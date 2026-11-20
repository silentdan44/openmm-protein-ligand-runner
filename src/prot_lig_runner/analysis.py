"""Trajectory analysis, reporting, and optional gmx_MMPBSA processing."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from .analysis_metrics import (
    estimate_membrane_planes,
    resolve_ligand,
    trajectory_metrics,
)
from .analysis_plots import (
    PLOT_FILES,
    _plot_mmpbsa,
    _plot_mmpbsa_entropy,
    _plot_qc,
    _plot_trajectory,
    build_report,
    plot_membrane_planes,
    read_mmpbsa_delta_csv,
)

logger = logging.getLogger(__name__)
K_B_KCAL_PER_MOL_K = 0.00198720425864083


def read_state_data(path):
    """Read OpenMM StateDataReporter CSV output into normalized records."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as source:
        rows = []
        for row in csv.DictReader(source):
            try:
                rows.append({key: float(value) if value not in (None, "") else math.nan
                             for key, value in row.items()})
            except (TypeError, ValueError):
                continue
    return rows


def interaction_entropy(electrostatics_kcal, temperature_k):
    """Estimate -TΔS by the interaction-entropy method (Duan et al.).

    −TΔS = kT ln⟨exp(ΔE_el/kT)⟩ − ⟨ΔE_el⟩. The mean electrostatics are already
    part of the MMPBSA enthalpy, so only the fluctuation term is the entropy
    contribution; without the subtraction the electrostatics would be counted
    twice. Uses a shifted log-mean-exp so strong favourable energies do not
    underflow.
    """
    import numpy as np

    values = np.asarray(electrostatics_kcal, dtype=float)
    if values.size == 0:
        raise ValueError("Interaction entropy needs at least one frame")
    kt = K_B_KCAL_PER_MOL_K * float(temperature_k)
    maximum = float(values.max())
    free_energy = maximum + kt * float(np.log(np.mean(np.exp((values - maximum) / kt))))
    return float(free_energy - values.mean())


def summarize_mmpbsa(path, *, temperature_k=300.0, entropy="ie"):
    """Summarise per-frame MMPBSA terms, add interaction entropy and errors."""
    import numpy as np

    rows = read_mmpbsa_delta_csv(path)
    if not rows:
        raise ValueError(f"No per-frame MMPBSA terms found in {path}")
    frame_key = next(key for key in rows[0] if key.lower().startswith("frame"))
    total_key = "TOTAL" if "TOTAL" in rows[0] else None
    eel_key = "EEL" if "EEL" in rows[0] else None
    if total_key is None or eel_key is None:
        raise ValueError("MMPBSA CSV has no TOTAL/EEL columns in the Delta section")
    frames = np.asarray([row[frame_key] for row in rows], dtype=float)
    enthalpy = np.asarray([row[total_key] for row in rows], dtype=float)
    eel = np.asarray([row[eel_key] for row in rows], dtype=float)

    running_enthalpy = np.cumsum(enthalpy) / np.arange(1, len(enthalpy) + 1)
    running_entropy = np.asarray(
        [interaction_entropy(eel[: index + 1], temperature_k) for index in range(len(eel))]
    ) if entropy == "ie" else np.zeros(len(eel))
    running_free = running_enthalpy + running_entropy

    def sem(values):
        return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")

    def block_sem(values, blocks=5):
        if len(values) < 2 * blocks:
            return float("nan")
        chunks = np.array_split(values, blocks)
        estimates = [interaction_entropy(chunk, temperature_k) for chunk in chunks]
        return float(np.std(estimates, ddof=1) / np.sqrt(blocks))

    delta_h = float(enthalpy.mean())
    minus_tds = float(running_entropy[-1]) if entropy == "ie" else 0.0
    sem_h = sem(enthalpy)
    sem_tds = block_sem(eel) if entropy == "ie" else 0.0
    delta_g = delta_h + minus_tds
    if np.isfinite(sem_tds):
        sem_g = float(np.sqrt(sem_h ** 2 + sem_tds ** 2))
    else:
        # Too few frames for a block estimate of the ensemble-level entropy;
        # report the enthalpy SEM and keep the unknown entropy SEM explicit.
        sem_g = sem_h
    return {
        "entropy_method": entropy,
        "temperature_k": float(temperature_k),
        "frame_count": int(len(frames)),
        "frames": frames.tolist(),
        "delta_h_kcal_mol": enthalpy.tolist(),
        "delta_eel_kcal_mol": eel.tolist(),
        "minus_tds_kcal_mol": running_entropy.tolist(),
        "delta_g_running_kcal_mol": running_free.tolist(),
        "delta_h_mean_kcal_mol": delta_h,
        "minus_tds_kcal_mol_value": minus_tds,
        "delta_g_kcal_mol": float(delta_g),
        "sem_delta_h_kcal_mol": sem_h,
        "sem_minus_tds_kcal_mol": sem_tds if np.isfinite(sem_tds) else None,
        "sem_delta_g_kcal_mol": sem_g,
        "note": "Single-trajectory MM/PBSA enthalpy plus interaction entropy; an "
                "approximate end-state estimate, not a calibrated absolute free energy.",
    }


def write_index_file(path, protein_indices, ligand_indices):
    """Write 1-based GROMACS index groups in trajectory/topology atom order."""
    if not protein_indices or not ligand_indices:
        raise ValueError("Protein and ligand atom groups must both be non-empty")
    lines = []
    for name, indices in (("Protein", protein_indices), ("LIG", ligand_indices)):
        lines.append(f"[ {name} ]")
        values = [index + 1 for index in indices]
        lines.extend(" ".join(str(value) for value in values[start:start + 15])
                     for start in range(0, len(values), 15))
        lines.append("")
    Path(path).write_text("\n".join(lines))


def _mmpbsa_window_bounds(trajectory_path, topology_path, last_ns):
    """Return absolute ps bounds for the final requested production window."""
    import mdtraj as md

    first_time = last_time = None
    for chunk in md.iterload(str(trajectory_path), top=str(topology_path), chunk=1000):
        if chunk.n_frames == 0:
            continue
        if first_time is None:
            first_time = float(chunk.time[0])
        last_time = float(chunk.time[-1])
    if last_time is None:
        raise ValueError("The production trajectory contains no frames for MMPBSA")
    cutoff = last_time - float(last_ns) * 1000.0
    if cutoff < first_time:
        logger.warning(
            "Requested final %.3f ns for MMPBSA, but the available production trajectory is "
            "only %.3f ns; using all available production frames",
            last_ns, (last_time - first_time) / 1000.0,
        )
        cutoff = first_time
    return cutoff


def _wrap_selection_into_box(trajectory):
    """Translate each frame so the selected complex sits in the box centre.

    The analysis unwraps trajectories for RMSD; gmx_MMPBSA expects the complex
    inside the box for its focusing/PB grid, so shift the whole selection by one
    lattice vector per frame instead of wrapping individual atoms.
    """
    import numpy as np

    boxes = trajectory.unitcell_vectors
    if boxes is None:
        return
    try:
        boxes = np.asarray(boxes, dtype=float)
        xyz = np.asarray(trajectory.xyz, dtype=float)
    except (TypeError, ValueError):
        return
    if boxes.ndim != 3 or xyz.ndim != 3:
        return
    for frame in range(min(len(xyz), len(boxes))):
        inverse = np.linalg.inv(boxes[frame])
        fractional = trajectory.xyz[frame].mean(axis=0) @ inverse
        shift = (0.5 - fractional) @ boxes[frame]
        trajectory.xyz[frame] += shift


def _copy_mmpbsa_window(source_path, topology_path, target_path, *, last_ns,
                        atom_indices=None):
    """Write a production trajectory window, optionally reduced to selected atoms."""
    import mdtraj as md

    cutoff = _mmpbsa_window_bounds(source_path, topology_path, last_ns)
    frames_written = 0
    with md.formats.XTCTrajectoryFile(str(target_path), mode="w") as writer:
        for chunk in md.iterload(str(source_path), top=str(topology_path), chunk=1000):
            if chunk.n_frames == 0:
                continue
            selected = chunk[chunk.time >= cutoff]
            if selected.n_frames == 0:
                continue
            if atom_indices is not None:
                selected = selected.atom_slice(atom_indices)
                _wrap_selection_into_box(selected)
            writer.write(selected.xyz, time=selected.time,
                         box=selected.unitcell_vectors)
            frames_written += selected.n_frames
    if frames_written == 0:
        raise ValueError("No production frames fall inside the requested MMPBSA window")
    return frames_written


def run_mmpbsa(output_dir, analysis_dir, *, model="gb", interval=1, last_ns=None,
               membrane_thickness_angstrom=None, membrane_dielectric=7.0,
               decomposition=False, ligand=None, temperature_k=300.0, entropy="ie"):
    """Run a single-trajectory gmx_MMPBSA estimate, optionally with decomposition."""
    executable = shutil.which("gmx_MMPBSA")
    if executable is None:
        raise RuntimeError("gmx_MMPBSA is not installed or is not available on PATH")
    output_dir, analysis_dir = Path(output_dir), Path(analysis_dir)
    complex_pdb = output_dir / "mmpbsa_complex.pdb"
    complex_top = output_dir / "mmpbsa_complex.top"
    complex_only = complex_pdb.is_file() and complex_top.is_file()
    required = (
        [complex_top, complex_pdb, output_dir / "system_final.pdb",
         analysis_dir / "trajectory_fitted.xtc"]
        if complex_only else
        [output_dir / "system.top", output_dir / "system_final.pdb",
         analysis_dir / "trajectory_fitted.xtc", analysis_dir / "index.ndx"]
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required MMPBSA inputs are missing: " + ", ".join(missing))
    mmpbsa_dir = analysis_dir / "mmpbsa"
    mmpbsa_dir.mkdir(exist_ok=True)
    input_path = mmpbsa_dir / "mmpbsa.in"
    if model not in {"gb", "pb", "pb_mem"}:
        raise ValueError(f"Unsupported MMPBSA model: {model}")
    if last_ns is not None and (not math.isfinite(float(last_ns)) or float(last_ns) <= 0):
        raise ValueError("The MMPBSA last-ns window must be a positive finite number")
    model_namelist = "&gb\n igb=5, saltcon=0.150,\n/\n" if model == "gb" else "&pb\n istrng=0.150,\n/\n"
    if model == "pb_mem":
        if membrane_thickness_angstrom is None:
            import mdtraj as md

            structure = md.load_pdb(str(output_dir / "system_final.pdb"))
            state = json.loads((output_dir / "run_state.json").read_text())
            configuration = state["configuration"]
            estimate = estimate_membrane_planes(
                structure.topology,
                structure.xyz[0],
                center_z_nm=configuration.get("membrane_center_z_nm") or 0.0,
                lipid_type=configuration.get("lipid_type", "POPC"),
            )
            membrane_thickness_angstrom = estimate["thickness_angstrom"]
            plot_membrane_planes(analysis_dir, estimate, filename="mmpbsa_membrane_planes.png")
            (mmpbsa_dir / "membrane_planes.json").write_text(
                json.dumps({key: value for key, value in estimate.items()
                            if key not in ("atom_indices", "atom_z_nm")}, indent=2, sort_keys=True) + "\n"
            )
            logger.info(
                "Auto membrane thickness %.2f A from %d/%d phosphorus atoms "
                "(%d outliers rejected)",
                membrane_thickness_angstrom,
                estimate["lower_atom_count"],
                estimate["upper_atom_count"],
                estimate["outlier_count"],
            )
        if (membrane_thickness_angstrom is None
                or not math.isfinite(float(membrane_thickness_angstrom))
                or float(membrane_thickness_angstrom) <= 0):
            raise ValueError("The pb_mem model requires a positive membrane thickness in angstroms")
        if not math.isfinite(float(membrane_dielectric)) or float(membrane_dielectric) <= 0:
            raise ValueError("The pb_mem model requires a positive finite membrane dielectric")
        state = json.loads((output_dir / "run_state.json").read_text())
        if not state["configuration"].get("membrane"):
            raise ValueError("The pb_mem model requires an OpenMM membrane run")
        center_z_angstrom = 10.0 * float(state["configuration"].get("membrane_center_z_nm") or 0.0)
        model_namelist = (
            "&pb\n memopt=1, emem=" + f"{membrane_dielectric:g}" + ", indi=4.0, "
            + f"mctrdz={center_z_angstrom:g}, mthick={membrane_thickness_angstrom:g}, "
            + "poretype=1, radiopt=0, istrng=0.150, fillratio=1.25, inp=2, "
            # Amber membrane PB needs the linear non-focusing periodic solver.
            + "sasopt=0, solvopt=2, ipb=1, bcopt=10, nfocus=1, linit=1000, "
            + "eneopt=1, cutfd=7.0, cutnb=99.0, maxarcdot=15000, npbverb=1,\n/\n"
        )
    decomposition_namelist = (
        '&decomp\n idecomp=2, dec_verbose=1, print_res="within 4",\n/\n'
        if decomposition else ""
    )
    # igb=5 conventionally pairs with mbondi2 radii (PBRadii=3); gmx_MMPBSA
    # otherwise reuses mbondi3 (4), which is calibrated for igb=8. PB runs take
    # radii from the topology (radiopt=0), where mbondi3 is the usual choice.
    pb_radii = 3 if model == "gb" else 4
    input_path.write_text(
        "Single-trajectory protein-ligand binding free-energy estimate.\n"
        + (f"Trajectory window: final {float(last_ns):g} ns of production; "
           "frames outside this window were excluded before applying interval.\n"
           if last_ns is not None else "Trajectory window: all available production frames.\n")
        + f"&general\n startframe=1, interval={interval}, PBRadii={pb_radii}, verbose=1,\n/\n"
        + model_namelist + decomposition_namelist
    )

    import mdtraj as md
    reference_path = mmpbsa_dir / "reference.pdb"
    source_trajectory = analysis_dir / "trajectory_fitted.xtc"
    full_topology = md.load_topology(str(output_dir / "system_final.pdb"))
    selection = resolve_ligand(full_topology, ligand)
    full_complex_indices = list(full_topology.select("protein")) + list(selection["atoms"])
    if not full_complex_indices:
        raise ValueError("Could not select protein and ligand atoms from the simulation topology")
    if complex_only:
        # The requested GB/PB model treats water and lipids implicitly. Keep
        # the MMPBSA topology small and stream its atoms from the full MD XTC.
        complex_topology = md.load_topology(str(complex_pdb))
        complex_ligands = [residue for residue in complex_topology.residues
                           if residue.name == "LIG"]
        if len(complex_ligands) != 1:
            raise ValueError(
                f"{complex_pdb} contains {len(complex_ligands)} ligand residues. "
                "Regenerate it with --mmpbsa-ligand so one ligand is exported."
            )
        if complex_topology.n_atoms != len(full_complex_indices):
            raise ValueError(
                "MMPBSA complex topology does not match the selected protein-ligand "
                "atoms in the simulation topology"
            )
        index_path = mmpbsa_dir / "index.ndx"
        write_index_file(
            index_path,
            list(complex_topology.select("protein")),
            [atom.index for atom in complex_ligands[0].atoms],
        )
        complex_trajectory_path = mmpbsa_dir / "complex.xtc"
        if last_ns is not None:
            _copy_mmpbsa_window(
                source_trajectory, output_dir / "system_final.pdb", complex_trajectory_path,
                last_ns=last_ns, atom_indices=full_complex_indices,
            )
        else:
            frames_written = 0
            with md.formats.XTCTrajectoryFile(str(complex_trajectory_path), mode="w") as writer:
                for chunk in md.iterload(str(source_trajectory), top=str(output_dir / "system_final.pdb"), chunk=1000):
                    selected = chunk.atom_slice(full_complex_indices)
                    _wrap_selection_into_box(selected)
                    writer.write(selected.xyz, time=selected.time, box=selected.unitcell_vectors)
                    frames_written += selected.n_frames
            if frames_written == 0:
                raise ValueError("The trajectory contains no frames for MMPBSA")
        reference_path.write_text(complex_pdb.read_text())
        complex_structure = complex_pdb
        complex_topology_path = complex_top
        mmpbsa_index = index_path
        mmpbsa_trajectory = complex_trajectory_path
    else:
        full_reference = md.load_topology(str(output_dir / "system_final.pdb"))
        reference_atoms = full_reference.select("protein or resname LIG")
        if len(reference_atoms) == 0:
            raise ValueError("Could not select protein and LIG atoms from the MMPBSA reference structure")
        md.load_pdb(str(output_dir / "system_final.pdb")).atom_slice(reference_atoms).save_pdb(str(reference_path))
        complex_structure = output_dir / "system_final.pdb"
        complex_topology_path = output_dir / "system.top"
        # The full GROMACS topology can contain several ligands; the index
        # groups decide which one is the ligand in this calculation.
        mmpbsa_index = mmpbsa_dir / "index.ndx"
        write_index_file(
            mmpbsa_index,
            list(full_reference.select("protein")),
            list(selection["atoms"]),
        )
        if last_ns is not None:
            mmpbsa_trajectory = mmpbsa_dir / "production_window.xtc"
            _copy_mmpbsa_window(
                source_trajectory, output_dir / "system_final.pdb", mmpbsa_trajectory,
                last_ns=last_ns,
            )
        else:
            mmpbsa_trajectory = source_trajectory
    command = [
        executable, "-O", "-i", str(input_path), "-cs", str(complex_structure),
        "-cp", str(complex_topology_path), "-cr", str(reference_path),
        "-ci", str(mmpbsa_index), "-cg", "Protein", "LIG",
        "-ct", str(mmpbsa_trajectory), "-o", str(mmpbsa_dir / "FINAL_RESULTS_MMPBSA.dat"),
        "-eo", str(mmpbsa_dir / "FINAL_RESULTS_MMPBSA.csv"), "-nogui",
    ]
    if decomposition:
        command.extend(["-do", str(mmpbsa_dir / "FINAL_DECOMP_MMPBSA.dat"),
                        "-deo", str(mmpbsa_dir / "FINAL_DECOMP_MMPBSA.csv")])
    (mmpbsa_dir / "command.txt").write_text(" ".join(command) + "\n")
    logger.info("Running gmx_MMPBSA (%s); this may take some time", model)
    with (mmpbsa_dir / "gmx_MMPBSA.log").open("w") as log:
        completed = subprocess.run(command, cwd=mmpbsa_dir, stdout=log, stderr=subprocess.STDOUT,
                                   text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"gmx_MMPBSA failed with exit code {completed.returncode}; see {mmpbsa_dir / 'gmx_MMPBSA.log'}")
    try:
        summary = summarize_mmpbsa(
            mmpbsa_dir / "FINAL_RESULTS_MMPBSA.csv",
            temperature_k=temperature_k,
            entropy=entropy,
        )
    except Exception as error:
        logger.warning("MMPBSA finished but entropy summarisation failed: %s", error)
    else:
        (mmpbsa_dir / "entropy.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        _plot_mmpbsa_entropy(analysis_dir, summary)
        logger.info(
            "MMPBSA: dH = %.2f, -TdS = %.2f, dG = %.2f kcal/mol over %d frames (%s)",
            summary["delta_h_mean_kcal_mol"], summary["minus_tds_kcal_mol_value"],
            summary["delta_g_kcal_mol"], summary["frame_count"], entropy,
        )
    return mmpbsa_dir


def analyze_run(output_dir, config, *, analysis_dir=None, preliminary_phase=None, perform_mmpbsa=False, mmpbsa_model="gb",
                mmpbsa_interval=1, mmpbsa_last_ns=None, mmpbsa_membrane_thickness_angstrom=None,
                mmpbsa_membrane_dielectric=7.0, mmpbsa_decomposition=False,
                analysis_stride=10, distance_pairs=None, mmpbsa_ligand=None,
                mmpbsa_entropy="ie"):
    output_dir = Path(output_dir)
    analysis_dir = Path(analysis_dir) if analysis_dir is not None else output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    for name in PLOT_FILES:
        (analysis_dir / name).unlink(missing_ok=True)
    for path in analysis_dir.glob("ligand_pocket_LIG*.png"):
        path.unlink(missing_ok=True)
    for name in ("protein_ligand_contacts.csv", "protein_ligand_hbonds.csv", "residue_distances.csv"):
        (analysis_dir / name).unlink(missing_ok=True)

    warnings = []
    if preliminary_phase is not None:
        warnings.append(f"PRELIMINARY report: snapshot taken during phase {preliminary_phase}; results may change as the simulation continues.")
    state_rows = read_state_data(output_dir / "state_data.csv")
    trajectory_data = None
    trajectory_path = output_dir / "trajectory.xtc"
    topology_path = output_dir / "system_final.pdb"
    if trajectory_path.is_file() and trajectory_path.stat().st_size and topology_path.is_file():
        try:
            trajectory_data = trajectory_metrics(
                trajectory_path, topology_path, analysis_dir, config,
                analysis_stride=analysis_stride,
                distance_pairs=distance_pairs if distance_pairs is not None else config.get("distance_pairs"),
            )
        except Exception as error:
            logger.warning("Trajectory metrics unavailable: %s", error)
            warnings.append(f"Trajectory metrics unavailable: {error}")
    else:
        warnings.append("Trajectory metrics skipped: trajectory.xtc or system_final.pdb is missing.")

    generated = []
    if _plot_qc(analysis_dir, state_rows, config):
        generated.append("simulation_qc.png")
    else:
        warnings.append("Simulation quality-control plot skipped: no usable state_data.csv samples.")
    generated.extend(_plot_trajectory(analysis_dir, trajectory_data))
    if trajectory_data and trajectory_data.get("distance_pair_warning"):
        warnings.append(f"Residue-distance graph skipped: {trajectory_data['distance_pair_warning']}")

    metrics = {
        "preliminary": preliminary_phase is not None,
        "snapshot_phase": preliminary_phase,
        "frame_count": trajectory_data["frame_count"] if trajectory_data else 0,
        "analysis_stride": trajectory_data["analysis_stride"] if trajectory_data else analysis_stride,
        "production_start_ns": trajectory_data["production_start_ns"] if trajectory_data else None,
        "trajectory_metrics": trajectory_data["statistics"] if trajectory_data else {},
        "membrane_core_selection": trajectory_data["membrane_core_selection"] if trajectory_data else None,
        "state_sample_count": len(state_rows),
        "temperature_k": config.get("temperature_k"),
        "pressure_bar": config.get("pressure_bar"),
        "distance_pairs": list(trajectory_data["distance_series"]) if trajectory_data else [],
        "note": "Frame-to-frame spread is descriptive; correlated frames do not replace independent replicas.",
    }

    if trajectory_data:
        topology = __import__("mdtraj").load_topology(str(topology_path))
        protein_indices = topology.select("protein")
        ligand_indices = topology.select("resname LIG")
        write_index_file(analysis_dir / "index.ndx", protein_indices.tolist(), ligand_indices.tolist())

    if perform_mmpbsa:
        mmpbsa_dir = run_mmpbsa(
            output_dir, analysis_dir, model=mmpbsa_model, interval=mmpbsa_interval,
            last_ns=mmpbsa_last_ns,
            membrane_thickness_angstrom=mmpbsa_membrane_thickness_angstrom,
            membrane_dielectric=mmpbsa_membrane_dielectric,
            decomposition=mmpbsa_decomposition,
            ligand=mmpbsa_ligand,
            temperature_k=float(config.get("temperature_k") or 300.0),
            entropy=mmpbsa_entropy,
        )
        mmpbsa_plots = _plot_mmpbsa(
            analysis_dir, mmpbsa_dir / "FINAL_RESULTS_MMPBSA.csv",
            mmpbsa_dir / "FINAL_DECOMP_MMPBSA.csv" if mmpbsa_decomposition else None,
        )
        generated.extend(mmpbsa_plots)
        entropy_path = mmpbsa_dir / "entropy.json"
        if entropy_path.is_file():
            summary = json.loads(entropy_path.read_text())
            metrics["mmpbsa"] = {
                key: value for key, value in summary.items() if not isinstance(value, list)
            }
            for name in ("mmpbsa_entropy.png", "mmpbsa_membrane_planes.png"):
                if (analysis_dir / name).is_file() and name not in generated:
                    generated.append(name)
        if not mmpbsa_plots:
            warnings.append("MMPBSA completed, but its CSV contained no supported frame table for plotting.")
    elif (analysis_dir / "mmpbsa" / "FINAL_RESULTS_MMPBSA.csv").is_file():
        generated.extend(_plot_mmpbsa(
            analysis_dir, analysis_dir / "mmpbsa" / "FINAL_RESULTS_MMPBSA.csv",
            analysis_dir / "mmpbsa" / "FINAL_DECOMP_MMPBSA.csv",
        ))
    else:
        warnings.append("MMPBSA plots skipped: this run has no MMPBSA results.")

    (analysis_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    build_report(analysis_dir, metrics, generated, warnings=warnings)
    return analysis_dir


def analyze_snapshot(output_dir, config, phase, *, analysis_stride=10, distance_pairs=None):
    """Analyze a bounded copy of a live run without modifying its output files."""
    output_dir = Path(output_dir)
    with tempfile.TemporaryDirectory(prefix="md-preview-") as directory:
        snapshot = Path(directory)
        topology = output_dir / "system_final.pdb"
        if topology.is_file():
            shutil.copyfile(topology, snapshot / topology.name)
        state_path = output_dir / "state_data.csv"
        if state_path.is_file():
            with state_path.open("rb") as source:
                data = source.read(state_path.stat().st_size)
            # A reporter may be writing its last row while the copy is made.
            (snapshot / state_path.name).write_bytes(data[:data.rfind(b"\n") + 1])
        trajectory = output_dir / "trajectory.xtc"
        if trajectory.is_file() and trajectory.stat().st_size:
            import mdtraj as md

            raw = snapshot / "raw.xtc"
            with trajectory.open("rb") as source, raw.open("wb") as target:
                remaining = trajectory.stat().st_size
                while remaining:
                    block = source.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    target.write(block)
                    remaining -= len(block)
            # A snapshot can end in a partially written frame. Keep complete
            # frames only; the live trajectory is never opened for writing.
            try:
                with md.formats.XTCTrajectoryFile(str(raw), "r") as reader:
                    with md.formats.XTCTrajectoryFile(str(snapshot / "trajectory.xtc"), "w") as writer:
                        while True:
                            try:
                                xyz, time, step, box = reader.read(n_frames=1)
                            except (EOFError, OSError, RuntimeError, ValueError) as error:
                                logger.warning("Discarding incomplete final XTC frame: %s", error)
                                break
                            if len(xyz) == 0:
                                break
                            writer.write(xyz, time=time, step=step, box=box)
            except (EOFError, OSError, RuntimeError, ValueError) as error:
                logger.warning("Trajectory snapshot unavailable: %s", error)
        return analyze_run(snapshot, config, analysis_dir=output_dir / "analysis_preview",
                           preliminary_phase=phase, analysis_stride=analysis_stride,
                           distance_pairs=distance_pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output", help="Completed OpenMM run directory")
    parser.add_argument("--preview", action="store_true",
                        help="Analyze a snapshot of an ongoing run in analysis_preview")
    parser.add_argument("--mmpbsa", action="store_true", help="Also run gmx_MMPBSA")
    parser.add_argument("--mmpbsa-model", choices=("gb", "pb", "pb_mem"), default="gb")
    parser.add_argument("--mmpbsa-interval", type=int, default=1)
    parser.add_argument("--mmpbsa-ligand",
                        help="Ligand for the MMPBSA estimate: 1-based index (2), label (LIG2), "
                             "or residue label; required when the run has several ligands")
    parser.add_argument("--mmpbsa-entropy", choices=("ie", "none"), default="ie",
                        help="Entropy correction: interaction entropy (default) or enthalpy only")
    parser.add_argument("--mmpbsa-last-ns", type=float,
                        help="Restrict MMPBSA to the final N ns of production")
    parser.add_argument("--analysis-stride", type=int, default=10)
    parser.add_argument("--mmpbsa-membrane-thickness-angstrom", type=float)
    parser.add_argument("--mmpbsa-membrane-dielectric", type=float, default=7.0)
    parser.add_argument("--mmpbsa-decomposition", action="store_true",
                        help="Also calculate per-residue MMPBSA decomposition")
    parser.add_argument("--distance-pair", action="append", nargs=2,
                        metavar=("CHAIN:RESID", "CHAIN:RESID"),
                        help="Plot Cα distance between two residues; repeat to add pairs")
    args = parser.parse_args()
    if args.preview and args.mmpbsa:
        parser.error("--preview cannot be combined with --mmpbsa")
    if args.mmpbsa_interval < 1:
        parser.error("--mmpbsa-interval must be a positive integer")
    if args.mmpbsa_last_ns is not None and (
        not math.isfinite(args.mmpbsa_last_ns) or args.mmpbsa_last_ns <= 0
    ):
        parser.error("--mmpbsa-last-ns must be a positive finite number")
    if args.analysis_stride < 1:
        parser.error("--analysis-stride must be a positive integer")
    if args.mmpbsa and args.mmpbsa_model == "pb_mem" and (
        args.mmpbsa_membrane_thickness_angstrom is not None
        and (not math.isfinite(args.mmpbsa_membrane_thickness_angstrom)
             or args.mmpbsa_membrane_thickness_angstrom <= 0)
    ):
        parser.error("--mmpbsa-membrane-thickness-angstrom must be positive when given")
    if args.mmpbsa and args.mmpbsa_model == "pb_mem" and (
        not math.isfinite(args.mmpbsa_membrane_dielectric)
        or args.mmpbsa_membrane_dielectric <= 0
    ):
        parser.error("pb_mem requires a positive finite --mmpbsa-membrane-dielectric")
    if args.mmpbsa_decomposition and not args.mmpbsa:
        parser.error("--mmpbsa-decomposition requires --mmpbsa")

    output_dir = Path(args.output)
    progress_path = output_dir / "run_state.json"
    if not progress_path.is_file():
        parser.error(f"No run_state.json found in {output_dir}")
    state = json.loads(progress_path.read_text())
    config = state["configuration"]
    if args.mmpbsa and args.mmpbsa_model == "pb_mem" and not config.get("membrane"):
        parser.error("pb_mem requires a completed membrane run")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.preview:
        result = analyze_snapshot(output_dir, config, state.get("phase", "unknown"),
                                  analysis_stride=args.analysis_stride,
                                  distance_pairs=args.distance_pair)
        print(f"Preliminary analysis report: {result / 'report.html'}")
        return
    result = analyze_run(
        output_dir, config, perform_mmpbsa=args.mmpbsa,
        mmpbsa_model=args.mmpbsa_model, mmpbsa_interval=args.mmpbsa_interval,
        mmpbsa_last_ns=args.mmpbsa_last_ns,
        mmpbsa_membrane_thickness_angstrom=args.mmpbsa_membrane_thickness_angstrom,
        mmpbsa_membrane_dielectric=args.mmpbsa_membrane_dielectric,
        analysis_stride=args.analysis_stride,
        mmpbsa_decomposition=args.mmpbsa_decomposition,
        mmpbsa_ligand=args.mmpbsa_ligand,
        mmpbsa_entropy=args.mmpbsa_entropy,
        distance_pairs=args.distance_pair,
    )
    print(f"Analysis report: {result / 'report.html'}")


if __name__ == "__main__":
    main()
