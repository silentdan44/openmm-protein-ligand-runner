"""Analysis orchestration and optional gmx_MMPBSA post-processing."""

from __future__ import annotations

import csv
import json
import logging
import math
import shutil
import subprocess
from pathlib import Path

from .analysis_metrics import trajectory_metrics
from .analysis_plots import _plot_mmpbsa, _plot_qc, _plot_trajectory, build_report

logger = logging.getLogger(__name__)


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


def phase_for_step(step, nvt_steps, npt_steps):
    if step <= nvt_steps:
        return "NVT"
    if step <= nvt_steps + npt_steps:
        return "NPT"
    return "Production"


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
            writer.write(selected.xyz, time=selected.time,
                         box=selected.unitcell_vectors)
            frames_written += selected.n_frames
    if frames_written == 0:
        raise ValueError("No production frames fall inside the requested MMPBSA window")
    return frames_written


def run_mmpbsa(output_dir, analysis_dir, *, model="gb", interval=1, last_ns=None,
               membrane_thickness_angstrom=None, membrane_dielectric=7.0,
               decomposition=False):
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
        if membrane_thickness_angstrom is None or membrane_thickness_angstrom <= 0:
            raise ValueError("The pb_mem model requires a positive membrane thickness in angstroms")
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
    input_path.write_text(
        "Single-trajectory protein-ligand binding free-energy estimate.\n"
        + (f"Trajectory window: final {float(last_ns):g} ns of production; "
           "frames outside this window were excluded before applying interval.\n"
           if last_ns is not None else "Trajectory window: all available production frames.\n")
        + f"&general\n startframe=1, interval={interval}, verbose=1,\n/\n"
        + model_namelist + decomposition_namelist
    )

    import mdtraj as md
    reference_path = mmpbsa_dir / "reference.pdb"
    source_trajectory = analysis_dir / "trajectory_fitted.xtc"
    if complex_only:
        # The requested GB/PB model treats water and lipids implicitly. Keep
        # the MMPBSA topology small and stream its atoms from the full MD XTC.
        full_topology = md.load_topology(str(output_dir / "system_final.pdb"))
        full_complex_indices = full_topology.select("protein or resname LIG")
        if len(full_complex_indices) == 0:
            raise ValueError("Could not select protein and LIG atoms from the simulation topology")
        complex_topology = md.load_topology(str(complex_pdb))
        if complex_topology.n_atoms != len(full_complex_indices):
            raise ValueError(
                "MMPBSA complex topology does not match protein-ligand atoms in the simulation topology"
            )
        index_path = mmpbsa_dir / "index.ndx"
        write_index_file(
            index_path,
            list(complex_topology.select("protein")),
            list(complex_topology.select("resname LIG")),
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
        mmpbsa_index = analysis_dir / "index.ndx"
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
    return mmpbsa_dir


def analyze_run(output_dir, config, *, perform_mmpbsa=False, mmpbsa_model="gb",
                mmpbsa_interval=1, mmpbsa_last_ns=None, mmpbsa_membrane_thickness_angstrom=None,
                mmpbsa_membrane_dielectric=7.0, mmpbsa_decomposition=False,
                analysis_stride=10, distance_pairs=None):
    output_dir = Path(output_dir)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    known_plots = {
        "simulation_qc.png", "protein_stability.png", "protein_rmsf.png", "ligand_pocket.png",
        "contact_occupancy.png", "hydrogen_bond_occupancy.png", "membrane_integrity.png",
        "residue_distances.png", "mmpbsa_energy_terms.png", "mmpbsa_energy_summary.png",
        "mmpbsa_residue_decomposition.png", "diagnostics.png", "temperature.png",
        "potential_energy.png", "kinetic_energy.png", "total_energy.png", "npt_density.png",
        "npt_box_volume.png", "protein_backbone_rmsd.png", "ligand_rmsd.png",
        "protein_radius_of_gyration.png", "ligand_radius_of_gyration.png", "ligand_protein_contacts.png",
    }
    for name in known_plots:
        (analysis_dir / name).unlink(missing_ok=True)
    for name in ("protein_ligand_contacts.csv", "protein_ligand_hbonds.csv", "residue_distances.csv"):
        (analysis_dir / name).unlink(missing_ok=True)

    warnings = []
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
        )
        mmpbsa_plots = _plot_mmpbsa(
            analysis_dir, mmpbsa_dir / "FINAL_RESULTS_MMPBSA.csv",
            mmpbsa_dir / "FINAL_DECOMP_MMPBSA.csv" if mmpbsa_decomposition else None,
        )
        generated.extend(mmpbsa_plots)
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
