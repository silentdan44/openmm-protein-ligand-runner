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


def run_mmpbsa(output_dir, analysis_dir, *, model="gb", interval=1,
               membrane_thickness_angstrom=None, membrane_dielectric=7.0,
               decomposition=False):
    """Run a single-trajectory gmx_MMPBSA estimate, optionally with decomposition."""
    executable = shutil.which("gmx_MMPBSA")
    if executable is None:
        raise RuntimeError("gmx_MMPBSA is not installed or is not available on PATH")
    output_dir, analysis_dir = Path(output_dir), Path(analysis_dir)
    required = [output_dir / "system.top", output_dir / "system_final.pdb",
                analysis_dir / "trajectory_fitted.xtc", analysis_dir / "index.ndx"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required MMPBSA inputs are missing: " + ", ".join(missing))
    mmpbsa_dir = analysis_dir / "mmpbsa"
    mmpbsa_dir.mkdir(exist_ok=True)
    input_path = mmpbsa_dir / "mmpbsa.in"
    if model not in {"gb", "pb", "pb_mem"}:
        raise ValueError(f"Unsupported MMPBSA model: {model}")
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
            + "poretype=1, istrng=0.150,\n/\n"
        )
    decomposition_namelist = (
        '&decomp\n idecomp=2, dec_verbose=1, print_res="within 4",\n/\n'
        if decomposition else ""
    )
    input_path.write_text(
        "Single-trajectory protein-ligand binding free-energy estimate.\n"
        f"&general\n startframe=1, interval={interval}, verbose=1,\n/\n"
        + model_namelist + decomposition_namelist
    )

    # gmx_MMPBSA needs a complex-only reference for chain and residue mapping.
    import mdtraj as md

    reference_path = mmpbsa_dir / "reference.pdb"
    reference = md.load_topology(str(output_dir / "system_final.pdb"))
    reference_atoms = reference.select("protein or resname LIG")
    if len(reference_atoms) == 0:
        raise ValueError("Could not select protein and LIG atoms for the MMPBSA reference structure")
    md.load_pdb(str(output_dir / "system_final.pdb")).atom_slice(reference_atoms).save_pdb(str(reference_path))
    command = [
        executable, "-O", "-i", str(input_path), "-cs", str(output_dir / "system_final.pdb"),
        "-cp", str(output_dir / "system.top"), "-cr", str(reference_path),
        "-ci", str(analysis_dir / "index.ndx"), "-cg", "Protein", "LIG",
        "-ct", str(analysis_dir / "trajectory_fitted.xtc"), "-o", str(mmpbsa_dir / "FINAL_RESULTS_MMPBSA.dat"),
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
                mmpbsa_interval=1, mmpbsa_membrane_thickness_angstrom=None,
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
