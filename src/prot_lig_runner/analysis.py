"""Trajectory analysis, reporting, and optional gmx_MMPBSA processing."""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import tempfile
from pathlib import Path

from .analysis_metrics import trajectory_metrics
from .analysis_plots import (
    PLOT_FILES,
    _plot_mmpbsa_decomposition,
    _plot_qc,
    _plot_trajectory,
)
from .mmpbsa import ionic_strength_from_config, run_mmpbsa
from .report import RunDataset, build_report
from .trajectory import read_state_data, write_index_file

logger = logging.getLogger(__name__)
K_B_KCAL_PER_MOL_K = 0.00198720425864083


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
    for pattern in ("ligand_pocket_LIG*.png", "ligand_structure_LIG*.png",
                    "ligand_structure_LIG*.svg"):
        for path in analysis_dir.glob(pattern):
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
            salt_molar=ionic_strength_from_config(config),
        )
        entropy_path = mmpbsa_dir / "entropy.json"
        if entropy_path.is_file():
            summary = json.loads(entropy_path.read_text())
            metrics["mmpbsa"] = {
                key: value for key, value in summary.items() if not isinstance(value, list)
            }
            generated.extend(_plot_mmpbsa_decomposition(
                analysis_dir, summary, mmpbsa_dir / "FINAL_RESULTS_MMPBSA.csv",
            ))
        for name in ("mmpbsa_decomposition.png", "mmpbsa_membrane_planes.png"):
            if (analysis_dir / name).is_file() and name not in generated:
                generated.append(name)
        if not any(name.startswith("mmpbsa_") for name in generated):
            warnings.append("MMPBSA completed, but its CSV contained no supported frame table for plotting.")
    elif (analysis_dir / "mmpbsa" / "FINAL_RESULTS_MMPBSA.csv").is_file():
        summary_path = analysis_dir / "mmpbsa" / "entropy.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        if summary:
            metrics["mmpbsa"] = {
                key: value for key, value in summary.items() if not isinstance(value, list)
            }
        generated.extend(_plot_mmpbsa_decomposition(
            analysis_dir, summary, analysis_dir / "mmpbsa" / "FINAL_RESULTS_MMPBSA.csv",
        ))
    else:
        warnings.append("MMPBSA plots skipped: this run has no MMPBSA results.")

    ligands = (metrics.get("trajectory_metrics") or {}).get("ligands") or {}
    if ligands:
        from .report.structures import draw_ligand

        for index, (label, stats) in enumerate(ligands.items(), start=1):
            sdf_path = output_dir / "inputs" / f"ligand_{index}.sdf"
            png = analysis_dir / f"ligand_structure_{label}.png"
            svg = analysis_dir / f"ligand_structure_{label}.svg"
            caption = f"{label} ({stats.get('residue', '')})"
            if draw_ligand(sdf_path, png, svg, label=caption):
                generated.append(png.name)

    (analysis_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    build_report(RunDataset(output_dir, analysis_dir=analysis_dir), warnings=warnings)
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
    parser.add_argument("--mmpbsa-entropy", choices=("ie", "c2", "nmode", "both", "none"), default="ie",
                        help="Entropy estimate: interaction entropy (default), C2 variance, "
                             "normal mode (small systems), both (IE+C2, ΔG from IE), or enthalpy only")
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
