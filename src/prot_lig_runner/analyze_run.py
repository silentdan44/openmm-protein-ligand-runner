"""Rebuild diagnostic plots and optionally run gmx_MMPBSA from a completed run."""

import argparse
import json
import logging
import math
from pathlib import Path

from .trajectory_analysis import analyze_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output", help="Completed OpenMM run directory")
    parser.add_argument("--mmpbsa", action="store_true", help="Also run gmx_MMPBSA")
    parser.add_argument("--mmpbsa-model", choices=("gb", "pb", "pb_mem"), default="gb")
    parser.add_argument("--mmpbsa-interval", type=int, default=1)
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
    output_dir = Path(args.output)
    if args.mmpbsa_interval < 1:
        parser.error("--mmpbsa-interval must be a positive integer")
    if args.mmpbsa_last_ns is not None and (
        not math.isfinite(args.mmpbsa_last_ns) or args.mmpbsa_last_ns <= 0
    ):
        parser.error("--mmpbsa-last-ns must be a positive finite number")
    if args.analysis_stride < 1:
        parser.error("--analysis-stride must be a positive integer")
    if args.mmpbsa and args.mmpbsa_model == "pb_mem" and (
        args.mmpbsa_membrane_thickness_angstrom is None
        or args.mmpbsa_membrane_thickness_angstrom <= 0
    ):
        parser.error("pb_mem requires --mmpbsa-membrane-thickness-angstrom")
    if args.mmpbsa_decomposition and not args.mmpbsa:
        parser.error("--mmpbsa-decomposition requires --mmpbsa")
    if args.mmpbsa and args.mmpbsa_model == "pb_mem":
        state = json.loads((output_dir / "run_state.json").read_text()) if (output_dir / "run_state.json").is_file() else {}
        if not state.get("configuration", {}).get("membrane"):
            parser.error("pb_mem requires a completed --membrane run")

    progress_path = output_dir / "run_state.json"
    if not progress_path.is_file():
        parser.error(f"No run_state.json found in {output_dir}")
    state = json.loads(progress_path.read_text())
    config = state["configuration"]
    config.update({key: config[key] for key in (
        "nvt_steps", "npt_steps", "prod_steps", "timestep_fs", "temperature_k", "pressure_bar"
    )})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = analyze_run(
        output_dir, config, perform_mmpbsa=args.mmpbsa,
        mmpbsa_model=args.mmpbsa_model, mmpbsa_interval=args.mmpbsa_interval,
        mmpbsa_last_ns=args.mmpbsa_last_ns,
        mmpbsa_membrane_thickness_angstrom=args.mmpbsa_membrane_thickness_angstrom,
        mmpbsa_membrane_dielectric=args.mmpbsa_membrane_dielectric,
        analysis_stride=args.analysis_stride,
        mmpbsa_decomposition=args.mmpbsa_decomposition,
        distance_pairs=args.distance_pair,
    )
    print(f"Analysis report: {result / 'report.html'}")


if __name__ == "__main__":
    main()
