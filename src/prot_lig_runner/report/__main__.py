"""Command line entry point: ``python -m prot_lig_runner.report --run DIR``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import RunDataset, SECTIONS
from .render import build_report


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m prot_lig_runner.report",
        description="Build a modular HTML report from a run directory, at any stage.",
    )
    parser.add_argument("--run", required=True, type=Path,
                        help="Run directory containing run_state.json and analysis/")
    parser.add_argument("--analysis-dir", type=Path,
                        help="Analysis directory (default: <run>/analysis)")
    parser.add_argument("--only", default="",
                        help="Comma-separated sections: " + ",".join(SECTIONS))
    parser.add_argument("--print-status", action="store_true",
                        help="Print the dataset status as JSON and exit")
    args = parser.parse_args(argv)

    only = tuple(item.strip() for item in args.only.split(",") if item.strip())
    unknown = [item for item in only if item not in SECTIONS]
    if unknown:
        parser.error(f"unknown sections: {', '.join(unknown)}")

    dataset = RunDataset(args.run, analysis_dir=args.analysis_dir)
    if args.print_status:
        print(json.dumps(dataset.status(), indent=2, sort_keys=True))
        return 0
    path = build_report(dataset, only=only or None)
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
