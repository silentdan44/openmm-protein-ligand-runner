"""Dataset view of a run directory, usable while the simulation is running."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SECTIONS = ("qc", "stability", "ligands", "membrane", "mmpbsa")

SECTION_TITLES = {
    "qc": "Simulation QC",
    "stability": "Protein stability",
    "ligands": "Ligands",
    "membrane": "Membrane",
    "mmpbsa": "MMPBSA",
}

FIGURE_TITLES = {
    "simulation_qc": "Temperature, pressure and energy traces",
    "protein_stability": "Backbone RMSD and radius of gyration",
    "protein_rmsf": "Per-residue RMSF",
    "ligand_comparison": "Ligand RMSD, contacts and hydrogen bonds",
    "ligand_pocket": "Ligand pocket (all ligands combined)",
    "contact_occupancy": "Protein-ligand contact occupancy",
    "hydrogen_bond_occupancy": "Hydrogen-bond occupancy",
    "membrane_integrity": "Membrane thickness, area per lipid and lipid counts",
    "mmpbsa_energy_terms": "MMPBSA energy terms",
    "mmpbsa_energy_summary": "MMPBSA per-frame energies",
    "mmpbsa_entropy": "MMPBSA running enthalpy, entropy and free energy",
    "mmpbsa_residue_decomposition": "MMPBSA per-residue decomposition",
    "residue_distances": "Selected residue-residue distances",
}


def figure_caption(path: Path, dataset: RunDataset) -> str:
    """Human caption, naming the ligand whenever the figure is ligand-specific."""
    stem = path.stem
    ligands = (dataset.metrics.get("trajectory_metrics") or {}).get("ligands") or {}
    for label, stats in ligands.items():
        residue = stats.get("residue", "")
        if stem.endswith(f"_{label}"):
            if stem.startswith("ligand_structure"):
                return f"{label} ({residue}): 2D structure"
            if stem.startswith("ligand_pocket"):
                return f"{label} ({residue}): RMSD, contacts and hydrogen bonds"
    return figure_title(path)


def figure_title(path) -> str:
    return FIGURE_TITLES.get(path.stem, path.stem.replace("_", " ").capitalize())


@dataclass
class RunDataset:
    """Read-only view of everything a report may need.

    Every accessor tolerates missing files: a report built mid-run simply has
    fewer sections, and ``status`` says what is missing and why.
    """

    root: Path
    analysis_dir: Path | None = None
    _time_series: list[dict[str, Any]] | None = field(default=None, repr=False)
    _metrics: dict[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self):
        self.root = Path(self.root)
        if self.analysis_dir is None:
            self.analysis_dir = self.root / "analysis"
        self.analysis_dir = Path(self.analysis_dir)

    @property
    def analysis(self) -> Path:
        return Path(self.analysis_dir or self.root / "analysis")

    # -- raw files ---------------------------------------------------------
    @property
    def state(self) -> dict[str, Any] | None:
        path = self.root / "run_state.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    @property
    def metrics(self) -> dict[str, Any]:
        if self._metrics is None:
            path = self.analysis / "metrics.json"
            self._metrics = json.loads(path.read_text()) if path.is_file() else {}
        return self._metrics

    @property
    def time_series(self) -> list[dict[str, Any]]:
        if self._time_series is None:
            path = self.analysis / "time_series.csv"
            rows: list[dict[str, Any]] = []
            if path.is_file():
                with path.open(newline="") as handle:
                    for row in csv.DictReader(handle):
                        parsed = {}
                        for key, value in row.items():
                            try:
                                parsed[key] = float(value)
                            except (TypeError, ValueError):
                                parsed[key] = value
                        rows.append(parsed)
            self._time_series = rows
        return self._time_series

    @property
    def mmpbsa(self) -> dict[str, Any] | None:
        # Only trust MMPBSA output that the current analysis has folded into
        # metrics.json. A stale analysis/mmpbsa/entropy.json from an earlier
        # analysis run must not be presented as part of this report.
        if not (self.metrics.get("mmpbsa")):
            return None
        path = self.analysis / "mmpbsa" / "entropy.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    # -- derived state -----------------------------------------------------
    @property
    def configuration(self) -> dict[str, Any]:
        state = self.state or {}
        return state.get("configuration") or {}

    @property
    def phase(self) -> str:
        metrics = self.metrics
        if metrics.get("snapshot_phase"):
            return str(metrics["snapshot_phase"])
        if metrics.get("preliminary"):
            return "preliminary snapshot"
        planned, simulated = self.planned_ns, self.simulated_ns
        if planned and simulated and simulated < planned * 0.999:
            return "production (partial trajectory)"
        if planned:
            return "finished"
        state = self.state or {}
        status = str(state.get("status") or "").lower()
        if status in {"finished", "complete", "completed"}:
            return "finished"
        if status in {"running", "production"}:
            return "production"
        if status:
            return status
        return "unknown"

    @property
    def simulated_ns(self) -> float:
        rows = self.time_series
        if rows and "time_ns" in rows[-1]:
            try:
                return float(rows[-1]["time_ns"])
            except (TypeError, ValueError):
                pass
        return float(self.metrics.get("production_start_ns") or 0.0)

    @property
    def planned_ns(self) -> float | None:
        config = self.configuration
        steps = config.get("prod_steps")
        dt_ps = config.get("timestep_ps") or config.get("timestep_fs", 0) / 1000.0
        if steps and dt_ps:
            return float(steps) * float(dt_ps) / 1000.0
        return None

    def has_membrane(self) -> bool:
        metrics = self.metrics.get("trajectory_metrics") or {}
        return any(key.startswith("membrane_") for key in metrics)

    def availability(self) -> dict[str, bool]:
        """What each section can show right now."""
        return {
            "qc": bool(self.metrics) or bool(self.time_series),
            "stability": bool(self.time_series),
            "ligands": bool((self.metrics.get("trajectory_metrics") or {}).get("ligands")),
            "membrane": self.has_membrane(),
            "mmpbsa": self.mmpbsa is not None,
        }

    def status(self) -> dict[str, Any]:
        planned = self.planned_ns
        simulated = self.simulated_ns
        return {
            "phase": self.phase,
            "preliminary": bool(self.metrics.get("preliminary")),
            "simulated_ns": simulated,
            "planned_ns": planned,
            "fraction": (simulated / planned) if planned else None,
            "frames": len(self.time_series),
            "availability": self.availability(),
            "temperature_k": self.metrics.get("temperature_k"),
        }
