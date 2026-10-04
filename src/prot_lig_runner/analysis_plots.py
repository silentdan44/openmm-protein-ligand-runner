"""Plot and HTML report generation for trajectory analysis."""

from __future__ import annotations

import csv
import html
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)


def _save_optional_plot(path, name, draw):
    """Create one figure; absent/invalid optional data never aborts the report."""
    path = Path(path)
    target = path / name
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = draw(plt)
        if fig is None:
            return False
        fig.savefig(target, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as error:
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception:
            pass
        target.unlink(missing_ok=True)
        logger.warning("Skipping optional chart %s: %s", name, error)
        return False


def _axis(ax, title, ylabel, xlabel="Time (ns)"):
    ax.set(title=title, ylabel=ylabel, xlabel=xlabel)
    ax.spines[["top", "right"]].set_visible(False)


def _annotate_statistics(ax, series, labels, unit):
    """Add compact mean/median/sample-SD summaries to a time-series axis."""
    import numpy as np

    entries = []
    for values, label in zip(series, labels):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        entries.append(
            f"{label}: mean {np.mean(values):.3g}, median {np.median(values):.3g}, "
            f"SD {std:.2g} {unit}"
        )
    if entries:
        ax.text(0.01, 0.99, "\n".join(entries), transform=ax.transAxes,
                ha="left", va="top", fontsize=8,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
                      "edgecolor": "#cbd5e1", "alpha": 0.92})


def _plot_distributions(ax, series, labels, *, colors=None, bins="auto"):
    """Plot overlaid horizontal histograms for comparable measured series."""
    import numpy as np

    colors = colors or [None] * len(series)
    for values, label, color in zip(series, labels, colors):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        ax.hist(values, bins=bins, orientation="horizontal", density=True,
                histtype="step", linewidth=1.5, label=label, color=color)
    ax.set_xlabel("Density")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelleft=False, left=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)


def _plot_qc(output_dir, state_rows, config):
    if not state_rows:
        return False
    import numpy as np

    def draw(plt):
        keys = list(state_rows[0])
        def column(part):
            key = next((candidate for candidate in keys if part.lower() in candidate.lower()), None)
            return np.asarray([row.get(key, math.nan) for row in state_rows], dtype=float) if key else None

        steps = column("Step")
        if steps is None:
            return None
        time_ns = steps * float(config["timestep_fs"]) / 1_000_000.0
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        temperature = column("Temperature")
        if temperature is not None:
            axes[0, 0].plot(time_ns, temperature, color="#2563eb", linewidth=1)
            axes[0, 0].axhline(config["temperature_k"], color="#dc2626", linestyle="--", linewidth=1)
        _axis(axes[0, 0], "Temperature", "K")

        energy = column("Potential Energy")
        if energy is not None:
            axes[0, 1].plot(time_ns, energy, color="#7c3aed", linewidth=1)
        _axis(axes[0, 1], "Potential energy", "kJ/mol")

        npt = (steps > config["nvt_steps"]) & (steps <= config["nvt_steps"] + config["npt_steps"])
        density, volume = column("Density"), column("Box Volume")
        if not np.any(npt):
            axes[1, 0].text(0.5, 0.5, "No NPT samples", ha="center", va="center",
                            transform=axes[1, 0].transAxes)
            axes[1, 1].text(0.5, 0.5, "No NPT samples", ha="center", va="center",
                            transform=axes[1, 1].transAxes)
        if density is not None and np.any(npt):
            axes[1, 0].plot(time_ns[npt], density[npt], color="#059669", linewidth=1)
        _axis(axes[1, 0], "NPT density", "g/mL")
        if volume is not None and np.any(npt):
            axes[1, 1].plot(time_ns[npt], volume[npt], color="#ea580c", linewidth=1)
        _axis(axes[1, 1], "NPT box volume", "nm³")
        return fig

    return _save_optional_plot(output_dir, "simulation_qc.png", draw)


def _plot_trajectory(output_dir, data):
    if not data:
        return []
    import numpy as np

    generated = []
    time = data["time_ns"]

    def stability(plt):
        fig, axes = plt.subplots(2 if data.get("membrane_core_rmsd_nm") is not None else 1,
                                 1, figsize=(10, 7 if data.get("membrane_core_rmsd_nm") is not None else 4.5),
                                 squeeze=False, constrained_layout=True)
        axes = axes[:, 0]
        axes[0].plot(time, data["protein_rmsd_nm"], color="#2563eb", linewidth=1)
        _axis(axes[0], "Whole-protein backbone RMSD", "nm")
        if data.get("membrane_core_rmsd_nm") is not None:
            axes[1].plot(time, data["membrane_core_rmsd_nm"], color="#0f766e", linewidth=1)
            _axis(axes[1], "Membrane-region Cα RMSD", "nm")
        return fig

    def rmsf(plt):
        chains = data.get("rmsf_by_chain", {})
        if not chains:
            return None
        fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
        for chain_id, chain_data in chains.items():
            ax.plot(chain_data["residue"], chain_data["rmsf_nm"], linewidth=1, label=f"Chain {chain_id}")
        if len(chains) > 1:
            ax.legend(frameon=False, ncol=2)
        _axis(ax, "Cα RMSF by residue", "RMSF (nm)", "Residue number")
        return fig

    def ligand(plt):
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
        axes[0].plot(time, data["ligand_rmsd_nm"], color="#dc2626", linewidth=1)
        _axis(axes[0], "Ligand RMSD after protein alignment", "nm")
        axes[1].plot(time, data["ligand_contacts"], color="#c2410c", linewidth=1,
                     label="Residue contacts < 0.45 nm")
        axes[1].plot(time, data["hbond_counts"], color="#7c3aed", linewidth=1,
                     label="Protein–ligand hydrogen bonds")
        axes[1].legend(frameon=False)
        _axis(axes[1], "Ligand pocket contacts", "Count")
        return fig

    for name, draw in (("protein_stability.png", stability),
                       ("protein_rmsf.png", rmsf), ("ligand_pocket.png", ligand)):
        if _save_optional_plot(output_dir, name, draw):
            generated.append(name)

    contacts = data.get("contact_occupancy", [])
    if contacts:
        def contact_plot(plt):
            rows = contacts[:20][::-1]
            fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.28)), constrained_layout=True)
            labels = [row["protein_residue"] for row in rows]
            ax.barh(labels, [row["occupancy"] for row in rows], color="#0f766e")
            ax.set_xlim(0, 1)
            _axis(ax, "Protein–ligand contact occupancy (top 20)", "Fraction of frames", "Residue")
            return fig
        if _save_optional_plot(output_dir, "contact_occupancy.png", contact_plot):
            generated.append("contact_occupancy.png")

    hbonds = data.get("hbond_occupancy", [])
    if hbonds:
        def hbond_plot(plt):
            rows = hbonds[:20][::-1]
            labels = [f'{row["protein_residue"]} {row["protein_atom"]}–{row["ligand_atom"]}' for row in rows]
            fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.3)), constrained_layout=True)
            ax.barh(labels, [row["occupancy"] for row in rows], color="#7c3aed")
            ax.set_xlim(0, 1)
            _axis(ax, "Protein–ligand hydrogen-bond occupancy", "Fraction of frames", "Interaction")
            return fig
        if _save_optional_plot(output_dir, "hydrogen_bond_occupancy.png", hbond_plot):
            generated.append("hydrogen_bond_occupancy.png")

    membrane = data.get("membrane_metrics")
    if membrane:
        def membrane_plot(plt):
            fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
            axes[0].plot(time, membrane["membrane_area_per_lipid_nm2"], color="#2563eb", linewidth=1)
            _axis(axes[0], "Area per lipid", "nm²")
            axes[1].plot(time, membrane["membrane_thickness_nm"], color="#059669", linewidth=1)
            _axis(axes[1], "Bilayer thickness (phosphate-plane separation)", "nm")
            axes[2].plot(time, membrane["membrane_lower_lipid_count"], label="Lower leaflet", linewidth=1)
            axes[2].plot(time, membrane["membrane_upper_lipid_count"], label="Upper leaflet", linewidth=1)
            axes[2].legend(frameon=False)
            _axis(axes[2], "Leaflet lipid counts", "Lipids")
            return fig
        if _save_optional_plot(output_dir, "membrane_integrity.png", membrane_plot):
            generated.append("membrane_integrity.png")

    distances = data.get("distance_series", {})
    if distances:
        def distance_plot(plt):
            labels = list(distances)
            series = [distances[label] for label in labels]
            fig, (ax, dist_ax) = plt.subplots(
                1, 2, figsize=(12, 5), sharey=True, constrained_layout=True,
                gridspec_kw={"width_ratios": [4, 1]},
            )
            colors = []
            for label, values in zip(labels, series):
                line, = ax.plot(time, values, linewidth=1, label=label)
                colors.append(line.get_color())
            ax.legend(frameon=False, fontsize=8, loc="best")
            _axis(ax, "Selected residue-pair Cα distances", "Distance (nm)")
            _annotate_statistics(ax, series, labels, "nm")
            _plot_distributions(dist_ax, series, labels, colors=colors)
            return fig
        if _save_optional_plot(output_dir, "residue_distances.png", distance_plot):
            generated.append("residue_distances.png")
    return generated


def read_mmpbsa_delta_csv(path):
    """Read the Delta Energy Terms frame table from gmx_MMPBSA's multi-section CSV."""
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        return None
    with path.open(newline="", errors="replace") as handle:
        rows = list(csv.reader(handle))
    section = next((i for i, row in enumerate(rows)
                    if row and "delta energy terms" in row[0].strip().lower()), None)
    if section is None:
        return None
    header_index = next((i for i in range(section + 1, len(rows))
                         if rows[i] and rows[i][0].strip().lower().startswith("frame")), None)
    if header_index is None:
        return None
    header = [value.strip() for value in rows[header_index]]
    data = []
    for row in rows[header_index + 1:]:
        if not row or len(row) < len(header):
            break
        try:
            data.append({key: float(value) for key, value in zip(header, row)})
        except ValueError:
            break
    return data or None


def _plot_mmpbsa(output_dir, mmpbsa_csv, decomposition_csv=None):
    try:
        rows = read_mmpbsa_delta_csv(mmpbsa_csv)
    except Exception as error:
        logger.warning("Skipping optional MMPBSA plots: could not parse %s: %s", mmpbsa_csv, error)
        return []
    if not rows:
        return []
    import numpy as np

    frame_key = next(key for key in rows[0] if key.lower().startswith("frame"))
    energy_keys = [key for key in rows[0] if key != frame_key and key.upper() in {
        "VDWAALS", "EEL", "EGB", "EPB", "ESURF", "ENPOLAR", "GGAS", "GSOLV", "TOTAL"
    }]
    if not energy_keys:
        return []
    generated = []

    def components(plt):
        labels = energy_keys
        series = [[row[key] for row in rows] for key in labels]
        fig, (ax, dist_ax) = plt.subplots(
            1, 2, figsize=(13, 5.5), sharey=True, constrained_layout=True,
            gridspec_kw={"width_ratios": [4, 1]},
        )
        colors = []
        frames = [row[frame_key] for row in rows]
        for key, values in zip(labels, series):
            line, = ax.plot(frames, values, linewidth=1, label=key)
            colors.append(line.get_color())
        ax.legend(frameon=False, ncol=2)
        _axis(ax, "MMPBSA energy terms by analyzed frame", "Energy (kcal/mol)", "Frame")
        _annotate_statistics(ax, series, labels, "kcal/mol")
        _plot_distributions(dist_ax, series, labels, colors=colors)
        return fig
    if _save_optional_plot(output_dir, "mmpbsa_energy_terms.png", components):
        generated.append("mmpbsa_energy_terms.png")

    def spread(plt):
        values = [[row[key] for row in rows] for key in energy_keys]
        means = [float(np.mean(series)) for series in values]
        deviations = [float(np.std(series, ddof=1)) if len(series) > 1 else 0.0 for series in values]
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        positions = np.arange(len(energy_keys))
        ax.errorbar(positions, means, yerr=deviations, fmt="o", capsize=4, color="#1d4ed8")
        ax.axhline(0, color="#64748b", linewidth=0.8)
        ax.set_xticks(positions, energy_keys, rotation=25, ha="right")
        _axis(ax, "MMPBSA mean ± frame standard deviation", "Energy (kcal/mol)", "Energy term")
        return fig
    if _save_optional_plot(output_dir, "mmpbsa_energy_summary.png", spread):
        generated.append("mmpbsa_energy_summary.png")

    if decomposition_csv:
        try:
            decomp_rows = _read_decomposition(decomposition_csv)
        except Exception as error:
            logger.warning("Skipping optional MMPBSA decomposition plot: %s", error)
            decomp_rows = None
        if decomp_rows:
            totals = {}
            for row in decomp_rows:
                if "Residue" not in row or "TOTAL" not in row:
                    continue
                totals.setdefault(row["Residue"], []).append(row["TOTAL"])
            ranked = sorted(((name, float(np.mean(values)),
                             float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
                             for name, values in totals.items()), key=lambda item: abs(item[1]), reverse=True)[:20]
            if ranked:
                def residue_decomposition(plt):
                    rows_plot = ranked[::-1]
                    fig, ax = plt.subplots(figsize=(10, max(4, len(rows_plot) * 0.3)), constrained_layout=True)
                    y = np.arange(len(rows_plot))
                    ax.barh(y, [row[1] for row in rows_plot], color=["#dc2626" if row[1] > 0 else "#0f766e" for row in rows_plot])
                    ax.errorbar([row[1] for row in rows_plot], y, xerr=[row[2] for row in rows_plot],
                                fmt="none", ecolor="#1f2937", capsize=2, linewidth=0.8)
                    ax.set_yticks(y, [row[0] for row in rows_plot])
                    ax.axvline(0, color="#64748b", linewidth=0.8)
                    ax.set(title="Largest per-residue MMPBSA contributions",
                           xlabel="Contribution (kcal/mol)", ylabel="Residue")
                    ax.spines[["top", "right"]].set_visible(False)
                    return fig
                if _save_optional_plot(output_dir, "mmpbsa_residue_decomposition.png", residue_decomposition):
                    generated.append("mmpbsa_residue_decomposition.png")
    return generated


def _read_decomposition(path):
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        return None
    with path.open(newline="", errors="replace") as handle:
        rows = list(csv.reader(handle))
    result = []
    active = False
    header = None
    for row in rows:
        first = row[0].strip().lower() if row else ""
        if "total decomposition contribution" in first:
            active = True
            header = None
            continue
        if not active:
            continue
        if row and first.startswith("frame") and any(cell.strip().lower() == "residue" for cell in row):
            header = [cell.strip() for cell in row]
            continue
        if not row or len(row) < 2:
            active = False
            continue
        if header is None:
            continue
        try:
            record = dict(zip(header, row))
            record["Frame #"] = float(record["Frame #"])
            for key in header:
                if key not in {"Frame #", "Residue"}:
                    record[key] = float(record[key])
            result.append(record)
        except (KeyError, ValueError):
            active = False
    return result or None


def build_report(output_dir, metrics, generated_plots, *, warnings=None):
    output_dir = Path(output_dir)
    summary_items = "".join(
        f"<li><strong>{html.escape(key.replace('_', ' '))}:</strong> {value:.3f}</li>"
        for key, value in metrics.get("trajectory_metrics", {}).items()
    )
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in (warnings or []))
    links = "\n".join(
        f'<section><h2>{html.escape(Path(name).stem.replace("_", " ").title())}</h2>'
        f'<p><img src="{html.escape(name)}" alt="{html.escape(name)}" style="max-width:100%"></p></section>'
        for name in generated_plots
    )
    report = "".join((
        "<!doctype html><meta charset='utf-8'><title>MD analysis report</title>",
        "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#172033}"
        "img{border:1px solid #dbe2ea;border-radius:8px;max-width:100%}section{margin:2rem 0}</style>",
        "<h1>Protein–ligand MD analysis</h1>",
        f"<p>Production frames analyzed: {metrics.get('frame_count', 0)}; "
        f"state samples: {metrics.get('state_sample_count', 0)}.</p>",
        f"<ul>{summary_items}</ul>" if summary_items else "",
        "<p>Frame-to-frame spread is descriptive; frames are correlated and do not replace independent replicas.</p>",
        f"<h2>Analysis notes</h2><ul>{notes}</ul>" if notes else "",
        links if links else "<p>No optional plots were available for this run.</p>",
    ))
    (output_dir / "report.html").write_text(report)
