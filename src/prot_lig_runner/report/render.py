"""Assemble an HTML report from a run dataset."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

from .dataset import RunDataset, SECTIONS, SECTION_TITLES, figure_title
from .pdf import build_pdf

SECTION_PATTERNS = {
    "qc": ("simulation_qc*.png", "pressure*.png", "temperature*.png"),
    "stability": ("protein_*.png", "rmsf*.png", "distance*.png"),
    "ligands": ("ligand_*.png", "contact_*.png", "hydrogen_bond_*.png"),
    "membrane": ("membrane_*.png",),
    "mmpbsa": ("mmpbsa_*.png",),
}


def collect_figures(dataset: RunDataset) -> dict[str, list[Path]]:
    """Group existing plot files by section, newest names last."""
    figures: dict[str, list[Path]] = {}
    for section in SECTIONS:
        seen: list[Path] = []
        for pattern in SECTION_PATTERNS[section]:
            for path in sorted(dataset.analysis.glob(pattern)):
                if path not in seen:
                    seen.append(path)
        # The combined ligand figure is superseded by the per-ligand ones.
        per_ligand = [path for path in seen if path.stem.startswith("ligand_pocket_LIG")]
        if per_ligand:
            seen = [path for path in seen if path.stem != "ligand_pocket"]
        figures[section] = seen
    return figures


def _status_block(status: dict) -> str:
    availability = status["availability"]
    badges = "".join(
        f'<span class="badge {"ok" if available else "missing"}">'
        f'{html.escape(SECTION_TITLES[section])}: {"ready" if available else "waiting"}</span>'
        for section, available in availability.items()
    )
    planned = status["planned_ns"]
    fraction = status["fraction"]
    if fraction is None:
        progress = f'{status["simulated_ns"]:.2f} ns simulated'
    else:
        progress = (
            f'{status["simulated_ns"]:.2f} of {planned:.2f} ns '
            f'({100 * fraction:.1f}%)'
        )
    return (
        '<section class="status">'
        f'<p><strong>Phase:</strong> {html.escape(status["phase"])}'
        f' · <strong>Progress:</strong> {progress}'
        f' · <strong>Frames:</strong> {status["frames"]}'
        f'{" · <strong>preliminary</strong>" if status["preliminary"] else ""}</p>'
        f'<p>{badges}</p>'
        "</section>"
    )


def _ligand_table(metrics: dict) -> str:
    ligands = (metrics.get("trajectory_metrics") or {}).get("ligands") or {}
    if not ligands:
        return ""
    rows = []
    for label, stats in ligands.items():
        pose = stats.get("ligand_pose_rmsd_nm_mean")
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(str(stats.get('residue', '')))}</td>"
            f"<td>{stats.get('ligand_rmsd_nm_mean', 0):.2f}</td>"
            f"<td>{'—' if pose is None else f'{pose:.2f}'}</td>"
            f"<td>{stats.get('ligand_contacts_mean', 0):.1f}</td>"
            f"<td>{stats.get('protein_ligand_hbonds_mean', 0):.1f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Ligand</th><th>Residue</th>"
        "<th>RMSD vs start (nm)</th><th>RMSD vs input (nm)</th>"
        "<th>Contacts</th><th>H-bonds</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _mmpbsa_table(summary: dict | None) -> str:
    if not summary:
        return "<p>MMPBSA has not run yet for this trajectory.</p>"
    variants = summary.get("entropy_variants") or {}
    if not variants:
        return (
            "<p>ΔH = {:.2f} kcal/mol (enthalpy only, {} frames).</p>".format(
                summary.get("delta_h_mean_kcal_mol", 0.0), summary.get("frame_count", 0)
            )
        )
    def _fmt(value):
        return "—" if value is None else f"{value:.2f}"

    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(variant.get('name', '')).upper())}</td>"
        f"<td>{summary.get('delta_h_mean_kcal_mol', 0.0):.2f}</td>"
        f"<td>{variant.get('minus_tds_kcal_mol', 0.0):.2f}</td>"
        f"<td>{variant.get('delta_g_kcal_mol', 0.0):.2f}</td>"
        f"<td>{_fmt(variant.get('sem_delta_g_kcal_mol'))}</td>"
        "</tr>"
        for variant in variants.values()
    )
    return (
        "<table><thead><tr><th>Entropy</th><th>ΔH</th><th>−TΔS</th>"
        f"<th>ΔG</th><th>SEM ΔG</th></tr></thead><tbody>{rows}</tbody></table>"
        f"<p>{summary.get('frame_count', 0)} frames at "
        f"{summary.get('temperature_k', 300):.0f} K.</p>"
    )


def build_report(dataset: RunDataset, *, only: tuple[str, ...] | None = None,
                 warnings: list[str] | None = None) -> Path:
    """Write ``report.html`` next to the analysis data and return its path."""
    figures = collect_figures(dataset)
    status = dataset.status()
    sections = [section for section in SECTIONS if not only or section in only]

    parts: list[str] = []
    for section in sections:
        available = status["availability"][section]
        body = ""
        if section == "ligands":
            body += "<h3>Per-ligand summary</h3>" + _ligand_table(dataset.metrics)
        if section == "mmpbsa":
            body += _mmpbsa_table(dataset.mmpbsa)
        if figures[section]:
            body += "".join(
                f'<figure><img src="{html.escape(path.name)}" alt="{html.escape(path.name)}">'
                f"<figcaption>{html.escape(figure_title(path))}</figcaption></figure>"
                for path in figures[section]
            )
        elif not available:
            body += "<p>Waiting for data from this stage.</p>"
        parts.append(
            f'<section id="{section}"><h2>{html.escape(SECTION_TITLES[section])}</h2>{body}</section>'
        )

    notes = "".join(f"<li>{html.escape(note)}</li>" for note in (warnings or []))
    notes_block = f'<section><h2>Notes</h2><ul>{notes}</ul></section>' if notes else ""
    nav = " · ".join(
        f'<a href="#{section}">{html.escape(SECTION_TITLES[section])}</a>' for section in sections
    )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run report</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; max-width: 1100px;
        margin: 0 auto; padding: 24px; line-height: 1.5; }}
h1 {{ margin-bottom: 4px; }}
.meta {{ color: #666; font-size: 0.9em; }}
nav {{ margin: 16px 0 24px; }}
nav a {{ margin-right: 8px; }}
.status {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; margin-bottom: 24px; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; margin: 2px 6px 2px 0;
          font-size: 0.8em; border: 1px solid currentColor; }}
.badge.ok {{ color: #067647; }}
.badge.missing {{ color: #b54708; opacity: 0.8; }}
figure {{ margin: 0 0 24px; }}
figure img {{ max-width: 100%; border: 1px solid #eee; border-radius: 6px; }}
figcaption {{ color: #555; font-size: 0.85em; }}
table {{ border-collapse: collapse; margin: 12px 0 20px; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 6px 12px; text-align: left; font-size: 0.9em; }}
section {{ margin-bottom: 40px; }}
</style>
</head>
<body>
<h1>Run report</h1>
<p class="meta">{html.escape(str(dataset.root))} · generated {generated} ·
<a href="report.pdf">download PDF</a></p>
{nav}
{_status_block(status)}
{notes_block}
{''.join(parts)}
</body>
</html>
"""
    path = dataset.analysis / "report.html"
    dataset.analysis.mkdir(parents=True, exist_ok=True)
    path.write_text(document)

    build_pdf(dataset, {section: figures[section] for section in sections}, warnings=warnings)

    manifest = {
        "generated_utc": generated,
        "root": str(dataset.root),
        "status": status,
        "sections": {
            section: [path.name for path in figures[section]] for section in sections
        },
    }
    (dataset.analysis / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return path
