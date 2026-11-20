"""PDF export of a run report, for reading on a phone or printing."""

from __future__ import annotations

import html
from pathlib import Path

from .dataset import RunDataset, SECTION_TITLES, figure_caption, figure_title


def _summary_rows(dataset: RunDataset):
    ligands = (dataset.metrics.get("trajectory_metrics") or {}).get("ligands") or {}
    rows = []
    for label, stats in ligands.items():
        rows.append([
            label,
            str(stats.get("residue", "")),
            f"{stats.get('ligand_rmsd_nm_mean', 0):.2f}",
            f"{stats.get('ligand_contacts_mean', 0):.1f}",
            f"{stats.get('protein_ligand_hbonds_mean', 0):.1f}",
        ])
    return rows


def _mmpbsa_rows(dataset: RunDataset):
    summary = dataset.mmpbsa
    if not summary:
        return [], ""
    ligand = summary.get("ligand")
    title = "MMPBSA free-energy estimate (kcal/mol)"
    if ligand:
        residue = summary.get("ligand_residue") or ""
        title += f" — {ligand}" + (f" ({residue})" if residue else "")
    rows = []
    for variant in (summary.get("entropy_variants") or {}).values():
        sem = variant.get("sem_delta_g_kcal_mol")
        rows.append([
            str(variant.get("name", "")).upper(),
            f"{summary.get('delta_h_mean_kcal_mol', 0):.2f}",
            f"{variant.get('minus_tds_kcal_mol', 0):.2f}",
            f"{variant.get('delta_g_kcal_mol', 0):.2f}",
            "—" if sem is None else f"{sem:.2f}",
        ])
    return rows, title


def _draw_table(figure, rows, header, *, top, height, left=0.08, width=0.62, fontsize=7):
    if not rows:
        return
    axes = figure.add_axes([left, top - height, width, height])
    axes.axis("off")
    table = axes.table(cellText=rows, colLabels=header, loc="upper left", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, 1.25)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor("0.75")
        if row == 0:
            cell.set_text_props(weight="bold")


def _draw_summary(figure, dataset: RunDataset, warnings: list[str], total_pages: int):
    status = dataset.status()
    planned = status["planned_ns"]
    progress = (f'{status["simulated_ns"]:.2f} of {planned:.2f} ns '
                f'({100 * status["fraction"]:.1f}%), {status["frames"]} frames'
                if planned else f'{status["simulated_ns"]:.2f} ns, {status["frames"]} frames')
    figure.text(0.08, 0.94, "Run report", fontsize=16, weight="bold", va="top")
    figure.text(0.08, 0.905, str(dataset.root), fontsize=8, color="0.35", va="top")
    figure.text(0.08, 0.875, f"{status['phase']} · {progress}", fontsize=9, va="top")

    metrics = dataset.metrics.get("trajectory_metrics") or {}
    keys = [
        ("Backbone RMSD", metrics.get("protein_backbone_rmsd_nm_mean"), "nm"),
        ("Radius of gyration", metrics.get("protein_rg_nm_mean"), "nm"),
        ("Ligand contacts", metrics.get("ligand_protein_contacts_mean"), ""),
        ("Protein-ligand H-bonds", metrics.get("protein_ligand_hbonds_mean"), ""),
        ("Membrane thickness", metrics.get("membrane_thickness_nm_mean"), "nm"),
    ]
    lines = [f"{name}: {value:.2f} {unit}".strip() for name, value, unit in keys
             if value is not None]
    if lines:
        figure.text(0.08, 0.83, "  ·  ".join(lines), fontsize=7.5, va="top", color="0.2")

    _draw_table(figure, _summary_rows(dataset),
                ["Ligand", "Residue", "RMSD (nm)", "Contacts", "H-bonds"],
                top=0.78, height=0.05 + 0.022 * len(_summary_rows(dataset)))
    mmpbsa_rows, title = _mmpbsa_rows(dataset)
    table_top = 0.78 - (0.07 + 0.022 * max(1, len(_summary_rows(dataset))))
    if mmpbsa_rows:
        figure.text(0.08, table_top + 0.02, title, fontsize=9, weight="bold", va="bottom")
        _draw_table(figure, mmpbsa_rows, ["Entropy", "dH", "-TdS", "dG", "SEM dG"],
                    top=table_top, height=0.04 + 0.022 * len(mmpbsa_rows))
    if warnings:
        figure.text(0.08, 0.16, "Notes", fontsize=8, weight="bold", va="top")
        figure.text(0.08, 0.14, "\n".join(f"- {note}" for note in warnings),
                    fontsize=7, va="top", color="0.25")
    figure.text(0.92, 0.03, f"1 / {total_pages}", fontsize=8, ha="right", color="0.4")


def build_pdf(dataset: RunDataset, figures: dict[str, list[Path]], *,
              warnings: list[str] | None = None, path: Path | None = None) -> Path:
    """Render a summary page plus one page per figure into ``report.pdf``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.image import imread

    path = Path(path) if path is not None else dataset.analysis / "report.pdf"
    dataset.analysis.mkdir(parents=True, exist_ok=True)
    pages = [(section, image_path) for section, paths in figures.items() for image_path in paths]
    total_pages = len(pages) + 1
    with PdfPages(path) as pdf:
        figure = plt.figure(figsize=(8.27, 11.69))
        _draw_summary(figure, dataset, warnings or [], total_pages)
        pdf.savefig(figure)
        plt.close(figure)

        page_number = 2
        for section, image_path in pages:
                try:
                    image = imread(image_path)
                except (OSError, ValueError, SyntaxError):
                    continue
                height, width = image.shape[:2]
                figure = plt.figure(figsize=(8.27, 11.69))
                figure.text(
                    0.06, 0.955,
                    f"{SECTION_TITLES.get(section, section)}: {figure_caption(image_path, dataset)}",
                    fontsize=11, weight="bold", va="top",
                )
                # Keep the plot's aspect ratio inside the A4 page.
                page_ratio = 0.86 * 8.27 / (0.84 * 11.69)
                image_ratio = width / height
                if image_ratio > page_ratio:
                    box = (0.05, 0.10, 0.90, 0.80)
                else:
                    box = (0.14, 0.10, 0.72, 0.80)
                axes = figure.add_axes(box)
                axes.imshow(image)
                axes.axis("off")
                figure.text(0.92, 0.03, f"{page_number} / {total_pages}", fontsize=8, ha="right", color="0.4")
                pdf.savefig(figure)
                plt.close(figure)
                page_number += 1
    return path
