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
            "",
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


def _draw_table(figure, rows, header, *, top, height, left=0.06, width=0.72, fontsize=7,
                images=None, col_widths=None):
    if not rows:
        return
    axes = figure.add_axes([left, top - height, width, height])
    axes.axis("off")
    table = axes.table(cellText=rows, colLabels=header, loc="upper left", cellLoc="left",
                       colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, 1.25)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor("0.75")
        if row == 0:
            cell.set_text_props(weight="bold")
    if images:
        from matplotlib.image import imread

        bbox = axes.get_position()
        for row_index, image_path in images.items():
            cell = table[(row_index, 0)]
            x, y, width_cell, height_cell = (
                cell.get_x(), cell.get_y(), cell.get_width(), cell.get_height()
            )
            if not image_path.is_file():
                continue
            try:
                image = imread(image_path)
            except (OSError, ValueError, SyntaxError):
                continue
            inset = figure.add_axes([
                bbox.x0 + x * bbox.width + 0.005,
                bbox.y0 + y * bbox.height + 0.004,
                width_cell * bbox.width * 0.95,
                height_cell * bbox.height * 0.9,
            ])
            inset.imshow(image)
            inset.axis("off")
    return table


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

    ligand_rows = _summary_rows(dataset)
    ligand_images = {
        index: dataset.analysis / f"ligand_structure_{row[1]}.png"
        for index, row in enumerate(ligand_rows, start=1)
    }
    table_height = 0.05 + 0.06 * len(ligand_rows)
    table_obj = _draw_table(figure, ligand_rows,
                ["", "Ligand", "Residue", "RMSD (nm)", "Contacts", "H-bonds"],
                top=0.80, height=table_height,
                col_widths=[0.26, 0.12, 0.18, 0.14, 0.14, 0.14])
    figure.canvas.draw()
    table_bbox = (table_obj.get_window_extent(figure.canvas.get_renderer()).transformed(
        figure.transFigure.inverted()
    ) if table_obj is not None else None)
    # Draw the structure thumbnails in the reserved first column; positions
    # come from the table geometry so they line up with their rows.
    import matplotlib.image as mpimg

    row_height = (table_bbox.height / (len(ligand_rows) + 1)) if table_bbox else 0
    for index, row in (enumerate(ligand_rows, start=1) if table_bbox else []):
        thumb = dataset.analysis / f"ligand_structure_{row[1]}_thumb.png"
        image_path = thumb if thumb.is_file() else dataset.analysis / f"ligand_structure_{row[1]}.png"
        if not image_path.is_file():
            continue
        try:
            image = mpimg.imread(image_path)
        except (OSError, ValueError, SyntaxError):
            continue
        y = table_bbox.y1 - (index + 0.93) * row_height
        inset = figure.add_axes([table_bbox.x0 + 0.004, y, 0.21, row_height * 0.9])
        inset.imshow(image)
        inset.axis("off")
    table_bottom = 0.80 - (0.05 + 0.115 * len(ligand_rows))
    mmpbsa_rows, title = _mmpbsa_rows(dataset)
    table_top = table_bottom - 0.04
    if mmpbsa_rows:
        figure.text(0.08, table_top + 0.02, title, fontsize=9, weight="bold", va="bottom")
        _draw_table(figure, mmpbsa_rows, ["Entropy", "dH", "-TdS", "dG", "SEM dG"],
                    top=table_top, height=0.04 + 0.022 * len(mmpbsa_rows))
        table_top = table_top - (0.06 + 0.022 * len(mmpbsa_rows))
    if warnings:
        notes_top = max(0.10, table_top - 0.10)
        figure.text(0.06, notes_top, "Notes", fontsize=8, weight="bold", va="top")
        figure.text(0.06, notes_top - 0.018, "\n".join(f"- {note}" for note in warnings),
                    fontsize=7.5, va="top", color="0.25")
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
                import textwrap

                title = f"{SECTION_TITLES.get(section, section)}: {figure_caption(image_path, dataset)}"
                figure.text(0.06, 0.96, textwrap.fill(title, 92), fontsize=9,
                            weight="bold", va="top")
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
