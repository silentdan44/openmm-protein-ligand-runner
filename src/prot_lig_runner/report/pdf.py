"""PDF export of a run report, for reading on a phone or printing."""

from __future__ import annotations

import html
from pathlib import Path

from .dataset import RunDataset, SECTION_TITLES, figure_caption, figure_title


def _summary_page_text(dataset: RunDataset, warnings: list[str]) -> list[str]:
    status = dataset.status()
    planned = status["planned_ns"]
    if planned:
        progress = (
            f'{status["simulated_ns"]:.2f} of {planned:.2f} ns '
            f'({100 * status["fraction"]:.1f}%), {status["frames"]} frames'
        )
    else:
        progress = f'{status["simulated_ns"]:.2f} ns, {status["frames"]} frames'
    lines = [
        "Run report",
        "",
        f"Run: {dataset.root}",
        f"Phase: {status['phase']}",
        f"Progress: {progress}",
    ]
    if status["temperature_k"]:
        lines.append(f"Temperature: {float(status['temperature_k']):.1f} K")

    metrics = dataset.metrics.get("trajectory_metrics") or {}
    if metrics:
        lines += ["", "Key metrics"]
        def metric(key, unit, digits=2):
            value = metrics.get(key)
            if value is None:
                return None
            label = key.replace("_mean", "").replace("_nm2", "").replace("_nm", "").replace("_", " ")
            return f"{label}: {value:.{digits}f} {unit}".strip()
        for key, unit, digits in (
            ("protein_backbone_rmsd_nm_mean", "nm", 2),
            ("protein_rg_nm_mean", "nm", 2),
            ("ligand_protein_contacts_mean", "contacts", 1),
            ("protein_ligand_hbonds_mean", "H-bonds", 1),
            ("membrane_thickness_nm_mean", "nm", 2),
            ("membrane_area_per_lipid_nm2_mean", "nm2", 3),
        ):
            line = metric(key, unit, digits)
            if line:
                lines.append(f"  {line}")

    ligands = (dataset.metrics.get("trajectory_metrics") or {}).get("ligands") or {}
    if ligands:
        lines += ["", "Per-ligand summary", "Ligand   Residue        RMSD(start)  RMSD(input)  Contacts  H-bonds"]
        for label, stats in ligands.items():
            pose = stats.get("ligand_pose_rmsd_nm_mean")
            lines.append(
                f"{label:<8} {str(stats.get('residue', '')):<14} "
                f"{stats.get('ligand_rmsd_nm_mean', 0):>11.2f}  "
                f"{'—' if pose is None else f'{pose:>11.2f}'}  "
                f"{stats.get('ligand_contacts_mean', 0):>8.1f}  "
                f"{stats.get('protein_ligand_hbonds_mean', 0):>7.1f}"
            )

    summary = dataset.mmpbsa
    if summary:
        ligand = summary.get("ligand")
        title = "MMPBSA free-energy estimate (kcal/mol)"
        if ligand:
            title += f" — {ligand}"
        lines += ["", title]
        variants = summary.get("entropy_variants") or {}
        if variants:
            lines.append("Entropy   dH       -TdS     dG       SEM dG")
            for variant in variants.values():
                sem = variant.get("sem_delta_g_kcal_mol")
                lines.append(
                    f"{str(variant.get('name', '')).upper():<8} "
                    f"{summary.get('delta_h_mean_kcal_mol', 0):>7.2f}  "
                    f"{variant.get('minus_tds_kcal_mol', 0):>7.2f}  "
                    f"{variant.get('delta_g_kcal_mol', 0):>7.2f}  "
                    f"{'—' if sem is None else f'{sem:>7.2f}'}"
                )
        else:
            lines.append(f"dH = {summary.get('delta_h_mean_kcal_mol', 0):.2f} (enthalpy only)")
        lines.append(f"{summary.get('frame_count', 0)} frames at "
                     f"{summary.get('temperature_k', 300):.0f} K")

    if warnings:
        lines += ["", "Notes"] + [f"- {note}" for note in warnings]
    return lines


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
        figure.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.06)
        lines = _summary_page_text(dataset, warnings or [])
        figure.text(
            0.08, 0.93, "\n".join(lines),
            family="monospace", fontsize=9, va="top", ha="left", wrap=True,
        )
        figure.text(0.92, 0.03, f"1 / {total_pages}", fontsize=8, ha="right", color="0.4")
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
