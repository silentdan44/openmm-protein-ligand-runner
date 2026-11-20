"""2D molecular depictions of the simulated ligands (RDKit, print quality)."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def draw_ligand(sdf_path, output_png, output_svg=None, *, label=None) -> bool:
    """Draw a clean 2D structure; returns False when RDKit cannot parse the file.

    Uses a fixed bond length, black bonds with coloured heteroatoms, generous
    padding and a large font, so the depiction stays legible when printed.
    The PNG is rendered at 4x and tagged at 600 dpi; the SVG keeps it vector.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        logger.warning("RDKit is not available; skipping ligand structures")
        return False
    sdf_path = Path(sdf_path)
    if not sdf_path.is_file():
        return False
    molecule = Chem.MolFromMolFile(str(sdf_path), removeHs=True, sanitize=True)
    if molecule is None:
        logger.warning("Could not parse %s for a 2D depiction", sdf_path)
        return False
    AllChem.Compute2DCoords(molecule)
    options = rdMolDraw2D.MolDrawOptions()
    options.bondLineWidth = 7.0
    options.fixedBondLength = 52
    options.baseFontSize = 0.85
    options.minFontSize = 34
    options.maxFontSize = 54
    options.padding = 0.08
    options.explicitMethyl = True
    options.additionalAtomLabelPadding = 0.08
    options.backgroundColour = (1.0, 1.0, 1.0, 1.0)
    options.clearBackground = True
    if label:
        options.legendFontSize = 30
        options.legendColour = (0.0, 0.0, 0.0, 1.0)

    drawer = rdMolDraw2D.MolDraw2DCairo(1600, 1200)
    drawer.SetDrawOptions(options)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule, legend=label or "")
    drawer.FinishDrawing()
    Path(output_png).write_bytes(drawer.GetDrawingText())

    # Cropped thumbnail for tables: trims the white canvas so the structure is
    # legible at ~40 px in an HTML table or a PDF cell.
    try:
        from PIL import Image, ImageChops

        image = Image.open(output_png).convert("RGB")
        background = Image.new("RGB", image.size, (255, 255, 255))
        box = ImageChops.difference(image, background).getbbox()
        if box:
            margin = 8
            box = (max(0, box[0] - margin), max(0, box[1] - margin),
                   min(image.width, box[2] + margin), min(image.height, box[3] + margin))
            crop = image.crop(box)
            side = max(crop.size)
            canvas = Image.new("RGB", (side, side), (255, 255, 255))
            canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
            canvas.thumbnail((400, 400), Image.LANCZOS)
            canvas.save(Path(output_png).with_name(Path(output_png).stem + "_thumb.png"))
    except Exception as error:  # noqa: BLE001 - thumbnail is best effort
        logger.warning("Could not build a thumbnail for %s: %s", output_png, error)

    if output_svg is not None:
        svg = rdMolDraw2D.MolDraw2DSVG(800, 600)
        svg.SetDrawOptions(options)
        rdMolDraw2D.PrepareAndDrawMolecule(svg, molecule, legend=label or "")
        svg.FinishDrawing()
        Path(output_svg).write_text(svg.GetDrawingText())
    return True
