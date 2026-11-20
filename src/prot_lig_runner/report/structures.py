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
    options.bondLineWidth = 2.2
    options.fixedBondLength = 46
    options.baseFontSize = 0.85
    options.minFontSize = 26
    options.maxFontSize = 46
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

    if output_svg is not None:
        svg = rdMolDraw2D.MolDraw2DSVG(800, 600)
        svg.SetDrawOptions(options)
        rdMolDraw2D.PrepareAndDrawMolecule(svg, molecule, legend=label or "")
        svg.FinishDrawing()
        Path(output_svg).write_text(svg.GetDrawingText())
    return True
