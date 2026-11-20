import tempfile
import unittest
from pathlib import Path

from prot_lig_runner.report.structures import draw_ligand


SDF = """benzene
  Mrv  0000

  6  6  0  0  0  0            999 V2000
    1.2990    0.7500    0.0000 C   0  0
    1.2990   -0.7500    0.0000 C   0  0
    0.0000   -1.5000    0.0000 C   0  0
   -1.2990   -0.7500    0.0000 C   0  0
   -1.2990    0.7500    0.0000 C   0  0
    0.0000    1.5000    0.0000 C   0  0
  1  2  2  0
  2  3  1  0
  3  4  2  0
  4  5  1  0
  5  6  2  0
  6  1  1  0
M  END
"""


class LigandStructureTests(unittest.TestCase):
    def test_draws_png_and_svg(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdf = root / "ligand.sdf"
            sdf.write_text(SDF)
            png, svg = root / "ligand.png", root / "ligand.svg"
            drawn = draw_ligand(sdf, png, svg, label="LIG1")
            self.assertTrue(drawn)
            self.assertTrue(png.read_bytes().startswith(b"\x89PNG"))
            self.assertIn("<svg", svg.read_text()[:500])

    def test_missing_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(draw_ligand(Path(directory) / "absent.sdf", Path(directory) / "x.png"))


if __name__ == "__main__":
    unittest.main()
