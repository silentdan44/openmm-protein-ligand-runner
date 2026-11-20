import json
import tempfile
import unittest
from pathlib import Path

from prot_lig_runner.report import RunDataset, build_report


def make_run(root: Path, *, with_metrics=True, with_trajectory=True) -> Path:
    (root / "analysis").mkdir(parents=True)
    (root / "run_state.json").write_text(json.dumps({
        "status": "running",
        "configuration": {"prod_steps": 5000000, "timestep_ps": 0.002},
    }))
    (root / "analysis" / "time_series.csv").write_text(
        "time_ns,protein_backbone_rmsd_nm\n1.0,0.1\n2.0,0.2\n"
    )
    if with_metrics:
        (root / "analysis" / "metrics.json").write_text(json.dumps({
            "preliminary": True,
            "trajectory_metrics": {
                "ligands": {"LIG1": {"residue": "A:1:LIG", "ligand_rmsd_nm_mean": 0.4}},
                "membrane_thickness_nm_mean": 3.7,
            },
        }))
    if with_trajectory:
        (root / "analysis" / "ligand_comparison.png").write_bytes(b"png")
    return root


class ReportDatasetTests(unittest.TestCase):
    def test_progress_and_availability(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = RunDataset(make_run(Path(directory)))
            status = dataset.status()
        self.assertAlmostEqual(status["simulated_ns"], 2.0)
        self.assertAlmostEqual(status["planned_ns"], 10.0)
        self.assertAlmostEqual(status["fraction"], 0.2)
        self.assertTrue(status["preliminary"])
        self.assertTrue(status["availability"]["ligands"])
        self.assertTrue(status["availability"]["membrane"])
        self.assertFalse(status["availability"]["mmpbsa"])

    def test_missing_metrics_still_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_run(Path(directory), with_metrics=False)
            dataset = RunDataset(root)
            path = build_report(dataset)
            text = path.read_text()
            manifest = json.loads((root / "analysis" / "report_manifest.json").read_text())
        self.assertIn("Waiting for data", text)
        self.assertIn("Simulation QC", text)
        self.assertFalse(manifest["status"]["availability"]["ligands"])

    def test_sections_can_be_limited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_run(Path(directory))
            build_report(RunDataset(root), only=("ligands",))
            text = (root / "analysis" / "report.html").read_text()
        self.assertIn('id="ligands"', text)
        self.assertNotIn('id="mmpbsa"', text)


if __name__ == "__main__":
    unittest.main()
