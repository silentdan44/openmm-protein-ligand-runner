import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from prot_lig_runner.trajectory_analysis import (
    phase_for_step,
    read_state_data,
    run_mmpbsa,
    write_index_file,
)


class TrajectoryAnalysisTests(unittest.TestCase):
    def test_phase_boundaries(self):
        self.assertEqual(phase_for_step(250, 250, 500), "NVT")
        self.assertEqual(phase_for_step(251, 250, 500), "NPT")
        self.assertEqual(phase_for_step(750, 250, 500), "NPT")
        self.assertEqual(phase_for_step(751, 250, 500), "Production")

    def test_reads_reporter_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state_data.csv"
            path.write_text("Step,Temperature (K),Density (g/mL)\n10,300.5,0.99\n")
            self.assertEqual(read_state_data(path), [
                {"Step": 10.0, "Temperature (K)": 300.5, "Density (g/mL)": 0.99}
            ])

    def test_writes_gromacs_index_groups_in_one_based_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.ndx"
            write_index_file(path, [0, 1, 17], [2, 3])
            self.assertEqual(path.read_text(), "[ Protein ]\n1 2 18\n\n[ LIG ]\n3 4\n")

    def test_mmpbsa_writes_gb_input_and_calls_single_trajectory_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis = root / "analysis"
            analysis.mkdir()
            for path in (root / "mmpbsa_complex.top", root / "mmpbsa_complex.pdb",
                         root / "system_final.pdb", analysis / "trajectory_fitted.xtc"):
                path.write_text("input")
            full_topology = MagicMock()
            full_topology.select.return_value = [0, 1]
            complex_topology = MagicMock()
            complex_topology.n_atoms = 2
            complex_topology.select.side_effect = [[0], [1]]
            complex_chunk = MagicMock()
            complex_chunk.xyz = "coordinates"
            complex_chunk.time = [0.0]
            complex_chunk.unitcell_vectors = "box"
            complex_chunk.n_frames = 1
            chunk = MagicMock()
            chunk.atom_slice.return_value = complex_chunk
            writer = MagicMock()
            writer_context = MagicMock()
            writer_context.__enter__.return_value = writer
            md = SimpleNamespace(
                load_topology=MagicMock(side_effect=[full_topology, complex_topology]),
                iterload=MagicMock(return_value=iter([chunk])),
                formats=SimpleNamespace(XTCTrajectoryFile=MagicMock(return_value=writer_context)),
            )
            with patch("prot_lig_runner.trajectory_analysis.shutil.which", return_value="/usr/bin/gmx_MMPBSA"):
                with patch("prot_lig_runner.trajectory_analysis.subprocess.run") as run:
                    run.return_value.returncode = 0
                    with patch.dict("sys.modules", {"mdtraj": md}):
                        result = run_mmpbsa(root, analysis, interval=2)
            self.assertEqual(result, analysis / "mmpbsa")
            input_text = (result / "mmpbsa.in").read_text()
            self.assertIn("interval=2", input_text)
            self.assertIn("igb=5", input_text)
            command = run.call_args.args[0]
            self.assertIn("-cp", command)
            self.assertEqual(command[command.index("-cp") + 1], str(root / "mmpbsa_complex.top"))
            self.assertIn("-ct", command)
            self.assertNotEqual(command[command.index("-cs") + 1],
                                command[command.index("-cr") + 1])
            self.assertEqual(Path(command[command.index("-cr") + 1]).read_text(), "input")
            self.assertEqual(command[command.index("-ct") + 1],
                             str(analysis / "mmpbsa" / "complex.xtc"))
            self.assertEqual(command[command.index("-cg") + 1:command.index("-cg") + 3],
                             ["Protein", "LIG"])

    def test_pb_mem_requires_explicit_membrane_thickness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis = root / "analysis"
            analysis.mkdir()
            for path in (root / "mmpbsa_complex.top", root / "mmpbsa_complex.pdb",
                         root / "system_final.pdb", analysis / "trajectory_fitted.xtc"):
                path.write_text("input")
            with patch("prot_lig_runner.trajectory_analysis.shutil.which", return_value="/usr/bin/gmx_MMPBSA"):
                with self.assertRaisesRegex(ValueError, "membrane thickness"):
                    run_mmpbsa(root, analysis, model="pb_mem")


if __name__ == "__main__":
    unittest.main()
