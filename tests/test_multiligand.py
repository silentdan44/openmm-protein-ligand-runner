import tempfile
import unittest
from pathlib import Path

import mdtraj as md
import numpy as np
from openmm import app

from prot_lig_runner.analysis_metrics import (
    ligand_groups,
    resolve_ligand,
    trajectory_metrics,
)
from prot_lig_runner.topology_export import mmpbsa_complex_selection

PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def make_mdtraj_topology():
    topology = md.Topology()
    chain = topology.add_chain()
    for index in range(2):
        residue = topology.add_residue("ALA", chain, resSeq=index + 1)
        for name, element in (
            ("N", md.element.nitrogen),
            ("CA", md.element.carbon),
            ("C", md.element.carbon),
            ("O", md.element.oxygen),
        ):
            topology.add_atom(name, element, residue)
    for index in range(2):
        residue = topology.add_residue("LIG", chain, resSeq=11 + index)
        for atom_index in range(3):
            topology.add_atom(f"C{atom_index}", md.element.carbon, residue)
    return topology


def make_openmm_topology(with_water=True):
    topology = app.Topology()
    chain = topology.addChain("A")
    protein = topology.addResidue("ALA", chain, "1")
    for name in ("N", "CA", "C", "O"):
        topology.addAtom(name, app.element.carbon, protein)
    for index in range(2):
        ligand = topology.addResidue("LIG", chain, str(10 + index))
        for atom_index in range(3):
            topology.addAtom(f"C{atom_index}", app.element.carbon, ligand)
    if with_water:
        water = topology.addResidue("HOH", chain, "50")
        topology.addAtom("O", app.element.oxygen, water)
    return topology


class LigandSelectionTests(unittest.TestCase):
    def test_labels_follow_input_order(self):
        groups = ligand_groups(make_mdtraj_topology())
        self.assertEqual([group["label"] for group in groups], ["LIG1", "LIG2"])
        self.assertEqual([group["res_seq"] for group in groups], [11, 12])
        self.assertTrue(all(len(group["heavy_atoms"]) == 3 for group in groups))

    def test_selects_by_index_label_and_residue_label(self):
        topology = make_mdtraj_topology()
        by_index = resolve_ligand(topology, "2")
        self.assertEqual(by_index["label"], "LIG2")
        self.assertEqual(resolve_ligand(topology, "LIG2")["res_seq"], 12)
        self.assertEqual(
            resolve_ligand(topology, by_index["residue_label"])["label"], "LIG2"
        )

    def test_requires_explicit_choice_with_multiple_ligands(self):
        with self.assertRaisesRegex(ValueError, "several ligands"):
            resolve_ligand(make_mdtraj_topology(), None)

    def test_single_ligand_is_unambiguous(self):
        topology = make_mdtraj_topology()
        single = md.Topology()
        chain = single.add_chain()
        protein = single.add_residue("ALA", chain, resSeq=1)
        single.add_atom("CA", md.element.carbon, protein)
        ligand = single.add_residue("LIG", chain, resSeq=11)
        single.add_atom("C0", md.element.carbon, ligand)
        self.assertEqual(resolve_ligand(single, None)["label"], "LIG1")
        self.assertEqual(len(ligand_groups(topology)), 2)

    def test_rejects_out_of_range_index(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            resolve_ligand(make_mdtraj_topology(), "3")

    def test_works_with_an_openmm_topology(self):
        selection = resolve_ligand(make_openmm_topology(), "2")
        self.assertEqual(selection["label"], "LIG2")
        self.assertEqual(str(selection["res_seq"]), "11")


class MmpbsaSelectionTests(unittest.TestCase):
    def test_keeps_protein_and_only_the_selected_ligand(self):
        topology = make_openmm_topology()
        selection, discarded = mmpbsa_complex_selection(topology, PROTEIN_RESIDUES, "2")
        self.assertEqual(selection["label"], "LIG2")
        discarded_residues = {(atom.residue.name, atom.residue.id) for atom in discarded}
        self.assertIn(("LIG", "10"), discarded_residues)
        self.assertIn(("HOH", "50"), discarded_residues)
        self.assertNotIn(("LIG", "11"), discarded_residues)
        kept = [
            atom for atom in topology.atoms()
            if atom not in discarded and atom.residue.name == "LIG"
        ]
        self.assertEqual({atom.residue.id for atom in kept}, {"11"})

    def test_multiple_ligands_need_a_selection(self):
        with self.assertRaisesRegex(ValueError, "several ligands"):
            mmpbsa_complex_selection(make_openmm_topology(), PROTEIN_RESIDUES, None)


class PerLigandMetricsTests(unittest.TestCase):
    def make_run(self, directory):
        topology = make_mdtraj_topology()
        frames, atoms = 5, topology.n_atoms
        xyz = np.zeros((frames, atoms, 3), dtype=np.float32)
        for frame in range(frames):
            xyz[frame, :, 0] = np.linspace(0.5, 1.6, atoms)
            xyz[frame, :, 1] = 0.8
            xyz[frame, :, 2] = 0.9
        # Ligand 1 (residue 11) stays put; ligand 2 (residue 12) drifts away.
        ligand1 = [atom.index for atom in topology.atoms if atom.residue.resSeq == 11]
        ligand2 = [atom.index for atom in topology.atoms if atom.residue.resSeq == 12]
        for frame in range(frames):
            xyz[frame, ligand2, 0] += 0.15 * frame
        trajectory = md.Trajectory(
            xyz, topology, time=np.arange(frames) * 1000.0,
            unitcell_lengths=np.tile([4.0, 4.0, 4.0], (frames, 1)),
            unitcell_angles=np.tile([90.0, 90.0, 90.0], (frames, 1)),
        )
        trajectory_path = Path(directory) / "trajectory.xtc"
        topology_path = Path(directory) / "system_final.pdb"
        trajectory.save_xtc(str(trajectory_path))
        trajectory[0].save_pdb(str(topology_path))
        analysis_dir = Path(directory) / "analysis"
        analysis_dir.mkdir()
        return trajectory_path, topology_path, analysis_dir, ligand1, ligand2

    def test_reports_one_series_per_ligand(self):
        config = {
            "nvt_steps": 0, "npt_steps": 0, "timestep_fs": 1.0,
            "membrane": False, "distance_pairs": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            trajectory_path, topology_path, analysis_dir, ligand1, ligand2 = self.make_run(directory)
            result = trajectory_metrics(
                trajectory_path, topology_path, analysis_dir, config, analysis_stride=1
            )

            self.assertEqual([ligand["label"] for ligand in result["ligands"]], ["LIG1", "LIG2"])
            statistics = result["statistics"]["ligands"]
            self.assertLess(
                statistics["LIG1"]["ligand_rmsd_nm_mean"],
                statistics["LIG2"]["ligand_rmsd_nm_mean"],
            )
            self.assertGreater(result["ligands"][1]["rmsd_nm"][-1], result["ligands"][0]["rmsd_nm"][-1])

            header = (analysis_dir / "time_series.csv").read_text().splitlines()[0]
            self.assertIn("ligand_rmsd_nm:LIG1", header)
            self.assertIn("ligand_rmsd_nm:LIG2", header)


class PerLigandReportTests(unittest.TestCase):
    def test_report_has_a_section_and_summary_for_each_ligand(self):
        from prot_lig_runner.analysis import analyze_run

        config = {
            "nvt_steps": 0, "npt_steps": 0, "timestep_fs": 1.0,
            "membrane": False, "distance_pairs": [],
            "temperature_k": 300.0, "pressure_bar": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            PerLigandMetricsTests().make_run(root)
            # make_run writes into <root>/analysis; analyze_run expects a fresh one.
            (root / "analysis").rename(root / "source_analysis")
            (root / "state_data.csv").write_text(
                'Step,Temperature (K)\n0,300.0\n1000,301.0\n'
            )
            analysis_dir = analyze_run(root, config, analysis_stride=1)
            report = (analysis_dir / "report.html").read_text()
            self.assertIn("Per-ligand summary", report)
            self.assertIn("LIG1", report)
            self.assertIn("LIG2", report)
            for name in (
                "ligand_pocket_LIG1.png",
                "ligand_pocket_LIG2.png",
                "ligand_comparison.png",
            ):
                self.assertTrue((analysis_dir / name).is_file(), name)


class PeriodicUnwrapTests(unittest.TestCase):
    def test_ligand_crossing_the_box_is_not_a_jump(self):
        topology = make_mdtraj_topology()
        box = 4.0
        frames, atoms = 3, topology.n_atoms
        xyz = np.zeros((frames, atoms, 3), dtype=np.float32)
        xyz[:, :, 0] = 1.0
        xyz[:, :, 1] = 1.0
        xyz[:, :, 2] = 1.0
        ligand2 = [atom.index for atom in topology.atoms if atom.residue.resSeq == 12]
        # Ligand 2 sits near the boundary and crosses it between frames: +0.15 nm
        # of real motion appears as -3.85 nm in wrapped coordinates.
        xyz[0, ligand2, 0] = 3.95
        xyz[1, ligand2, 0] = 0.10
        xyz[2, ligand2, 0] = 0.25
        trajectory = md.Trajectory(
            xyz, topology, time=np.arange(frames) * 1000.0,
            unitcell_lengths=np.tile([box, box, box], (frames, 1)),
            unitcell_angles=np.tile([90.0, 90.0, 90.0], (frames, 1)),
        )
        config = {
            "nvt_steps": 0, "npt_steps": 0, "timestep_fs": 1.0,
            "membrane": False, "distance_pairs": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            trajectory_path = Path(directory) / "trajectory.xtc"
            topology_path = Path(directory) / "system_final.pdb"
            trajectory.save_xtc(str(trajectory_path))
            trajectory[0].save_pdb(str(topology_path))
            analysis_dir = Path(directory) / "analysis"
            analysis_dir.mkdir()
            result = trajectory_metrics(
                trajectory_path, topology_path, analysis_dir, config, analysis_stride=1
            )
        rmsd = result["ligands"][1]["rmsd_nm"]
        self.assertLess(float(rmsd[-1]), 0.5, f"wrapped ligand produced a jump: {rmsd}")


if __name__ == "__main__":
    unittest.main()
