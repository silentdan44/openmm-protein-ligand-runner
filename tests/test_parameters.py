import tempfile
import unittest
from pathlib import Path

import numpy as np

from prot_lig_runner.analysis import (
    _wrap_selection_into_box,
    interaction_entropy,
    summarize_mmpbsa,
)
from prot_lig_runner.analysis_metrics import estimate_membrane_planes

K_B = 0.00198720425864083


def make_membrane_topology(outlier=True):
    import mdtraj as md

    topology = md.Topology()
    chain = topology.add_chain()
    positions = []
    for index in range(6):
        residue = topology.add_residue("POPC", chain, resSeq=index + 1)
        topology.add_atom("P", md.element.phosphorus, residue)
        topology.add_atom("C1", md.element.carbon, residue)
        positions.extend([None, None])
    # Six lipids: three lower (-2 nm), three upper (+2 nm) plus one outlier.
    z = [-2.02, -1.98, -2.00, 1.99, 2.01, 2.00]
    if outlier:
        z[5] = 0.4
    xyz = np.zeros((topology.n_atoms, 3), dtype=float)
    for lipid, value in enumerate(z):
        xyz[2 * lipid, 2] = value
        xyz[2 * lipid, 0] = 1.0 + lipid * 0.1
    return topology, xyz


class MembranePlaneTests(unittest.TestCase):
    def test_estimates_planes_from_median_and_rejects_outliers(self):
        topology, xyz = make_membrane_topology(outlier=True)
        estimate = estimate_membrane_planes(topology, xyz, lipid_type="POPC")
        self.assertAlmostEqual(estimate["lower_plane_nm"], -2.00, places=2)
        self.assertAlmostEqual(estimate["upper_plane_nm"], 2.00, places=2)
        self.assertAlmostEqual(estimate["thickness_angstrom"], 40.0, places=1)
        self.assertEqual(estimate["outlier_count"], 1)
        self.assertEqual(estimate["outliers"][0]["leaflet"], "upper")

    def test_mean_would_be_misled_but_median_is_not(self):
        topology, xyz = make_membrane_topology(outlier=True)
        estimate = estimate_membrane_planes(topology, xyz, lipid_type="POPC")
        upper_mean = float(np.mean([1.99, 2.01, 0.4]))
        self.assertLess(abs(estimate["upper_plane_nm"] - 2.0), abs(upper_mean - 2.0))

    def test_requires_both_leaflets(self):
        topology, xyz = make_membrane_topology(outlier=False)
        xyz[:, 2] = 2.0
        with self.assertRaisesRegex(ValueError, "both membrane leaflets"):
            estimate_membrane_planes(topology, xyz, lipid_type="POPC")


class InteractionEntropyTests(unittest.TestCase):
    def test_constant_electrostatics_returns_that_energy(self):
        self.assertAlmostEqual(interaction_entropy([-10.0, -10.0, -10.0], 300.0), -10.0, places=6)

    def test_matches_explicit_log_mean_exp(self):
        values = np.array([-20.0, -5.0, -8.0, -12.0])
        kt = K_B * 300.0
        expected = kt * np.log(np.mean(np.exp(values / kt)))
        self.assertAlmostEqual(interaction_entropy(values, 300.0), expected, places=6)

    def test_entropy_correction_is_unfavourable(self):
        values = np.array([-25.0, -10.0, -3.0])
        self.assertGreater(interaction_entropy(values, 300.0), values.mean())


class SummarizeMmpbsaTests(unittest.TestCase):
    CSV = (
        "GENERALIZED BORN:\r\n"
        "Delta Energy Terms\r\n"
        "Frame #,VDWAALS,EEL,EGB,ESURF,TOTAL\r\n"
        "1,-5.0,-20.0,8.0,1.0,-16.0\r\n"
        "2,-6.0,-12.0,7.0,1.0,-10.0\r\n"
        "3,-4.0,-16.0,6.0,1.0,-13.0\r\n"
        "\r\n"
    )

    def test_adds_interaction_entropy_and_reports_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "FINAL_RESULTS_MMPBSA.csv"
            path.write_text(self.CSV)
            summary = summarize_mmpbsa(path, temperature_k=300.0, entropy="ie")
        self.assertEqual(summary["frame_count"], 3)
        self.assertAlmostEqual(summary["delta_h_mean_kcal_mol"], -13.0, places=6)
        expected_ie = interaction_entropy([-20.0, -12.0, -16.0], 300.0)
        self.assertAlmostEqual(summary["minus_tds_kcal_mol_value"], expected_ie, places=6)
        self.assertAlmostEqual(summary["delta_g_kcal_mol"], -13.0 + expected_ie, places=6)
        self.assertIn("sem_delta_g_kcal_mol", summary)

    def test_entropy_none_reports_enthalpy_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "FINAL_RESULTS_MMPBSA.csv"
            path.write_text(self.CSV)
            summary = summarize_mmpbsa(path, temperature_k=300.0, entropy="none")
        self.assertAlmostEqual(summary["delta_g_kcal_mol"], -13.0, places=6)
        self.assertEqual(summary["minus_tds_kcal_mol_value"], 0.0)


class WrapComplexTests(unittest.TestCase):
    def test_shifts_the_complex_into_the_box_without_changing_geometry(self):
        import mdtraj as md

        topology = md.Topology()
        chain = topology.add_chain()
        residue = topology.add_residue("LIG", chain)
        for index in range(3):
            topology.add_atom(f"C{index}", md.element.carbon, residue)
        xyz = np.array([[[20.0, 0.0, 0.0], [20.4, 0.0, 0.0], [20.0, 0.3, 0.0]]])
        trajectory = md.Trajectory(
            xyz, topology, unitcell_lengths=[[10.0, 10.0, 10.0]],
            unitcell_angles=[[90.0, 90.0, 90.0]],
        )
        before = np.linalg.norm(xyz[0, 1] - xyz[0, 0])
        _wrap_selection_into_box(trajectory)
        centroid = trajectory.xyz[0].mean(axis=0)
        self.assertTrue(np.all(centroid >= 0.0) and np.all(centroid <= 10.0))
        after = np.linalg.norm(trajectory.xyz[0, 1] - trajectory.xyz[0, 0])
        self.assertAlmostEqual(before, after, places=6)


if __name__ == "__main__":
    unittest.main()
