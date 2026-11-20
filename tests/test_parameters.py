import tempfile
import unittest
from pathlib import Path

import numpy as np

from prot_lig_runner.analysis import (
    _wrap_selection_into_box,
    c2_entropy,
    interaction_entropy,
    ionic_strength_from_config,
    read_gmx_entropy,
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
    def test_constant_electrostatics_has_no_entropy(self):
        self.assertAlmostEqual(interaction_entropy([-10.0, -10.0, -10.0], 300.0), 0.0, places=9)

    def test_matches_explicit_log_mean_exp_minus_mean(self):
        values = np.array([-20.0, -5.0, -8.0, -12.0])
        kt = K_B * 300.0
        expected = kt * np.log(np.mean(np.exp(values / kt))) - values.mean()
        self.assertAlmostEqual(interaction_entropy(values, 300.0), expected, places=6)

    def test_entropy_correction_is_unfavourable(self):
        values = np.array([-25.0, -10.0, -3.0])
        self.assertGreater(interaction_entropy(values, 300.0), 0.0)


    def test_c2_matches_variance_formula(self):
        values = np.array([-10.0, -12.0, -8.0, -14.0])
        expected = np.var(values, ddof=1) / (2.0 * K_B * 300.0)
        self.assertAlmostEqual(c2_entropy(values, 300.0), expected, places=9)

    def test_c2_is_zero_for_constant_electrostatics(self):
        self.assertAlmostEqual(c2_entropy([-9.0, -9.0, -9.0], 300.0), 0.0, places=9)


class IonicStrengthTests(unittest.TestCase):
    def test_follows_configuration(self):
        self.assertAlmostEqual(ionic_strength_from_config({"ionic_strength_molar": 0.2}), 0.2)

    def test_explicit_zero_stays_zero(self):
        self.assertEqual(ionic_strength_from_config({"ionic_strength_molar": 0.0}), 0.0)

    def test_absent_key_uses_historical_default(self):
        self.assertAlmostEqual(ionic_strength_from_config({}), 0.15)


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
        self.assertEqual(summary["entropy_variants"], {})

    def test_reads_gmx_mmpbsa_entropy_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "FINAL_RESULTS_MMPBSA.csv"
            path.write_text(self.CSV)
            (Path(directory) / "_GMXMMPBSA_normal_gb_IE.dat").write_text(
                "| Interaction Entropy results for gb calculations\n"
                "IE-frames: last 3\n"
                "Full-ensemble Interaction Entropy (-TΔS):    2.5000\n"
                "Block diagnostic: size 4 blocks 5 SD 0.1000 SEM 0.0500\n"
            )
            (Path(directory) / "_GMXMMPBSA_normal_gb_c2_entropy.dat").write_text(
                "C2 Entropy (-TΔS): 3.1000\n"
                "C2 Block SD: 1.2000\n"
                "C2 Block SEM: 0.3913\n"
            )
            summary = summarize_mmpbsa(
                path, temperature_k=300.0, entropy="both",
                mmpbsa_dir=directory, model="gb",
            )
        self.assertAlmostEqual(summary["entropy_variants"]["ie"]["minus_tds_kcal_mol"], 2.5, places=6)
        self.assertAlmostEqual(summary["entropy_variants"]["ie"]["sem_minus_tds_kcal_mol"], 0.05, places=6)
        self.assertAlmostEqual(summary["entropy_variants"]["c2"]["minus_tds_kcal_mol"], 3.1, places=6)
        self.assertEqual(summary["entropy_variants"]["ie"]["source"], "_GMXMMPBSA_normal_gb_IE.dat")
        self.assertAlmostEqual(summary["delta_g_kcal_mol"], -13.0 + 2.5, places=6)

    def test_reads_nmode_entropy_from_dat(self):
        with tempfile.TemporaryDirectory() as directory:
            dat = Path(directory) / "FINAL_RESULTS_MMPBSA.dat"
            dat.write_text(
                "Normal [ -TΔS ]\n"
                "ENTROPY RESULTS (NMODE APPROXIMATION):\n"
                "Delta (Complex - Receptor - Ligand):\n"
                "Entropy Term           Average     SD(Prop.)         SD   SEM(Prop.)        SEM\n"
                "ΔTRANSLATIONAL           12.19          0.00       0.00         0.00       0.00\n"
                "ΔVIBRATIONAL            -22.56          4.87       4.90         2.81       2.83\n"
                "ΔTOTAL                   -2.60          4.87       4.89         2.81       2.83\n"
            )
            estimates = read_gmx_entropy(directory, "gb")
        self.assertAlmostEqual(estimates["nmode"]["minus_tds_kcal_mol"], -2.6, places=6)
        self.assertAlmostEqual(estimates["nmode"]["sem_minus_tds_kcal_mol"], 2.83, places=6)
        self.assertEqual(estimates["nmode"]["source"], "FINAL_RESULTS_MMPBSA.dat")

    def test_both_reports_ie_and_c2_but_uses_ie_for_delta_g(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "FINAL_RESULTS_MMPBSA.csv"
            path.write_text(self.CSV)
            summary = summarize_mmpbsa(path, temperature_k=300.0, entropy="both")
        variants = summary["entropy_variants"]
        self.assertEqual(set(variants), {"ie", "c2"})
        self.assertAlmostEqual(variants["ie"]["minus_tds_kcal_mol"], summary["minus_tds_kcal_mol_value"], places=9)
        self.assertAlmostEqual(
            variants["c2"]["minus_tds_kcal_mol"],
            c2_entropy([-20.0, -12.0, -16.0], 300.0), places=9,
        )
        self.assertAlmostEqual(summary["delta_g_kcal_mol"], -13.0 + variants["ie"]["minus_tds_kcal_mol"], places=9)


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


class AtomtypeDeduplicationTests(unittest.TestCase):
    def test_equivalent_types_are_merged(self):
        import tempfile
        from pathlib import Path
        from prot_lig_runner.topology_export import deduplicate_atomtypes

        top = """
[ atomtypes ]
;name  at.num  mass  charge  ptype  sigma  epsilon
T0       6  12.01  0.0  A  0.3400  0.3600
T1       6  12.01  0.0  A  0.3400  0.3600
T2       8  16.00  0.0  A  0.2950  0.8800
T3       6  12.01  0.0  A  0.3400  0.3600

[ atoms ]
1  T1  1  LIG  C1  1  0.1  12.01
2  T2  1  LIG  O1  2 -0.2  16.00
3  T3  1  LIG  C2  3  0.1  12.01

[ nonbond_params ]
T0  T2  1  0.30  0.50
T1  T2  1  0.30  0.50
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "system.top"
            path.write_text(top)
            deduplicate_atomtypes(path)
            text = path.read_text()
        self.assertNotIn("T1", text)
        self.assertNotIn("T3", text)
        self.assertIn("T0", text)
        self.assertIn("T2", text)
        self.assertEqual(text.count("T0 T2 1 0.30 0.50"), 1)


if __name__ == "__main__":
    unittest.main()
