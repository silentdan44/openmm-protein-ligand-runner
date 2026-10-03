import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from stability_diagnostics import (
    StabilityReporter,
    diagnose_failure,
    find_severe_overlaps,
    write_diagnostics,
)


class FakeChain:
    id = "A"


class FakeResidue:
    def __init__(self, name, residue_id):
        self.name = name
        self.id = residue_id
        self.chain = FakeChain()


class FakeAtom:
    def __init__(self, index, name, symbol, residue):
        self.index = index
        self.name = name
        self.element = types.SimpleNamespace(symbol=symbol)
        self.residue = residue


class FakeTopology:
    def __init__(self, atoms, bonds=()):
        self._atoms = atoms
        self._bonds = bonds

    def atoms(self):
        return iter(self._atoms)

    def bonds(self):
        return iter((self._atoms[a], self._atoms[b]) for a, b in self._bonds)


class FakePositions:
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def value_in_unit(self, _unit):
        return self.coordinates


class StabilityDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.original_openmm = sys.modules.get("openmm")
        sys.modules["openmm"] = types.SimpleNamespace(
            unit=types.SimpleNamespace(nanometer=object(), kilojoule_per_mole=object())
        )

    def tearDown(self):
        if self.original_openmm is None:
            sys.modules.pop("openmm", None)
        else:
            sys.modules["openmm"] = self.original_openmm

    def test_finds_and_labels_severe_nonbonded_overlap(self):
        atoms = [
            FakeAtom(0, "C1", "C", FakeResidue("LIG", "1")),
            FakeAtom(1, "O1", "O", FakeResidue("ALA", "2")),
            FakeAtom(2, "H1", "H", FakeResidue("ALA", "2")),
        ]
        topology = FakeTopology(atoms, bonds=((1, 2),))
        positions = FakePositions(((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.11, 0.0, 0.0)))

        pairs = find_severe_overlaps(topology, positions)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["atom_1"], "A:LIG1:C1[0]")
        self.assertEqual(pairs[0]["atom_2"], "A:ALA2:O1[1]")
        self.assertAlmostEqual(pairs[0]["distance_nm"], 0.1)

    def test_explains_constraint_failure_and_includes_metrics(self):
        result = diagnose_failure(
            RuntimeError("Could not enforce constraints"), step=250, phase="nvt",
            temperature_k=425.0, energy_kj_mol=-1200.0,
        )

        self.assertIn("constrained bonds", result["cause"])
        self.assertIn("smaller timestep", result["suggested_action"])
        self.assertEqual(result["step"], 250)

    def test_overlap_is_reported_as_a_possible_contributor(self):
        overlap = [{"atom_1": "A:ALA1:C[0]", "atom_2": "A:LIG1:O[1]", "distance_nm": 0.1}]
        result = diagnose_failure(RuntimeError("integration failed"), overlaps=overlap)

        self.assertIn("may contribute", result["cause"])
        self.assertEqual(result["possible_severe_overlaps"], overlap)

    def test_writes_json_diagnostics_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failure_diagnostics.json"
            write_diagnostics(path, {"phase": "npt", "step": 12})
            self.assertEqual(json.loads(path.read_text()), {"phase": "npt", "step": 12})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_monitor_stops_on_non_finite_energy(self):
        class Energy:
            def value_in_unit(self, _unit):
                return float("nan")

        class State:
            def getPotentialEnergy(self):
                return Energy()

            def getKineticEnergy(self):
                return Energy()

        class Simulation:
            currentStep = 50

        reporter = StabilityReporter(interval=50, target_temperature_k=300, particle_count=10)
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            reporter.report(Simulation(), State())
        self.assertEqual(reporter.last_step, 50)

    def test_monitor_requests_energies_with_openmm_82_reporter_protocol(self):
        reporter = StabilityReporter(interval=100, target_temperature_k=300, particle_count=10)
        simulation = types.SimpleNamespace(currentStep=125)

        self.assertEqual(reporter.describeNextReport(simulation), (75, False, False, False, True))

    def test_monitor_stops_on_extreme_temperature(self):
        class Energy:
            def __init__(self, value):
                self.value = value

            def value_in_unit(self, _unit):
                return self.value

        class State:
            def getPotentialEnergy(self):
                return Energy(0.0)

            def getKineticEnergy(self):
                return Energy(3 * 10 * 0.008314462618 * 1600)

        reporter = StabilityReporter(interval=50, target_temperature_k=300, particle_count=10)
        with self.assertRaisesRegex(RuntimeError, "temperature"):
            reporter.report(types.SimpleNamespace(currentStep=50), State())

    def test_monitor_stops_on_large_energy_jump(self):
        class Energy:
            def __init__(self, value):
                self.value = value

            def value_in_unit(self, _unit):
                return self.value

        class State:
            def __init__(self, potential):
                self.potential = potential

            def getPotentialEnergy(self):
                return Energy(self.potential)

            def getKineticEnergy(self):
                return Energy(0.0)

        reporter = StabilityReporter(interval=50, target_temperature_k=300, particle_count=100)
        simulation = types.SimpleNamespace(currentStep=50)
        reporter.report(simulation, State(0.0))
        simulation.currentStep = 100
        with self.assertRaisesRegex(RuntimeError, "Potential energy changed"):
            reporter.report(simulation, State(2600.0))


if __name__ == "__main__":
    unittest.main()
