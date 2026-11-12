import tempfile
import unittest
from pathlib import Path

import openmm as mm
from openmm import app
from openmm import unit as u

from prot_lig_runner.simulate import (
    load_serialized_resume,
    serialize_xml_atomic,
    serialized_system_path,
    truncate_state_data_to_step,
)


def make_toy_system():
    system = mm.System()
    for _ in range(4):
        system.addParticle(2.0 * u.amu)
    restraint = mm.CustomExternalForce("0.5*k*periodicdistance(x,y,z,x0,y0,z0)^2")
    restraint.addGlobalParameter("k", 100.0)
    for parameter in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(parameter)
    for index in range(4):
        restraint.addParticle(index, (0.1 * index, 0.0, 0.0))
    system.addForce(restraint)
    bond = mm.CustomBondForce("0.5*(r-0.2)^2")
    bond.addBond(0, 1, [])
    system.addForce(bond)
    return system


def make_toy_topology(num_atoms=4):
    topology = app.Topology()
    chain = topology.addChain("A")
    residue = topology.addResidue("LIG", chain)
    for index in range(num_atoms):
        topology.addAtom(f"C{index}", app.element.carbon, residue)
    return topology


def make_toy_integrator():
    return mm.LangevinMiddleIntegrator(
        300 * u.kelvin, 1.0 / u.picosecond, 2.0 * u.femtosecond
    )


def write_resume_artifacts(directory, simulation, system, integrator, topology, positions):
    directory = Path(directory)
    checkpoint = directory / "checkpoint-prod.chk"
    simulation.saveCheckpoint(str(checkpoint))
    serialize_xml_atomic(system, serialized_system_path(checkpoint))
    serialize_xml_atomic(integrator, directory / "integrator.xml")
    with open(directory / "system_final.pdb", "w") as handle:
        app.PDBFile.writeFile(topology, positions, handle)
    return checkpoint


class SerializedSystemPathTests(unittest.TestCase):
    def test_maps_checkpoint_stages_to_ensemble_files(self):
        cases = {
            "checkpoint-nvt1.chk": "system-nvt.xml",
            "checkpoint-nvt2.chk": "system-nvt.xml",
            "checkpoint-npt1.chk": "system-npt.xml",
            "checkpoint-npt4.chk": "system-npt.xml",
            "checkpoint-prod.chk": "system-prod.xml",
            "checkpoint-complete.chk": "system-prod.xml",
        }
        for checkpoint_name, system_name in cases.items():
            with self.subTest(checkpoint=checkpoint_name):
                path = serialized_system_path(Path(checkpoint_name))
                self.assertEqual(path.name, system_name)

    def test_rejects_unknown_checkpoint_names(self):
        for bad in ("checkpoint-foo.chk", "checkpoint.chk", "state.chk"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    serialized_system_path(Path(bad))


class SerializeXmlAtomicTests(unittest.TestCase):
    def test_roundtrip_preserves_particles_forces_and_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "system-prod.xml"
            serialize_xml_atomic(make_toy_system(), path)
            restored = mm.XmlSerializer.deserialize(path.read_text())
        self.assertEqual(restored.getNumParticles(), 4)
        self.assertEqual(restored.getNumForces(), 2)
        self.assertAlmostEqual(restored.getParticleMass(0).value_in_unit(u.amu), 2.0)
        restraint = restored.getForce(0)
        self.assertIsInstance(restraint, mm.CustomExternalForce)
        self.assertEqual(restraint.getNumParticles(), 4)
        self.assertEqual(restraint.getNumGlobalParameters(), 1)
        self.assertAlmostEqual(restraint.getGlobalParameterDefaultValue(0), 100.0)

    def test_roundtrip_preserves_integrator_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "integrator.xml"
            serialize_xml_atomic(make_toy_integrator(), path)
            restored = mm.XmlSerializer.deserialize(path.read_text())
        self.assertIsInstance(restored, mm.LangevinMiddleIntegrator)
        self.assertAlmostEqual(restored.getTemperature().value_in_unit(u.kelvin), 300.0)
        self.assertAlmostEqual(
            restored.getFriction().value_in_unit(u.picosecond ** -1), 1.0
        )
        self.assertAlmostEqual(
            restored.getStepSize().value_in_unit(u.femtosecond), 2.0
        )


class LoadSerializedResumeTests(unittest.TestCase):
    def test_resume_continues_from_checkpoint_state(self):
        platform = mm.Platform.getPlatformByName("CPU")
        system = make_toy_system()
        integrator = make_toy_integrator()
        topology = make_toy_topology()
        positions = [
            mm.Vec3(0.5 + 0.2 * index, 0.1 * index, -0.1 * index) for index in range(4)
        ] * u.nanometer
        simulation = app.Simulation(topology, system, integrator, platform)
        simulation.context.setPositions(positions)
        simulation.step(10)
        reference = simulation.context.getState(getPositions=True, getVelocities=True)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = write_resume_artifacts(
                directory, simulation, system, integrator, topology, positions
            )
            resumed = load_serialized_resume(
                checkpoint,
                Path(directory) / "integrator.xml",
                Path(directory) / "system_final.pdb",
                platform,
                None,
            )
            self.assertEqual(resumed.currentStep, 10)
            state = resumed.context.getState(getPositions=True, getVelocities=True)
            for value, expected in zip(state.getPositions(), reference.getPositions()):
                for component, expected_component in zip(value, expected):
                    self.assertAlmostEqual(
                        component.value_in_unit(u.nanometer),
                        expected_component.value_in_unit(u.nanometer),
                        places=5,
                    )
            for value, expected in zip(state.getVelocities(), reference.getVelocities()):
                for component, expected_component in zip(value, expected):
                    self.assertAlmostEqual(
                        component.value_in_unit(u.nanometer / u.picosecond),
                        expected_component.value_in_unit(u.nanometer / u.picosecond),
                        places=4,
                    )
            resumed.step(5)
            self.assertEqual(resumed.currentStep, 15)

    def test_rejects_topology_atom_count_mismatch(self):
        platform = mm.Platform.getPlatformByName("CPU")
        system = make_toy_system()
        integrator = make_toy_integrator()
        topology = make_toy_topology()
        positions = [
            mm.Vec3(0.5 * index, 0.0, 0.0) for index in range(4)
        ] * u.nanometer
        simulation = app.Simulation(topology, system, integrator, platform)
        simulation.context.setPositions(positions)
        simulation.step(2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = write_resume_artifacts(
                directory, simulation, system, integrator, topology, positions
            )
            short_topology = make_toy_topology(num_atoms=3)
            with open(Path(directory) / "system_final.pdb", "w") as handle:
                app.PDBFile.writeFile(short_topology, positions[:3], handle)
            with self.assertRaises(ValueError):
                load_serialized_resume(
                    checkpoint,
                    Path(directory) / "integrator.xml",
                    Path(directory) / "system_final.pdb",
                    platform,
                    None,
                )

    def test_missing_artifact_raises_file_not_found(self):
        platform = mm.Platform.getPlatformByName("CPU")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "checkpoint-prod.chk"
            with self.assertRaises(FileNotFoundError):
                load_serialized_resume(
                    checkpoint,
                    directory / "integrator.xml",
                    directory / "system_final.pdb",
                    platform,
                    None,
                )


class TruncateStateDataTests(unittest.TestCase):
    HEADER = '#"Step","Potential Energy (kJ/mole)","Temperature (K)"\n'

    def test_keeps_rows_up_to_checkpoint_and_preserves_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state_data.csv"
            path.write_text(
                self.HEADER
                + "250,-1.0,300.0\n"
                + "500,-2.0,300.5\n"
                + "750,-3.0,301.0\n"
            )
            kept = truncate_state_data_to_step(path, 500)
            self.assertEqual(kept, 2)
            self.assertEqual(
                path.read_text(),
                self.HEADER + "250,-1.0,300.0\n500,-2.0,300.5\n",
            )

    def test_raises_when_step_column_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state_data.csv"
            path.write_text('#"Time","Temperature (K)"\n1.0,300.0\n')
            with self.assertRaises(ValueError):
                truncate_state_data_to_step(path, 500)


if __name__ == "__main__":
    unittest.main()
