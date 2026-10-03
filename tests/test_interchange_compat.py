import unittest

import numpy as np
import openmm as mm
from openmm import app, unit

from openff.interchange import Interchange
from prot_lig_runner.interchange_compat import enable_batched_openmm_topology_conversion


class InterchangeCompatibilityTests(unittest.TestCase):
    def test_batched_openmm_conversion_preserves_disconnected_molecules(self):
        enable_batched_openmm_topology_conversion(batch_size=2)
        topology = app.Topology()
        chain = topology.addChain("A")
        system = mm.System()
        nonbonded = mm.NonbondedForce()
        nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
        nonbonded.setCutoffDistance(1.0 * unit.nanometers)
        box_vectors = (
            mm.Vec3(3.0, 0.0, 0.0),
            mm.Vec3(0.0, 3.0, 0.0),
            mm.Vec3(0.0, 0.0, 3.0),
        ) * unit.nanometers
        topology.setPeriodicBoxVectors(box_vectors)
        system.setDefaultPeriodicBoxVectors(*box_vectors)
        positions = []

        for residue_index in range(8):
            residue = topology.addResidue("HOH", chain, id=str(residue_index + 1))
            oxygen = topology.addAtom("O", app.element.oxygen, residue)
            hydrogen_1 = topology.addAtom("H1", app.element.hydrogen, residue)
            hydrogen_2 = topology.addAtom("H2", app.element.hydrogen, residue)
            topology.addBond(oxygen, hydrogen_1)
            topology.addBond(oxygen, hydrogen_2)
            for _ in range(3):
                system.addParticle(18.0 * unit.dalton)
                nonbonded.addParticle(0.0, 0.3, 0.0)
                positions.append([0.0, 0.0, 0.0])

        system.addForce(nonbonded)
        interchange = Interchange.from_openmm(
            topology=topology,
            system=system,
            positions=np.asarray(positions) * unit.nanometers,
        )

        self.assertEqual(interchange.topology.n_atoms, 24)
        self.assertEqual(len(list(interchange.topology.molecules)), 8)
