"""Small, dependency-free diagnostics for common OpenMM instability symptoms."""

from __future__ import annotations

import json
import math
from pathlib import Path



_COVALENT_RADII_NM = {
    "C": 0.076,
    "N": 0.071,
    "O": 0.066,
    "F": 0.057,
    "P": 0.107,
    "S": 0.105,
    "CL": 0.102,
}
_K_B_KJ_PER_MOL_K = 0.008314462618


def _units():
    import openmm

    return openmm.unit


def _atom_label(atom):
    residue = atom.residue
    chain = getattr(residue, "chain", None)
    chain_id = getattr(chain, "id", "") or "?"
    return f"{chain_id}:{residue.name}{residue.id}:{atom.name}[{atom.index}]"


def find_severe_overlaps(topology, positions, max_pairs=10):
    """Return unusually close nonbonded heavy-atom pairs using a spatial grid."""
    atoms = list(topology.atoms())
    xyz = positions.value_in_unit(_units().nanometer)
    bonded = {tuple(sorted((a.index, b.index))) for a, b in topology.bonds()}
    cell_size = 0.22
    cells = {}
    found = []

    for atom in atoms:
        symbol = (atom.element.symbol if atom.element else "").upper()
        radius = _COVALENT_RADII_NM.get(symbol)
        if radius is None:
            continue
        point = tuple(float(x) for x in xyz[atom.index])
        cell = tuple(math.floor(x / cell_size) for x in point)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for other, other_point, other_radius in cells.get(
                        (cell[0] + dx, cell[1] + dy, cell[2] + dz), ()
                    ):
                        if tuple(sorted((atom.index, other.index))) in bonded:
                            continue
                        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(point, other_point)))
                        cutoff = 0.9 * (radius + other_radius)
                        if distance < cutoff:
                            found.append(
                                {
                                    "atom_1": _atom_label(other),
                                    "atom_2": _atom_label(atom),
                                    "distance_nm": round(distance, 4),
                                    "warning_cutoff_nm": round(cutoff, 4),
                                }
                            )
        cells.setdefault(cell, []).append((atom, point, radius))

    found.sort(key=lambda pair: pair["distance_nm"])
    return found[:max_pairs]


def diagnose_failure(error, *, step=None, phase=None, temperature_k=None, energy_kj_mol=None,
                     overlaps=None):
    """Create a concise, actionable explanation from an OpenMM error and last metrics."""
    message = str(error)
    text = message.lower()
    if any(token in text for token in ("nan", "infinite", "not finite", "non-finite")):
        cause = "The simulation produced a non-finite coordinate or energy, usually after a force spike."
        action = "Inspect the closest-atom warnings and starting geometry; minimize the structure and retry with a smaller timestep."
    elif "potential energy changed" in text:
        cause = "Potential energy changed sharply between reports, a common sign of a sudden force spike."
        action = "Inspect the closest-atom warnings and starting geometry; minimize the structure and retry with a smaller timestep."
    elif any(token in text for token in ("constraint", "shake", "rattle")):
        cause = "OpenMM could not keep constrained bonds at their required lengths."
        action = "Check bond geometry and try a smaller timestep; confirm the topology and hydrogen placement."
    elif any(token in text for token in ("cuda", "opencl", "device", "context lost")):
        cause = "The compute platform or device reported an execution failure."
        action = "Retry on CPU or another available platform and check GPU/runtime compatibility."
    elif temperature_k is not None and math.isfinite(temperature_k):
        cause = f"The measured kinetic temperature rose to {temperature_k:.0f} K, suggesting the system gained excessive energy."
        action = "Inspect atom-overlap warnings and energy trends; minimize first and retry with a smaller timestep."
    else:
        cause = "The simulation stopped with an error that does not match a known instability signature."
        action = "Use the original error and last recorded metrics to inspect the failing phase; no single cause could be confirmed."

    result = {
        "phase": phase,
        "step": step,
        "cause": cause,
        "suggested_action": action,
        "error": message,
        "last_temperature_k": round(temperature_k, 2) if temperature_k is not None and math.isfinite(temperature_k) else None,
        "last_potential_energy_kj_mol": round(energy_kj_mol, 3) if energy_kj_mol is not None and math.isfinite(energy_kj_mol) else None,
        "possible_severe_overlaps": overlaps or [],
    }
    if overlaps:
        result["cause"] += " Severe close contacts were also found in the prepared coordinates; they may contribute to the force spike."
        result["suggested_action"] = "Inspect the listed atom pairs, correct the starting geometry, minimize, and retry with a smaller timestep."
    return result


def write_diagnostics(path, diagnostics):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class StabilityReporter:
    """Sample kinetic temperature and energy, stopping on clearly catastrophic values."""

    def __init__(self, interval, target_temperature_k, particle_count):
        self.interval = interval
        self.target_temperature_k = target_temperature_k
        self.particle_count = max(particle_count, 1)
        self.last_step = None
        self.last_temperature_k = None
        self.last_energy_kj_mol = None
        self.previous_energy_kj_mol = None

    def describeNextReport(self, simulation):
        steps = self.interval - simulation.currentStep % self.interval
        # OpenMM 8.2 uses the five-field reporter protocol (steps, positions,
        # velocities, forces, energies); environment.yml pins that version.
        return (steps, False, False, False, True)

    def report(self, simulation, state):
        energy = state.getPotentialEnergy().value_in_unit(_units().kilojoule_per_mole)
        kinetic = state.getKineticEnergy().value_in_unit(_units().kilojoule_per_mole)
        temperature = 2.0 * kinetic / (
            3.0 * self.particle_count * _K_B_KJ_PER_MOL_K
        )
        self.last_step = simulation.currentStep
        self.last_temperature_k = temperature
        self.last_energy_kj_mol = energy

        if not math.isfinite(energy) or not math.isfinite(temperature):
            raise RuntimeError("Non-finite energy or temperature detected by stability monitor")
        if temperature > max(1500.0, 5.0 * self.target_temperature_k):
            raise RuntimeError("Kinetic temperature exceeded the stability limit")
        if self.previous_energy_kj_mol is not None:
            jump_per_atom = abs(energy - self.previous_energy_kj_mol) / self.particle_count
            if jump_per_atom > 25.0:
                raise RuntimeError("Potential energy changed by more than 25 kJ/mol per atom between reports")
        self.previous_energy_kj_mol = energy
