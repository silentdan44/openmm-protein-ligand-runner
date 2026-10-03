import argparse
import hashlib
import json
import logging
import math
import random
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import openmm as mm
from openmm import app
from openmm import unit as u
from openff.interchange import Interchange
from openff.toolkit import Molecule, Topology
from openff.units.openmm import to_openmm as off_to_omm
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from pdbfixer import PDBFixer
from stability_diagnostics import (
    StabilityReporter,
    diagnose_failure,
    find_severe_overlaps,
    write_diagnostics,
)

TEMPERATURE_K = 300.0
PRESSURE_BAR = 1.0
FRICTION_PER_PS = 1.0
TIMESTEP_FS = 2.0
IONIC_STRENGTH_MOLAR = 0.15
PADDING_NM = 1.0
RESTRAINT_K = 1000.0 * u.kilojoule_per_mole / u.nanometer**2
PROTEIN_FF = "amber14/protein.ff14SB.xml"
WATER_FF = "amber14/tip3p.xml"
PROTEIN_RESIDUES = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)
PROGRESS_FILE = "run_state.json"
logger = logging.getLogger(__name__)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description="Run a protein-ligand OpenMM simulation.")
    parser.add_argument("--receptor", default="input/AQP3.pdb", help="Receptor PDB path")
    parser.add_argument("--ligand", default="input/rank1.sdf", help="Single-ligand SDF path")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--nvt-steps", type=positive_int, default=250_000)
    parser.add_argument("--npt-steps", type=positive_int, default=500_000)
    parser.add_argument("--prod-steps", type=positive_int, default=250_000_000)
    parser.add_argument("--report-interval", type=positive_int, default=10_000)
    parser.add_argument("--xtc-interval", type=positive_int, default=50_000)
    parser.add_argument("--checkpoint-interval", type=positive_int, default=50_000)
    parser.add_argument("--minimize-iterations", type=positive_int, default=10_000)
    parser.add_argument("--temperature-k", type=positive_float, default=TEMPERATURE_K)
    parser.add_argument("--pressure-bar", type=positive_float, default=PRESSURE_BAR)
    parser.add_argument("--friction-per-ps", type=positive_float, default=FRICTION_PER_PS)
    parser.add_argument("--timestep-fs", type=positive_float, default=TIMESTEP_FS)
    parser.add_argument("--ionic-strength-molar", type=float, default=IONIC_STRENGTH_MOLAR)
    parser.add_argument("--padding-nm", type=positive_float, default=PADDING_NM)
    parser.add_argument("--seed", type=positive_int, default=2026)
    parser.add_argument(
        "--platform", choices=("auto", "CUDA", "OpenCL", "CPU", "Reference"), default="auto"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from the checkpoint in run_state.json")
    return parser.parse_args()


def add_positional_restraints(system, positions, atom_indices, k):
    force = mm.CustomExternalForce("0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", k)
    for parameter in ("x0", "y0", "z0"):
        force.addPerParticleParameter(parameter)
    for index in atom_indices:
        force.addParticle(index, positions[index].value_in_unit(u.nanometers))
    system.addForce(force)
    return force


def make_barostat(args):
    barostat = mm.MonteCarloBarostat(
        args.pressure_bar * u.bar,
        args.temperature_k * u.kelvin,
        25,
    )
    barostat.setRandomNumberSeed(derived_seed(args.seed, 1))
    return barostat


def get_platform(requested):
    if requested != "auto":
        platform = mm.Platform.getPlatformByName(requested)
        properties = {"CudaPrecision": "mixed"} if requested == "CUDA" else {}
        return platform, properties

    for name, properties in (("CUDA", {"CudaPrecision": "mixed"}), ("OpenCL", {}), ("CPU", {}), ("Reference", {})):
        try:
            return mm.Platform.getPlatformByName(name), properties
        except mm.OpenMMException:
            continue
    raise RuntimeError("No OpenMM platform is available")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_configuration(args, receptor, ligand):
    return {
        "inputs": {"receptor_sha256": sha256_file(receptor), "ligand_sha256": sha256_file(ligand)},
        "runner_sha256": sha256_file(Path(__file__)),
        "nvt_steps": args.nvt_steps,
        "npt_steps": args.npt_steps,
        "prod_steps": args.prod_steps,
        "report_interval": args.report_interval,
        "xtc_interval": args.xtc_interval,
        "temperature_k": args.temperature_k,
        "pressure_bar": args.pressure_bar,
        "friction_per_ps": args.friction_per_ps,
        "timestep_fs": args.timestep_fs,
        "ionic_strength_molar": args.ionic_strength_molar,
        "padding_nm": args.padding_nm,
        "seed": args.seed,
        "minimize_iterations": args.minimize_iterations,
        "protein_forcefield": PROTEIN_FF,
        "water_forcefield": WATER_FF,
        "software": {
            "openmm": mm.__version__,
            "openff_toolkit": package_version("openff-toolkit"),
            "openff_interchange": package_version("openff-interchange"),
            "openmmforcefields": package_version("openmmforcefields"),
        },
    }


def write_progress(path, phase, configuration, platform_name, checkpoint_name):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "phase": phase,
                "configuration": configuration,
                "platform": platform_name,
                "checkpoint": checkpoint_name,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(path)


def save_checkpoint_atomically(simulation, path):
    temporary = path.with_name(path.name + ".tmp")
    simulation.saveCheckpoint(str(temporary))
    temporary.replace(path)


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def derived_seed(seed, offset):
    return (seed + offset - 1) % 2_147_483_647 + 1


def main():
    args = parse_args()
    receptor_path = Path(args.receptor)
    ligand_path = Path(args.ligand)
    output_dir = Path(args.output)

    for label, path in (("receptor", receptor_path), ("ligand", ligand_path)):
        if not path.is_file():
            raise SystemExit(f"{label} file does not exist or is not a file: {path}")
    if not math.isfinite(args.ionic_strength_molar) or args.ionic_strength_molar < 0:
        raise SystemExit("--ionic-strength-molar must be a finite number that is zero or greater")

    output_dir.mkdir(parents=True, exist_ok=True)
    resume = args.resume
    progress_path = output_dir / PROGRESS_FILE
    config = run_configuration(args, receptor_path, ligand_path)
    previous_progress = None
    if resume:
        if not progress_path.is_file():
            raise SystemExit(f"Cannot resume: {progress_path} is required")
        previous_progress = json.loads(progress_path.read_text())
        checkpoint_path = output_dir / previous_progress.get("checkpoint", "")
        if not checkpoint_path.is_file():
            raise SystemExit(f"Cannot resume: checkpoint file is missing: {checkpoint_path}")
        if previous_progress["configuration"] != config:
            raise SystemExit("Cannot resume: input files or simulation settings differ from checkpoint run")
        if previous_progress["phase"] not in {"nvt", "npt", "prod", "complete"}:
            raise SystemExit(f"Cannot resume: invalid run phase {previous_progress['phase']!r}")
        if previous_progress["phase"] == "complete":
            raise SystemExit("This run is already complete; start a new run without --resume")

    log_mode = "a" if resume else "w"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(output_dir / "simulation.log", mode=log_mode), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger.info("Run configuration: %s", json.dumps(config, sort_keys=True))
    logger.info(
        "Versions: OpenMM=%s OpenFF-Toolkit=%s OpenFF-Interchange=%s openmmforcefields=%s",
        mm.__version__,
        package_version("openff-toolkit"),
        package_version("openff-interchange"),
        package_version("openmmforcefields"),
    )

    diagnostic_reporter = None
    current_phase = "preparation"
    initial_overlaps = []
    try:
        logger.info("Loading ligand from %s", ligand_path)
        ligand = Molecule.from_file(str(ligand_path))
        if isinstance(ligand, list):
            if len(ligand) != 1:
                raise ValueError(f"SDF must contain exactly one ligand; found {len(ligand)}")
            ligand = ligand[0]
        if ligand.n_atoms == 0 or not ligand.conformers:
            raise ValueError("Ligand SDF must contain atoms and at least one 3D conformer")

        logger.info("Preparing receptor %s with PDBFixer", receptor_path)
        fixer = PDBFixer(filename=str(receptor_path))
        modeller = app.Modeller(fixer.topology, fixer.positions)
        hydrogens = [atom for atom in modeller.topology.atoms() if atom.element.symbol == "H"]
        modeller.delete(hydrogens)
        fixer.topology, fixer.positions = modeller.topology, modeller.positions

        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

        smirnoff = SMIRNOFFTemplateGenerator(molecules=ligand)
        forcefield = app.ForceField(PROTEIN_FF, WATER_FF)
        forcefield.registerTemplateGenerator(smirnoff.generator)
        modeller = app.Modeller(fixer.topology, fixer.positions)

        ligand_topology = Topology.from_molecules([ligand]).to_openmm()
        ligand_positions = off_to_omm(ligand.conformers[0])
        for residue in ligand_topology.residues():
            residue.name = "LIG"
        modeller.add(ligand_topology, ligand_positions)

        logger.info("Adding hydrogens and solvating system")
        python_rng_state = random.getstate()
        random.seed(args.seed)
        try:
            modeller.addHydrogens(forcefield=forcefield, pH=7.0)
            modeller.addSolvent(
                forcefield,
                padding=args.padding_nm * u.nanometer,
                ionicStrength=args.ionic_strength_molar * u.molar,
            )
        finally:
            random.setstate(python_rng_state)

        with open(output_dir / "system_final.pdb", "w") as pdb_file:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, pdb_file)

        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * u.nanometer,
            constraints=app.HBonds,
        )
        initial_overlaps = find_severe_overlaps(modeller.topology, modeller.positions)
        if initial_overlaps:
            logger.warning(
                "Found %d possible severe nonbonded close contacts in prepared coordinates; "
                "the first pairs are included in failure_diagnostics.json: %s",
                len(initial_overlaps), json.dumps(initial_overlaps),
            )
        for index, residue in enumerate(modeller.topology.residues()):
            residue.id = str(index + 1)

        interchange = Interchange.from_openmm(
            topology=modeller.topology, system=system, positions=modeller.positions
        )
        if modeller.topology.getPeriodicBoxVectors():
            interchange.box = modeller.topology.getPeriodicBoxVectors()
        interchange.to_top(str(output_dir / "system.top"))
        interchange.to_gro(str(output_dir / "system.gro"))

        protein_indices = [
            atom.index for atom in modeller.topology.atoms() if atom.residue.name in PROTEIN_RESIDUES
        ]
        ligand_indices = [atom.index for atom in modeller.topology.atoms() if atom.residue.name == "LIG"]
        if not protein_indices:
            raise ValueError("No standard protein residues found; cannot apply protein restraints")
        if not ligand_indices:
            raise ValueError("No LIG residue found; cannot apply ligand restraints")
        logger.info(
            "Applying restraints to %d protein atoms and %d ligand atoms",
            len(protein_indices), len(ligand_indices)
        )

        add_positional_restraints(
            system, modeller.positions, protein_indices + ligand_indices, RESTRAINT_K
        )
        phase = "nvt"
        if not resume:
            checkpoint_path = output_dir / f"checkpoint-{phase}.chk"
        else:
            checkpoint_path = output_dir / previous_progress["checkpoint"]
        current_platform = args.platform
        if resume:
            phase = previous_progress["phase"]
            current_platform = previous_progress["platform"]
            if args.platform != "auto" and args.platform != current_platform:
                raise ValueError("Requested platform differs from the checkpoint platform")
        if phase in ("npt", "prod"):
            system.addForce(make_barostat(args))
            if phase == "prod":
                for force_index in reversed(range(system.getNumForces())):
                    if isinstance(system.getForce(force_index), mm.CustomExternalForce):
                        system.removeForce(force_index)

        platform, platform_properties = get_platform(current_platform)
        logger.info("Using OpenMM platform %s", platform.getName())
        integrator = mm.LangevinMiddleIntegrator(
            args.temperature_k * u.kelvin,
            args.friction_per_ps / u.picosecond,
            args.timestep_fs * u.femtoseconds,
        )
        integrator.setRandomNumberSeed(derived_seed(args.seed, 2))
        simulation = app.Simulation(
            modeller.topology, system, integrator, platform, platform_properties
        )
        if resume:
            simulation.loadCheckpoint(str(checkpoint_path))
            logger.info("Resumed %s phase at step %d", phase, simulation.currentStep)
        else:
            simulation.context.setPositions(modeller.positions)
            logger.info("Minimizing energy")
            simulation.minimizeEnergy(maxIterations=args.minimize_iterations)
            simulation.context.setVelocitiesToTemperature(
                args.temperature_k * u.kelvin, derived_seed(args.seed, 0)
            )

        state_data_file = output_dir / "state_data.csv"
        trajectory_file = output_dir / "trajectory.xtc"
        append_outputs = resume and state_data_file.exists() and state_data_file.stat().st_size > 0
        append_trajectory = resume and trajectory_file.exists() and trajectory_file.stat().st_size > 0
        diagnostic_reporter = StabilityReporter(
            args.report_interval, args.temperature_k, system.getNumParticles()
        )
        simulation.reporters.extend(
            [
                diagnostic_reporter,
                app.StateDataReporter(
                    sys.stdout, args.report_interval, step=True, potentialEnergy=True,
                    kineticEnergy=True, density=True, temperature=True, speed=True, separator="\t"
                ),
                app.StateDataReporter(
                    str(state_data_file), args.report_interval, step=True, potentialEnergy=True,
                    kineticEnergy=True, density=True, temperature=True, speed=True,
                    separator=",", append=append_outputs
                ),
                app.XTCReporter(
                    str(trajectory_file), args.xtc_interval, append=append_trajectory
                ),
                app.CheckpointReporter(str(checkpoint_path), args.checkpoint_interval),
            ]
        )

        def checkpoint(phase_name):
            nonlocal checkpoint_path
            checkpoint_path = output_dir / f"checkpoint-{phase_name}.chk"
            save_checkpoint_atomically(simulation, checkpoint_path)
            simulation.reporters = [
                reporter
                for reporter in simulation.reporters
                if not isinstance(reporter, app.CheckpointReporter)
            ]
            simulation.reporters.append(
                app.CheckpointReporter(str(checkpoint_path), args.checkpoint_interval)
            )
            write_progress(
                progress_path, phase_name, config, platform.getName(), checkpoint_path.name
            )

        nvt_end = args.nvt_steps
        npt_end = args.nvt_steps + args.npt_steps
        total_end = npt_end + args.prod_steps
        current_phase = phase
        if phase == "nvt":
            current_phase = "nvt"
            checkpoint("nvt")
            remaining = max(0, nvt_end - simulation.currentStep)
            logger.info("Running NVT for %d remaining steps", remaining)
            simulation.step(remaining)
            simulation.system.addForce(make_barostat(args))
            simulation.context.reinitialize(preserveState=True)
            diagnostic_reporter.previous_energy_kj_mol = None
            phase = "npt"
            checkpoint(phase)

        if phase == "npt":
            current_phase = "npt"
            remaining = max(0, npt_end - simulation.currentStep)
            logger.info("Running NPT for %d remaining steps", remaining)
            simulation.step(remaining)
            for force_index in reversed(range(simulation.system.getNumForces())):
                if isinstance(simulation.system.getForce(force_index), mm.CustomExternalForce):
                    simulation.system.removeForce(force_index)
            simulation.context.reinitialize(preserveState=True)
            diagnostic_reporter.previous_energy_kj_mol = None
            phase = "prod"
            checkpoint(phase)

        if phase == "prod":
            current_phase = "prod"
            remaining = max(0, total_end - simulation.currentStep)
            logger.info("Running production for %d remaining steps", remaining)
            simulation.step(remaining)
            simulation.saveState(str(output_dir / "final_state.xml"))
            checkpoint("complete")
            logger.info("Simulation completed at step %d", simulation.currentStep)

    except Exception as error:
        logger.exception("Simulation failed")
        metrics = diagnostic_reporter
        diagnostics = diagnose_failure(
            error,
            step=metrics.last_step if metrics else getattr(locals().get("simulation"), "currentStep", None),
            phase=current_phase,
            temperature_k=metrics.last_temperature_k if metrics else None,
            energy_kj_mol=metrics.last_energy_kj_mol if metrics else None,
            overlaps=initial_overlaps,
        )
        try:
            write_diagnostics(output_dir / "failure_diagnostics.json", diagnostics)
            logger.error(
                "Failure diagnosis: %s Suggested action: %s Diagnostics saved to %s",
                diagnostics["cause"], diagnostics["suggested_action"],
                output_dir / "failure_diagnostics.json",
            )
        except OSError:
            logger.exception("Could not write failure_diagnostics.json")
        raise


if __name__ == "__main__":
    main()
