import argparse
import hashlib
import json
import logging
import math
import random
import shutil
import subprocess
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
    parser.add_argument(
        "--receptor", default="input/aqp3_solution_test_receptor.pdb", help="Receptor PDB path"
    )
    parser.add_argument(
        "--ligand", default="input/aqp3_solution_test_ligand.sdf", help="Single-ligand SDF path"
    )
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
    parser.add_argument("--membrane", action="store_true", help="Build a lipid membrane and water system")
    parser.add_argument(
        "--orientation", choices=("tmdet", "preoriented", "manual"), default="tmdet",
        help="How to orient the receptor for a membrane run",
    )
    parser.add_argument(
        "--lipid-type", choices=("POPC", "POPE", "DLPC", "DLPE", "DMPC", "DOPC", "DPPC"),
        default="POPC", help="Single lipid type for the membrane",
    )
    parser.add_argument("--membrane-center-z-nm", type=float, default=0.0)
    parser.add_argument("--membrane-padding-nm", type=positive_float, default=1.0)
    parser.add_argument("--manual-membrane-normal", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--manual-membrane-center-angstrom", type=float, nargs=3, metavar=("X", "Y", "Z"),
        help="Point on the bilayer center plane in the input coordinates",
    )
    parser.add_argument("--seed", type=positive_int, default=2026)
    parser.add_argument(
        "--platform", choices=("auto", "CUDA", "OpenCL", "CPU", "Reference"), default="auto"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from the checkpoint in run_state.json")
    args = parser.parse_args()
    if args.membrane:
        if not math.isfinite(args.membrane_center_z_nm):
            parser.error("--membrane-center-z-nm must be finite")
        if args.orientation == "manual":
            if args.manual_membrane_normal is None or args.manual_membrane_center_angstrom is None:
                parser.error("manual membrane orientation requires --manual-membrane-normal and --manual-membrane-center-angstrom")
            if not all(math.isfinite(value) for value in args.manual_membrane_normal + args.manual_membrane_center_angstrom):
                parser.error("manual membrane coordinates must be finite")
            if math.sqrt(sum(value * value for value in args.manual_membrane_normal)) < 1e-8:
                parser.error("--manual-membrane-normal must be nonzero")
        elif args.manual_membrane_normal is not None or args.manual_membrane_center_angstrom is not None:
            parser.error("manual orientation parameters are only valid with --orientation manual")
    return args


def vector_rotation(source, target):
    """Return a 3x3 rotation matrix that maps source onto target."""
    import numpy as np

    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.dot(source, target))
    if sine < 1e-12:
        if cosine > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis -= source * np.dot(axis, source)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = cross / sine
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + skew * sine + (skew @ skew) * (1.0 - cosine)


def transform_positions(positions, rotation, source_center, target_center):
    import numpy as np

    coords = np.asarray(positions.value_in_unit(u.nanometers), dtype=float)
    transformed = (coords - source_center) @ rotation.T + target_center
    return transformed * u.nanometers


def atom_signature(atom):
    return (atom.residue.chain.id, atom.residue.id, atom.residue.name, atom.name)


def membrane_geometry_summary(topology, positions, lipid_type, membrane_center_z_nm):
    import numpy as np

    coords = np.asarray(positions.value_in_unit(u.nanometers), dtype=float)
    # OpenMM's bundled lipid patches are PDB files, whose residue-name field is
    # only three characters (POPC/POPE become POP, DLPC/DLPE become DLP, etc.).
    lipid_residue_names = {lipid_type, lipid_type[:3]}
    groups = {"protein": [], "ligand": [], "lipid": [], "water": []}
    protein_coords = []
    ligand_coords = []
    for atom in topology.atoms():
        if atom.element is None or atom.element.symbol == "H":
            continue
        residue_name = atom.residue.name
        xyz = coords[atom.index]
        if residue_name in PROTEIN_RESIDUES:
            groups["protein"].append(xyz)
            protein_coords.append(xyz)
        elif residue_name == "LIG":
            groups["ligand"].append(xyz)
            ligand_coords.append(xyz)
        elif residue_name in lipid_residue_names:
            groups["lipid"].append(xyz)
        elif residue_name in {"HOH", "WAT", "SOL"}:
            groups["water"].append(xyz)
    result = {}
    for name, atoms in groups.items():
        array = np.asarray(atoms, dtype=float)
        result[name] = {
            "heavy_atom_count": int(len(array)),
            "z_range_nm": [float(array[:, 2].min()), float(array[:, 2].max())] if len(array) else None,
            "z_mean_nm": float(array[:, 2].mean()) if len(array) else None,
        }
    lipid_z = np.asarray(groups["lipid"], dtype=float)[:, 2]
    result["lipid_leaflets"] = {
        "below_center_heavy_atoms": int(np.sum(lipid_z < membrane_center_z_nm)),
        "above_center_heavy_atoms": int(np.sum(lipid_z > membrane_center_z_nm)),
    }
    if protein_coords and ligand_coords:
        delta = np.asarray(ligand_coords)[:, None, :] - np.asarray(protein_coords)[None, :, :]
        result["minimum_ligand_protein_heavy_distance_nm"] = float(
            np.sqrt(np.sum(delta * delta, axis=2)).min()
        )
    return result


def get_tmdet_transform(topology, positions, receptor_path, output_dir):
    """Run standalone TmDet and recover its rigid transform from matched heavy atoms."""
    import numpy as np

    executable = shutil.which("tmdet")
    if executable is None:
        raise RuntimeError("TmDet was requested but the tmdet executable is not installed")
    oriented_path = output_dir / "tmdet_oriented.cif"
    annotation_path = output_dir / "tmdet_annotation.xml"
    command = [executable, "-pi", str(receptor_path), "-po", str(oriented_path), "-x", str(annotation_path)]
    logger.info("Running TmDet: %s", " ".join(command))
    completed = subprocess.run(command, text=True, capture_output=True)
    (output_dir / "tmdet.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0 or not oriented_path.is_file():
        raise RuntimeError(f"TmDet failed (exit {completed.returncode}); see {output_dir / 'tmdet.log'}")
    oriented = app.PDBxFile(str(oriented_path))
    # Keep duplicate signatures: PDB inputs can repeat an auth chain ID after
    # TER records, while TmDet may emit those subunits as separate CIF chains.
    source_atoms = {}
    for atom in topology.atoms():
        if atom.element is not None and atom.element.symbol != "H":
            source_atoms.setdefault(atom_signature(atom), []).append(atom.index)
    target_atoms = {}
    for atom in oriented.topology.atoms():
        if atom.element is not None and atom.element.symbol != "H":
            target_atoms.setdefault(atom_signature(atom), []).append(atom.index)
    source_indices, target_indices = [], []
    for key in sorted(source_atoms.keys() & target_atoms.keys()):
        count = min(len(source_atoms[key]), len(target_atoms[key]))
        source_indices.extend(source_atoms[key][:count])
        target_indices.extend(target_atoms[key][:count])
    if len(source_indices) < 100:
        raise RuntimeError(
            f"TmDet output matches only {len(source_indices)} receptor heavy atoms; cannot safely transfer orientation"
        )
    source = np.asarray([positions[i].value_in_unit(u.nanometers) for i in source_indices])
    target = np.asarray([oriented.positions[i].value_in_unit(u.nanometers) for i in target_indices])
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(right.T @ left.T)
    rotation = right.T @ correction @ left.T
    fitted = (source - source_center) @ rotation.T + target_center
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))
    if rmsd > 0.02:
        raise RuntimeError(f"TmDet transform is not rigid (matched-atom RMSD {rmsd:.4f} nm)")
    logger.info("Recovered TmDet rigid transform from %d heavy atoms (RMSD %.4f nm)", len(source_indices), rmsd)
    return rotation, source_center, target_center


def prepare_membrane_orientation(args, fixer, ligand_positions, receptor_path, output_dir):
    import numpy as np

    if args.orientation == "preoriented":
        return fixer.positions, ligand_positions
    if args.orientation == "tmdet":
        rotation, source_center, target_center = get_tmdet_transform(
            fixer.topology, fixer.positions, receptor_path, output_dir
        )
        target_center = target_center.copy()
        target_center[2] += args.membrane_center_z_nm
    else:
        normal = np.asarray(args.manual_membrane_normal, dtype=float)
        center = np.asarray(args.manual_membrane_center_angstrom, dtype=float) / 10.0
        rotation = vector_rotation(normal, (0.0, 0.0, 1.0))
        source_center = center
        target_center = np.asarray((0.0, 0.0, args.membrane_center_z_nm))
    return (
        transform_positions(fixer.positions, rotation, source_center, target_center),
        transform_positions(ligand_positions, rotation, source_center, target_center),
    )


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
    if args.membrane:
        barostat = mm.MonteCarloMembraneBarostat(
            args.pressure_bar * u.bar,
            0.0 * u.bar * u.nanometer,
            args.temperature_k * u.kelvin,
            mm.MonteCarloMembraneBarostat.XYIsotropic,
            mm.MonteCarloMembraneBarostat.ZFree,
            25,
        )
    else:
        barostat = mm.MonteCarloBarostat(
            args.pressure_bar * u.bar,
            args.temperature_k * u.kelvin,
            25,
        )
    barostat.setRandomNumberSeed(derived_seed(args.seed, 1))
    return barostat


def get_platform(requested):
    candidates = (requested,) if requested != "auto" else ("CUDA", "OpenCL", "CPU")
    failures = []
    for name in candidates:
        properties = {"CudaPrecision": "mixed"} if name == "CUDA" else {}
        try:
            platform = mm.Platform.getPlatformByName(name)
            # A registered plugin does not guarantee that its device/context
            # can actually be initialized. Probe it before committing a run.
            probe_system = mm.System()
            probe_system.addParticle(1.0 * u.dalton)
            probe_integrator = mm.VerletIntegrator(0.001 * u.picoseconds)
            probe_context = mm.Context(probe_system, probe_integrator, platform, properties)
            probe_context.setPositions([[0.0, 0.0, 0.0]] * u.nanometer)
            del probe_context
            del probe_integrator
            logger.info("OpenMM platform %s is available", name)
            return platform, properties
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            if requested != "auto":
                break
            logger.warning("OpenMM platform %s could not initialize; trying next platform: %s", name, exc)
    if requested == "auto":
        raise RuntimeError("No usable OpenMM platform found (tried CUDA -> OpenCL -> CPU): " + "; ".join(failures))
    raise RuntimeError(f"Requested OpenMM platform {requested} could not initialize: " + "; ".join(failures))


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
        "membrane": args.membrane,
        "orientation": args.orientation if args.membrane else None,
        "lipid_type": args.lipid_type if args.membrane else None,
        "membrane_center_z_nm": args.membrane_center_z_nm if args.membrane else None,
        "membrane_padding_nm": args.membrane_padding_nm if args.membrane else None,
        "manual_membrane_normal": args.manual_membrane_normal if args.membrane else None,
        "manual_membrane_center_angstrom": args.manual_membrane_center_angstrom if args.membrane else None,
        "seed": args.seed,
        "minimize_iterations": args.minimize_iterations,
        "protein_forcefield": PROTEIN_FF,
        "lipid_forcefield": "amber14/lipid17.xml" if args.membrane else None,
        "water_forcefield": WATER_FF,
        "software": {
            "openmm": mm.__version__,
            "tmdet": "4.1.2" if args.membrane and args.orientation == "tmdet" else None,
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
    current_platform = args.platform
    if resume:
        current_platform = previous_progress["platform"]
        if args.platform != "auto" and args.platform != current_platform:
            raise ValueError("Requested platform differs from the checkpoint platform")
    platform, platform_properties = get_platform(current_platform)
    logger.info("Using OpenMM platform %s", platform.getName())

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
        ligand_topology = Topology.from_molecules([ligand]).to_openmm()
        ligand_positions = off_to_omm(ligand.conformers[0])
        if args.membrane:
            fixer.positions, ligand_positions = prepare_membrane_orientation(
                args, fixer, ligand_positions, receptor_path, output_dir
            )
            logger.info("Membrane orientation mode: %s", args.orientation)
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
        forcefield_files = [PROTEIN_FF]
        if args.membrane:
            forcefield_files.append("amber14/lipid17.xml")
        forcefield_files.append(WATER_FF)
        forcefield = app.ForceField(*forcefield_files)
        forcefield.registerTemplateGenerator(smirnoff.generator)
        modeller = app.Modeller(fixer.topology, fixer.positions)

        for residue in ligand_topology.residues():
            residue.name = "LIG"
        modeller.add(ligand_topology, ligand_positions)

        logger.info("Adding hydrogens and %s", "membrane plus water" if args.membrane else "solvent")
        python_rng_state = random.getstate()
        random.seed(args.seed)
        try:
            modeller.addHydrogens(forcefield=forcefield, pH=7.0)
            if args.membrane:
                modeller.addMembrane(
                    forcefield,
                    lipidType=args.lipid_type,
                    membraneCenterZ=args.membrane_center_z_nm * u.nanometer,
                    minimumPadding=args.membrane_padding_nm * u.nanometer,
                    ionicStrength=args.ionic_strength_molar * u.molar,
                    platform=platform,
                )
            else:
                modeller.addSolvent(
                    forcefield,
                    padding=args.padding_nm * u.nanometer,
                    ionicStrength=args.ionic_strength_molar * u.molar,
                )
        finally:
            random.setstate(python_rng_state)

        if args.membrane:
            residues = list(modeller.topology.residues())
            lipid_residue_names = {args.lipid_type, args.lipid_type[:3]}
            lipid_residues = [residue for residue in residues if residue.name in lipid_residue_names]
            water_residues = [residue for residue in residues if residue.name in {"HOH", "WAT", "SOL"}]
            if len(lipid_residues) < 10 or not water_residues:
                raise ValueError(
                    f"Membrane builder produced {len(lipid_residues)} {args.lipid_type} residues and "
                    f"{len(water_residues)} waters; expected a membrane and solvated system"
                )
            logger.info(
                "Built membrane system: %d %s lipids, %d water residues, box=%s",
                len(lipid_residues), args.lipid_type, len(water_residues),
                modeller.topology.getPeriodicBoxVectors(),
            )
            geometry = membrane_geometry_summary(
                modeller.topology, modeller.positions, args.lipid_type, args.membrane_center_z_nm
            )
            lipid_leaflets = geometry["lipid_leaflets"]
            if not lipid_leaflets["below_center_heavy_atoms"] or not lipid_leaflets["above_center_heavy_atoms"]:
                raise ValueError("Membrane builder did not produce lipid atoms in both leaflets")
            protein_z = geometry["protein"]["z_range_nm"]
            if protein_z is None or not protein_z[0] < args.membrane_center_z_nm < protein_z[1]:
                raise ValueError("The receptor does not span the requested membrane center plane")
            ligand_distance = geometry.get("minimum_ligand_protein_heavy_distance_nm")
            if ligand_distance is None or ligand_distance > 0.8:
                raise ValueError(
                    "The transformed ligand is not near the receptor "
                    f"(minimum heavy-atom distance {ligand_distance} nm)"
                )
            box_vectors = modeller.topology.getPeriodicBoxVectors()
            geometry["box_vectors_nm"] = [
                [float(value) for value in vector.value_in_unit(u.nanometer)] for vector in box_vectors
            ]
            (output_dir / "membrane_geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
            logger.info(
                "Membrane geometry: protein z=%s nm, lipid z=%s nm, minimum ligand-protein distance=%.3f nm",
                geometry["protein"]["z_range_nm"], geometry["lipid"]["z_range_nm"], ligand_distance,
            )

        with open(output_dir / "system_final.pdb", "w") as pdb_file:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, pdb_file)

        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * u.nanometer,
            constraints=app.HBonds,
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
        if resume:
            phase = previous_progress["phase"]
        if phase in ("npt", "prod"):
            system.addForce(make_barostat(args))
            if phase == "prod":
                for force_index in reversed(range(system.getNumForces())):
                    if isinstance(system.getForce(force_index), mm.CustomExternalForce):
                        system.removeForce(force_index)

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
        simulation.reporters.extend(
            [
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
        if phase == "nvt":
            checkpoint("nvt")
            remaining = max(0, nvt_end - simulation.currentStep)
            logger.info("Running NVT for %d remaining steps", remaining)
            simulation.step(remaining)
            simulation.system.addForce(make_barostat(args))
            simulation.context.reinitialize(preserveState=True)
            phase = "npt"
            checkpoint(phase)

        if phase == "npt":
            remaining = max(0, npt_end - simulation.currentStep)
            logger.info("Running NPT for %d remaining steps", remaining)
            simulation.step(remaining)
            for force_index in reversed(range(simulation.system.getNumForces())):
                if isinstance(simulation.system.getForce(force_index), mm.CustomExternalForce):
                    simulation.system.removeForce(force_index)
            simulation.context.reinitialize(preserveState=True)
            phase = "prod"
            checkpoint(phase)

        if phase == "prod":
            remaining = max(0, total_end - simulation.currentStep)
            logger.info("Running production for %d remaining steps", remaining)
            simulation.step(remaining)
            simulation.saveState(str(output_dir / "final_state.xml"))
            checkpoint("complete")
            logger.info("Simulation completed at step %d", simulation.currentStep)

    except Exception:
        logger.exception("Simulation failed")
        raise


if __name__ == "__main__":
    main()
