import argparse
import hashlib
import json
import logging
import math
import os
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
from .stability import (
    StabilityReporter,
    diagnose_failure,
    find_severe_overlaps,
    write_diagnostics,
)
from .topology_export import (
    enable_batched_openmm_topology_conversion,
    export_mmpbsa_complex,
    keep_interchange_molecules_separate,
)

TEMPERATURE_K = 300.0
PRESSURE_BAR = 1.0
FRICTION_PER_PS = 1.0
TIMESTEP_FS = 2.0
IONIC_STRENGTH_MOLAR = 0.15
PADDING_NM = 1.0
EQUILIBRATION_STAGE_PS = 250.0
INITIAL_RESTRAINT_K = 1000.0
RESTRAINT_PARAMETERS = {
    "protein_bb": "k_protein_bb",
    "protein_sc": "k_protein_sc",
    "ligand": "k_ligand",
    "lipid_pz": "k_lipid_pz",
}
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
        "--receptor", default="tests/fixtures/aqp3_receptor.pdb", help="Receptor PDB path"
    )
    parser.add_argument(
        "--ligand", action="append", dest="ligands",
        help="Ligand SDF path; repeat to include multiple ligands in one simulation",
    )
    parser.add_argument("--output", default="output", help="Output directory")
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
    parser.add_argument(
        "--skip-gromacs-export", action="store_true",
        help="Skip OpenFF Interchange conversion to GROMACS files (useful for large membrane systems)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from the checkpoint in run_state.json")
    parser.add_argument(
        "--mmpbsa", action="store_true",
        help="Run an optional single-trajectory gmx_MMPBSA analysis after MD (GB by default)",
    )
    parser.add_argument(
        "--mmpbsa-model", choices=("gb", "pb", "pb_mem"), default="gb",
        help="Implicit-solvent model for --mmpbsa; pb_mem requires a membrane thickness",
    )
    parser.add_argument("--mmpbsa-interval", type=positive_int, default=1,
                        help="Analyze every Nth saved frame in gmx_MMPBSA")
    parser.add_argument("--mmpbsa-last-ns", type=positive_float,
                        help="Restrict MMPBSA to the final N ns of production")
    parser.add_argument("--analysis-stride", type=positive_int, default=10,
                        help="Load every Nth XTC frame for plots and the prepared MMPBSA trajectory")
    parser.add_argument("--mmpbsa-membrane-thickness-angstrom", type=positive_float,
                        help="Required membrane slab thickness for --mmpbsa-model pb_mem")
    parser.add_argument("--mmpbsa-membrane-dielectric", type=positive_float, default=7.0,
                        help="Membrane dielectric used by the pb_mem model (default: 7)")
    parser.add_argument(
        "--mmpbsa-decomposition", action="store_true",
        help="Also calculate per-residue MMPBSA energy decomposition (more expensive)",
    )
    parser.add_argument(
        "--distance-pair", action="append", nargs=2, metavar=("CHAIN:RESID", "CHAIN:RESID"),
        help="Plot Cα distance between two residues; repeat to add pairs, e.g. --distance-pair A:550 A:600",
    )
    args = parser.parse_args()
    args.ligands = args.ligands or ["tests/fixtures/aqp3_ligand.sdf"]
    if args.mmpbsa and args.skip_gromacs_export:
        parser.error("--mmpbsa requires its protein-ligand-only GROMACS topology; remove --skip-gromacs-export")
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
    if args.mmpbsa and args.mmpbsa_model == "pb_mem" and args.mmpbsa_membrane_thickness_angstrom is None:
        parser.error("--mmpbsa-model pb_mem requires --mmpbsa-membrane-thickness-angstrom")
    if args.mmpbsa and args.mmpbsa_model == "pb_mem" and not args.membrane:
        parser.error("--mmpbsa-model pb_mem is only valid with --membrane")
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
    command = [
        executable,
        "-e", "/app/tmdet.env",
        "-pi", str(receptor_path),
        "-po", str(oriented_path),
        "-x", str(annotation_path),
    ]
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
        [transform_positions(positions, rotation, source_center, target_center)
         for positions in ligand_positions],
    )


def add_positional_restraints(system, positions, atom_indices, parameter_name, axes=("x", "y", "z")):
    expression = " + ".join(f"({axis}-{axis}0)^2" for axis in axes)
    force = mm.CustomExternalForce(f"0.5*{parameter_name}*({expression})")
    force.addGlobalParameter(parameter_name, INITIAL_RESTRAINT_K)
    for parameter in ("x0", "y0", "z0"):
        force.addPerParticleParameter(parameter)
    for index in atom_indices:
        force.addParticle(index, positions[index].value_in_unit(u.nanometers))
    system.addForce(force)
    return force


def staged_equilibration_schedule(membrane):
    if membrane:
        return [
            ("nvt1", "nvt", {"protein_bb": 1000, "protein_sc": 1000, "ligand": 1000, "lipid_pz": 1000}),
            ("nvt2", "nvt", {"protein_bb": 500, "protein_sc": 500, "ligand": 500, "lipid_pz": 500}),
            ("npt1", "npt", {"protein_bb": 500, "protein_sc": 250, "ligand": 500, "lipid_pz": 500}),
            ("npt2", "npt", {"protein_bb": 250, "protein_sc": 100, "ligand": 250, "lipid_pz": 250}),
            ("npt3", "npt", {"protein_bb": 100, "protein_sc": 50, "ligand": 100, "lipid_pz": 100}),
            ("npt4", "npt", {"protein_bb": 0, "protein_sc": 0, "ligand": 0, "lipid_pz": 0}),
            ("prod", "npt", {"protein_bb": 0, "protein_sc": 0, "ligand": 0, "lipid_pz": 0}),
        ]
    return [
        ("nvt1", "nvt", {"protein_bb": 1000, "protein_sc": 1000, "ligand": 1000}),
        ("nvt2", "nvt", {"protein_bb": 500, "protein_sc": 500, "ligand": 500}),
        ("npt1", "npt", {"protein_bb": 500, "protein_sc": 250, "ligand": 500}),
        ("npt2", "npt", {"protein_bb": 250, "protein_sc": 100, "ligand": 250}),
        ("npt3", "npt", {"protein_bb": 100, "protein_sc": 50, "ligand": 100}),
        ("npt4", "npt", {"protein_bb": 0, "protein_sc": 0, "ligand": 0}),
        ("prod", "npt", {"protein_bb": 0, "protein_sc": 0, "ligand": 0}),
    ]


def equilibration_stage_steps(timestep_fs):
    return max(1, round(EQUILIBRATION_STAGE_PS * 1000.0 / timestep_fs))


def set_restraint_strengths(context, strengths):
    for group, value in strengths.items():
        context.setParameter(RESTRAINT_PARAMETERS[group], float(value))


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
    if requested in {"auto", "CUDA"} and Path("/dev/nvidiactl").exists():
        cuda_plugin_dir = os.environ.get("OPENMM_CUDA_PLUGIN_DIR")
        if cuda_plugin_dir:
            try:
                plugin_errors = mm.Platform.loadPluginsFromDirectory(cuda_plugin_dir)
                if plugin_errors:
                    logger.warning("Some OpenMM CUDA plugins failed to load: %s", plugin_errors)
            except Exception as exc:
                logger.warning("OpenMM CUDA plugins could not be loaded: %s", exc)
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


def run_configuration(args, receptor, ligands):
    package_dir = Path(__file__).parent
    return {
        "inputs": {"receptor_sha256": sha256_file(receptor),
                   "ligand_sha256": [sha256_file(ligand) for ligand in ligands]},
        "source_sha256": {
            path.name: sha256_file(path)
            for path in sorted(package_dir.glob("*.py"))
        },
        "nvt_steps": 2 * equilibration_stage_steps(args.timestep_fs),
        "npt_steps": 4 * equilibration_stage_steps(args.timestep_fs),
        "prod_steps": args.prod_steps,
        "equilibration_protocol": "staged-restraints-fixed-duration-v2",
        "equilibration_schedule": [
            {
                "phase": name,
                "ensemble": ensemble,
                "duration_ps": EQUILIBRATION_STAGE_PS if name not in {"prod"} else None,
                "restraints_kj_mol_nm2": strengths,
            }
            for name, ensemble, strengths in staged_equilibration_schedule(args.membrane)
        ],
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
        "skip_gromacs_export": args.skip_gromacs_export,
        "distance_pairs": args.distance_pair or [],
        "mmpbsa_decomposition": args.mmpbsa_decomposition,
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
    ligand_paths = [Path(path) for path in args.ligands]
    output_dir = Path(args.output)

    for label, path in (("receptor", receptor_path), *[(f"ligand {index}", path) for index, path in enumerate(ligand_paths, 1)]):
        if not path.is_file():
            raise SystemExit(f"{label} file does not exist or is not a file: {path}")
    if not math.isfinite(args.ionic_strength_molar) or args.ionic_strength_molar < 0:
        raise SystemExit("--ionic-strength-molar must be a finite number that is zero or greater")
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = args.resume
    progress_path = output_dir / PROGRESS_FILE
    config = run_configuration(args, receptor_path, ligand_paths)
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
        valid_phases = {"nvt1", "nvt2", "npt1", "npt2", "npt3", "npt4", "prod", "complete"}
        if previous_progress["phase"] not in valid_phases:
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

    diagnostic_reporter = None
    current_phase = "preparation"
    initial_overlaps = []
    try:
        ligands = []
        for ligand_path in ligand_paths:
            logger.info("Loading ligand from %s", ligand_path)
            loaded = Molecule.from_file(str(ligand_path))
            if isinstance(loaded, list):
                if len(loaded) != 1:
                    raise ValueError(f"Each ligand SDF must contain exactly one molecule; found {len(loaded)} in {ligand_path}")
                loaded = loaded[0]
            if loaded.n_atoms == 0 or not loaded.conformers:
                raise ValueError(f"Ligand SDF must contain atoms and at least one 3D conformer: {ligand_path}")
            ligands.append(loaded)

        logger.info("Preparing receptor %s with PDBFixer", receptor_path)
        fixer = PDBFixer(filename=str(receptor_path))
        ligand_topologies = [Topology.from_molecules([ligand]).to_openmm() for ligand in ligands]
        ligand_positions = [off_to_omm(ligand.conformers[0]) for ligand in ligands]
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

        smirnoff = SMIRNOFFTemplateGenerator(molecules=ligands)
        forcefield_files = [PROTEIN_FF]
        if args.membrane:
            forcefield_files.append("amber14/lipid17.xml")
        forcefield_files.append(WATER_FF)
        forcefield = app.ForceField(*forcefield_files)
        forcefield.registerTemplateGenerator(smirnoff.generator)
        modeller = app.Modeller(fixer.topology, fixer.positions)

        for ligand_topology, positions in zip(ligand_topologies, ligand_positions):
            for residue in ligand_topology.residues():
                residue.name = "LIG"
            modeller.add(ligand_topology, positions)

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
        initial_overlaps = find_severe_overlaps(modeller.topology, modeller.positions)
        if initial_overlaps:
            logger.warning(
                "Found %d possible severe nonbonded close contacts in prepared coordinates; "
                "the first pairs are included in failure_diagnostics.json: %s",
                len(initial_overlaps), json.dumps(initial_overlaps),
            )
        for index, residue in enumerate(modeller.topology.residues()):
            residue.id = str(index + 1)

        if args.mmpbsa:
            logger.info("Exporting protein-ligand-only topology for MMPBSA; membrane and solvent stay implicit")
            export_mmpbsa_complex(forcefield, modeller, output_dir)
        elif args.skip_gromacs_export:
            logger.info("Skipping GROMACS export as requested")
        else:
            # Keep constrained H-X bonds in OpenMM dynamics, but include them
            # in the exported topology so ParmEd can construct Amber radii.
            enable_batched_openmm_topology_conversion()
            topology_system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=app.PME,
                nonbondedCutoff=1.0 * u.nanometer,
                constraints=None,
            )
            interchange = Interchange.from_openmm(
                topology=modeller.topology, system=topology_system, positions=modeller.positions
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
        protein_atoms = [
            atom for atom in modeller.topology.atoms()
            if atom.residue.name in PROTEIN_RESIDUES
            and atom.element is not None and atom.element.symbol != "H"
        ]
        backbone_names = {"N", "CA", "C", "O", "OXT"}
        protein_bb_indices = [atom.index for atom in protein_atoms if atom.name.upper() in backbone_names]
        protein_sc_indices = [atom.index for atom in protein_atoms if atom.name.upper() not in backbone_names]
        ligand_heavy_indices = [
            atom.index for atom in modeller.topology.atoms()
            if atom.residue.name == "LIG" and atom.element is not None and atom.element.symbol != "H"
        ]
        if not protein_bb_indices:
            raise ValueError("No protein backbone heavy atoms found; cannot apply protein restraints")
        if not ligand_heavy_indices:
            raise ValueError("No ligand heavy atoms found; cannot apply ligand restraints")
        lipid_p_indices = []
        if args.membrane:
            lipid_residue_names = {args.lipid_type, args.lipid_type[:3]}
            lipid_p_indices = [
                atom.index for atom in modeller.topology.atoms()
                if atom.residue.name in lipid_residue_names and atom.name.upper() == "P"
            ]
            if not lipid_p_indices:
                raise ValueError("No lipid phosphorus atoms found; cannot apply membrane equilibration restraints")
        logger.info(
            "Applying staged restraints to %d protein backbone heavy atoms, %d side-chain heavy atoms, "
            "%d ligand heavy atoms%s",
            len(protein_bb_indices), len(protein_sc_indices), len(ligand_heavy_indices),
            f", and {len(lipid_p_indices)} lipid phosphorus atoms along Z" if args.membrane else "",
        )
        add_positional_restraints(
            system, modeller.positions, protein_bb_indices, RESTRAINT_PARAMETERS["protein_bb"]
        )
        add_positional_restraints(
            system, modeller.positions, protein_sc_indices, RESTRAINT_PARAMETERS["protein_sc"]
        )
        add_positional_restraints(
            system, modeller.positions, ligand_heavy_indices, RESTRAINT_PARAMETERS["ligand"]
        )
        if args.membrane:
            add_positional_restraints(
                system, modeller.positions, lipid_p_indices, RESTRAINT_PARAMETERS["lipid_pz"], axes=("z",)
            )

        raw_schedule = staged_equilibration_schedule(args.membrane)
        stage_steps = equilibration_stage_steps(args.timestep_fs)
        schedule_steps = [stage_steps] * (len(raw_schedule) - 1) + [args.prod_steps]
        stages = []
        cumulative_steps = 0
        for (stage_name, ensemble, strengths), step_count in zip(raw_schedule, schedule_steps):
            cumulative_steps += step_count
            stages.append({
                "name": stage_name,
                "ensemble": ensemble,
                "restraints": strengths,
                "end_step": cumulative_steps,
            })
        stage_names = [stage["name"] for stage in stages]
        phase = "nvt1"
        if not resume:
            checkpoint_path = output_dir / f"checkpoint-{phase}.chk"
        else:
            checkpoint_path = output_dir / previous_progress["checkpoint"]
        if resume:
            phase = previous_progress["phase"]
        start_stage_index = stage_names.index(phase)
        if stages[start_stage_index]["ensemble"] == "npt":
            system.addForce(make_barostat(args))

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
                    kineticEnergy=True, totalEnergy=True, volume=True, density=True,
                    temperature=True, speed=True, separator="\t"
                ),
                app.StateDataReporter(
                    str(state_data_file), args.report_interval, step=True, potentialEnergy=True,
                    kineticEnergy=True, totalEnergy=True, volume=True, density=True,
                    temperature=True, speed=True,
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

        if not resume:
            checkpoint("nvt1")
        current_phase = phase
        for stage_index in range(start_stage_index, len(stages)):
            stage = stages[stage_index]
            current_phase = stage["name"]
            remaining = max(0, stage["end_step"] - simulation.currentStep)
            logger.info(
                "Running %s for %d remaining steps with restraints (kJ/mol/nm^2): %s",
                current_phase.upper(), remaining, json.dumps(stage["restraints"], sort_keys=True),
            )
            if remaining:
                simulation.step(remaining)
            diagnostic_reporter.previous_energy_kj_mol = None

            if stage["name"] == "prod":
                simulation.saveState(str(output_dir / "final_state.xml"))
                checkpoint("complete")
                logger.info("Simulation completed at step %d", simulation.currentStep)
                from .analysis import analyze_run

                analyze_run(
                    output_dir,
                    {
                        **config,
                        "prod_steps": args.prod_steps,
                        "timestep_fs": args.timestep_fs,
                        "temperature_k": args.temperature_k,
                        "pressure_bar": args.pressure_bar,
                    },
                    perform_mmpbsa=args.mmpbsa,
                    mmpbsa_model=args.mmpbsa_model,
                    mmpbsa_interval=args.mmpbsa_interval,
                    mmpbsa_last_ns=args.mmpbsa_last_ns,
                    mmpbsa_membrane_thickness_angstrom=args.mmpbsa_membrane_thickness_angstrom,
                    mmpbsa_membrane_dielectric=args.mmpbsa_membrane_dielectric,
                    analysis_stride=args.analysis_stride,
                    mmpbsa_decomposition=args.mmpbsa_decomposition,
                    distance_pairs=args.distance_pair,
                )
                break

            if stage_index + 1 < len(stages):
                next_stage = stages[stage_index + 1]
                if stage["ensemble"] == "nvt" and next_stage["ensemble"] == "npt":
                    simulation.system.addForce(make_barostat(args))
                    simulation.context.reinitialize(preserveState=True)
                set_restraint_strengths(simulation.context, next_stage["restraints"])
                diagnostic_reporter.previous_energy_kj_mol = None
                checkpoint(next_stage["name"])

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
