"""Single-trajectory gmx_MMPBSA: input preparation, execution and summarisation."""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
from pathlib import Path

from .analysis_metrics import estimate_membrane_planes, resolve_ligand
from .analysis_plots import (
    _plot_mmpbsa_decomposition,
    plot_membrane_planes,
    read_mmpbsa_delta_csv,
)
from .trajectory import (
    _copy_mmpbsa_window,
    _wrap_selection_into_box,
    image_ligand_to_protein,
    write_index_file,
)

logger = logging.getLogger(__name__)
K_B_KCAL_PER_MOL_K = 0.00198720425864083
R_KCAL_PER_MOL_K = K_B_KCAL_PER_MOL_K


def interaction_entropy(electrostatics_kcal, temperature_k):
    """Estimate -TΔS by the interaction-entropy method (Duan et al.).

    −TΔS = kT ln⟨exp(ΔE_el/kT)⟩ − ⟨ΔE_el⟩. The mean electrostatics are already
    part of the MMPBSA enthalpy, so only the fluctuation term is the entropy
    contribution; without the subtraction the electrostatics would be counted
    twice. Uses a shifted log-mean-exp so strong favourable energies do not
    underflow.
    """
    import numpy as np

    values = np.asarray(electrostatics_kcal, dtype=float)
    if values.size == 0:
        raise ValueError("Interaction entropy needs at least one frame")
    kt = K_B_KCAL_PER_MOL_K * float(temperature_k)
    maximum = float(values.max())
    free_energy = maximum + kt * float(np.log(np.mean(np.exp((values - maximum) / kt))))
    # The fluctuation term is non-negative in the thermodynamic limit; small
    # negative values are sampling noise. gmx_MMPBSA clamps the same way.
    return max(0.0, float(free_energy - values.mean()))


def c2_entropy(electrostatics_kcal, temperature_k):
    """Estimate -TΔS from the C2 (variance) entropy approximation.

    −TΔS = σ²(E_el) / (2 R T). Like interaction entropy this uses only the
    fluctuation of the protein–ligand electrostatic energy, but it keeps just
    the second moment, so it is cheaper and less sensitive to rare high-energy
    frames than the log-mean-exp form.
    """
    import numpy as np

    values = np.asarray(electrostatics_kcal, dtype=float)
    if values.size == 0:
        raise ValueError("C2 entropy needs at least one frame")
    temperature = float(temperature_k)
    return float(np.var(values, ddof=1) / (2.0 * R_KCAL_PER_MOL_K * temperature)) if values.size > 1 else 0.0


def read_gmx_entropy(directory, model="gb"):
    """Read entropy estimates that gmx_MMPBSA wrote for this calculation.

    gmx_MMPBSA stores them next to the results as ``*normal_<model>_IE.dat``
    and ``*normal_<model>_c2_entropy.dat``. Returns a mapping method ->
    {minus_tds_kcal_mol, sem_minus_tds_kcal_mol, source}.
    """
    import re
    from pathlib import Path

    directory = Path(directory)
    specs = {
        "ie": ("*normal_{}_IE.dat".format(model),
               r"Full-ensemble Interaction Entropy \(-TΔS\):\s*(-?[\d.]+)"),
        "c2": ("*normal_{}_c2_entropy.dat".format(model),
               r"C2 Entropy \(-TΔS\):\s*(-?[\d.]+)"),
    }
    results = {}
    for method, (pattern, regex) in specs.items():
        matches = sorted(directory.glob(pattern))
        if not matches:
            continue
        text = matches[0].read_text(errors="replace")
        value = re.search(regex, text)
        if value is None:
            continue
        sem = re.search(r"(?:Block diagnostic:.*?SEM\s+|C2 Block SEM:\s*)(-?[\d.]+)", text)
        results[method] = {
            "minus_tds_kcal_mol": float(value.group(1)),
            "sem_minus_tds_kcal_mol": float(sem.group(1)) if sem else None,
            "source": matches[0].name,
        }
    dat = directory / "FINAL_RESULTS_MMPBSA.dat"
    if dat.is_file():
        text = dat.read_text(errors="replace")
        marker = text.find("ENTROPY RESULTS (NMODE APPROXIMATION)")
        if marker >= 0:
            match = re.search(
                r"^ΔTOTAL\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)",
                text[marker:], re.MULTILINE,
            )
            if match:
                results["nmode"] = {
                    "minus_tds_kcal_mol": float(match.group(1)),
                    "sem_minus_tds_kcal_mol": float(match.group(5)),
                    "source": dat.name,
                }
    return results


def summarize_mmpbsa(path, *, temperature_k=300.0, entropy="ie", mmpbsa_dir=None, model="gb"):
    """Summarise per-frame MMPBSA terms, add entropy estimates and errors.

    ``entropy`` selects the entropy model: ``ie`` (interaction entropy),
    ``c2`` (variance entropy), ``both`` (report both, ΔG from IE) or ``none``
    (enthalpy only). The two estimates are alternatives, not additive.
    """
    import numpy as np

    rows = read_mmpbsa_delta_csv(path)
    if not rows:
        raise ValueError(f"No per-frame MMPBSA terms found in {path}")
    frame_key = next(key for key in rows[0] if key.lower().startswith("frame"))
    total_key = "TOTAL" if "TOTAL" in rows[0] else None
    eel_key = "EEL" if "EEL" in rows[0] else None
    if total_key is None or eel_key is None:
        raise ValueError("MMPBSA CSV has no TOTAL/EEL columns in the Delta section")
    frames = np.asarray([row[frame_key] for row in rows], dtype=float)
    enthalpy = np.asarray([row[total_key] for row in rows], dtype=float)
    eel = np.asarray([row[eel_key] for row in rows], dtype=float)

    def sem(values):
        return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")

    def block_sem(values, estimator, blocks=5):
        if len(values) < 2 * blocks:
            return None
        chunks = np.array_split(values, blocks)
        estimates = [estimator(chunk, temperature_k) for chunk in chunks]
        return float(np.std(estimates, ddof=1) / np.sqrt(blocks))

    estimators = {"ie": interaction_entropy, "c2": c2_entropy}
    if entropy == "both":
        methods = ["ie", "c2"]
    elif entropy in ("ie", "c2", "nmode"):
        methods = [entropy]
    elif entropy == "none":
        methods = []
    else:
        raise ValueError(f"Unknown entropy model: {entropy}")

    delta_h = float(enthalpy.mean())
    sem_h = sem(enthalpy)
    running_enthalpy = np.cumsum(enthalpy) / np.arange(1, len(enthalpy) + 1)
    variants = {}
    gmx_estimates = read_gmx_entropy(mmpbsa_dir, model) if mmpbsa_dir is not None else {}
    for method in methods:
        estimator = estimators.get(method)
        if estimator is not None:
            running = np.asarray([estimator(eel[: index + 1], temperature_k) for index in range(len(eel))])
        else:
            running = None
        if method in gmx_estimates:
            minus_tds = gmx_estimates[method]["minus_tds_kcal_mol"]
            sem_tds = gmx_estimates[method]["sem_minus_tds_kcal_mol"]
            source = gmx_estimates[method]["source"]
        elif estimator is not None and running is not None:
            minus_tds = float(running[-1])
            sem_tds = block_sem(eel, estimator)
            source = "python"
        else:
            raise ValueError(
                f"{method} entropy needs gmx_MMPBSA output; rerun the MMPBSA calculation"
            )
        if running is None:
            running = np.full(len(eel), minus_tds)
        delta_g = delta_h + minus_tds
        sem_g = float(np.sqrt(sem_h ** 2 + sem_tds ** 2)) if sem_tds is not None else sem_h
        variants[method] = {
            "name": method,
            "minus_tds_kcal_mol": float(minus_tds),
            "minus_tds_running_kcal_mol": running.tolist(),
            "delta_g_kcal_mol": float(delta_g),
            "delta_g_running_kcal_mol": (running_enthalpy + running).tolist(),
            "sem_minus_tds_kcal_mol": sem_tds,
            "sem_delta_g_kcal_mol": sem_g,
            "source": source,
        }

    primary = methods[0] if methods else None
    running_enthalpy = np.cumsum(enthalpy) / np.arange(1, len(enthalpy) + 1)
    if primary:
        minus_tds = variants[primary]["minus_tds_kcal_mol"]
        delta_g = variants[primary]["delta_g_kcal_mol"]
        sem_tds = variants[primary]["sem_minus_tds_kcal_mol"]
        sem_g = variants[primary]["sem_delta_g_kcal_mol"]
        running_entropy = np.asarray(variants[primary]["minus_tds_running_kcal_mol"])
    else:
        minus_tds = 0.0
        delta_g = delta_h
        sem_tds = 0.0
        sem_g = sem_h
        running_entropy = np.zeros(len(eel))
    return {
        "entropy_method": entropy,
        "temperature_k": float(temperature_k),
        "frame_count": int(len(frames)),
        "frames": frames.tolist(),
        "delta_h_kcal_mol": enthalpy.tolist(),
        "delta_eel_kcal_mol": eel.tolist(),
        "minus_tds_kcal_mol": running_entropy.tolist(),
        "delta_g_running_kcal_mol": (running_enthalpy + running_entropy).tolist(),
        "delta_h_mean_kcal_mol": delta_h,
        "minus_tds_kcal_mol_value": float(minus_tds),
        "delta_g_kcal_mol": float(delta_g),
        "sem_delta_h_kcal_mol": sem_h,
        "sem_minus_tds_kcal_mol": sem_tds,
        "sem_delta_g_kcal_mol": sem_g,
        "entropy_variants": variants,
        "note": "Single-trajectory MM/PBSA enthalpy plus an entropy estimate "
                "(interaction entropy, C2 variance, or both); an approximate "
                "end-state value, not a calibrated absolute free energy.",
    }


def ionic_strength_from_config(config):
    """MMPBSA ionic strength in mol/L, taken from the MD configuration.

    Older runs may not record the value; the historical 0.15 M default is used
    only when the key is absent, while an explicit 0.0 stays zero.
    """
    value = (config or {}).get("ionic_strength_molar")
    return 0.15 if value is None else float(value)


def run_mmpbsa(output_dir, analysis_dir, *, model="gb", interval=1, last_ns=None,
               membrane_thickness_angstrom=None, membrane_dielectric=7.0,
               decomposition=False, ligand=None, temperature_k=300.0, entropy="ie",
               salt_molar=0.150):
    """Run a single-trajectory gmx_MMPBSA estimate, optionally with decomposition."""
    executable = shutil.which("gmx_MMPBSA")
    if executable is None:
        raise RuntimeError("gmx_MMPBSA is not installed or is not available on PATH")
    output_dir, analysis_dir = Path(output_dir), Path(analysis_dir)
    complex_pdb = output_dir / "mmpbsa_complex.pdb"
    complex_top = output_dir / "mmpbsa_complex.top"
    complex_only = complex_pdb.is_file() and complex_top.is_file()
    required = (
        [complex_top, complex_pdb, output_dir / "system_final.pdb",
         analysis_dir / "trajectory_fitted.xtc"]
        if complex_only else
        [output_dir / "system.top", output_dir / "system_final.pdb",
         analysis_dir / "trajectory_fitted.xtc", analysis_dir / "index.ndx"]
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required MMPBSA inputs are missing: " + ", ".join(missing))
    mmpbsa_dir = analysis_dir / "mmpbsa"
    mmpbsa_dir.mkdir(exist_ok=True)
    input_path = mmpbsa_dir / "mmpbsa.in"
    if model not in {"gb", "pb", "pb_mem"}:
        raise ValueError(f"Unsupported MMPBSA model: {model}")
    if last_ns is not None and (not math.isfinite(float(last_ns)) or float(last_ns) <= 0):
        raise ValueError("The MMPBSA last-ns window must be a positive finite number")
    # The MMPBSA ionic strength follows the MD configuration, so the implicit
    # solvent screens charges at the same salt concentration as the simulation.
    salt = float(salt_molar)
    if not math.isfinite(salt) or salt < 0:
        raise ValueError("The MMPBSA ionic strength must be a non-negative finite number")
    model_namelist = (
        f"&gb\n igb=5, saltcon={salt:g},\n/\n" if model == "gb"
        else f"&pb\n istrng={salt:g},\n/\n"
    )
    if model == "pb_mem":
        if membrane_thickness_angstrom is None:
            import mdtraj as md

            structure = md.load_pdb(str(output_dir / "system_final.pdb"))
            state = json.loads((output_dir / "run_state.json").read_text())
            configuration = state["configuration"]
            estimate = estimate_membrane_planes(
                structure.topology,
                structure.xyz[0],
                center_z_nm=configuration.get("membrane_center_z_nm") or 0.0,
                lipid_type=configuration.get("lipid_type", "POPC"),
            )
            membrane_thickness_angstrom = estimate["thickness_angstrom"]
            plot_membrane_planes(analysis_dir, estimate, filename="mmpbsa_membrane_planes.png")
            (mmpbsa_dir / "membrane_planes.json").write_text(
                json.dumps({key: value for key, value in estimate.items()
                            if key not in ("atom_indices", "atom_z_nm")}, indent=2, sort_keys=True) + "\n"
            )
            logger.info(
                "Auto membrane thickness %.2f A from %d/%d phosphorus atoms "
                "(%d outliers rejected)",
                membrane_thickness_angstrom,
                estimate["lower_atom_count"],
                estimate["upper_atom_count"],
                estimate["outlier_count"],
            )
        if (membrane_thickness_angstrom is None
                or not math.isfinite(float(membrane_thickness_angstrom))
                or float(membrane_thickness_angstrom) <= 0):
            raise ValueError("The pb_mem model requires a positive membrane thickness in angstroms")
        if not math.isfinite(float(membrane_dielectric)) or float(membrane_dielectric) <= 0:
            raise ValueError("The pb_mem model requires a positive finite membrane dielectric")
        state = json.loads((output_dir / "run_state.json").read_text())
        if not state["configuration"].get("membrane"):
            raise ValueError("The pb_mem model requires an OpenMM membrane run")
        center_z_angstrom = 10.0 * float(state["configuration"].get("membrane_center_z_nm") or 0.0)
        model_namelist = (
            "&pb\n memopt=1, emem=" + f"{membrane_dielectric:g}" + ", indi=4.0, "
            + f"mctrdz={center_z_angstrom:g}, mthick={membrane_thickness_angstrom:g}, "
            + f"poretype=1, radiopt=0, istrng={salt:g}, fillratio=1.25, inp=2, "
            # Amber membrane PB needs the linear non-focusing periodic solver.
            + "sasopt=0, solvopt=2, ipb=1, bcopt=10, nfocus=1, linit=1000, "
            + "eneopt=1, cutfd=7.0, cutnb=99.0, maxarcdot=15000, npbverb=1,\n/\n"
        )
    decomposition_namelist = (
        '&decomp\n idecomp=2, dec_verbose=1, print_res="within 4",\n/\n'
        if decomposition else ""
    )
    # igb=5 conventionally pairs with mbondi2 radii (PBRadii=3); gmx_MMPBSA
    # otherwise reuses mbondi3 (4), which is calibrated for igb=8. PB runs take
    # radii from the topology (radiopt=0), where mbondi3 is the usual choice.
    # Let gmx_MMPBSA itself compute the entropy estimates; they are parsed
    # from its *_IE.dat / *_c2_entropy.dat output below.
    entropy_flags = {
        "ie": "interaction_entropy=1,",
        "c2": "c2_entropy=1,",
        "both": "interaction_entropy=1, c2_entropy=1,",
    }.get(entropy, "")
    nmode_namelist = (
        "&nmode\n nmstartframe=1, nmendframe=1000000, nminterval=1, nmode_igb=1, "
        f"nmode_istrng={salt:g}, dielc=1.0, drms=0.001, maxcyc=10000,\n/\n"
        if entropy == "nmode" else ""
    )
    pb_radii = 3 if model == "gb" else 4
    input_path.write_text(
        "Single-trajectory protein-ligand binding free-energy estimate.\n"
        + (f"Trajectory window: final {float(last_ns):g} ns of production; "
           "frames outside this window were excluded before applying interval.\n"
           if last_ns is not None else "Trajectory window: all available production frames.\n")
        + f"&general\n startframe=1, interval={interval}, PBRadii={pb_radii}, verbose=1, "
        + (entropy_flags + "\n/\n" if entropy_flags else "\n/\n")
        + model_namelist + nmode_namelist + decomposition_namelist
    )

    import mdtraj as md
    reference_path = mmpbsa_dir / "reference.pdb"
    source_trajectory = analysis_dir / "trajectory_fitted.xtc"
    full_topology = md.load_topology(str(output_dir / "system_final.pdb"))
    selection = resolve_ligand(full_topology, ligand)
    protein_indices = list(full_topology.select("protein"))
    full_complex_indices = protein_indices + list(selection["atoms"])
    if not full_complex_indices:
        raise ValueError("Could not select protein and ligand atoms from the simulation topology")
    if complex_only:
        # The requested GB/PB model treats water and lipids implicitly. Keep
        # the MMPBSA topology small and stream its atoms from the full MD XTC.
        complex_topology = md.load_topology(str(complex_pdb))
        complex_ligands = [residue for residue in complex_topology.residues
                           if residue.name == "LIG"]
        if len(complex_ligands) != 1:
            raise ValueError(
                f"{complex_pdb} contains {len(complex_ligands)} ligand residues. "
                "Regenerate it with --mmpbsa-ligand so one ligand is exported."
            )
        if complex_topology.n_atoms != len(full_complex_indices):
            raise ValueError(
                "MMPBSA complex topology does not match the selected protein-ligand "
                "atoms in the simulation topology"
            )
        index_path = mmpbsa_dir / "index.ndx"
        write_index_file(
            index_path,
            list(complex_topology.select("protein")),
            [atom.index for atom in complex_ligands[0].atoms],
        )
        complex_trajectory_path = mmpbsa_dir / "complex.xtc"
        if last_ns is not None:
            _copy_mmpbsa_window(
                source_trajectory, output_dir / "system_final.pdb", complex_trajectory_path,
                last_ns=last_ns, atom_indices=full_complex_indices,
                protein_indices=range(len(protein_indices)),
            )
        else:
            frames_written = 0
            with md.formats.XTCTrajectoryFile(str(complex_trajectory_path), mode="w") as writer:
                for chunk in md.iterload(str(source_trajectory), top=str(output_dir / "system_final.pdb"), chunk=1000):
                    selected = chunk.atom_slice(full_complex_indices)
                    selected = image_ligand_to_protein(selected, range(len(protein_indices)))
                    _wrap_selection_into_box(selected)
                    writer.write(selected.xyz, time=selected.time, box=selected.unitcell_vectors)
                    frames_written += selected.n_frames
            if frames_written == 0:
                raise ValueError("The trajectory contains no frames for MMPBSA")
        reference_path.write_text(complex_pdb.read_text())
        complex_structure = complex_pdb
        complex_topology_path = complex_top
        mmpbsa_index = index_path
        mmpbsa_trajectory = complex_trajectory_path
    else:
        full_reference = md.load_topology(str(output_dir / "system_final.pdb"))
        reference_atoms = full_reference.select("protein or resname LIG")
        if len(reference_atoms) == 0:
            raise ValueError("Could not select protein and LIG atoms from the MMPBSA reference structure")
        md.load_pdb(str(output_dir / "system_final.pdb")).atom_slice(reference_atoms).save_pdb(str(reference_path))
        complex_structure = output_dir / "system_final.pdb"
        complex_topology_path = output_dir / "system.top"
        # The full GROMACS topology can contain several ligands; the index
        # groups decide which one is the ligand in this calculation.
        mmpbsa_index = mmpbsa_dir / "index.ndx"
        write_index_file(
            mmpbsa_index,
            list(full_reference.select("protein")),
            list(selection["atoms"]),
        )
        if last_ns is not None:
            mmpbsa_trajectory = mmpbsa_dir / "production_window.xtc"
            _copy_mmpbsa_window(
                source_trajectory, output_dir / "system_final.pdb", mmpbsa_trajectory,
                last_ns=last_ns,
            )
        else:
            mmpbsa_trajectory = source_trajectory
    command = [
        executable, "-O", "-i", str(input_path), "-cs", str(complex_structure),
        "-cp", str(complex_topology_path), "-cr", str(reference_path),
        "-ci", str(mmpbsa_index), "-cg", "Protein", "LIG",
        "-ct", str(mmpbsa_trajectory), "-o", str(mmpbsa_dir / "FINAL_RESULTS_MMPBSA.dat"),
        "-eo", str(mmpbsa_dir / "FINAL_RESULTS_MMPBSA.csv"), "-nogui",
    ]
    if decomposition:
        command.extend(["-do", str(mmpbsa_dir / "FINAL_DECOMP_MMPBSA.dat"),
                        "-deo", str(mmpbsa_dir / "FINAL_DECOMP_MMPBSA.csv")])
    (mmpbsa_dir / "command.txt").write_text(" ".join(command) + "\n")
    logger.info("Running gmx_MMPBSA (%s); this may take some time", model)
    with (mmpbsa_dir / "gmx_MMPBSA.log").open("w") as log:
        completed = subprocess.run(command, cwd=mmpbsa_dir, stdout=log, stderr=subprocess.STDOUT,
                                   text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"gmx_MMPBSA failed with exit code {completed.returncode}; see {mmpbsa_dir / 'gmx_MMPBSA.log'}")
    try:
        summary = summarize_mmpbsa(
            mmpbsa_dir / "FINAL_RESULTS_MMPBSA.csv",
            temperature_k=temperature_k,
            entropy=entropy,
            mmpbsa_dir=mmpbsa_dir,
            model=model,
        )
    except Exception as error:
        logger.warning("MMPBSA finished but entropy summarisation failed: %s", error)
    else:
        summary["ligand"] = selection.get("label")
        summary["ligand_residue"] = selection.get("residue_label")
        (mmpbsa_dir / "entropy.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        _plot_mmpbsa_decomposition(
            analysis_dir, summary, mmpbsa_dir / "FINAL_RESULTS_MMPBSA.csv"
        )
        logger.info(
            "MMPBSA: dH = %.2f, -TdS = %.2f, dG = %.2f kcal/mol over %d frames (%s)",
            summary["delta_h_mean_kcal_mol"], summary["minus_tds_kcal_mol_value"],
            summary["delta_g_kcal_mol"], summary["frame_count"], entropy,
        )
    return mmpbsa_dir
