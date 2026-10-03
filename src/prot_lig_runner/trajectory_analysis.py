"""Trajectory diagnostics and optional gmx_MMPBSA post-processing."""

from __future__ import annotations

import csv
import html
import json
import logging
import math
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

PHASE_COLORS = {"NVT": "#dbeafe", "NPT": "#dcfce7", "Production": "#fef3c7"}


def read_state_data(path):
    """Read OpenMM StateDataReporter CSV output into normalized records."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        rows = []
        for row in reader:
            try:
                rows.append({key: float(value) if value not in (None, "") else math.nan
                             for key, value in row.items()})
            except (TypeError, ValueError):
                continue
    return rows


def phase_for_step(step, nvt_steps, npt_steps):
    if step <= nvt_steps:
        return "NVT"
    if step <= nvt_steps + npt_steps:
        return "NPT"
    return "Production"


def write_index_file(path, protein_indices, ligand_indices):
    """Write 1-based GROMACS index groups in trajectory/topology atom order."""
    if not protein_indices or not ligand_indices:
        raise ValueError("Protein and ligand atom groups must both be non-empty")
    lines = []
    for name, indices in (("Protein", protein_indices), ("LIG", ligand_indices)):
        lines.append(f"[ {name} ]")
        values = [index + 1 for index in indices]
        lines.extend(" ".join(str(value) for value in values[start:start + 15])
                     for start in range(0, len(values), 15))
        lines.append("")
    Path(path).write_text("\n".join(lines))


def _column(rows, fragment):
    if not rows:
        return []
    key = next((key for key in rows[0] if fragment.lower() in key.lower()), None)
    return [row.get(key, math.nan) for row in rows] if key else []


def _make_plots(output_dir, state_rows, metrics, trajectory_data, config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle("Protein–ligand MD diagnostics", fontsize=19, fontweight="bold")

    if state_rows:
        step = np.asarray(_column(state_rows, "Step"), dtype=float)
        time_ns = step * float(config["timestep_fs"]) / 1_000_000.0
        for ax in axes.flat[:3]:
            for start, end, phase in (
                (0, config["nvt_steps"], "NVT"),
                (config["nvt_steps"], config["nvt_steps"] + config["npt_steps"], "NPT"),
                (config["nvt_steps"] + config["npt_steps"],
                 config["nvt_steps"] + config["npt_steps"] + config["prod_steps"], "Production"),
            ):
                ax.axvspan(start * config["timestep_fs"] / 1_000_000,
                           end * config["timestep_fs"] / 1_000_000,
                           color=PHASE_COLORS[phase], alpha=0.30, linewidth=0)

        temperature = _column(state_rows, "Temperature")
        if temperature:
            axes[0, 0].plot(time_ns, temperature, color="#2563eb", linewidth=1.2)
        axes[0, 0].axhline(config["temperature_k"], color="#dc2626", linestyle="--",
                           label=f"Target {config['temperature_k']:g} K")
        axes[0, 0].set(title="Thermostat · temperature", xlabel="Time (ns)", ylabel="Temperature (K)")
        target_handles, target_labels = axes[0, 0].get_legend_handles_labels()
        phase_handles = [Patch(facecolor=PHASE_COLORS[name], label=name, alpha=0.65)
                         for name in ("NVT", "NPT", "Production")]
        axes[0, 0].legend(target_handles + phase_handles,
                          target_labels + ["NVT", "NPT", "Production"],
                          frameon=False, ncol=2, fontsize=8)

        density = _column(state_rows, "Density")
        volume = _column(state_rows, "Box Volume")
        npt_mask = (step > config["nvt_steps"]) & (step <= config["nvt_steps"] + config["npt_steps"])
        if density:
            axes[0, 1].plot(time_ns[npt_mask], np.asarray(density)[npt_mask],
                            color="#059669", linewidth=1.2, label="Density (g/mL)")
        ax_volume = axes[0, 1].twinx()
        if volume:
            ax_volume.plot(time_ns[npt_mask], np.asarray(volume)[npt_mask],
                           color="#7c3aed", linewidth=1.0, alpha=0.8, label="Volume (nm³)")
        axes[0, 1].set(title=f"Barostat · NPT (target {config['pressure_bar']:g} bar)",
                       xlabel="Time (ns)", ylabel="Density (g/mL)")
        ax_volume.set_ylabel("Volume (nm³)")
        handles, labels = axes[0, 1].get_legend_handles_labels()
        h2, l2 = ax_volume.get_legend_handles_labels()
        if handles or h2:
            axes[0, 1].legend(handles + h2, labels + l2, frameon=False, loc="best")

        for fragment, color in (("Potential Energy", "#2563eb"),
                                ("Kinetic Energy", "#ea580c"),
                                ("Total Energy", "#475569")):
            values = _column(state_rows, fragment)
            if values:
                axes[1, 0].plot(time_ns, values, label=fragment.replace(" Energy", ""),
                                color=color, linewidth=0.9, alpha=0.85)
        axes[1, 0].set(title="Energy", xlabel="Time (ns)", ylabel="Energy (kJ/mol)")
        axes[1, 0].legend(frameon=False)
    else:
        for ax in axes.flat[:3]:
            ax.text(0.5, 0.5, "No state_data.csv samples", ha="center", va="center",
                    transform=ax.transAxes)

    if trajectory_data:
        time_ns = trajectory_data["time_ns"]
        axes[1, 1].plot(time_ns, trajectory_data["protein_rmsd_nm"],
                        label="Protein backbone", color="#2563eb", linewidth=1.2)
        axes[1, 1].plot(time_ns, trajectory_data["ligand_rmsd_nm"],
                        label="Ligand (protein aligned)", color="#dc2626", linewidth=1.2)
        axes[1, 1].set(title="RMSD", xlabel="Time (ns)", ylabel="RMSD (nm)")
        axes[1, 1].legend(frameon=False)
        axes[2, 0].plot(time_ns, trajectory_data["protein_rg_nm"],
                        label="Protein", color="#0f766e", linewidth=1.2)
        axes[2, 0].plot(time_ns, trajectory_data["ligand_rg_nm"],
                        label="Ligand", color="#9333ea", linewidth=1.2)
        axes[2, 0].set(title="Radius of gyration", xlabel="Time (ns)", ylabel="Rg (nm)")
        axes[2, 0].legend(frameon=False)
        axes[2, 1].plot(time_ns, trajectory_data["ligand_contacts"],
                        color="#c2410c", linewidth=1.2)
        axes[2, 1].set(title="Protein–ligand contacts (< 0.45 nm)", xlabel="Time (ns)",
                       ylabel="Residue contacts")
    else:
        for ax in axes.flat[3:]:
            ax.text(0.5, 0.5, "Trajectory metrics unavailable\n(check trajectory.xtc and system_final.pdb)",
                    ha="center", va="center", transform=ax.transAxes)

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    fig.savefig(output_dir / "diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    if trajectory_data and trajectory_data.get("rmsf_residue"):
        fig, ax = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
        ax.fill_between(trajectory_data["rmsf_residue"], 0,
                        trajectory_data["protein_rmsf_nm"], color="#60a5fa", alpha=0.35)
        ax.plot(trajectory_data["rmsf_residue"], trajectory_data["protein_rmsf_nm"],
                color="#1d4ed8", linewidth=1.2)
        ax.set(title="Protein Cα RMSF by residue", xlabel="Residue index", ylabel="RMSF (nm)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(output_dir / "protein_rmsf.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def _trajectory_metrics(trajectory_path, topology_path, analysis_dir, config, analysis_stride=10):
    import mdtraj as md
    import numpy as np

    trajectory = md.load(str(trajectory_path), top=str(topology_path), stride=analysis_stride)
    production_start_ps = (config["nvt_steps"] + config["npt_steps"]) * config["timestep_fs"] / 1000.0
    production_frames = trajectory.time >= production_start_ps
    trajectory = trajectory[production_frames]
    if trajectory.n_frames == 0:
        raise ValueError("No production frames were found in trajectory.xtc; increase production length or adjust --xtc-interval")

    # Reconstruct each bonded molecule before fitting; then move the ligand to
    # its nearest periodic image around the protein for every saved frame.
    trajectory.make_molecules_whole(inplace=True)
    protein_atoms = trajectory.topology.select("protein and not element H")
    protein_ca = trajectory.topology.select("protein and name CA")
    ligand_atoms = trajectory.topology.select("resname LIG and not element H")
    ligand_all = trajectory.topology.select("resname LIG")
    if len(protein_atoms) == 0 or len(protein_ca) == 0 or len(ligand_atoms) == 0:
        raise ValueError("Could not identify protein heavy atoms, protein Cα atoms, and LIG atoms in system_final.pdb")

    if trajectory.unitcell_vectors is not None:
        boxes = trajectory.unitcell_vectors
        for frame in range(trajectory.n_frames):
            box = boxes[frame]
            delta = trajectory.xyz[frame, ligand_all].mean(axis=0) - trajectory.xyz[frame, protein_atoms].mean(axis=0)
            fractional = delta @ np.linalg.inv(box)
            image_delta = (fractional - np.round(fractional)) @ box
            trajectory.xyz[frame, ligand_all] += image_delta - delta

    trajectory.superpose(trajectory, 0, atom_indices=protein_ca)
    reference = trajectory[0]
    protein_backbone = trajectory.topology.select("protein and backbone")
    if len(protein_backbone) == 0:
        protein_backbone = protein_ca
    # Protein fitting has already been applied. Calculate raw coordinate RMSD
    # afterwards so the ligand RMSD remains relative to the protein frame.
    protein_delta = trajectory.xyz[:, protein_backbone] - reference.xyz[0, protein_backbone]
    ligand_delta = trajectory.xyz[:, ligand_all] - reference.xyz[0, ligand_all]
    protein_rmsd = np.sqrt(np.mean(np.sum(protein_delta * protein_delta, axis=2), axis=1))
    ligand_rmsd = np.sqrt(np.mean(np.sum(ligand_delta * ligand_delta, axis=2), axis=1))
    protein_rg = md.compute_rg(trajectory.atom_slice(protein_atoms))
    ligand_rg = md.compute_rg(trajectory.atom_slice(ligand_atoms))
    ca_rmsf = md.rmsf(trajectory, reference, atom_indices=protein_ca)
    ca_residues = [trajectory.topology.atom(index).residue for index in protein_ca]
    rmsf_residue, rmsf_values = [], []
    for residue in ca_residues:
        rmsf_residue.append(residue.resSeq)
        rmsf_values.append(float(ca_rmsf[len(rmsf_values)]))

    protein_residues = sorted({atom.residue.index for atom in trajectory.topology.atoms
                               if atom.residue.is_protein})
    ligand_residues = sorted({atom.residue.index for atom in trajectory.topology.atoms
                              if atom.residue.name == "LIG"})
    residue_pairs = np.asarray([(protein, ligand) for protein in protein_residues
                                for ligand in ligand_residues], dtype=int)
    if residue_pairs.size:
        distances, _ = md.compute_contacts(trajectory, contacts=residue_pairs,
                                           scheme="closest-heavy", periodic=False)
        contacts = (distances < 0.45).sum(axis=1)
    else:
        contacts = np.zeros(trajectory.n_frames)

    time_ns = trajectory.time / 1000.0
    cleaned_path = analysis_dir / "trajectory_fitted.xtc"
    trajectory.save_xtc(str(cleaned_path))
    return {
        "time_ns": time_ns,
        "protein_rmsd_nm": protein_rmsd,
        "ligand_rmsd_nm": ligand_rmsd,
        "protein_rg_nm": protein_rg,
        "ligand_rg_nm": ligand_rg,
        "protein_rmsf_nm": rmsf_values,
        "rmsf_residue": rmsf_residue,
        "ligand_contacts": contacts,
        "cleaned_trajectory": cleaned_path,
        "frame_count": int(trajectory.n_frames),
        "analysis_stride": int(analysis_stride),
        "production_start_ns": production_start_ps / 1000.0,
        "statistics": {
            "protein_backbone_rmsd_nm_mean": float(np.mean(protein_rmsd)),
            "protein_backbone_rmsd_nm_std": float(np.std(protein_rmsd)),
            "ligand_rmsd_nm_mean": float(np.mean(ligand_rmsd)),
            "protein_rg_nm_mean": float(np.mean(protein_rg)),
            "ligand_rg_nm_mean": float(np.mean(ligand_rg)),
            "ligand_contacts_mean": float(np.mean(contacts)),
        },
    }


def run_mmpbsa(output_dir, analysis_dir, *, model="gb", interval=1,
               membrane_thickness_angstrom=None, membrane_dielectric=7.0):
    """Run a single-trajectory gmx_MMPBSA calculation on the fitted trajectory."""
    executable = shutil.which("gmx_MMPBSA")
    if executable is None:
        raise RuntimeError("gmx_MMPBSA is not installed or is not available on PATH")
    output_dir = Path(output_dir)
    analysis_dir = Path(analysis_dir)
    required = [output_dir / "system.top", output_dir / "system_final.pdb",
                analysis_dir / "trajectory_fitted.xtc", analysis_dir / "index.ndx"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required MMPBSA inputs are missing: " + ", ".join(missing))
    mmpbsa_dir = analysis_dir / "mmpbsa"
    mmpbsa_dir.mkdir(exist_ok=True)
    input_path = mmpbsa_dir / "mmpbsa.in"
    if model not in {"gb", "pb", "pb_mem"}:
        raise ValueError(f"Unsupported MMPBSA model: {model}")
    model_namelist = "&gb\n igb=5, saltcon=0.150,\n/\n" if model == "gb" else "&pb\n istrng=0.150,\n/\n"
    if model == "pb_mem":
        if membrane_thickness_angstrom is None or membrane_thickness_angstrom <= 0:
            raise ValueError("The pb_mem model requires a positive membrane thickness in angstroms")
        state = json.loads((output_dir / "run_state.json").read_text())
        if not state["configuration"].get("membrane"):
            raise ValueError("The pb_mem model requires an OpenMM membrane run")
        center_z_angstrom = 10.0 * float(state["configuration"].get("membrane_center_z_nm") or 0.0)
        model_namelist = (
            "&pb\n memopt=1, emem=" + f"{membrane_dielectric:g}" + ", indi=4.0, "
            + f"mctrdz={center_z_angstrom:g}, mthick={membrane_thickness_angstrom:g}, "
            + "poretype=1, istrng=0.150,\n/\n"
        )
    input_path.write_text(
        "Single-trajectory protein-ligand binding free-energy estimate.\n"
        f"&general\n startframe=1, interval={interval}, verbose=1,\n/\n"
        + model_namelist
    )
    # gmx_MMPBSA uses -cr for residue/chain mapping, so it must contain just
    # the analyzed complex, not the solvent and ions present in system_final.
    # It also rejects duplicate command-line inputs by path.
    reference_path = mmpbsa_dir / "reference.pdb"
    import mdtraj as md

    reference = md.load_topology(str(output_dir / "system_final.pdb"))
    reference_atoms = reference.select("protein or resname LIG")
    if len(reference_atoms) == 0:
        raise ValueError("Could not select protein and LIG atoms for the MMPBSA reference structure")
    md.load_pdb(str(output_dir / "system_final.pdb")).atom_slice(reference_atoms).save_pdb(str(reference_path))
    command = [
        executable, "-O", "-i", str(input_path), "-cs", str(output_dir / "system_final.pdb"),
        "-cp", str(output_dir / "system.top"), "-cr", str(reference_path),
        "-ci", str(analysis_dir / "index.ndx"), "-cg", "Protein", "LIG",
        "-ct", str(analysis_dir / "trajectory_fitted.xtc"), "-o", str(mmpbsa_dir / "FINAL_RESULTS_MMPBSA.dat"),
        "-eo", str(mmpbsa_dir / "FINAL_RESULTS_MMPBSA.csv"), "-nogui",
    ]
    (mmpbsa_dir / "command.txt").write_text(" ".join(command) + "\n")
    logger.info("Running gmx_MMPBSA (%s); this may take some time", model)
    with (mmpbsa_dir / "gmx_MMPBSA.log").open("w") as log:
        completed = subprocess.run(command, cwd=mmpbsa_dir, stdout=log, stderr=subprocess.STDOUT,
                                   text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"gmx_MMPBSA failed with exit code {completed.returncode}; see {mmpbsa_dir / 'gmx_MMPBSA.log'}")
    return mmpbsa_dir


def analyze_run(output_dir, config, *, perform_mmpbsa=False, mmpbsa_model="gb",
                mmpbsa_interval=1, mmpbsa_membrane_thickness_angstrom=None,
                mmpbsa_membrane_dielectric=7.0, analysis_stride=10):
    output_dir = Path(output_dir)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    rows = read_state_data(output_dir / "state_data.csv")
    trajectory_path = output_dir / "trajectory.xtc"
    topology_path = output_dir / "system_final.pdb"
    trajectory_data = None
    if trajectory_path.is_file() and trajectory_path.stat().st_size:
        try:
            trajectory_data = _trajectory_metrics(
                trajectory_path, topology_path, analysis_dir, config, analysis_stride=analysis_stride
            )
        except Exception as error:
            logger.warning("Trajectory metrics unavailable: %s", error)
    metrics = {
        "frame_count": trajectory_data["frame_count"] if trajectory_data else 0,
        "analysis_stride": trajectory_data["analysis_stride"] if trajectory_data else analysis_stride,
        "production_start_ns": trajectory_data["production_start_ns"] if trajectory_data else None,
        "trajectory_metrics": trajectory_data["statistics"] if trajectory_data else {},
        "state_sample_count": len(rows),
        "temperature_k": config["temperature_k"],
        "pressure_bar": config["pressure_bar"],
        "note": "NPT density and volume are equilibration diagnostics. Instantaneous pressure is not reported by OpenMM StateDataReporter.",
    }
    _make_plots(analysis_dir, rows, metrics, trajectory_data, config)
    (analysis_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    if trajectory_data:
        topology = __import__("mdtraj").load_topology(str(topology_path))
        protein_indices = topology.select("protein")
        ligand_indices = topology.select("resname LIG")
        write_index_file(analysis_dir / "index.ndx", protein_indices.tolist(), ligand_indices.tolist())
    if perform_mmpbsa:
        run_mmpbsa(
            output_dir, analysis_dir, model=mmpbsa_model, interval=mmpbsa_interval,
            membrane_thickness_angstrom=mmpbsa_membrane_thickness_angstrom,
            membrane_dielectric=mmpbsa_membrane_dielectric,
        )

    image_names = ["diagnostics.png"]
    if (analysis_dir / "protein_rmsf.png").exists():
        image_names.append("protein_rmsf.png")
    links = "\n".join(f'<p><img src="{name}" alt="{html.escape(name)}" style="max-width:100%"></p>'
                       for name in image_names)
    summary_items = "".join(
        f"<li><strong>{html.escape(key.replace('_', ' '))}:</strong> {value:.3f}</li>"
        for key, value in metrics["trajectory_metrics"].items()
    )
    report = "".join((
        "<!doctype html><meta charset='utf-8'><title>MD diagnostics</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#172033}"
        "img{border:1px solid #dbe2ea;border-radius:8px}</style>"
        "<h1>Protein–ligand MD diagnostics</h1>",
        f"<p>Frames analyzed: {metrics['frame_count']}; state samples: {metrics['state_sample_count']}.</p>",
        f"<ul>{summary_items}</ul>" if summary_items else "",
        "<p>Temperature and NPT density/volume are diagnostics, not proof of thermodynamic convergence."
        " Check independent replicas and production-window stability for quantitative conclusions.</p>",
        links,
    ))
    (analysis_dir / "report.html").write_text(report)
    return analysis_dir
