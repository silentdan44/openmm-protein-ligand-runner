"""Production-trajectory metrics for protein–ligand and membrane systems."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DistancePairError(ValueError):
    """Invalid explicitly requested residue-distance pair."""


def _residue_number(residue):
    """Residue number on both mdtraj (resSeq) and OpenMM (id) topologies."""
    value = getattr(residue, "resSeq", None)
    if value is None:
        value = getattr(residue, "id", None)
    return value


def _residue_label(residue):
    chain_id = _chain_label(residue.chain)
    insertion = getattr(residue, "insertion_code", "") or ""
    return f"{chain_id}:{_residue_number(residue)}{insertion}:{residue.name}"


def _chain_label(chain):
    value = getattr(chain, "chain_id", None)
    if value is None and not hasattr(chain, "chain_id"):
        value = getattr(chain, "id", None)
    return str(value) if value not in (None, "") else "_"


def _iter_residues(topology):
    """Iterate residues on both mdtraj (property) and OpenMM (method) topologies."""
    residues = getattr(topology, "residues", [])
    return residues() if callable(residues) else residues


def _iter_atoms(residue):
    """Iterate atoms on both mdtraj (property) and OpenMM (method) residues."""
    atoms = getattr(residue, "atoms", [])
    return atoms() if callable(atoms) else atoms


def ligand_groups(topology):
    """Describe each ligand residue with a stable 1-based label.

    Ligands are added in --ligand order, so LIG1 is the first SDF passed to the
    runner. Every LIG residue gets its own label even when two inputs are the
    same molecule (for example two GABA copies).
    """
    groups = []
    for index, residue in enumerate(
        (residue for residue in _iter_residues(topology) if residue.name == "LIG"), 1
    ):
        heavy = [atom.index for atom in _iter_atoms(residue) if atom.element.symbol != "H"]
        groups.append(
            {
                "label": f"LIG{index}",
                "residue_label": _residue_label(residue),
                "residue_index": residue.index,
                "chain_index": residue.chain.index,
                "res_seq": _residue_number(residue),
                "atoms": [atom.index for atom in _iter_atoms(residue)],
                "heavy_atoms": heavy,
            }
        )
    return groups


def resolve_ligand(topology, selector):
    """Resolve a ligand selector to one entry from ligand_groups().

    Accepts a 1-based index ("2"), the label ("LIG2"), or the residue label
    printed in the report (for example "A:123:LIG"). With no selector a run
    that contains exactly one ligand is unambiguous; several ligands must be
    chosen explicitly.
    """
    groups = ligand_groups(topology)
    if not groups:
        raise ValueError("The topology contains no LIG residues")
    if selector is None:
        if len(groups) == 1:
            return groups[0]
        options = ", ".join(f"{group['label']} ({group['residue_label']})" for group in groups)
        raise ValueError(
            "The topology contains several ligands; choose one with --mmpbsa-ligand. "
            f"Available: {options}"
        )
    text = str(selector).strip()
    if text.isdigit():
        index = int(text)
        if not 1 <= index <= len(groups):
            raise ValueError(
                f"Ligand index {index} is out of range; this run has {len(groups)} ligand(s)"
            )
        return groups[index - 1]
    for group in groups:
        if text in (group["label"], group["residue_label"]):
            return group
    options = ", ".join(f"{group['label']} ({group['residue_label']})" for group in groups)
    raise ValueError(f"Unknown ligand {selector!r}. Available: {options}")


def _write_rows(path, fieldnames, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_distance_pairs(topology, pair_specs):
    resolved = []
    residues = list(topology.residues)
    for first_spec, second_spec in pair_specs or []:
        atoms = []
        labels = []
        for spec in (first_spec, second_spec):
            try:
                chain_id, residue_id = spec.split(":", 1)
                resseq = int(residue_id)
            except (ValueError, AttributeError):
                raise DistancePairError(f"Invalid residue selector {spec!r}; expected CHAIN:RESID, e.g. A:550")
            requested_chain = chain_id or "_"
            matches = [residue for residue in residues
                       if _chain_label(residue.chain) == requested_chain
                       and residue.resSeq == resseq]
            if len(matches) != 1:
                raise DistancePairError(
                    f"Residue selector {spec!r} matched {len(matches)} residues; "
                    "use an unambiguous chain ID and residue number from system_final.pdb"
                )
            ca = [atom.index for atom in matches[0].atoms if atom.name == "CA"]
            if len(ca) != 1:
                raise DistancePairError(f"Residue {spec!r} has no unique C-alpha atom")
            atoms.append(ca[0])
            labels.append(_residue_label(matches[0]))
        resolved.append((atoms[0], atoms[1], f"{labels[0]} — {labels[1]}"))
    return resolved


def _membrane_core_ca(topology, reference_xyz, box_lengths, lipid_type, center_z_nm):
    """Select protein Cα atoms between the two lipid phosphate headgroup planes."""
    lipid_names = {lipid_type, lipid_type[:3]}
    phosphate_by_residue = {}
    for residue in topology.residues:
        if residue.name not in lipid_names:
            continue
        phosphate_atoms = [atom.index for atom in residue.atoms
                           if atom.name.strip().upper() == "P"]
        if len(phosphate_atoms) == 1:
            phosphate_by_residue[residue.index] = reference_xyz[phosphate_atoms[0], 2]
    if len(phosphate_by_residue) < 4:
        return [], None

    center = float(center_z_nm or 0.0)
    length_z = float(box_lengths[2]) if box_lengths is not None else 0.0

    def relative_z(z):
        delta = z - center
        if length_z > 0:
            delta = (delta + length_z / 2) % length_z - length_z / 2
        return delta

    phosphate_z = [relative_z(z) for z in phosphate_by_residue.values()]
    lower = [z for z in phosphate_z if z < 0]
    upper = [z for z in phosphate_z if z > 0]
    if not lower or not upper:
        return [], None
    lower_plane = sum(lower) / len(lower)
    upper_plane = sum(upper) / len(upper)
    if lower_plane >= upper_plane:
        return [], None

    selected = []
    for atom in topology.atoms:
        residue = atom.residue
        if residue.name in {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
                            "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
                            "THR", "TRP", "TYR", "VAL"} and atom.name == "CA":
            z = relative_z(reference_xyz[atom.index, 2])
            if lower_plane <= z <= upper_plane:
                selected.append(atom.index)
    return selected, {
        "lower_phosphate_plane_nm": lower_plane + center,
        "upper_phosphate_plane_nm": upper_plane + center,
        "selected_ca_count": len(selected),
        "selection": "protein C-alpha atoms between mean lipid phosphate z planes in system_final.pdb",
    }


def _membrane_timeseries(trajectory, lipid_type, center_z_nm):
    import numpy as np

    if trajectory.unitcell_vectors is None:
        return None
    lipid_names = {lipid_type, lipid_type[:3]}
    phosphate_indices = []
    for residue in trajectory.topology.residues:
        if residue.name == lipid_type or residue.name == lipid_type[:3]:
            phosphate = [atom.index for atom in residue.atoms if atom.name.strip().upper() == "P"]
            if len(phosphate) == 1:
                phosphate_indices.append(phosphate[0])
    if len(phosphate_indices) < 4:
        return None

    centers, thickness, apl, lower_count, upper_count = [], [], [], [], []
    center = float(center_z_nm or 0.0)
    for frame in range(trajectory.n_frames):
        box = trajectory.unitcell_vectors[frame]
        length_z = np.linalg.norm(box[2])
        z = trajectory.xyz[frame, phosphate_indices, 2] - center
        if length_z > 0:
            z = (z + length_z / 2) % length_z - length_z / 2
        lower = z[z < 0]
        upper = z[z > 0]
        if not len(lower) or not len(upper):
            centers.append((float("nan"), float("nan")))
            thickness.append(float("nan"))
            apl.append(float("nan"))
            lower_count.append(int(len(lower)))
            upper_count.append(int(len(upper)))
            continue
        low_mean, high_mean = float(lower.mean()), float(upper.mean())
        area = float(np.linalg.norm(np.cross(box[0], box[1])))
        centers.append((low_mean + center, high_mean + center))
        thickness.append(high_mean - low_mean)
        apl.append(area / ((len(lower) + len(upper)) / 2))
        lower_count.append(int(len(lower)))
        upper_count.append(int(len(upper)))
    return {
        "membrane_lower_headgroup_z_nm": np.asarray([v[0] for v in centers]),
        "membrane_upper_headgroup_z_nm": np.asarray([v[1] for v in centers]),
        "membrane_thickness_nm": np.asarray(thickness),
        "membrane_area_per_lipid_nm2": np.asarray(apl),
        "membrane_lower_lipid_count": np.asarray(lower_count),
        "membrane_upper_lipid_count": np.asarray(upper_count),
    }


def _unwrap_periodic(trajectory):
    """Make coordinates continuous across frames, in place.

    XTC stores each frame as written by the integrator, so a molecule that
    crosses a periodic boundary appears on the opposite side in the next
    frame. That produces box-sized jumps in RMSD and contact counts that look
    like binding/unbinding events. Minimum-image unwrapping per atom, relative
    to the previous frame, removes them; it is valid because atoms move far
    less than half a box between saved frames.
    """
    import numpy as np

    boxes = trajectory.unitcell_vectors
    if boxes is None or len(trajectory.xyz) < 2:
        return
    xyz = trajectory.xyz
    for frame in range(1, xyz.shape[0]):
        delta = xyz[frame] - xyz[frame - 1]
        fractional = delta @ np.linalg.inv(boxes[frame])
        delta = (fractional - np.round(fractional)) @ boxes[frame]
        xyz[frame] = xyz[frame - 1] + delta


def _place_ligands_in_protein_image(trajectory, protein_atoms, ligand_atoms):
    """Shift each ligand into the protein's periodic image in the first frame."""
    import numpy as np

    boxes = trajectory.unitcell_vectors
    if boxes is None or len(trajectory.xyz) == 0:
        return
    protein_center = trajectory.xyz[0, protein_atoms].mean(axis=0)
    for atoms in ligand_atoms:
        shift = trajectory.xyz[0, atoms].mean(axis=0) - protein_center
        fractional = shift @ np.linalg.inv(boxes[0])
        image_shift = (fractional - np.round(fractional)) @ boxes[0] - shift
        trajectory.xyz[:, atoms] += image_shift


def trajectory_metrics(trajectory_path, topology_path, analysis_dir, config, analysis_stride=10,
                       distance_pairs=None):
    import mdtraj as md
    import numpy as np

    trajectory = md.load(str(trajectory_path), top=str(topology_path), stride=analysis_stride)
    production_start_ps = (config["nvt_steps"] + config["npt_steps"]) * config["timestep_fs"] / 1000.0
    trajectory = trajectory[trajectory.time >= production_start_ps]
    if trajectory.n_frames == 0:
        raise ValueError("No production frames were found in trajectory.xtc; increase production length or adjust --xtc-interval")

    trajectory.make_molecules_whole(inplace=True)
    topology = trajectory.topology
    protein_atoms = topology.select("protein and not element H")
    protein_ca = topology.select("protein and name CA")
    ligand_atoms = topology.select("resname LIG and not element H")
    ligand_all = topology.select("resname LIG")
    if len(protein_atoms) == 0 or len(protein_ca) == 0 or len(ligand_atoms) == 0:
        raise ValueError("Could not identify protein heavy atoms, protein Cα atoms, and LIG atoms in system_final.pdb")
    groups = [group for group in ligand_groups(topology) if group["heavy_atoms"]]

    membrane_core_ca, membrane_selection = [], None
    membrane_data = None
    if config.get("membrane"):
        initial_structure = md.load_pdb(str(topology_path))
        membrane_core_ca, membrane_selection = _membrane_core_ca(
            topology, initial_structure.xyz[0],
            trajectory.unitcell_lengths[0] if trajectory.unitcell_lengths is not None else None,
            config.get("lipid_type", "POPC"), config.get("membrane_center_z_nm", 0.0),
        )
        if len(membrane_core_ca) >= 3:
            membrane_data = _membrane_timeseries(
                trajectory, config.get("lipid_type", "POPC"), config.get("membrane_center_z_nm", 0.0)
            )

    if trajectory.unitcell_vectors is not None:
        # Put every ligand in the protein's image in the reference frame, then
        # make the whole trajectory continuous. Without the second step a
        # molecule crossing a boundary jumps by a box length between frames and
        # shows up as a fake unbinding event in RMSD and contact curves.
        _place_ligands_in_protein_image(
            trajectory, protein_atoms, [group["atoms"] for group in groups]
        )
        _unwrap_periodic(trajectory)

    trajectory.superpose(trajectory, 0, atom_indices=protein_ca)
    reference = trajectory[0]
    protein_backbone = topology.select("protein and backbone")
    if len(protein_backbone) == 0:
        protein_backbone = protein_ca
    protein_delta = trajectory.xyz[:, protein_backbone] - reference.xyz[0, protein_backbone]
    ligand_delta = trajectory.xyz[:, ligand_all] - reference.xyz[0, ligand_all]
    protein_rmsd = np.sqrt(np.mean(np.sum(protein_delta * protein_delta, axis=2), axis=1))
    ligand_rmsd = np.sqrt(np.mean(np.sum(ligand_delta * ligand_delta, axis=2), axis=1))
    protein_rg = md.compute_rg(trajectory.atom_slice(protein_atoms))
    ligand_rg = md.compute_rg(trajectory.atom_slice(ligand_atoms))
    for group in groups:
        group_delta = trajectory.xyz[:, group["atoms"]] - reference.xyz[0, group["atoms"]]
        group["rmsd_nm"] = np.sqrt(np.mean(np.sum(group_delta * group_delta, axis=2), axis=1))
        group["rg_nm"] = md.compute_rg(trajectory.atom_slice(group["heavy_atoms"]))
        group["contacts"] = np.zeros(trajectory.n_frames)
        group["hbonds"] = np.zeros(trajectory.n_frames, dtype=int)

    protein_residues = sorted({atom.residue.index for atom in topology.atoms if atom.residue.is_protein})
    ligand_residues = sorted({atom.residue.index for atom in topology.atoms if atom.residue.name == "LIG"})
    residue_pairs = np.asarray([(protein, ligand) for protein in protein_residues
                                for ligand in ligand_residues], dtype=int)
    contact_rows = []
    if residue_pairs.size:
        distances, actual_pairs = md.compute_contacts(
            trajectory, contacts=residue_pairs, scheme="closest-heavy", periodic=False
        )
        contacts = (distances < 0.45).sum(axis=1)
        for pair_index, (protein_idx, ligand_idx) in enumerate(actual_pairs):
            in_contact = distances[:, pair_index] < 0.45
            for group in groups:
                if group["residue_index"] == int(ligand_idx):
                    group["contacts"] += in_contact
            occupancy = float(np.mean(in_contact))
            if occupancy > 0:
                contact_rows.append({
                    "protein_residue": _residue_label(topology.residue(int(protein_idx))),
                    "ligand_residue": _residue_label(topology.residue(int(ligand_idx))),
                    "occupancy": occupancy,
                    "mean_min_distance_nm": float(np.mean(distances[:, pair_index])),
                    "minimum_distance_nm": float(np.min(distances[:, pair_index])),
                })
        contact_rows.sort(key=lambda row: row["occupancy"], reverse=True)
        _write_rows(Path(analysis_dir) / "protein_ligand_contacts.csv",
                    ["protein_residue", "ligand_residue", "occupancy", "mean_min_distance_nm",
                     "minimum_distance_nm"], contact_rows)
    else:
        contacts = np.zeros(trajectory.n_frames)

    hbonds_per_frame = np.zeros(trajectory.n_frames, dtype=int)
    hbond_rows = []
    try:
        complex_atoms = np.concatenate([topology.select("protein"), ligand_all])
        complex_trajectory = trajectory.atom_slice(complex_atoms)
        complex_topology = complex_trajectory.topology
        complex_protein = complex_topology.select("protein")
        complex_ligand = complex_topology.select("resname LIG")
        candidates = md.baker_hubbard(complex_trajectory, freq=0.0, exclude_water=True, periodic=False)
        protein_set, ligand_set = set(complex_protein), set(complex_ligand)
        candidates = [tuple(map(int, triplet)) for triplet in candidates
                      if (triplet[0] in protein_set and triplet[2] in ligand_set)
                      or (triplet[0] in ligand_set and triplet[2] in protein_set)]
        if candidates:
            triplets = np.asarray(candidates, dtype=int)
            donor_acceptor_pairs = triplets[:, [1, 2]]
            distances = md.compute_distances(complex_trajectory, donor_acceptor_pairs, periodic=False)
            angles = md.compute_angles(complex_trajectory, triplets, periodic=False)
            present = (distances < 0.25) & (angles > np.deg2rad(120.0))
            hbonds_per_frame = present.sum(axis=1)
            for index, triplet in enumerate(candidates):
                donor = complex_topology.atom(triplet[0])
                acceptor = complex_topology.atom(triplet[2])
                protein_atom = donor if donor.residue.name != "LIG" else acceptor
                ligand_atom = acceptor if acceptor.residue.name == "LIG" else donor
                for group in groups:
                    if (ligand_atom.residue.chain.index == group["chain_index"]
                            and ligand_atom.residue.resSeq == group["res_seq"]):
                        group["hbonds"] += present[:, index]
                hbond_rows.append({
                    "protein_residue": _residue_label(protein_atom.residue),
                    "protein_atom": protein_atom.name,
                    "ligand_residue": _residue_label(ligand_atom.residue),
                    "ligand_atom": ligand_atom.name,
                    "occupancy": float(np.mean(present[:, index])),
                    "mean_donor_acceptor_distance_nm": float(np.mean(distances[:, index])),
                })
            hbond_rows.sort(key=lambda row: row["occupancy"], reverse=True)
            _write_rows(Path(analysis_dir) / "protein_ligand_hbonds.csv",
                        ["protein_residue", "protein_atom", "ligand_residue", "ligand_atom",
                         "occupancy", "mean_donor_acceptor_distance_nm"], hbond_rows)
    except Exception as error:
        # Hydrogen-bond analysis is optional and must not suppress other plots.
        hbond_rows = []
        logger.warning("Hydrogen-bond analysis skipped: %s", error)

    membrane_rmsd = None
    if membrane_core_ca:
        core_fitted = trajectory[:]
        core_fitted.superpose(core_fitted, 0, atom_indices=membrane_core_ca)
        delta = core_fitted.xyz[:, membrane_core_ca] - core_fitted.xyz[0, membrane_core_ca]
        membrane_rmsd = np.sqrt(np.mean(np.sum(delta * delta, axis=2), axis=1))
        ca_rmsf = md.rmsf(core_fitted, core_fitted[0], atom_indices=membrane_core_ca)
        rmsf_atoms = membrane_core_ca
        rmsf_residue = [topology.atom(index).residue.resSeq for index in rmsf_atoms]
    else:
        ca_rmsf = md.rmsf(trajectory, reference, atom_indices=protein_ca)
        rmsf_atoms = list(protein_ca)
        rmsf_residue = [topology.atom(index).residue.resSeq for index in rmsf_atoms]
    rmsf_values = [float(value) for value in ca_rmsf]
    rmsf_by_chain = {}
    for atom_index, rmsf_value in zip(rmsf_atoms, rmsf_values):
        residue = topology.atom(atom_index).residue
        chain_id = _chain_label(residue.chain)
        chain_data = rmsf_by_chain.setdefault(chain_id, {"residue": [], "rmsf_nm": []})
        chain_data["residue"].append(residue.resSeq)
        chain_data["rmsf_nm"].append(rmsf_value)

    distance_pair_warning = None
    try:
        resolved_pairs = _resolve_distance_pairs(topology, distance_pairs)
    except DistancePairError as error:
        logger.warning("Skipping requested residue-distance plots: %s", error)
        resolved_pairs = []
        distance_pair_warning = str(error)
    distance_series = {}
    for atom_a, atom_b, label in resolved_pairs:
        pair_distances = md.compute_distances(
            trajectory, np.asarray([[atom_a, atom_b]], dtype=int), periodic=False
        )[:, 0]
        distance_series[label] = pair_distances
    if distance_series:
        distance_path = Path(analysis_dir) / "residue_distances.csv"
        with distance_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            labels = list(distance_series)
            writer.writerow(["time_ns", *labels])
            writer.writerows(zip(trajectory.time / 1000.0, *(distance_series[label] for label in labels)))

    time_ns = trajectory.time / 1000.0
    cleaned_path = Path(analysis_dir) / "trajectory_fitted.xtc"
    trajectory.save_xtc(str(cleaned_path))
    trajectory_rows = []
    for index, time in enumerate(time_ns):
        row = {
            "time_ns": float(time),
            "protein_backbone_rmsd_nm": float(protein_rmsd[index]),
            "ligand_rmsd_nm": float(ligand_rmsd[index]),
            "protein_rg_nm": float(protein_rg[index]),
            "ligand_rg_nm": float(ligand_rg[index]),
            "ligand_protein_contacts": int(contacts[index]),
            "protein_ligand_hbonds": int(hbonds_per_frame[index]),
        }
        if membrane_rmsd is not None:
            row["membrane_core_rmsd_nm"] = float(membrane_rmsd[index])
        if membrane_data:
            for key, values in membrane_data.items():
                row[key] = float(values[index]) if values.dtype.kind == "f" else int(values[index])
        for label, values in distance_series.items():
            row[f"distance_nm:{label}"] = float(values[index])
        for group in groups:
            row[f"ligand_rmsd_nm:{group['label']}"] = float(group["rmsd_nm"][index])
            row[f"ligand_rg_nm:{group['label']}"] = float(group["rg_nm"][index])
            row[f"ligand_contacts:{group['label']}"] = int(group["contacts"][index])
            row[f"protein_ligand_hbonds:{group['label']}"] = int(group["hbonds"][index])
        trajectory_rows.append(row)
    if trajectory_rows:
        _write_rows(Path(analysis_dir) / "time_series.csv", list(trajectory_rows[0]), trajectory_rows)

    statistics: dict = {
        "protein_backbone_rmsd_nm_mean": float(np.mean(protein_rmsd)),
        "protein_backbone_rmsd_nm_std": float(np.std(protein_rmsd)),
        "ligand_rmsd_nm_mean": float(np.mean(ligand_rmsd)),
        "protein_rg_nm_mean": float(np.mean(protein_rg)),
        "ligand_rg_nm_mean": float(np.mean(ligand_rg)),
        "ligand_contacts_mean": float(np.mean(contacts)),
        "protein_ligand_hbonds_mean": float(np.mean(hbonds_per_frame)),
    }
    if membrane_rmsd is not None:
        statistics["membrane_core_rmsd_nm_mean"] = float(np.mean(membrane_rmsd))
        statistics["membrane_core_rmsd_nm_std"] = float(np.std(membrane_rmsd))
    if membrane_data:
        statistics["membrane_thickness_nm_mean"] = float(np.nanmean(membrane_data["membrane_thickness_nm"]))
        statistics["membrane_area_per_lipid_nm2_mean"] = float(np.nanmean(membrane_data["membrane_area_per_lipid_nm2"]))
    statistics["ligands"] = {
        group["label"]: {
            "residue": group["residue_label"],
            "ligand_rmsd_nm_mean": float(np.mean(group["rmsd_nm"])),
            "ligand_rmsd_nm_std": float(np.std(group["rmsd_nm"])),
            "ligand_rg_nm_mean": float(np.mean(group["rg_nm"])),
            "ligand_contacts_mean": float(np.mean(group["contacts"])),
            "protein_ligand_hbonds_mean": float(np.mean(group["hbonds"])),
        }
        for group in groups
    }

    return {
        "time_ns": time_ns,
        "protein_rmsd_nm": protein_rmsd,
        "membrane_core_rmsd_nm": membrane_rmsd,
        "membrane_core_selection": membrane_selection,
        "ligand_rmsd_nm": ligand_rmsd,
        "protein_rg_nm": protein_rg,
        "ligand_rg_nm": ligand_rg,
        "ligands": [
            {
                "label": group["label"],
                "residue": group["residue_label"],
                "rmsd_nm": group["rmsd_nm"],
                "rg_nm": group["rg_nm"],
                "contacts": group["contacts"],
                "hbonds": group["hbonds"],
            }
            for group in groups
        ],
        "protein_rmsf_nm": rmsf_values,
        "rmsf_residue": rmsf_residue,
        "rmsf_by_chain": rmsf_by_chain,
        "ligand_contacts": contacts,
        "contact_occupancy": contact_rows,
        "hbond_counts": hbonds_per_frame,
        "hbond_occupancy": hbond_rows,
        "membrane_metrics": membrane_data,
        "distance_series": distance_series,
        "distance_pair_warning": distance_pair_warning,
        "cleaned_trajectory": cleaned_path,
        "frame_count": int(trajectory.n_frames),
        "analysis_stride": int(analysis_stride),
        "production_start_ns": production_start_ps / 1000.0,
        "statistics": statistics,
    }
