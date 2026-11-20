"""Trajectory and state-file helpers shared by analysis and MMPBSA."""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)


def read_state_data(path):
    """Read OpenMM StateDataReporter CSV output into normalized records."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as source:
        rows = []
        for row in csv.DictReader(source):
            try:
                rows.append({key: float(value) if value not in (None, "") else math.nan
                             for key, value in row.items()})
            except (TypeError, ValueError):
                continue
    return rows


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


def _mmpbsa_window_bounds(trajectory_path, topology_path, last_ns):
    """Return absolute ps bounds for the final requested production window."""
    import mdtraj as md

    first_time = last_time = None
    for chunk in md.iterload(str(trajectory_path), top=str(topology_path), chunk=1000):
        if chunk.n_frames == 0:
            continue
        if first_time is None:
            first_time = float(chunk.time[0])
        last_time = float(chunk.time[-1])
    if last_time is None:
        raise ValueError("The production trajectory contains no frames for MMPBSA")
    cutoff = last_time - float(last_ns) * 1000.0
    if cutoff < first_time:
        logger.warning(
            "Requested final %.3f ns for MMPBSA, but the available production trajectory is "
            "only %.3f ns; using all available production frames",
            last_ns, (last_time - first_time) / 1000.0,
        )
        cutoff = first_time
    return cutoff


def _wrap_selection_into_box(trajectory):
    """Translate each frame so the selected complex sits in the box centre.

    The analysis unwraps trajectories for RMSD; gmx_MMPBSA expects the complex
    inside the box for its focusing/PB grid, so shift the whole selection by one
    lattice vector per frame instead of wrapping individual atoms.
    """
    import numpy as np

    boxes = trajectory.unitcell_vectors
    if boxes is None:
        return
    try:
        boxes = np.asarray(boxes, dtype=float)
        xyz = np.asarray(trajectory.xyz, dtype=float)
    except (TypeError, ValueError):
        return
    if boxes.ndim != 3 or xyz.ndim != 3:
        return
    for frame in range(min(len(xyz), len(boxes))):
        inverse = np.linalg.inv(boxes[frame])
        fractional = trajectory.xyz[frame].mean(axis=0) @ inverse
        shift = (0.5 - fractional) @ boxes[frame]
        trajectory.xyz[frame] += shift


def _copy_mmpbsa_window(source_path, topology_path, target_path, *, last_ns,
                        atom_indices=None):
    """Write a production trajectory window, optionally reduced to selected atoms."""
    import mdtraj as md

    cutoff = _mmpbsa_window_bounds(source_path, topology_path, last_ns)
    frames_written = 0
    with md.formats.XTCTrajectoryFile(str(target_path), mode="w") as writer:
        for chunk in md.iterload(str(source_path), top=str(topology_path), chunk=1000):
            if chunk.n_frames == 0:
                continue
            selected = chunk[chunk.time >= cutoff]
            if selected.n_frames == 0:
                continue
            if atom_indices is not None:
                selected = selected.atom_slice(atom_indices)
                _wrap_selection_into_box(selected)
            writer.write(selected.xyz, time=selected.time,
                         box=selected.unitcell_vectors)
            frames_written += selected.n_frames
    if frames_written == 0:
        raise ValueError("No production frames fall inside the requested MMPBSA window")
    return frames_written
