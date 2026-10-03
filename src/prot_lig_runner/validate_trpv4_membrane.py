#!/usr/bin/env python3
"""Validate artifacts produced by the short Docker TRPV4 membrane run."""

import json
import sys
from pathlib import Path


def require(condition, message):
    if not condition:
        raise SystemExit(f"VALIDATION FAILED: {message}")


def main():
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "/results")
    geometry_path = output / "membrane_geometry.json"
    progress_path = output / "run_state.json"
    trajectory_path = output / "trajectory.xtc"
    structure_path = output / "system_final.pdb"
    for path in (geometry_path, progress_path, trajectory_path, structure_path):
        require(path.is_file() and path.stat().st_size > 0, f"missing or empty {path.name}")

    geometry = json.loads(geometry_path.read_text())
    progress = json.loads(progress_path.read_text())
    require(progress.get("phase") == "complete", "toy MD did not reach the complete phase")
    require(progress.get("platform") in {"CUDA", "OpenCL", "CPU"}, "invalid recorded OpenMM platform")

    protein = geometry["protein"]
    ligand = geometry["ligand"]
    lipid = geometry["lipid"]
    water = geometry["water"]
    leaflets = geometry["lipid_leaflets"]
    require(protein["heavy_atom_count"] > 1000, "protein heavy atoms were not found")
    require(ligand["heavy_atom_count"] > 0, "ligand heavy atoms were not found")
    require(lipid["heavy_atom_count"] > 1000, "membrane lipids were not found")
    require(water["heavy_atom_count"] > 10000, "the system is not sufficiently solvated")
    require(leaflets["below_center_heavy_atoms"] > 0, "lower lipid leaflet is missing")
    require(leaflets["above_center_heavy_atoms"] > 0, "upper lipid leaflet is missing")
    protein_z = protein["z_range_nm"]
    lipid_z = lipid["z_range_nm"]
    require(protein_z[0] < 0 < protein_z[1], "protein does not span the membrane center plane")
    require(lipid_z[0] < 0 < lipid_z[1], "lipid bilayer does not straddle the membrane center plane")
    distance = geometry["minimum_ligand_protein_heavy_distance_nm"]
    require(0 < distance < 0.8, f"ligand is not close to the protein (distance={distance:.3f} nm)")

    print(
        "VALIDATION PASSED: "
        f"phase={progress['phase']}, platform={progress['platform']}, "
        f"protein={protein['heavy_atom_count']} heavy atoms, "
        f"ligand={ligand['heavy_atom_count']}, lipid={lipid['heavy_atom_count']}, "
        f"water={water['heavy_atom_count']}, ligand-protein={distance:.3f} nm, "
        f"trajectory={trajectory_path.stat().st_size} bytes"
    )


if __name__ == "__main__":
    main()
