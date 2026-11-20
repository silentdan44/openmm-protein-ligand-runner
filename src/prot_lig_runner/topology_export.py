"""Topology export helpers for GROMACS and MMPBSA."""

from pathlib import Path

from .analysis_metrics import resolve_ligand, _residue_number


def enable_batched_openmm_topology_conversion(batch_size=4096):
    """Batch molecule insertion to avoid quadratic OpenFF topology invalidation.

    OpenFF Interchange 0.4.1 builds one simple molecule per disconnected
    component. Its default helper calls ``Topology.add_molecule`` for every
    component, and each call invalidates atom-index caches across the full
    topology. Solvated membrane systems can contain hundreds of thousands of
    water molecules, so batch the same molecules through ``add_molecules``.
    """
    import networkx
    from openff.interchange.components import toolkit as interchange_toolkit
    from openff.toolkit import Topology

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if getattr(interchange_toolkit._simple_topology_from_graph, "_batched", False):
        return

    def simple_topology_from_graph(graph):
        topology = Topology()
        molecules = []
        expected_first_atom = 0
        for component in networkx.connected_components(graph):
            subgraph = interchange_toolkit._reorder_subgraph(graph.subgraph(component))
            first_atom = next(iter(subgraph.nodes))
            if first_atom != expected_first_atom:
                raise AssertionError(
                    f"OpenMM topology components are out of atom order: "
                    f"expected {expected_first_atom}, found {first_atom}"
                )
            molecule = interchange_toolkit._SimpleMolecule._from_subgraph(subgraph)
            expected_first_atom += molecule.n_atoms
            molecules.append(molecule)
            if len(molecules) >= batch_size:
                topology.add_molecules(molecules)
                molecules = []

        if molecules:
            topology.add_molecules(molecules)
        return topology

    simple_topology_from_graph._batched = True
    interchange_toolkit._simple_topology_from_graph = simple_topology_from_graph


def keep_interchange_molecules_separate(interchange):
    """Skip recursive graph-isomorphism grouping for large protein chains.

    GROMACS can represent repeated chains as separate molecule types. Keeping
    them separate avoids expensive all-by-all graph matching while preserving
    each chain's atom and force-field data.
    """
    topology = interchange.topology
    topology._cached_chemically_identical_molecules = {
        molecule_index: [
            (molecule_index, {atom_index: atom_index for atom_index in range(molecule.n_atoms)})
        ]
        for molecule_index, molecule in enumerate(topology.molecules)
    }


def mmpbsa_complex_selection(topology, protein_residues, ligand_selector=None):
    """Pick the ligand for MMPBSA and list atoms that must be deleted.

    Keeps protein residues plus exactly one LIG residue, so a multi-ligand run
    exports a complex whose single ligand group is the requested one.
    """
    selection = resolve_ligand(topology, ligand_selector)
    kept_ligand = (selection["chain_index"], selection["res_seq"])
    atoms = getattr(topology, "atoms")
    atoms = atoms() if callable(atoms) else atoms
    discarded = [
        atom for atom in atoms
        if atom.residue.name not in protein_residues
        and not (
            atom.residue.name == "LIG"
            and (atom.residue.chain.index, _residue_number(atom.residue)) == kept_ligand
        )
    ]
    return selection, discarded


def deduplicate_atomtypes(top_path):
    """Collapse per-atom SMIRNOFF atom types into equivalent 1-4-9 types.

    Interchange writes one atom type per atom for SMIRNOFF systems. ParmEd,
    which gmx_MMPBSA uses for the GROMACS→Amber conversion, turns that into
    NTYPES == NATOM, and the quadratic NONBONDED_PARM_INDEX makes the Amber
    prmtop tens of gigabytes (observed: 18 GB for a 27k-atom complex), which
    then exhausts memory in sander. Types that share element, particle type,
    sigma and epsilon are physically identical, so they are merged.
    """
    path = Path(top_path)
    lines = path.read_text().splitlines()
    section = None
    type_map = {}
    canonical = {}
    emitted = set()
    pair_seen = set()
    output = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip()
            output.append(line)
            continue
        if not stripped or stripped.startswith(";"):
            output.append(line)
            continue
        columns = stripped.split()
        if section == "atomtypes" and len(columns) >= 7:
            name = columns[0]
            offset = 1 if columns[1].isdigit() else 2
            key = (
                columns[offset],
                columns[offset + 3] if len(columns) >= offset + 6 else "",
                format(float(columns[-2]), ".6f"),
                format(float(columns[-1]), ".6f"),
            )
            canonical.setdefault(key, name)
            type_map[name] = canonical[key]
            if canonical[key] == name and name not in emitted:
                emitted.add(name)
                output.append(line)
            continue
        if section == "atoms" and len(columns) >= 6:
            columns[1] = type_map.get(columns[1], columns[1])
            output.append(" ".join(columns))
            continue
        if section == "nonbond_params" and len(columns) >= 5:
            first = type_map.get(columns[0], columns[0])
            second = type_map.get(columns[1], columns[1])
            if first > second:
                first, second = second, first
            if (first, second) in pair_seen:
                continue
            pair_seen.add((first, second))
            columns[0], columns[1] = first, second
            output.append(" ".join(columns))
            continue
        output.append(line)
    path.write_text("\n".join(output) + "\n")


def export_mmpbsa_complex(forcefield, modeller, output_dir, protein_residues, ligand_selector=None):
    """Export a dry protein-ligand topology; solvent and lipids stay implicit.

    A run can contain several ligands, but gmx_MMPBSA estimates one binding
    affinity per calculation, so the export keeps protein plus exactly one
    ligand residue: the one selected with --mmpbsa-ligand.
    """
    import openmm.app as app
    import openmm.unit as unit
    from openff.interchange import Interchange

    _selection, discarded = mmpbsa_complex_selection(
        modeller.topology, protein_residues, ligand_selector
    )
    complex_modeller = app.Modeller(modeller.topology, modeller.positions)
    complex_modeller.delete(discarded)
    if not any(atom.residue.name == "LIG" for atom in complex_modeller.topology.atoms()):
        raise ValueError("Could not create a protein-ligand-only topology for MMPBSA")

    with (output_dir / "mmpbsa_complex.pdb").open("w") as pdb_file:
        app.PDBFile.writeFile(complex_modeller.topology, complex_modeller.positions, pdb_file)

    topology_system = forcefield.createSystem(
        complex_modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,
    )
    enable_batched_openmm_topology_conversion()
    interchange = Interchange.from_openmm(
        topology=complex_modeller.topology,
        system=topology_system,
        positions=complex_modeller.positions,
    )
    keep_interchange_molecules_separate(interchange)
    if complex_modeller.topology.getPeriodicBoxVectors():
        interchange.box = complex_modeller.topology.getPeriodicBoxVectors()
    interchange.to_top(str(output_dir / "mmpbsa_complex.top"))
    deduplicate_atomtypes(output_dir / "mmpbsa_complex.top")
    interchange.to_gro(str(output_dir / "mmpbsa_complex.gro"))
