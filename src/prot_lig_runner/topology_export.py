"""Topology export helpers for GROMACS and MMPBSA."""


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


def export_mmpbsa_complex(forcefield, modeller, output_dir, protein_residues):
    """Export a dry protein-ligand topology; solvent and lipids stay implicit."""
    import openmm.app as app
    import openmm.unit as unit
    from openff.interchange import Interchange

    complex_modeller = app.Modeller(modeller.topology, modeller.positions)
    discarded = [
        atom for atom in complex_modeller.topology.atoms()
        if atom.residue.name not in protein_residues and atom.residue.name != "LIG"
    ]
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
    interchange.to_gro(str(output_dir / "mmpbsa_complex.gro"))
