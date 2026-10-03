"""Performance compatibility helpers for large OpenMM to Interchange exports."""


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
