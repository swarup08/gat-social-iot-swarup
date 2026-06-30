import networkx as nx


def pyg_to_networkx(data):
    """
    Convert a PyTorch Geometric Data object
    into a NetworkX directed graph.
    """

    G = nx.DiGraph()

    # -------------------------
    # Add nodes
    # -------------------------
    for node in range(data.num_nodes):
        G.add_node(node)

    # -------------------------
    # Add edges
    # -------------------------
    edge_index = data.edge_index.t().tolist()

    G.add_edges_from(edge_index)

    return G