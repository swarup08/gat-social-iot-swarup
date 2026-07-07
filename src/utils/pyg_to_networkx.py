import networkx as nx


def pyg_to_networkx(data):
    """
    Convert a PyTorch Geometric Data object
    into a NetworkX directed graph.

    IMPORTANT: node and edge features are carried over as attributes.
    Without this, the botnet simulator running inside the RL
    environment has no access to the features that are supposed to
    drive infection probability (it would silently fall back to a
    flat constant rate, which is exactly the bug that made pruning
    decisions irrelevant to infection spread).

    Feature layout must match src/models/social_iot_to_pyg.py:
        x        = [avg_packet_rate, anomaly_score, degree]
        edge_attr= [is_mqtt, bandwidth, latency]
    """

    G = nx.DiGraph()

    # -------------------------
    # Add nodes (with features)
    # -------------------------
    x = data.x.tolist() if data.x is not None else None
    for node in range(data.num_nodes):
        attrs = {}
        if x is not None:
            attrs["avg_packet_rate"] = x[node][0]
            attrs["anomaly_score"] = x[node][1]
            attrs["degree"] = x[node][2]
        G.add_node(node, **attrs)

    # -------------------------
    # Add edges (with features)
    # -------------------------
    edge_index = data.edge_index.t().tolist()
    edge_attr = data.edge_attr.tolist() if data.edge_attr is not None else None

    for i, (u, v) in enumerate(edge_index):
        attrs = {}
        if edge_attr is not None:
            attrs["is_mqtt"] = edge_attr[i][0]
            attrs["bandwidth"] = edge_attr[i][1]
            attrs["latency"] = edge_attr[i][2]
        G.add_edge(u, v, **attrs)

    return G