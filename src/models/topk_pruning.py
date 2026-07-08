import torch
from collections import defaultdict
from torch_geometric.data import Data

from src.utils.data_loader import load_social_iot_dataset
from src.models.edge_importance_from_gat import compute_edge_importance

def topk_pruning(k=3):
    """
    Keep only top-k attention edges
    for every source node.
    """

    data = load_social_iot_dataset()

    edge_index, s_uv = compute_edge_importance()

    edge_scores = []

    for i in range(edge_index.shape[1]):

        u = edge_index[0, i].item()
        v = edge_index[1, i].item()

        edge_scores.append(
            (
                u,
                v,
                float(s_uv[i]),
                i
            )
        )

    node_edges = defaultdict(list)

    for edge in edge_scores:
        node_edges[edge[0]].append(edge)

    # ---------------------------------------------------
    # Keep only Top-K edges for every source node
    # ---------------------------------------------------

    selected_indices = []

    for node in node_edges:

        # Sort edges by attention score (descending)
        sorted_edges = sorted(
            node_edges[node],
            key=lambda x: x[2],
            reverse=True
        )

        # Keep only Top-K
        top_edges = sorted_edges[:k]

        for edge in top_edges:
            selected_indices.append(edge[3])

    selected_indices = sorted(selected_indices)
    pruned_edge_index = edge_index[:, selected_indices]

    if data.edge_attr is not None:
        pruned_edge_attr = data.edge_attr[selected_indices]
    else:
        pruned_edge_attr = None

    pruned_graph = Data(
        x=data.x,
        edge_index=pruned_edge_index,
        edge_attr=pruned_edge_attr,
        y=data.y
    )
    print("\n==============================")
    print("Top-K Pruning")
    print("==============================")

    print(f"Top-K Value        : {k}")
    print(f"Original Edges     : {edge_index.shape[1]}")
    print(f"Remaining Edges    : {pruned_edge_index.shape[1]}")
    print(f"Removed Edges      : {edge_index.shape[1] - pruned_edge_index.shape[1]}")

    return pruned_graph


def topk_pruning_by_budget(budget, data=None, model_path=None):
    """
    Keep only the top-k attention edges per source node, choosing the
    smallest k whose total removed-edge count is closest to `budget`.

    Top-k's per-node fairness constraint (every node keeps its k best
    edges) means an exact global edge-count budget usually isn't
    achievable, unlike threshold/random/degree/betweenness pruning.
    Removed-edge count is non-increasing in k, so a linear scan finds
    the closest match. The actual removed-edge count (which may differ
    slightly from `budget`) is reported alongside the result so any
    mismatch stays visible rather than hidden.

    `data`/`model_path` default to the canonical frozen dataset/GAT
    checkpoint; pass both to score a freshly generated graph against
    a freshly trained model (e.g. multi-graph evaluation).
    """

    if data is None:
        data = load_social_iot_dataset()

    edge_index, s_uv = compute_edge_importance(data=data, model_path=model_path)

    node_edges = defaultdict(list)

    for i in range(edge_index.shape[1]):
        u = edge_index[0, i].item()
        node_edges[u].append((float(s_uv[i]), i))

    for node in node_edges:
        node_edges[node].sort(key=lambda x: x[0], reverse=True)

    max_k = max(len(edges) for edges in node_edges.values())

    def removed_for_k(k):
        return sum(max(0, len(edges) - k) for edges in node_edges.values())

    best_k = max_k
    best_diff = abs(removed_for_k(max_k) - budget)

    for k in range(1, max_k + 1):
        diff = abs(removed_for_k(k) - budget)
        if diff < best_diff:
            best_diff = diff
            best_k = k

    selected_indices = []
    for node, edges in node_edges.items():
        for _, i in edges[:best_k]:
            selected_indices.append(i)

    selected_indices = sorted(selected_indices)
    pruned_edge_index = edge_index[:, selected_indices]

    if data.edge_attr is not None:
        pruned_edge_attr = data.edge_attr[selected_indices]
    else:
        pruned_edge_attr = None

    pruned_graph = Data(
        x=data.x,
        edge_index=pruned_edge_index,
        edge_attr=pruned_edge_attr,
        y=data.y
    )

    print("\n==============================")
    print("Top-K Pruning (Budget)")
    print("==============================")

    print(f"Budget               : {budget}")
    print(f"Chosen K             : {best_k}")
    print(f"Original Edges       : {edge_index.shape[1]}")
    print(f"Remaining Edges      : {pruned_edge_index.shape[1]}")
    print(f"Removed Edges        : {edge_index.shape[1] - pruned_edge_index.shape[1]}")

    return pruned_graph


if __name__ == "__main__":

    topk_pruning(k=3)