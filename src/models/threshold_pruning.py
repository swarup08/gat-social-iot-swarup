import torch
import numpy as np
from torch_geometric.data import Data

from src.utils.data_loader import load_social_iot_dataset
from src.models.edge_importance_from_gat import compute_edge_importance

def threshold_pruning(percentile=20):
    """
    Remove edges whose attention score is below a percentile threshold.
    """

    data = load_social_iot_dataset()

    edge_index, s_uv = compute_edge_importance()

    threshold = np.percentile(
        s_uv.numpy(),
        percentile
    )

    keep_mask = s_uv >= threshold

    pruned_edge_index = edge_index[:, keep_mask]

    if data.edge_attr is not None:
        pruned_edge_attr = data.edge_attr[keep_mask]
    else:
        pruned_edge_attr = None

    pruned_graph = Data(
        x=data.x,
        edge_index=pruned_edge_index,
        edge_attr=pruned_edge_attr,
        y=data.y
    )

    print("\n==============================")
    print("Threshold Pruning")
    print("==============================")

    print(f"Threshold Percentile : {percentile}%")
    print(f"Threshold Value      : {threshold:.4f}")
    print(f"Original Edges       : {edge_index.shape[1]}")
    print(f"Remaining Edges      : {pruned_edge_index.shape[1]}")
    print(f"Removed Edges cls       : {edge_index.shape[1]-pruned_edge_index.shape[1]}")

    return pruned_graph


def threshold_pruning_by_budget(budget, data=None, model_path=None):
    """
    Remove exactly `budget` lowest-attention-score edges (global
    threshold, not per-node). Unlike threshold_pruning(percentile),
    which removes a variable count depending on the score
    distribution, this hits an exact edge count so it can be matched
    against other pruning methods at a shared budget.

    `data`/`model_path` default to the canonical frozen dataset/GAT
    checkpoint; pass both to score a freshly generated graph against
    a freshly trained model (e.g. multi-graph evaluation).
    """

    if data is None:
        data = load_social_iot_dataset()

    edge_index, s_uv = compute_edge_importance(data=data, model_path=model_path)

    total_edges = edge_index.shape[1]
    budget = min(budget, total_edges)

    # Ascending order: lowest-importance edges first.
    order = torch.argsort(s_uv)
    remove_idx = set(order[:budget].tolist())

    keep_mask = torch.tensor(
        [i not in remove_idx for i in range(total_edges)]
    )

    pruned_edge_index = edge_index[:, keep_mask]

    if data.edge_attr is not None:
        pruned_edge_attr = data.edge_attr[keep_mask]
    else:
        pruned_edge_attr = None

    pruned_graph = Data(
        x=data.x,
        edge_index=pruned_edge_index,
        edge_attr=pruned_edge_attr,
        y=data.y
    )

    print("\n==============================")
    print("Threshold Pruning (Budget)")
    print("==============================")

    print(f"Budget               : {budget}")
    print(f"Original Edges       : {total_edges}")
    print(f"Remaining Edges      : {pruned_edge_index.shape[1]}")
    print(f"Removed Edges        : {total_edges - pruned_edge_index.shape[1]}")

    return pruned_graph


if __name__ == "__main__":

    threshold_pruning(percentile=20)