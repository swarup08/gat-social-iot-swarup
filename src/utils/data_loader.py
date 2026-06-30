"""
====================================================
Social IoT Dataset Loader
Project : GAT-based Secure Social IoT Framework
Author  : Swarup Sarkar
====================================================

Loads the frozen Social IoT dataset for all experiments.
"""

import os
import torch


DATASET_PATH = "data/social_iot_data.pt"


def load_social_iot_dataset():
    """
    Load the frozen Social IoT dataset.

    Returns
    -------
    torch_geometric.data.Data
        PyTorch Geometric Data object.
    """

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"\nDataset not found: {DATASET_PATH}\n"
            "Run:\n"
            "python -m src.utils.save_social_iot_dataset"
        )

    data = torch.load(DATASET_PATH, weights_only=False)
    return data


if __name__ == "__main__":

    data = load_social_iot_dataset()

    print("=" * 50)
    print("Social IoT Dataset Loaded Successfully")
    print("=" * 50)

    print(f"Nodes          : {data.num_nodes}")
    print(f"Edges          : {data.edge_index.shape[1]}")
    print(f"Node Features  : {data.x.shape[1]}")
    print(f"Edge Features  : {data.edge_attr.shape[1]}")
    print(f"Classes        : {len(torch.unique(data.y))}")