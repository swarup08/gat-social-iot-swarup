"""
====================================================
Save Social IoT Dataset
Project : GAT-based Secure Social IoT Framework
Author  : Swarup Sarkar
====================================================

Generates the synthetic Social IoT graph once,
converts it to PyTorch Geometric format,
and saves it for reproducible experiments.
"""

import os
import torch

from src.models.social_iot_to_pyg import convert_social_iot_to_pyg


def main():
    print("=" * 60)
    print("Generating Social IoT Dataset")
    print("=" * 60)

    # Generate dataset
    data = convert_social_iot_to_pyg()

    # Create data directory if it does not exist
    os.makedirs("data", exist_ok=True)

    # Save dataset
    save_path = "data/social_iot_data.pt"
    torch.save(data, save_path)

    print("\nDataset saved successfully.")
    print(f"Location : {save_path}")

    print("\nDataset Summary")
    print("-" * 40)
    print(f"Nodes        : {data.num_nodes}")
    print(f"Edges        : {data.edge_index.shape[1]}")
    print(f"Node Features: {data.x.shape[1]}")
    print(f"Edge Features: {data.edge_attr.shape[1]}")
    print(f"Classes      : {len(torch.unique(data.y))}")


if __name__ == "__main__":
    main()