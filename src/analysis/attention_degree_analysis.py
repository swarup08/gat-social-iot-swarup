"""
====================================================
Attention vs Degree Correlation Analysis
Project : GAT-based Secure Social IoT Framework
Author  : Swarup Sarkar
====================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from scipy.stats import pearsonr

from src.models.social_iot_to_pyg import convert_social_iot_to_pyg
from src.models.edge_importance_from_gat import GATWithAttention

def load_graph():
    """
    Load the Social IoT graph in PyTorch Geometric format.
    """
    data = convert_social_iot_to_pyg()
    return data

def main():

    print("=" * 50)
    print("Attention vs Degree Analysis")
    print("=" * 50)

    data = load_graph()

    print()

    print("Number of Nodes :", data.num_nodes)
    print("Number of Edges :", data.edge_index.shape[1])


if __name__ == "__main__":
    main()