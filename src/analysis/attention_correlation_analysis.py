import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from scipy.stats import pearsonr

from src.utils.data_loader import load_social_iot_dataset
from src.models.edge_importance_from_gat import compute_edge_importance

os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

def compute_node_degree(edge_index, num_nodes):
    """
    Compute node degree from edge index.
    """

    G = nx.DiGraph()

    edges = edge_index.t().tolist()

    G.add_edges_from(edges)

    degrees = []

    for node in range(num_nodes):
        degrees.append(G.degree(node))

    return np.array(degrees)

def compute_average_attention(edge_index, s_uv, num_nodes):
    """
    Compute average attention score received by each node.
    """

    attention_sum = np.zeros(num_nodes)
    attention_count = np.zeros(num_nodes)

    edges = edge_index.t().numpy()

    for i, (u, v) in enumerate(edges):
        attention_sum[v] += float(s_uv[i])
        attention_count[v] += 1

    average_attention = np.divide(
        attention_sum,
        attention_count,
        out=np.zeros_like(attention_sum),
        where=attention_count != 0
    )

    return average_attention

def correlation_analysis():

    data = load_social_iot_dataset()

    edge_index, s_uv = compute_edge_importance()

    degrees = compute_node_degree(
        edge_index,
        data.num_nodes
    )

    avg_attention = compute_average_attention(
        edge_index,
        s_uv,
        data.num_nodes
    )

    r, p = pearsonr(degrees, avg_attention)

    print("\n==============================")
    print("Attention Correlation Analysis")
    print("==============================")

    print(f"Pearson Correlation : {r:.4f}")
    print(f"P-value             : {p:.6f}")

    plot_attention_histogram(avg_attention)

    plot_degree_attention_scatter(
        degrees,
        avg_attention
    )

    return degrees, avg_attention, r, p

def plot_attention_histogram(avg_attention):

    plt.figure(figsize=(7,5))

    plt.hist(
        avg_attention,
        bins=10,
        edgecolor="black"
    )

    plt.xlabel("Average Attention Score")
    plt.ylabel("Number of Nodes")
    plt.title("Distribution of Node Attention Scores")

    plt.grid(True)

    plt.savefig(
        "results/plots/attention_histogram.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_degree_attention_scatter(degrees, avg_attention):

    plt.figure(figsize=(7,5))

    plt.scatter(
        degrees,
        avg_attention,
        s=70,
        alpha=0.8
    )

    plt.xlabel("Node Degree")
    plt.ylabel("Average Attention Score")
    plt.title("Node Degree vs Attention Score")

    plt.grid(True)

    plt.savefig(
        "results/plots/degree_attention_scatter.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
if __name__ == "__main__":

    correlation_analysis()