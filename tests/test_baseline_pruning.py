"""
Test the baseline pruning strategies (random, degree, betweenness).
"""

from src.utils.data_loader import load_social_iot_dataset
from src.utils.pyg_to_networkx import pyg_to_networkx

from src.models.baseline_pruning import (
    random_edge_pruning,
    degree_centrality_pruning,
    betweenness_centrality_pruning,
)


def main():

    G = pyg_to_networkx(load_social_iot_dataset())

    print("=" * 50)
    print("Original Graph")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print("=" * 50)

    budget = 20

    random_G = random_edge_pruning(G, budget, seed=0)
    degree_G = degree_centrality_pruning(G, budget)
    betweenness_G = betweenness_centrality_pruning(G, budget)

    print("Random Pruning       -> Edges:", random_G.number_of_edges())
    print("Degree Pruning       -> Edges:", degree_G.number_of_edges())
    print("Betweenness Pruning  -> Edges:", betweenness_G.number_of_edges())

    assert random_G.number_of_edges() == G.number_of_edges() - budget
    assert degree_G.number_of_edges() == G.number_of_edges() - budget
    assert betweenness_G.number_of_edges() == G.number_of_edges() - budget

    # Node/edge features must survive pruning (the botnet simulator
    # depends on them for feature-dependent infection probability).
    sample_node = next(iter(degree_G.nodes()))
    assert "anomaly_score" in degree_G.nodes[sample_node]

    sample_edge = next(iter(degree_G.edges()))
    assert "bandwidth" in degree_G.edges[sample_edge]

    print("\nAll baseline pruning checks passed.")


if __name__ == "__main__":
    main()
