"""
====================================================
Baseline Edge-Pruning Strategies
Project : GAT-based Secure Social IoT Framework
====================================================

Graph-theoretic baselines that GAT-attention + RL pruning must beat to
justify the learned components (advisor priority #2). Each strategy
takes the same NetworkX graph the RL/threshold/topk pruning methods
consume (see src/utils/pyg_to_networkx.py) and a pruning budget
(number of directed edges to remove), so results are directly
comparable at matched budgets.
"""

import random

import networkx as nx


def random_edge_pruning(graph, num_edges_to_remove, seed=None):
    """
    Remove num_edges_to_remove edges chosen uniformly at random.

    Stochastic by design — callers should repeat over many seeds and
    average, rather than trusting a single draw.
    """

    G = graph.copy()

    edges = list(G.edges())
    num_edges_to_remove = min(num_edges_to_remove, len(edges))

    rng = random.Random(seed)
    to_remove = rng.sample(edges, num_edges_to_remove)

    G.remove_edges_from(to_remove)

    return G


def degree_centrality_pruning(graph, num_edges_to_remove):
    """
    Remove the edges touching the highest-degree nodes first.

    Edge score = degree(u) + degree(v). This is the "obvious" hub-aware
    strategy scale-free topology invites: cut around the busiest nodes.
    """

    G = graph.copy()

    degree = dict(G.degree())

    scored_edges = sorted(
        G.edges(),
        key=lambda e: degree[e[0]] + degree[e[1]],
        reverse=True,
    )

    num_edges_to_remove = min(num_edges_to_remove, len(scored_edges))

    G.remove_edges_from(scored_edges[:num_edges_to_remove])

    return G


def betweenness_centrality_pruning(graph, num_edges_to_remove):
    """
    Remove the highest edge-betweenness-centrality edges first.

    Targets edges that lie on the most shortest paths, i.e. structural
    bottlenecks/bridges rather than raw hub degree.
    """

    G = graph.copy()

    betweenness = nx.edge_betweenness_centrality(G)

    scored_edges = sorted(
        betweenness.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    num_edges_to_remove = min(num_edges_to_remove, len(scored_edges))

    to_remove = [edge for edge, _ in scored_edges[:num_edges_to_remove]]

    G.remove_edges_from(to_remove)

    return G
