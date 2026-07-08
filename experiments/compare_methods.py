"""
====================================================
Method Comparison
Project : GAT-based Secure Social IoT Framework
Author  : Swarup Sarkar
====================================================

Compare

1. No Pruning
2. Static GAT Pruning
3. Random / Degree-Centrality / Betweenness-Centrality Pruning (baselines)
4. Dynamic RL Pruning
"""

from src.rl.environment import GraphPruningEnv
from src.simulator.botnet_simulator import run_botnet_simulation

from src.models.threshold_pruning import threshold_pruning_by_budget
from src.models.topk_pruning import topk_pruning_by_budget
from src.utils.data_loader import load_social_iot_dataset
from src.utils.pyg_to_networkx import pyg_to_networkx
from src.simulator.botnet_simulator import run_botnet_simulation
from experiments.evaluate_dynamic_rl import evaluate_dynamic_rl

from src.models.baseline_pruning import (
    random_edge_pruning,
    degree_centrality_pruning,
    betweenness_centrality_pruning,
)

# Shared pruning budget: every method below removes (or targets removing,
# for Top-k's per-node constraint) exactly this many edges, so infection/
# connectivity numbers are compared at a matched cost rather than whatever
# count each method's own hyperparameter happens to produce.
BUDGET = 20

SIM_KWARGS = dict(infection_prob=0.25, recovery_prob=0.02, steps=20)


def _original_graph():
    """Original (unpruned) NetworkX graph, freshly loaded from the frozen dataset."""
    return pyg_to_networkx(load_social_iot_dataset())


def _evaluate_pruned_graph(method_name, G, original_edge_count):
    """
    Shared evaluation harness: run the botnet simulation on a pruned
    graph and compute the same (infection, connectivity, removed_edges)
    metrics used by every other pruning method, so results are directly
    comparable.
    """

    infection_history = run_botnet_simulation(G, **SIM_KWARGS)

    infection_ratio = infection_history[-1] / G.number_of_nodes()

    connectivity_ratio = G.number_of_edges() / original_edge_count

    removed_edges = original_edge_count - G.number_of_edges()

    return {
        "method": method_name,
        "infection": infection_ratio,
        "connectivity": connectivity_ratio,
        "removed_edges": removed_edges,
    }

def evaluate_no_pruning():
    """
    Baseline:
    No edge pruning.
    """

    env = GraphPruningEnv()

    env.reset()

    print("=" * 60)
    print("No Pruning Evaluation")
    print("=" * 60)

    infection_history = run_botnet_simulation(
        env.graph,
        infection_prob=0.25,
        recovery_prob=0.02,
        steps=20
    )

    infection_ratio = (
        infection_history[-1]
        /
        env.graph.number_of_nodes()
    )

    connectivity_ratio = 1.0

    removed_edges = 0

    print(f"Nodes              : {env.graph.number_of_nodes()}")
    print(f"Edges              : {env.graph.number_of_edges()}")
    print(f"Final Infection    : {infection_ratio:.3f}")
    print(f"Connectivity Ratio : {connectivity_ratio:.3f}")
    print(f"Removed Edges      : {removed_edges}")

    return {
        "method": "No Pruning",
        "infection": infection_ratio,
        "connectivity": connectivity_ratio,
        "removed_edges": removed_edges,
    }

def evaluate_static_pruning(budget):
    """
    Static Threshold Pruning Evaluation.

    Removes exactly `budget` lowest-attention-score edges (global
    threshold), so it is directly comparable at the same budget as the
    other methods rather than whatever a fixed percentile happens to
    produce.
    """

    print("=" * 60)
    print("Static Threshold Pruning Evaluation")
    print("=" * 60)

    pruned_graph = threshold_pruning_by_budget(budget)

    # Convert PyG -> NetworkX
    G = pyg_to_networkx(pruned_graph)

    original_edge_count = _original_graph().number_of_edges()

    result = _evaluate_pruned_graph("Threshold", G, original_edge_count)

    print(f"Nodes              : {G.number_of_nodes()}")
    print(f"Edges              : {G.number_of_edges()}")
    print(f"Final Infection    : {result['infection']:.3f}")
    print(f"Connectivity Ratio : {result['connectivity']:.3f}")
    print(f"Removed Edges      : {result['removed_edges']}")

    return result


def evaluate_topk_pruning(budget):
    """
    Top-K Pruning Evaluation.

    Keeps the top-k attention edges per source node, choosing k to
    best match `budget` total removed edges (see
    topk_pruning_by_budget's docstring for why an exact match isn't
    always achievable under Top-k's per-node constraint).
    """

    print("=" * 60)
    print("Top-K Pruning Evaluation")
    print("=" * 60)

    pruned_graph = topk_pruning_by_budget(budget)

    G = pyg_to_networkx(pruned_graph)

    original_edge_count = _original_graph().number_of_edges()

    result = _evaluate_pruned_graph("Top-K", G, original_edge_count)

    print(f"Nodes              : {G.number_of_nodes()}")
    print(f"Edges              : {G.number_of_edges()}")
    print(f"Final Infection    : {result['infection']:.3f}")
    print(f"Connectivity Ratio : {result['connectivity']:.3f}")
    print(f"Removed Edges      : {result['removed_edges']}")

    return result


def evaluate_random_pruning(num_edges_to_remove, trials=10):
    """
    Random edge pruning baseline: remove num_edges_to_remove edges
    chosen uniformly at random, repeated over many seeds and averaged
    (a single random draw is not representative of the strategy).
    """

    print("=" * 60)
    print("Random Pruning Evaluation")
    print("=" * 60)

    original_graph = _original_graph()
    original_edge_count = original_graph.number_of_edges()

    infections = []
    connectivities = []

    for trial in range(trials):
        G = random_edge_pruning(original_graph, num_edges_to_remove, seed=trial)
        trial_result = _evaluate_pruned_graph("Random", G, original_edge_count)
        infections.append(trial_result["infection"])
        connectivities.append(trial_result["connectivity"])

    result = {
        "method": "Random",
        "infection": sum(infections) / trials,
        "connectivity": sum(connectivities) / trials,
        "removed_edges": min(num_edges_to_remove, original_edge_count),
    }

    print(f"Trials             : {trials}")
    print(f"Final Infection    : {result['infection']:.3f}")
    print(f"Connectivity Ratio : {result['connectivity']:.3f}")
    print(f"Removed Edges      : {result['removed_edges']}")

    return result


def evaluate_degree_pruning(num_edges_to_remove):
    """
    Degree-centrality pruning baseline: remove edges touching the
    highest-degree (hub) nodes first.
    """

    print("=" * 60)
    print("Degree-Centrality Pruning Evaluation")
    print("=" * 60)

    original_graph = _original_graph()
    original_edge_count = original_graph.number_of_edges()

    G = degree_centrality_pruning(original_graph, num_edges_to_remove)
    result = _evaluate_pruned_graph("Degree Centrality", G, original_edge_count)

    print(f"Final Infection    : {result['infection']:.3f}")
    print(f"Connectivity Ratio : {result['connectivity']:.3f}")
    print(f"Removed Edges      : {result['removed_edges']}")

    return result


def evaluate_betweenness_pruning(num_edges_to_remove):
    """
    Betweenness-centrality pruning baseline: remove the edges lying on
    the most shortest paths first (structural bridges/bottlenecks).
    """

    print("=" * 60)
    print("Betweenness-Centrality Pruning Evaluation")
    print("=" * 60)

    original_graph = _original_graph()
    original_edge_count = original_graph.number_of_edges()

    G = betweenness_centrality_pruning(original_graph, num_edges_to_remove)
    result = _evaluate_pruned_graph("Betweenness Centrality", G, original_edge_count)

    print(f"Final Infection    : {result['infection']:.3f}")
    print(f"Connectivity Ratio : {result['connectivity']:.3f}")
    print(f"Removed Edges      : {result['removed_edges']}")

    return result


def evaluate_dynamic_pruning():
    """
    Placeholder for Dynamic RL pruning.
    """

    print("=" * 60)
    print("Dynamic RL Evaluation")
    print("=" * 60)

    print("Coming next...")


def main():

    print(f"\nRunning evaluations at shared budget={BUDGET} edges...\n")

    print("\nRunning No Pruning Evaluation...\n")
    no_pruning = evaluate_no_pruning()

    print("\nRunning Static Threshold Evaluation...\n")
    static_pruning = evaluate_static_pruning(BUDGET)

    print("\nRunning Top-K Evaluation...\n")
    topk_pruning = evaluate_topk_pruning(BUDGET)

    print("\nRunning Random Pruning Evaluation...\n")
    random_pruning = evaluate_random_pruning(BUDGET)

    print("\nRunning Degree-Centrality Evaluation...\n")
    degree_pruning = evaluate_degree_pruning(BUDGET)

    print("\nRunning Betweenness-Centrality Evaluation...\n")
    betweenness_pruning = evaluate_betweenness_pruning(BUDGET)

    print("\nRunning Dynamic RL Evaluation...\n")
    dynamic_rl = evaluate_dynamic_rl(verbose=False, max_steps=BUDGET)

    print("\n" + "=" * 80)
    print("Comparison of Pruning Methods")
    print("=" * 80)

    print(
        f"{'Method':<25}"
        f"{'Infection':<15}"
        f"{'Connectivity':<18}"
        f"{'Removed Edges':<18}"
        f"{'Reward':<12}"
    )

    print("-" * 80)

    def _print_row(result, reward=None):
        reward_str = f"{reward:.3f}" if reward is not None else "-"
        print(
            f"{result['method']:<25}"
            f"{result['infection']:<15.3f}"
            f"{result['connectivity']:<18.3f}"
            f"{result['removed_edges']:<18}"
            f"{reward_str:<12}"
        )

    _print_row(no_pruning)
    _print_row(static_pruning)
    _print_row(topk_pruning)
    _print_row(random_pruning)
    _print_row(degree_pruning)
    _print_row(betweenness_pruning)
    _print_row(dynamic_rl, reward=dynamic_rl["total_reward"])

if __name__ == "__main__":
    main()



