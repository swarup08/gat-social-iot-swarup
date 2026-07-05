"""
====================================================
Method Comparison
Project : GAT-based Secure Social IoT Framework
Author  : Swarup Sarkar
====================================================

Compare

1. No Pruning
2. Static GAT Pruning
3. Dynamic RL Pruning
"""

from src.rl.environment import GraphPruningEnv
from src.simulator.botnet_simulator import run_botnet_simulation

from src.models.threshold_pruning import threshold_pruning
from src.utils.pyg_to_networkx import pyg_to_networkx
from src.simulator.botnet_simulator import run_botnet_simulation
from experiments.evaluate_dynamic_rl import evaluate_dynamic_rl

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

def evaluate_static_pruning():
    """
    Static Threshold Pruning Evaluation.
    """

    print("=" * 60)
    print("Static Threshold Pruning Evaluation")
    print("=" * 60)

    # Threshold pruning
    pruned_graph = threshold_pruning(percentile=20)

    # Convert PyG -> NetworkX
    G = pyg_to_networkx(pruned_graph)

    # Run botnet simulation
    infection_history = run_botnet_simulation(
        G,
        infection_prob=0.25,
        recovery_prob=0.02,
        steps=20
    )

    infection_ratio = (
        infection_history[-1]
        /
        G.number_of_nodes()
    )

    connectivity_ratio = (
        G.number_of_edges()
        /
        124        # Original graph edges
    )

    removed_edges = (
        124
        -
        G.number_of_edges()
    )

    print(f"Nodes              : {G.number_of_nodes()}")
    print(f"Edges              : {G.number_of_edges()}")
    print(f"Final Infection    : {infection_ratio:.3f}")
    print(f"Connectivity Ratio : {connectivity_ratio:.3f}")
    print(f"Removed Edges      : {removed_edges}")

    return {
        "method": "Threshold",
        "infection": infection_ratio,
        "connectivity": connectivity_ratio,
        "removed_edges": removed_edges,
    }
def evaluate_dynamic_pruning():
    """
    Placeholder for Dynamic RL pruning.
    """

    print("=" * 60)
    print("Dynamic RL Evaluation")
    print("=" * 60)

    print("Coming next...")


def main():

    print("\nRunning No Pruning Evaluation...\n")
    no_pruning = evaluate_no_pruning()

    print("\nRunning Static Threshold Evaluation...\n")
    static_pruning = evaluate_static_pruning()

    print("\nRunning Dynamic RL Evaluation...\n")
    dynamic_rl = evaluate_dynamic_rl(verbose=False)

    print("\n" + "=" * 80)
    print("Comparison of Pruning Methods")
    print("=" * 80)

    print(
        f"{'Method':<20}"
        f"{'Infection':<15}"
        f"{'Connectivity':<18}"
        f"{'Removed Edges':<18}"
        f"{'Reward':<12}"
    )

    print("-" * 80)

    print(
        f"{no_pruning['method']:<20}"
        f"{no_pruning['infection']:<15.3f}"
        f"{no_pruning['connectivity']:<18.3f}"
        f"{no_pruning['removed_edges']:<18}"
        f"{'-':<12}"
    )

    print(
        f"{static_pruning['method']:<20}"
        f"{static_pruning['infection']:<15.3f}"
        f"{static_pruning['connectivity']:<18.3f}"
        f"{static_pruning['removed_edges']:<18}"
        f"{'-':<12}"
    )

    print(
        f"{dynamic_rl['method']:<20}"
        f"{dynamic_rl['infection']:<15.3f}"
        f"{dynamic_rl['connectivity']:<18.3f}"
        f"{dynamic_rl['removed_edges']:<18}"
        f"{dynamic_rl['total_reward']:<12.3f}"
    )

if __name__ == "__main__":
    main()



