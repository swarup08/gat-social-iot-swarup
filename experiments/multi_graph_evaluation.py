"""
====================================================
Multi-Graph Evaluation (Step 3)
Project : GAT-based Secure Social IoT Framework
====================================================

Re-runs the full method comparison (No Pruning, Random, Degree
Centrality, Betweenness Centrality, Threshold, Top-K, Dynamic RL)
across many independently generated graph instances (different
seeds), at a shared matched pruning budget, and reports mean +/- std
for infection ratio and connectivity ratio per method. A single
graph's result is not evidence (advisor feedback); this is.

For each seed: generates a fresh graph, trains a fresh GAT (for
Threshold/Top-K attention scores and the RL state features) and a
fresh DQN (for Dynamic RL) on that specific instance. Both retrain
cheaply on this graph size (~2s / ~30-40s), and neither transfers
across independently generated instances: Top-K/Threshold ranking is
instance-specific, and the DQN's output layer is one Q-value slot per
edge INDEX in this instance's own edge ordering, so a checkpoint
trained on one graph is meaningless applied to another.

NOTE on Dynamic RL metric definition: evaluate_dynamic_rl() (used by
compare_methods.py) reports infection/connectivity averaged *across
the pruning episode's steps*, which is a different quantity than the
other 6 methods (final infection/connectivity after all budget edges
are already removed). Averaging across steps systematically inflates
RL's connectivity ratio, since early steps (before all budget edges
are pruned) have more edges remaining. This script instead reports
RL's *final-state* infection/connectivity after all budget edges are
removed, matching every other method, so the "matched budget"
comparison is actually apples-to-apples.
"""

import os
import io
import time
import contextlib

import pandas as pd

from src.models.social_iot_to_pyg import convert_social_iot_to_pyg
from src.models.gat_social_iot import main as train_gat
from src.models.threshold_pruning import threshold_pruning_by_budget
from src.models.topk_pruning import topk_pruning_by_budget
from src.models.baseline_pruning import (
    random_edge_pruning,
    degree_centrality_pruning,
    betweenness_centrality_pruning,
)
from src.utils.pyg_to_networkx import pyg_to_networkx
from src.simulator.botnet_simulator import run_botnet_simulation
from src.rl.environment import GraphPruningEnv
from src.rl.train_dqn import train as train_dqn

BUDGET = 20
SIM_KWARGS = dict(infection_prob=0.25, recovery_prob=0.02, steps=20)
RANDOM_TRIALS = 10
DQN_EPISODES = 400
SCRATCH_GAT_PATH = "results/models/_scratch_multi_graph_gat.pth"

RAW_RESULTS_PATH = "results/tables/multi_graph_raw_results.csv"
SUMMARY_PATH = "results/tables/multi_graph_summary.csv"


def _evaluate_pruned_graph(method_name, G, original_edge_count):
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


def _evaluate_dynamic_rl(data, budget, dqn_episodes):
    """
    Train a fresh DQN on this graph instance and evaluate greedily
    (epsilon=0 equivalent via predict_action) for exactly `budget`
    valid steps, then report the FINAL state — matching the other six
    methods' "final infection/connectivity after all budget edges are
    removed" convention (see module docstring for why this differs
    from evaluate_dynamic_rl.py's across-episode average).
    """

    train_env = GraphPruningEnv(
        max_steps=budget, data=data, gat_model_path=SCRATCH_GAT_PATH,
    )

    agent, _, _ = train_dqn(
        episodes=dqn_episodes,
        env=train_env,
        save_path=None,
        save_history=False,
        verbose=False,
    )

    eval_env = GraphPruningEnv(
        max_steps=budget, data=data, gat_model_path=SCRATCH_GAT_PATH,
    )
    state = eval_env.reset()

    done = False
    infection_ratio = 0.0

    while not done:
        valid_actions = eval_env.get_valid_actions()
        if not valid_actions:
            break
        action = agent.predict_action(state, valid_actions)
        state, reward, done, info = eval_env.step(action)
        infection_ratio = info["infection_ratio"]

    original_edge_count = eval_env.original_graph.number_of_edges()
    removed_edges = original_edge_count - eval_env.graph.number_of_edges()
    connectivity_ratio = eval_env.graph.number_of_edges() / original_edge_count

    return {
        "method": "Dynamic RL",
        "infection": infection_ratio,
        "connectivity": connectivity_ratio,
        "removed_edges": removed_edges,
    }


def evaluate_one_graph(seed, budget=BUDGET, dqn_episodes=DQN_EPISODES):
    """Run all 7 methods on one freshly generated graph instance."""

    data = convert_social_iot_to_pyg(seed=seed)
    original_graph = pyg_to_networkx(data)
    original_edge_count = original_graph.number_of_edges()

    results = []

    # --- No Pruning ---
    infection_history = run_botnet_simulation(original_graph, **SIM_KWARGS)
    results.append({
        "method": "No Pruning",
        "infection": infection_history[-1] / original_graph.number_of_nodes(),
        "connectivity": 1.0,
        "removed_edges": 0,
    })

    # --- Random (averaged over RANDOM_TRIALS seed draws) ---
    infections, connectivities = [], []
    for trial in range(RANDOM_TRIALS):
        G = random_edge_pruning(original_graph, budget, seed=trial)
        r = _evaluate_pruned_graph("Random", G, original_edge_count)
        infections.append(r["infection"])
        connectivities.append(r["connectivity"])
    results.append({
        "method": "Random",
        "infection": sum(infections) / RANDOM_TRIALS,
        "connectivity": sum(connectivities) / RANDOM_TRIALS,
        "removed_edges": min(budget, original_edge_count),
    })

    # --- Degree Centrality ---
    G = degree_centrality_pruning(original_graph, budget)
    results.append(_evaluate_pruned_graph("Degree Centrality", G, original_edge_count))

    # --- Betweenness Centrality ---
    G = betweenness_centrality_pruning(original_graph, budget)
    results.append(_evaluate_pruned_graph("Betweenness Centrality", G, original_edge_count))

    # --- Fresh GAT for this instance (Threshold / Top-K / RL state features) ---
    with contextlib.redirect_stdout(io.StringIO()):
        train_gat(data=data, save_model_path=SCRATCH_GAT_PATH)

    # --- Threshold (budget-exact) ---
    with contextlib.redirect_stdout(io.StringIO()):
        pruned = threshold_pruning_by_budget(budget, data=data, model_path=SCRATCH_GAT_PATH)
    G = pyg_to_networkx(pruned)
    results.append(_evaluate_pruned_graph("Threshold", G, original_edge_count))

    # --- Top-K (closest achievable match to budget) ---
    with contextlib.redirect_stdout(io.StringIO()):
        pruned = topk_pruning_by_budget(budget, data=data, model_path=SCRATCH_GAT_PATH)
    G = pyg_to_networkx(pruned)
    results.append(_evaluate_pruned_graph("Top-K", G, original_edge_count))

    # --- Dynamic RL (fresh DQN trained on this instance) ---
    results.append(_evaluate_dynamic_rl(data, budget, dqn_episodes))

    for r in results:
        r["seed"] = seed

    return results


def run_multi_graph_evaluation(num_graphs=40, budget=BUDGET, dqn_episodes=DQN_EPISODES, seed_offset=1000):

    os.makedirs("results/tables", exist_ok=True)

    print("=" * 70)
    print(f"Multi-Graph Evaluation: {num_graphs} graphs, budget={budget} edges")
    print("=" * 70)

    all_rows = []
    first_write = True

    for i in range(num_graphs):
        seed = seed_offset + i
        t0 = time.time()

        rows = evaluate_one_graph(seed, budget=budget, dqn_episodes=dqn_episodes)
        all_rows.extend(rows)

        elapsed = time.time() - t0

        # Append incrementally so partial progress survives an interruption.
        df_chunk = pd.DataFrame(rows)
        df_chunk.to_csv(
            RAW_RESULTS_PATH,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )
        first_write = False

        rl_row = next(r for r in rows if r["method"] == "Dynamic RL")
        print(
            f"[{i+1:02d}/{num_graphs}] seed={seed} | {elapsed:5.1f}s | "
            f"RL infection={rl_row['infection']:.3f} connectivity={rl_row['connectivity']:.3f}"
        )

    df = pd.DataFrame(all_rows)

    summary = (
        df.groupby("method")
        .agg(
            infection_mean=("infection", "mean"),
            infection_std=("infection", "std"),
            connectivity_mean=("connectivity", "mean"),
            connectivity_std=("connectivity", "std"),
            removed_edges_mean=("removed_edges", "mean"),
            n_graphs=("seed", "count"),
        )
        .reset_index()
    )

    method_order = [
        "No Pruning", "Threshold", "Top-K", "Random",
        "Degree Centrality", "Betweenness Centrality", "Dynamic RL",
    ]
    summary["method"] = pd.Categorical(summary["method"], categories=method_order, ordered=True)
    summary = summary.sort_values("method").reset_index(drop=True)

    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n" + "=" * 70)
    print("Summary (mean +/- std across graphs)")
    print("=" * 70)
    print(summary.to_string(index=False))

    print(f"\nRaw per-graph results : {RAW_RESULTS_PATH}")
    print(f"Summary statistics    : {SUMMARY_PATH}")

    return summary


if __name__ == "__main__":

    run_multi_graph_evaluation()
