"""
====================================================
Generate Publication Figures
Project : GAT-based Secure Social IoT Framework
====================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.data_loader import load_social_iot_dataset
from src.utils.pyg_to_networkx import pyg_to_networkx

from src.models.threshold_pruning import threshold_pruning

from src.simulator.botnet_simulator import run_botnet_simulation

from experiments.evaluate_dynamic_rl import evaluate_dynamic_rl


os.makedirs("results/figures", exist_ok=True)


def generate_infection_curve():

    # ------------------------
    # Original Graph
    # ------------------------
    original_graph = pyg_to_networkx(
        load_social_iot_dataset()
    )

    # ------------------------
    # Threshold Graph
    # ------------------------
    threshold_graph = pyg_to_networkx(
        threshold_pruning(percentile=20)
    )

    # ------------------------
    # Infection Curves
    # ------------------------
    original_history = run_botnet_simulation(
        original_graph
    )

    threshold_history = run_botnet_simulation(
        threshold_graph
    )

    original_history = [
     x / original_graph.number_of_nodes()
        for x in original_history
    ]

    threshold_history = [
        x / threshold_graph.number_of_nodes()
        for x in threshold_history
    ]
    rl_history, _ = evaluate_dynamic_rl(
        verbose=False,
        return_history=True,
    )

    # ------------------------
    # Plot
    # ------------------------
    plt.figure(figsize=(8,5))

    plt.plot(
        original_history,
        linewidth=2,
        marker="o",
        label="No Pruning"
    )

    plt.plot(
        threshold_history,
        linewidth=2,
        marker="s",
        label="Static Threshold Pruning"
    )

    plt.plot(
        rl_history,
        linewidth=2,
        marker="^",
        label="Dynamic RL Pruning"
    )

    plt.xlabel("Simulation Step")

    plt.ylabel("Infection Ratio")

    plt.title("Botnet Infection under Different Edge Pruning Strategies")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/figures/infection_curve_comparison.png",
        dpi=300
    )

    plt.show()

import pandas as pd
import numpy as np


def generate_performance_comparison():

    print("=" * 60)
    print("Generating Performance Comparison Figure")
    print("=" * 60)

    df = pd.read_csv(
        "results/tables/comparison_results.csv"
    )

    methods = df["Method"]

    infection = df["Infection"]

    connectivity = df["Connectivity"]

    reward = df["Reward"].fillna(0)

    x = np.arange(len(methods))

    width = 0.25

    plt.figure(figsize=(9,6))

    plt.bar(
        x - width,
        infection,
        width,
        label="Infection"
    )

    plt.bar(
        x,
        connectivity,
        width,
        label="Connectivity"
    )

    plt.bar(
        x + width,
        reward / reward.max(),
        width,
        label="Removed Edges (Normalized)"
    )

    plt.xticks(
        x,
        methods
    )

    plt.ylabel("Performance Metric")

    plt.title(
        "Performance Comparison of Edge Pruning Strategies"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/figures/performance_comparison.png",
        dpi=300
    )

    plt.show()
def generate_xai_figure():

    print("=" * 60)
    print("Generating XAI Figure")
    print("=" * 60)

    edge_df = pd.read_csv(
        "results/tables/edge_summary.csv"
    )

    edge_df = (
    edge_df
    .sort_values(
        by="Average_Attention",
        ascending=False
    )
    .head(10)
)

    plt.figure(figsize=(9,5))

    bars = plt.barh(
        edge_df["Source_Node"].astype(str)
        + "→"
        + edge_df["Destination_Node"].astype(str),
        edge_df["Average_Attention"]
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Average Attention Score")

    plt.ylabel("Pruned Edge")

    plt.title("Top-10 Most Influential Edges Selected for Pruning by GAT Attention")

    for bar in bars:
        width = bar.get_width()

        plt.text(
            width + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            clip_on=False,
        )

    plt.xlim(
        0,
        edge_df["Average_Attention"].max() + 0.05
    )
    plt.tight_layout()

    plt.savefig(
        "results/figures/xai_top_edges.png",
        dpi=300
    )

    plt.show()

if __name__ == "__main__":

    generate_infection_curve()

    generate_performance_comparison()

    generate_xai_figure()