"""
====================================================
Edge Pruning Summary
Project : GAT-based Secure Social IoT Framework
Author  : Swarup Sarkar
====================================================
"""

import pandas as pd


def summarize_pruned_edges(log_file="results/logs/pruning_log.csv"):
    """
    Generate summary of frequently pruned edges.
    """

    df = pd.read_csv(log_file)

    summary = (
        df.groupby(
            ["Source_Node", "Destination_Node"]
        )
        .agg(
            Times_Pruned=("Step", "count"),
            Average_Attention=("Attention_Score", "mean"),
            Average_Reward=("Reward", "mean"),
        )
        .reset_index()
        .sort_values(
            by="Times_Pruned",
            ascending=False
        )
    )

    print("=" * 60)
    print("Most Frequently Pruned Edges")
    print("=" * 60)

    print(summary)

    summary.to_csv(
        "results/tables/edge_summary.csv",
        index=False,
    )

    print("\nSaved to:")
    print("results/tables/edge_summary.csv")

    return summary


if __name__ == "__main__":

    summarize_pruned_edges()