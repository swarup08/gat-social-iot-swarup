"""
====================================================
Node Explanation Summary
Project : GAT-based Secure Social IoT Framework
Author  : Swarup Sarkar
====================================================
"""

import pandas as pd


def summarize_nodes(
    log_file="results/logs/pruning_log.csv"
):
    """
    Generate node-wise pruning summary.
    """

    df = pd.read_csv(log_file)

    summary = (
        df.groupby("Source_Node")
        .agg(
            Pruned_Neighbors=("Destination_Node", list),
            Number_of_Pruned_Edges=("Destination_Node", "count"),
            Average_Attention=("Attention_Score", "mean"),
            Average_Reward=("Reward", "mean"),
        )
        .reset_index()
        .sort_values(
            by="Number_of_Pruned_Edges",
            ascending=False
        )
    )

    print("=" * 60)
    print("Per-Node Explanation Summary")
    print("=" * 60)

    print(summary)

    return summary


if __name__ == "__main__":

    summarize_nodes()