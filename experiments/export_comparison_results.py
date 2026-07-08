"""
====================================================
Export Comparison Results
Project : GAT-based Secure Social IoT Framework
Author  : Swarup Sarkar
====================================================

Exports comparison results to CSV.
"""

import os
import pandas as pd

from experiments.compare_methods import (
    BUDGET,
    evaluate_no_pruning,
    evaluate_static_pruning,
    evaluate_topk_pruning,
    evaluate_random_pruning,
    evaluate_degree_pruning,
    evaluate_betweenness_pruning,
)

from experiments.evaluate_dynamic_rl import (
    evaluate_dynamic_rl,
)


def export_results():

    print("=" * 60)
    print("Exporting Comparison Results")
    print("=" * 60)

    no_pruning = evaluate_no_pruning()

    static_pruning = evaluate_static_pruning(BUDGET)

    topk_pruning = evaluate_topk_pruning(BUDGET)

    random_pruning = evaluate_random_pruning(BUDGET)
    degree_pruning = evaluate_degree_pruning(BUDGET)
    betweenness_pruning = evaluate_betweenness_pruning(BUDGET)

    dynamic_rl = evaluate_dynamic_rl(
        verbose=False,
        max_steps=BUDGET,
    )

    results = [

        {
            "Method": no_pruning["method"],
            "Infection": no_pruning["infection"],
            "Connectivity": no_pruning["connectivity"],
            "Removed_Edges": no_pruning["removed_edges"],
            "Reward": None,
        },

        {
            "Method": static_pruning["method"],
            "Infection": static_pruning["infection"],
            "Connectivity": static_pruning["connectivity"],
            "Removed_Edges": static_pruning["removed_edges"],
            "Reward": None,
        },

        {
            "Method": topk_pruning["method"],
            "Infection": topk_pruning["infection"],
            "Connectivity": topk_pruning["connectivity"],
            "Removed_Edges": topk_pruning["removed_edges"],
            "Reward": None,
        },

        {
            "Method": random_pruning["method"],
            "Infection": random_pruning["infection"],
            "Connectivity": random_pruning["connectivity"],
            "Removed_Edges": random_pruning["removed_edges"],
            "Reward": None,
        },

        {
            "Method": degree_pruning["method"],
            "Infection": degree_pruning["infection"],
            "Connectivity": degree_pruning["connectivity"],
            "Removed_Edges": degree_pruning["removed_edges"],
            "Reward": None,
        },

        {
            "Method": betweenness_pruning["method"],
            "Infection": betweenness_pruning["infection"],
            "Connectivity": betweenness_pruning["connectivity"],
            "Removed_Edges": betweenness_pruning["removed_edges"],
            "Reward": None,
        },

        {
            "Method": dynamic_rl["method"],
            "Infection": dynamic_rl["infection"],
            "Connectivity": dynamic_rl["connectivity"],
            "Removed_Edges": dynamic_rl["removed_edges"],
            "Reward": dynamic_rl["total_reward"],
        },

    ]

    df = pd.DataFrame(results)

    os.makedirs(
        "results/tables",
        exist_ok=True
    )

    output_path = (
        "results/tables/comparison_results.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\nSaved Successfully")

    print(output_path)

    print("\n")

    print(df)


if __name__ == "__main__":

    export_results()