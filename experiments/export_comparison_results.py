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
    evaluate_no_pruning,
    evaluate_static_pruning,
)

from experiments.evaluate_dynamic_rl import (
    evaluate_dynamic_rl,
)


def export_results():

    print("=" * 60)
    print("Exporting Comparison Results")
    print("=" * 60)

    no_pruning = evaluate_no_pruning()

    static_pruning = evaluate_static_pruning()

    dynamic_rl = evaluate_dynamic_rl(
        verbose=False
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