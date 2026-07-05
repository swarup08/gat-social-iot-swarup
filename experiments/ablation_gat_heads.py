"""
====================================================
GAT Head Ablation Study
Project : GAT-based Secure Social IoT Framework
====================================================
"""

import pandas as pd
from src.models.gat_social_iot import main as train_gat

HEADS = [2, 4, 8]


def run_ablation():

    results = []

    print("=" * 60)
    print("GAT Head Ablation")
    print("=" * 60)

    for head in HEADS:

        print(f"\nRunning GAT with Heads = {head}")

        model_path = f"results/models/gat_heads_{head}.pth"

        best_val_acc, test_acc = train_gat(
            heads=head,
            save_model_path=model_path,
        )

        results.append(
            {
                "Heads": head,
                "Best_Validation_Accuracy": best_val_acc,
                "Test_Accuracy": test_acc,
            }
        )

    df = pd.DataFrame(results)

    df.to_csv(
        "results/tables/gat_heads_ablation.csv",
        index=False,
    )

    print("\n")
    print(df)


if __name__ == "__main__":

    run_ablation()