"""
====================================================
Noisy Feature Ablation Study
Project : GAT-based Secure Social IoT Framework
====================================================
"""

import copy
import torch
import pandas as pd

from src.utils.data_loader import load_social_iot_dataset
from src.models.gat_social_iot import main as train_gat

NOISE_LEVELS = [
    0.00,
    0.05,
    0.10,
]
def add_noise(data, noise_level):

    noisy_data = copy.deepcopy(data)

    noise = torch.randn_like(
        noisy_data.x
    ) * noise_level

    noisy_data.x = noisy_data.x + noise

    return noisy_data

def run_ablation():

    results = []

    for noise_level in NOISE_LEVELS:

        print("\n" + "=" * 60)
        print(f"Noise Level : {noise_level}")
        print("=" * 60)

        data = load_social_iot_dataset()

        noisy_data = add_noise(
            data,
            noise_level,
        )

        model_path = (
            f"results/models/gat_noise_{noise_level:.2f}.pth"
        )

        best_val_acc, test_acc = train_gat(
            data=noisy_data,
            save_model_path=model_path,
        )

        results.append({
            "Noise_Level": noise_level,
            "Best_Validation_Accuracy": best_val_acc,
            "Test_Accuracy": test_acc,
        })

    df = pd.DataFrame(results)

    print("\n")
    print("=" * 70)
    print("Noisy Feature Ablation Results")
    print("=" * 70)
    print(df)

    df.to_csv(
        "results/tables/noisy_feature_ablation.csv",
        index=False,
    )


if __name__ == "__main__":

    run_ablation()