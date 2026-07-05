"""
====================================================
Generate Publication Quality Plots
Project : GAT-based Secure Social IoT Framework
Author  : Swarup Sarkar
====================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def comparison_plot():

    df = pd.read_csv(
        "results/tables/comparison_results.csv"
    )

    os.makedirs(
        "results/plots",
        exist_ok=True
    )

    methods = df["Method"]

    infection = df["Infection"]

    connectivity = df["Connectivity"]

    x = range(len(methods))

    width = 0.35

    plt.figure(figsize=(8,5))

    plt.bar(
        [i-width/2 for i in x],
        infection,
        width,
        label="Infection"
    )

    plt.bar(
        [i+width/2 for i in x],
        connectivity,
        width,
        label="Connectivity"
    )

    plt.xticks(x, methods)

    plt.ylabel("Ratio")

    plt.title("Comparison of Graph Pruning Methods")

    plt.legend()

    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(
        "results/plots/comparison_bar_chart.png",
        dpi=300
    )

    plt.show()

def robustness_initial_seeds_plot():

    seeds = [3, 5, 7]

    infection = [
        0.889,
        0.886,
        0.894,
    ]

    reward = [
        6.266,
        6.291,
        6.216,
    ]

    plt.figure(figsize=(8,5))

    plt.plot(
        seeds,
        infection,
        marker="o",
        linewidth=2,
        label="Infection"
    )

    plt.plot(
        seeds,
        reward,
        marker="s",
        linewidth=2,
        label="Reward"
    )

    plt.xlabel("Initial Infected Nodes")

    plt.ylabel("Metric")

    plt.title("Robustness under Different Initial Infection Sizes")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/plots/robustness_initial_seeds.png",
        dpi=300
    )

    plt.show()
def robustness_probability_plot():

    probability = [
        0.15,
        0.25,
        0.35,
    ]

    infection = [
        0.807,
        0.890,
        0.912,
    ]

    reward = [
        7.078,
        6.253,
        6.028,
    ]

    plt.figure(figsize=(8,5))

    plt.plot(
        probability,
        infection,
        marker="o",
        linewidth=2,
        label="Infection"
    )

    plt.plot(
        probability,
        reward,
        marker="s",
        linewidth=2,
        label="Reward"
    )

    plt.xlabel("Infection Probability")

    plt.ylabel("Metric")

    plt.title("Robustness under Different Infection Probabilities")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/plots/robustness_infection_probability.png",
        dpi=300
    )

    plt.show()
def main():

    comparison_plot()

    robustness_initial_seeds_plot()

    robustness_probability_plot()

if __name__ == "__main__":

    main()