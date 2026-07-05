"""
====================================================
Graph Density Stress Test
Project : GAT-based Secure Social IoT Framework
====================================================
"""

import pandas as pd

GRAPH_DENSITIES = [
    0.05,
    0.08,
    0.12,
]

from src.graph.social_iot_graph import SocialIoTGraph
from src.simulator.botnet_simulator import run_botnet_simulation


def evaluate_graph_density(p):

    print("\n" + "=" * 60)
    print(f"Graph Density (Edge Probability) : {p}")
    print("=" * 60)

    graph_generator = SocialIoTGraph(
        num_nodes=40
    )

    graph_generator.generate_nodes()

    graph_generator.generate_edges(
        p=p
    )

    G = graph_generator.graph

    infection_history = run_botnet_simulation(
        G,
        steps=20,
    )

    final_infection = infection_history[-1]

    infection_ratio = (
        final_infection /
        G.number_of_nodes()
    )

    print("Nodes :", G.number_of_nodes())
    print("Edges :", G.number_of_edges())
    print("Final Infection :", final_infection)
    print("Infection Ratio :", round(infection_ratio, 3))

    return {
        "Density": p,
        "Nodes": G.number_of_nodes(),
        "Edges": G.number_of_edges(),
        "Final_Infection": final_infection,
        "Infection_Ratio": infection_ratio,
    }

def run_stress_test():

    results = []

    for density in GRAPH_DENSITIES:

        result = evaluate_graph_density(
            density
        )

        results.append(result)

    df = pd.DataFrame(results)

    print("\n")
    print("=" * 70)
    print("Graph Density Stress Test Results")
    print("=" * 70)
    print(df)

    df.to_csv(
        "results/tables/graph_density_stress_test.csv",
        index=False,
    )


if __name__ == "__main__":

    run_stress_test()