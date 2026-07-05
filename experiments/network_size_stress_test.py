"""
====================================================
Network Size Stress Test
Project : GAT-based Secure Social IoT Framework
====================================================
"""

import pandas as pd
from src.graph.social_iot_graph import SocialIoTGraph
from src.simulator.botnet_simulator import run_botnet_simulation

NETWORK_SIZES = [
    40,
    80,
    120,
]
def evaluate_network_size(num_nodes):

    print("\n" + "=" * 60)
    print(f"Network Size : {num_nodes}")
    print("=" * 60)

    graph_generator = SocialIoTGraph(
        num_nodes=num_nodes
    )

    graph_generator.generate_nodes()

    graph_generator.generate_edges(
        p=0.08
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

    return {
        "Nodes": G.number_of_nodes(),
        "Edges": G.number_of_edges(),
        "Final_Infection": final_infection,
        "Infection_Ratio": infection_ratio,
}
def run_stress_test():

    results = []

    for size in NETWORK_SIZES:

        result = evaluate_network_size(size)

        results.append(result)

    df = pd.DataFrame(results)

    print("\n")
    print("=" * 70)
    print("Network Size Stress Test Results")
    print("=" * 70)
    print(df)

    df.to_csv(
        "results/tables/network_size_stress_test.csv",
        index=False,
    )


if __name__ == "__main__":

    run_stress_test()