import matplotlib.pyplot as plt
from src.utils.pyg_to_networkx import pyg_to_networkx

from src.utils.data_loader import load_social_iot_dataset

from src.models.threshold_pruning import threshold_pruning
from src.models.topk_pruning import topk_pruning

from src.simulator.botnet_simulator import run_botnet_simulation

from src.utils.pyg_to_networkx import pyg_to_networkx

original_graph = pyg_to_networkx(load_social_iot_dataset())

threshold_graph = pyg_to_networkx(threshold_pruning())

topk_graph = pyg_to_networkx(topk_pruning())

def evaluate_pruning():

    print("=" * 60)
    print("Botnet Spread Evaluation")
    print("=" * 60)

    # ----------------------------
    # Convert all graphs to NetworkX
    # ----------------------------
    original_graph = pyg_to_networkx(
        load_social_iot_dataset()
    )

    threshold_graph = pyg_to_networkx(
        threshold_pruning(percentile=20)
    )

    topk_graph = pyg_to_networkx(
        topk_pruning(k=3)
    )

    # ----------------------------
    # Run simulations
    # ----------------------------
    print("\nRunning simulation on Original Graph...")
    original_history = run_botnet_simulation(original_graph)

    print("\nRunning simulation on Threshold Pruned Graph...")
    threshold_history = run_botnet_simulation(threshold_graph)

    print("\nRunning simulation on Top-K Pruned Graph...")
    topk_history = run_botnet_simulation(topk_graph)

    return original_history, threshold_history, topk_history
def plot_results(original, threshold, topk):

    plt.figure(figsize=(8,5))

    plt.plot(original, label="Original Graph")
    plt.plot(threshold, label="Threshold Pruning")
    plt.plot(topk, label="Top-K Pruning")

    plt.xlabel("Time Step")
    plt.ylabel("Infected Nodes")
    plt.title("Botnet Spread Comparison")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        "results/plots/pruning_evaluation.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

if __name__ == "__main__":

    original, threshold, topk = evaluate_pruning()

    plot_results(
        original,
        threshold,
        topk
    )