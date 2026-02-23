import torch
import copy
import matplotlib.pyplot as plt
import networkx as nx

from src.models.social_iot_to_pyg import convert_social_iot_to_pyg
from src.models.edge_importance_from_gat import GATWithAttention
from src.graph.social_iot_graph import SocialIoTGraph
from src.simulator.botnet_simulator import run_botnet_simulation  # ensure this function exists

def compute_edge_importance(data, heads=8):
    model = GATWithAttention(
        in_channels=data.x.size(1),
        hidden_channels=16,
        out_channels=2,
        heads=heads,
        dropout=0.3
    )
    model.eval()
    with torch.no_grad():
        _, (ei1, attn1), (ei2, attn2) = model(data.x, data.edge_index, return_attention=True)
    attn1_mean = attn1.mean(dim=1)
    attn2_mean = attn2.squeeze(dim=1)
    s_uv = 0.5 * attn1_mean + 0.5 * attn2_mean
    return ei1, s_uv

def pyg_to_networkx(data):
    G = nx.DiGraph()
    G.add_nodes_from(range(data.num_nodes))
    edges = data.edge_index.t().tolist()
    G.add_edges_from(edges)
    return G

def threshold_prune(G, edge_scores, edge_index, percentile=30):
    scores = edge_scores.cpu().numpy()
    thresh = float(torch.quantile(edge_scores, percentile/100.0))
    keep_mask = scores >= thresh

    pruned_edges = []
    for keep, (u, v) in zip(keep_mask, edge_index.t().tolist()):
        if keep and u != v:
            pruned_edges.append((u, v))

    Gp = nx.DiGraph()
    Gp.add_nodes_from(G.nodes())
    Gp.add_edges_from(pruned_edges)
    return Gp

def topk_prune_per_node(G, edge_scores, edge_index, k=3):
    edge_list = edge_index.t().tolist()
    scores = edge_scores.tolist()

    per_node = {}
    for (u, v), s in zip(edge_list, scores):
        if u == v:
            continue
        per_node.setdefault(u, []).append((v, s))

    pruned_edges = []
    for u, vs in per_node.items():
        vs_sorted = sorted(vs, key=lambda x: x[1], reverse=True)[:k]
        for v, s in vs_sorted:
            pruned_edges.append((u, v))

    Gp = nx.DiGraph()
    Gp.add_nodes_from(G.nodes())
    Gp.add_edges_from(pruned_edges)
    return Gp

def compare_botnet_spread(G_orig, G_pruned, steps=20):
    orig_curve = run_botnet_simulation(G_orig, steps=steps)
    pruned_curve = run_botnet_simulation(G_pruned, steps=steps)
    return orig_curve, pruned_curve

def plot_curves(orig, pruned):
    plt.figure()
    plt.plot(orig, label="Original Graph")
    plt.plot(pruned, label="Pruned Graph")
    plt.xlabel("Time step")
    plt.ylabel("Infected nodes")
    plt.title("Botnet Spread: Original vs Pruned Graph")
    plt.legend()
    plt.show()

def main():
    data = convert_social_iot_to_pyg()
    edge_index, scores = compute_edge_importance(data)

    G_orig = pyg_to_networkx(data)

    # A) Threshold pruning (remove bottom 30%)
    G_pruned_thresh = threshold_prune(G_orig, scores, edge_index, percentile=30)

    # B) Top-k per node pruning
    G_pruned_topk = topk_prune_per_node(G_orig, scores, edge_index, k=3)

    print("Edges (original):", G_orig.number_of_edges())
    print("Edges (threshold pruned):", G_pruned_thresh.number_of_edges())
    print("Edges (top-k pruned):", G_pruned_topk.number_of_edges())

    # Compare botnet spread
    orig_curve, thresh_curve = compare_botnet_spread(G_orig, G_pruned_thresh, steps=20)
    _, topk_curve = compare_botnet_spread(G_orig, G_pruned_topk, steps=20)

    plot_curves(orig_curve, thresh_curve)
    plot_curves(orig_curve, topk_curve)

if __name__ == "__main__":
    main()