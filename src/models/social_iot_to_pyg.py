import torch
from torch_geometric.data import Data
from src.graph.social_iot_graph import SocialIoTGraph

import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def generate_node_labels(G):
    degrees = [G.degree(n) for n in G.nodes()]
    median_degree = sorted(degrees)[len(degrees)//2]

    labels = []
    for node in G.nodes():
        feat = G.nodes[node]
        anomaly = feat["anomaly_score"]
        degree = feat["degree"]

        if anomaly > 0.6 and degree > median_degree:
            labels.append(1)   # vulnerable
        else:
            labels.append(0)   # benign

    return labels

def convert_social_iot_to_pyg(num_nodes=40, edge_prob=0.08, seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    graph_gen = SocialIoTGraph(num_nodes=num_nodes, seed=seed)
    graph_gen.generate_nodes()
    graph_gen.generate_edges(p=edge_prob)
    graph_gen.add_degree_features()

    G = graph_gen.graph

    # ---------------------------
    # Build node feature matrix
    # ---------------------------
    x_list = []
    for node in G.nodes():
        feat = G.nodes[node]
        x_list.append([
            feat["avg_packet_rate"],   # traffic feature
            feat["anomaly_score"],     # anomaly feature
            feat["degree"]             # topological feature
        ])
    x = torch.tensor(x_list, dtype=torch.float)

    # ---------------------------
    # Build edge index
    # ---------------------------
    edge_index_list = []
    for u, v in G.edges():
        edge_index_list.append([u, v])
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    # ---------------------------
    # Edge attributes (optional)
    # ---------------------------
    edge_attr_list = []
    for u, v in G.edges():
        e = G.edges[u, v]
        edge_attr_list.append([
            1.0 if e["protocol"] == "MQTT" else 0.0,
            e["bandwidth"],
            e["latency"]
        ])
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    # ---------------------------
    # Generate node labels
    # ---------------------------
    labels = generate_node_labels(G)
    y = torch.tensor(labels, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data

if __name__ == "__main__":
    data = convert_social_iot_to_pyg()
    print(data)
    print("Node feature shape (x):", data.x.shape)
    print("Edge index shape:", data.edge_index.shape)
    print("Edge attribute shape:", data.edge_attr.shape)
    print("Label distribution (0=benign, 1=vulnerable):", torch.bincount(data.y))