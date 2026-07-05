import torch
import torch.nn.functional as F

import os
import random
import numpy as np


from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# Import your PyG conversion
#from src.models.social_iot_to_pyg import convert_social_iot_to_pyg
from src.utils.data_loader import load_social_iot_dataset

# =====================================================
# Reproducibility
# =====================================================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
# ---------------------------
# GAT Model
# ---------------------------
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x

# ---------------------------
# Train / Eval utilities
# ---------------------------
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[mask] == data.y[mask]).sum().item()
    acc = correct / mask.sum().item()
    return acc

# ---------------------------
# Main
# ---------------------------

def main(
        heads=8,
        hidden_channels=16,
        save_model_path="results/models/best_gat_social_iot.pth",
        data=None,
):
    os.makedirs("results/models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #clsdata = convert_social_iot_to_pyg()
    if data is None:
        data = load_social_iot_dataset()
    data = data.to(device)

    # Create train/val/test masks
    num_nodes = data.num_nodes
    idx = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)

    train_idx = idx[:train_size]
    val_idx = idx[train_size:train_size + val_size]
    test_idx = idx[train_size + val_size:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    model = GAT(
        in_channels=data.x.size(1),
        hidden_channels=hidden_channels,
        out_channels=2,
        heads=heads,
        dropout=0.5
    ).to(device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=5e-4)
    print("Starting GAT training on Social IoT graph...")

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(1, 201):

        loss = train(model, data, optimizer)
        train_acc = evaluate(model, data, data.train_mask)
        val_acc = evaluate(model, data, data.val_mask)
        test_acc = evaluate(model, data, data.test_mask)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                save_model_path
            )
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                  f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | Test Acc: {test_acc:.3f}")

    print("GAT training completed.")

    print("\n====================================")
    print("Training Summary")
    print("====================================")

    print(f"Best Epoch           : {best_epoch}")
    print(f"Best Validation Acc  : {best_val_acc:.3f}")
    print(f"Test Accuracy        : {best_test_acc:.3f}")

    print("\nBest model saved at:")
    print(save_model_path)
    return best_val_acc, best_test_acc

if __name__ == "__main__":
    main()