import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# Import your PyG conversion
from src.models.social_iot_to_pyg import convert_social_iot_to_pyg

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
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = convert_social_iot_to_pyg()
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
        hidden_channels=16, #8
        out_channels=2,   # benign vs vulnerable
        heads=8, #4
        dropout=0.5
    ).to(device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=5e-4)
    print("Starting GAT training on Social IoT graph...")
    for epoch in range(1, 201):
        loss = train(model, data, optimizer)
        train_acc = evaluate(model, data, data.train_mask)
        val_acc = evaluate(model, data, data.val_mask)
        test_acc = evaluate(model, data, data.test_mask)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                  f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | Test Acc: {test_acc:.3f}")

    print("GAT training completed.")

if __name__ == "__main__":
    main()