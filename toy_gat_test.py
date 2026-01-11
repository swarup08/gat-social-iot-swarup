import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv

# Load citation dataset (Cora)
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

print("Dataset:", dataset)
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Number of classes:", dataset.num_classes)
print("Node feature dimension:", dataset.num_node_features)


# Define a simple GAT model
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=8, dropout=0.6)
        self.gat2 = GATConv(hidden_channels * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)


# Model, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(
    in_channels=dataset.num_node_features,
    hidden_channels=8,
    out_channels=dataset.num_classes
).to(device)

data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


# Test function
def test():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())
        accs.append(acc)
    return accs


# Run training
print("\nStarting Toy GAT Training...\n")
for epoch in range(1, 101):
    loss = train()
    train_acc, val_acc, test_acc = test()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Test Acc: {test_acc:.4f}")

print("\nToy GAT training completed successfully!")
