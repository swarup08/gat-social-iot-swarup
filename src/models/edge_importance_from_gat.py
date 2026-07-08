import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

from src.models.social_iot_to_pyg import convert_social_iot_to_pyg

# ---------------------------
# GAT with attention capture
# ---------------------------
class GATWithAttention(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.3):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, return_attention=False):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, (edge_index1, attn1) = self.gat1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, (edge_index2, attn2) = self.gat2(x, edge_index, return_attention_weights=True)

        if return_attention:
            return x, (edge_index1, attn1), (edge_index2, attn2)
        return x


def compute_edge_importance(data=None, model_path=None):
    """
    Compute attention-based edge importance scores.

    Parameters
    ----------
    data : torch_geometric.data.Data, optional
        Graph to score. Defaults to the frozen canonical dataset.
    model_path : str, optional
        Trained GAT checkpoint to load. Defaults to the canonical
        checkpoint. Both params exist so callers (e.g. a multi-graph
        evaluation loop) can score a freshly generated graph against
        a freshly trained model without touching the canonical files.

    Returns
    -------
    edge_index : Tensor
    s_uv : Tensor
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data is None:
        from src.utils.data_loader import load_social_iot_dataset
        data = load_social_iot_dataset()

    data = data.to(device)

    if model_path is None:
        model_path = "results/models/best_gat_social_iot.pth"

    model = GATWithAttention(
        in_channels=data.x.size(1),
        hidden_channels=16,
        out_channels=2,
        heads=8,
        dropout=0.3
    ).to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=False
        )
    )

    model.eval()

    with torch.no_grad():

        out, (ei1, attn1), (ei2, attn2) = model(
            data.x,
            data.edge_index,
            return_attention=True
        )

        attn1_mean = attn1.mean(dim=1)
        attn2_mean = attn2.squeeze()

        # Final attention score
        s_uv = 0.5 * attn1_mean + 0.5 * attn2_mean

        # ---------------------------------------------------
        # Remove self-loops added automatically by GATConv
        # ---------------------------------------------------
        num_original_edges = data.edge_index.shape[1]

        ei1 = ei1[:, :num_original_edges]
        s_uv = s_uv[:num_original_edges]

        return ei1.cpu(), s_uv.cpu()
# ---------------------------
# Main
# ---------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = convert_social_iot_to_pyg().to(device)

    model = GATWithAttention(
        in_channels=data.x.size(1),
        hidden_channels=16,
        out_channels=2,
        heads=8,
        dropout=0.3
    ).to(device)
# ---------------------------------------------------
# Load trained GAT model
# ---------------------------------------------------
    model.load_state_dict(
        torch.load(
            "results/models/best_gat_social_iot.pth",
            map_location=device,
            weights_only=False
        )
    )

    model.eval()

    print("Trained GAT model loaded successfully.")

    with torch.no_grad():
        out, (ei1, attn1), (ei2, attn2) = model(data.x, data.edge_index, return_attention=True)

    # ---------------------------
    # Aggregate attention to edge importance
    # ---------------------------
    # attn1 shape: [num_edges, heads]
    # attn2 shape: [num_edges, 1]  (single head)
    attn1_mean = attn1.mean(dim=1)           # average over heads
    attn2_mean = attn2.squeeze(dim=1)        # shape [num_edges]

    # Final importance score (simple average of layers)
    s_uv = 0.5 * attn1_mean + 0.5 * attn2_mean

    # ---------------------------------------------------
# Keep only original edges (ignore added self-loops)
# ---------------------------------------------------
    num_original_edges = data.edge_index.shape[1]

    ei1 = ei1[:, :num_original_edges]
    s_uv = s_uv[:num_original_edges]

    print("Edge importance computed.")
    print("Top-10 most important edges (index, score):")
    topk = torch.topk(s_uv, k=min(10, s_uv.numel()))
    for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
        u = ei1[0, idx].item()
        v = ei1[1, idx].item()
        print(f"Edge ({u} -> {v}) : importance = {score:.4f}")

    print("\nImportance statistics:")
    print("Min:", float(s_uv.min()))
    print("Max:", float(s_uv.max()))
    print("Mean:", float(s_uv.mean()))
    print("Std:", float(s_uv.std()))

if __name__ == "__main__":
    main()