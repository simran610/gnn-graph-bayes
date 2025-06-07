# GAT model with dropout
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.5):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)

        # Use node type == 0 to find root nodes
        node_types = data.x[:, 0]
        root_mask = (node_types == 0)
        root_indices = root_mask.nonzero(as_tuple=False).squeeze()

        # If batch > 1, pick one root per graph
        if batch is not None and batch.numel() > 0:
            roots_per_graph = []
            for i in range(batch.max().item() + 1):
                # For each graph in batch, find root node index
                idx = ((batch == i) & (node_types == 0)).nonzero(as_tuple=False)
                if idx.numel() == 0:
                    raise ValueError(f"No root node found in graph {i}")
                roots_per_graph.append(idx[0].item())
            root_indices = torch.tensor(roots_per_graph, device=x.device)

        return x[root_indices]
