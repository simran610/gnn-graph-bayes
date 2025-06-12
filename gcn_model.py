# GCN model with dropout and root node extraction
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn1(x, edge_index)
        x = F.relu(x) 
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)

        # Find root nodes (type == 0)
        node_types = data.x[:, 0]
        root_mask = (node_types == 0)
        root_indices = root_mask.nonzero(as_tuple=False).squeeze()

        # Handle batches - select 1 root per graph
        if batch is not None and batch.numel() > 0:
            roots_per_graph = []
            for i in range(batch.max().item() + 1):
                idx = ((batch == i) & (node_types == 0)).nonzero(as_tuple=False)
                if idx.numel() == 0:
                    raise ValueError(f"No root node found in graph {i}")
                roots_per_graph.append(idx[0].item())
            root_indices = torch.tensor(roots_per_graph, device=x.device)

        return x[root_indices]
