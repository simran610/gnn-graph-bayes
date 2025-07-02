import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2, use_sigmoid=False):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        self.fc_out = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    # def forward(self, data):
    #     x, edge_index = data.x, data.edge_index
    #     batch = data.batch if hasattr(data, 'batch') else None

    #     x = F.dropout(x, p=self.dropout, training=self.training)
    #     x = self.sage1(x, edge_index)
    #     x = F.relu(x)
    #     x = F.dropout(x, p=self.dropout, training=self.training)
    #     x = self.sage2(x, edge_index)
        
    #     # Extract root node indices (node_type == 0)
    #     node_types = data.x[:, 0]
    #     root_mask = (node_types == 0)
    #     root_indices = root_mask.nonzero(as_tuple=False).squeeze()

    #     if batch is not None and batch.numel() > 0:
    #         roots_per_graph = []
    #         for i in range(batch.max().item() + 1):
    #             idx = ((batch == i) & (node_types == 0)).nonzero(as_tuple=False)
    #             if idx.numel() == 0:
    #                 raise ValueError(f"No root node found in graph {i}")
    #             roots_per_graph.append(idx[0].item())
    #         root_indices = torch.tensor(roots_per_graph, device=x.device)

    #     x = x[root_indices]
    #     x = self.fc_out(x)  # <== this line is critical
    #     if self.use_sigmoid:
    #         x = torch.sigmoid(x)
    #     return x
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
      
        x = F.relu(self.sage1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.sage2(x, edge_index))
        x = self.fc_out(x)
        
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        
        # Extract root node output
        node_types = data.x[:, 0]  
        root_mask = (node_types == 0) 
        
        if batch is not None:
            # Handle batched graphs
            root_outputs = []
            for i in range(batch.max().item() + 1):
                graph_mask = (batch == i)
                graph_root_mask = graph_mask & root_mask
                root_indices = graph_root_mask.nonzero(as_tuple=False).squeeze()
                
                if root_indices.numel() == 0:
                    raise ValueError(f"No root node found in graph {i}")
                
                # Take first root node if multiple (shouldn't happen in tree)
                if root_indices.dim() > 0 and root_indices.shape[0] > 1:
                    root_idx = root_indices[0]
                else:
                    root_idx = root_indices
                
                root_outputs.append(x[root_idx])
            
            return torch.stack(root_outputs)  
        else:
            # Single graph case
            root_indices = root_mask.nonzero(as_tuple=False).squeeze()
            if root_indices.numel() == 0:
                raise ValueError("No root node found")
            return x[root_indices[0]].unsqueeze(0)  

