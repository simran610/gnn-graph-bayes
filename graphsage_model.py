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
        self.sage3 = SAGEConv(hidden_channels, hidden_channels)
        self.sage4 = SAGEConv(hidden_channels, hidden_channels)
        self.fc_out = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)

        print("Input node features (sample):", x[:5])

       # 1st layer
        x = F.relu(self.sage1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        #print("After conv1:", x[:5])

        # 2nd layer
        x = F.relu(self.sage2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        #print("After conv2:", x[:5])
        
        # 3rd layer
        x = F.relu(self.sage3(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 4th layer 
        x = F.relu(self.sage4(x, edge_index))
        x = self.fc_out(x)

        #print("After conv3:", x[:5])
        
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


