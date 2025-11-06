
# with batchnorm and with out graph pooling


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
        # self.sage3 = SAGEConv(hidden_channels, hidden_channels)
        # self.sage4 = SAGEConv(hidden_channels, hidden_channels)
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
        # x = F.dropout(x, p=self.dropout, training=self.training)

        #print("After conv2:", x[:5])
        
        # 3rd layer
        # x = F.relu(self.sage3(x, edge_index))
        # x = F.dropout(x, p=self.dropout, training=self.training)

        # 4th layer 
        # x = F.relu(self.sage4(x, edge_index))
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

# with batchnorm and graph pooling

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool

# class GraphSAGE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
#         super().__init__()
        
#         # Graph convolution layers
#         self.sage1 = SAGEConv(in_channels, hidden_channels)
#         self.sage2 = SAGEConv(hidden_channels, hidden_channels)
#         self.sage3 = SAGEConv(hidden_channels, hidden_channels)
#         self.sage4 = SAGEConv(hidden_channels, hidden_channels)
        
#         # Batch normalization (add after each conv layer)
#         self.bn1 = nn.BatchNorm1d(hidden_channels)
#         self.bn2 = nn.BatchNorm1d(hidden_channels)
#         self.bn3 = nn.BatchNorm1d(hidden_channels)
#         self.bn4 = nn.BatchNorm1d(hidden_channels)
        
#         # Output layer: concatenate [root_features + graph_mean + graph_max]
#         self.fc_out = nn.Linear(hidden_channels * 3, out_channels)
#         self.dropout = dropout

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         batch = getattr(data, 'batch', None)
        
#         # Layer 1: Conv -> BN -> ReLU -> Dropout
#         x = self.sage1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
        
#         # Layer 2
#         x = self.sage2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
        
#         # Layer 3
#         x = self.sage3(x, edge_index)
#         x = self.bn3(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
        
#         # Layer 4 (no dropout after last layer)
#         x = self.sage4(x, edge_index)
#         x = self.bn4(x)
#         x = F.relu(x)
        
#         # Extract features
#         node_types = data.x[:, 0]
#         root_mask = (node_types == 0)
        
#         if batch is not None:
#             # Batched graphs
#             root_outputs = []
            
#             for i in range(batch.max().item() + 1):
#                 graph_mask = (batch == i)
                
#                 # 1. Get root node features
#                 graph_root_mask = graph_mask & root_mask
#                 root_indices = graph_root_mask.nonzero(as_tuple=False).squeeze()
                
#                 if root_indices.numel() == 0:
#                     raise ValueError(f"No root node found in graph {i}")
                
#                 if root_indices.dim() > 0 and root_indices.shape[0] > 1:
#                     root_idx = root_indices[0]
#                 else:
#                     root_idx = root_indices
                
#                 root_feat = x[root_idx]
                
#                 # 2. Get graph-level features (pooling over ALL nodes)
#                 graph_nodes = x[graph_mask]
#                 mean_feat = graph_nodes.mean(dim=0)  # Global mean pooling
#                 max_feat = graph_nodes.max(dim=0)[0]  # Global max pooling
                
#                 # 3. Concatenate: [root + graph_mean + graph_max]
#                 combined = torch.cat([root_feat, mean_feat, max_feat], dim=-1)
#                 root_outputs.append(combined)
            
#             x = torch.stack(root_outputs)
#         else:
#             # Single graph case
#             root_indices = root_mask.nonzero(as_tuple=False).squeeze()
#             if root_indices.numel() == 0:
#                 raise ValueError("No root node found")
            
#             root_feat = x[root_indices[0]]
#             mean_feat = x.mean(dim=0)
#             max_feat = x.max(dim=0)[0]
            
#             x = torch.cat([root_feat, mean_feat, max_feat], dim=-1).unsqueeze(0)
        
#         # Final output layer + sigmoid
#         x = self.fc_out(x)
#         x = torch.sigmoid(x)
        
#         return x


# with Layer norm and graph pooling


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool

# class GraphSAGE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
#         super().__init__()
        
#         # Graph convolution layers
#         self.sage1 = SAGEConv(in_channels, hidden_channels)
#         self.sage2 = SAGEConv(hidden_channels, hidden_channels)
#         self.sage3 = SAGEConv(hidden_channels, hidden_channels)
#         self.sage4 = SAGEConv(hidden_channels, hidden_channels)
        
#         # LayerNorm (fixes inference collapse)
#         self.ln1 = nn.LayerNorm(hidden_channels)  # ← FIX: nn.LayerNorm, not LayerNorm
#         self.ln2 = nn.LayerNorm(hidden_channels)
#         self.ln3 = nn.LayerNorm(hidden_channels)
#         self.ln4 = nn.LayerNorm(hidden_channels)
        
#         # Output layer
#         self.fc_out = nn.Linear(hidden_channels * 3, out_channels)
#         self.dropout = dropout

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         batch = getattr(data, 'batch', None)
        
#         # Layer 1
#         x = self.sage1(x, edge_index)
#         x = self.ln1(x)  # ← FIX: was self.bn1
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
        
#         # Layer 2
#         x = self.sage2(x, edge_index)
#         x = self.ln2(x)  # ← FIX: was self.bn2
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
        
#         # Layer 3
#         x = self.sage3(x, edge_index)
#         x = self.ln3(x)  # ← FIX: was self.bn3
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
        
#         # Layer 4 (no dropout after last layer)
#         x = self.sage4(x, edge_index)
#         x = self.ln4(x)  # ← FIX: was self.bn4
#         x = F.relu(x)
        
#         # Extract features
#         node_types = data.x[:, 0]
#         root_mask = (node_types == 0)
        
#         if batch is not None:
#             root_outputs = []
            
#             for i in range(batch.max().item() + 1):
#                 graph_mask = (batch == i)
                
#                 # Get root node features
#                 graph_root_mask = graph_mask & root_mask
#                 root_indices = graph_root_mask.nonzero(as_tuple=False).squeeze()
                
#                 if root_indices.numel() == 0:
#                     raise ValueError(f"No root node found in graph {i}")
                
#                 if root_indices.dim() > 0 and root_indices.shape[0] > 1:
#                     root_idx = root_indices[0]
#                 else:
#                     root_idx = root_indices
                
#                 root_feat = x[root_idx]
                
#                 # Get graph-level features
#                 graph_nodes = x[graph_mask]
#                 mean_feat = graph_nodes.mean(dim=0)
#                 max_feat = graph_nodes.max(dim=0)[0]
                
#                 # Concatenate
#                 combined = torch.cat([root_feat, mean_feat, max_feat], dim=-1)
#                 root_outputs.append(combined)
            
#             x = torch.stack(root_outputs)
#         else:
#             # Single graph case
#             root_indices = root_mask.nonzero(as_tuple=False).squeeze()
#             if root_indices.numel() == 0:
#                 raise ValueError("No root node found")
            
#             root_feat = x[root_indices[0]]
#             mean_feat = x.mean(dim=0)
#             max_feat = x.max(dim=0)[0]
            
#             x = torch.cat([root_feat, mean_feat, max_feat], dim=-1).unsqueeze(0)
        
#         # Final output
#         x = self.fc_out(x)
#         x = torch.sigmoid(x)
        
#         return x