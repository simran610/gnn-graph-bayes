# gcn_model.py
# GCN with 4 layers matching GraphSAGE structure
# Supports both raw probability and log-probability modes

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# class GCN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2, 
#                  mode="root_probability", use_log_prob=False):
#         """
#         GCN model for Bayesian Network inference.
        
#         Args:
#             in_channels: Number of input features
#             hidden_channels: Number of hidden features
#             out_channels: Number of output features
#             dropout: Dropout probability
#             mode: "root_probability", "distribution", or "regression"
#             use_log_prob: If True, outputs log-probabilities (negative values)
#         """
#         super().__init__()
#         self.mode = mode
#         self.use_log_prob = use_log_prob
        
#         # GCN layers
#         self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=False)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
#         self.conv3 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
#         self.conv4 = GCNConv(hidden_channels, hidden_channels, )
        
#         # Output layer
#         self.fc_out = nn.Linear(hidden_channels, out_channels)
#         self.dropout = dropout
        
#         print(f"GCN initialized: mode={mode}, use_log_prob={use_log_prob}")
    
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         batch = getattr(data, 'batch', None)

#         # Uncomment for debugging:
#         # print("Input node features (sample):", x[:5])

#         # 1st layer
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         # 2nd layer
#         x = F.relu(self.conv2(x, edge_index))
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         # Uncomment for debugging:
#         # print("After conv2:", x[:5])
        
#         # 3rd layer
#         x = F.relu(self.conv3(x, edge_index))
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         # 4th layer 
#         x = F.relu(self.conv4(x, edge_index))
#         x = self.fc_out(x)

#         # Apply appropriate output activation based on mode
#         if self.mode == "root_probability":
#             if self.use_log_prob:
#                 # For log-prob: force output to be negative
#                 # -softplus ensures output is in range (-∞, 0]
#                 # x = -F.softplus(x)
#                 # Clamp output to reasonable log-prob range
#                 # log(0.0001) ≈ -9.21, log(0.9999) ≈ -0.0001
#                 x = torch.clamp(x, min=-9.0, max=-0.0001)
#             else:
#                 # For raw prob: force output to be in [0, 1]
#                 x = torch.sigmoid(x)
#         elif self.mode == "distribution":
#             # No activation - will use log_softmax in loss function
#             pass
#         # For regression mode, also no activation needed
        
#         # Extract root node output
#         node_types = data.x[:, 0]  
#         root_mask = (node_types == 0) 
        
#         if batch is not None:
#             # Handle batched graphs
#             root_outputs = []
#             for i in range(batch.max().item() + 1):
#                 graph_mask = (batch == i)
#                 graph_root_mask = graph_mask & root_mask
#                 root_indices = graph_root_mask.nonzero(as_tuple=False).squeeze()
                
#                 if root_indices.numel() == 0:
#                     raise ValueError(f"No root node found in graph {i}")
                
#                 # Take first root node if multiple (shouldn't happen in tree)
#                 if root_indices.dim() > 0 and root_indices.shape[0] > 1:
#                     root_idx = root_indices[0]
#                 else:
#                     root_idx = root_indices
                
#                 root_outputs.append(x[root_idx])
            
#             return torch.stack(root_outputs)  
#         else:
#             # Single graph case
#             root_indices = root_mask.nonzero(as_tuple=False).squeeze()
#             if root_indices.numel() == 0:
#                 raise ValueError("No root node found")
#             return x[root_indices[0]].unsqueeze(0)

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2, 
                 mode="root_probability", use_log_prob=False):
        super().__init__()
        self.mode = mode
        self.use_log_prob = use_log_prob
        
        # GCN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)  
        
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels) 
        
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)  
        
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.bn4 = nn.BatchNorm1d(hidden_channels) 
        
        self.fc_out = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        
        print(f"GCN initialized: mode={mode}, use_log_prob={use_log_prob}")
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)

        # 1st layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)  
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 2nd layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)  
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 3rd layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 4th layer 
        x = self.conv4(x, edge_index)
        x = self.bn4(x) 
        x = F.relu(x)
        x = self.fc_out(x)

        # Apply appropriate output activation based on mode
        if self.mode == "root_probability":
            if self.use_log_prob:
                x = torch.clamp(x, min=-9.0, max=-0.0001)
            else:
                x = torch.sigmoid(x)
        elif self.mode == "distribution":
            pass
        
        # Extract root node output (same as before)
        node_types = data.x[:, 0]  
        root_mask = (node_types == 0) 
        
        if batch is not None:
            root_outputs = []
            for i in range(batch.max().item() + 1):
                graph_mask = (batch == i)
                graph_root_mask = graph_mask & root_mask
                root_indices = graph_root_mask.nonzero(as_tuple=False).squeeze()
                
                if root_indices.numel() == 0:
                    raise ValueError(f"No root node found in graph {i}")
                
                if root_indices.dim() > 0 and root_indices.shape[0] > 1:
                    root_idx = root_indices[0]
                else:
                    root_idx = root_indices
                
                root_outputs.append(x[root_idx])
            
            return torch.stack(root_outputs)  
        else:
            root_indices = root_mask.nonzero(as_tuple=False).squeeze()
            if root_indices.numel() == 0:
                raise ValueError("No root node found")
            return x[root_indices[0]].unsqueeze(0)