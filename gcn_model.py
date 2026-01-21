"""
GCN Model Module

Graph Convolutional Network (GCN) implementation for Bayesian network inference.
Uses spectral convolution operations with batch normalization for stable training.
Supports both raw probability and log-probability output modes.

Architecture:
    - 4 GCN layers with batch normalization
    - Output layer: Linear projection to target dimension
    - Activation: ReLU between layers, configurable output activation

Modes:
    - root_probability: Predict single root node probability
    - distribution: Predict full probability distribution
    - regression: Regression task on continuous targets
"""

# gcn_model.py
# GCN with 4 layers matching GraphSAGE structure
# Supports both raw probability and log-probability modes

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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