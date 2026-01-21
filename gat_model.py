"""
GAT Model Module

Graph Attention Network (GAT) implementation for Bayesian network inference.
Uses multi-head attention mechanisms to learn which graph edges are most important
for predictions. Supports both raw probability and log-probability output modes.

Architecture:
    - 4 GAT layers with multi-head attention (GATv2)
    - Output layer: Linear projection to target dimension
    - Activation: ReLU between layers, configurable output activation

Modes:
    - root_probability: Predict single root node probability
    - distribution: Predict full probability distribution
    - regression: Regression task on continuous targets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import GATv2Conv


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2, 
                 heads=2, mode="root_probability", use_log_prob=False):
        """
        GAT model for Bayesian Network inference.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features
            dropout: Dropout probability
            heads: Number of attention heads
            mode: "root_probability", "distribution", or "regression"
            use_log_prob: If True, outputs log-probabilities (negative values)
        """
        super().__init__()
        self.mode = mode
        self.use_log_prob = use_log_prob
        self.dropout = dropout

        # GAT layers
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=False)
        self.gat2 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False)
        self.gat3 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False)
        self.gat4 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_channels, out_channels)
        
        print(f"GAT initialized: mode={mode}, use_log_prob={use_log_prob}")

    def forward(self, data, debug_attention=False):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)

        # Uncomment for debugging:
        # print("Input node features (sample):", x[:5])

        # Layer 1
        if debug_attention:
            x, (_, att1) = self.gat1(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        if debug_attention:
            x, (_, att2) = self.gat2(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Uncomment for debugging:
        # print("After gat2:", x[:5])

        # Layer 3
        if debug_attention:
            x, (_, att3) = self.gat3(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 4
        if debug_attention:
            x, (_, att4) = self.gat4(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat4(x, edge_index)
        x = F.relu(x)
        x = self.fc_out(x)

        # Apply appropriate output activation based on mode
        if self.mode == "root_probability":
            if self.use_log_prob:
                # For log-prob: force output to be negative
                # -softplus ensures output is in range (-∞, 0]
                # x = -F.softplus(x)
                # Clamp output to reasonable log-prob range
                # log(0.0001) ≈ -9.21, log(0.9999) ≈ -0.0001
                x = torch.clamp(x, min=-9.0, max=-0.0001)
            else:
                # For raw prob: force output to be in [0, 1]
                x = torch.sigmoid(x)
        elif self.mode == "distribution":
            # No activation - will use log_softmax in loss function
            pass
        # For regression mode, also no activation needed

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
            
            outs = torch.stack(root_outputs)
        else:
            # Single graph case
            root_indices = root_mask.nonzero(as_tuple=False).squeeze()
            if root_indices.numel() == 0:
                raise ValueError("No root node found")
            outs = x[root_indices[0]].unsqueeze(0)

        if debug_attention:
            return outs, {'layer1': att1, 'layer2': att2, 'layer3': att3, 'layer4': att4}
        else:
            return outs