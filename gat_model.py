import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import GATv2Conv


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2, heads=2, use_sigmoid=False):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.dropout = dropout

        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=False, dropout=dropout)
        self.gat2 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=dropout)
        self.gat3 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=dropout)


        #self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        #self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        #self.gat3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=False, dropout=dropout)
        
        self.fc_out = nn.Linear(hidden_channels, out_channels)

    # def forward(self, data):
    #     x, edge_index = data.x, data.edge_index
    #     batch = getattr(data, 'batch', None)

    #     #x = F.elu(self.gat1(x, edge_index))
    #     x = F.relu(self.gat1(x, edge_index))
    #     x = F.dropout(x, p=self.dropout, training=self.training)

    #     #x = F.elu(self.gat2(x, edge_index))
    #     x = F.relu(self.gat2(x, edge_index))
    #     x = F.dropout(x, p=self.dropout, training=self.training)

    #     #x = F.elu(self.gat3(x, edge_index))
    #     x = F.relu(self.gat3(x, edge_index))
    #     x = self.fc_out(x)

    #     if self.use_sigmoid:
    #         x = torch.sigmoid(x)


    #     # Extract root node output
    #     node_types = data.x[:, 0]  
    #     root_mask = (node_types == 0) 
        
    #     if batch is not None:
    #         root_outputs = []
    #         for i in range(batch.max().item() + 1):
    #             graph_mask = (batch == i)
    #             graph_root_mask = graph_mask & root_mask
    #             root_indices = graph_root_mask.nonzero(as_tuple=False).squeeze()

    #             if root_indices.numel() == 0:
    #                 raise ValueError(f"No root node found in graph {i}")
    #             if root_indices.dim() > 0 and root_indices.shape[0] > 1:
    #                 root_idx = root_indices[0]
    #             else:
    #                 root_idx = root_indices
                
    #             root_outputs.append(x[root_idx])
    #         return torch.stack(root_outputs)
    #     else:
    #         root_indices = root_mask.nonzero(as_tuple=False).squeeze()
    #         if root_indices.numel() == 0:
    #             raise ValueError("No root node found")
    #         return x[root_indices[0]].unsqueeze(0)

    def forward(self, data, debug_attention=True):
        x, edge_index = data.x, data.edge_index

        # Layer 1 (get attention only if debugging)
        gat1_args = (x, edge_index)
        if debug_attention:
            x, (_, att1) = self.gat1(*gat1_args, return_attention_weights=True)
        else:
            x = self.gat1(*gat1_args)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        if debug_attention:
            x, (_, att2) = self.gat2(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3
        if debug_attention:
            x, (_, att3) = self.gat3(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat3(x, edge_index)
        x = F.relu(x)
        x = self.fc_out(x)
        if self.use_sigmoid:
            x = torch.sigmoid(x)

        # Extract root node output (your existing code)

        node_types = data.x[:, 0]  
        root_mask = (node_types == 0) 
        batch = getattr(data, 'batch', None)
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
            outs = torch.stack(root_outputs)
        else:
            root_indices = root_mask.nonzero(as_tuple=False).squeeze()
            if root_indices.numel() == 0:
                raise ValueError("No root node found")
            outs = x[root_indices[0]].unsqueeze(0)

        if debug_attention:
            return outs, {'layer1': att1, 'layer2': att2, 'layer3': att3}
        else:
            return outs
