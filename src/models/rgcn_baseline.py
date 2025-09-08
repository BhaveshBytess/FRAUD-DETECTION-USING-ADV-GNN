# src/models/rgcn_baseline.py
import torch
from torch import nn
from torch_geometric.nn import RGCNConv, Linear

class SimpleRGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(RGCNConv(in_dim, out_dim, num_relations))
        else:
            self.convs.append(RGCNConv(in_dim, hidden_dim, num_relations))
            for _ in range(num_layers - 2):
                self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))
            self.convs.append(RGCNConv(hidden_dim, out_dim, num_relations))

        # Fallback linear layers for no-edge case
        self.lin1 = Linear(in_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_type):
        # If there are no edges, just use linear layers (MLP)
        if edge_index is None or edge_index.numel() == 0:
            x = self.lin1(x).relu()
            x = self.lin2(x)
            return x

        # RGCNConv expects edge_index and edge_type
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_type)
            x = torch.relu(x)
        x = self.convs[-1](x, edge_index, edge_type)
        return x
