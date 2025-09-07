# src/models/rgcn_baseline.py
import torch
from torch import nn
from torch_geometric.nn import RGCNConv

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

    def forward(self, x, edge_index, edge_type):
        # RGCNConv expects edge_index and edge_type
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_type)
            x = torch.relu(x)
        x = self.convs[-1](x, edge_index, edge_type)
        return x
