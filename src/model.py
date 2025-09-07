import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, Linear

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, data):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: GCNConv(-1, hidden_channels) for edge_type in data.edge_types
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        # Apply the final linear layer to the 'transaction' nodes
        return self.lin(x_dict['transaction'])
