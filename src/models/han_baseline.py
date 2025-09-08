# src/models/han_baseline.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv, Linear

class SimpleHAN(nn.Module):
    """
    Heterogeneous Attention Network (HAN) baseline for fraud detection.
    
    This model uses HANConv layers which implement hierarchical attention
    (node-level and semantic-level) for heterogeneous graphs.
    """
    
    def __init__(self, metadata, hidden_dim=64, out_dim=1, num_heads=4, num_layers=2, dropout=0.1):
        """
        Initialize HAN model.
        
        Args:
            metadata: Tuple containing (node_types, edge_types) from HeteroData
            hidden_dim: Hidden dimension for all node types
            out_dim: Output dimension (1 for binary classification) 
            num_heads: Number of attention heads
            num_layers: Number of HAN layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.node_types, self.edge_types = metadata
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input linear projections for each node type
        self.node_lin = nn.ModuleDict()
        for node_type in self.node_types:
            # We'll set input_dim dynamically based on actual data
            self.node_lin[node_type] = Linear(-1, hidden_dim)
        
        # Build meta-paths for HAN
        # For Elliptic++, we'll use meta-paths like:
        # transaction -> transaction (direct)
        # transaction -> wallet -> transaction (through wallet)
        self.metapaths = self._build_metapaths()
        
        # HAN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HANConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=num_heads,
                dropout=dropout
            )
            self.convs.append(conv)
        
        # Output classifier for transaction nodes
        self.classifier = Linear(hidden_dim, out_dim)
        
    def _build_metapaths(self):
        """
        Build meta-paths for the heterogeneous graph.
        Meta-paths define the sequence of node and edge types to consider.
        """
        metapaths = []
        
        # Direct transaction connections (if they exist)
        if ('transaction', 'to', 'transaction') in self.edge_types:
            metapaths.append([('transaction', 'to', 'transaction')])
        
        # Transaction -> Wallet -> Transaction paths
        if ('transaction', 'to', 'wallet') in self.edge_types and ('wallet', 'to', 'transaction') in self.edge_types:
            metapaths.append([('transaction', 'to', 'wallet'), ('wallet', 'to', 'transaction')])
        
        # If no specific paths found, create a default direct path
        if not metapaths and self.edge_types:
            # Use the first available edge type as default
            first_edge = self.edge_types[0] 
            metapaths.append([first_edge])
        
        return metapaths
    
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass through HAN.
        
        Args:
            x_dict: Dictionary of node features {node_type: features}
            edge_index_dict: Dictionary of edge indices {edge_type: edge_index}
            
        Returns:
            logits: Output logits for transaction nodes
        """
        # Check if we have enough heterogeneous structure for HAN
        if len(x_dict) == 1 and len(edge_index_dict) == 0:
            # Fallback to simple linear classification for homogeneous case
            print("Warning: Using fallback mode - no heterogeneous structure detected")
            if 'transaction' in x_dict:
                x = x_dict['transaction']
                x = torch.nan_to_num(x, nan=0.0)
                # Use first linear layer as feature extractor
                h = self.node_lin['transaction'](x)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                logits = self.classifier(h)
                return logits.squeeze(-1)
            else:
                raise ValueError("No transaction nodes found")
        
        # Normal HAN processing
        # Project input features to hidden dimension
        x_dict_proj = {}
        for node_type, x in x_dict.items():
            # Handle NaN values
            x = torch.nan_to_num(x, nan=0.0)
            x_dict_proj[node_type] = self.node_lin[node_type](x)
        
        # Apply HAN layers only if we have edges
        if len(edge_index_dict) > 0:
            for conv in self.convs:
                x_dict_proj = conv(x_dict_proj, edge_index_dict)
                # Apply dropout and activation - check for None values
                for node_type in x_dict_proj:
                    if x_dict_proj[node_type] is not None:
                        x_dict_proj[node_type] = F.relu(x_dict_proj[node_type])
                        x_dict_proj[node_type] = F.dropout(x_dict_proj[node_type], 
                                                         p=self.dropout, training=self.training)
                    else:
                        print(f"Warning: {node_type} features are None after HANConv")
        else:
            # No edges - just apply basic transformations
            for node_type in x_dict_proj:
                x_dict_proj[node_type] = F.relu(x_dict_proj[node_type])
                x_dict_proj[node_type] = F.dropout(x_dict_proj[node_type], 
                                                 p=self.dropout, training=self.training)
        
        # Get transaction node embeddings and classify
        if 'transaction' in x_dict_proj:
            tx_embeddings = x_dict_proj['transaction']
            logits = self.classifier(tx_embeddings)
            return logits.squeeze(-1)
        else:
            raise ValueError("Transaction node type not found in the graph")
            raise ValueError("Transaction node type not found in the graph")
    
    def update_input_dims(self, x_dict):
        """
        Update input linear layers based on actual feature dimensions.
        Call this once after seeing the actual data.
        """
        for node_type, x in x_dict.items():
            if node_type in self.node_lin:
                input_dim = x.size(-1)
                self.node_lin[node_type] = Linear(input_dim, self.hidden_dim)
