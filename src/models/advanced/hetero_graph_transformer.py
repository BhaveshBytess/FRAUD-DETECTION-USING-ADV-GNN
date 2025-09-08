"""
Heterogeneous Graph Transformer Network (HGTN) for Stage 5

This module implements a sophisticated HGTN that combines heterogeneous graph modeling
with transformer architecture for advanced fraud detection on multi-type graphs.

Key Features:
- Multi-type node and edge modeling
- Type-specific transformations and attention
- Cross-type information propagation
- Advanced meta-path reasoning
- Scalable heterogeneous attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Any, Union
import math
from collections import defaultdict

from .graph_transformer import GraphMultiHeadAttention, GraphPositionalEncoding


class HeteroTypeEmbedding(nn.Module):
    """
    Learnable embeddings for different node and edge types in heterogeneous graphs.
    """
    
    def __init__(self, node_types: List[str], edge_types: List[str], hidden_dim: int):
        super(HeteroTypeEmbedding, self).__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        
        # Type embeddings
        self.node_type_embeddings = nn.ModuleDict({
            node_type: nn.Embedding(1, hidden_dim)
            for node_type in node_types
        })
        
        self.edge_type_embeddings = nn.ModuleDict({
            edge_type: nn.Embedding(1, hidden_dim)
            for edge_type in edge_types
        })
        
    def get_node_type_embedding(self, node_type: str) -> torch.Tensor:
        """Get embedding for a specific node type."""
        return self.node_type_embeddings[node_type](torch.tensor([0]))
    
    def get_edge_type_embedding(self, edge_type: str) -> torch.Tensor:
        """Get embedding for a specific edge type."""
        return self.edge_type_embeddings[edge_type](torch.tensor([0]))


class HeteroAttention(nn.Module):
    """
    Heterogeneous attention mechanism that handles different node/edge types.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        node_types: List[str],
        edge_types: List[str],
        dropout: float = 0.1
    ):
        super(HeteroAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.node_types = node_types
        self.edge_types = edge_types
        
        # Type-specific projections
        self.node_projections = nn.ModuleDict()
        for node_type in node_types:
            self.node_projections[node_type] = nn.ModuleDict({
                'q': nn.Linear(hidden_dim, hidden_dim),
                'k': nn.Linear(hidden_dim, hidden_dim),
                'v': nn.Linear(hidden_dim, hidden_dim)
            })
        
        # Edge type projections
        self.edge_projections = nn.ModuleDict({
            edge_type: nn.Linear(hidden_dim, hidden_dim)
            for edge_type in edge_types
        })
        
        # Cross-type attention weights
        self.cross_type_weights = nn.ParameterDict({
            f"{src_type}_{edge_type}_{dst_type}": nn.Parameter(torch.randn(num_heads))
            for src_type in node_types
            for dst_type in node_types
            for edge_type in edge_types
        })
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[str, torch.Tensor],
        edge_attr_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of heterogeneous attention.
        
        Args:
            x_dict: Node features for each type {node_type: [num_nodes, hidden_dim]}
            edge_index_dict: Edge indices for each type {edge_type: [2, num_edges]}
            edge_attr_dict: Edge features for each type {edge_type: [num_edges, hidden_dim]}
            
        Returns:
            Updated node features for each type
        """
        output_dict = {}
        
        for node_type in self.node_types:
            if node_type not in x_dict:
                continue
                
            x = x_dict[node_type]
            num_nodes = x.size(0)
            
            # Compute Q, K, V for this node type
            q = self.node_projections[node_type]['q'](x)
            q = q.view(num_nodes, self.num_heads, self.head_dim)
            
            # Collect attention from all relevant edge types
            aggregated_attention = torch.zeros_like(x)
            total_attention_weights = torch.zeros(num_nodes, device=x.device)
            
            for edge_type, edge_index in edge_index_dict.items():
                # Parse edge type (e.g., "user__to__transaction")
                if '__to__' in edge_type:
                    src_type, dst_type = edge_type.split('__to__')
                else:
                    # Assume homogeneous edge if format not recognized
                    src_type = dst_type = node_type
                
                # Skip if this edge type doesn't involve current node type
                if dst_type != node_type:
                    continue
                
                if src_type not in x_dict:
                    continue
                
                src_x = x_dict[src_type]
                
                # Compute K, V from source nodes
                k = self.node_projections[src_type]['k'](src_x)
                v = self.node_projections[src_type]['v'](src_x)
                
                k = k.view(src_x.size(0), self.num_heads, self.head_dim)
                v = v.view(src_x.size(0), self.num_heads, self.head_dim)
                
                # Get edge connections
                row, col = edge_index  # row: source, col: destination
                
                if len(row) == 0:
                    continue
                
                # Get Q, K, V for connected nodes
                q_dst = q[col]  # [num_edges, num_heads, head_dim]
                k_src = k[row]  # [num_edges, num_heads, head_dim]
                v_src = v[row]  # [num_edges, num_heads, head_dim]
                
                # Compute attention scores
                attention_scores = (q_dst * k_src).sum(dim=-1) * self.scale  # [num_edges, num_heads]
                
                # Add cross-type weights
                cross_type_key = f"{src_type}_{edge_type}_{dst_type}"
                if cross_type_key in self.cross_type_weights:
                    type_weights = self.cross_type_weights[cross_type_key]
                    attention_scores = attention_scores + type_weights.unsqueeze(0)
                
                # Add edge features if available
                if edge_attr_dict and edge_type in edge_attr_dict:
                    edge_features = edge_attr_dict[edge_type]
                    if edge_type in self.edge_projections:
                        edge_emb = self.edge_projections[edge_type](edge_features)
                        edge_emb = edge_emb.view(-1, self.num_heads, self.head_dim)
                        edge_bias = (q_dst * edge_emb).sum(dim=-1)
                        attention_scores = attention_scores + edge_bias
                
                # Apply softmax
                attention_weights = F.softmax(attention_scores, dim=0)  # [num_edges, num_heads]
                attention_weights = self.dropout(attention_weights)
                
                # Apply attention to values
                attended_values = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
                
                for head in range(self.num_heads):
                    head_weights = attention_weights[:, head].unsqueeze(-1)
                    head_values = v_src[:, head, :]
                    
                    # Aggregate using scatter
                    head_aggregated = torch.zeros(num_nodes, self.head_dim, device=x.device)
                    head_aggregated.index_add_(0, col, head_weights * head_values)
                    attended_values[:, head, :] = head_aggregated
                
                # Reshape and add to aggregated attention
                attended_values = attended_values.view(num_nodes, self.hidden_dim)
                aggregated_attention += attended_values
                
                # Track attention weights for normalization
                edge_attention_weights = torch.zeros(num_nodes, device=x.device)
                edge_attention_weights.index_add_(0, col, attention_weights.mean(dim=1))
                total_attention_weights += edge_attention_weights
            
            # Normalize aggregated attention
            total_attention_weights = torch.clamp(total_attention_weights, min=1e-8)
            aggregated_attention = aggregated_attention / total_attention_weights.unsqueeze(-1)
            
            # Apply output projection
            output = self.output_projection(aggregated_attention)
            output_dict[node_type] = output
        
        return output_dict


class HeteroTransformerLayer(nn.Module):
    """
    Single layer of Heterogeneous Graph Transformer.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        node_types: List[str],
        edge_types: List[str],
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        norm_first: bool = True
    ):
        super(HeteroTransformerLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.norm_first = norm_first
        ff_dim = ff_dim or 4 * hidden_dim
        
        self.node_types = node_types
        self.edge_types = edge_types
        
        # Heterogeneous attention
        self.hetero_attention = HeteroAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            node_types=node_types,
            edge_types=edge_types,
            dropout=dropout
        )
        
        # Type-specific feed-forward networks
        self.ff_networks = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(hidden_dim, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, hidden_dim),
                nn.Dropout(dropout)
            )
            for node_type in node_types
        })
        
        # Type-specific layer normalization
        self.norm1 = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim)
            for node_type in node_types
        })
        
        self.norm2 = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim)
            for node_type in node_types
        })
        
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[str, torch.Tensor],
        edge_attr_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of heterogeneous transformer layer.
        """
        output_dict = {}
        
        if self.norm_first:
            # Pre-norm architecture
            norm_x_dict = {
                node_type: self.norm1[node_type](x)
                for node_type, x in x_dict.items()
                if node_type in self.norm1
            }
            
            # Attention
            attn_dict = self.hetero_attention(norm_x_dict, edge_index_dict, edge_attr_dict)
            
            # Residual connection
            for node_type in x_dict.keys():
                if node_type in attn_dict:
                    x_dict[node_type] = x_dict[node_type] + attn_dict[node_type]
            
            # Feed-forward
            for node_type, x in x_dict.items():
                if node_type in self.norm2:
                    norm_x = self.norm2[node_type](x)
                    ff_out = self.ff_networks[node_type](norm_x)
                    output_dict[node_type] = x + ff_out
                else:
                    output_dict[node_type] = x
        else:
            # Post-norm architecture
            attn_dict = self.hetero_attention(x_dict, edge_index_dict, edge_attr_dict)
            
            # Attention + residual + norm
            for node_type in x_dict.keys():
                if node_type in attn_dict and node_type in self.norm1:
                    x_dict[node_type] = self.norm1[node_type](x_dict[node_type] + attn_dict[node_type])
            
            # Feed-forward + residual + norm
            for node_type, x in x_dict.items():
                if node_type in self.ff_networks and node_type in self.norm2:
                    ff_out = self.ff_networks[node_type](x)
                    output_dict[node_type] = self.norm2[node_type](x + ff_out)
                else:
                    output_dict[node_type] = x
        
        return output_dict


class HeterogeneousGraphTransformer(nn.Module):
    """
    Complete Heterogeneous Graph Transformer Network (HGTN) for fraud detection.
    
    This model handles heterogeneous graphs with multiple node and edge types,
    using type-aware attention mechanisms and transformations.
    
    Features:
    - Multi-type node and edge handling
    - Type-specific transformations
    - Cross-type attention mechanisms
    - Meta-path reasoning capabilities
    - Advanced heterogeneous graph modeling
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        num_classes: int = 2,
        target_node_type: str = 'transaction',
        norm_first: bool = True,
        use_type_embeddings: bool = True
    ):
        super(HeterogeneousGraphTransformer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.target_node_type = target_node_type
        self.use_type_embeddings = use_type_embeddings
        
        # Extract node and edge types from input
        self.node_types = list(input_dims.keys())
        
        # Input projections for each node type
        self.input_projections = nn.ModuleDict({
            node_type: nn.Linear(input_dim, hidden_dim)
            for node_type, input_dim in input_dims.items()
        })
        
        # Type embeddings
        if use_type_embeddings:
            # We'll infer edge types during forward pass
            self.type_embeddings = None  # Will be initialized when we see the first batch
        
        # Transformer layers (will be initialized when we know edge types)
        self.transformer_layers = nn.ModuleList()
        
        # Output layers for target node type
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Store configuration for lazy initialization
        self.config = {
            'num_heads': num_heads,
            'ff_dim': ff_dim,
            'dropout': dropout,
            'norm_first': norm_first
        }
        
        self._initialized = False
        
    def _initialize_layers(self, edge_types: List[str]):
        """Initialize layers after seeing the first batch to get edge types."""
        if self._initialized:
            return
            
        self.edge_types = edge_types
        
        # Initialize type embeddings
        if self.use_type_embeddings:
            self.type_embeddings = HeteroTypeEmbedding(
                self.node_types, edge_types, self.hidden_dim
            )
        
        # Initialize transformer layers
        for _ in range(self.num_layers):
            layer = HeteroTransformerLayer(
                hidden_dim=self.hidden_dim,
                node_types=self.node_types,
                edge_types=edge_types,
                **self.config
            )
            self.transformer_layers.append(layer)
        
        self._initialized = True
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[str, torch.Tensor],
        edge_attr_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of HGTN.
        
        Args:
            x_dict: Node features for each type {node_type: [num_nodes, input_dim]}
            edge_index_dict: Edge indices for each type {edge_type: [2, num_edges]}
            edge_attr_dict: Edge features for each type {edge_type: [num_edges, hidden_dim]}
            
        Returns:
            Dictionary containing:
            - 'logits': Output logits for target node type
            - 'node_embeddings': Final embeddings for all node types
        """
        # Initialize layers if needed
        if not self._initialized:
            edge_types = list(edge_index_dict.keys())
            self._initialize_layers(edge_types)
        
        # Project input features to hidden dimension
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.input_projections:
                h = self.input_projections[node_type](x)
                
                # Add type embeddings if enabled
                if self.use_type_embeddings and self.type_embeddings:
                    type_emb = self.type_embeddings.get_node_type_embedding(node_type)
                    h = h + type_emb.expand_as(h)
                
                h_dict[node_type] = h
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            h_dict = layer(h_dict, edge_index_dict, edge_attr_dict)
        
        # Final normalization
        for node_type in h_dict:
            h_dict[node_type] = self.output_norm(h_dict[node_type])
        
        # Generate predictions for target node type
        logits = None
        if self.target_node_type in h_dict:
            target_embeddings = h_dict[self.target_node_type]
            target_embeddings = self.output_dropout(target_embeddings)
            logits = self.output_projection(target_embeddings)
        
        return {
            'logits': logits,
            'node_embeddings': h_dict
        }


def create_heterogeneous_graph_transformer(
    input_dims: Dict[str, int],
    config: Dict[str, Any]
) -> HeterogeneousGraphTransformer:
    """
    Factory function to create Heterogeneous Graph Transformer.
    
    Args:
        input_dims: Input dimensions for each node type
        config: Model configuration dictionary
        
    Returns:
        Initialized HGTN model
    """
    return HeterogeneousGraphTransformer(
        input_dims=input_dims,
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 4),
        num_heads=config.get('num_heads', 8),
        ff_dim=config.get('ff_dim', None),
        dropout=config.get('dropout', 0.1),
        num_classes=config.get('num_classes', 2),
        target_node_type=config.get('target_node_type', 'transaction'),
        norm_first=config.get('norm_first', True),
        use_type_embeddings=config.get('use_type_embeddings', True)
    )


if __name__ == "__main__":
    # Test Heterogeneous Graph Transformer
    print("Testing Heterogeneous Graph Transformer...")
    
    # Create test heterogeneous data
    num_transactions = 1000
    num_wallets = 500
    
    x_dict = {
        'transaction': torch.randn(num_transactions, 186),
        'wallet': torch.randn(num_wallets, 64)
    }
    
    edge_index_dict = {
        'wallet__to__transaction': torch.randint(0, min(num_wallets, num_transactions), (2, 2000)),
        'transaction__to__wallet': torch.randint(0, min(num_transactions, num_wallets), (2, 1500)),
        'transaction__to__transaction': torch.randint(0, num_transactions, (2, 3000))
    }
    
    # Model configuration
    input_dims = {'transaction': 186, 'wallet': 64}
    config = {
        'hidden_dim': 128,
        'num_layers': 3,
        'num_heads': 4,
        'dropout': 0.1,
        'target_node_type': 'transaction'
    }
    
    # Create model
    model = create_heterogeneous_graph_transformer(input_dims, config)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(x_dict, edge_index_dict)
    
    print(f"Transaction logits shape: {outputs['logits'].shape}")
    print(f"Transaction embeddings shape: {outputs['node_embeddings']['transaction'].shape}")
    print(f"Wallet embeddings shape: {outputs['node_embeddings']['wallet'].shape}")
    
    print("✅ Heterogeneous Graph Transformer test completed successfully!")
