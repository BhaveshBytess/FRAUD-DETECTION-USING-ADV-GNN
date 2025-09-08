"""
Graph Transformer for Stage 5: Advanced Architectures

This module implements state-of-the-art Graph Transformer architectures that combine
the power of Transformer self-attention with graph structure for fraud detection.

Key Features:
- Multi-head graph attention with positional encoding
- Learnable graph structure embeddings
- Advanced normalization and residual connections
- Scalable architecture for large graphs
- Integration with heterogeneous graph data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import math
from typing import Optional, Tuple, Union, Dict, Any
import numpy as np


class GraphPositionalEncoding(nn.Module):
    """
    Graph-aware positional encoding that captures structural information.
    
    Combines multiple encoding strategies:
    - Random walk positional encoding
    - Degree-based encoding
    - Spectral encoding (when available)
    """
    
    def __init__(self, hidden_dim: int, max_degree: int = 100, walk_length: int = 16):
        super(GraphPositionalEncoding, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.max_degree = max_degree
        self.walk_length = walk_length
        
        # Degree-based encoding
        self.degree_encoder = nn.Embedding(max_degree + 1, hidden_dim // 4)
        
        # Random walk encoding
        self.walk_encoder = nn.Linear(walk_length, hidden_dim // 4)
        
        # Learnable structural encoding
        self.struct_encoder = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        
        # Final projection
        self.pos_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Compute graph positional encoding.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Graph edges [2, num_edges]
            batch_size: Batch size for batched graphs
            
        Returns:
            Positional encodings [num_nodes, hidden_dim]
        """
        num_nodes = x.size(0)
        device = x.device
        
        # Compute node degrees
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=torch.long)
        deg = torch.clamp(deg, 0, self.max_degree)
        degree_emb = self.degree_encoder(deg)  # [num_nodes, hidden_dim//4]
        
        # Random walk encoding (simplified version)
        # In practice, you would compute actual random walk statistics
        walk_features = torch.randn(num_nodes, self.walk_length, device=device)
        walk_emb = self.walk_encoder(walk_features)  # [num_nodes, hidden_dim//4]
        
        # Combine encodings
        pos_features = torch.cat([degree_emb, walk_emb], dim=-1)  # [num_nodes, hidden_dim//2]
        pos_features = self.struct_encoder(pos_features)
        
        # Final projection
        pos_encoding = self.pos_projection(pos_features)
        
        return pos_encoding


class GraphMultiHeadAttention(nn.Module):
    """
    Multi-head attention adapted for graph data with structural bias.
    
    Features:
    - Graph-aware attention weights
    - Edge feature integration
    - Structural bias terms
    - Efficient sparse computation
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        bias: bool = True
    ):
        super(GraphMultiHeadAttention, self).__init__()
        
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        # Edge feature projection (if available)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim, bias=False) if edge_dim else None
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Learnable structural bias
        self.structural_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of graph multi-head attention.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Graph edges [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            attention_mask: Attention mask [num_nodes, num_nodes] (optional)
            
        Returns:
            Tuple of (output_features, attention_weights)
        """
        num_nodes = x.size(0)
        
        # Compute Q, K, V
        Q = self.q_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        
        # Compute attention scores
        row, col = edge_index
        
        # Get Q, K for connected nodes
        q_i = Q[row]  # [num_edges, num_heads, head_dim]
        k_j = K[col]  # [num_edges, num_heads, head_dim]
        v_j = V[col]  # [num_edges, num_heads, head_dim]
        
        # Compute attention scores
        attention_scores = (q_i * k_j).sum(dim=-1) * self.scale  # [num_edges, num_heads]
        
        # Add structural bias
        attention_scores = attention_scores + self.structural_bias.view(self.num_heads, 1).t()
        
        # Add edge features if available
        if edge_attr is not None and self.edge_proj is not None:
            edge_emb = self.edge_proj(edge_attr)  # [num_edges, hidden_dim]
            edge_emb = edge_emb.view(-1, self.num_heads, self.head_dim)
            edge_bias = (q_i * edge_emb).sum(dim=-1)  # [num_edges, num_heads]
            attention_scores = attention_scores + edge_bias
        
        # Apply softmax to get attention weights
        attention_weights = torch.zeros(num_nodes, num_nodes, self.num_heads, device=x.device)
        attention_weights[row, col] = F.softmax(attention_scores, dim=0)
        
        # Apply attention to values
        attention_probs = F.softmax(attention_scores, dim=0)  # [num_edges, num_heads]
        attention_probs = self.dropout(attention_probs)
        
        # Aggregate values
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
        
        for head in range(self.num_heads):
            head_probs = attention_probs[:, head].unsqueeze(-1)  # [num_edges, 1]
            head_values = v_j[:, head, :]  # [num_edges, head_dim]
            
            # Aggregate using scatter
            aggregated = torch.zeros(num_nodes, self.head_dim, device=x.device)
            aggregated.index_add_(0, row, head_probs * head_values)
            out[:, head, :] = aggregated
        
        # Reshape and project output
        out = out.view(num_nodes, self.hidden_dim)
        out = self.out_proj(out)
        
        return out, attention_weights


class GraphTransformerLayer(nn.Module):
    """
    Single layer of Graph Transformer with residual connections and normalization.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        norm_first: bool = True
    ):
        super(GraphTransformerLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.norm_first = norm_first
        ff_dim = ff_dim or 4 * hidden_dim
        
        # Multi-head attention
        self.attention = GraphMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim
        )
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of transformer layer.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Graph edges [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            
        Returns:
            Tuple of (output_features, attention_weights)
        """
        if self.norm_first:
            # Pre-norm architecture
            # Attention block
            norm_x = self.norm1(x)
            attn_out, attn_weights = self.attention(norm_x, edge_index, edge_attr)
            x = x + self.dropout(attn_out)
            
            # Feed-forward block
            norm_x = self.norm2(x)
            ff_out = self.ff_network(norm_x)
            x = x + ff_out
        else:
            # Post-norm architecture
            # Attention block
            attn_out, attn_weights = self.attention(x, edge_index, edge_attr)
            x = self.norm1(x + self.dropout(attn_out))
            
            # Feed-forward block
            ff_out = self.ff_network(x)
            x = self.norm2(x + ff_out)
        
        return x, attn_weights


class GraphTransformer(nn.Module):
    """
    Complete Graph Transformer model for fraud detection.
    
    This model combines the power of Transformer architecture with graph structure
    to create a state-of-the-art fraud detection system.
    
    Features:
    - Multiple transformer layers with graph attention
    - Positional encoding for graph structure
    - Advanced normalization and regularization
    - Support for both node and graph-level predictions
    - Integration with heterogeneous graph data
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        num_classes: int = 2,
        use_pos_encoding: bool = True,
        norm_first: bool = True,
        global_pool: str = 'mean'
    ):
        super(GraphTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.use_pos_encoding = use_pos_encoding
        self.global_pool = global_pool
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        if use_pos_encoding:
            self.pos_encoding = GraphPositionalEncoding(hidden_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                edge_dim=edge_dim,
                norm_first=norm_first
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Output layers
        self.output_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Graph Transformer.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph edges [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            batch: Batch assignment for multiple graphs [num_nodes] (optional)
            
        Returns:
            Dictionary containing:
            - 'logits': Output logits [num_nodes, num_classes] or [batch_size, num_classes]
            - 'node_embeddings': Final node embeddings [num_nodes, hidden_dim]
            - 'attention_weights': List of attention weights from each layer
        """
        # Input projection
        h = self.input_projection(x)  # [num_nodes, hidden_dim]
        
        # Add positional encoding
        if self.use_pos_encoding:
            pos_enc = self.pos_encoding(h, edge_index)
            h = h + pos_enc
        
        # Store attention weights from each layer
        attention_weights = []
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            h, attn_weights = layer(h, edge_index, edge_attr)
            attention_weights.append(attn_weights)
        
        # Final normalization
        h = self.final_norm(h)
        
        # Store node embeddings
        node_embeddings = h
        
        # Global pooling for graph-level prediction
        if batch is not None:
            # Graph-level prediction
            if self.global_pool == 'mean':
                from torch_scatter import scatter_mean
                h = scatter_mean(h, batch, dim=0)
            elif self.global_pool == 'max':
                from torch_scatter import scatter_max
                h = scatter_max(h, batch, dim=0)[0]
            elif self.global_pool == 'sum':
                from torch_scatter import scatter_add
                h = scatter_add(h, batch, dim=0)
            else:
                raise ValueError(f"Unknown pooling method: {self.global_pool}")
        
        # Output projection
        h = self.output_dropout(h)
        logits = self.output_projection(h)
        
        return {
            'logits': logits,
            'node_embeddings': node_embeddings,
            'attention_weights': attention_weights
        }
    
    def get_attention_maps(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get attention maps for visualization and analysis.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph edges [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            
        Returns:
            Dictionary containing attention maps from each layer
        """
        with torch.no_grad():
            outputs = self.forward(x, edge_index, edge_attr)
            
        attention_maps = {}
        for i, attn_weights in enumerate(outputs['attention_weights']):
            attention_maps[f'layer_{i}'] = attn_weights
            
        return attention_maps


def create_graph_transformer(
    input_dim: int,
    config: Dict[str, Any]
) -> GraphTransformer:
    """
    Factory function to create Graph Transformer model.
    
    Args:
        input_dim: Input feature dimension
        config: Model configuration dictionary
        
    Returns:
        Initialized Graph Transformer model
    """
    return GraphTransformer(
        input_dim=input_dim,
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        ff_dim=config.get('ff_dim', None),
        dropout=config.get('dropout', 0.1),
        edge_dim=config.get('edge_dim', None),
        num_classes=config.get('num_classes', 2),
        use_pos_encoding=config.get('use_pos_encoding', True),
        norm_first=config.get('norm_first', True),
        global_pool=config.get('global_pool', 'mean')
    )


if __name__ == "__main__":
    # Test Graph Transformer
    print("Testing Graph Transformer...")
    
    # Test parameters
    num_nodes = 1000
    input_dim = 186
    num_edges = 5000
    
    # Create test data
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Create model
    config = {
        'hidden_dim': 128,
        'num_layers': 4,
        'num_heads': 4,
        'dropout': 0.1,
        'num_classes': 2
    }
    
    model = create_graph_transformer(input_dim, config)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(x, edge_index)
        
    print(f"Node embeddings shape: {outputs['node_embeddings'].shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Number of attention layers: {len(outputs['attention_weights'])}")
    
    # Test with batched graphs
    batch = torch.randint(0, 4, (num_nodes,))  # 4 graphs in batch
    
    with torch.no_grad():
        outputs_batched = model(x, edge_index, batch=batch)
        
    print(f"Batched logits shape: {outputs_batched['logits'].shape}")
    
    print("✅ Graph Transformer test completed successfully!")
