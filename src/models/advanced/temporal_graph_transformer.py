"""
Temporal Graph Transformer for Stage 5: Advanced Architectures

This module implements a sophisticated Temporal Graph Transformer that combines
temporal sequence modeling with graph attention mechanisms for fraud detection.

Key Features:
- Joint temporal and graph attention
- Multi-scale temporal reasoning
- Dynamic graph structure over time
- Temporal positional encoding
- Advanced spatio-temporal fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import math

from .graph_transformer import GraphMultiHeadAttention, GraphPositionalEncoding
from ..temporal_stable import SimpleLSTM, SimpleGRU


class TemporalPositionalEncoding(nn.Module):
    """
    Temporal positional encoding for time-aware sequence modeling.
    
    Combines multiple temporal encoding strategies:
    - Sinusoidal positional encoding
    - Learnable temporal embeddings
    - Time step aware encoding
    """
    
    def __init__(self, hidden_dim: int, max_seq_len: int = 1000, dropout: float = 0.1):
        super(TemporalPositionalEncoding, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        
        # Create sinusoidal positional encoding
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Learnable temporal embeddings
        self.temporal_embedding = nn.Parameter(torch.randn(max_seq_len, hidden_dim) * 0.1)
        
    def forward(self, x: torch.Tensor, time_steps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add temporal positional encoding to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            time_steps: Actual time steps [batch_size, seq_len] (optional)
            
        Returns:
            Encoded tensor with temporal positional information
        """
        batch_size, seq_len, _ = x.shape
        
        # Add sinusoidal encoding
        pos_encoding = self.pe[:, :seq_len, :]
        
        # Add learnable temporal embedding
        temp_encoding = self.temporal_embedding[:seq_len, :].unsqueeze(0)
        
        # Combine encodings
        x = x + pos_encoding + temp_encoding
        
        # If actual time steps are provided, add time-aware encoding
        if time_steps is not None:
            time_emb = self._encode_time_steps(time_steps)
            x = x + time_emb
        
        return self.dropout(x)
    
    def _encode_time_steps(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Encode actual time step values."""
        # Normalize time steps
        time_normalized = time_steps.float() / (time_steps.max() + 1e-8)
        
        # Create sinusoidal encoding for actual time values
        time_emb = torch.zeros(*time_steps.shape, self.hidden_dim, device=time_steps.device)
        
        for i in range(self.hidden_dim // 2):
            freq = 1.0 / (10000 ** (2 * i / self.hidden_dim))
            time_emb[:, :, 2*i] = torch.sin(time_normalized * freq)
            time_emb[:, :, 2*i + 1] = torch.cos(time_normalized * freq)
        
        return time_emb


class TemporalGraphAttention(nn.Module):
    """
    Joint temporal and graph attention mechanism.
    
    This module computes attention weights that consider both:
    1. Temporal relationships (sequence order, time proximity)
    2. Graph relationships (structural connections)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        temporal_weight: float = 0.5
    ):
        super(TemporalGraphAttention, self).__init__()
        
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.temporal_weight = temporal_weight
        
        # Temporal attention components
        self.temporal_q = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_k = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Graph attention components
        self.graph_q = nn.Linear(hidden_dim, hidden_dim)
        self.graph_k = nn.Linear(hidden_dim, hidden_dim)
        self.graph_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Fusion mechanism
        self.fusion_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
        graph_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of temporal graph attention.
        
        Args:
            x: Input features [batch_size, seq_len, hidden_dim] or [num_nodes, hidden_dim]
            edge_index: Graph edges [2, num_edges] (for graph attention)
            temporal_mask: Temporal attention mask [batch_size, seq_len, seq_len]
            graph_mask: Graph attention mask [num_nodes, num_nodes]
            
        Returns:
            Tuple of (output_features, attention_weights_dict)
        """
        if x.dim() == 3:
            # Temporal sequence input
            return self._temporal_graph_attention_sequence(x, edge_index, temporal_mask, graph_mask)
        else:
            # Graph node input
            return self._temporal_graph_attention_nodes(x, edge_index, graph_mask)
    
    def _temporal_graph_attention_sequence(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        temporal_mask: Optional[torch.Tensor],
        graph_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Handle sequence input with temporal and graph attention."""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Temporal attention
        temporal_q = self.temporal_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        temporal_k = self.temporal_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        temporal_v = self.temporal_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Compute temporal attention scores
        temporal_scores = torch.matmul(temporal_q, temporal_k.transpose(-2, -1)) * self.scale
        
        if temporal_mask is not None:
            temporal_scores = temporal_scores.masked_fill(temporal_mask == 0, float('-inf'))
        
        temporal_weights = F.softmax(temporal_scores, dim=-1)
        temporal_weights = self.dropout(temporal_weights)
        
        temporal_out = torch.matmul(temporal_weights, temporal_v)
        temporal_out = temporal_out.view(batch_size, seq_len, hidden_dim)
        
        # Graph attention (if edge_index provided)
        if edge_index is not None:
            # Flatten sequence for graph processing
            x_flat = x.view(-1, hidden_dim)  # [batch_size * seq_len, hidden_dim]
            
            graph_q = self.graph_q(x_flat).view(-1, self.num_heads, self.head_dim)
            graph_k = self.graph_k(x_flat).view(-1, self.num_heads, self.head_dim)
            graph_v = self.graph_v(x_flat).view(-1, self.num_heads, self.head_dim)
            
            # Adjust edge indices for batched processing
            batch_offset = torch.arange(batch_size, device=x.device) * seq_len
            edge_index_batched = edge_index.unsqueeze(0) + batch_offset.view(-1, 1, 1)
            edge_index_batched = edge_index_batched.view(2, -1)
            
            # Compute graph attention
            row, col = edge_index_batched
            q_i = graph_q[row]
            k_j = graph_k[col]
            v_j = graph_v[col]
            
            graph_scores = (q_i * k_j).sum(dim=-1) * self.scale
            graph_weights = F.softmax(graph_scores, dim=0)
            graph_weights = self.dropout(graph_weights)
            
            # Aggregate graph attention
            graph_out = torch.zeros_like(x_flat)
            for head in range(self.num_heads):
                head_weights = graph_weights[:, head].unsqueeze(-1)
                head_values = v_j[:, head, :]
                
                head_aggregated = torch.zeros(x_flat.size(0), self.head_dim, device=x.device)
                head_aggregated.index_add_(0, row, head_weights * head_values)
                
                start_idx = head * self.head_dim
                end_idx = (head + 1) * self.head_dim
                graph_out[:, start_idx:end_idx] = head_aggregated
            
            graph_out = graph_out.view(batch_size, seq_len, hidden_dim)
        else:
            graph_out = torch.zeros_like(temporal_out)
            graph_weights = None
        
        # Fusion of temporal and graph attention
        combined = torch.cat([temporal_out, graph_out], dim=-1)
        fusion_weights = torch.sigmoid(self.fusion_gate(combined))
        
        fused_out = (fusion_weights * temporal_out + 
                    (1 - fusion_weights) * graph_out)
        
        output = self.output_proj(fused_out)
        
        attention_weights = {
            'temporal': temporal_weights,
            'graph': graph_weights,
            'fusion': fusion_weights
        }
        
        return output, attention_weights
    
    def _temporal_graph_attention_nodes(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        graph_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Handle node input with graph attention only."""
        # For node-level input, we only apply graph attention
        if edge_index is None:
            return x, {'graph': None}
        
        num_nodes = x.size(0)
        
        graph_q = self.graph_q(x).view(num_nodes, self.num_heads, self.head_dim)
        graph_k = self.graph_k(x).view(num_nodes, self.num_heads, self.head_dim)
        graph_v = self.graph_v(x).view(num_nodes, self.num_heads, self.head_dim)
        
        row, col = edge_index
        q_i = graph_q[row]
        k_j = graph_k[col]
        v_j = graph_v[col]
        
        graph_scores = (q_i * k_j).sum(dim=-1) * self.scale
        graph_weights = F.softmax(graph_scores, dim=0)
        graph_weights = self.dropout(graph_weights)
        
        # Aggregate
        graph_out = torch.zeros_like(x)
        for head in range(self.num_heads):
            head_weights = graph_weights[:, head].unsqueeze(-1)
            head_values = v_j[:, head, :]
            
            head_aggregated = torch.zeros(num_nodes, self.head_dim, device=x.device)
            head_aggregated.index_add_(0, row, head_weights * head_values)
            
            start_idx = head * self.head_dim
            end_idx = (head + 1) * self.head_dim
            graph_out[:, start_idx:end_idx] = head_aggregated
        
        output = self.output_proj(graph_out)
        
        return output, {'graph': graph_weights}


class TemporalGraphTransformerLayer(nn.Module):
    """
    Single layer of Temporal Graph Transformer.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        temporal_weight: float = 0.5,
        norm_first: bool = True
    ):
        super(TemporalGraphTransformerLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.norm_first = norm_first
        ff_dim = ff_dim or 4 * hidden_dim
        
        # Temporal graph attention
        self.tg_attention = TemporalGraphAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            temporal_weight=temporal_weight
        )
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
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
        edge_index: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of temporal graph transformer layer.
        """
        if self.norm_first:
            # Pre-norm architecture
            norm_x = self.norm1(x)
            attn_out, attn_weights = self.tg_attention(norm_x, edge_index, temporal_mask)
            x = x + self.dropout(attn_out)
            
            norm_x = self.norm2(x)
            ff_out = self.ff_network(norm_x)
            x = x + ff_out
        else:
            # Post-norm architecture
            attn_out, attn_weights = self.tg_attention(x, edge_index, temporal_mask)
            x = self.norm1(x + self.dropout(attn_out))
            
            ff_out = self.ff_network(x)
            x = self.norm2(x + ff_out)
        
        return x, attn_weights


class TemporalGraphTransformer(nn.Module):
    """
    Complete Temporal Graph Transformer for advanced fraud detection.
    
    This model combines temporal sequence modeling with graph attention
    to capture both temporal patterns and structural relationships.
    
    Features:
    - Joint temporal and graph attention
    - Multi-scale temporal reasoning
    - Dynamic graph structure awareness
    - Advanced spatio-temporal fusion
    - Support for both sequence and node prediction
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        num_classes: int = 2,
        max_seq_len: int = 100,
        temporal_weight: float = 0.5,
        use_temporal_pos_encoding: bool = True,
        use_graph_pos_encoding: bool = True,
        prediction_mode: str = 'sequence'  # 'sequence' or 'node'
    ):
        super(TemporalGraphTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.prediction_mode = prediction_mode
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encodings
        self.use_temporal_pos_encoding = use_temporal_pos_encoding
        self.use_graph_pos_encoding = use_graph_pos_encoding
        
        if use_temporal_pos_encoding:
            self.temporal_pos_encoding = TemporalPositionalEncoding(
                hidden_dim, max_seq_len, dropout
            )
        
        if use_graph_pos_encoding:
            self.graph_pos_encoding = GraphPositionalEncoding(hidden_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TemporalGraphTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                temporal_weight=temporal_weight
            )
            for _ in range(num_layers)
        ])
        
        # Final layers
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
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
        edge_index: Optional[torch.Tensor] = None,
        time_steps: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Temporal Graph Transformer.
        
        Args:
            x: Input features 
               - For sequence mode: [batch_size, seq_len, input_dim]
               - For node mode: [num_nodes, input_dim]
            edge_index: Graph edges [2, num_edges] (optional)
            time_steps: Time step indices [batch_size, seq_len] (optional)
            temporal_mask: Temporal attention mask [batch_size, seq_len, seq_len] (optional)
            
        Returns:
            Dictionary containing:
            - 'logits': Output logits
            - 'embeddings': Final embeddings
            - 'attention_weights': Attention weights from each layer
        """
        # Input projection
        h = self.input_projection(x)
        
        # Add positional encodings
        if self.use_temporal_pos_encoding and h.dim() == 3:
            h = self.temporal_pos_encoding(h, time_steps)
        
        if self.use_graph_pos_encoding and edge_index is not None:
            if h.dim() == 3:
                # For sequence input, apply graph pos encoding to flattened version
                batch_size, seq_len, hidden_dim = h.shape
                h_flat = h.view(-1, hidden_dim)
                
                # Adjust edge indices for batched processing
                batch_offset = torch.arange(batch_size, device=h.device) * seq_len
                edge_index_batched = edge_index.unsqueeze(0) + batch_offset.view(-1, 1, 1)
                edge_index_batched = edge_index_batched.view(2, -1)
                
                graph_pos = self.graph_pos_encoding(h_flat, edge_index_batched)
                graph_pos = graph_pos.view(batch_size, seq_len, hidden_dim)
                h = h + graph_pos
            else:
                graph_pos = self.graph_pos_encoding(h, edge_index)
                h = h + graph_pos
        
        # Store attention weights
        all_attention_weights = []
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            h, attn_weights = layer(h, edge_index, temporal_mask)
            all_attention_weights.append(attn_weights)
        
        # Final normalization
        h = self.final_norm(h)
        embeddings = h
        
        # Output projection
        h = self.output_dropout(h)
        logits = self.output_projection(h)
        
        return {
            'logits': logits,
            'embeddings': embeddings,
            'attention_weights': all_attention_weights
        }


def create_temporal_graph_transformer(
    input_dim: int,
    config: Dict[str, Any]
) -> TemporalGraphTransformer:
    """
    Factory function to create Temporal Graph Transformer.
    
    Args:
        input_dim: Input feature dimension
        config: Model configuration dictionary
        
    Returns:
        Initialized Temporal Graph Transformer model
    """
    return TemporalGraphTransformer(
        input_dim=input_dim,
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        ff_dim=config.get('ff_dim', None),
        dropout=config.get('dropout', 0.1),
        num_classes=config.get('num_classes', 2),
        max_seq_len=config.get('max_seq_len', 100),
        temporal_weight=config.get('temporal_weight', 0.5),
        use_temporal_pos_encoding=config.get('use_temporal_pos_encoding', True),
        use_graph_pos_encoding=config.get('use_graph_pos_encoding', True),
        prediction_mode=config.get('prediction_mode', 'sequence')
    )


if __name__ == "__main__":
    # Test Temporal Graph Transformer
    print("Testing Temporal Graph Transformer...")
    
    # Test sequence mode
    print("\n1. Testing sequence mode...")
    batch_size = 4
    seq_len = 20
    input_dim = 186
    num_edges = 100
    
    x_seq = torch.randn(batch_size, seq_len, input_dim)
    edge_index = torch.randint(0, seq_len, (2, num_edges))
    time_steps = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    
    config = {
        'hidden_dim': 128,
        'num_layers': 3,
        'num_heads': 4,
        'dropout': 0.1,
        'prediction_mode': 'sequence'
    }
    
    model_seq = create_temporal_graph_transformer(input_dim, config)
    
    with torch.no_grad():
        outputs_seq = model_seq(x_seq, edge_index, time_steps)
    
    print(f"Sequence logits shape: {outputs_seq['logits'].shape}")
    print(f"Sequence embeddings shape: {outputs_seq['embeddings'].shape}")
    
    # Test node mode
    print("\n2. Testing node mode...")
    num_nodes = 1000
    x_nodes = torch.randn(num_nodes, input_dim)
    edge_index_nodes = torch.randint(0, num_nodes, (2, 2000))
    
    config['prediction_mode'] = 'node'
    model_nodes = create_temporal_graph_transformer(input_dim, config)
    
    with torch.no_grad():
        outputs_nodes = model_nodes(x_nodes, edge_index_nodes)
    
    print(f"Node logits shape: {outputs_nodes['logits'].shape}")
    print(f"Node embeddings shape: {outputs_nodes['embeddings'].shape}")
    
    print(f"\nModel parameters: {sum(p.numel() for p in model_seq.parameters()):,}")
    print("✅ Temporal Graph Transformer test completed successfully!")
