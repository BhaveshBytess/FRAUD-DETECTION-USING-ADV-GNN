"""
Temporal Models for Stage 4: Time-Series Fraud Detection

This module implements various temporal models for fraud detection including:
- LSTM/GRU based sequence models
- Temporal Graph Attention Network (TGAN)
- Hybrid temporal-graph models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class TemporalLSTM(nn.Module):
    """
    LSTM-based temporal model for fraud detection.
    
    Processes temporal sequences of transaction features to detect fraudulent patterns.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
        bidirectional: bool = True,
        use_attention: bool = True
    ):
        super(TemporalLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output dimension from LSTM
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Attention mechanism
        if use_attention:
            self.attention = TemporalAttention(lstm_output_dim)
            classifier_input_dim = lstm_output_dim
        else:
            classifier_input_dim = lstm_output_dim
            
        # Dropout and classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM and linear layer weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through temporal LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            lengths: Optional sequence lengths for packed sequences
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM forward pass
        if lengths is not None:
            # Pack sequences for variable length
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention or use final hidden state
        if self.use_attention:
            # Attention over sequence
            attended_output = self.attention(lstm_out, lengths)
        else:
            # Use final hidden state
            if self.bidirectional:
                # Concatenate final forward and backward hidden states
                attended_output = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                attended_output = h_n[-1]
        
        # Apply dropout and classifier
        output = self.dropout(attended_output)
        logits = self.classifier(output)
        
        return logits


class TemporalGRU(nn.Module):
    """
    GRU-based temporal model for fraud detection.
    
    Similar to TemporalLSTM but uses GRU cells which are computationally lighter.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
        bidirectional: bool = True,
        use_attention: bool = True
    ):
        super(TemporalGRU, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output dimension from GRU
        gru_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Attention mechanism
        if use_attention:
            self.attention = TemporalAttention(gru_output_dim)
            classifier_input_dim = gru_output_dim
        else:
            classifier_input_dim = gru_output_dim
            
        # Dropout and classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through temporal GRU.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            lengths: Optional sequence lengths for packed sequences
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # GRU forward pass
        if lengths is not None:
            # Pack sequences for variable length
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            gru_out, h_n = self.gru(x_packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        else:
            gru_out, h_n = self.gru(x)
        
        # Apply attention or use final hidden state
        if self.use_attention:
            # Attention over sequence
            attended_output = self.attention(gru_out, lengths)
        else:
            # Use final hidden state
            if self.bidirectional:
                # Concatenate final forward and backward hidden states
                attended_output = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                attended_output = h_n[-1]
        
        # Apply dropout and classifier
        output = self.dropout(attended_output)
        logits = self.classifier(output)
        
        return logits


class TemporalAttention(nn.Module):
    """
    Attention mechanism for temporal sequences.
    
    Computes attention weights over sequence positions to focus on
    most relevant time steps for fraud detection.
    """
    
    def __init__(self, hidden_dim: int, attention_dim: int = None):
        super(TemporalAttention, self).__init__()
        
        if attention_dim is None:
            attention_dim = hidden_dim
        
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # Attention layers
        self.attention_linear = nn.Linear(hidden_dim, attention_dim)
        self.attention_vector = nn.Parameter(torch.randn(attention_dim))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.attention_linear.weight)
        nn.init.normal_(self.attention_vector, std=0.1)
    
    def forward(self, hidden_states: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention to hidden states.
        
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim)
            lengths: Optional sequence lengths for masking
            
        Returns:
            Attended output of shape (batch_size, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Compute attention scores
        attention_weights = self.attention_linear(hidden_states)  # (batch_size, seq_len, attention_dim)
        attention_weights = torch.tanh(attention_weights)
        
        # Compute attention scores using attention vector
        attention_scores = torch.matmul(attention_weights, self.attention_vector)  # (batch_size, seq_len)
        
        # Apply length mask if provided
        if lengths is not None:
            mask = torch.arange(seq_len, device=hidden_states.device)[None, :] >= lengths[:, None]
            attention_scores.masked_fill_(mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        
        # Apply attention to hidden states
        attended_output = torch.sum(
            attention_weights.unsqueeze(-1) * hidden_states, 
            dim=1
        )  # (batch_size, hidden_dim)
        
        return attended_output


class TemporalGraphAttentionNetwork(nn.Module):
    """
    Temporal Graph Attention Network (TGAN) combining graph structure with temporal modeling.
    
    This model combines:
    1. Graph attention mechanisms from Stage 3 (HAN)
    2. Temporal sequence modeling from LSTM/GRU
    3. Joint temporal-graph attention
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        temporal_dim: int = 64,
        graph_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
        use_temporal: bool = True,
        use_graph: bool = True
    ):
        super(TemporalGraphAttentionNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.temporal_dim = temporal_dim
        self.graph_dim = graph_dim
        self.use_temporal = use_temporal
        self.use_graph = use_graph
        
        # Feature projection
        self.feature_projection = nn.Linear(input_dim, hidden_dim)
        
        # Temporal branch
        if use_temporal:
            self.temporal_lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=temporal_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True,
                batch_first=True
            )
            self.temporal_attention = TemporalAttention(temporal_dim * 2)
            temporal_output_dim = temporal_dim * 2
        else:
            temporal_output_dim = 0
        
        # Graph branch (simplified for now - will integrate with HAN from Stage 3)
        if use_graph:
            self.graph_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.graph_projection = nn.Linear(hidden_dim, graph_dim)
            graph_output_dim = graph_dim
        else:
            graph_output_dim = 0
        
        # Fusion mechanism
        fusion_input_dim = temporal_output_dim + graph_output_dim
        if fusion_input_dim == 0:
            fusion_input_dim = hidden_dim
            
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=fusion_input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        temporal_lengths: Optional[torch.Tensor] = None,
        graph_edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through TGAN.
        
        Args:
            x: Input features (batch_size, seq_len, input_dim)
            temporal_lengths: Sequence lengths for temporal modeling
            graph_edge_index: Edge indices for graph modeling
            
        Returns:
            Output logits (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project features
        x_proj = self.feature_projection(x)  # (batch_size, seq_len, hidden_dim)
        
        outputs = []
        
        # Temporal branch
        if self.use_temporal:
            temporal_out, _ = self.temporal_lstm(x_proj)
            temporal_attended = self.temporal_attention(temporal_out, temporal_lengths)
            outputs.append(temporal_attended)
        
        # Graph branch (simplified - will enhance with HAN integration)
        if self.use_graph:
            # Apply self-attention as a simplified graph mechanism
            graph_out, _ = self.graph_attention(x_proj, x_proj, x_proj)
            # Global average pooling over sequence
            graph_pooled = torch.mean(graph_out, dim=1)  # (batch_size, hidden_dim)
            graph_projected = self.graph_projection(graph_pooled)
            outputs.append(graph_projected)
        
        # Fusion
        if len(outputs) > 1:
            fused = torch.cat(outputs, dim=-1)  # (batch_size, fusion_dim)
        elif len(outputs) == 1:
            fused = outputs[0]
        else:
            # Fallback to simple pooling
            fused = torch.mean(x_proj, dim=1)
        
        # Apply fusion attention
        fused_expanded = fused.unsqueeze(1)  # (batch_size, 1, fusion_dim)
        attended_fusion, _ = self.fusion_attention(fused_expanded, fused_expanded, fused_expanded)
        final_output = attended_fusion.squeeze(1)  # (batch_size, fusion_dim)
        
        # Final classification
        output = self.dropout(final_output)
        logits = self.classifier(output)
        
        return logits


class TemporalDataLoader:
    """
    Data loader for temporal fraud detection models.
    
    Handles batching of temporal sequences with proper padding and masking.
    """
    
    def __init__(
        self,
        temporal_data: Dict,
        batch_size: int = 32,
        shuffle: bool = True,
        max_seq_len: int = 20
    ):
        self.temporal_data = temporal_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_seq_len = max_seq_len
        
        # Prepare sequences
        self.sequences = self._prepare_sequences()
    
    def _prepare_sequences(self) -> List[Dict]:
        """Prepare temporal sequences from window data."""
        sequences = []
        
        windows = self.temporal_data['windows']
        
        for i, window in enumerate(windows):
            seq_data = {
                'features': window['data'],
                'window_id': i,
                'time_range': window['window_range'],
                'length': min(window['data'].shape[0], self.max_seq_len)
            }
            sequences.append(seq_data)
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """Get a batch of temporal sequences."""
        batch_sequences = [self.sequences[i] for i in indices]
        
        # Determine max length in batch
        max_len = min(max(seq['length'] for seq in batch_sequences), self.max_seq_len)
        
        # Prepare batch tensors
        batch_features = []
        batch_lengths = []
        
        for seq in batch_sequences:
            features = seq['features'][:max_len]  # Truncate if necessary
            
            # Pad if necessary
            if features.shape[0] < max_len:
                padding = torch.zeros(max_len - features.shape[0], features.shape[1])
                features = torch.cat([features, padding], dim=0)
            
            batch_features.append(features)
            batch_lengths.append(seq['length'])
        
        return {
            'features': torch.stack(batch_features),
            'lengths': torch.tensor(batch_lengths),
            'max_length': max_len
        }


def create_temporal_model(
    model_type: str,
    input_dim: int,
    config: Dict
) -> nn.Module:
    """
    Factory function to create temporal models.
    
    Args:
        model_type: Type of model ('lstm', 'gru', 'tgan')
        input_dim: Input feature dimension
        config: Model configuration
        
    Returns:
        Temporal model instance
    """
    if model_type.lower() == 'lstm':
        return TemporalLSTM(
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.3),
            num_classes=config.get('num_classes', 2),
            bidirectional=config.get('bidirectional', True),
            use_attention=config.get('use_attention', True)
        )
    
    elif model_type.lower() == 'gru':
        return TemporalGRU(
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.3),
            num_classes=config.get('num_classes', 2),
            bidirectional=config.get('bidirectional', True),
            use_attention=config.get('use_attention', True)
        )
    
    elif model_type.lower() == 'tgan':
        return TemporalGraphAttentionNetwork(
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 128),
            temporal_dim=config.get('temporal_dim', 64),
            graph_dim=config.get('graph_dim', 64),
            num_heads=config.get('num_heads', 4),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.3),
            num_classes=config.get('num_classes', 2),
            use_temporal=config.get('use_temporal', True),
            use_graph=config.get('use_graph', True)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Quick test of temporal models
    print("Testing temporal models...")
    
    # Test parameters
    batch_size = 4
    seq_len = 10
    input_dim = 186  # From enhanced features
    
    # Create test data
    x = torch.randn(batch_size, seq_len, input_dim)
    lengths = torch.tensor([10, 8, 6, 9])
    
    # Test LSTM
    print("\nTesting TemporalLSTM...")
    lstm_model = TemporalLSTM(input_dim)
    lstm_out = lstm_model(x, lengths)
    print(f"LSTM output shape: {lstm_out.shape}")
    
    # Test GRU
    print("\nTesting TemporalGRU...")
    gru_model = TemporalGRU(input_dim)
    gru_out = gru_model(x, lengths)
    print(f"GRU output shape: {gru_out.shape}")
    
    # Test TGAN
    print("\nTesting TGAN...")
    tgan_model = TemporalGraphAttentionNetwork(input_dim)
    tgan_out = tgan_model(x, lengths)
    print(f"TGAN output shape: {tgan_out.shape}")
    
    print("\nAll temporal models working correctly!")
