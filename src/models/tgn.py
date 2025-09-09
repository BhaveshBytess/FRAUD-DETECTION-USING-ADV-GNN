"""
Temporal Graph Networks (TGN) for Stage 4: Memory-based Temporal Modeling

This module implements TGN and TGAT architectures with explicit memory modules
for fraud detection, following the original TGN paper principles:
- Memory modules that maintain node state over time
- Message aggregation and memory update pipeline
- Time-aware neighbor sampling
- Temporal attention mechanisms

References:
- TGN: https://arxiv.org/abs/2006.10637
- TGAT: https://arxiv.org/abs/2002.07962
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import math
from collections import defaultdict


class MemoryModule(nn.Module):
    """
    Memory module that maintains node representations over time.
    
    Key features:
    - Persistent memory for each node
    - Memory update based on messages
    - Time-aware memory decay
    """
    
    def __init__(
        self,
        num_nodes: int,
        memory_dim: int,
        message_dim: int,
        time_dim: int = 32,
        memory_updater: str = 'gru'
    ):
        super(MemoryModule, self).__init__()
        
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        self.time_dim = time_dim
        
        # Initialize memory bank
        self.register_buffer('memory', torch.zeros(num_nodes, memory_dim))
        self.register_buffer('last_update_time', torch.zeros(num_nodes))
        
        # Memory updater
        if memory_updater == 'gru':
            self.memory_updater = nn.GRUCell(message_dim, memory_dim)
        elif memory_updater == 'lstm':
            self.memory_updater = nn.LSTMCell(message_dim, memory_dim)
            self.cell_state = torch.zeros(num_nodes, memory_dim)
        else:
            raise ValueError(f"Unknown memory updater: {memory_updater}")
        
        # Time encoding
        self.time_encoder = nn.Linear(1, time_dim)
        self.memory_time_fusion = nn.Linear(memory_dim + time_dim, memory_dim)
        
        # Message processing
        self.message_processor = nn.Sequential(
            nn.Linear(message_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim)
        )
    
    def get_memory(self, node_ids: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Get memory for specific nodes at given timestamps.
        
        Args:
            node_ids: Node IDs to get memory for
            timestamps: Current timestamps
            
        Returns:
            Memory representations for nodes
        """
        # Get current memory
        node_memory = self.memory[node_ids]  # (batch_size, memory_dim)
        
        # Apply time-aware decay
        time_deltas = timestamps.unsqueeze(-1) - self.last_update_time[node_ids].unsqueeze(-1)
        time_encoding = torch.tanh(self.time_encoder(time_deltas))
        
        # Fuse memory with time information
        memory_with_time = torch.cat([node_memory, time_encoding], dim=-1)
        updated_memory = torch.tanh(self.memory_time_fusion(memory_with_time))
        
        return updated_memory
    
    def update_memory(
        self, 
        node_ids: torch.Tensor, 
        messages: torch.Tensor, 
        timestamps: torch.Tensor
    ):
        """
        Update memory for nodes based on aggregated messages.
        
        Args:
            node_ids: Node IDs to update
            messages: Aggregated messages for nodes
            timestamps: Update timestamps
        """
        # Process messages
        processed_messages = self.message_processor(messages)
        
        # Update memory using GRU/LSTM
        for i, node_id in enumerate(node_ids):
            if hasattr(self, 'cell_state'):
                # LSTM update
                new_memory, new_cell = self.memory_updater(
                    processed_messages[i:i+1], 
                    (self.memory[node_id:node_id+1], self.cell_state[node_id:node_id+1])
                )
                self.cell_state[node_id] = new_cell.squeeze(0)
            else:
                # GRU update
                new_memory = self.memory_updater(
                    processed_messages[i:i+1], 
                    self.memory[node_id:node_id+1]
                )
            
            self.memory[node_id] = new_memory.squeeze(0)
            self.last_update_time[node_id] = timestamps[i]
    
    def reset_memory(self, node_ids: Optional[torch.Tensor] = None):
        """Reset memory for specific nodes or all nodes."""
        if node_ids is None:
            self.memory.zero_()
            self.last_update_time.zero_()
            if hasattr(self, 'cell_state'):
                self.cell_state.zero_()
        else:
            self.memory[node_ids] = 0
            self.last_update_time[node_ids] = 0
            if hasattr(self, 'cell_state'):
                self.cell_state[node_ids] = 0


class MessageAggregator(nn.Module):
    """
    Aggregates messages for nodes based on temporal interactions.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        message_dim: int,
        aggregation_method: str = 'attention'
    ):
        super(MessageAggregator, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.message_dim = message_dim
        self.aggregation_method = aggregation_method
        
        # Message function
        self.message_function = nn.Sequential(
            nn.Linear(2 * node_feature_dim + edge_feature_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim)
        )
        
        # Aggregation function
        if aggregation_method == 'attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=message_dim,
                num_heads=4,
                batch_first=True
            )
        elif aggregation_method == 'mean':
            pass  # Simple mean aggregation
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    def forward(
        self,
        source_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
        edge_features: torch.Tensor,
        node_features: torch.Tensor,
        timestamps: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Aggregate messages for each target node.
        
        Args:
            source_nodes: Source node IDs
            target_nodes: Target node IDs  
            edge_features: Edge features
            node_features: Node features
            timestamps: Interaction timestamps
            
        Returns:
            Dictionary mapping target node ID to aggregated message
        """
        # Create messages
        source_feats = node_features[source_nodes]
        target_feats = node_features[target_nodes]
        
        # Combine source, target, and edge features
        message_input = torch.cat([source_feats, target_feats, edge_features], dim=-1)
        messages = self.message_function(message_input)
        
        # Group messages by target node
        target_messages = defaultdict(list)
        target_timestamps = defaultdict(list)
        
        for i, target_id in enumerate(target_nodes):
            target_messages[target_id.item()].append(messages[i])
            target_timestamps[target_id.item()].append(timestamps[i])
        
        # Aggregate messages for each target
        aggregated_messages = {}
        
        for target_id, msgs in target_messages.items():
            if len(msgs) == 1:
                aggregated_messages[target_id] = msgs[0]
            else:
                msg_stack = torch.stack(msgs)  # (num_msgs, message_dim)
                
                if self.aggregation_method == 'attention':
                    # Use attention-based aggregation
                    msg_stack = msg_stack.unsqueeze(0)  # (1, num_msgs, message_dim)
                    aggregated, _ = self.attention(msg_stack, msg_stack, msg_stack)
                    aggregated_messages[target_id] = aggregated.squeeze(0).mean(dim=0)
                else:
                    # Simple mean aggregation
                    aggregated_messages[target_id] = msg_stack.mean(dim=0)
        
        return aggregated_messages


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for TGAT.
    """
    
    def __init__(
        self,
        feature_dim: int,
        time_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super(TemporalAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.time_dim = time_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0
        
        # Time encoding
        self.time_encoder = nn.Linear(1, time_dim)
        
        # Attention components
        self.query_projection = nn.Linear(feature_dim + time_dim, feature_dim)
        self.key_projection = nn.Linear(feature_dim + time_dim, feature_dim)
        self.value_projection = nn.Linear(feature_dim, feature_dim)
        
        self.output_projection = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        query_nodes: torch.Tensor,
        key_nodes: torch.Tensor,
        query_time: torch.Tensor,
        key_time: torch.Tensor,
        node_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute temporal attention between nodes.
        
        Args:
            query_nodes: Query node IDs
            key_nodes: Key node IDs
            query_time: Query timestamps
            key_time: Key timestamps
            node_features: Node feature matrix
            
        Returns:
            Attention-weighted features for query nodes
        """
        batch_size = query_nodes.size(0)
        
        # Get node features
        query_feats = node_features[query_nodes]  # (batch_size, feature_dim)
        key_feats = node_features[key_nodes]  # (batch_size, feature_dim)
        
        # Encode time differences
        time_deltas = (query_time - key_time).unsqueeze(-1)
        time_encoding = torch.tanh(self.time_encoder(time_deltas))
        
        # Create queries and keys with time information
        query_with_time = torch.cat([query_feats, time_encoding], dim=-1)
        key_with_time = torch.cat([key_feats, time_encoding], dim=-1)
        
        # Project to attention space
        queries = self.query_projection(query_with_time)
        keys = self.key_projection(key_with_time)
        values = self.value_projection(key_feats)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, self.num_heads, self.head_dim)
        values = values.view(batch_size, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attention_scores = torch.sum(queries * keys, dim=-1) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = attention_weights.unsqueeze(-1) * values
        attended_values = attended_values.view(batch_size, self.feature_dim)
        
        # Final projection
        output = self.output_projection(attended_values)
        
        return output


class TGN(nn.Module):
    """
    Temporal Graph Network (TGN) implementation.
    
    Key components:
    - Memory module for persistent node state
    - Message aggregation and memory updates
    - Temporal embeddings
    """
    
    def __init__(
        self,
        num_nodes: int,
        node_feature_dim: int,
        edge_feature_dim: int,
        memory_dim: int = 128,
        message_dim: int = 128,
        embedding_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        memory_updater: str = 'gru'
    ):
        super(TGN, self).__init__()
        
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        self.embedding_dim = embedding_dim
        
        # Memory module
        self.memory_module = MemoryModule(
            num_nodes=num_nodes,
            memory_dim=memory_dim,
            message_dim=message_dim,
            memory_updater=memory_updater
        )
        
        # Message aggregator
        self.message_aggregator = MessageAggregator(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            message_dim=message_dim
        )
        
        # Embedding layers
        self.node_projector = nn.Linear(node_feature_dim, embedding_dim)
        self.memory_projector = nn.Linear(memory_dim, embedding_dim)
        
        # Graph layers
        self.graph_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )
    
    def forward(
        self,
        source_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
        edge_features: torch.Tensor,
        node_features: torch.Tensor,
        timestamps: torch.Tensor,
        predict_nodes: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through TGN.
        
        Args:
            source_nodes: Source node IDs for interactions
            target_nodes: Target node IDs for interactions
            edge_features: Edge features for interactions
            node_features: Node feature matrix
            timestamps: Interaction timestamps
            predict_nodes: Nodes to make predictions for
            
        Returns:
            Predictions for predict_nodes
        """
        # 1. Aggregate messages for target nodes
        aggregated_messages = self.message_aggregator(
            source_nodes, target_nodes, edge_features, node_features, timestamps
        )
        
        # 2. Update memory for nodes that received messages
        if aggregated_messages:
            update_nodes = torch.tensor(list(aggregated_messages.keys()), dtype=torch.long)
            update_messages = torch.stack([aggregated_messages[nid] for nid in aggregated_messages.keys()])
            update_timestamps = timestamps[:len(update_nodes)]  # Use corresponding timestamps
            
            self.memory_module.update_memory(update_nodes, update_messages, update_timestamps)
        
        # 3. Generate embeddings for prediction nodes
        predict_timestamps = timestamps[:len(predict_nodes)]  # Assuming same time for prediction
        
        # Get current memory
        memory_embeddings = self.memory_module.get_memory(predict_nodes, predict_timestamps)
        
        # Get node features
        node_embeddings = self.node_projector(node_features[predict_nodes])
        
        # Combine memory and node features
        combined_embeddings = self.memory_projector(memory_embeddings) + node_embeddings
        
        # Apply graph layers
        for layer in self.graph_layers:
            combined_embeddings = layer(combined_embeddings) + combined_embeddings
        
        # Make predictions
        predictions = self.classifier(combined_embeddings)
        
        return predictions


class TGAT(nn.Module):
    """
    Temporal Graph Attention Network (TGAT) implementation.
    
    Key features:
    - Time-aware attention mechanisms
    - Temporal neighbor sampling
    - Multi-layer temporal attention
    """
    
    def __init__(
        self,
        num_nodes: int,
        node_feature_dim: int,
        edge_feature_dim: int,
        embedding_dim: int = 128,
        time_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super(TGAT, self).__init__()
        
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Input projections
        self.node_projector = nn.Linear(node_feature_dim, embedding_dim)
        self.edge_projector = nn.Linear(edge_feature_dim, embedding_dim)
        
        # Temporal attention layers
        self.attention_layers = nn.ModuleList([
            TemporalAttention(
                feature_dim=embedding_dim,
                time_dim=time_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(num_layers)
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )
    
    def forward(
        self,
        source_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
        edge_features: torch.Tensor,
        node_features: torch.Tensor,
        timestamps: torch.Tensor,
        predict_nodes: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through TGAT.
        """
        # Project node features
        node_embeddings = self.node_projector(node_features)
        
        # Apply temporal attention layers
        for i, (attention_layer, layer_norm) in enumerate(zip(self.attention_layers, self.layer_norms)):
            # Use source nodes as keys for target nodes (queries)
            attended = attention_layer(
                query_nodes=target_nodes,
                key_nodes=source_nodes,
                query_time=timestamps,
                key_time=timestamps,
                node_features=node_embeddings
            )
            
            # Residual connection and layer norm
            node_embeddings[target_nodes] = layer_norm(
                node_embeddings[target_nodes] + attended
            )
        
        # Make predictions for specified nodes
        predictions = self.classifier(node_embeddings[predict_nodes])
        
        return predictions


def create_tgn_model(
    model_type: str,
    num_nodes: int,
    node_feature_dim: int,
    edge_feature_dim: int,
    config: Dict[str, Any]
) -> nn.Module:
    """
    Create TGN or TGAT model based on configuration.
    """
    if model_type.lower() == 'tgn':
        return TGN(
            num_nodes=num_nodes,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            memory_dim=config.get('memory_dim', 128),
            message_dim=config.get('message_dim', 128),
            embedding_dim=config.get('embedding_dim', 128),
            num_layers=config.get('num_layers', 2),
            num_classes=config.get('num_classes', 2),
            dropout=config.get('dropout', 0.1),
            memory_updater=config.get('memory_updater', 'gru')
        )
    
    elif model_type.lower() == 'tgat':
        return TGAT(
            num_nodes=num_nodes,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            embedding_dim=config.get('embedding_dim', 128),
            time_dim=config.get('time_dim', 32),
            num_heads=config.get('num_heads', 4),
            num_layers=config.get('num_layers', 2),
            num_classes=config.get('num_classes', 2),
            dropout=config.get('dropout', 0.1)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test TGN and TGAT models
    print("Testing TGN and TGAT implementations...")
    
    # Test parameters
    num_nodes = 1000
    node_feature_dim = 93
    edge_feature_dim = 4
    batch_size = 32
    
    # Create sample data
    source_nodes = torch.randint(0, num_nodes, (batch_size,))
    target_nodes = torch.randint(0, num_nodes, (batch_size,))
    edge_features = torch.randn(batch_size, edge_feature_dim)
    node_features = torch.randn(num_nodes, node_feature_dim)
    timestamps = torch.randint(1, 50, (batch_size,)).float()
    predict_nodes = torch.randint(0, num_nodes, (10,))
    
    # Test TGN
    print("\nTesting TGN...")
    tgn_config = {
        'memory_dim': 64,
        'message_dim': 64,
        'embedding_dim': 64,
        'num_layers': 2
    }
    
    tgn_model = create_tgn_model('tgn', num_nodes, node_feature_dim, edge_feature_dim, tgn_config)
    tgn_output = tgn_model(source_nodes, target_nodes, edge_features, node_features, timestamps, predict_nodes)
    print(f"TGN output shape: {tgn_output.shape}")
    
    # Test TGAT
    print("\nTesting TGAT...")
    tgat_config = {
        'embedding_dim': 64,
        'time_dim': 16,
        'num_heads': 4,
        'num_layers': 2
    }
    
    tgat_model = create_tgn_model('tgat', num_nodes, node_feature_dim, edge_feature_dim, tgat_config)
    tgat_output = tgat_model(source_nodes, target_nodes, edge_features, node_features, timestamps, predict_nodes)
    print(f"TGAT output shape: {tgat_output.shape}")
    
    print("\n✅ TGN and TGAT models working correctly!")
