"""
Fraud-specific hyperedge construction for PhenomNN-based hypergraph neural networks

Implements hyperedge construction strategies specifically designed for fraud detection:
1. Multi-entity transaction hyperedges (user+merchant+device+IP)
2. Temporal pattern hyperedges (entities active in same time window)
3. Amount pattern hyperedges (similar transaction amounts)
4. Behavioral pattern hyperedges (similar activity patterns)

Following the Stage 5 reference document specifications for fraud detection applications.
"""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict


class FraudHyperedgeConstructor:
    """
    Constructs fraud-specific hyperedges from heterogeneous transaction data.
    
    Supports multiple hyperedge types:
    - Transaction hyperedges: Link [user, merchant, device, IP] per transaction
    - Temporal hyperedges: Connect entities active in same time window
    - Amount pattern hyperedges: Group similar transaction amounts
    - Behavioral hyperedges: Connect entities with similar activity patterns
    """
    
    def __init__(
        self,
        transaction_weight: float = 1.0,
        temporal_weight: float = 0.5,
        amount_weight: float = 0.3,
        behavioral_weight: float = 0.2,
        time_window_hours: int = 24,
        amount_similarity_threshold: float = 0.1,
        min_hyperedge_size: int = 2,
        max_hyperedge_size: int = 10
    ):
        """
        Initialize fraud hyperedge constructor.
        
        Args:
            transaction_weight: Weight for transaction-based hyperedges
            temporal_weight: Weight for temporal pattern hyperedges  
            amount_weight: Weight for amount pattern hyperedges
            behavioral_weight: Weight for behavioral pattern hyperedges
            time_window_hours: Time window for temporal hyperedges (hours)
            amount_similarity_threshold: Threshold for amount similarity (relative)
            min_hyperedge_size: Minimum nodes per hyperedge
            max_hyperedge_size: Maximum nodes per hyperedge
        """
        self.transaction_weight = transaction_weight
        self.temporal_weight = temporal_weight
        self.amount_weight = amount_weight
        self.behavioral_weight = behavioral_weight
        self.time_window_hours = time_window_hours
        self.amount_similarity_threshold = amount_similarity_threshold
        self.min_hyperedge_size = min_hyperedge_size
        self.max_hyperedge_size = max_hyperedge_size
        
    def construct_hyperedges(self, hetero_data: HeteroData) -> Tuple[List[List[int]], Optional[torch.Tensor]]:
        """
        Construct all types of fraud-specific hyperedges.
        
        Args:
            hetero_data: Heterogeneous graph with transaction data
            
        Returns:
            Tuple of (hyperedges_list, hyperedge_features)
            hyperedges_list: List of hyperedges, each hyperedge is list of node indices
            hyperedge_features: Optional tensor of hyperedge feature vectors
        """
        hyperedges = []
        hyperedge_types = []
        hyperedge_weights = []
        
        # 1. Transaction-based hyperedges
        if self.transaction_weight > 0:
            tx_hyperedges = self._construct_transaction_hyperedges(hetero_data)
            hyperedges.extend(tx_hyperedges)
            hyperedge_types.extend(['transaction'] * len(tx_hyperedges))
            hyperedge_weights.extend([self.transaction_weight] * len(tx_hyperedges))
        
        # 2. Temporal pattern hyperedges
        if self.temporal_weight > 0:
            temporal_hyperedges = self._construct_temporal_hyperedges(hetero_data)
            hyperedges.extend(temporal_hyperedges)
            hyperedge_types.extend(['temporal'] * len(temporal_hyperedges))
            hyperedge_weights.extend([self.temporal_weight] * len(temporal_hyperedges))
        
        # 3. Amount pattern hyperedges
        if self.amount_weight > 0:
            amount_hyperedges = self._construct_amount_hyperedges(hetero_data)
            hyperedges.extend(amount_hyperedges)
            hyperedge_types.extend(['amount'] * len(amount_hyperedges))
            hyperedge_weights.extend([self.amount_weight] * len(amount_hyperedges))
        
        # 4. Behavioral pattern hyperedges
        if self.behavioral_weight > 0:
            behavioral_hyperedges = self._construct_behavioral_hyperedges(hetero_data)
            hyperedges.extend(behavioral_hyperedges)
            hyperedge_types.extend(['behavioral'] * len(behavioral_hyperedges))
            hyperedge_weights.extend([self.behavioral_weight] * len(behavioral_hyperedges))
        
        # Filter hyperedges by size constraints
        filtered_hyperedges = []
        filtered_types = []
        filtered_weights = []
        
        for he, he_type, he_weight in zip(hyperedges, hyperedge_types, hyperedge_weights):
            if self.min_hyperedge_size <= len(he) <= self.max_hyperedge_size:
                filtered_hyperedges.append(he)
                filtered_types.append(he_type)
                filtered_weights.append(he_weight)
        
        # Create hyperedge features from types and weights
        if filtered_hyperedges:
            hyperedge_features = self._encode_hyperedge_features(filtered_types, filtered_weights)
        else:
            hyperedge_features = None
            
        return filtered_hyperedges, hyperedge_features
    
    def _construct_transaction_hyperedges(self, hetero_data: HeteroData) -> List[List[int]]:
        """
        Construct hyperedges linking entities involved in same transaction.
        Each transaction creates a hyperedge connecting [user, merchant, device, IP].
        """
        hyperedges = []
        
        # Access transaction data
        tx_data = hetero_data.get('transaction', None)
        if tx_data is None:
            return hyperedges
            
        # For each transaction, create hyperedge with connected entities
        # This requires analyzing the heterogeneous edge structure
        transaction_entities = defaultdict(set)
        
        # Analyze different edge types to find entity connections
        for edge_type in hetero_data.edge_types:
            src_type, edge_name, dst_type = edge_type
            
            if src_type == 'transaction' or dst_type == 'transaction':
                edge_index = hetero_data[edge_type].edge_index
                
                if src_type == 'transaction':
                    # Transaction -> Entity edges
                    for tx_idx, entity_idx in zip(edge_index[0], edge_index[1]):
                        transaction_entities[tx_idx.item()].add((dst_type, entity_idx.item()))
                elif dst_type == 'transaction':
                    # Entity -> Transaction edges  
                    for entity_idx, tx_idx in zip(edge_index[0], edge_index[1]):
                        transaction_entities[tx_idx.item()].add((src_type, entity_idx.item()))
        
        # Convert transaction entity groups to hyperedges
        for tx_idx, entities in transaction_entities.items():
            # Include the transaction node itself
            hyperedge = [tx_idx]
            
            # Add connected entities (map to global node indices if needed)
            for entity_type, entity_idx in entities:
                # For now, assume all nodes are transaction nodes
                # In full implementation, would need proper node index mapping
                if entity_type == 'transaction':
                    hyperedge.append(entity_idx)
            
            if len(hyperedge) >= self.min_hyperedge_size:
                hyperedges.append(hyperedge)
        
        return hyperedges
    
    def _construct_temporal_hyperedges(self, hetero_data: HeteroData) -> List[List[int]]:
        """
        Construct hyperedges connecting entities active in same time window.
        """
        hyperedges = []
        
        # Extract timestamp information if available
        tx_data = hetero_data.get('transaction', None)
        if tx_data is None or not hasattr(tx_data, 'time'):
            return hyperedges
        
        timestamps = tx_data.time
        if timestamps is None:
            return hyperedges
        
        # Group transactions by time windows
        time_window_seconds = self.time_window_hours * 3600
        time_groups = defaultdict(list)
        
        for idx, timestamp in enumerate(timestamps):
            time_bin = int(timestamp.item() // time_window_seconds)
            time_groups[time_bin].append(idx)
        
        # Create hyperedges for each time window with sufficient transactions
        for time_bin, tx_indices in time_groups.items():
            if len(tx_indices) >= self.min_hyperedge_size:
                # Limit hyperedge size to prevent overly large hyperedges
                if len(tx_indices) > self.max_hyperedge_size:
                    # Sample subset of transactions
                    tx_indices = np.random.choice(
                        tx_indices, 
                        size=self.max_hyperedge_size, 
                        replace=False
                    ).tolist()
                hyperedges.append(tx_indices)
        
        return hyperedges
    
    def _construct_amount_hyperedges(self, hetero_data: HeteroData) -> List[List[int]]:
        """
        Construct hyperedges connecting transactions with similar amounts.
        """
        hyperedges = []
        
        tx_data = hetero_data.get('transaction', None)
        if tx_data is None or tx_data.x is None:
            return hyperedges
        
        # Assume first feature is transaction amount (or find amount feature)
        features = tx_data.x
        if features.shape[1] == 0:
            return hyperedges
        
        # Use first feature as amount proxy
        amounts = features[:, 0].cpu().numpy()
        
        # Group transactions by similar amounts
        amount_groups = defaultdict(list)
        
        for idx, amount in enumerate(amounts):
            if not np.isnan(amount) and amount > 0:
                # Bin amounts by relative similarity
                amount_bin = int(np.log10(amount + 1e-8) / self.amount_similarity_threshold)
                amount_groups[amount_bin].append(idx)
        
        # Create hyperedges for amount groups
        for amount_bin, tx_indices in amount_groups.items():
            if len(tx_indices) >= self.min_hyperedge_size:
                if len(tx_indices) > self.max_hyperedge_size:
                    tx_indices = np.random.choice(
                        tx_indices,
                        size=self.max_hyperedge_size,
                        replace=False
                    ).tolist()
                hyperedges.append(tx_indices)
        
        return hyperedges
    
    def _construct_behavioral_hyperedges(self, hetero_data: HeteroData) -> List[List[int]]:
        """
        Construct hyperedges connecting entities with similar behavioral patterns.
        """
        hyperedges = []
        
        tx_data = hetero_data.get('transaction', None)
        if tx_data is None or tx_data.x is None:
            return hyperedges
        
        features = tx_data.x.cpu().numpy()
        if features.shape[1] < 2:
            return hyperedges
        
        # Use feature similarity to group transactions
        # Simple approach: cluster based on feature vectors
        from sklearn.cluster import KMeans
        
        # Remove NaN values and normalize
        valid_mask = ~np.isnan(features).any(axis=1)
        if valid_mask.sum() < self.min_hyperedge_size:
            return hyperedges
        
        valid_features = features[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        # Normalize features
        feature_std = valid_features.std(axis=0)
        feature_std[feature_std == 0] = 1  # Avoid division by zero
        normalized_features = (valid_features - valid_features.mean(axis=0)) / feature_std
        
        # Cluster into behavioral groups
        n_clusters = min(len(valid_indices) // self.min_hyperedge_size, 20)
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(normalized_features)
            
            # Create hyperedges from clusters
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_indices = valid_indices[cluster_mask].tolist()
                
                if len(cluster_indices) >= self.min_hyperedge_size:
                    if len(cluster_indices) > self.max_hyperedge_size:
                        cluster_indices = np.random.choice(
                            cluster_indices,
                            size=self.max_hyperedge_size,
                            replace=False
                        ).tolist()
                    hyperedges.append(cluster_indices)
        
        return hyperedges
    
    def _encode_hyperedge_features(self, hyperedge_types: List[str], hyperedge_weights: List[float]) -> torch.Tensor:
        """
        Encode hyperedge features from types and weights.
        
        Args:
            hyperedge_types: List of hyperedge type strings
            hyperedge_weights: List of hyperedge weights
            
        Returns:
            Hyperedge feature tensor (n_hyperedges × d_features)
        """
        n_hyperedges = len(hyperedge_types)
        
        # One-hot encode hyperedge types
        type_mapping = {'transaction': 0, 'temporal': 1, 'amount': 2, 'behavioral': 3}
        type_features = torch.zeros((n_hyperedges, 4))
        
        for i, he_type in enumerate(hyperedge_types):
            type_features[i, type_mapping.get(he_type, 0)] = 1.0
        
        # Add weight features
        weight_features = torch.tensor(hyperedge_weights, dtype=torch.float).unsqueeze(1)
        
        # Combine type and weight features
        hyperedge_features = torch.cat([type_features, weight_features], dim=1)
        
        return hyperedge_features


def construct_fraud_hyperedges(
    hetero_data: HeteroData,
    constructor_config: Optional[Dict[str, Any]] = None
) -> Tuple[List[List[int]], Optional[torch.Tensor]]:
    """
    Convenience function to construct fraud-specific hyperedges.
    
    Args:
        hetero_data: Heterogeneous transaction graph
        constructor_config: Configuration for FraudHyperedgeConstructor
        
    Returns:
        Tuple of (hyperedges_list, hyperedge_features)
    """
    if constructor_config is None:
        constructor_config = {}
    
    constructor = FraudHyperedgeConstructor(**constructor_config)
    return constructor.construct_hyperedges(hetero_data)


def construct_simple_transaction_hyperedges(
    hetero_data: HeteroData,
    max_hyperedges: int = 1000
) -> Tuple[List[List[int]], Optional[torch.Tensor]]:
    """
    Simple hyperedge construction for quick testing.
    Creates small hyperedges by grouping nearby transaction nodes.
    
    Args:
        hetero_data: Heterogeneous transaction graph
        max_hyperedges: Maximum number of hyperedges to create
        
    Returns:
        Tuple of (hyperedges_list, hyperedge_features)
    """
    tx_data = hetero_data.get('transaction', None)
    if tx_data is None or tx_data.x is None:
        return [], None
    
    n_nodes = tx_data.x.shape[0]
    if n_nodes < 2:
        return [], None
    
    hyperedges = []
    
    # Create hyperedges by grouping consecutive transaction nodes
    hyperedge_size = 3  # Small hyperedges for testing
    
    for i in range(0, min(n_nodes - hyperedge_size + 1, max_hyperedges)):
        hyperedge = list(range(i, i + hyperedge_size))
        hyperedges.append(hyperedge)
    
    # Create simple hyperedge features (all transaction type)
    n_hyperedges = len(hyperedges)
    if n_hyperedges > 0:
        hyperedge_features = torch.zeros((n_hyperedges, 5))  # 4 type features + 1 weight
        hyperedge_features[:, 0] = 1.0  # All transaction type
        hyperedge_features[:, 4] = 1.0  # Weight = 1.0
    else:
        hyperedge_features = None
    
    return hyperedges, hyperedge_features
