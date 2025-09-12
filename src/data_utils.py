# src/data_utils.py
import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import HeteroData
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

def load_csv_nodes_edges(nodes_csv, edges_csv):
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)
    return nodes, edges

def build_hetero_data(nodes_df, edges_df,
                      node_type_col='type', node_id_col='id',
                      src_col='src', dst_col='dst', edge_type_col='etype',
                      time_col='time', label_col='label', labels_df=None):
    """
    Minimal converter: returns a PyG HeteroData object.
    Assumes nodes_df has: id, type, features... ; edges_df has: src, dst, etype, time, features...
    """
    data = HeteroData()
    
    # Create a mapping from original node ID to a new integer index for each node type
    node_mappings = {}
    for nt in nodes_df[node_type_col].unique():
        nt_nodes = nodes_df[nodes_df[node_type_col] == nt]
        node_mappings[nt] = {old_id: new_id for new_id, old_id in enumerate(nt_nodes[node_id_col])}

    # If labels are provided, merge them into the nodes_df
    if labels_df is not None:
        # Assuming labels_df has [node_id_col, label_col]
        nodes_df = pd.merge(nodes_df, labels_df, on=node_id_col, how='left')

    # Build node collections by type
    for nt, mapping in node_mappings.items():
        nt_df = nodes_df[nodes_df[node_type_col] == nt]
        data[nt].num_nodes = len(nt_df)
        
        # Store features if present
        if 'feature_0' in nt_df.columns:
            feats = torch.tensor(nt_df.filter(regex='^feature_').values, dtype=torch.float)
            data[nt].x = feats
        
        # Store labels if present
        if label_col in nt_df.columns:
            # Fill NaNs with a value indicating no label, e.g., -1
            labels = torch.tensor(nt_df[label_col].fillna(-1).values, dtype=torch.long)
            data[nt].y = labels

    # Build edges per relation
    for rel in edges_df[edge_type_col].unique():
        rel_df = edges_df[edges_df[edge_type_col] == rel]
        src_type, _, dst_type = rel.partition('->')
        
        # Map original source and destination IDs to the new integer indices
        src_indices = rel_df[src_col].map(node_mappings[src_type]).values
        dst_indices = rel_df[dst_col].map(node_mappings[dst_type]).values
        
        edge_index = torch.tensor(np.vstack([src_indices, dst_indices]), dtype=torch.long)
        data[(src_type, rel, dst_type)].edge_index = edge_index
        
        # Attach timestamps if present
        if time_col in rel_df.columns:
            data[(src_type, rel, dst_type)].time = torch.tensor(rel_df[time_col].values, dtype=torch.float)

    return data


def build_hypergraph_data(
    hetero_data: HeteroData,
    hypergraph_config: Optional[Dict[str, Any]] = None
) -> Tuple[Any, torch.Tensor, torch.Tensor]:
    """
    Convert HeteroData to hypergraph representation for fraud detection.
    
    Args:
        hetero_data: PyG HeteroData object
        hypergraph_config: Configuration for hypergraph construction
        
    Returns:
        hypergraph_data: HypergraphData object
        node_features: Combined node features tensor
        labels: Node labels for transaction nodes
    """
    from models.hypergraph import FraudHyperedgeConstructor, construct_hypergraph_from_hetero
    
    # Default configuration
    if hypergraph_config is None:
        hypergraph_config = {
            'fraud_pattern_types': [
                'transaction_patterns',
                'temporal_patterns', 
                'amount_patterns',
                'behavioral_patterns'
            ],
            'min_hyperedge_size': 2,
            'max_hyperedge_size': 10,
            'temporal_window': 24,  # hours
            'amount_bins': 5,
            'clustering_eps': 0.1,
            'clustering_min_samples': 3,
            'random_seed': 42
        }
    
    logger.info(f"Building hypergraph with config: {hypergraph_config}")
    
    # Create fraud hyperedge constructor
    constructor = FraudHyperedgeConstructor(
        time_window_hours=hypergraph_config.get('temporal_window', 24),
        min_hyperedge_size=hypergraph_config.get('min_hyperedge_size', 2),
        max_hyperedge_size=hypergraph_config.get('max_hyperedge_size', 10)
    )
    
    # Construct hypergraph from heterogeneous data
    hypergraph_data = construct_hypergraph_from_hetero(
        hetero_data, 
        hyperedge_construction_fn=constructor.construct_hyperedges,
        node_type='transaction'
    )
    
    # Extract node features and labels
    node_features = hypergraph_data.X
    
    # Extract labels from transaction nodes
    if 'transaction' in hetero_data.node_types:
        tx_data = hetero_data['transaction']
        if hasattr(tx_data, 'y') and tx_data.y is not None:
            # Filter out unknown labels and remap to binary
            known_mask = tx_data.y != 3
            labels = tx_data.y[known_mask].clone()
            labels[labels == 1] = 0  # licit
            labels[labels == 2] = 1  # illicit
            
            # Apply same mask to features
            if known_mask.sum() != node_features.size(0):
                logger.warning(f"Label mask size {known_mask.sum()} != feature size {node_features.size(0)}")
                # Truncate to match smaller size
                min_size = min(known_mask.sum().item(), node_features.size(0))
                labels = labels[:min_size]
                node_features = node_features[:min_size]
        else:
            logger.warning("No labels found in transaction data")
            labels = torch.zeros(node_features.size(0), dtype=torch.long)
    else:
        logger.warning("No transaction nodes found")
        labels = torch.zeros(node_features.size(0), dtype=torch.long)
    
    # Ensure features and labels are the same size
    min_size = min(node_features.size(0), labels.size(0))
    node_features = node_features[:min_size]
    labels = labels[:min_size]
    
    logger.info(f"Created hypergraph: {hypergraph_data.n_nodes} nodes, "
               f"{hypergraph_data.n_hyperedges} hyperedges, "
               f"features: {node_features.shape}, labels: {labels.shape}")
    
    return hypergraph_data, node_features, labels


def create_hypergraph_masks(
    num_nodes: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create train/validation/test masks for hypergraph nodes.
    
    Args:
        num_nodes: Total number of nodes
        train_ratio: Proportion for training
        val_ratio: Proportion for validation  
        test_ratio: Proportion for testing
        seed: Random seed
        
    Returns:
        train_mask, val_mask, test_mask: Boolean tensors
    """
    torch.manual_seed(seed)
    
    # Ensure ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    # Create random permutation
    perm = torch.randperm(num_nodes)
    
    # Split indices
    train_end = int(train_ratio * num_nodes)
    val_end = train_end + int(val_ratio * num_nodes)
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True
    
    logger.info(f"Created masks: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")
    
    return train_mask, val_mask, test_mask
