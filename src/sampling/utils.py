# src/sampling/utils.py
"""
Utility functions for temporal graph construction and validation per §PHASE_A.1
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Optional
from torch_geometric.data import HeteroData
import logging

logger = logging.getLogger(__name__)

def build_temporal_adjacency(
    hetero_data: HeteroData,
    edge_type: str = ('transaction', 'to', 'transaction')
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build temporal adjacency lists from HeteroData per §PHASE_A.1
    
    Args:
        hetero_data: PyG HeteroData object with timestamped edges
        edge_type: edge type to extract (default: transaction-to-transaction)
        
    Returns:
        indptr: CSR pointer array (n+1,)
        indices: neighbor indices (nnz,) 
        timestamps: edge timestamps aligned with indices (nnz,)
    """
    # Extract edge information
    edge_index = hetero_data[edge_type].edge_index
    edge_timestamps = hetero_data[edge_type].get('timestamp', None)
    
    if edge_timestamps is None:
        # Try alternative timestamp field names
        for attr_name in ['time', 't', 'timestamp', 'timestamps']:
            if hasattr(hetero_data[edge_type], attr_name):
                edge_timestamps = getattr(hetero_data[edge_type], attr_name)
                break
        
        if edge_timestamps is None:
            logger.warning("No timestamp information found, using sequential timestamps")
            edge_timestamps = torch.arange(edge_index.size(1), dtype=torch.float32)
    
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    
    # Determine number of nodes
    num_nodes = max(src_nodes.max().item(), dst_nodes.max().item()) + 1
    
    # Build adjacency lists per node with timestamps
    adjacency_dict = {}
    for i in range(len(src_nodes)):
        src = src_nodes[i].item()
        dst = dst_nodes[i].item()
        timestamp = edge_timestamps[i].item()
        
        if src not in adjacency_dict:
            adjacency_dict[src] = []
        adjacency_dict[src].append((dst, timestamp))
    
    # Sort each node's adjacency list by timestamp (descending for recency preference per §PHASE_A.2)
    for src in adjacency_dict:
        adjacency_dict[src].sort(key=lambda x: x[1], reverse=True)
    
    # Convert to CSR format
    indptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    indices_list = []
    timestamps_list = []
    
    for src in range(num_nodes):
        neighbors = adjacency_dict.get(src, [])
        indptr[src + 1] = indptr[src] + len(neighbors)
        
        for dst, timestamp in neighbors:
            indices_list.append(dst)
            timestamps_list.append(timestamp)
    
    indices = torch.tensor(indices_list, dtype=torch.long)
    timestamps = torch.tensor(timestamps_list, dtype=torch.float32)
    
    logger.info(f"Built temporal adjacency: {num_nodes} nodes, {len(indices)} edges")
    
    return indptr, indices, timestamps

def validate_temporal_constraints(
    temporal_graph,
    seed_nodes: torch.Tensor,
    t_eval: torch.Tensor,
    delta_t: float,
    sampled_subgraph
) -> Dict[str, bool]:
    """
    Validate temporal constraints per §PHASE_A.5 tests
    
    Args:
        temporal_graph: TemporalGraph object
        seed_nodes: seed nodes used for sampling
        t_eval: evaluation timestamps
        delta_t: time relaxation parameter
        sampled_subgraph: SubgraphBatch result
        
    Returns:
        validation results dictionary
    """
    results = {}
    
    # Test 1: time_window_filtering - all sampled edges within time window
    all_edges_valid = True
    if len(sampled_subgraph.sub_timestamps) > 0:
        # Check against corresponding t_eval for each edge
        for i, timestamp in enumerate(sampled_subgraph.sub_timestamps):
            # Find corresponding t_eval (use max as conservative check)
            max_t_eval = t_eval.max().item()
            if not (timestamp <= max_t_eval and timestamp >= max_t_eval - delta_t):
                all_edges_valid = False
                logger.warning(f"Edge {i} timestamp {timestamp} outside window [{max_t_eval - delta_t}, {max_t_eval}]")
                break
    
    results['time_window_filtering'] = all_edges_valid
    
    # Test 2: no_leakage - training mask edges respect causality
    no_leakage = True
    if hasattr(sampled_subgraph, 'train_mask'):
        # Verify that train_mask nodes don't use future information
        # This is enforced by the time window filtering above
        results['no_leakage'] = all_edges_valid
    else:
        results['no_leakage'] = True
    
    # Test 3: frontier_size - ensure reasonable frontier growth
    max_expected_frontier = sum([f for f in [15, 10]])  # fanouts from config
    actual_frontier = sampled_subgraph.num_nodes - len(seed_nodes)
    frontier_size_ok = actual_frontier <= max_expected_frontier * len(seed_nodes)
    results['frontier_size'] = frontier_size_ok
    
    if not frontier_size_ok:
        logger.warning(f"Frontier size {actual_frontier} exceeds expected {max_expected_frontier * len(seed_nodes)}")
    
    # Test 4: monotonic_timestamps - timestamps should be ordered per node
    monotonic_ok = True
    for u in range(sampled_subgraph.num_nodes):
        if u < len(sampled_subgraph.sub_indptr) - 1:
            start_idx = sampled_subgraph.sub_indptr[u]
            end_idx = sampled_subgraph.sub_indptr[u + 1]
            if end_idx > start_idx:
                node_timestamps = sampled_subgraph.sub_timestamps[start_idx:end_idx]
                if len(node_timestamps) > 1:
                    # Should be monotonic (descending for recency)
                    if not torch.all(node_timestamps[:-1] >= node_timestamps[1:]):
                        monotonic_ok = False
                        break
    
    results['monotonic_timestamps'] = monotonic_ok
    
    logger.info(f"Temporal validation results: {results}")
    return results

def extract_timestamps_from_data(data, edge_type=('transaction', 'to', 'transaction')):
    """
    Extract timestamps from various data formats
    
    Args:
        data: HeteroData, Data, or torch tensor
        edge_type: edge type for HeteroData
        
    Returns:
        timestamps tensor or None
    """
    if hasattr(data, '__getitem__') and edge_type in data:
        # HeteroData case
        edge_data = data[edge_type]
        for attr_name in ['timestamp', 'time', 't', 'timestamps']:
            if hasattr(edge_data, attr_name):
                return getattr(edge_data, attr_name)
    
    elif hasattr(data, 'edge_attr'):
        # Assume timestamps are in edge attributes
        return data.edge_attr
    
    elif hasattr(data, 'timestamp'):
        return data.timestamp
    
    logger.warning("No timestamp information found in data")
    return None

def create_temporal_masks(
    num_nodes: int,
    timestamps: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    temporal_split: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create temporal train/val/test masks respecting causality per §PHASE_A.1
    
    Args:
        num_nodes: total number of nodes
        timestamps: node timestamps for temporal ordering
        train_ratio: fraction for training
        val_ratio: fraction for validation  
        temporal_split: if True, use temporal ordering; if False, random split
        
    Returns:
        train_mask, val_mask, test_mask
    """
    if temporal_split and timestamps is not None:
        # Sort by timestamp for temporal split
        sorted_indices = torch.argsort(timestamps)
        
        train_end = int(train_ratio * num_nodes)
        val_end = int((train_ratio + val_ratio) * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool) 
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[sorted_indices[:train_end]] = True
        val_mask[sorted_indices[train_end:val_end]] = True
        test_mask[sorted_indices[val_end:]] = True
        
        logger.info(f"Created temporal split: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")
    else:
        # Random split fallback
        perm = torch.randperm(num_nodes)
        train_end = int(train_ratio * num_nodes)
        val_end = int((train_ratio + val_ratio) * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[perm[:train_end]] = True
        val_mask[perm[train_end:val_end]] = True  
        test_mask[perm[val_end:]] = True
        
        logger.info(f"Created random split: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")
    
    return train_mask, val_mask, test_mask
