# src/sampling/temporal_data_loader.py
"""
Temporal data loading extensions per §PHASE_A.1
Extends existing data loading to handle timestamps & directed edges
"""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from typing import Optional, Dict, Any, Tuple
import logging
import os

from sampling.cpu_fallback import TemporalGraph
from sampling.utils import build_temporal_adjacency, create_temporal_masks

logger = logging.getLogger(__name__)

class TemporalGraphDataLoader:
    """Data loader for temporal graph structures per §PHASE_A.1"""
    
    def __init__(self):
        """Initialize temporal graph data loader"""
        pass
    
    def create_temporal_graph(self, hetero_data: HeteroData) -> TemporalGraph:
        """Create temporal graph from HeteroData"""
        return create_temporal_graph_from_hetero(hetero_data)

def load_ellipticpp_with_timestamps(data_path: str = "data/ellipticpp.pt") -> HeteroData:
    """
    Load ellipticpp data and add synthetic timestamps per §PHASE_A.1
    
    Args:
        data_path: path to ellipticpp.pt file
        
    Returns:
        HeteroData with timestamp information
    """
    # Load existing data
    data = torch.load(data_path, weights_only=False)
    
    # Find transaction node type (could be 'transaction', 'user', or other)
    node_types = [k for k in data.keys() if isinstance(k, str) and hasattr(data[k], 'num_nodes')]
    if not node_types:
        raise ValueError("No node types found in data")
    
    # Use first node type as primary (usually 'user' or 'transaction')
    primary_node_type = node_types[0]
    logger.info(f"Using primary node type: {primary_node_type}")
    
    # Find edge types that connect primary nodes to themselves or others
    edge_types = getattr(data, 'edge_types', [])
    if not edge_types:
        # Fallback: manually check keys
        edge_types = [k for k in data.keys() if isinstance(k, tuple) and len(k) == 3]
    if not edge_types:
        logger.warning("No edge types found in data")
        return data
    
    # Use first available edge type
    edge_type = edge_types[0]
    logger.info(f"Using edge type: {edge_type}")
    
    edge_index = data[edge_type].edge_index
    num_edges = edge_index.size(1)
    
    # Check if timestamps already exist
    if hasattr(data[edge_type], 'timestamp') or hasattr(data[edge_type], 'time'):
        logger.info("Timestamps already exist in data")
        if not hasattr(data[edge_type], 'timestamp') and hasattr(data[edge_type], 'time'):
            data[edge_type].timestamp = data[edge_type].time
        
        # Ensure node timestamps are compatible with edge timestamps
        edge_timestamps = getattr(data[edge_type], 'timestamp', data[edge_type].time)
        max_edge_time = edge_timestamps.max().item()
        
        # Get number of nodes safely
        try:
            num_primary_nodes = data[primary_node_type].num_nodes
            if num_primary_nodes is None:
                raise ValueError("num_nodes is None")
        except:
            # Fallback: estimate from max node ID in edges
            max_node_id = edge_index.max().item() if edge_index.numel() > 0 else 0
            num_primary_nodes = max_node_id + 1
            logger.warning(f"Could not get num_nodes, estimated {num_primary_nodes} from edge indices")
        
        # Create node timestamps that are compatible with edge timestamps
        # Each node gets a timestamp that allows it to see edges up to that time
        if num_primary_nodes > 1:
            node_timestamps = torch.linspace(0, max_edge_time, num_primary_nodes)
        else:
            node_timestamps = torch.tensor([max_edge_time])
        data[primary_node_type].timestamp = node_timestamps
        
        logger.info(f"Synchronized timestamps: nodes=[{node_timestamps.min():.1f}, {node_timestamps.max():.1f}], edges=[{edge_timestamps.min():.1f}, {edge_timestamps.max():.1f}]")
        return data
    
    # Generate synthetic timestamps per §PHASE_A.1 concepts
    # Strategy: use transaction IDs as proxies for time ordering
    # This is common in fraud detection when explicit timestamps unavailable
    
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    
    # Create timestamps based on source node IDs (simulating chronological ordering)
    # Add some noise to avoid perfect ordering
    base_timestamps = src_nodes.float() * 100.0  # scale up for realistic timestamps
    noise = torch.randn(num_edges) * 10.0  # small noise
    timestamps = base_timestamps + noise
    
    # Ensure positive timestamps
    timestamps = torch.clamp(timestamps, min=1.0)
    
    # Add timestamp attribute to edge data
    data[edge_type].timestamp = timestamps
    data[edge_type].time = timestamps  # alternative field name
    
    # Also add node-level timestamps (for t_eval)
    num_primary_nodes = data[primary_node_type].num_nodes
    node_timestamps = torch.arange(num_primary_nodes, dtype=torch.float32) * 50.0 + 1000.0
    data[primary_node_type].timestamp = node_timestamps
    
    logger.info(f"Added timestamps to {num_edges} edges and {num_primary_nodes} nodes")
    logger.info(f"Timestamp range: [{timestamps.min():.1f}, {timestamps.max():.1f}]")
    
    return data

def create_temporal_graph_from_hetero(
    hetero_data: HeteroData,
    edge_type: Optional[str] = None
) -> TemporalGraph:
    """
    Create TemporalGraph from HeteroData per §PHASE_A.1
    
    Args:
        hetero_data: PyG HeteroData with timestamp information
        edge_type: edge type to extract for temporal graph (auto-detect if None)
        
    Returns:
        TemporalGraph object for time-relaxed sampling
    """
    # Auto-detect edge type if not provided
    if edge_type is None:
        edge_types = getattr(hetero_data, 'edge_types', [])
        if not edge_types:
            # Fallback: manually check keys 
            edge_types = [k for k in hetero_data.keys() if isinstance(k, tuple) and len(k) == 3]
        if not edge_types:
            raise ValueError("No edge types found in data")
        edge_type = edge_types[0]
        logger.info(f"Auto-detected edge type: {edge_type}")
    
    # Auto-detect primary node type
    node_types = getattr(hetero_data, 'node_types', [])
    if not node_types:
        node_types = [k for k in hetero_data.keys() if isinstance(k, str) and hasattr(hetero_data[k], 'num_nodes')]
    primary_node_type = node_types[0] if node_types else 'user'
    
    # Build temporal adjacency using existing utility
    indptr, indices, timestamps = build_temporal_adjacency(hetero_data, edge_type)
    
    num_nodes = hetero_data[primary_node_type].num_nodes
    num_edges = len(indices)
    
    temporal_graph = TemporalGraph(
        indptr=indptr,
        indices=indices,
        timestamps=timestamps,
        num_nodes=num_nodes,
        num_edges=num_edges
    )
    
    logger.info(f"Created TemporalGraph: {num_nodes} nodes, {num_edges} edges")
    return temporal_graph

def create_temporal_dataloaders(
    hetero_data: HeteroData,
    temporal_graph: TemporalGraph,
    batch_size: int = 256,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[Any, Any, Any]:
    """
    Create temporal-aware data loaders per §PHASE_A.1
    
    Args:
        hetero_data: HeteroData with timestamps
        temporal_graph: TemporalGraph for sampling
        batch_size: batch size for training
        train_ratio: fraction for training
        val_ratio: fraction for validation
        
    Returns:
        train_loader, val_loader, test_loader with temporal constraints
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    # Auto-detect primary node type
    node_types = getattr(hetero_data, 'node_types', [])
    if not node_types:
        node_types = [k for k in hetero_data.keys() if isinstance(k, str) and hasattr(hetero_data[k], 'num_nodes')]
    primary_node_type = node_types[0] if node_types else 'user'
    
    # Get transaction data
    primary_data = hetero_data[primary_node_type]
    num_nodes = primary_data.num_nodes
    
    # Get node timestamps for temporal splitting
    node_timestamps = getattr(primary_data, 'timestamp', torch.arange(num_nodes, dtype=torch.float32))
    
    # Create temporal masks per §PHASE_A.1
    train_mask, val_mask, test_mask = create_temporal_masks(
        num_nodes=num_nodes,
        timestamps=node_timestamps,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        temporal_split=True
    )
    
    # Create datasets with node indices and corresponding timestamps
    train_indices = torch.where(train_mask)[0]
    val_indices = torch.where(val_mask)[0]
    test_indices = torch.where(test_mask)[0]
    
    train_timestamps = node_timestamps[train_indices]
    val_timestamps = node_timestamps[val_indices]
    test_timestamps = node_timestamps[test_indices]
    
    # Get labels if available
    labels = getattr(primary_data, 'y', torch.zeros(num_nodes, dtype=torch.long))
    train_labels = labels[train_indices]
    val_labels = labels[val_indices] 
    test_labels = labels[test_indices]
    
    # Create datasets
    train_dataset = TensorDataset(train_indices, train_timestamps, train_labels)
    val_dataset = TensorDataset(val_indices, val_timestamps, val_labels)
    test_dataset = TensorDataset(test_indices, test_timestamps, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Created temporal dataloaders: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    return train_loader, val_loader, test_loader

def prepare_stage6_data(
    data_path: str = "data/ellipticpp.pt",
    config: Optional[Dict[str, Any]] = None
) -> Tuple[HeteroData, TemporalGraph, Any, Any, Any]:
    """
    Complete data preparation pipeline for Stage 6 per §PHASE_A.1
    
    Args:
        data_path: path to data file
        config: Stage 6 configuration
        
    Returns:
        hetero_data: HeteroData with timestamps
        temporal_graph: TemporalGraph for sampling
        train_loader: temporal train data loader
        val_loader: temporal validation data loader  
        test_loader: temporal test data loader
    """
    if config is None:
        config = {
            'batch_size': 256,
            'train_ratio': 0.7,
            'val_ratio': 0.15
        }
    
    # Load data with timestamps
    hetero_data = load_ellipticpp_with_timestamps(data_path)
    
    # Create temporal graph
    temporal_graph = create_temporal_graph_from_hetero(hetero_data)
    
    # Create temporal data loaders
    train_loader, val_loader, test_loader = create_temporal_dataloaders(
        hetero_data=hetero_data,
        temporal_graph=temporal_graph,
        batch_size=config.get('batch_size', 256),
        train_ratio=config.get('train_ratio', 0.7),
        val_ratio=config.get('val_ratio', 0.15)
    )
    
    logger.info("Stage 6 data preparation complete")
    return hetero_data, temporal_graph, train_loader, val_loader, test_loader

# Debug utility for validating temporal data per APPENDIX
def debug_temporal_data(hetero_data: HeteroData, temporal_graph: TemporalGraph):
    """Print debug information about temporal data per APPENDIX requirements"""
    print(f"[Stage6 Data Debug]")
    print(f"  Nodes: {temporal_graph.num_nodes}")
    print(f"  Edges: {temporal_graph.num_edges}")
    
    # Check timestamp range
    edge_timestamps = temporal_graph.timestamps
    
    # Auto-detect primary node type
    node_types = getattr(hetero_data, 'node_types', [])
    if not node_types:
        node_types = [k for k in hetero_data.keys() if isinstance(k, str) and hasattr(hetero_data[k], 'num_nodes')]
    primary_node_type = node_types[0] if node_types else 'user'
    
    node_timestamps = getattr(hetero_data[primary_node_type], 'timestamp', torch.arange(temporal_graph.num_nodes, dtype=torch.float32))
    
    print(f"  Edge timestamp range: [{edge_timestamps.min():.1f}, {edge_timestamps.max():.1f}]")
    print(f"  Node timestamp range: [{node_timestamps.min():.1f}, {node_timestamps.max():.1f}]")
    
    # Check for temporal ordering per node
    ordered_count = 0
    for u in range(temporal_graph.num_nodes):
        start_idx = temporal_graph.indptr[u]
        end_idx = temporal_graph.indptr[u + 1]
        if end_idx > start_idx:
            node_edge_timestamps = edge_timestamps[start_idx:end_idx]
            if len(node_edge_timestamps) > 1:
                is_ordered = torch.all(node_edge_timestamps[:-1] >= node_edge_timestamps[1:])
                if is_ordered:
                    ordered_count += 1
    
    print(f"  Nodes with properly ordered edges: {ordered_count}/{temporal_graph.num_nodes}")
    
    # Check sparsity
    avg_degree = temporal_graph.num_edges / temporal_graph.num_nodes if temporal_graph.num_nodes > 0 else 0
    print(f"  Average degree: {avg_degree:.2f}")
    
    print(f"[Stage6 Data Debug Complete]")
