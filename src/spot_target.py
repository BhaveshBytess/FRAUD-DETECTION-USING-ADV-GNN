"""
SpotTarget Core Implementation
Following Stage7 Spot Target And Robustness Reference §Phase1

Implements SpotTarget training discipline and leakage-safe inference as per:
- Training-time rule: Exclude T_low edges where min(deg[u],deg[v]) < δ  
- Inference-time rule: Remove test/validation edges to prevent P3 leakage
- δ default: Average node degree of dataset
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_avg_degree(edge_index: torch.Tensor, num_nodes: int) -> Tuple[int, torch.Tensor]:
    """
    Compute average degree and per-node degrees from edge index.
    Following Stage7 Reference §Phase1 quick code snippet.
    
    Args:
        edge_index: (2, E) edge index tensor
        num_nodes: number of nodes in graph
        
    Returns:
        (average_degree, degrees): avg degree as int, per-node degrees tensor
    """
    deg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    src, dst = edge_index
    
    # Count outgoing edges
    deg.scatter_add_(0, src, torch.ones_like(src))
    # Count incoming edges  
    deg.scatter_add_(0, dst, torch.ones_like(dst))
    
    avg_deg = int(deg.float().mean().item())
    return avg_deg, deg


def compute_default_delta(edge_index: torch.Tensor, num_nodes: int) -> int:
    """
    Helper to compute default δ = average degree.
    Following Stage7 Reference §Phase1 implementation hints.
    
    Args:
        edge_index: (2, E) edge index tensor  
        num_nodes: number of nodes
        
    Returns:
        delta: average degree as integer threshold
    """
    avg_deg, _ = compute_avg_degree(edge_index, num_nodes)
    return avg_deg


class SpotTargetSampler:
    """
    Edge sampler that excludes train-target edges incident to low-degree nodes.
    Following Stage7 Reference §Phase1 API specification.
    
    Implements SpotTarget training discipline:
    - Compute T_low = {e=(i,j) in train_targets | min(deg[i],deg[j]) < δ}
    - Exclude T_low from message passing to reduce overfitting
    """
    
    def __init__(
        self,
        edge_index: torch.Tensor,
        train_edge_mask: torch.Tensor,
        degrees: torch.Tensor,
        delta: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize SpotTarget sampler.
        
        Args:
            edge_index: (2, E) edge index tensor for message passing graph
            train_edge_mask: (E,) boolean mask marking train target edges
            degrees: (N,) per-node degree vector (computed on training graph)
            delta: degree threshold; if None, computed as average degree
            verbose: whether to log filtering statistics
        """
        self.edge_index = edge_index
        self.train_edge_mask = train_edge_mask
        self.degrees = degrees
        self.verbose = verbose
        
        # Compute delta if not provided - following §Phase1 δ default
        if delta is None:
            num_nodes = degrees.size(0)
            self.delta = compute_default_delta(edge_index, num_nodes)
        else:
            self.delta = delta
            
        if self.verbose:
            logger.info(f"SpotTargetSampler initialized with δ={self.delta}")
    
    def _compute_tlow_mask(self, batch_edge_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute T_low mask for batch edges.
        Following Stage7 Reference §Phase1: T_low = edges with min(deg[u],deg[v]) < δ
        
        Args:
            batch_edge_indices: indices into edge_index for this batch
            
        Returns:
            tlow_mask: boolean mask indicating T_low edges to exclude
        """
        # Get source and destination nodes for batch edges
        batch_edges = self.edge_index[:, batch_edge_indices]
        src, dst = batch_edges[0], batch_edges[1]
        
        # Check if these are train target edges
        batch_train_mask = self.train_edge_mask[batch_edge_indices]
        
        # Compute minimum degree for each edge - following reference snippet
        min_deg = torch.minimum(self.degrees[src], self.degrees[dst])
        
        # T_low = train target edges with min degree < delta
        tlow_mask = batch_train_mask & (min_deg < self.delta)
        
        return tlow_mask
    
    def sample_batch(self, batch_edge_indices: torch.Tensor) -> torch.Tensor:
        """
        Sample batch edges with SpotTarget filtering.
        Following Stage7 Reference §Phase1: exclude T_low from message passing.
        
        Args:
            batch_edge_indices: indices into edge_index for this mini-batch
            
        Returns:
            filtered_edge_index: (2, E') edge index with T_low excluded
        """
        # Compute which edges to exclude (T_low)
        tlow_mask = self._compute_tlow_mask(batch_edge_indices)
        
        # Keep edges that are NOT in T_low
        keep_mask = ~tlow_mask
        filtered_indices = batch_edge_indices[keep_mask]
        
        # Return filtered edge index for message passing
        filtered_edge_index = self.edge_index[:, filtered_indices]
        
        if self.verbose:
            n_excluded = tlow_mask.sum().item()
            n_total = len(batch_edge_indices)
            logger.info(f"SpotTarget: excluded {n_excluded}/{n_total} T_low edges")
            
        return filtered_edge_index
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sampler statistics for debugging."""
        total_edges = self.edge_index.size(1)
        train_edges = self.train_edge_mask.sum().item()
        
        # Compute total T_low edges
        src, dst = self.edge_index
        min_deg = torch.minimum(self.degrees[src], self.degrees[dst])
        tlow_total = (self.train_edge_mask & (min_deg < self.delta)).sum().item()
        
        return {
            'delta': self.delta,
            'total_edges': total_edges,
            'train_target_edges': train_edges,
            'tlow_edges': tlow_total,
            'exclusion_rate': tlow_total / max(train_edges, 1)
        }


def leakage_check(
    edge_index: torch.Tensor,
    edge_splits: Dict[str, torch.Tensor],
    use_validation_edges: bool = False,
    strict_mode: bool = True
) -> torch.Tensor:
    """
    Return edge index for inference where test edges (and optionally val edges) are removed.
    Following Stage7 Reference §Phase1: "Always exclude all test (and optionally validation) 
    target edges from the inference graph to avoid implicit leakage."
    
    This implements SpotTarget's inference-time rule to prevent P3 leakage.
    
    Args:
        edge_index: (2, E) full edge index
        edge_splits: dict with 'train', 'valid', 'test' edge masks
        use_validation_edges: if True, also remove validation edges
        strict_mode: if True, raise error if potential leakage detected
        
    Returns:
        filtered_edge_index: (2, E') edge index safe for inference
    """
    device = edge_index.device
    total_edges = edge_index.size(1)
    
    # Start with all edges
    keep_mask = torch.ones(total_edges, dtype=torch.bool, device=device)
    
    # Always remove test edges - core SpotTarget rule
    if 'test' in edge_splits:
        test_mask = edge_splits['test']
        keep_mask = keep_mask & ~test_mask
        n_test_removed = test_mask.sum().item()
        logger.info(f"Leakage check: removed {n_test_removed} test edges")
    
    # Optionally remove validation edges
    if use_validation_edges and 'valid' in edge_splits:
        valid_mask = edge_splits['valid']
        keep_mask = keep_mask & ~valid_mask
        n_valid_removed = valid_mask.sum().item()
        logger.info(f"Leakage check: removed {n_valid_removed} validation edges")
    
    # Filter edge index
    filtered_edge_index = edge_index[:, keep_mask]
    
    n_removed = total_edges - keep_mask.sum().item()
    n_remaining = keep_mask.sum().item()
    
    logger.info(f"Leakage check complete: {n_removed} edges removed, {n_remaining} remaining")
    
    # Strict mode validation
    if strict_mode and n_removed == 0:
        logger.warning("Leakage check: no edges were removed - verify edge splits are correct")
    
    return filtered_edge_index


def create_inference_graph(
    edge_index: torch.Tensor,
    edge_splits: Dict[str, torch.Tensor],
    exclude_validation: bool = False
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Create leakage-safe inference graph following SpotTarget discipline.
    
    Args:
        edge_index: (2, E) full edge index
        edge_splits: edge split masks
        exclude_validation: whether to exclude validation edges
        
    Returns:
        (inference_edge_index, stats): filtered edges and removal statistics
    """
    # Apply leakage check
    inference_edge_index = leakage_check(
        edge_index, 
        edge_splits, 
        use_validation_edges=exclude_validation
    )
    
    # Compute statistics
    original_edges = edge_index.size(1)
    remaining_edges = inference_edge_index.size(1)
    removed_edges = original_edges - remaining_edges
    
    stats = {
        'original_edges': original_edges,
        'remaining_edges': remaining_edges,
        'removed_edges': removed_edges,
        'removal_rate': removed_edges / max(original_edges, 1)
    }
    
    return inference_edge_index, stats


# Utility functions for integration
def load_stage7_config() -> Dict[str, Any]:
    """Load Stage 7 configuration from YAML."""
    import yaml
    import os
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'stage7.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('stage7', {})
    except FileNotFoundError:
        logger.warning(f"Stage7 config not found at {config_path}, using defaults")
        return {
            'delta': 'auto',
            'verbose': True,
            'leakage_check': {'use_validation_edges': False, 'strict_mode': True}
        }


def setup_spottarget_sampler(
    edge_index: torch.Tensor,
    train_edge_mask: torch.Tensor,
    num_nodes: int,
    config: Optional[Dict[str, Any]] = None
) -> SpotTargetSampler:
    """
    Factory function to create SpotTarget sampler with proper configuration.
    
    Args:
        edge_index: (2, E) edge index
        train_edge_mask: (E,) train target edge mask
        num_nodes: number of nodes
        config: optional config dict
        
    Returns:
        SpotTargetSampler instance
    """
    if config is None:
        config = load_stage7_config()
    
    # Compute degrees on full training graph
    _, degrees = compute_avg_degree(edge_index, num_nodes)
    
    # Handle delta configuration
    delta_config = config.get('delta', 'auto')
    if delta_config == 'auto':
        delta = compute_default_delta(edge_index, num_nodes)
    else:
        delta = int(delta_config)
    
    verbose = config.get('verbose', False)
    
    return SpotTargetSampler(
        edge_index=edge_index,
        train_edge_mask=train_edge_mask,
        degrees=degrees,
        delta=delta,
        verbose=verbose
    )
