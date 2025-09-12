# src/sampling/cpu_fallback.py
"""
CPU Implementation of Time-Relaxed Neighbor Sampling per §PHASE_A.2
Implements exact pseudocode from STAGE6_TDGNN_GSAMPLER_REFERENCE.md
"""

import torch
import numpy as np
from typing import List, Set, Dict, Union, Tuple, Optional
from dataclasses import dataclass
import bisect
import logging

logger = logging.getLogger(__name__)

@dataclass
class TemporalGraph:
    """Temporal graph data structure for time-aware sampling per §PHASE_A.1"""
    # CSR-like structure with timestamps
    indptr: torch.Tensor  # (n+1,) node pointer array
    indices: torch.Tensor  # (nnz,) neighbor indices
    timestamps: torch.Tensor  # (nnz,) edge timestamps aligned with indices
    num_nodes: int
    num_edges: int
    
    def __post_init__(self):
        """Validate temporal graph constraints per §PHASE_A.1"""
        assert len(self.indptr) == self.num_nodes + 1
        assert len(self.indices) == self.num_edges
        assert len(self.timestamps) == self.num_edges
        
        # Validate timestamps are sorted per node (required for binary search)
        for u in range(self.num_nodes):
            start_idx = self.indptr[u]
            end_idx = self.indptr[u + 1]
            if end_idx > start_idx:
                node_timestamps = self.timestamps[start_idx:end_idx]
                # Should be sorted in descending order for recency preference
                if not torch.all(node_timestamps[:-1] >= node_timestamps[1:]):
                    logger.warning(f"Node {u} timestamps not sorted descending")

@dataclass 
class SubgraphBatch:
    """Sampled subgraph batch for training per §PHASE_C.1"""
    seed_nodes: torch.Tensor
    sub_indptr: torch.Tensor  
    sub_indices: torch.Tensor
    sub_timestamps: torch.Tensor
    node_mapping: Dict[int, int]  # original_id -> batch_id
    train_mask: torch.Tensor
    num_nodes: int
    num_edges: int

def get_neighbors_in_time_range(
    neighbors: torch.Tensor, 
    neighbor_timestamps: torch.Tensor,
    t_end: float, 
    t_start: float
) -> torch.Tensor:
    """
    Extract neighbors within time range [t_start, t_end] using binary search.
    Per §PHASE_A.2: neighbors should be pre-sorted by timestamp descending.
    
    Args:
        neighbors: neighbor node indices
        neighbor_timestamps: corresponding timestamps (sorted descending)
        t_end: maximum timestamp (evaluation time)
        t_start: minimum timestamp (t_eval - delta_t)
    
    Returns:
        filtered neighbor indices within time window
    """
    if len(neighbors) == 0:
        return torch.tensor([], dtype=neighbors.dtype)
    
    # Binary search for time window bounds
    # Since timestamps are sorted descending, we need:
    # - left bound: first index where timestamp <= t_end  
    # - right bound: last index where timestamp >= t_start
    
    timestamps_np = neighbor_timestamps.numpy()
    
    # Find first index where timestamp <= t_end (since descending sorted)
    left_idx = 0
    while left_idx < len(timestamps_np) and timestamps_np[left_idx] > t_end:
        left_idx += 1
    
    # Find last index where timestamp >= t_start
    right_idx = len(timestamps_np) - 1
    while right_idx >= 0 and timestamps_np[right_idx] < t_start:
        right_idx -= 1
    
    if left_idx > right_idx:
        return torch.tensor([], dtype=neighbors.dtype)
    
    return neighbors[left_idx:right_idx + 1]

def sample_candidates(
    candidates: torch.Tensor, 
    k: int, 
    strategy: str = 'recency'
) -> torch.Tensor:
    """
    Sample k candidates using specified strategy per §PHASE_A.2
    
    Args:
        candidates: candidate neighbor indices (already time-filtered)
        k: number of neighbors to sample
        strategy: 'recency' (top-k recent) or 'random'
    
    Returns:
        sampled neighbor indices
    """
    if len(candidates) <= k:
        return candidates
    
    if strategy == 'recency':
        # Take top-k (already sorted by recency due to time filtering)
        return candidates[:k]
    elif strategy == 'random':
        # Random sampling without replacement
        perm = torch.randperm(len(candidates))[:k]
        return candidates[perm]
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

def sample_time_relaxed_neighbors(
    node_ids: torch.Tensor,
    t_eval: Union[torch.Tensor, float], 
    depth: int,
    fanouts: List[int],
    delta_t: float,
    temporal_graph: TemporalGraph,
    strategy: str = 'recency'
) -> SubgraphBatch:
    """
    Time-Relaxed Neighbor Sampling - EXACT implementation per §PHASE_A.2 pseudocode
    
    Function: sample_time_relaxed_neighbors(node_ids, t_eval, depth, fanouts, delta_t)
    
    Args:
        node_ids: seed nodes to sample neighborhoods for (could be transaction nodes)
        t_eval: evaluation timestamp (scalar or per-seed array)  
        depth: number of hops
        fanouts: [f1, f2, ...] neighbors per hop
        delta_t: time relaxation value (>=0)
        temporal_graph: TemporalGraph with sorted adjacency lists (Batches in pseudocode)
        strategy: sampling strategy ('recency' or 'random')
    
    Returns:
        SubgraphBatch with sampled nodes and edges
    """
    # Debug prints per APPENDIX requirements
    if len(node_ids) > 0:
        t_eval_min = t_eval.min() if isinstance(t_eval, torch.Tensor) and t_eval.numel() > 0 else (t_eval if not isinstance(t_eval, torch.Tensor) else 0.0)
        t_eval_max = t_eval.max() if isinstance(t_eval, torch.Tensor) and t_eval.numel() > 0 else (t_eval if not isinstance(t_eval, torch.Tensor) else 0.0)
        print(f"[Stage6] seed_nodes={len(node_ids)} t_eval_min={t_eval_min} max={t_eval_max}")
    else:
        print(f"[Stage6] seed_nodes=0 (empty input)")
        # Handle empty input case
        return SubgraphBatch(
            seed_nodes=node_ids,
            sub_indptr=torch.zeros(1, dtype=torch.long),
            sub_indices=torch.tensor([], dtype=torch.long),
            sub_timestamps=torch.tensor([], dtype=torch.float32),
            node_mapping={},
            train_mask=torch.tensor([], dtype=torch.bool),
            num_nodes=0,
            num_edges=0
        )
    
    # Convert t_eval to per-seed array if scalar
    if isinstance(t_eval, (int, float)):
        t_eval = torch.full((len(node_ids),), t_eval, dtype=torch.float32)
    elif isinstance(t_eval, torch.Tensor) and t_eval.dim() == 0:
        t_eval = t_eval.expand(len(node_ids))
    
    assert len(t_eval) == len(node_ids), "t_eval must match node_ids length"
    assert len(fanouts) >= depth, f"fanouts length {len(fanouts)} < depth {depth}"
    
    sampled_nodes = set(node_ids.tolist())
    frontier = set(node_ids.tolist())
    
    # Track frontier sizes for debugging per APPENDIX
    frontier_sizes = []
    
    for hop in range(depth):
        next_frontier = set()
        f = fanouts[hop]
        
        for u in frontier:
            # Get evaluation time for this seed node
            u_idx = node_ids.tolist().index(u) if u in node_ids.tolist() else 0
            t_eval_u = t_eval[u_idx].item()
            
            # Extract neighbors for node u from temporal graph
            start_idx = temporal_graph.indptr[u]
            end_idx = temporal_graph.indptr[u + 1]
            
            if end_idx <= start_idx:
                continue  # No neighbors
            
            neighbors = temporal_graph.indices[start_idx:end_idx]
            neighbor_timestamps = temporal_graph.timestamps[start_idx:end_idx]
            
            # Candidate neighbors: neighbors v with timestamp <= t_eval(u) and timestamp >= t_eval(u) - delta_t
            # Use binary search for time range extraction per §PHASE_A.2
            candidates = get_neighbors_in_time_range(
                neighbors, neighbor_timestamps,
                t_end=t_eval_u, 
                t_start=t_eval_u - delta_t
            )
            
            # Sample candidates per strategy
            # If len(candidates) <= f: take all; else sample = top-k by recency if prefer recency else random sample
            sampled_neighbors = sample_candidates(candidates, k=f, strategy=strategy)
            next_frontier.update(sampled_neighbors.tolist())
        
        sampled_nodes.update(next_frontier)
        frontier = next_frontier
        frontier_sizes.append(len(frontier))
        
        # Debug prints per APPENDIX
        print(f"[TDGNN] hop={hop} frontier_size={len(frontier)}")
    
    # Debug frontier statistics per APPENDIX  
    max_frontier_size = max(frontier_sizes) if frontier_sizes else 0
    avg_frontier_size = np.mean(frontier_sizes) if frontier_sizes else 0
    print(f"[TDGNN] max_frontier={max_frontier_size} avg_frontier={avg_frontier_size:.1f}")
    
    # Convert to induced subgraph
    sampled_node_list = sorted(list(sampled_nodes))
    node_mapping = {orig_id: new_id for new_id, orig_id in enumerate(sampled_node_list)}
    
    # Build subgraph CSR structure
    sub_edges = []
    sub_timestamps = []
    
    for orig_u in sampled_node_list:
        start_idx = temporal_graph.indptr[orig_u]
        end_idx = temporal_graph.indptr[orig_u + 1]
        
        for edge_idx in range(start_idx, end_idx):
            orig_v = temporal_graph.indices[edge_idx].item()
            if orig_v in node_mapping:
                new_u = node_mapping[orig_u]
                new_v = node_mapping[orig_v]
                sub_edges.append((new_u, new_v))
                sub_timestamps.append(temporal_graph.timestamps[edge_idx].item())
    
    # Build CSR for subgraph
    num_sub_nodes = len(sampled_node_list)
    sub_indptr = torch.zeros(num_sub_nodes + 1, dtype=torch.long)
    
    if sub_edges:
        sub_edges_tensor = torch.tensor(sub_edges)
        sub_indices = sub_edges_tensor[:, 1]
        sub_timestamps_tensor = torch.tensor(sub_timestamps)
        
        # Count edges per node
        for u, v in sub_edges:
            sub_indptr[u + 1] += 1
        
        # Cumulative sum to get pointers
        torch.cumsum(sub_indptr, dim=0, out=sub_indptr)
    else:
        sub_indices = torch.tensor([], dtype=torch.long)
        sub_timestamps_tensor = torch.tensor([], dtype=torch.float32)
    
    # Create train mask for seed nodes
    train_mask = torch.zeros(num_sub_nodes, dtype=torch.bool)
    for orig_seed in node_ids:
        if orig_seed.item() in node_mapping:
            train_mask[node_mapping[orig_seed.item()]] = True
    
    return SubgraphBatch(
        seed_nodes=node_ids,
        sub_indptr=sub_indptr,
        sub_indices=sub_indices,
        sub_timestamps=sub_timestamps_tensor,
        node_mapping=node_mapping,
        train_mask=train_mask,
        num_nodes=num_sub_nodes,
        num_edges=len(sub_edges)
    )
