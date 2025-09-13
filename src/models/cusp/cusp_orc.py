"""
Ollivier-Ricci Curvature (ORC) computation for CUSP module
Implements compute_orc per STAGE8_CUSP_Reference §Phase1
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix, degree
from scipy.sparse import csr_matrix
import warnings


def compute_orc(
    edge_index: torch.Tensor, 
    num_nodes: int,
    delta: float = 0.2,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Ollivier-Ricci curvature per edge and per node.
    
    # Implements compute_orc per STAGE8_CUSP_Reference §Phase1
    
    Computes Wasserstein-1 distance between neighbor distributions for each edge,
    then derives discrete Ricci curvature. Provides CPU reference implementation
    with numerical stability safeguards.
    
    Args:
        edge_index: [2, num_edges] tensor of edge connections
        num_nodes: number of nodes in the graph
        delta: neighborhood mass parameter (default 0.2 per paper)
        eps: numerical stability epsilon (default 1e-8)
        
    Returns:
        edge_orc: [num_edges] tensor of edge curvatures in [-1+eps, 1-eps]
        node_orc: [num_nodes] tensor of node curvatures (averaged from incident edges)
    """
    device = edge_index.device
    num_edges = edge_index.size(1)
    
    # Convert to scipy for efficient neighborhood operations
    adj_scipy = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    # Convert to CSR format for efficient row slicing
    adj_scipy = adj_scipy.tocsr()
    
    # Compute node degrees
    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    
    # Initialize edge curvatures
    edge_orc = torch.zeros(num_edges, device=device, dtype=torch.float)
    
    # Process each edge
    for edge_idx in range(num_edges):
        u, v = edge_index[:, edge_idx].cpu().numpy()
        
        # Get neighborhoods (including self-loops with mass delta)
        nbrs_u = adj_scipy[u].nonzero()[1]
        nbrs_v = adj_scipy[v].nonzero()[1]
        
        # Add self-loops to neighborhoods
        nbrs_u = np.concatenate([nbrs_u, [u]])
        nbrs_v = np.concatenate([nbrs_v, [v]])
        
        # Create probability distributions with delta mass on central node
        deg_u = max(deg[u].item(), eps)
        deg_v = max(deg[v].item(), eps)
        
        # Distribution for node u: delta mass on u, (1-delta) distributed on neighbors
        prob_u = {}
        for w in nbrs_u:
            if w == u:
                prob_u[w] = delta + (1 - delta) / deg_u
            else:
                prob_u[w] = (1 - delta) / deg_u
                
        # Distribution for node v: delta mass on v, (1-delta) distributed on neighbors  
        prob_v = {}
        for w in nbrs_v:
            if w == v:
                prob_v[w] = delta + (1 - delta) / deg_v
            else:
                prob_v[w] = (1 - delta) / deg_v
        
        # Compute Wasserstein-1 distance (approximation)
        # Get all nodes involved in either distribution
        all_nodes = set(nbrs_u) | set(nbrs_v)
        
        # Create distance matrix (hop distance approximation)
        wasserstein_dist = 0.0
        for w in all_nodes:
            mass_u = prob_u.get(w, 0.0)
            mass_v = prob_v.get(w, 0.0)
            
            # Simple hop distance (can be improved with shortest path)
            if w in prob_u and w in prob_v:
                dist_w = 0.0  # same node
            elif w in prob_u:
                dist_w = 1.0  # one hop from v
            else:  # w in prob_v
                dist_w = 1.0  # one hop from u
                
            wasserstein_dist += abs(mass_u - mass_v) * dist_w
        
        # Ricci curvature: κ(u,v) = 1 - W(μ_u, μ_v)
        curvature = 1.0 - wasserstein_dist
        
        # Clip for numerical stability per §Numerical notes
        curvature = max(-1.0 + eps, min(1.0 - eps, curvature))
        
        edge_orc[edge_idx] = curvature
    
    # Compute node curvatures as average of incident edge curvatures
    node_orc = torch.zeros(num_nodes, device=device, dtype=torch.float)
    node_counts = torch.zeros(num_nodes, device=device, dtype=torch.float)
    
    # Accumulate curvatures for each node
    for edge_idx in range(num_edges):
        u, v = edge_index[:, edge_idx]
        curv = edge_orc[edge_idx]
        
        node_orc[u] += curv
        node_orc[v] += curv
        node_counts[u] += 1
        node_counts[v] += 1
    
    # Average (avoid division by zero)
    nonzero_mask = node_counts > 0
    node_orc[nonzero_mask] /= node_counts[nonzero_mask]
    
    return edge_orc, node_orc


def compute_orc_fast_approximation(
    edge_index: torch.Tensor,
    num_nodes: int,
    delta: float = 0.2,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast approximation of ORC using degree-based heuristic.
    
    # Implements fast ORC approximation per STAGE8_CUSP_Reference §Error Handling
    
    For large graphs where full ORC computation is too slow,
    this provides a degree-based approximation that captures
    local connectivity patterns.
    
    Args:
        edge_index: [2, num_edges] tensor of edge connections
        num_nodes: number of nodes in the graph
        delta: neighborhood mass parameter
        eps: numerical stability epsilon
        
    Returns:
        edge_orc: [num_edges] tensor of approximate edge curvatures
        node_orc: [num_nodes] tensor of approximate node curvatures
    """
    device = edge_index.device
    num_edges = edge_index.size(1)
    
    # Compute node degrees
    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    deg = torch.clamp(deg, min=eps)  # avoid division by zero
    
    # Edge curvature approximation based on degree difference
    u_indices, v_indices = edge_index[0], edge_index[1]
    deg_u = deg[u_indices]
    deg_v = deg[v_indices]
    
    # Approximate curvature: higher for edges connecting similar-degree nodes
    deg_ratio = torch.min(deg_u, deg_v) / (torch.max(deg_u, deg_v) + eps)
    edge_orc = 2 * deg_ratio - 1  # map [0,1] to [-1,1]
    
    # Add some randomness based on delta parameter
    noise = delta * (torch.rand_like(edge_orc) - 0.5)
    edge_orc = torch.clamp(edge_orc + noise, -1.0 + eps, 1.0 - eps)
    
    # Node curvatures as average of incident edges
    node_orc = torch.zeros(num_nodes, device=device, dtype=torch.float)
    node_counts = torch.zeros(num_nodes, device=device, dtype=torch.float)
    
    for edge_idx in range(num_edges):
        u, v = edge_index[:, edge_idx]
        curv = edge_orc[edge_idx]
        
        node_orc[u] += curv
        node_orc[v] += curv
        node_counts[u] += 1
        node_counts[v] += 1
    
    nonzero_mask = node_counts > 0
    node_orc[nonzero_mask] /= node_counts[nonzero_mask]
    
    return edge_orc, node_orc
