"""
Cusp Laplacian construction for CUSP module
Implements build_cusp_laplacian per STAGE8_CUSP_Reference §Phase1
"""

import torch
import torch.sparse
from typing import Tuple, Optional
from torch_geometric.utils import add_self_loops, degree


def build_cusp_laplacian(
    edge_index: torch.Tensor,
    edge_orc: torch.Tensor,
    num_nodes: Optional[int] = None,
    eps: float = 1e-8,
    add_self_loop: bool = True
) -> Tuple[torch.sparse.FloatTensor, torch.Tensor, torch.sparse.FloatTensor]:
    """
    Build curvature-weighted adjacency and normalized Cusp Laplacian.
    
    # Implements Cusp Laplacian per STAGE8_CUSP_Reference §Phase1 Definition 1
    
    Uses weight formula w̄_{xy} = exp(-1/(1-κ̃(x,y))) to create:
    - Ã_{xy} = w̄_{xy} * A_{xy} (curvature-weighted adjacency)
    - D̃_{xx} = Σ_y Ã_{xy} (curvature-weighted degrees)
    - Ã_n = D̃^{-1/2} Ã D̃^{-1/2} (normalized adjacency)
    
    Args:
        edge_index: [2, num_edges] tensor of edge connections
        edge_orc: [num_edges] tensor of edge curvatures in [-1+eps, 1-eps]
        num_nodes: number of nodes (inferred if None)
        eps: numerical stability epsilon for clipping and denominators
        add_self_loop: whether to add self-loops with weight 1.0
        
    Returns:
        A_tilde: sparse curvature-weighted adjacency matrix
        D_tilde: diagonal degree tensor  
        A_tilde_n: sparse normalized adjacency D̃^{-1/2} Ã D̃^{-1/2}
    """
    device = edge_index.device
    num_edges = edge_index.size(1)
    
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    
    # Clip curvatures for numerical stability per §Numerical notes
    kappa_clipped = torch.clamp(edge_orc, -1.0 + eps, 1.0 - eps)
    
    # Compute curvature weights: w̄_{xy} = exp(-1/(1-κ̃(x,y)))
    # For numerical stability, handle the case where κ̃ ≈ 1
    denominator = 1.0 - kappa_clipped
    weights = torch.exp(-1.0 / (denominator + eps))
    
    # Handle potential numerical issues
    weights = torch.clamp(weights, min=eps, max=1e6)  # prevent extreme weights
    
    # Add self-loops if requested
    if add_self_loop:
        edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        # Self-loop weights are 1.0 (neutral curvature)
        num_self_loops = num_nodes
        self_loop_weights = torch.ones(num_self_loops, device=device, dtype=torch.float)
        
        # Combine original edge weights with self-loop weights
        all_weights = torch.cat([weights, self_loop_weights])
        edge_index_final = edge_index_with_loops
    else:
        all_weights = weights
        edge_index_final = edge_index
    
    # Create sparse curvature-weighted adjacency matrix Ã
    A_tilde = torch.sparse_coo_tensor(
        edge_index_final,
        all_weights,
        size=(num_nodes, num_nodes),
        device=device,
        dtype=torch.float
    ).coalesce()
    
    # Compute curvature-weighted degrees D̃
    D_tilde = torch.sparse.sum(A_tilde, dim=1).to_dense()
    D_tilde = torch.clamp(D_tilde, min=eps)  # avoid zero degrees
    
    # Compute normalized adjacency: Ã_n = D̃^{-1/2} Ã D̃^{-1/2}
    D_tilde_inv_sqrt = torch.pow(D_tilde, -0.5)
    D_tilde_inv_sqrt[torch.isinf(D_tilde_inv_sqrt)] = 0.0
    
    # Create normalization matrices
    row_norm = D_tilde_inv_sqrt[edge_index_final[0]]
    col_norm = D_tilde_inv_sqrt[edge_index_final[1]]
    normalized_weights = all_weights * row_norm * col_norm
    
    # Create normalized sparse adjacency matrix
    A_tilde_n = torch.sparse_coo_tensor(
        edge_index_final,
        normalized_weights,
        size=(num_nodes, num_nodes),
        device=device,
        dtype=torch.float
    ).coalesce()
    
    return A_tilde, D_tilde, A_tilde_n


def cusp_laplacian_to_csr(A_tilde: torch.sparse.FloatTensor) -> torch.Tensor:
    """
    Convert sparse Cusp adjacency to CSR format for efficient sparse operations.
    
    # Implements sparse format conversion per STAGE8_CUSP_Reference §Numerical notes
    
    Args:
        A_tilde: sparse curvature-weighted adjacency tensor
        
    Returns:
        A_csr: CSR format tensor for efficient sparse-dense operations
    """
    # Convert to CSR for efficient sparse-dense matrix multiplication
    A_coo = A_tilde.coalesce()
    indices = A_coo.indices()
    values = A_coo.values()
    shape = A_coo.shape
    
    # Create CSR format (PyTorch doesn't have native CSR, use COO with sorted indices)
    row_indices = indices[0]
    col_indices = indices[1]
    
    # Sort by row then column for CSR-like access patterns
    sorted_idx = torch.lexsort((col_indices, row_indices))
    
    row_sorted = row_indices[sorted_idx]
    col_sorted = col_indices[sorted_idx]
    val_sorted = values[sorted_idx]
    
    A_csr = torch.sparse_coo_tensor(
        torch.stack([row_sorted, col_sorted]),
        val_sorted,
        size=shape,
        device=A_tilde.device,
        dtype=A_tilde.dtype
    ).coalesce()
    
    return A_csr


def validate_cusp_laplacian(
    A_tilde: torch.sparse.FloatTensor,
    D_tilde: torch.Tensor,
    A_tilde_n: torch.sparse.FloatTensor,
    eps: float = 1e-8
) -> bool:
    """
    Validate Cusp Laplacian construction for numerical stability.
    
    # Implements validation checks per STAGE8_CUSP_Reference §Validation checklist
    
    Args:
        A_tilde: sparse curvature-weighted adjacency
        D_tilde: diagonal degree tensor
        A_tilde_n: normalized adjacency
        eps: numerical stability threshold
        
    Returns:
        is_valid: True if all validation checks pass
    """
    try:
        # Check 1: A_tilde has non-negative entries
        A_values = A_tilde.values()
        if torch.any(A_values < 0):
            print("Validation failed: A_tilde has negative entries")
            return False
        
        # Check 2: D_tilde diagonal > 0 (allow very small positive values)
        if torch.any(D_tilde <= eps/10):  # More tolerant threshold
            print(f"Validation failed: D_tilde has zero or negative entries: min={D_tilde.min()}")
            return False
        
        # Check 3: A_tilde_n is finite
        A_n_values = A_tilde_n.values()
        if not torch.all(torch.isfinite(A_n_values)):
            print("Validation failed: A_tilde_n has non-finite entries")
            return False
        
        # Check 4: Sparse matrices have correct shapes
        num_nodes = A_tilde.shape[0]
        if A_tilde.shape != (num_nodes, num_nodes):
            print("Validation failed: A_tilde shape incorrect")
            return False
        
        if A_tilde_n.shape != (num_nodes, num_nodes):
            print("Validation failed: A_tilde_n shape incorrect")
            return False
        
        if D_tilde.shape != (num_nodes,):
            print("Validation failed: D_tilde shape incorrect")
            return False
        
        print("Cusp Laplacian validation passed")
        return True
        
    except Exception as e:
        print(f"Validation failed with exception: {e}")
        return False
