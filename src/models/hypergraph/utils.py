"""
Utility functions for PhenomNN-based hypergraph neural networks

Implements core mathematical operations and validation functions according to the
"From Hypergraph Energy Functions to Hypergraph Neural Networks" paper.

Key functions:
- Matrix computation helpers (degree matrices, expansions)
- Hypergraph structure validation
- Debugging and visualization utilities
- Performance optimization helpers
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings


def compute_degree_matrices(incidence_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute degree matrices for hypergraph following PhenomNN paper.
    
    Args:
        incidence_matrix: Binary incidence matrix B (n_nodes × n_hyperedges)
        
    Returns:
        Tuple of (DH, DC, DS_bar) degree matrices:
        - DH: Hyperedge degree matrix diag(B.sum(dim=0))
        - DC: Clique node degree matrix diag(AC.sum(dim=1))  
        - DS_bar: Star node degree matrix diag(AS_bar.sum(dim=1))
    """
    device = incidence_matrix.device
    
    # Hyperedge degree matrix: DH = diag(B.sum(dim=0))
    hyperedge_degrees = incidence_matrix.sum(dim=0)  # Sum over nodes
    hyperedge_degrees = torch.clamp(hyperedge_degrees, min=1e-8)  # Prevent singularity
    DH = torch.diag(hyperedge_degrees)
    
    # Compute expansion matrices to get node degrees
    AC, AS_bar = compute_expansion_matrices(incidence_matrix, DH)
    
    # Clique node degree matrix: DC = diag(AC.sum(dim=1))
    clique_degrees = AC.sum(dim=1)
    DC = torch.diag(clique_degrees)
    
    # Star node degree matrix: DS_bar = diag(AS_bar.sum(dim=1))
    star_degrees = AS_bar.sum(dim=1)
    DS_bar = torch.diag(star_degrees)
    
    return DH, DC, DS_bar


def compute_expansion_matrices(
    incidence_matrix: torch.Tensor, 
    DH: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute clique and star expansion matrices following PhenomNN paper.
    
    Mathematical formulation:
    - Clique expansion: AC = B @ inv(DH) @ B.T
    - Star expansion: AS_bar = B @ inv(DH) @ B.T (same for simplified version)
    
    Args:
        incidence_matrix: Binary incidence matrix B (n_nodes × n_hyperedges)
        DH: Hyperedge degree matrix, computed if not provided
        
    Returns:
        Tuple of (AC, AS_bar) expansion matrices
    """
    B = incidence_matrix
    device = B.device
    
    # Compute DH if not provided
    if DH is None:
        hyperedge_degrees = B.sum(dim=0)
        hyperedge_degrees = torch.clamp(hyperedge_degrees, min=1e-8)
        DH = torch.diag(hyperedge_degrees)
    
    # Add regularization to prevent singular matrix
    epsilon = 1e-8
    DH_reg = DH + epsilon * torch.eye(DH.shape[0], device=device)
    
    # Compute DH inverse
    try:
        DH_inv = torch.inverse(DH_reg)
    except RuntimeError as e:
        warnings.warn(f"Matrix inversion failed, using pseudo-inverse: {e}")
        DH_inv = torch.pinverse(DH_reg)
    
    # Clique expansion: AC = B @ inv(DH) @ B.T
    AC = B @ DH_inv @ B.T
    
    # Star expansion: AS_bar = B @ inv(DH) @ B.T (simplified version)
    # In the full PhenomNN formulation, this could be different
    AS_bar = AC  # Simplified: same as clique expansion
    
    return AC, AS_bar


def compute_preconditioner(
    DC: torch.Tensor,
    DS_bar: torch.Tensor, 
    lambda0: float = 1.0,
    lambda1: float = 1.0,
    regularization: float = 1e-8
) -> torch.Tensor:
    """
    Compute preconditioner matrix D̃ = λ0*DC + λ1*DS_bar + I
    
    Args:
        DC: Clique node degree matrix
        DS_bar: Star node degree matrix
        lambda0: Weight for clique expansion
        lambda1: Weight for star expansion  
        regularization: Regularization parameter for numerical stability
        
    Returns:
        Preconditioner matrix D̃
    """
    n_nodes = DC.shape[0]
    device = DC.device
    
    I = torch.eye(n_nodes, device=device)
    D_tilde = lambda0 * DC + lambda1 * DS_bar + I
    
    # Add small regularization to diagonal for numerical stability
    D_tilde += regularization * I
    
    return D_tilde


def validate_hypergraph_structure(
    incidence_matrix: torch.Tensor,
    node_features: torch.Tensor,
    hyperedge_features: Optional[torch.Tensor] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive validation of hypergraph structure and data consistency.
    
    Args:
        incidence_matrix: Binary incidence matrix B
        node_features: Node feature matrix X
        hyperedge_features: Optional hyperedge feature matrix U
        verbose: Whether to print validation results
        
    Returns:
        Dictionary with validation results and statistics
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    B = incidence_matrix
    X = node_features
    U = hyperedge_features
    
    # 1. Dimension consistency checks
    if B.shape[0] != X.shape[0]:
        results['valid'] = False
        results['errors'].append(f"Incidence matrix rows ({B.shape[0]}) != node features rows ({X.shape[0]})")
    
    if U is not None and B.shape[1] != U.shape[0]:
        results['valid'] = False
        results['errors'].append(f"Incidence matrix cols ({B.shape[1]}) != hyperedge features rows ({U.shape[0]})")
    
    # 2. Binary incidence matrix check
    if not torch.all((B == 0) | (B == 1)):
        results['warnings'].append("Incidence matrix is not binary")
    
    # 3. Empty hyperedge check
    hyperedge_sizes = B.sum(dim=0)
    empty_hyperedges = (hyperedge_sizes == 0).sum().item()
    if empty_hyperedges > 0:
        results['valid'] = False
        results['errors'].append(f"Found {empty_hyperedges} empty hyperedges")
    
    # 4. Isolated node check
    node_degrees = B.sum(dim=1) 
    isolated_nodes = (node_degrees == 0).sum().item()
    if isolated_nodes > 0:
        results['warnings'].append(f"Found {isolated_nodes} isolated nodes")
    
    # 5. NaN/Inf checks
    if not torch.isfinite(B).all():
        results['valid'] = False
        results['errors'].append("Incidence matrix contains NaN/Inf values")
    
    if not torch.isfinite(X).all():
        results['valid'] = False
        results['errors'].append("Node features contain NaN/Inf values")
    
    if U is not None and not torch.isfinite(U).all():
        results['valid'] = False
        results['errors'].append("Hyperedge features contain NaN/Inf values")
    
    # 6. Compute statistics
    results['statistics'] = {
        'n_nodes': B.shape[0],
        'n_hyperedges': B.shape[1],
        'n_node_features': X.shape[1],
        'n_hyperedge_features': U.shape[1] if U is not None else 0,
        'avg_hyperedge_size': hyperedge_sizes.float().mean().item(),
        'max_hyperedge_size': hyperedge_sizes.max().item(),
        'min_hyperedge_size': hyperedge_sizes.min().item(),
        'avg_node_degree': node_degrees.float().mean().item(),
        'max_node_degree': node_degrees.max().item(),
        'min_node_degree': node_degrees.min().item(),
        'density': (B.sum() / (B.shape[0] * B.shape[1])).item(),
        'empty_hyperedges': empty_hyperedges,
        'isolated_nodes': isolated_nodes
    }
    
    # 7. Matrix computation validation
    try:
        DH, DC, DS_bar = compute_degree_matrices(B)
        AC, AS_bar = compute_expansion_matrices(B, DH)
        
        results['statistics']['degree_matrix_shapes'] = {
            'DH': list(DH.shape),
            'DC': list(DC.shape), 
            'DS_bar': list(DS_bar.shape),
            'AC': list(AC.shape),
            'AS_bar': list(AS_bar.shape)
        }
        
        # Check for positive definiteness
        if torch.any(torch.diag(DH) <= 0):
            results['errors'].append("DH matrix has non-positive diagonal elements")
        if torch.any(torch.diag(DC) < 0):
            results['warnings'].append("DC matrix has negative diagonal elements")
        if torch.any(torch.diag(DS_bar) < 0):
            results['warnings'].append("DS_bar matrix has negative diagonal elements")
            
    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Matrix computation failed: {str(e)}")
    
    if verbose:
        print(f"Hypergraph Validation Results:")
        print(f"Valid: {results['valid']}")
        if results['errors']:
            print(f"Errors: {results['errors']}")
        if results['warnings']:
            print(f"Warnings: {results['warnings']}")
        print(f"Statistics: {results['statistics']}")
    
    return results


def debug_hypergraph_matrices(
    incidence_matrix: torch.Tensor,
    lambda0: float = 1.0,
    lambda1: float = 1.0
) -> Dict[str, Any]:
    """
    Debug helper to analyze hypergraph matrix properties.
    
    Args:
        incidence_matrix: Binary incidence matrix B
        lambda0: Weight for clique expansion
        lambda1: Weight for star expansion
        
    Returns:
        Dictionary with matrix analysis results
    """
    B = incidence_matrix
    
    # Compute all matrices
    DH, DC, DS_bar = compute_degree_matrices(B)
    AC, AS_bar = compute_expansion_matrices(B, DH)
    D_tilde = compute_preconditioner(DC, DS_bar, lambda0, lambda1)
    
    debug_info = {
        'matrix_shapes': {
            'B': list(B.shape),
            'DH': list(DH.shape),
            'DC': list(DC.shape),
            'DS_bar': list(DS_bar.shape), 
            'AC': list(AC.shape),
            'AS_bar': list(AS_bar.shape),
            'D_tilde': list(D_tilde.shape)
        },
        'matrix_properties': {
            'B_sparsity': 1.0 - (B.sum() / B.numel()).item(),
            'AC_sparsity': 1.0 - (AC.nonzero().shape[0] / AC.numel()),
            'AS_bar_sparsity': 1.0 - (AS_bar.nonzero().shape[0] / AS_bar.numel()),
            'DH_min_eigenvalue': torch.min(torch.diag(DH)).item(),
            'DC_min_eigenvalue': torch.min(torch.diag(DC)).item(),
            'DS_bar_min_eigenvalue': torch.min(torch.diag(DS_bar)).item(),
            'D_tilde_condition_number': torch.linalg.cond(D_tilde).item() if D_tilde.shape[0] < 1000 else float('inf')
        },
        'numerical_stability': {
            'DH_singular': torch.any(torch.diag(DH) <= 1e-10).item(),
            'D_tilde_singular': torch.any(torch.diag(D_tilde) <= 1e-10).item(),
            'AC_symmetric': torch.allclose(AC, AC.T, atol=1e-6),
            'AS_bar_symmetric': torch.allclose(AS_bar, AS_bar.T, atol=1e-6)
        }
    }
    
    return debug_info


def optimize_hypergraph_memory(
    incidence_matrix: torch.Tensor,
    use_sparse: bool = True,
    max_hyperedges: Optional[int] = None
) -> torch.Tensor:
    """
    Optimize hypergraph for memory efficiency.
    
    Args:
        incidence_matrix: Dense incidence matrix
        use_sparse: Whether to convert to sparse representation
        max_hyperedges: Maximum number of hyperedges to keep
        
    Returns:
        Optimized incidence matrix
    """
    B = incidence_matrix
    
    # Limit number of hyperedges if specified
    if max_hyperedges is not None and B.shape[1] > max_hyperedges:
        # Keep hyperedges with most connections
        hyperedge_sizes = B.sum(dim=0)
        _, top_indices = torch.topk(hyperedge_sizes, k=max_hyperedges)
        B = B[:, top_indices]
    
    # Convert to sparse if requested and beneficial
    if use_sparse and B.sum() / B.numel() < 0.1:  # Less than 10% dense
        # Convert to sparse COO format
        B = B.to_sparse()
    
    return B


def get_hypergraph_statistics_summary(validation_results: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of hypergraph statistics.
    
    Args:
        validation_results: Results from validate_hypergraph_structure()
        
    Returns:
        Formatted string summary
    """
    stats = validation_results['statistics']
    
    summary = f"""
Hypergraph Statistics Summary:
=============================
Structure:
  - Nodes: {stats['n_nodes']:,}
  - Hyperedges: {stats['n_hyperedges']:,}
  - Node features: {stats['n_node_features']}
  - Density: {stats['density']:.4f}

Hyperedge Analysis:
  - Average size: {stats['avg_hyperedge_size']:.2f}
  - Size range: [{stats['min_hyperedge_size']}, {stats['max_hyperedge_size']}]
  - Empty hyperedges: {stats['empty_hyperedges']}

Node Analysis:
  - Average degree: {stats['avg_node_degree']:.2f} 
  - Degree range: [{stats['min_node_degree']}, {stats['max_node_degree']}]
  - Isolated nodes: {stats['isolated_nodes']}

Validation:
  - Valid structure: {validation_results['valid']}
  - Errors: {len(validation_results['errors'])}
  - Warnings: {len(validation_results['warnings'])}
"""
    
    return summary.strip()
