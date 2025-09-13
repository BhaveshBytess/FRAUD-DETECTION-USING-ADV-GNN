"""
GPR Filter Bank for CUSP module
Implements gpr_filter_bank per STAGE8_CUSP_Reference §Phase2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class GPRFilterBank(nn.Module):
    """
    GPR (Generalized PageRank) Filter Bank for spectral graph filtering.
    
    # Implements GPR filter bank per STAGE8_CUSP_Reference §Phase2
    
    Performs GPRGNN-style propagation on curvature-normalized adjacency A_tilde_n
    with L learnable filters to extract features across different spectrum parts.
    """
    
    def __init__(
        self,
        filter_count_L: int = 10,
        alpha: float = 0.3,
        dropout: float = 0.2,
        eps: float = 1e-8
    ):
        """
        Initialize GPR Filter Bank.
        
        Args:
            filter_count_L: number of filters L {5,10,15,20,25} per §Hyperparameters
            alpha: GPR propagation parameter {0.1,0.3,0.5,0.9} per §Hyperparameters
            dropout: dropout rate {0.2,0.3,0.5} per §Hyperparameters
            eps: numerical stability epsilon
        """
        super().__init__()
        self.filter_count_L = filter_count_L
        self.alpha = alpha
        self.eps = eps
        
        # Learnable GPR weights for each filter
        self.gpr_weights = nn.Parameter(torch.zeros(filter_count_L + 1))
        self.reset_parameters()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def reset_parameters(self):
        """Initialize GPR weights with proper initialization."""
        # Initialize with geometric decay as in GPR-GNN
        for i in range(self.filter_count_L + 1):
            self.gpr_weights.data[i] = self.alpha * (1 - self.alpha) ** i
            
    def forward(
        self,
        A_tilde_n: torch.sparse.FloatTensor,
        X: torch.Tensor,
        return_all_filters: bool = False
    ) -> torch.Tensor:
        """
        Apply GPR filter bank to input features.
        
        # Implements GPR propagation per STAGE8_CUSP_Reference §Phase2
        
        Args:
            A_tilde_n: [n, n] sparse normalized Cusp adjacency matrix
            X: [n, d] input node features
            return_all_filters: if True, return all L filter outputs
            
        Returns:
            Z: [n, d] or [n, d * L] filtered features (concatenated if return_all_filters)
        """
        device = X.device
        n_nodes, feat_dim = X.shape
        
        # Store propagation results for each hop
        propagation_results = []
        
        # Initial features (0-hop)
        H = X
        propagation_results.append(H)
        
        # Multi-hop propagation up to L hops
        for l in range(1, self.filter_count_L + 1):
            # Sparse-dense matrix multiplication: A_tilde_n @ H
            H = torch.sparse.mm(A_tilde_n, H)
            
            # Apply dropout for regularization
            H = self.dropout(H)
            
            # Numerical stability check
            if not torch.all(torch.isfinite(H)):
                print(f"Warning: Non-finite values detected at hop {l}")
                H = torch.where(torch.isfinite(H), H, torch.zeros_like(H))
            
            propagation_results.append(H)
        
        # Combine results using learnable GPR weights
        if return_all_filters:
            # Return concatenated filters for manifold-specific processing
            return torch.cat(propagation_results, dim=1)
        else:
            # Weighted combination using GPR weights
            weighted_sum = torch.zeros_like(X)
            
            for l, H_l in enumerate(propagation_results):
                weight = self.gpr_weights[l]
                weighted_sum += weight * H_l
                
            return weighted_sum


def gpr_filter_bank(
    A_tilde_n: torch.sparse.FloatTensor,
    X: torch.Tensor,
    filter_count_L: int = 10,
    alpha: float = 0.3,
    return_filters: bool = False
) -> torch.Tensor:
    """
    Functional interface for GPR filter bank.
    
    # Implements gpr_filter_bank per STAGE8_CUSP_Reference §Phase2
    
    Args:
        A_tilde_n: [n, n] sparse normalized Cusp adjacency 
        X: [n, d] input node features
        filter_count_L: number of filters L
        alpha: GPR propagation parameter
        return_filters: return individual filter outputs
        
    Returns:
        filtered_features: [n, d] or [n, d * L] GPR filtered features
    """
    device = X.device
    n_nodes, feat_dim = X.shape
    
    # Manual GPR computation for functional interface
    propagation_results = []
    
    # Initial features (0-hop)
    H = X
    propagation_results.append(H)
    
    # Multi-hop propagation
    for l in range(1, filter_count_L + 1):
        H = torch.sparse.mm(A_tilde_n, H)
        
        # Numerical stability
        H = torch.clamp(H, min=-1e6, max=1e6)
        propagation_results.append(H)
    
    if return_filters:
        # Return all filter outputs concatenated
        return torch.cat(propagation_results, dim=1)
    else:
        # Weighted combination with geometric weights
        weighted_sum = torch.zeros_like(X)
        
        for l, H_l in enumerate(propagation_results):
            weight = alpha * (1 - alpha) ** l
            weighted_sum += weight * H_l
            
        return weighted_sum


class ManifoldGPRFilter(nn.Module):
    """
    Manifold-aware GPR filter for product-manifold components.
    
    # Implements manifold-aware propagation per STAGE8_CUSP_Reference §Phase2
    
    Performs GPR filtering with manifold-specific operations for different
    curvature components (Hyperbolic, Spherical, Euclidean).
    """
    
    def __init__(
        self,
        manifold_type: str = "euclidean",  # "hyperbolic", "spherical", "euclidean"
        filter_count_L: int = 10,
        alpha: float = 0.3,
        eps: float = 1e-8
    ):
        """
        Initialize manifold-aware GPR filter.
        
        Args:
            manifold_type: type of manifold geometry
            filter_count_L: number of filters
            alpha: GPR propagation parameter
            eps: numerical stability epsilon
        """
        super().__init__()
        self.manifold_type = manifold_type
        self.filter_count_L = filter_count_L
        self.alpha = alpha
        self.eps = eps
        
        # Learnable weights for manifold-specific filtering
        self.manifold_weights = nn.Parameter(torch.zeros(filter_count_L + 1))
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize manifold-specific weights."""
        if self.manifold_type == "hyperbolic":
            # Hyperbolic manifolds: emphasize local structure
            for i in range(self.filter_count_L + 1):
                self.manifold_weights.data[i] = self.alpha * (1 - self.alpha) ** (i * 1.5)
        elif self.manifold_type == "spherical":
            # Spherical manifolds: emphasize global structure
            for i in range(self.filter_count_L + 1):
                self.manifold_weights.data[i] = self.alpha * (1 - self.alpha) ** (i * 0.5)
        else:  # euclidean
            # Standard GPR weights
            for i in range(self.filter_count_L + 1):
                self.manifold_weights.data[i] = self.alpha * (1 - self.alpha) ** i
                
    def forward(
        self,
        A_tilde_n: torch.sparse.FloatTensor,
        X_manifold: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply manifold-aware GPR filtering.
        
        Args:
            A_tilde_n: [n, n] sparse normalized Cusp adjacency
            X_manifold: [n, d] features in manifold space
            
        Returns:
            Z_manifold: [n, d] filtered manifold features
        """
        # Store propagation results
        propagation_results = []
        
        # Initial features
        H = X_manifold
        propagation_results.append(H)
        
        # Multi-hop propagation with manifold-specific operations
        for l in range(1, self.filter_count_L + 1):
            if self.manifold_type == "hyperbolic":
                # Hyperbolic propagation (simplified Möbius operations)
                H = torch.sparse.mm(A_tilde_n, H)
                # Apply hyperbolic normalization
                H_norm = torch.norm(H, dim=1, keepdim=True)
                H = H / (H_norm + self.eps) * torch.tanh(H_norm + self.eps)
                
            elif self.manifold_type == "spherical":
                # Spherical propagation
                H = torch.sparse.mm(A_tilde_n, H)
                # Project to unit sphere
                H = F.normalize(H, p=2, dim=1, eps=self.eps)
                
            else:  # euclidean
                # Standard Euclidean propagation
                H = torch.sparse.mm(A_tilde_n, H)
            
            propagation_results.append(H)
        
        # Weighted combination using manifold-specific weights
        weighted_sum = torch.zeros_like(X_manifold)
        
        for l, H_l in enumerate(propagation_results):
            weight = self.manifold_weights[l]
            weighted_sum += weight * H_l
            
        return weighted_sum


def validate_gpr_output(
    Z: torch.Tensor,
    expected_shape: Tuple[int, int],
    eps: float = 1e-8
) -> bool:
    """
    Validate GPR filter bank output.
    
    # Implements GPR validation per STAGE8_CUSP_Reference §Phase2 validation
    
    Args:
        Z: GPR filter output tensor
        expected_shape: expected (n_nodes, feat_dim) shape
        eps: numerical stability threshold
        
    Returns:
        is_valid: True if validation passes
    """
    try:
        # Check shape
        if Z.shape != expected_shape:
            print(f"Shape validation failed: expected {expected_shape}, got {Z.shape}")
            return False
        
        # Check finite values
        if not torch.all(torch.isfinite(Z)):
            print("Finite values validation failed: output contains NaN/Inf")
            return False
        
        # Check reasonable magnitude (not too large)
        if torch.max(torch.abs(Z)) > 1e6:
            print(f"Magnitude validation failed: max absolute value {torch.max(torch.abs(Z))}")
            return False
        
        print("GPR filter bank output validation passed")
        return True
        
    except Exception as e:
        print(f"GPR validation failed with exception: {e}")
        return False
