"""
Functional Curvature Encoding for CUSP module
Implements curvature_positional_encoding per STAGE8_CUSP_Reference §Phase3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def curvature_positional_encoding(
    node_orc: torch.Tensor,
    dC: int = 16,
    max_curvature: float = 1.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Produce curvature positional encodings based on harmonic analysis.
    
    # Implements curvature positional encoding per STAGE8_CUSP_Reference §Phase3
    
    Creates harmonic-like encodings from node curvature values to augment
    node features with geometric information about local graph structure.
    
    Args:
        node_orc: [n] tensor of node curvature values in [-1, 1]
        dC: curvature embedding dimension {8,16,32,64} per §Hyperparameters
        max_curvature: maximum expected curvature for normalization
        eps: numerical stability epsilon
        
    Returns:
        Phi: [n, dC] curvature positional encoding matrix
    """
    device = node_orc.device
    n_nodes = node_orc.shape[0]
    
    # Normalize curvatures to [0, 1] range for encoding
    normalized_orc = (node_orc + max_curvature) / (2 * max_curvature + eps)
    normalized_orc = torch.clamp(normalized_orc, 0.0, 1.0)
    
    # Create frequency components for harmonic encoding
    # Use different frequencies for different encoding dimensions
    frequencies = torch.arange(1, dC // 2 + 1, dtype=torch.float, device=device)
    frequencies = frequencies * math.pi  # Scale frequencies
    
    # Compute sine and cosine components
    curvature_expanded = normalized_orc.unsqueeze(1)  # [n, 1]
    freq_expanded = frequencies.unsqueeze(0)  # [1, dC//2]
    
    # Harmonic encoding: alternating sin/cos
    angles = curvature_expanded * freq_expanded  # [n, dC//2]
    
    sin_components = torch.sin(angles)  # [n, dC//2]
    cos_components = torch.cos(angles)  # [n, dC//2]
    
    # Interleave sin and cos components
    if dC % 2 == 0:
        # Even dimension: equal sin/cos
        Phi = torch.stack([sin_components, cos_components], dim=2)  # [n, dC//2, 2]
        Phi = Phi.view(n_nodes, dC)  # [n, dC]
    else:
        # Odd dimension: extra sin component
        Phi = torch.stack([sin_components, cos_components], dim=2)  # [n, dC//2, 2]
        Phi = Phi.view(n_nodes, dC - 1)  # [n, dC-1]
        
        # Add extra sin component for the last dimension
        extra_freq = frequencies[-1] * 1.5
        extra_sin = torch.sin(curvature_expanded.squeeze(1) * extra_freq).unsqueeze(1)
        Phi = torch.cat([Phi, extra_sin], dim=1)  # [n, dC]
    
    return Phi


def advanced_curvature_encoding(
    node_orc: torch.Tensor,
    edge_index: torch.Tensor,
    dC: int = 16,
    encoding_type: str = "harmonic",
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Advanced curvature encoding with neighborhood context.
    
    # Implements advanced curvature encoding per STAGE8_CUSP_Reference §Phase3
    
    Incorporates local neighborhood curvature statistics for richer
    positional encoding that captures both node and local structure information.
    
    Args:
        node_orc: [n] tensor of node curvature values
        edge_index: [2, m] edge connectivity for neighborhood context
        dC: curvature embedding dimension
        encoding_type: "harmonic", "polynomial", or "mixed"
        eps: numerical stability epsilon
        
    Returns:
        Phi_advanced: [n, dC] advanced curvature positional encoding
    """
    device = node_orc.device
    n_nodes = node_orc.shape[0]
    
    # Compute neighborhood curvature statistics
    neighbor_curvature_mean = torch.zeros_like(node_orc)
    neighbor_curvature_std = torch.zeros_like(node_orc)
    neighbor_count = torch.zeros_like(node_orc)
    
    # Accumulate neighbor curvatures
    for edge_idx in range(edge_index.size(1)):
        u, v = edge_index[:, edge_idx]
        neighbor_curvature_mean[u] += node_orc[v]
        neighbor_curvature_mean[v] += node_orc[u]
        neighbor_count[u] += 1
        neighbor_count[v] += 1
    
    # Compute means (avoid division by zero)
    nonzero_mask = neighbor_count > 0
    neighbor_curvature_mean[nonzero_mask] /= neighbor_count[nonzero_mask]
    
    # Compute standard deviations
    for edge_idx in range(edge_index.size(1)):
        u, v = edge_index[:, edge_idx]
        if neighbor_count[u] > 0:
            neighbor_curvature_std[u] += (node_orc[v] - neighbor_curvature_mean[u]) ** 2
        if neighbor_count[v] > 0:
            neighbor_curvature_std[v] += (node_orc[u] - neighbor_curvature_mean[v]) ** 2
    
    neighbor_curvature_std[nonzero_mask] /= neighbor_count[nonzero_mask]
    neighbor_curvature_std = torch.sqrt(neighbor_curvature_std + eps)
    
    # Create encoding based on type
    if encoding_type == "harmonic":
        # Use node curvature for base harmonic encoding
        Phi_base = curvature_positional_encoding(node_orc, dC // 2)
        
        # Use neighborhood mean for additional harmonic encoding
        Phi_neighbor = curvature_positional_encoding(neighbor_curvature_mean, dC // 2)
        
        Phi_advanced = torch.cat([Phi_base, Phi_neighbor], dim=1)
        
    elif encoding_type == "polynomial":
        # Polynomial basis encoding with controlled range
        max_degree = min(dC, 8)  # Limit polynomial degree to prevent explosion
        powers = torch.arange(1, max_degree + 1, dtype=torch.float, device=device)
        
        # Normalize inputs to [0, 1] and add small scaling
        norm_orc = (node_orc + 1) / 2  # Map [-1,1] to [0,1]
        norm_neighbor = (neighbor_curvature_mean + 1) / 2
        
        # Scale inputs to prevent large polynomial values
        norm_orc = norm_orc * 0.8 + 0.1  # Map to [0.1, 0.9]
        norm_neighbor = norm_neighbor * 0.8 + 0.1
        
        # Create polynomial features
        orc_powers = norm_orc.unsqueeze(1) ** powers.unsqueeze(0)  # [n, max_degree]
        neighbor_powers = norm_neighbor.unsqueeze(1) ** powers.unsqueeze(0)  # [n, max_degree]
        
        # Mix polynomial features
        poly_features = 0.5 * orc_powers + 0.5 * neighbor_powers
        
        # Pad or truncate to desired dimension
        if max_degree < dC:
            # Repeat features to fill dimension
            repeat_factor = dC // max_degree
            remainder = dC % max_degree
            
            repeated = poly_features.repeat(1, repeat_factor)
            if remainder > 0:
                extra = poly_features[:, :remainder]
                Phi_advanced = torch.cat([repeated, extra], dim=1)
            else:
                Phi_advanced = repeated
        else:
            Phi_advanced = poly_features[:, :dC]
        
    else:  # mixed
        # Combine harmonic and polynomial
        dC_half = dC // 2
        
        # Harmonic part
        Phi_harmonic = curvature_positional_encoding(node_orc, dC_half)
        
        # Polynomial part
        powers = torch.arange(1, dC_half + 1, dtype=torch.float, device=device)
        norm_orc = (node_orc + 1) / 2
        Phi_poly = norm_orc.unsqueeze(1) ** powers.unsqueeze(0)
        
        Phi_advanced = torch.cat([Phi_harmonic, Phi_poly], dim=1)
    
    return Phi_advanced


class CurvatureEncodingLayer(nn.Module):
    """
    Learnable curvature encoding layer.
    
    # Implements learnable curvature encoding per STAGE8_CUSP_Reference §Phase3
    
    Combines fixed positional encoding with learnable transformations
    for adaptive curvature feature extraction.
    """
    
    def __init__(
        self,
        dC: int = 16,
        hidden_dim: int = 32,
        encoding_type: str = "harmonic",
        learnable: bool = True
    ):
        """
        Initialize curvature encoding layer.
        
        Args:
            dC: output curvature embedding dimension
            hidden_dim: hidden dimension for learnable transformation
            encoding_type: type of base encoding ("harmonic", "polynomial", "mixed")
            learnable: whether to add learnable transformation
        """
        super().__init__()
        self.dC = dC
        self.encoding_type = encoding_type
        self.learnable = learnable
        
        if learnable:
            # Learnable transformation layers
            self.curvature_mlp = nn.Sequential(
                nn.Linear(dC, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, dC),
                nn.LayerNorm(dC)
            )
    
    def forward(
        self,
        node_orc: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute curvature encoding.
        
        Args:
            node_orc: [n] node curvature values
            edge_index: [2, m] optional edge connectivity for advanced encoding
            
        Returns:
            Phi: [n, dC] curvature positional encoding
        """
        if edge_index is not None:
            # Advanced encoding with neighborhood context
            Phi_base = advanced_curvature_encoding(
                node_orc, edge_index, self.dC, self.encoding_type
            )
        else:
            # Standard harmonic encoding
            Phi_base = curvature_positional_encoding(node_orc, self.dC)
        
        if self.learnable:
            # Apply learnable transformation
            Phi = self.curvature_mlp(Phi_base)
            # Residual connection
            Phi = Phi + Phi_base
        else:
            Phi = Phi_base
        
        return Phi


def validate_curvature_encoding(
    Phi: torch.Tensor,
    expected_shape: Tuple[int, int],
    eps: float = 1e-8
) -> bool:
    """
    Validate curvature encoding output.
    
    # Implements encoding validation per STAGE8_CUSP_Reference §Phase3
    
    Args:
        Phi: curvature encoding tensor
        expected_shape: expected (n_nodes, dC) shape
        eps: numerical stability threshold
        
    Returns:
        is_valid: True if validation passes
    """
    try:
        # Check shape
        if Phi.shape != expected_shape:
            print(f"Shape validation failed: expected {expected_shape}, got {Phi.shape}")
            return False
        
        # Check finite values
        if not torch.all(torch.isfinite(Phi)):
            print("Finite values validation failed: encoding contains NaN/Inf")
            return False
        
        # Check reasonable range (encodings shouldn't be too large)
        if torch.max(torch.abs(Phi)) > 100:
            print(f"Range validation failed: max absolute value {torch.max(torch.abs(Phi))}")
            return False
        
        # Check for diversity (not all zeros)
        if torch.allclose(Phi, torch.zeros_like(Phi), atol=eps):
            print("Diversity validation failed: all encodings are zero")
            return False
        
        print("Curvature encoding validation passed")
        return True
        
    except Exception as e:
        print(f"Encoding validation failed with exception: {e}")
        return False
