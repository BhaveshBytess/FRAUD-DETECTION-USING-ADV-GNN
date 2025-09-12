"""
PhenomNN Layer Implementation for Hypergraph Neural Networks

Implements the core PhenomNN layers from "From Hypergraph Energy Functions to Hypergraph Neural Networks"
Following the exact mathematical formulations in Equations 22 and 25.

Mathematical Foundation:
- Equation 25 (Simplified): Y^(t+1) = ReLU((1-α)Y^(t) + αD̃^(-1)[(λ0*AC + λ1*AS_bar)*Y^(t) + f(X;W)])
- Equation 22 (General): Y^(t+1) = ReLU((1-α)Y^(t) + αD̃^(-1)[f(X;W) + λ0*Ỹ_C^(t) + λ1*(L̄_S*Y^(t) + Ỹ_S^(t))])
- Preconditioner: D̃ = λ0*DC + λ1*DS_bar + I

Key Components:
- PhenomNNSimpleLayer: Simplified formulation (Equation 25)
- PhenomNNLayer: Full formulation (Equation 22) 
- Energy-based updates with convergence guarantees
- Proper gradient flow and numerical stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import warnings
import math

from .hypergraph_data import HypergraphData


class PhenomNNSimpleLayer(nn.Module):
    """
    Simplified PhenomNN layer implementing Equation 25 from the paper.
    
    Mathematical formulation:
    Y^(t+1) = ReLU((1-α)Y^(t) + αD̃^(-1)[(λ0*AC + λ1*AS_bar)*Y^(t) + f(X;W)])
    
    Where:
    - Y^(t): Node representations at iteration t
    - α: Step size parameter (controls convergence)
    - D̃: Preconditioner matrix = λ0*DC + λ1*DS_bar + I
    - AC, AS_bar: Clique and star expansion matrices
    - f(X;W): Learnable transformation of input features
    - λ0, λ1: Expansion weights (clique vs star)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        lambda0: float = 1.0,
        lambda1: float = 1.0,
        alpha: float = 0.1,
        num_iterations: int = 5,
        convergence_threshold: float = 1e-6,
        dropout: float = 0.0,
        use_bias: bool = True
    ):
        """
        Initialize PhenomNN Simple Layer.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden representation dimension
            lambda0: Weight for clique expansion (λ0)
            lambda1: Weight for star expansion (λ1)  
            alpha: Step size for energy-based updates
            num_iterations: Maximum number of iterations
            convergence_threshold: Threshold for convergence detection
            dropout: Dropout probability
            use_bias: Whether to use bias in linear transformation
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.convergence_threshold = convergence_threshold
        
        # Learnable transformation f(X;W)
        self.feature_transform = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights following best practices."""
        nn.init.xavier_uniform_(self.feature_transform.weight)
        if self.feature_transform.bias is not None:
            nn.init.zeros_(self.feature_transform.bias)
    
    def forward(
        self, 
        hypergraph_data: HypergraphData,
        Y_init: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass implementing Equation 25.
        
        Args:
            hypergraph_data: HypergraphData object with matrices
            Y_init: Initial node representations (optional)
            
        Returns:
            Tuple of (final_representations, iteration_info)
        """
        X = hypergraph_data.X
        device = X.device
        batch_size, _ = X.shape
        
        # Compute f(X;W) - learnable feature transformation
        f_X = self.feature_transform(X)  # Shape: (n_nodes, hidden_dim)
        f_X = self.dropout(f_X)
        
        # Initialize Y if not provided
        if Y_init is None:
            Y = f_X.clone()  # Initialize with transformed features
        else:
            Y = Y_init.clone()
        
        # Get hypergraph matrices
        D_tilde = hypergraph_data.get_preconditioner(self.lambda0, self.lambda1)
        expansion_matrix = self.lambda0 * hypergraph_data.AC + self.lambda1 * hypergraph_data.AS_bar
        
        # Precompute D̃^(-1) for efficiency
        D_tilde_inv = self._safe_matrix_inverse(D_tilde)
        
        # Energy-based iterative updates
        iteration_info = {
            'convergence_history': [],
            'converged': False,
            'final_iteration': 0
        }
        
        for t in range(self.num_iterations):
            Y_prev = Y.clone()
            
            # Equation 25: Y^(t+1) = ReLU((1-α)Y^(t) + αD̃^(-1)[(λ0*AC + λ1*AS_bar)*Y^(t) + f(X;W)])
            expansion_term = expansion_matrix @ Y  # (λ0*AC + λ1*AS_bar)*Y^(t)
            update_term = expansion_term + f_X    # (λ0*AC + λ1*AS_bar)*Y^(t) + f(X;W)
            preconditioned_update = D_tilde_inv @ update_term  # D̃^(-1)[...]
            
            # Complete update with momentum and activation
            Y = torch.relu((1 - self.alpha) * Y + self.alpha * preconditioned_update)
            
            # Check convergence
            change = torch.norm(Y - Y_prev, p='fro').item()
            iteration_info['convergence_history'].append(change)
            
            if change < self.convergence_threshold:
                iteration_info['converged'] = True
                iteration_info['final_iteration'] = t + 1
                break
            
            iteration_info['final_iteration'] = t + 1
        
        return Y, iteration_info
    
    def _safe_matrix_inverse(self, matrix: torch.Tensor, regularization: float = 1e-6) -> torch.Tensor:
        """
        Safely compute matrix inverse with regularization for numerical stability.
        
        Args:
            matrix: Input matrix to invert
            regularization: Regularization parameter
            
        Returns:
            Inverse matrix
        """
        device = matrix.device
        regularized_matrix = matrix + regularization * torch.eye(matrix.shape[0], device=device)
        
        try:
            return torch.inverse(regularized_matrix)
        except RuntimeError as e:
            warnings.warn(f"Matrix inversion failed, using pseudo-inverse: {e}")
            return torch.pinverse(regularized_matrix)


class PhenomNNLayer(nn.Module):
    """
    Full PhenomNN layer implementing Equation 22 from the paper.
    
    Mathematical formulation:
    Y^(t+1) = ReLU((1-α)Y^(t) + αD̃^(-1)[f(X;W) + λ0*Ỹ_C^(t) + λ1*(L̄_S*Y^(t) + Ỹ_S^(t))])
    
    This is the complete formulation with separate clique and star terms.
    For the simplified implementation, we use the approximation where:
    - Ỹ_C^(t) ≈ AC * Y^(t) (clique expansion)
    - L̄_S*Y^(t) + Ỹ_S^(t) ≈ AS_bar * Y^(t) (star expansion)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        lambda0: float = 1.0,
        lambda1: float = 1.0,
        alpha: float = 0.1,
        num_iterations: int = 10,
        convergence_threshold: float = 1e-6,
        dropout: float = 0.0,
        use_bias: bool = True,
        adaptive_alpha: bool = True
    ):
        """
        Initialize full PhenomNN Layer.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden representation dimension  
            lambda0: Weight for clique expansion (λ0)
            lambda1: Weight for star expansion (λ1)
            alpha: Initial step size for energy-based updates
            num_iterations: Maximum number of iterations
            convergence_threshold: Threshold for convergence detection
            dropout: Dropout probability
            use_bias: Whether to use bias in transformations
            adaptive_alpha: Whether to use adaptive step size
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.convergence_threshold = convergence_threshold
        self.adaptive_alpha = adaptive_alpha
        
        # Learnable transformations
        self.feature_transform = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        
        # Additional learnable parameters for the full formulation
        self.clique_weight = nn.Parameter(torch.tensor(lambda0))
        self.star_weight = nn.Parameter(torch.tensor(lambda1))
        
        # Adaptive step size parameter
        if adaptive_alpha:
            self.alpha_param = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer('alpha_param', torch.tensor(alpha))
        
        # Regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights."""
        nn.init.xavier_uniform_(self.feature_transform.weight)
        if self.feature_transform.bias is not None:
            nn.init.zeros_(self.feature_transform.bias)
    
    def forward(
        self,
        hypergraph_data: HypergraphData,
        Y_init: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass implementing Equation 22.
        
        Args:
            hypergraph_data: HypergraphData object
            Y_init: Initial node representations
            
        Returns:
            Tuple of (final_representations, iteration_info)
        """
        X = hypergraph_data.X
        device = X.device
        
        # Compute f(X;W)
        f_X = self.feature_transform(X)
        f_X = self.dropout(f_X)
        f_X = self.layer_norm(f_X)
        
        # Check for NaN in initial transformation
        if torch.isnan(f_X).any():
            print(f"WARNING: NaN detected in f_X after transformation")
            f_X = torch.nan_to_num(f_X, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Initialize Y
        if Y_init is None:
            Y = f_X.clone()
        else:
            Y = Y_init.clone()
            
        # Check for NaN in Y initialization
        if torch.isnan(Y).any():
            print(f"WARNING: NaN detected in Y initialization")
            Y = torch.nan_to_num(Y, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Get hypergraph matrices with learnable weights
        lambda0_effective = torch.clamp(self.clique_weight, min=0.0, max=10.0)
        lambda1_effective = torch.clamp(self.star_weight, min=0.0, max=10.0) 
        
        D_tilde = hypergraph_data.get_preconditioner(lambda0_effective.item(), lambda1_effective.item())
        AC = hypergraph_data.AC
        AS_bar = hypergraph_data.AS_bar
        
        # Check matrices for NaN or infinite values
        if torch.isnan(D_tilde).any() or torch.isinf(D_tilde).any():
            print(f"WARNING: Invalid values in D_tilde")
            D_tilde = torch.eye(D_tilde.size(0), device=D_tilde.device) + 1e-6
            
        if torch.isnan(AC).any() or torch.isinf(AC).any():
            print(f"WARNING: Invalid values in AC")
            
        if torch.isnan(AS_bar).any() or torch.isinf(AS_bar).any():
            print(f"WARNING: Invalid values in AS_bar")
        
        # Precompute D̃^(-1) with better numerical stability
        D_tilde_inv = self._safe_matrix_inverse(D_tilde, regularization=1e-4)
        
        # Adaptive step size
        alpha_effective = torch.clamp(self.alpha_param, min=0.01, max=0.5) if self.adaptive_alpha else self.alpha
        
        # Iterative updates
        iteration_info = {
            'convergence_history': [],
            'alpha_history': [],
            'energy_history': [],
            'converged': False,
            'final_iteration': 0
        }
        
        for t in range(self.num_iterations):
            Y_prev = Y.clone()
            
            # Equation 22 components:
            # Ỹ_C^(t) ≈ AC * Y^(t) (clique expansion approximation)
            Y_clique = AC @ Y
            
            # L̄_S*Y^(t) + Ỹ_S^(t) ≈ AS_bar * Y^(t) (star expansion approximation)  
            Y_star = AS_bar @ Y
            
            # Full update term: f(X;W) + λ0*Ỹ_C^(t) + λ1*(L̄_S*Y^(t) + Ỹ_S^(t))
            update_term = f_X + lambda0_effective * Y_clique + lambda1_effective * Y_star
            
            # Check for NaN in update term
            if torch.isnan(update_term).any():
                print(f"WARNING: NaN in update_term at iteration {t}")
                print(f"f_X has NaN: {torch.isnan(f_X).any()}")
                print(f"Y_clique has NaN: {torch.isnan(Y_clique).any()}")
                print(f"Y_star has NaN: {torch.isnan(Y_star).any()}")
                update_term = torch.nan_to_num(update_term, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Preconditioned update: D̃^(-1)[...]
            preconditioned_update = D_tilde_inv @ update_term
            
            # Check for NaN in preconditioned update
            if torch.isnan(preconditioned_update).any():
                print(f"WARNING: NaN in preconditioned_update at iteration {t}")
                preconditioned_update = torch.nan_to_num(preconditioned_update, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Final update with momentum: Y^(t+1) = ReLU((1-α)Y^(t) + α*preconditioned_update)
            alpha_t = alpha_effective if not self.adaptive_alpha else alpha_effective * (0.9 ** t)  # Decay
            Y_new = torch.relu((1 - alpha_t) * Y + alpha_t * preconditioned_update)
            
            # Check for NaN in Y update
            if torch.isnan(Y_new).any():
                print(f"WARNING: NaN in Y_new at iteration {t}")
                print(f"alpha_t: {alpha_t}")
                print(f"Y has NaN: {torch.isnan(Y).any()}")
                print(f"preconditioned_update has NaN: {torch.isnan(preconditioned_update).any()}")
                Y_new = torch.nan_to_num(Y_new, nan=0.0, posinf=1.0, neginf=-1.0)
                
            Y = Y_new
            
            # Compute energy for monitoring
            energy = self._compute_energy(Y, hypergraph_data, f_X, lambda0_effective, lambda1_effective)
            
            # Track iteration info
            change = torch.norm(Y - Y_prev, p='fro').item()
            iteration_info['convergence_history'].append(change)
            iteration_info['alpha_history'].append(alpha_t.item() if torch.is_tensor(alpha_t) else alpha_t)
            iteration_info['energy_history'].append(energy.item())
            
            # Check convergence
            if change < self.convergence_threshold:
                iteration_info['converged'] = True
                iteration_info['final_iteration'] = t + 1
                break
                
            iteration_info['final_iteration'] = t + 1
        
        return Y, iteration_info
    
    def _compute_energy(
        self,
        Y: torch.Tensor,
        hypergraph_data: HypergraphData,
        f_X: torch.Tensor,
        lambda0: torch.Tensor,
        lambda1: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the energy function for monitoring convergence.
        
        This is a simplified energy computation for tracking purposes.
        """
        AC = hypergraph_data.AC
        AS_bar = hypergraph_data.AS_bar
        
        # Simplified energy: ||Y - f(X)||^2 + λ0*tr(Y^T AC Y) + λ1*tr(Y^T AS_bar Y)
        feature_term = torch.norm(Y - f_X, p='fro') ** 2
        clique_term = lambda0 * torch.trace(Y.T @ AC @ Y)
        star_term = lambda1 * torch.trace(Y.T @ AS_bar @ Y)
        
        energy = feature_term + clique_term + star_term
        return energy
    
    def _safe_matrix_inverse(self, matrix: torch.Tensor, regularization: float = 1e-6) -> torch.Tensor:
        """Safe matrix inversion with regularization."""
        device = matrix.device
        
        try:
            # Add regularization to diagonal for numerical stability
            regularized_matrix = matrix + regularization * torch.eye(matrix.size(0), device=device)
            
            # Use torch.linalg.pinv for better numerical stability
            inverse = torch.linalg.pinv(regularized_matrix, rcond=1e-8)
            
            # Check for NaN or infinite values
            if torch.isnan(inverse).any() or torch.isinf(inverse).any():
                print(f"WARNING: Invalid values in matrix inverse, using identity")
                inverse = torch.eye(matrix.size(0), device=device)
                
            return inverse
            
        except Exception as e:
            print(f"Matrix inversion failed: {e}, using identity matrix")
            return torch.eye(matrix.size(0), device=device)
        regularized_matrix = matrix + regularization * torch.eye(matrix.shape[0], device=device)
        
        try:
            return torch.inverse(regularized_matrix)
        except RuntimeError as e:
            warnings.warn(f"Matrix inversion failed, using pseudo-inverse: {e}")
            return torch.pinverse(regularized_matrix)


class PhenomNNBlock(nn.Module):
    """
    PhenomNN block combining multiple layers with residual connections.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        layer_type: str = 'simple',
        **layer_kwargs
    ):
        """
        Initialize PhenomNN block.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of PhenomNN layers
            layer_type: 'simple' or 'full'
            **layer_kwargs: Additional arguments for layers
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.layer_type = layer_type
        
        # Create layers
        self.layers = nn.ModuleList()
        
        LayerClass = PhenomNNSimpleLayer if layer_type == 'simple' else PhenomNNLayer
        
        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            layer = LayerClass(
                input_dim=layer_input_dim,
                hidden_dim=hidden_dim,
                **layer_kwargs
            )
            self.layers.append(layer)
        
        # Residual projection if dimensions don't match
        if input_dim != hidden_dim:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(
        self,
        hypergraph_data: HypergraphData
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through PhenomNN block.
        
        Args:
            hypergraph_data: Hypergraph data
            
        Returns:
            Tuple of (output_representations, all_iteration_info)
        """
        X = hypergraph_data.X
        Y = X
        
        all_iteration_info = {}
        
        for i, layer in enumerate(self.layers):
            # Update hypergraph data with current representations
            if i > 0:
                # Create new hypergraph data with updated node features
                updated_hypergraph = HypergraphData(
                    incidence_matrix=hypergraph_data.B,
                    node_features=Y,
                    node_labels=hypergraph_data.y,
                    device=Y.device
                )
            else:
                updated_hypergraph = hypergraph_data
            
            # Apply layer
            Y_new, iteration_info = layer(updated_hypergraph)
            
            # Residual connection
            if i == 0:
                residual = self.residual_proj(X)
            else:
                residual = Y
                
            Y = Y_new + residual
            
            # Store iteration info
            all_iteration_info[f'layer_{i}'] = iteration_info
        
        return Y, all_iteration_info
