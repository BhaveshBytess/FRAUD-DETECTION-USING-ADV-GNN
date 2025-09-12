"""
HypergraphData: Core data structure for PhenomNN-based hypergraph neural networks

Implements the mathematical foundation from "From Hypergraph Energy Functions to Hypergraph Neural Networks"
Following the reference document specifications for fraud detection applications.

Key Components:
- Incidence matrix B (n_nodes × n_hyperedges) 
- Node features X (n_nodes × d_features)
- Hyperedge features U (n_hyperedges × d_edge_features) [optional]
- Degree matrices: DH, DC, DS_bar
- Expansion matrices: AC, AS_bar

Mathematical Foundation:
- Hyperedge degree matrix: DH = diag(B.sum(dim=0))
- Clique expansion: AC = B @ inv(DH) @ B.T
- Star expansion: AS_bar = B @ inv(DH) @ B.T  
- Node degrees: DC = diag(AC.sum(dim=1)), DS_bar = diag(AS_bar.sum(dim=1))
"""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from scipy import sparse
import pandas as pd


class HypergraphData:
    """
    Core hypergraph data structure implementing PhenomNN paper specifications.
    
    Attributes:
        B (torch.Tensor): Incidence matrix (n_nodes × n_hyperedges), binary
        X (torch.Tensor): Node features (n_nodes × d_features)  
        U (torch.Tensor): Hyperedge features (n_hyperedges × d_edge_features), optional
        y (torch.Tensor): Node labels (n_nodes,), optional
        
        # Computed matrices (lazy evaluation)
        DH (torch.Tensor): Hyperedge degree matrix diag(B.sum(dim=0))
        DC (torch.Tensor): Clique node degree matrix  
        DS_bar (torch.Tensor): Star node degree matrix
        AC (torch.Tensor): Clique expansion matrix
        AS_bar (torch.Tensor): Star expansion matrix
    """
    
    def __init__(
        self, 
        incidence_matrix: torch.Tensor,
        node_features: torch.Tensor,
        hyperedge_features: Optional[torch.Tensor] = None,
        node_labels: Optional[torch.Tensor] = None,
        hyperedge_labels: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize HypergraphData with incidence matrix and features.
        
        Args:
            incidence_matrix: Binary matrix (n_nodes × n_hyperedges)
            node_features: Node feature matrix (n_nodes × d_features)
            hyperedge_features: Optional hyperedge features (n_hyperedges × d_edge_features)
            node_labels: Optional node labels (n_nodes,)
            hyperedge_labels: Optional hyperedge labels (n_hyperedges,)
            device: Device to store tensors on
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Core data
        self.B = incidence_matrix.to(device)
        self.X = node_features.to(device)
        self.U = hyperedge_features.to(device) if hyperedge_features is not None else None
        self.y = node_labels.to(device) if node_labels is not None else None
        self.hyperedge_y = hyperedge_labels.to(device) if hyperedge_labels is not None else None
        
        # Validate dimensions
        assert self.B.shape[0] == self.X.shape[0], f"Incidence matrix rows ({self.B.shape[0]}) must match node features ({self.X.shape[0]})"
        if self.U is not None:
            assert self.B.shape[1] == self.U.shape[0], f"Incidence matrix cols ({self.B.shape[1]}) must match hyperedge features ({self.U.shape[0]})"
        if self.y is not None:
            assert self.X.shape[0] == self.y.shape[0], f"Node features ({self.X.shape[0]}) must match labels ({self.y.shape[0]})"
        
        # Cached matrices (computed on demand)
        self._DH = None
        self._DC = None  
        self._DS_bar = None
        self._AC = None
        self._AS_bar = None
        
        self.device = device
        
    @property
    def n_nodes(self) -> int:
        """Number of nodes in hypergraph."""
        return self.B.shape[0]
    
    @property  
    def n_hyperedges(self) -> int:
        """Number of hyperedges in hypergraph."""
        return self.B.shape[1]
        
    @property
    def n_features(self) -> int:
        """Number of node features."""
        return self.X.shape[1]
    
    @property
    def DH(self) -> torch.Tensor:
        """Hyperedge degree matrix: DH = diag(B.sum(dim=0))"""
        if self._DH is None:
            hyperedge_degrees = self.B.sum(dim=0)  # Sum over nodes for each hyperedge
            # Add small epsilon to prevent singular matrices
            hyperedge_degrees = torch.clamp(hyperedge_degrees, min=1e-8)
            self._DH = torch.diag(hyperedge_degrees)
        return self._DH
    
    @property
    def DC(self) -> torch.Tensor:
        """Clique node degree matrix: DC = diag(AC.sum(dim=1))"""
        if self._DC is None:
            clique_degrees = self.AC.sum(dim=1)
            self._DC = torch.diag(clique_degrees)
        return self._DC
    
    @property
    def DS_bar(self) -> torch.Tensor:
        """Star node degree matrix: DS_bar = diag(AS_bar.sum(dim=1))"""
        if self._DS_bar is None:
            star_degrees = self.AS_bar.sum(dim=1)
            self._DS_bar = torch.diag(star_degrees)
        return self._DS_bar
    
    @property
    def AC(self) -> torch.Tensor:
        """Clique expansion matrix: AC = B @ inv(DH) @ B.T"""
        if self._AC is None:
            # Compute B @ inv(DH) @ B.T with improved numerical stability
            try:
                # Use pseudoinverse for better numerical stability
                DH_regularized = self.DH + 1e-6 * torch.eye(self.DH.shape[0], device=self.device)
                DH_inv = torch.linalg.pinv(DH_regularized, rcond=1e-8)
                
                # Check for NaN or infinite values
                if torch.isnan(DH_inv).any() or torch.isinf(DH_inv).any():
                    print("WARNING: Invalid values in DH_inv, using regularized identity")
                    DH_inv = torch.eye(self.DH.shape[0], device=self.device) * 0.1
                
                self._AC = self.B @ DH_inv @ self.B.T
                
                # Additional safety check
                if torch.isnan(self._AC).any() or torch.isinf(self._AC).any():
                    print("WARNING: Invalid values in AC, using regularized identity")
                    self._AC = torch.eye(self.n_nodes, device=self.device) * 0.1
                    
            except Exception as e:
                print(f"AC computation failed: {e}, using identity matrix")
                self._AC = torch.eye(self.n_nodes, device=self.device) * 0.1
                
        return self._AC
    
    @property
    def AS_bar(self) -> torch.Tensor:
        """Star expansion matrix: AS_bar = B @ inv(DH) @ B.T (same as AC for this formulation)"""
        if self._AS_bar is None:
            # In the simplified PhenomNN formulation, star expansion is same as clique
            # This follows the paper's notation where both use the same hyperedge-mediated expansion
            self._AS_bar = self.AC
        return self._AS_bar
    
    def get_preconditioner(self, lambda0: float = 1.0, lambda1: float = 1.0) -> torch.Tensor:
        """
        Compute preconditioner matrix: D̃ = λ0*DC + λ1*DS_bar + I
        
        Args:
            lambda0: Weight for clique expansion
            lambda1: Weight for star expansion
            
        Returns:
            Preconditioner matrix D̃
        """
        I = torch.eye(self.n_nodes, device=self.device)
        D_tilde = lambda0 * self.DC + lambda1 * self.DS_bar + I
        
        # Add extra regularization for numerical stability
        D_tilde = D_tilde + 1e-4 * I
        
        # Check for invalid values
        if torch.isnan(D_tilde).any() or torch.isinf(D_tilde).any():
            print("WARNING: Invalid values in preconditioner, using identity")
            D_tilde = I + 1e-4 * I
            
        return D_tilde
    
    def validate_structure(self) -> Dict[str, bool]:
        """
        Validate hypergraph structure and matrix computations.
        
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Check incidence matrix is binary
        results['B_binary'] = torch.all((self.B == 0) | (self.B == 1))
        
        # Check no empty hyperedges
        hyperedge_sizes = self.B.sum(dim=0)
        results['no_empty_hyperedges'] = torch.all(hyperedge_sizes > 0)
        
        # Check degree matrices are positive definite
        results['DH_positive'] = torch.all(torch.diag(self.DH) > 0)
        results['DC_positive'] = torch.all(torch.diag(self.DC) > 0)
        results['DS_bar_positive'] = torch.all(torch.diag(self.DS_bar) > 0)
        
        # Check matrix dimensions
        results['matrix_dims'] = (
            self.AC.shape == (self.n_nodes, self.n_nodes) and
            self.AS_bar.shape == (self.n_nodes, self.n_nodes) and
            self.DH.shape == (self.n_hyperedges, self.n_hyperedges)
        )
        
        # Check for NaN/Inf values
        results['no_nan_inf'] = (
            torch.isfinite(self.B).all() and
            torch.isfinite(self.X).all() and
            torch.isfinite(self.AC).all() and
            torch.isfinite(self.AS_bar).all()
        )
        
        return results
    
    def get_hyperedge_statistics(self) -> Dict[str, float]:
        """Get statistics about hyperedge structure."""
        hyperedge_sizes = self.B.sum(dim=0).float()
        node_degrees = self.B.sum(dim=1).float()
        
        return {
            'n_nodes': self.n_nodes,
            'n_hyperedges': self.n_hyperedges,
            'n_features': self.n_features,
            'avg_hyperedge_size': hyperedge_sizes.mean().item(),
            'max_hyperedge_size': hyperedge_sizes.max().item(),
            'min_hyperedge_size': hyperedge_sizes.min().item(),
            'avg_node_degree': node_degrees.mean().item(),
            'max_node_degree': node_degrees.max().item(),
            'min_node_degree': node_degrees.min().item(),
            'density': (self.B.sum() / (self.n_nodes * self.n_hyperedges)).item()
        }
    
    def to(self, device: torch.device) -> 'HypergraphData':
        """Move hypergraph data to specified device."""
        self.B = self.B.to(device)
        self.X = self.X.to(device)
        if self.U is not None:
            self.U = self.U.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        if self.hyperedge_y is not None:
            self.hyperedge_y = self.hyperedge_y.to(device)
            
        # Clear cached matrices so they get recomputed on new device
        self._DH = None
        self._DC = None
        self._DS_bar = None  
        self._AC = None
        self._AS_bar = None
        
        self.device = device
        return self


def construct_hypergraph_from_hetero(
    hetero_data: HeteroData,
    hyperedge_construction_fn,
    node_type: str = 'transaction'
) -> HypergraphData:
    """
    Convert HeteroData to HypergraphData using specified hyperedge construction.
    
    Args:
        hetero_data: PyTorch Geometric HeteroData object
        hyperedge_construction_fn: Function to construct hyperedges
        node_type: Primary node type for classification
        
    Returns:
        HypergraphData object ready for PhenomNN training
    """
    # Extract transaction node features and labels
    tx_data = hetero_data[node_type]
    node_features = tx_data.x
    node_labels = tx_data.y if hasattr(tx_data, 'y') else None
    
    # Construct hyperedges from heterogeneous graph structure
    hyperedges, hyperedge_features = hyperedge_construction_fn(hetero_data)
    
    # Build incidence matrix
    n_nodes = node_features.shape[0]
    n_hyperedges = len(hyperedges)
    
    incidence_matrix = torch.zeros((n_nodes, n_hyperedges), dtype=torch.float)
    
    for he_idx, hyperedge in enumerate(hyperedges):
        for node_idx in hyperedge:
            if node_idx < n_nodes:  # Ensure valid node index
                incidence_matrix[node_idx, he_idx] = 1.0
    
    return HypergraphData(
        incidence_matrix=incidence_matrix,
        node_features=node_features,
        hyperedge_features=hyperedge_features,
        node_labels=node_labels
    )
