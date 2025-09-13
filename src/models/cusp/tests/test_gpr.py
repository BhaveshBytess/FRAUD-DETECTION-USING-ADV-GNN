"""
Unit tests for GPR Filter Bank
Tests gpr_filter_bank per STAGE8_CUSP_Reference §Phase2 validation checklist
"""

import pytest
import torch
import numpy as np
from torch_geometric.utils import erdos_renyi_graph, to_undirected

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cusp_orc import compute_orc
from cusp_laplacian import build_cusp_laplacian
from cusp_gpr import GPRFilterBank, gpr_filter_bank, ManifoldGPRFilter, validate_gpr_output


class TestGPRFilterBank:
    """Test suite for GPR Filter Bank."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.eps = 1e-8
        
        # Create test graph
        torch.manual_seed(42)
        self.edge_index = erdos_renyi_graph(8, 0.4)
        self.edge_index = to_undirected(self.edge_index)
        self.num_nodes = 8
        self.feat_dim = 16
        
        # Create test features
        self.X = torch.randn(self.num_nodes, self.feat_dim)
        
        # Compute ORC and Cusp Laplacian for testing
        edge_orc, _ = compute_orc(self.edge_index, self.num_nodes, delta=0.2)
        _, _, self.A_tilde_n = build_cusp_laplacian(
            self.edge_index, edge_orc, num_nodes=self.num_nodes
        )
    
    def test_gpr_filter_bank_basic_functionality(self):
        """Test basic GPR filter bank functionality."""
        # Implements test per STAGE8_CUSP_Reference §Phase2 validation
        
        filter_bank = GPRFilterBank(filter_count_L=5, alpha=0.3, dropout=0.0)
        
        # Forward pass
        Z = filter_bank(self.A_tilde_n, self.X)
        
        # Check output shape
        assert Z.shape == self.X.shape, f"Expected shape {self.X.shape}, got {Z.shape}"
        
        # Check output type
        assert isinstance(Z, torch.Tensor)
        assert Z.dtype == torch.float
        
        print(f"✓ Basic functionality test passed")
        print(f"  Input shape: {self.X.shape}")
        print(f"  Output shape: {Z.shape}")
    
    def test_gpr_filter_bank_finite_outputs(self):
        """Test that GPR produces stable finite outputs."""
        # Implements finite outputs check per STAGE8_CUSP_Reference §Phase2 validation
        
        filter_counts = [5, 10, 15]
        
        for L in filter_counts:
            filter_bank = GPRFilterBank(filter_count_L=L, alpha=0.3, dropout=0.1)
            Z = filter_bank(self.A_tilde_n, self.X)
            
            # Check finite values
            assert torch.all(torch.isfinite(Z)), f"Non-finite values for L={L}"
            
            # Validate using validation function
            is_valid = validate_gpr_output(Z, self.X.shape)
            assert is_valid, f"Validation failed for L={L}"
        
        print(f"✓ Finite outputs test passed for L={filter_counts}")
    
    def test_gpr_functional_interface(self):
        """Test functional GPR filter bank interface."""
        # Implements functional interface per STAGE8_CUSP_Reference §Phase2
        
        # Test standard output
        Z_standard = gpr_filter_bank(
            self.A_tilde_n, self.X, 
            filter_count_L=5, alpha=0.3
        )
        
        # Test return all filters
        Z_all_filters = gpr_filter_bank(
            self.A_tilde_n, self.X,
            filter_count_L=5, alpha=0.3,
            return_filters=True
        )
        
        # Check shapes
        assert Z_standard.shape == self.X.shape
        assert Z_all_filters.shape == (self.num_nodes, self.feat_dim * 6)  # L+1 filters
        
        # Check finite values
        assert torch.all(torch.isfinite(Z_standard))
        assert torch.all(torch.isfinite(Z_all_filters))
        
        print(f"✓ Functional interface test passed")
        print(f"  Standard output shape: {Z_standard.shape}")
        print(f"  All filters shape: {Z_all_filters.shape}")
    
    def test_gpr_parameter_sensitivity(self):
        """Test GPR with different parameters."""
        # Implements parameter sensitivity per STAGE8_CUSP_Reference §Hyperparameters
        
        alpha_values = [0.1, 0.3, 0.5, 0.9]
        results = {}
        
        for alpha in alpha_values:
            filter_bank = GPRFilterBank(filter_count_L=5, alpha=alpha, dropout=0.0)
            Z = filter_bank(self.A_tilde_n, self.X)
            
            # Check stability
            assert torch.all(torch.isfinite(Z))
            results[alpha] = Z
            
        # Check that different alphas produce different results
        Z_01 = results[0.1]
        Z_09 = results[0.9]
        assert not torch.allclose(Z_01, Z_09, atol=1e-6), "Different alphas should produce different results"
        
        print(f"✓ Parameter sensitivity test passed for alpha={alpha_values}")
    
    def test_manifold_gpr_filter(self):
        """Test manifold-aware GPR filters."""
        # Implements manifold filtering per STAGE8_CUSP_Reference §Phase2
        
        manifold_types = ["euclidean", "hyperbolic", "spherical"]
        results = {}
        
        for manifold_type in manifold_types:
            manifold_filter = ManifoldGPRFilter(
                manifold_type=manifold_type,
                filter_count_L=5,
                alpha=0.3
            )
            
            Z_manifold = manifold_filter(self.A_tilde_n, self.X)
            
            # Check output shape and finite values
            assert Z_manifold.shape == self.X.shape
            assert torch.all(torch.isfinite(Z_manifold))
            
            results[manifold_type] = Z_manifold
        
        # Check that different manifold types produce different results
        assert not torch.allclose(results["euclidean"], results["hyperbolic"], atol=1e-6)
        assert not torch.allclose(results["euclidean"], results["spherical"], atol=1e-6)
        
        print(f"✓ Manifold GPR filter test passed for {manifold_types}")
    
    def test_gpr_gradient_flow(self):
        """Test that gradients flow through GPR layers."""
        # Implements gradient flow test for training
        
        filter_bank = GPRFilterBank(filter_count_L=5, alpha=0.3, dropout=0.0)
        
        # Enable gradients
        self.X.requires_grad_(True)
        
        # Forward pass
        Z = filter_bank(self.A_tilde_n, self.X)
        
        # Compute loss and backward pass
        loss = Z.sum()
        loss.backward()
        
        # Check that gradients exist
        assert self.X.grad is not None
        assert torch.any(self.X.grad != 0)
        
        # Check GPR weights have gradients
        assert filter_bank.gpr_weights.grad is not None
        
        print(f"✓ Gradient flow test passed")
    
    def test_gpr_edge_cases(self):
        """Test GPR on edge cases."""
        # Implements edge case testing
        
        # Test with single node
        single_edge_index = torch.empty((2, 0), dtype=torch.long)
        single_X = torch.randn(1, self.feat_dim)
        single_edge_orc = torch.empty(0)
        
        _, _, single_A_tilde_n = build_cusp_laplacian(
            single_edge_index, single_edge_orc, num_nodes=1
        )
        
        filter_bank = GPRFilterBank(filter_count_L=3, alpha=0.3, dropout=0.0)
        Z_single = filter_bank(single_A_tilde_n, single_X)
        
        assert Z_single.shape == single_X.shape
        assert torch.all(torch.isfinite(Z_single))
        
        # Test with very small features
        small_X = torch.ones_like(self.X) * 1e-8
        Z_small = filter_bank(self.A_tilde_n, small_X)
        
        assert torch.all(torch.isfinite(Z_small))
        
        print(f"✓ Edge cases test passed")
    
    def test_gpr_numerical_stability(self):
        """Test numerical stability with extreme inputs."""
        # Implements numerical stability test
        
        filter_bank = GPRFilterBank(filter_count_L=10, alpha=0.9, dropout=0.0)
        
        # Test with large features
        large_X = torch.randn_like(self.X) * 100
        Z_large = filter_bank(self.A_tilde_n, large_X)
        
        assert torch.all(torch.isfinite(Z_large))
        assert torch.max(torch.abs(Z_large)) < 1e6  # Reasonable magnitude
        
        # Test with small alpha (near 1)
        extreme_filter = GPRFilterBank(filter_count_L=5, alpha=0.99, dropout=0.0)
        Z_extreme = extreme_filter(self.A_tilde_n, self.X)
        
        assert torch.all(torch.isfinite(Z_extreme))
        
        print(f"✓ Numerical stability test passed")


if __name__ == "__main__":
    # Run tests directly
    test_gpr = TestGPRFilterBank()
    test_gpr.setup_method()
    
    print("Running GPR Filter Bank unit tests...")
    print("=" * 50)
    
    try:
        test_gpr.test_gpr_filter_bank_basic_functionality()
        test_gpr.test_gpr_filter_bank_finite_outputs()
        test_gpr.test_gpr_functional_interface()
        test_gpr.test_gpr_parameter_sensitivity()
        test_gpr.test_manifold_gpr_filter()
        test_gpr.test_gpr_gradient_flow()
        test_gpr.test_gpr_edge_cases()
        test_gpr.test_gpr_numerical_stability()
        
        print("=" * 50)
        print("✅ All GPR Filter Bank tests passed!")
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
