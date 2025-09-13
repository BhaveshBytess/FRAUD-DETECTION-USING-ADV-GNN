"""
Unit tests for Cusp Laplacian construction
Tests build_cusp_laplacian per STAGE8_CUSP_Reference §Phase1 validation checklist
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
from cusp_laplacian import build_cusp_laplacian, validate_cusp_laplacian


class TestCuspLaplacian:
    """Test suite for Cusp Laplacian construction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.eps = 1e-8
        
        # Create test graph
        torch.manual_seed(42)
        self.edge_index = erdos_renyi_graph(8, 0.4)
        self.edge_index = to_undirected(self.edge_index)
        self.num_nodes = 8
        
        # Compute ORC for the test graph
        self.edge_orc, self.node_orc = compute_orc(
            self.edge_index, 
            self.num_nodes, 
            delta=0.2
        )
    
    def test_cusp_laplacian_basic_functionality(self):
        """Test basic Cusp Laplacian construction."""
        # Implements test per STAGE8_CUSP_Reference §Phase1 validation
        
        A_tilde, D_tilde, A_tilde_n = build_cusp_laplacian(
            self.edge_index,
            self.edge_orc,
            num_nodes=self.num_nodes
        )
        
        # Check output types
        assert isinstance(A_tilde, torch.sparse.FloatTensor)
        assert isinstance(D_tilde, torch.Tensor)
        assert isinstance(A_tilde_n, torch.sparse.FloatTensor)
        
        # Check shapes
        assert A_tilde.shape == (self.num_nodes, self.num_nodes)
        assert D_tilde.shape == (self.num_nodes,)
        assert A_tilde_n.shape == (self.num_nodes, self.num_nodes)
        
        print(f"✓ Basic functionality test passed")
        print(f"  A_tilde shape: {A_tilde.shape}")
        print(f"  D_tilde shape: {D_tilde.shape}")
        print(f"  A_tilde_n shape: {A_tilde_n.shape}")
    
    def test_cusp_laplacian_non_negative_entries(self):
        """Test that A_tilde has non-negative entries."""
        # Implements non-negative check per STAGE8_CUSP_Reference §Phase1 validation
        
        A_tilde, D_tilde, A_tilde_n = build_cusp_laplacian(
            self.edge_index,
            self.edge_orc,
            num_nodes=self.num_nodes
        )
        
        # Check A_tilde non-negative
        A_values = A_tilde.values()
        assert torch.all(A_values >= 0), f"A_tilde has negative values: {A_values.min()}"
        
        # Check D_tilde positive
        assert torch.all(D_tilde > 0), f"D_tilde has non-positive values: {D_tilde.min()}"
        
        print(f"✓ Non-negative entries test passed")
        print(f"  A_tilde values range: [{A_values.min():.6f}, {A_values.max():.6f}]")
        print(f"  D_tilde values range: [{D_tilde.min():.6f}, {D_tilde.max():.6f}]")
    
    def test_cusp_laplacian_finite_values(self):
        """Test that all outputs are finite."""
        # Implements finite values check per STAGE8_CUSP_Reference §Phase1 validation
        
        A_tilde, D_tilde, A_tilde_n = build_cusp_laplacian(
            self.edge_index,
            self.edge_orc,
            num_nodes=self.num_nodes
        )
        
        # Check finite values
        assert torch.all(torch.isfinite(A_tilde.values())), "A_tilde contains non-finite values"
        assert torch.all(torch.isfinite(D_tilde)), "D_tilde contains non-finite values"
        assert torch.all(torch.isfinite(A_tilde_n.values())), "A_tilde_n contains non-finite values"
        
        print(f"✓ Finite values test passed")
    
    def test_cusp_laplacian_validation_function(self):
        """Test the validation function."""
        # Implements validation function per STAGE8_CUSP_Reference §Phase1
        
        A_tilde, D_tilde, A_tilde_n = build_cusp_laplacian(
            self.edge_index,
            self.edge_orc,
            num_nodes=self.num_nodes
        )
        
        # Run validation
        is_valid = validate_cusp_laplacian(A_tilde, D_tilde, A_tilde_n)
        assert is_valid, "Cusp Laplacian validation failed"
        
        print(f"✓ Validation function test passed")
    
    def test_cusp_laplacian_extreme_curvatures(self):
        """Test Cusp Laplacian with extreme curvature values."""
        # Implements extreme values test for numerical stability
        
        # Create extreme edge curvatures (near boundaries)
        extreme_edge_orc = torch.tensor([
            -1.0 + self.eps,  # Very negative
            1.0 - self.eps,   # Very positive  
            0.0,              # Neutral
            0.5,              # Moderate positive
            -0.5              # Moderate negative
        ], dtype=torch.float)
        
        # Create simple edge index for these edges
        extreme_edge_index = torch.tensor([
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0]
        ], dtype=torch.long)
        
        A_tilde, D_tilde, A_tilde_n = build_cusp_laplacian(
            extreme_edge_index,
            extreme_edge_orc,
            num_nodes=5
        )
        
        # Check that extreme values are handled properly
        assert torch.all(torch.isfinite(A_tilde.values()))
        assert torch.all(torch.isfinite(D_tilde))
        assert torch.all(torch.isfinite(A_tilde_n.values()))
        
        # Validate
        is_valid = validate_cusp_laplacian(A_tilde, D_tilde, A_tilde_n)
        assert is_valid
        
        print(f"✓ Extreme curvatures test passed")
    
    def test_cusp_laplacian_self_loops(self):
        """Test Cusp Laplacian with and without self-loops."""
        # Implements self-loop handling test
        
        # With self-loops
        A_tilde_with, D_tilde_with, A_tilde_n_with = build_cusp_laplacian(
            self.edge_index,
            self.edge_orc,
            num_nodes=self.num_nodes,
            add_self_loop=True
        )
        
        # Without self-loops
        A_tilde_without, D_tilde_without, A_tilde_n_without = build_cusp_laplacian(
            self.edge_index,
            self.edge_orc,
            num_nodes=self.num_nodes,
            add_self_loop=False
        )
        
        # Check that self-loop version has more edges
        assert A_tilde_with._nnz() >= A_tilde_without._nnz()
        
        # Both should be valid
        assert validate_cusp_laplacian(A_tilde_with, D_tilde_with, A_tilde_n_with)
        assert validate_cusp_laplacian(A_tilde_without, D_tilde_without, A_tilde_n_without)
        
        print(f"✓ Self-loops test passed")
        print(f"  With self-loops: {A_tilde_with._nnz()} edges")
        print(f"  Without self-loops: {A_tilde_without._nnz()} edges")
    
    def test_cusp_laplacian_weight_formula(self):
        """Test the weight formula implementation."""
        # Implements weight formula test per STAGE8_CUSP_Reference §Phase1 Definition 1
        
        # Test specific curvature values and expected weights
        test_curvatures = torch.tensor([0.0, 0.5, -0.5, 0.9, -0.9])
        
        for kappa in test_curvatures:
            # Manual weight calculation: w = exp(-1/(1-kappa))
            expected_weight = torch.exp(-1.0 / (1.0 - kappa + self.eps))
            
            # Create single edge with this curvature
            single_edge_index = torch.tensor([[0], [1]], dtype=torch.long)
            single_edge_orc = torch.tensor([kappa], dtype=torch.float)
            
            A_tilde, _, _ = build_cusp_laplacian(
                single_edge_index,
                single_edge_orc,
                num_nodes=2,
                add_self_loop=False
            )
            
            # Extract the computed weight
            computed_weight = A_tilde.values()[0]
            
            # Check that it matches expected (within tolerance)
            assert torch.isclose(computed_weight, expected_weight, atol=1e-6), \
                f"Weight mismatch for κ={kappa}: expected {expected_weight}, got {computed_weight}"
        
        print(f"✓ Weight formula test passed")


if __name__ == "__main__":
    # Run tests directly
    test_laplacian = TestCuspLaplacian()
    test_laplacian.setup_method()
    
    print("Running Cusp Laplacian unit tests...")
    print("=" * 50)
    
    try:
        test_laplacian.test_cusp_laplacian_basic_functionality()
        test_laplacian.test_cusp_laplacian_non_negative_entries()
        test_laplacian.test_cusp_laplacian_finite_values()
        test_laplacian.test_cusp_laplacian_validation_function()
        test_laplacian.test_cusp_laplacian_extreme_curvatures()
        test_laplacian.test_cusp_laplacian_self_loops()
        test_laplacian.test_cusp_laplacian_weight_formula()
        
        print("=" * 50)
        print("✅ All Cusp Laplacian tests passed!")
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
