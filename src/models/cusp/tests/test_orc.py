"""
Unit tests for ORC computation
Tests compute_orc per STAGE8_CUSP_Reference §Phase1 validation checklist
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, to_undirected

# Import our ORC functions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cusp_orc import compute_orc, compute_orc_fast_approximation


class TestORC:
    """Test suite for Ollivier-Ricci curvature computation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.eps = 1e-8
        
        # Create small test graph
        self.small_edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2]
        ], dtype=torch.long)
        self.small_num_nodes = 4
        
        # Create medium test graph (random)
        torch.manual_seed(42)
        self.medium_edge_index = erdos_renyi_graph(10, 0.3)
        self.medium_edge_index = to_undirected(self.medium_edge_index)
        self.medium_num_nodes = 10
    
    def test_orc_basic_functionality(self):
        """Test basic ORC computation functionality."""
        # Implements test per STAGE8_CUSP_Reference §Phase1 validation
        
        edge_orc, node_orc = compute_orc(
            self.small_edge_index, 
            self.small_num_nodes,
            delta=0.2
        )
        
        # Check output shapes
        assert edge_orc.shape == (self.small_edge_index.size(1),)
        assert node_orc.shape == (self.small_num_nodes,)
        
        # Check output types
        assert edge_orc.dtype == torch.float
        assert node_orc.dtype == torch.float
        
        print(f"✓ Basic functionality test passed")
        print(f"  Edge ORC shape: {edge_orc.shape}")
        print(f"  Node ORC shape: {node_orc.shape}")
    
    def test_orc_finite_values(self):
        """Test that ORC returns finite values in expected range."""
        # Implements finite values check per STAGE8_CUSP_Reference §Phase1 validation
        
        edge_orc, node_orc = compute_orc(
            self.medium_edge_index,
            self.medium_num_nodes,
            delta=0.2
        )
        
        # Check all values are finite
        assert torch.all(torch.isfinite(edge_orc)), "Edge ORC contains non-finite values"
        assert torch.all(torch.isfinite(node_orc)), "Node ORC contains non-finite values"
        
        # Check values are in expected range [-1, 1]
        assert torch.all(edge_orc >= -1.0 - self.eps), f"Edge ORC below -1: {edge_orc.min()}"
        assert torch.all(edge_orc <= 1.0 + self.eps), f"Edge ORC above 1: {edge_orc.max()}"
        assert torch.all(node_orc >= -1.0 - self.eps), f"Node ORC below -1: {node_orc.min()}"
        assert torch.all(node_orc <= 1.0 + self.eps), f"Node ORC above 1: {node_orc.max()}"
        
        print(f"✓ Finite values test passed")
        print(f"  Edge ORC range: [{edge_orc.min():.4f}, {edge_orc.max():.4f}]")
        print(f"  Node ORC range: [{node_orc.min():.4f}, {node_orc.max():.4f}]")
    
    def test_orc_parameter_sensitivity(self):
        """Test ORC computation with different delta parameters."""
        # Implements parameter sensitivity per STAGE8_CUSP_Reference §Hyperparameters
        
        delta_values = [0.2, 0.5, 0.7]
        results = {}
        
        for delta in delta_values:
            edge_orc, node_orc = compute_orc(
                self.small_edge_index,
                self.small_num_nodes,
                delta=delta
            )
            results[delta] = (edge_orc, node_orc)
            
            # Check stability for each delta
            assert torch.all(torch.isfinite(edge_orc))
            assert torch.all(torch.isfinite(node_orc))
        
        print(f"✓ Parameter sensitivity test passed for delta={delta_values}")
    
    def test_orc_fast_approximation(self):
        """Test fast ORC approximation."""
        # Implements fast approximation per STAGE8_CUSP_Reference §Error Handling
        
        edge_orc_fast, node_orc_fast = compute_orc_fast_approximation(
            self.medium_edge_index,
            self.medium_num_nodes,
            delta=0.2
        )
        
        # Check output shapes and types
        assert edge_orc_fast.shape == (self.medium_edge_index.size(1),)
        assert node_orc_fast.shape == (self.medium_num_nodes,)
        
        # Check finite values and range
        assert torch.all(torch.isfinite(edge_orc_fast))
        assert torch.all(torch.isfinite(node_orc_fast))
        assert torch.all(edge_orc_fast >= -1.0 - self.eps)
        assert torch.all(edge_orc_fast <= 1.0 + self.eps)
        
        print(f"✓ Fast approximation test passed")
        print(f"  Fast edge ORC range: [{edge_orc_fast.min():.4f}, {edge_orc_fast.max():.4f}]")
    
    def test_orc_consistency(self):
        """Test ORC computation consistency with fixed seed."""
        # Implements consistency check for reproducibility
        
        torch.manual_seed(123)
        edge_orc1, node_orc1 = compute_orc(
            self.small_edge_index,
            self.small_num_nodes,
            delta=0.2
        )
        
        torch.manual_seed(123)
        edge_orc2, node_orc2 = compute_orc(
            self.small_edge_index,
            self.small_num_nodes,
            delta=0.2
        )
        
        # Check consistency (should be identical)
        assert torch.allclose(edge_orc1, edge_orc2, atol=1e-6)
        assert torch.allclose(node_orc1, node_orc2, atol=1e-6)
        
        print(f"✓ Consistency test passed")
    
    def test_orc_empty_graph(self):
        """Test ORC computation on edge cases."""
        # Implements edge case testing
        
        # Empty graph
        empty_edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_orc, node_orc = compute_orc(empty_edge_index, 1, delta=0.2)
        
        assert edge_orc.shape == (0,)
        assert node_orc.shape == (1,)
        assert node_orc[0] == 0.0  # isolated node should have zero curvature
        
        print(f"✓ Empty graph test passed")


if __name__ == "__main__":
    # Run tests directly
    test_orc = TestORC()
    test_orc.setup_method()
    
    print("Running ORC unit tests...")
    print("=" * 50)
    
    try:
        test_orc.test_orc_basic_functionality()
        test_orc.test_orc_finite_values()
        test_orc.test_orc_parameter_sensitivity()
        test_orc.test_orc_fast_approximation()
        test_orc.test_orc_consistency()
        test_orc.test_orc_empty_graph()
        
        print("=" * 50)
        print("✅ All ORC tests passed!")
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
