"""
Unit tests for CUSP Curvature Encoding module
Tests curvature_positional_encoding per STAGE8_CUSP_Reference §Phase3
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
from torch_geometric.utils import to_undirected

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.cusp.cusp_encoding import (
    curvature_positional_encoding,
    advanced_curvature_encoding,
    CurvatureEncodingLayer,
    validate_curvature_encoding
)


class TestCurvaturePositionalEncoding:
    """Test basic curvature positional encoding functionality."""
    
    def test_basic_encoding_shape(self):
        """Test basic encoding produces correct shape."""
        n_nodes = 100
        dC = 16
        
        # Random curvatures in [-1, 1]
        node_orc = torch.randn(n_nodes) * 0.8
        
        Phi = curvature_positional_encoding(node_orc, dC)
        
        assert Phi.shape == (n_nodes, dC)
        assert torch.all(torch.isfinite(Phi))
    
    def test_encoding_deterministic(self):
        """Test encoding is deterministic for same input."""
        n_nodes = 50
        dC = 16
        
        node_orc = torch.randn(n_nodes)
        
        Phi1 = curvature_positional_encoding(node_orc, dC)
        Phi2 = curvature_positional_encoding(node_orc, dC)
        
        assert torch.allclose(Phi1, Phi2, atol=1e-6)
    
    def test_encoding_different_dimensions(self):
        """Test encoding works for different dimensions."""
        n_nodes = 30
        node_orc = torch.randn(n_nodes)
        
        for dC in [8, 16, 32, 64]:
            Phi = curvature_positional_encoding(node_orc, dC)
            assert Phi.shape == (n_nodes, dC)
            assert torch.all(torch.isfinite(Phi))
    
    def test_encoding_extreme_curvatures(self):
        """Test encoding handles extreme curvature values."""
        n_nodes = 20
        dC = 16
        
        # Test with extreme values
        extreme_orc = torch.tensor([-1.0, -0.9, 0.0, 0.9, 1.0] * 4)
        
        Phi = curvature_positional_encoding(extreme_orc, dC)
        
        assert Phi.shape == (n_nodes, dC)
        assert torch.all(torch.isfinite(Phi))
        
        # Check encoding diversity for different curvatures
        unique_rows = torch.unique(Phi, dim=0)
        assert unique_rows.shape[0] > 1  # Should produce different encodings
    
    def test_encoding_odd_dimension(self):
        """Test encoding works for odd dimensions."""
        n_nodes = 25
        dC_odd = 17
        
        node_orc = torch.randn(n_nodes)
        Phi = curvature_positional_encoding(node_orc, dC_odd)
        
        assert Phi.shape == (n_nodes, dC_odd)
        assert torch.all(torch.isfinite(Phi))
    
    def test_encoding_device_compatibility(self):
        """Test encoding works on different devices."""
        n_nodes = 20
        dC = 16
        
        # CPU test
        node_orc_cpu = torch.randn(n_nodes)
        Phi_cpu = curvature_positional_encoding(node_orc_cpu, dC)
        assert Phi_cpu.device == node_orc_cpu.device
        
        # GPU test (if available)
        if torch.cuda.is_available():
            node_orc_gpu = node_orc_cpu.cuda()
            Phi_gpu = curvature_positional_encoding(node_orc_gpu, dC)
            assert Phi_gpu.device == node_orc_gpu.device
            
            # Should produce same results
            assert torch.allclose(Phi_cpu, Phi_gpu.cpu(), atol=1e-5)


class TestAdvancedCurvatureEncoding:
    """Test advanced curvature encoding with neighborhood context."""
    
    def test_advanced_encoding_basic(self):
        """Test advanced encoding basic functionality."""
        n_nodes = 20
        dC = 16
        
        # Create simple graph
        edge_index = torch.tensor([
            [0, 1, 2, 3, 1, 2],
            [1, 0, 3, 2, 2, 1]
        ])
        edge_index = to_undirected(edge_index)
        
        node_orc = torch.randn(n_nodes) * 0.5
        
        for encoding_type in ["harmonic", "polynomial", "mixed"]:
            Phi = advanced_curvature_encoding(
                node_orc, edge_index, dC, encoding_type
            )
            
            assert Phi.shape == (n_nodes, dC)
            assert torch.all(torch.isfinite(Phi))
    
    def test_advanced_encoding_isolated_nodes(self):
        """Test advanced encoding handles isolated nodes."""
        n_nodes = 10
        dC = 16
        
        # Graph with isolated nodes
        edge_index = torch.tensor([[0, 1], [1, 0]])  # Only nodes 0,1 connected
        
        node_orc = torch.randn(n_nodes)
        
        Phi = advanced_curvature_encoding(node_orc, edge_index, dC)
        
        assert Phi.shape == (n_nodes, dC)
        assert torch.all(torch.isfinite(Phi))
    
    def test_advanced_encoding_comparison(self):
        """Test advanced encoding differs from basic encoding."""
        n_nodes = 15
        dC = 16
        
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0]
        ])
        edge_index = to_undirected(edge_index)
        
        node_orc = torch.randn(n_nodes)
        
        Phi_basic = curvature_positional_encoding(node_orc, dC)
        Phi_advanced = advanced_curvature_encoding(node_orc, edge_index, dC)
        
        # Should be different (neighborhood context matters)
        assert not torch.allclose(Phi_basic, Phi_advanced, atol=1e-3)


class TestCurvatureEncodingLayer:
    """Test learnable curvature encoding layer."""
    
    def test_layer_initialization(self):
        """Test layer initializes correctly."""
        layer = CurvatureEncodingLayer(dC=16, hidden_dim=32)
        
        assert layer.dC == 16
        assert layer.learnable == True
        assert hasattr(layer, 'curvature_mlp')
    
    def test_layer_forward_basic(self):
        """Test layer forward pass."""
        n_nodes = 25
        dC = 16
        
        layer = CurvatureEncodingLayer(dC=dC)
        node_orc = torch.randn(n_nodes)
        
        Phi = layer(node_orc)
        
        assert Phi.shape == (n_nodes, dC)
        assert torch.all(torch.isfinite(Phi))
    
    def test_layer_forward_with_edges(self):
        """Test layer forward with edge connectivity."""
        n_nodes = 20
        dC = 16
        
        layer = CurvatureEncodingLayer(dC=dC, encoding_type="mixed")
        
        node_orc = torch.randn(n_nodes)
        edge_index = torch.tensor([
            [0, 1, 2, 3],
            [1, 2, 3, 0]
        ])
        edge_index = to_undirected(edge_index)
        
        Phi = layer(node_orc, edge_index)
        
        assert Phi.shape == (n_nodes, dC)
        assert torch.all(torch.isfinite(Phi))
    
    def test_layer_non_learnable(self):
        """Test non-learnable layer."""
        n_nodes = 15
        dC = 16
        
        layer = CurvatureEncodingLayer(dC=dC, learnable=False)
        node_orc = torch.randn(n_nodes)
        
        # Should be equivalent to basic encoding
        Phi_layer = layer(node_orc)
        Phi_basic = curvature_positional_encoding(node_orc, dC)
        
        assert torch.allclose(Phi_layer, Phi_basic, atol=1e-6)
    
    def test_layer_gradient_flow(self):
        """Test gradients flow through learnable layer."""
        n_nodes = 10
        dC = 8
        
        layer = CurvatureEncodingLayer(dC=dC)
        node_orc = torch.randn(n_nodes, requires_grad=True)
        
        Phi = layer(node_orc)
        loss = Phi.sum()
        loss.backward()
        
        # Check gradients exist
        assert node_orc.grad is not None
        assert not torch.allclose(node_orc.grad, torch.zeros_like(node_orc.grad))


class TestCurvatureEncodingValidation:
    """Test curvature encoding validation functions."""
    
    def test_validation_success(self):
        """Test validation passes for valid encoding."""
        n_nodes, dC = 20, 16
        
        # Valid encoding
        Phi = torch.randn(n_nodes, dC)
        
        is_valid = validate_curvature_encoding(Phi, (n_nodes, dC))
        assert is_valid == True
    
    def test_validation_shape_failure(self):
        """Test validation fails for wrong shape."""
        n_nodes, dC = 20, 16
        
        # Wrong shape
        Phi = torch.randn(n_nodes, dC + 2)
        
        is_valid = validate_curvature_encoding(Phi, (n_nodes, dC))
        assert is_valid == False
    
    def test_validation_nan_failure(self):
        """Test validation fails for NaN values."""
        n_nodes, dC = 15, 16
        
        Phi = torch.randn(n_nodes, dC)
        Phi[0, 0] = float('nan')
        
        is_valid = validate_curvature_encoding(Phi, (n_nodes, dC))
        assert is_valid == False
    
    def test_validation_inf_failure(self):
        """Test validation fails for infinite values."""
        n_nodes, dC = 15, 16
        
        Phi = torch.randn(n_nodes, dC)
        Phi[5, 3] = float('inf')
        
        is_valid = validate_curvature_encoding(Phi, (n_nodes, dC))
        assert is_valid == False
    
    def test_validation_zero_failure(self):
        """Test validation fails for all-zero encoding."""
        n_nodes, dC = 10, 16
        
        Phi = torch.zeros(n_nodes, dC)
        
        is_valid = validate_curvature_encoding(Phi, (n_nodes, dC))
        assert is_valid == False
    
    def test_validation_large_values_failure(self):
        """Test validation fails for extremely large values."""
        n_nodes, dC = 10, 16
        
        Phi = torch.randn(n_nodes, dC) * 200  # Too large
        
        is_valid = validate_curvature_encoding(Phi, (n_nodes, dC))
        assert is_valid == False


class TestCurvatureEncodingIntegration:
    """Integration tests for complete curvature encoding pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete curvature encoding pipeline."""
        n_nodes = 50
        dC = 32
        
        # Generate test graph
        edge_index = torch.randint(0, n_nodes, (2, n_nodes * 2))
        edge_index = to_undirected(edge_index)
        
        # Random curvatures
        node_orc = torch.randn(n_nodes) * 0.7
        
        # Test all encoding variants
        encodings = {}
        
        # Basic encoding
        encodings['basic'] = curvature_positional_encoding(node_orc, dC)
        
        # Advanced encodings
        for enc_type in ["harmonic", "polynomial", "mixed"]:
            encodings[f'advanced_{enc_type}'] = advanced_curvature_encoding(
                node_orc, edge_index, dC, enc_type
            )
        
        # Learnable layer
        layer = CurvatureEncodingLayer(dC=dC)
        encodings['learnable'] = layer(node_orc, edge_index)
        
        # Validate all encodings
        for name, Phi in encodings.items():
            assert validate_curvature_encoding(Phi, (n_nodes, dC)), \
                f"Validation failed for {name} encoding"
    
    def test_encoding_consistency_across_batches(self):
        """Test encoding consistency when processing in batches."""
        n_nodes = 40
        dC = 16
        
        node_orc = torch.randn(n_nodes)
        
        # Encode all at once
        Phi_full = curvature_positional_encoding(node_orc, dC)
        
        # Encode in batches
        batch_size = 10
        Phi_batched = []
        for i in range(0, n_nodes, batch_size):
            batch_orc = node_orc[i:i+batch_size]
            batch_phi = curvature_positional_encoding(batch_orc, dC)
            Phi_batched.append(batch_phi)
        
        Phi_batched = torch.cat(Phi_batched, dim=0)
        
        # Should be identical
        assert torch.allclose(Phi_full, Phi_batched, atol=1e-6)


if __name__ == "__main__":
    # Run tests
    test_classes = [
        TestCurvaturePositionalEncoding,
        TestAdvancedCurvatureEncoding,
        TestCurvatureEncodingLayer,
        TestCurvatureEncodingValidation,
        TestCurvatureEncodingIntegration
    ]
    
    for test_class in test_classes:
        print(f"\n=== Running {test_class.__name__} ===")
        test_instance = test_class()
        
        for method_name in dir(test_instance):
            if method_name.startswith('test_'):
                print(f"Running {method_name}...")
                try:
                    getattr(test_instance, method_name)()
                    print(f"✓ {method_name} passed")
                except Exception as e:
                    print(f"✗ {method_name} failed: {e}")
    
    print("\n🎯 All Curvature Encoding tests completed!")
