"""
Unit tests for CUSP Module Integration
Tests cusp_core.py per STAGE8_CUSP_Reference §Phase5
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, Batch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.cusp.cusp_core import CuspModule, create_cusp_model


class TestCuspModule:
    """Test complete CUSP module integration."""
    
    def test_cusp_module_initialization(self):
        """Test CUSP module initializes correctly."""
        model = CuspModule(
            input_dim=16,
            hidden_dim=32,
            output_dim=24,
            num_layers=2
        )
        
        assert model.input_dim == 16
        assert model.hidden_dim == 32
        assert model.output_dim == 24
        assert model.num_layers == 2
        assert len(model.cusp_layers) == 2
    
    def test_cusp_module_forward_single_graph(self):
        """Test CUSP forward pass for single graph."""
        input_dim = 20
        hidden_dim = 32
        output_dim = 16
        n_nodes = 25
        
        model = CuspModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=2
        )
        
        # Create test graph
        x = torch.randn(n_nodes, input_dim)
        edge_index = torch.randint(0, n_nodes, (2, n_nodes * 2))
        edge_index = to_undirected(edge_index)
        
        # Forward pass
        output = model(x, edge_index)
        
        assert output.shape == (1, output_dim)  # Single graph pooling
        assert torch.all(torch.isfinite(output))
    
    def test_cusp_module_forward_batched(self):
        """Test CUSP forward pass for batched graphs."""
        input_dim = 16
        hidden_dim = 24
        output_dim = 12
        n_nodes = 40
        batch_size = 3
        
        model = CuspModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            pooling_strategy="mean"  # Use mean pooling for simplicity
        )
        
        # Create test batch
        x = torch.randn(n_nodes, input_dim)
        edge_index = torch.randint(0, n_nodes, (2, n_nodes * 2))
        edge_index = to_undirected(edge_index)
        
        # Batch assignment
        batch = torch.cat([
            torch.full((15,), 0),
            torch.full((12,), 1),
            torch.full((13,), 2)
        ])
        
        # Forward pass
        output = model(x, edge_index, batch)
        
        assert output.shape == (batch_size, output_dim)
        assert torch.all(torch.isfinite(output))
    
    def test_cusp_module_different_manifolds(self):
        """Test CUSP with different manifold configurations."""
        input_dim = 12
        n_nodes = 20
        
        manifold_configs = [
            ["euclidean"],
            ["hyperbolic"],
            ["spherical"],
            ["euclidean", "hyperbolic"],
            ["euclidean", "spherical"],
            ["euclidean", "hyperbolic", "spherical"]
        ]
        
        x = torch.randn(n_nodes, input_dim)
        edge_index = torch.randint(0, n_nodes, (2, n_nodes))
        edge_index = to_undirected(edge_index)
        
        for manifolds in manifold_configs:
            model = CuspModule(
                input_dim=input_dim,
                hidden_dim=16,
                output_dim=8,
                manifold_types=manifolds,
                pooling_strategy="mean"
            )
            
            output = model(x, edge_index)
            
            assert output.shape == (1, 8)
            assert torch.all(torch.isfinite(output))
    
    def test_cusp_module_return_attention(self):
        """Test CUSP module returns attention weights."""
        input_dim = 16
        n_nodes = 20
        
        model = CuspModule(
            input_dim=input_dim,
            hidden_dim=24,
            output_dim=12,
            pooling_strategy="attention"
        )
        
        x = torch.randn(n_nodes, input_dim)
        edge_index = torch.randint(0, n_nodes, (2, n_nodes))
        edge_index = to_undirected(edge_index)
        
        output, extras = model(x, edge_index, return_attention=True)
        
        assert output.shape == (1, 12)
        assert 'attention_weights' in extras
        assert torch.all(torch.isfinite(output))
    
    def test_cusp_module_return_intermediate(self):
        """Test CUSP module returns intermediate results."""
        input_dim = 14
        n_nodes = 18
        
        model = CuspModule(
            input_dim=input_dim,
            hidden_dim=20,
            output_dim=10,
            num_layers=2,
            pooling_strategy="mean"
        )
        
        x = torch.randn(n_nodes, input_dim)
        edge_index = torch.randint(0, n_nodes, (2, n_nodes))
        edge_index = to_undirected(edge_index)
        
        output, extras = model(x, edge_index, return_intermediate=True)
        
        assert output.shape == (1, 10)
        assert 'intermediate_results' in extras
        assert 'final_node_features' in extras
        assert 'curvature' in extras
        
        # Check intermediate results structure
        intermediate = extras['intermediate_results']
        assert len(intermediate) >= 2  # Input + at least one CUSP layer
        
        for result in intermediate:
            assert 'layer' in result
            assert 'features' in result
    
    def test_cusp_module_gradient_flow(self):
        """Test gradients flow through CUSP module."""
        input_dim = 12
        n_nodes = 15
        
        model = CuspModule(
            input_dim=input_dim,
            hidden_dim=16,
            output_dim=8,
            pooling_strategy="mean"
        )
        
        x = torch.randn(n_nodes, input_dim, requires_grad=True)
        edge_index = torch.randint(0, n_nodes, (2, n_nodes))
        edge_index = to_undirected(edge_index)
        
        output = model(x, edge_index)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
        
        # Check model parameters have gradients
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        assert has_gradients
    
    def test_cusp_module_curvature_statistics(self):
        """Test curvature statistics computation."""
        model = CuspModule(
            input_dim=16,
            hidden_dim=24,
            output_dim=12
        )
        
        n_nodes = 20
        edge_index = torch.randint(0, n_nodes, (2, n_nodes))
        edge_index = to_undirected(edge_index)
        
        stats = model.get_curvature_statistics(edge_index, n_nodes)
        
        required_keys = [
            'mean_curvature', 'std_curvature', 'min_curvature', 'max_curvature',
            'positive_curvature_ratio', 'negative_curvature_ratio'
        ]
        
        for key in required_keys:
            assert key in stats
            if key != 'error':  # Skip error key if present
                assert isinstance(stats[key], float)
    
    def test_cusp_module_ablation_study(self):
        """Test ablation study functionality."""
        input_dim = 14
        n_nodes = 16
        
        model = CuspModule(
            input_dim=input_dim,
            hidden_dim=20,
            output_dim=10,
            pooling_strategy="mean"
        )
        
        x = torch.randn(n_nodes, input_dim)
        edge_index = torch.randint(0, n_nodes, (2, n_nodes))
        edge_index = to_undirected(edge_index)
        
        ablation_results = model.ablation_study(x, edge_index)
        
        expected_configs = ['no_curvature', 'euclidean_only', 'no_gpr', 'full_cusp']
        
        for config in expected_configs:
            assert config in ablation_results
            result = ablation_results[config]
            assert result.shape == (1, 10)
            assert torch.all(torch.isfinite(result))
    
    def test_cusp_module_different_layers(self):
        """Test CUSP module with different number of layers."""
        input_dim = 16
        n_nodes = 20
        
        x = torch.randn(n_nodes, input_dim)
        edge_index = torch.randint(0, n_nodes, (2, n_nodes))
        edge_index = to_undirected(edge_index)
        
        for num_layers in [1, 2, 3]:
            model = CuspModule(
                input_dim=input_dim,
                hidden_dim=24,
                output_dim=12,
                num_layers=num_layers,
                pooling_strategy="mean"
            )
            
            output = model(x, edge_index)
            
            assert output.shape == (1, 12)
            assert torch.all(torch.isfinite(output))
            assert len(model.cusp_layers) == num_layers


class TestCuspModelFactory:
    """Test CUSP model factory function."""
    
    def test_create_cusp_model_default(self):
        """Test creating CUSP model with default configuration."""
        input_dim = 20
        num_classes = 2
        
        model = create_cusp_model(input_dim, num_classes)
        
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'classifier')
        
        # Test forward pass
        n_nodes = 25
        x = torch.randn(n_nodes, input_dim)
        edge_index = torch.randint(0, n_nodes, (2, n_nodes))
        edge_index = to_undirected(edge_index)
        
        logits = model(x, edge_index)
        assert logits.shape == (1, num_classes)
        assert torch.all(torch.isfinite(logits))
    
    def test_create_cusp_model_custom_config(self):
        """Test creating CUSP model with custom configuration."""
        input_dim = 24
        num_classes = 3
        
        config = {
            'hidden_dim': 48,
            'output_dim': 24,
            'num_layers': 3,
            'manifold_types': ["euclidean", "hyperbolic"],
            'pooling_strategy': "mean"
        }
        
        model = create_cusp_model(input_dim, num_classes, config)
        
        # Test that configuration was applied
        assert model.backbone.hidden_dim == 48
        assert model.backbone.output_dim == 24
        assert model.backbone.num_layers == 3
        assert model.backbone.manifold_types == ["euclidean", "hyperbolic"]
        
        # Test forward pass
        n_nodes = 20
        x = torch.randn(n_nodes, input_dim)
        edge_index = torch.randint(0, n_nodes, (2, n_nodes))
        edge_index = to_undirected(edge_index)
        
        logits = model(x, edge_index)
        assert logits.shape == (1, num_classes)
        assert torch.all(torch.isfinite(logits))
    
    def test_create_cusp_model_return_features(self):
        """Test CUSP model returning features."""
        input_dim = 16
        num_classes = 2
        
        model = create_cusp_model(input_dim, num_classes)
        
        n_nodes = 18
        x = torch.randn(n_nodes, input_dim)
        edge_index = torch.randint(0, n_nodes, (2, n_nodes))
        edge_index = to_undirected(edge_index)
        
        logits, features = model(x, edge_index, return_features=True)
        
        assert logits.shape == (1, num_classes)
        assert features.shape == (1, 32)  # Default output_dim
        assert torch.all(torch.isfinite(logits))
        assert torch.all(torch.isfinite(features))
    
    def test_create_cusp_model_gradient_flow(self):
        """Test gradient flow in CUSP classifier."""
        input_dim = 14
        num_classes = 2
        
        model = create_cusp_model(input_dim, num_classes)
        
        n_nodes = 15
        x = torch.randn(n_nodes, input_dim, requires_grad=True)
        edge_index = torch.randint(0, n_nodes, (2, n_nodes))
        edge_index = to_undirected(edge_index)
        
        logits = model(x, edge_index)
        loss = logits.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestCuspModuleIntegration:
    """Integration tests for complete CUSP module."""
    
    def test_end_to_end_fraud_detection_pipeline(self):
        """Test complete fraud detection pipeline with CUSP."""
        # Simulate fraud detection scenario
        input_dim = 32  # Transaction features
        num_classes = 2  # Fraud/legitimate
        batch_size = 3
        
        # Create model
        config = {
            'hidden_dim': 64,
            'output_dim': 32,
            'num_layers': 2,
            'manifold_types': ["euclidean", "hyperbolic", "spherical"],
            'pooling_strategy': "attention"
        }
        
        model = create_cusp_model(input_dim, num_classes, config)
        
        # Create test batch
        graphs = []
        for i in range(batch_size):
            n_nodes = 20 + i * 5  # Different graph sizes
            x = torch.randn(n_nodes, input_dim)
            edge_index = torch.randint(0, n_nodes, (2, n_nodes * 2))
            edge_index = to_undirected(edge_index)
            
            graphs.append(Data(x=x, edge_index=edge_index))
        
        # Create batch
        batch_data = Batch.from_data_list(graphs)
        
        # Forward pass
        logits = model(batch_data.x, batch_data.edge_index, batch_data.batch)
        
        assert logits.shape == (batch_size, num_classes)
        assert torch.all(torch.isfinite(logits))
        
        # Test with return features
        logits, features = model(
            batch_data.x, batch_data.edge_index, batch_data.batch, 
            return_features=True
        )
        
        assert logits.shape == (batch_size, num_classes)
        assert features.shape == (batch_size, 32)
    
    def test_cusp_module_robustness(self):
        """Test CUSP module robustness to edge cases."""
        input_dim = 16
        model = CuspModule(
            input_dim=input_dim,
            hidden_dim=24,
            output_dim=12,
            pooling_strategy="mean"
        )
        
        # Test with very small graph
        x_small = torch.randn(3, input_dim)
        edge_index_small = torch.tensor([[0, 1, 2], [1, 2, 0]])
        
        output_small = model(x_small, edge_index_small)
        assert output_small.shape == (1, 12)
        assert torch.all(torch.isfinite(output_small))
        
        # Test with disconnected nodes
        x_disconnected = torch.randn(5, input_dim)
        edge_index_disconnected = torch.tensor([[0, 1], [1, 0]])  # Only 2 nodes connected
        
        output_disconnected = model(x_disconnected, edge_index_disconnected)
        assert output_disconnected.shape == (1, 12)
        assert torch.all(torch.isfinite(output_disconnected))
    
    def test_cusp_module_memory_efficiency(self):
        """Test CUSP module memory usage."""
        input_dim = 32
        model = CuspModule(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=32,
            num_layers=2,
            pooling_strategy="mean"
        )
        
        # Test with larger graph
        n_nodes = 100
        x = torch.randn(n_nodes, input_dim)
        edge_index = torch.randint(0, n_nodes, (2, n_nodes * 3))
        edge_index = to_undirected(edge_index)
        
        # Forward pass
        output = model(x, edge_index)
        
        assert output.shape == (1, 32)
        assert torch.all(torch.isfinite(output))
        
        # Check memory usage is reasonable (no specific assertion, just runs)
        # In practice, would use torch.cuda.memory_allocated() if on GPU


if __name__ == "__main__":
    # Run tests
    test_classes = [
        TestCuspModule,
        TestCuspModelFactory,
        TestCuspModuleIntegration
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
                    import traceback
                    traceback.print_exc()
    
    print("\n🎯 All CUSP Integration tests completed!")
