"""
Tests for robustness modules.
Following Stage7 Reference §Phase3: test DropEdge determinism and RGNN wrapper stability.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from robustness import (
    DropEdge,
    RGNNWrapper,
    AdversarialEdgeTrainer,
    RobustnessAugmentations,
    create_robust_model,
    create_dropedge_transform,
    benchmark_robustness_overhead
)


class SimpleTestModel(nn.Module):
    """Simple model for testing robustness modules."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 16, output_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr=None, **kwargs) -> torch.Tensor:
        """Simple forward pass."""
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)


class TestDropEdge:
    """Test DropEdge functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_dropedge_initialization(self):
        """Test DropEdge initialization."""
        drop_edge = DropEdge(p_drop=0.2, training=True)
        
        assert drop_edge.p_drop == 0.2
        assert drop_edge.training_mode == True
    
    def test_dropedge_no_drop_during_eval(self):
        """Test that DropEdge doesn't drop edges during evaluation."""
        drop_edge = DropEdge(p_drop=0.5, training=False)
        
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        original_edges = edge_index.size(1)
        
        dropped_edge_index, dropped_attr = drop_edge(edge_index)
        
        # Should keep all edges during eval
        assert dropped_edge_index.size(1) == original_edges
        assert torch.equal(dropped_edge_index, edge_index)
    
    def test_dropedge_reproducible(self):
        """
        Test DropEdge reproducibility with seed.
        Following Stage7 Reference §Phase3: deterministic seeding requirement.
        """
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        
        # First run
        torch.manual_seed(42)
        drop_edge1 = DropEdge(p_drop=0.4, training=True)
        result1, _ = drop_edge1(edge_index)
        
        # Second run with same seed
        torch.manual_seed(42)
        drop_edge2 = DropEdge(p_drop=0.4, training=True)
        result2, _ = drop_edge2(edge_index)
        
        # Results should be identical
        assert torch.equal(result1, result2), "DropEdge should be deterministic with same seed"
        
        print(f"✅ DropEdge determinism: {result1.size(1)}/{edge_index.size(1)} edges kept consistently")
    
    def test_dropedge_with_edge_attributes(self):
        """Test DropEdge with edge attributes."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        edge_attr = torch.randn(3, 4)  # 3 edges, 4 features each
        
        drop_edge = DropEdge(p_drop=0.3, training=True)
        dropped_edge_index, dropped_attr = drop_edge(edge_index, edge_attr)
        
        # Edge index and attributes should have same filtering
        assert dropped_edge_index.size(1) == dropped_attr.size(0)
        assert dropped_attr.size(1) == edge_attr.size(1)  # Feature dimension preserved
        
        print(f"✅ DropEdge with attributes: {dropped_edge_index.size(1)}/3 edges kept")
    
    def test_dropedge_empty_graph(self):
        """Test DropEdge with empty graph."""
        empty_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        drop_edge = DropEdge(p_drop=0.5, training=True)
        result, _ = drop_edge(empty_edge_index)
        
        assert result.size() == empty_edge_index.size()
        assert result.size(1) == 0
    
    def test_dropedge_training_mode_switch(self):
        """Test DropEdge training mode switching."""
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
        
        drop_edge = DropEdge(p_drop=0.5, training=True)
        
        # Training mode - should drop edges
        drop_edge.train()
        train_result, _ = drop_edge(edge_index)
        
        # Eval mode - should keep all edges
        drop_edge.eval()
        eval_result, _ = drop_edge(edge_index)
        
        assert eval_result.size(1) == edge_index.size(1)  # All edges kept in eval
        print(f"✅ Mode switching: train={train_result.size(1)}, eval={eval_result.size(1)} edges")


class TestRGNNWrapper:
    """Test RGNN wrapper functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        
        self.base_model = SimpleTestModel(input_dim=8, hidden_dim=16, output_dim=3)
        self.test_data = {
            'x': torch.randn(10, 8),
            'edge_index': torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long),
            'y': torch.randint(0, 3, (10,))
        }
    
    def test_rgnn_wrapper_initialization(self):
        """Test RGNN wrapper initialization."""
        rgnn_model = RGNNWrapper(
            base_model=self.base_model,
            hidden_dim=16,
            enable_spectral_norm=True,
            enable_attention_gating=True
        )
        
        assert rgnn_model.base_model is self.base_model
        assert rgnn_model.hidden_dim == 16
        assert rgnn_model.enable_spectral_norm == True
        assert rgnn_model.enable_attention_gating == True
        assert hasattr(rgnn_model, 'attention_net')
    
    def test_rgnn_wrapper_forward_pass(self):
        """Test RGNN wrapper forward pass."""
        rgnn_model = RGNNWrapper(
            base_model=self.base_model,
            hidden_dim=16,
            enable_spectral_norm=False,  # Disable for simpler test
            enable_attention_gating=True
        )
        
        # Forward pass
        output = rgnn_model(self.test_data['x'], self.test_data['edge_index'])
        
        # Verify output shape
        assert output.size(0) == self.test_data['x'].size(0)  # Same number of nodes
        assert output.size(1) == 3  # Output dimension
        
        print(f"✅ RGNN forward pass: input {self.test_data['x'].shape} -> output {output.shape}")
    
    def test_rgnn_wrapper_stability(self):
        """
        Test RGNN wrapper doesn't catastrophically degrade performance.
        Following Stage7 Reference §Phase3: stability requirement.
        """
        # Original model
        original_output = self.base_model(self.test_data['x'], self.test_data['edge_index'])
        
        # RGNN wrapped model
        rgnn_model = RGNNWrapper(
            base_model=self.base_model,
            hidden_dim=16,
            enable_spectral_norm=True,
            enable_attention_gating=True,
            residual_weight=0.1
        )
        
        rgnn_output = rgnn_model(self.test_data['x'], self.test_data['edge_index'])
        
        # Outputs should have same shape
        assert original_output.shape == rgnn_output.shape
        
        # Compute difference (should not be catastrophically different)
        diff = (original_output - rgnn_output).abs().mean().item()
        assert diff < 10.0, f"RGNN wrapper caused too large change: {diff}"
        
        print(f"✅ RGNN stability: mean output difference = {diff:.4f}")
    
    def test_rgnn_attention_computation(self):
        """Test attention weight computation."""
        rgnn_model = RGNNWrapper(
            base_model=self.base_model,
            hidden_dim=16,
            enable_attention_gating=True
        )
        
        attention_weights = rgnn_model._compute_attention_weights(
            self.test_data['edge_index'],
            self.test_data['x']
        )
        
        # Should have one weight per edge
        assert attention_weights.size(0) == self.test_data['edge_index'].size(1)
        
        # Weights should be in reasonable range (sigmoid output)
        assert torch.all(attention_weights >= 0)
        assert torch.all(attention_weights <= rgnn_model.attention_clip)
        
        print(f"✅ Attention weights: range [{attention_weights.min():.3f}, {attention_weights.max():.3f}]")
    
    def test_rgnn_with_empty_edges(self):
        """Test RGNN wrapper with empty edge list."""
        empty_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        rgnn_model = RGNNWrapper(
            base_model=self.base_model,
            hidden_dim=16,
            enable_attention_gating=True
        )
        
        # Should not crash with empty edges
        output = rgnn_model(self.test_data['x'], empty_edge_index)
        assert output.shape == (10, 3)
        
        print("✅ RGNN handles empty edges gracefully")


class TestAdversarialEdgeTrainer:
    """Test adversarial edge trainer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        
        self.model = SimpleTestModel()
        self.batch_data = {
            'x': torch.randn(8, 10),
            'edge_index': torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
            'y': torch.randint(0, 2, (8,)),
            'train_mask': torch.tensor([True, True, False, False, True, True, False, False])
        }
    
    def test_adversarial_trainer_disabled(self):
        """Test adversarial trainer when disabled."""
        trainer = AdversarialEdgeTrainer(enabled=False)
        
        assert trainer.enabled == False
        
        # Should perform standard training when disabled
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        
        clean_loss, adv_loss = trainer.perturb_and_train(
            self.batch_data, self.model, optimizer, loss_fn
        )
        
        # Both losses should be the same when disabled
        assert clean_loss == adv_loss
        print(f"✅ Adversarial trainer disabled: loss={clean_loss:.4f}")
    
    def test_adversarial_trainer_enabled_basic(self):
        """Test basic adversarial trainer functionality."""
        trainer = AdversarialEdgeTrainer(
            epsilon=0.01,
            steps=2,
            enabled=True
        )
        
        assert trainer.enabled == True
        assert trainer.epsilon == 0.01
        assert trainer.steps == 2
        
        print("✅ Adversarial trainer enabled (basic test)")
    
    def test_edge_weight_perturbation(self):
        """Test edge weight perturbation generation."""
        trainer = AdversarialEdgeTrainer(
            epsilon=0.1,
            steps=2,
            enabled=True
        )
        
        edge_index = self.batch_data['edge_index']
        edge_weights = torch.ones(edge_index.size(1))
        loss_fn = nn.CrossEntropyLoss()
        
        # Generate perturbations
        perturbed_weights = trainer.perturb_edge_weights(
            edge_index, edge_weights, self.model, self.batch_data, loss_fn
        )
        
        # Verify perturbation properties
        assert perturbed_weights.shape == edge_weights.shape
        
        # Should be within epsilon ball (with some tolerance)
        diff = (perturbed_weights - edge_weights).abs()
        max_diff = diff.max().item()
        assert max_diff <= trainer.epsilon + 1e-4, f"Perturbation exceeds epsilon: max diff = {max_diff}"
        
        # Should be in valid range [0, 1] (with tolerance)
        assert torch.all(perturbed_weights >= -1e-6), f"Weights below 0: min = {perturbed_weights.min()}"
        assert torch.all(perturbed_weights <= 1 + 1e-6), f"Weights above 1: max = {perturbed_weights.max()}"
        
        print(f"✅ Edge perturbation: max change = {max_diff:.4f} (ε={trainer.epsilon})")


class TestRobustnessAugmentations:
    """Test additional robustness techniques."""
    
    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        
        self.model = SimpleTestModel(input_dim=8, hidden_dim=16, output_dim=2)  # Match feature dim
        self.features = torch.randn(10, 8)  # Match input_dim
        self.labels = torch.randint(0, 3, (10,))
    
    def test_label_smoothing(self):
        """Test label smoothing functionality."""
        augmentations = RobustnessAugmentations({
            'label_smoothing': 0.1
        })
        
        smoothed = augmentations.apply_label_smoothing(self.labels, num_classes=3)
        
        # Should be proper probability distribution
        assert torch.allclose(smoothed.sum(dim=1), torch.ones(10))
        
        # Max probability should be less than 1 due to smoothing
        assert torch.all(smoothed.max(dim=1)[0] < 1.0)
        
        print(f"✅ Label smoothing: max prob = {smoothed.max().item():.3f}")
    
    def test_feature_noise(self):
        """Test feature noise addition."""
        augmentations = RobustnessAugmentations({
            'feature_noise_std': 0.1
        })
        
        noisy_features = augmentations.add_feature_noise(self.features)
        
        # Should have same shape
        assert noisy_features.shape == self.features.shape
        
        # Should be different from original
        diff = (noisy_features - self.features).abs().mean()
        assert diff > 0, "Noise should change features"
        
        print(f"✅ Feature noise: mean change = {diff:.4f}")
    
    def test_gradient_clipping(self):
        """Test gradient clipping."""
        augmentations = RobustnessAugmentations({
            'gradient_clip': 1.0
        })
        
        # Create some gradients
        loss = torch.sum(self.model(self.features, torch.empty((2, 0), dtype=torch.long))**2)
        loss.backward()
        
        # Apply clipping
        grad_norm = augmentations.clip_gradients(self.model)
        
        assert isinstance(grad_norm, float)
        assert grad_norm >= 0
        
        print(f"✅ Gradient clipping: norm = {grad_norm:.4f}")


class TestRobustnessIntegration:
    """Integration tests for robustness modules."""
    
    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        
        self.base_model = SimpleTestModel()
        self.test_data = {
            'x': torch.randn(15, 10),
            'edge_index': torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
        }
    
    def test_create_robust_model(self):
        """Test robust model creation factory."""
        config = {
            'rgnn': {
                'enabled': True,
                'spectral_norm': True,
                'attention_clip': 3.0
            }
        }
        
        robust_model = create_robust_model(self.base_model, config)
        
        # Should be wrapped with RGNN
        assert isinstance(robust_model, RGNNWrapper)
        assert robust_model.base_model is self.base_model
        
        print("✅ Robust model factory: RGNN wrapper applied")
    
    def test_create_dropedge_transform(self):
        """Test DropEdge factory."""
        config = {'dropedge_p': 0.3}
        
        drop_edge = create_dropedge_transform(config)
        
        assert isinstance(drop_edge, DropEdge)
        assert drop_edge.p_drop == 0.3
        
        print("✅ DropEdge factory: p_drop=0.3")
    
    def test_benchmark_robustness_overhead(self):
        """Test robustness overhead benchmarking."""
        # Create robust model
        config = {
            'rgnn': {
                'enabled': True,
                'spectral_norm': False  # Disable for faster testing
            }
        }
        robust_model = create_robust_model(self.base_model, config)
        
        # Benchmark overhead
        results = benchmark_robustness_overhead(
            model=self.base_model,
            robust_model=robust_model,
            test_data=self.test_data,
            num_trials=3  # Small number for testing
        )
        
        # Verify results structure
        assert 'original_time_ms' in results
        assert 'robust_time_ms' in results
        assert 'overhead_ratio' in results
        assert 'overhead_ms' in results
        
        # Times should be positive
        assert results['original_time_ms'] > 0
        assert results['robust_time_ms'] > 0
        
        print(f"✅ Overhead benchmark: {results['overhead_ratio']:.2f}x slower, +{results['overhead_ms']:.2f}ms")


def test_dropedge_functionality():
    """Test runner for DropEdge."""
    print("\n🧪 Running DropEdge Tests...")
    
    test_dropedge = TestDropEdge()
    test_dropedge.setup_method()
    
    test_dropedge.test_dropedge_initialization()
    test_dropedge.test_dropedge_no_drop_during_eval()
    test_dropedge.test_dropedge_reproducible()
    test_dropedge.test_dropedge_with_edge_attributes()
    test_dropedge.test_dropedge_empty_graph()
    test_dropedge.test_dropedge_training_mode_switch()
    
    print("✅ All DropEdge tests passed!")


def test_rgnn_wrapper_functionality():
    """Test runner for RGNN wrapper."""
    print("\n🧪 Running RGNN Wrapper Tests...")
    
    test_rgnn = TestRGNNWrapper()
    test_rgnn.setup_method()
    
    test_rgnn.test_rgnn_wrapper_initialization()
    test_rgnn.test_rgnn_wrapper_forward_pass()
    test_rgnn.test_rgnn_wrapper_stability()
    test_rgnn.test_rgnn_attention_computation()
    test_rgnn.test_rgnn_with_empty_edges()
    
    print("✅ All RGNN wrapper tests passed!")


def test_robustness_modules():
    """Main test runner for all robustness modules."""
    print("\n🧪 Running Robustness Module Tests...")
    
    # DropEdge tests
    test_dropedge_functionality()
    
    # RGNN wrapper tests
    test_rgnn_wrapper_functionality()
    
    # Adversarial trainer tests
    test_adversarial = TestAdversarialEdgeTrainer()
    test_adversarial.setup_method()
    test_adversarial.test_adversarial_trainer_disabled()
    test_adversarial.test_adversarial_trainer_enabled_basic()
    test_adversarial.test_edge_weight_perturbation()
    
    # Augmentations tests
    test_augmentations = TestRobustnessAugmentations()
    test_augmentations.setup_method()
    test_augmentations.test_label_smoothing()
    test_augmentations.test_feature_noise()
    test_augmentations.test_gradient_clipping()
    
    # Integration tests
    test_integration = TestRobustnessIntegration()
    test_integration.setup_method()
    test_integration.test_create_robust_model()
    test_integration.test_create_dropedge_transform()
    test_integration.test_benchmark_robustness_overhead()
    
    print("✅ All robustness module tests passed!")


if __name__ == "__main__":
    test_robustness_modules()
