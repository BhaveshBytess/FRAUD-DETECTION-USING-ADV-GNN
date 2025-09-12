"""
Integration tests for SpotTarget training wrapper.
Following Stage7 Reference §Phase2: test training wrapper on small synthetic dataset.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from training_wrapper import (
    SpotTargetTrainingWrapper,
    train_epoch_with_spottarget,
    validate_with_leakage_check,
    SpotTargetTrainer,
    create_spottarget_trainer,
    benchmark_filtering_overhead
)
from spot_target import SpotTargetSampler, compute_avg_degree, setup_spottarget_sampler


class SimpleGNN(nn.Module):
    """Simple GNN for testing."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """Simple message passing."""
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        
        # Simple aggregation - mean of neighbors
        num_nodes = x.size(0)
        aggregated = torch.zeros_like(h)
        
        if edge_index.size(1) > 0:
            src, dst = edge_index
            aggregated.scatter_add_(0, dst.unsqueeze(1).expand(-1, h.size(1)), h[src])
            
            # Normalize by degree
            degree = torch.zeros(num_nodes, device=x.device)
            degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
            degree = torch.clamp(degree, min=1.0)
            aggregated = aggregated / degree.unsqueeze(1)
        
        output = self.fc2(aggregated + h)  # Skip connection
        return output


class SyntheticDataset:
    """Synthetic dataset for testing."""
    
    def __init__(self, num_nodes: int = 100, num_features: int = 10, num_classes: int = 2):
        torch.manual_seed(42)
        
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Create random features
        self.x = torch.randn(num_nodes, num_features)
        
        # Create random graph structure
        edge_prob = 0.1
        edges = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if torch.rand(1).item() < edge_prob:
                    edges.extend([(i, j), (j, i)])  # Undirected
        
        if len(edges) == 0:
            # Ensure at least some edges
            edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
        
        self.edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        # Create random labels
        self.y = torch.randint(0, num_classes, (num_nodes,))
        
        # Create train/test/val splits
        indices = torch.randperm(num_nodes)
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        self.train_mask[train_indices] = True
        self.val_mask[val_indices] = True
        self.test_mask[test_indices] = True
        
        # Create edge splits for SpotTarget
        num_edges = self.edge_index.size(1)
        edge_indices = torch.randperm(num_edges)
        
        train_edge_size = int(0.7 * num_edges)
        val_edge_size = int(0.15 * num_edges)
        
        self.edge_splits = {
            'train': torch.zeros(num_edges, dtype=torch.bool),
            'valid': torch.zeros(num_edges, dtype=torch.bool),
            'test': torch.zeros(num_edges, dtype=torch.bool)
        }
        
        self.edge_splits['train'][edge_indices[:train_edge_size]] = True
        self.edge_splits['valid'][edge_indices[train_edge_size:train_edge_size+val_edge_size]] = True
        self.edge_splits['test'][edge_indices[train_edge_size+val_edge_size:]] = True
    
    def to(self, device):
        """Move data to device."""
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.y = self.y.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        
        for key in self.edge_splits:
            self.edge_splits[key] = self.edge_splits[key].to(device)
        
        return self
    
    def get_batch_data(self) -> Dict[str, torch.Tensor]:
        """Get batch data dict for training."""
        return {
            'x': self.x,
            'edge_index': self.edge_index,
            'y': self.y,
            'train_mask': self.train_mask,
            'val_mask': self.val_mask,
            'batch_edge_indices': torch.arange(self.edge_index.size(1))
        }


class MockDataLoader:
    """Mock data loader for testing."""
    
    def __init__(self, dataset: SyntheticDataset, batch_size: int = 1):
        self.dataset = dataset
        self.batch_size = batch_size
    
    def __iter__(self):
        # Simple mock - yield same batch
        for _ in range(self.batch_size):
            yield self.dataset.get_batch_data()
    
    def __len__(self):
        return self.batch_size


class TestTrainingWrapperIntegration:
    """Integration tests for training wrapper."""
    
    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create synthetic dataset
        self.dataset = SyntheticDataset(num_nodes=50, num_features=8, num_classes=2)
        
        # Create model
        self.model = SimpleGNN(
            input_dim=8,
            hidden_dim=16,
            output_dim=2
        )
        
        # Create SpotTarget sampler
        _, degrees = compute_avg_degree(self.dataset.edge_index, self.dataset.num_nodes)
        self.sampler = SpotTargetSampler(
            edge_index=self.dataset.edge_index,
            train_edge_mask=self.dataset.edge_splits['train'],
            degrees=degrees,
            delta=3,  # reasonable threshold
            verbose=True
        )
    
    def test_training_wrapper_initialization(self):
        """Test training wrapper initialization."""
        config = {'verbose': True, 'delta': 3}
        
        wrapper = SpotTargetTrainingWrapper(
            model=self.model,
            sampler=self.sampler,
            config=config
        )
        
        assert wrapper.model is self.model
        assert wrapper.sampler is self.sampler
        assert wrapper.config['verbose'] == True
        assert wrapper.batch_count == 0
    
    def test_apply_spottarget_to_batch(self):
        """Test SpotTarget filtering on batch data."""
        wrapper = SpotTargetTrainingWrapper(
            model=self.model,
            sampler=self.sampler,
            config={'verbose': False}
        )
        
        batch_data = self.dataset.get_batch_data()
        original_edges = batch_data['edge_index'].size(1)
        
        # Apply filtering
        filtered_batch = wrapper.apply_spottarget_to_batch(batch_data)
        
        # Verify filtering occurred
        filtered_edges = filtered_batch['edge_index'].size(1)
        assert filtered_edges <= original_edges, "Filtering should not add edges"
        
        # Verify other data preserved
        assert torch.equal(filtered_batch['x'], batch_data['x'])
        assert torch.equal(filtered_batch['y'], batch_data['y'])
        
        # Verify statistics updated
        assert wrapper.batch_count == 1
        assert wrapper.total_original_edges > 0
        
        print(f"✅ Batch filtering: {original_edges} -> {filtered_edges} edges")
    
    def test_train_step(self):
        """Test single training step with SpotTarget."""
        wrapper = SpotTargetTrainingWrapper(
            model=self.model,
            sampler=self.sampler,
            config={'verbose': False}
        )
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        batch_data = self.dataset.get_batch_data()
        
        # Training step
        loss, accuracy = wrapper.train_step(batch_data, optimizer, criterion)
        
        # Verify outputs
        assert isinstance(loss, float) and loss >= 0
        assert isinstance(accuracy, float) and 0 <= accuracy <= 1
        
        print(f"✅ Training step: loss={loss:.4f}, accuracy={accuracy:.4f}")
    
    def test_train_epoch_with_spottarget(self):
        """
        Test full epoch training with SpotTarget.
        Following Stage7 Reference §Phase2: integration test on small synthetic dataset.
        """
        train_loader = MockDataLoader(self.dataset, batch_size=3)
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Train one epoch
        stats = train_epoch_with_spottarget(
            model=self.model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            sampler=self.sampler,
            config={'verbose': True}
        )
        
        # Verify statistics
        assert stats.batch_count == 3
        assert stats.epoch_loss >= 0
        assert 0 <= stats.epoch_accuracy <= 1
        assert stats.edges_original_total > 0
        assert stats.filtering_overhead_ms >= 0
        
        print(f"✅ Epoch training: {stats.batch_count} batches, loss={stats.epoch_loss:.4f}")
        print(f"   Filtering: {stats.edges_filtered_total}/{stats.edges_original_total} edges")
    
    def test_validate_with_leakage_check(self):
        """Test validation with leakage-safe inference."""
        val_loader = MockDataLoader(self.dataset, batch_size=2)
        criterion = nn.CrossEntropyLoss()
        
        # Validation with leakage check
        val_loss, val_accuracy = validate_with_leakage_check(
            model=self.model,
            val_loader=val_loader,
            criterion=criterion,
            edge_splits=self.dataset.edge_splits,
            original_edge_index=self.dataset.edge_index,
            exclude_validation=False
        )
        
        # Verify outputs
        assert isinstance(val_loss, float) and val_loss >= 0
        assert isinstance(val_accuracy, float) and 0 <= val_accuracy <= 1
        
        print(f"✅ Validation (leakage-safe): loss={val_loss:.4f}, accuracy={val_accuracy:.4f}")
    
    def test_spottarget_trainer_class(self):
        """Test complete SpotTarget trainer class."""
        trainer = SpotTargetTrainer(
            model=self.model,
            edge_index=self.dataset.edge_index,
            edge_splits=self.dataset.edge_splits,
            num_nodes=self.dataset.num_nodes,
            config={'verbose': True, 'delta': 'auto'}
        )
        
        # Test training
        train_loader = MockDataLoader(self.dataset, batch_size=2)
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        stats = trainer.train_epoch(train_loader, optimizer, criterion)
        assert stats.batch_count == 2
        
        # Test validation
        val_loader = MockDataLoader(self.dataset, batch_size=1)
        val_loss, val_acc = trainer.validate(val_loader, criterion)
        assert val_loss >= 0 and 0 <= val_acc <= 1
        
        # Test statistics
        sampler_stats = trainer.get_sampler_stats()
        assert 'delta' in sampler_stats
        assert 'exclusion_rate' in sampler_stats
        
        print(f"✅ Complete trainer: δ={sampler_stats['delta']}, exclusion_rate={sampler_stats['exclusion_rate']:.3f}")
    
    def test_create_spottarget_trainer_factory(self):
        """Test factory function for trainer creation."""
        graph_data = {
            'edge_index': self.dataset.edge_index,
            'edge_splits': self.dataset.edge_splits,
            'num_nodes': self.dataset.num_nodes
        }
        
        trainer = create_spottarget_trainer(
            model=self.model,
            graph_data=graph_data
        )
        
        assert isinstance(trainer, SpotTargetTrainer)
        assert trainer.model is self.model
        
        print(f"✅ Factory creation: trainer with δ={trainer.sampler.delta}")
    
    def test_no_crash_with_minimal_edges(self):
        """Test training doesn't crash with minimal edges after filtering."""
        # Create dataset with very few edges
        minimal_dataset = SyntheticDataset(num_nodes=10, num_features=4, num_classes=2)
        
        # Force most edges to be train targets with low degrees
        num_edges = minimal_dataset.edge_index.size(1)
        minimal_dataset.edge_splits['train'] = torch.ones(num_edges, dtype=torch.bool)
        
        # Create sampler with high delta to filter most edges
        _, degrees = compute_avg_degree(minimal_dataset.edge_index, minimal_dataset.num_nodes)
        high_delta_sampler = SpotTargetSampler(
            edge_index=minimal_dataset.edge_index,
            train_edge_mask=minimal_dataset.edge_splits['train'],
            degrees=degrees,
            delta=100,  # very high threshold
            verbose=True
        )
        
        # Test training doesn't crash
        model = SimpleGNN(input_dim=4, hidden_dim=8, output_dim=2)
        train_loader = MockDataLoader(minimal_dataset, batch_size=1)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        stats = train_epoch_with_spottarget(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            sampler=high_delta_sampler,
            config={'verbose': True}
        )
        
        # Should complete without crashing
        assert stats.batch_count == 1
        print(f"✅ Minimal edges test: completed with {stats.edges_original_total - stats.edges_filtered_total} remaining edges")
    
    def test_benchmark_filtering_overhead(self):
        """Test filtering overhead benchmark."""
        results = benchmark_filtering_overhead(
            sampler=self.sampler,
            batch_sizes=[10, 20, 30],
            num_trials=3
        )
        
        assert len(results['mean_times']) == 3
        assert len(results['std_times']) == 3
        assert all(t >= 0 for t in results['mean_times'])
        
        print(f"✅ Overhead benchmark: {results['mean_times']} ms (mean times)")


def test_training_wrapper_integration():
    """Main test runner for training wrapper integration."""
    print("\n🧪 Running Training Wrapper Integration Tests...")
    
    test_integration = TestTrainingWrapperIntegration()
    test_integration.setup_method()
    
    # Run all integration tests
    test_integration.test_training_wrapper_initialization()
    test_integration.test_apply_spottarget_to_batch()
    test_integration.test_train_step()
    test_integration.test_train_epoch_with_spottarget()
    test_integration.test_validate_with_leakage_check()
    test_integration.test_spottarget_trainer_class()
    test_integration.test_create_spottarget_trainer_factory()
    test_integration.test_no_crash_with_minimal_edges()
    test_integration.test_benchmark_filtering_overhead()
    
    print("✅ All training wrapper integration tests passed!")


if __name__ == "__main__":
    test_training_wrapper_integration()
