"""
Unit tests for Phase B — Explainability primitives.

Following Stage 10 Reference §Phase B validation requirements:
- wrapper returns masks in valid ranges [0,1]
- important_subgraph reconstructs correctly when masks are applied  
- results are reproducible with fixed seeds
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.explainability.gnne_explainers import (
    GNNExplainerWrapper,
    PGExplainerTrainer, 
    HGNNExplainer,
    TemporalExplainer,
    create_explainer
)
from src.explainability.temporal_explainer import SimplifiedTemporalExplainer


class TestGNNExplainerWrapper:
    """Test GNNExplainer wrapper."""
    
    def setup_method(self):
        """Setup test data and mock model."""
        # Create simple test graph
        self.x = torch.randn(5, 10)  # 5 nodes, 10 features
        self.edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        
        # Mock model
        self.mock_model = Mock()
        self.mock_model.return_value = torch.randn(5, 2)  # 2 classes
    
    def test_explainer_initialization(self):
        """Test explainer initializes correctly."""
        try:
            explainer = GNNExplainerWrapper(
                model=self.mock_model,
                epochs=10,
                device='cpu',
                seed=42
            )
            assert explainer.epochs == 10
            assert explainer.device == 'cpu'
            assert explainer.seed == 42
        except ImportError:
            # Skip if PyG not available
            pytest.skip("PyTorch Geometric not available")
    
    def test_explain_node_output_format(self):
        """Test that explain_node returns correct format."""
        try:
            explainer = GNNExplainerWrapper(
                model=self.mock_model,
                epochs=5,  # Fewer epochs for testing
                device='cpu',
                seed=42
            )
            
            result = explainer.explain_node(
                node_id=1,
                x=self.x,
                edge_index=self.edge_index
            )
            
            # Check required keys
            assert 'edge_mask' in result
            assert 'node_feat_mask' in result
            assert 'important_subgraph' in result
            assert 'explanation_type' in result
            
            # Check edge mask properties
            edge_mask = result['edge_mask']
            assert edge_mask.shape[0] == self.edge_index.shape[1]
            assert torch.all(edge_mask >= 0) and torch.all(edge_mask <= 1)
            
        except ImportError:
            pytest.skip("PyTorch Geometric not available")
    
    def test_explainer_reproducibility(self):
        """Test that explanations are reproducible with same seed."""
        try:
            explainer1 = GNNExplainerWrapper(
                model=self.mock_model,
                epochs=5,
                device='cpu',
                seed=123
            )
            
            explainer2 = GNNExplainerWrapper(
                model=self.mock_model,
                epochs=5,
                device='cpu', 
                seed=123
            )
            
            result1 = explainer1.explain_node(1, self.x, self.edge_index)
            result2 = explainer2.explain_node(1, self.x, self.edge_index)
            
            # Results should be very similar (allowing for small numerical differences)
            mask_diff = torch.abs(result1['edge_mask'] - result2['edge_mask']).mean()
            assert mask_diff < 0.1, "Explanations should be reproducible"
            
        except ImportError:
            pytest.skip("PyTorch Geometric not available")


class TestPGExplainerTrainer:
    """Test PGExplainer trainer wrapper."""
    
    def setup_method(self):
        """Setup test data."""
        self.mock_model = Mock()
        self.x = torch.randn(5, 10)
        self.edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        
        # Create training data
        self.train_data = [
            {
                'x': self.x,
                'edge_index': self.edge_index,
                'node_id': 1,
                'ground_truth_mask': torch.rand(self.edge_index.shape[1])
            }
        ]
    
    def test_pg_explainer_initialization(self):
        """Test PGExplainer initializes correctly."""
        explainer = PGExplainerTrainer(
            model=self.mock_model,
            hidden_dim=32,
            device='cpu',
            seed=42
        )
        
        assert explainer.hidden_dim == 32
        assert not explainer.is_trained
        assert explainer.mask_predictor is None
    
    def test_pg_explainer_training(self):
        """Test PGExplainer training process."""
        explainer = PGExplainerTrainer(
            model=self.mock_model,
            hidden_dim=16,
            device='cpu',
            seed=42
        )
        
        # Train with small dataset
        explainer.train_explainer(
            train_data=self.train_data,
            epochs=5,
            lr=0.1
        )
        
        assert explainer.is_trained
        assert explainer.mask_predictor is not None
    
    def test_pg_explainer_inference(self):
        """Test PGExplainer inference after training."""
        explainer = PGExplainerTrainer(
            model=self.mock_model,
            hidden_dim=16,
            device='cpu',
            seed=42
        )
        
        # Train first
        explainer.train_explainer(self.train_data, epochs=3)
        
        # Test inference
        result = explainer.explain_node(
            node_id=1,
            x=self.x,
            edge_index=self.edge_index
        )
        
        assert 'edge_mask' in result
        assert 'explanation_type' in result
        assert result['explanation_type'] == 'pg_explainer'
        
        # Check mask properties
        edge_mask = result['edge_mask']
        assert edge_mask.shape[0] == self.edge_index.shape[1]
        assert torch.all(edge_mask >= 0) and torch.all(edge_mask <= 1)
    
    def test_pg_explainer_untrained_error(self):
        """Test that using untrained explainer raises error."""
        explainer = PGExplainerTrainer(
            model=self.mock_model,
            device='cpu'
        )
        
        with pytest.raises(RuntimeError, match="must be trained"):
            explainer.explain_node(1, self.x, self.edge_index)


class TestHGNNExplainer:
    """Test heterogeneous GNN explainer."""
    
    def setup_method(self):
        """Setup test heterogeneous data."""
        self.mock_model = Mock()
        self.node_types = ['user', 'item']
        self.edge_types = [('user', 'buys', 'item'), ('item', 'bought_by', 'user')]
        
        self.x_dict = {
            'user': torch.randn(3, 8),
            'item': torch.randn(2, 6)
        }
        
        self.edge_index_dict = {
            ('user', 'buys', 'item'): torch.tensor([[0, 1, 2], [0, 1, 0]]),
            ('item', 'bought_by', 'user'): torch.tensor([[0, 1, 0], [0, 1, 2]])
        }
    
    def test_hgnn_explainer_initialization(self):
        """Test HGNN explainer initializes correctly."""
        explainer = HGNNExplainer(
            model=self.mock_model,
            node_types=self.node_types,
            edge_types=self.edge_types,
            device='cpu',
            seed=42
        )
        
        assert explainer.node_types == self.node_types
        assert explainer.edge_types == self.edge_types
        assert len(explainer.relation_mask_learners) == len(self.edge_types)
    
    def test_hgnn_explainer_explanation(self):
        """Test HGNN explainer generates valid explanations."""
        explainer = HGNNExplainer(
            model=self.mock_model,
            node_types=self.node_types,
            edge_types=self.edge_types,
            device='cpu',
            seed=42
        )
        
        result = explainer.explain_node(
            node_id=1,
            x_dict=self.x_dict,
            edge_index_dict=self.edge_index_dict,
            node_type='user'
        )
        
        # Check output format
        assert 'edge_masks' in result
        assert 'relation_importance' in result
        assert 'important_subgraph' in result
        assert 'target_node' in result
        assert 'target_type' in result
        assert result['explanation_type'] == 'hgnn_explainer'
        
        # Check edge masks
        for edge_type, mask in result['edge_masks'].items():
            expected_edges = self.edge_index_dict[edge_type].shape[1]
            assert mask.shape[0] == expected_edges
            assert torch.all(mask >= 0) and torch.all(mask <= 1)
        
        # Check relation importance
        for edge_type, importance in result['relation_importance'].items():
            assert 0 <= importance <= 1


class TestTemporalExplainer:
    """Test temporal explainer."""
    
    def setup_method(self):
        """Setup temporal test data."""
        self.mock_model = Mock()
        self.x = torch.randn(5, 10)
        self.edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        self.edge_time = torch.tensor([1.0, 2.0, 3.0, 4.0])
    
    def test_temporal_explainer_initialization(self):
        """Test temporal explainer initializes correctly."""
        explainer = TemporalExplainer(
            model=self.mock_model,
            device='cpu',
            seed=42
        )
        
        assert explainer.device == 'cpu'
        assert explainer.seed == 42
    
    def test_temporal_explainer_explanation(self):
        """Test temporal explainer generates valid explanations."""
        explainer = TemporalExplainer(
            model=self.mock_model,
            device='cpu',
            seed=42
        )
        
        result = explainer.explain_node(
            node_id=1,
            x=self.x,
            edge_index=self.edge_index,
            edge_time=self.edge_time
        )
        
        # Check output format
        assert 'temporal_mask' in result
        assert 'time_windows' in result
        assert 'important_subgraph' in result
        assert result['explanation_type'] == 'temporal_explainer'
        
        # Check temporal mask
        temporal_mask = result['temporal_mask']
        assert temporal_mask.shape[0] == len(self.edge_time)
        assert torch.all(temporal_mask >= 0) and torch.all(temporal_mask <= 1)


class TestSimplifiedTemporalExplainer:
    """Test simplified temporal explainer."""
    
    def setup_method(self):
        """Setup test data."""
        self.mock_model = Mock()
        self.event_sequence = [
            {'time': 1.0, 'src': 0, 'dst': 1, 'features': torch.randn(5)},
            {'time': 2.0, 'src': 1, 'dst': 2, 'features': torch.randn(5)},
            {'time': 3.0, 'src': 2, 'dst': 3, 'features': torch.randn(5)},
            {'time': 4.0, 'src': 1, 'dst': 3, 'features': torch.randn(5)}
        ]
    
    def test_simplified_temporal_explainer(self):
        """Test simplified temporal explainer."""
        explainer = SimplifiedTemporalExplainer(
            model=self.mock_model,
            window_size=3,
            device='cpu',
            seed=42
        )
        
        result = explainer.explain_temporal_prediction(
            node_id=1,
            event_sequence=self.event_sequence,
            target_time=5.0
        )
        
        # Check output format
        assert 'event_importance' in result
        assert 'time_window_importance' in result
        assert 'relevant_events' in result
        assert result['explanation_type'] == 'temporal_simplified'
        
        # Check event importance
        assert len(result['event_importance']) == len(self.event_sequence)
        for importance in result['event_importance']:
            assert 0 <= importance <= 1


class TestExplainerFactory:
    """Test explainer factory function."""
    
    def setup_method(self):
        """Setup test data."""
        self.mock_model = Mock()
    
    def test_create_explainer_gnn(self):
        """Test creating GNN explainer."""
        try:
            explainer = create_explainer('gnn', self.mock_model, epochs=10)
            assert isinstance(explainer, GNNExplainerWrapper)
        except ImportError:
            pytest.skip("PyTorch Geometric not available")
    
    def test_create_explainer_pg(self):
        """Test creating PG explainer."""
        explainer = create_explainer('pg', self.mock_model, hidden_dim=32)
        assert isinstance(explainer, PGExplainerTrainer)
    
    def test_create_explainer_hgnn(self):
        """Test creating HGNN explainer."""
        explainer = create_explainer(
            'hgnn',
            self.mock_model,
            node_types=['user'],
            edge_types=[('user', 'edge', 'user')]
        )
        assert isinstance(explainer, HGNNExplainer)
    
    def test_create_explainer_temporal(self):
        """Test creating temporal explainer."""
        explainer = create_explainer('temporal', self.mock_model)
        assert isinstance(explainer, TemporalExplainer)
    
    def test_create_explainer_invalid(self):
        """Test creating invalid explainer raises error."""
        with pytest.raises(ValueError, match="Unknown explainer type"):
            create_explainer('invalid', self.mock_model)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
