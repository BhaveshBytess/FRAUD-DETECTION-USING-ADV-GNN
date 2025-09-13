"""
Unit tests for Phase D — Integration API.

Tests explain_instance() pipeline, ExplainabilityPipeline, and API endpoints.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.explainability.integration import (
    explain_instance,
    ExplainabilityPipeline,
    ExplainabilityConfig,
    _generate_explanation_text,
    _get_node_type
)

try:
    from src.explainability.api import ExplainabilityAPI
    HAS_API = True
except ImportError:
    HAS_API = False


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, output_dim=2):
        super().__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(10, output_dim)
    
    def forward(self, data):
        # Create mock output based on data
        if hasattr(data, 'x') and data.x is not None:
            batch_size = data.x.size(0)
        else:
            batch_size = 10  # Default
        
        # Return logits that will result in predictable probabilities
        logits = torch.randn(batch_size, self.output_dim)
        logits[0] = torch.tensor([0.5, 2.0])  # High fraud probability for node 0
        return logits


class MockData:
    """Mock graph data for testing."""
    
    def __init__(self, num_nodes=10, num_features=5):
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.x = torch.randn(num_nodes, num_features)
        self.edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        self.num_edges = 4
        self.feature_names = [f'feature_{i}' for i in range(num_features)]
    
    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        return self


class TestExplainabilityConfig:
    """Test explainability configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ExplainabilityConfig()
        
        assert config.explainer_type == 'gnn_explainer'
        assert config.k_hops == 2
        assert config.max_nodes == 50
        assert config.edge_mask_threshold == 0.1
        assert config.feature_threshold == 0.05
        assert config.top_k_features == 10
        assert config.visualization is True
        assert config.save_reports is True
        assert config.seed == 42
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ExplainabilityConfig(
            explainer_type='pg_explainer',
            k_hops=3,
            max_nodes=100,
            top_k_features=5,
            visualization=False,
            seed=123
        )
        
        assert config.explainer_type == 'pg_explainer'
        assert config.k_hops == 3
        assert config.max_nodes == 100
        assert config.top_k_features == 5
        assert config.visualization is False
        assert config.seed == 123
    
    def test_output_dir_creation(self):
        """Test that output directory is created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_explanations')
            config = ExplainabilityConfig(output_dir=output_path)
            
            assert config.output_dir.exists()
            assert config.output_dir.is_dir()


class TestExplainInstance:
    """Test explain_instance() function."""
    
    def setup_method(self):
        """Setup test data."""
        self.model = MockModel()
        self.data = MockData()
        self.temp_dir = tempfile.mkdtemp()
        self.config = ExplainabilityConfig(
            output_dir=self.temp_dir,
            visualization=False,  # Skip viz for faster tests
            save_reports=False    # Skip reports for faster tests
        )
    
    def teardown_method(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.explainability.integration.SubgraphExtractor')
    @patch('src.explainability.integration.GNNExplainerWrapper')
    def test_explain_instance_basic(self, mock_explainer_class, mock_extractor_class):
        """Test basic explain_instance functionality."""
        # Mock extractor
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract.return_value = self.data
        
        # Mock explainer
        mock_explainer = Mock()
        mock_explainer_class.return_value = mock_explainer
        mock_explainer.explain.return_value = {
            'edge_mask': torch.tensor([0.9, 0.7, 0.3, 0.8]),
            'node_feat_mask': torch.tensor([0.5, 0.6, 0.4, 0.7, 0.2])
        }
        
        result = explain_instance(
            model=self.model,
            data=self.data,
            node_id=0,
            config=self.config,
            device='cpu'
        )
        
        # Check result structure
        assert 'node_id' in result
        assert 'prediction' in result
        assert 'explanation_masks' in result
        assert 'top_features' in result
        assert 'subgraph_info' in result
        assert 'explanation_text' in result
        
        assert result['node_id'] == 0
        assert isinstance(result['prediction'], float)
        assert 0 <= result['prediction'] <= 1
    
    @patch('src.explainability.integration.SubgraphExtractor')
    @patch('src.explainability.integration.PGExplainerTrainer')
    def test_explain_instance_pg_explainer(self, mock_explainer_class, mock_extractor_class):
        """Test explain_instance with PG explainer."""
        # Setup mocks
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract.return_value = self.data
        
        mock_explainer = Mock()
        mock_explainer_class.return_value = mock_explainer
        mock_explainer.explain.return_value = {
            'edge_mask': torch.tensor([0.9, 0.5]),
            'node_feat_mask': torch.tensor([0.8, 0.6])
        }
        
        config = ExplainabilityConfig(
            explainer_type='pg_explainer',
            output_dir=self.temp_dir,
            visualization=False,
            save_reports=False
        )
        
        result = explain_instance(
            model=self.model,
            data=self.data,
            node_id=1,
            config=config,
            device='cpu'
        )
        
        assert result['node_id'] == 1
        assert 'edge_mask' in result['explanation_masks']
        assert 'node_feat_mask' in result['explanation_masks']
    
    @patch('src.explainability.integration.SubgraphExtractor')
    def test_explain_instance_invalid_explainer(self, mock_extractor_class):
        """Test explain_instance with invalid explainer type."""
        # Mock extractor to avoid graph type issues
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract.return_value = self.data
        
        config = ExplainabilityConfig(explainer_type='invalid_explainer')
        
        with pytest.raises(ValueError, match="Unknown explainer type"):
            explain_instance(
                model=self.model,
                data=self.data,
                node_id=0,
                config=config,
                device='cpu'
            )
    
    @patch('src.explainability.integration.SubgraphExtractor')
    @patch('src.explainability.integration.GNNExplainerWrapper')
    def test_explain_instance_with_visualization(self, mock_explainer_class, mock_extractor_class):
        """Test explain_instance with visualization enabled."""
        # Setup mocks
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract.return_value = self.data
        
        mock_explainer = Mock()
        mock_explainer_class.return_value = mock_explainer
        mock_explainer.explain.return_value = {
            'edge_mask': torch.tensor([0.9, 0.7]),
            'node_feat_mask': torch.tensor([0.8, 0.6, 0.5])
        }
        
        config = ExplainabilityConfig(
            output_dir=self.temp_dir,
            visualization=True,
            save_reports=False
        )
        
        with patch('src.explainability.integration.visualize_subgraph') as mock_viz:
            mock_viz.return_value = {'static': 'test.png'}
            
            with patch('src.explainability.integration.create_feature_importance_plot') as mock_feat_plot:
                mock_feat_plot.return_value = 'features.png'
                
                result = explain_instance(
                    model=self.model,
                    data=self.data,
                    node_id=0,
                    config=config,
                    device='cpu'
                )
        
        assert 'visualization_paths' in result
        assert result['visualization_paths'] is not None


class TestExplainabilityPipeline:
    """Test ExplainabilityPipeline class."""
    
    def setup_method(self):
        """Setup test data."""
        self.model = MockModel()
        self.data = MockData()
        self.temp_dir = tempfile.mkdtemp()
        self.config = ExplainabilityConfig(
            output_dir=self.temp_dir,
            visualization=False,
            save_reports=False
        )
        self.pipeline = ExplainabilityPipeline(self.model, self.config, 'cpu')
    
    def teardown_method(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.explainability.integration.explain_instance')
    def test_explain_nodes(self, mock_explain):
        """Test explaining multiple nodes."""
        # Mock explain_instance to return predictable results
        def mock_explain_side_effect(model, data, node_id, config, device):
            return {
                'node_id': node_id,
                'prediction': 0.7,
                'explanation_masks': {},
                'top_features': [],
                'explanation_text': f'Explanation for node {node_id}'
            }
        
        mock_explain.side_effect = mock_explain_side_effect
        
        node_ids = [0, 1, 2]
        results = self.pipeline.explain_nodes(self.data, node_ids)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result['node_id'] == node_ids[i]
            assert 'error' not in result
    
    @patch('src.explainability.integration.explain_instance')
    def test_explain_nodes_with_error(self, mock_explain):
        """Test handling errors during node explanation."""
        # First call succeeds, second fails
        mock_explain.side_effect = [
            {'node_id': 0, 'prediction': 0.7},
            Exception("Test error"),
            {'node_id': 2, 'prediction': 0.3}
        ]
        
        node_ids = [0, 1, 2]
        results = self.pipeline.explain_nodes(self.data, node_ids)
        
        assert len(results) == 3
        assert 'error' not in results[0]
        assert 'error' in results[1]
        assert results[1]['error'] == 'Test error'
        assert 'error' not in results[2]
    
    def test_explain_suspicious_nodes(self):
        """Test automatic suspicious node detection."""
        # Mock model to return predictable outputs
        with patch.object(self.model, 'forward') as mock_forward:
            # Create logits that result in different fraud probabilities
            logits = torch.tensor([
                [0.0, 2.0],  # High fraud probability (node 0)
                [1.0, 0.0],  # Low fraud probability (node 1)
                [0.0, 1.5],  # Medium fraud probability (node 2)
                [2.0, 0.0],  # Very low fraud probability (node 3)
            ])
            mock_forward.return_value = logits
            
            with patch.object(self.pipeline, 'explain_nodes') as mock_explain_nodes:
                mock_explain_nodes.return_value = [
                    {'node_id': 0, 'prediction': 0.88},
                    {'node_id': 2, 'prediction': 0.82}
                ]
                
                results = self.pipeline.explain_suspicious_nodes(
                    self.data, threshold=0.6, max_nodes=10
                )
                
                # Should have called explain_nodes with suspicious nodes
                mock_explain_nodes.assert_called_once()
                call_args = mock_explain_nodes.call_args[0]
                suspicious_nodes = call_args[1]
                
                # Node 0 and 2 should be identified as suspicious
                assert 0 in suspicious_nodes
                assert 2 in suspicious_nodes
                assert 1 not in suspicious_nodes  # Below threshold
                assert 3 not in suspicious_nodes  # Below threshold


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_generate_explanation_text_high_risk(self):
        """Test explanation text generation for high risk."""
        top_features = [
            ('transaction_amount', 0.85),
            ('num_connections', -0.72),
            ('account_age', 0.48)
        ]
        
        text = _generate_explanation_text(
            node_id=123,
            fraud_prob=0.85,
            top_features=top_features,
            significant_edges=5,
            explainer_type='gnn_explainer'
        )
        
        assert 'Node 123' in text
        assert 'high fraud risk' in text
        assert '85.0%' in text
        assert 'transaction_amount' in text
        assert 'num_connections' in text
        assert '5 significant connections' in text
        assert 'gnn explainer' in text
    
    def test_generate_explanation_text_low_risk(self):
        """Test explanation text generation for low risk."""
        text = _generate_explanation_text(
            node_id=456,
            fraud_prob=0.25,
            top_features=[],
            significant_edges=1,
            explainer_type='pg_explainer'
        )
        
        assert 'Node 456' in text
        assert 'low fraud risk' in text
        assert '25.0%' in text
        assert 'pg explainer' in text
    
    def test_get_node_type_with_transaction(self):
        """Test node type detection with transaction nodes."""
        mock_data = Mock()
        mock_data.node_types = ['transaction', 'account', 'merchant']
        
        node_type = _get_node_type(mock_data, 0)
        assert node_type == 'transaction'
    
    def test_get_node_type_without_transaction(self):
        """Test node type detection without transaction nodes."""
        mock_data = Mock()
        mock_data.node_types = ['account', 'merchant']
        
        node_type = _get_node_type(mock_data, 0)
        assert node_type == 'account'  # First type
    
    def test_get_node_type_no_types(self):
        """Test node type detection with no node types."""
        mock_data = Mock()
        mock_data.node_types = None
        
        node_type = _get_node_type(mock_data, 0)
        assert node_type == 'default'


@pytest.mark.skipif(not HAS_API, reason="Flask not available")
class TestExplainabilityAPI:
    """Test REST API functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.model = MockModel()
        self.data = MockData()
        self.temp_dir = tempfile.mkdtemp()
        self.config = ExplainabilityConfig(
            output_dir=self.temp_dir,
            visualization=False,
            save_reports=False
        )
        self.api = ExplainabilityAPI(self.model, self.data, self.config, 'cpu')
        self.client = self.api.app.test_client()
    
    def teardown_method(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['model_loaded'] is True
        assert data['data_loaded'] is True
        assert data['pipeline_ready'] is True
    
    def test_get_config(self):
        """Test configuration endpoint."""
        response = self.client.get('/config')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'explainer_type' in data
        assert 'k_hops' in data
        assert data['explainer_type'] == 'gnn_explainer'
    
    def test_update_config(self):
        """Test configuration update endpoint."""
        new_config = {
            'explainer_type': 'pg_explainer',
            'k_hops': 3,
            'top_k_features': 5
        }
        
        response = self.client.post('/config', 
                                  data=json.dumps(new_config),
                                  content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'Configuration updated' in data['status']
        
        # Verify config was updated
        assert self.api.config.explainer_type == 'pg_explainer'
        assert self.api.config.k_hops == 3
        assert self.api.config.top_k_features == 5
    
    @patch('src.explainability.integration.explain_instance')
    def test_explain_node_endpoint(self, mock_explain):
        """Test single node explanation endpoint."""
        mock_explain.return_value = {
            'node_id': 123,
            'prediction': 0.75,
            'explanation_masks': {
                'edge_mask': np.array([0.9, 0.7]),
                'node_feat_mask': np.array([0.8, 0.6])
            },
            'top_features': [('feature_1', 0.8)],
            'explanation_text': 'Test explanation'
        }
        
        request_data = {'node_id': 123}
        
        response = self.client.post('/explain',
                                  data=json.dumps(request_data),
                                  content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['node_id'] == 123
        assert data['prediction'] == 0.75
        assert 'explanation_masks' in data
    
    def test_explain_node_missing_id(self):
        """Test explanation endpoint with missing node ID."""
        response = self.client.post('/explain',
                                  data=json.dumps({}),
                                  content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'node_id is required' in data['error']
    
    @patch('src.explainability.integration.ExplainabilityPipeline.explain_nodes')
    def test_explain_batch_endpoint(self, mock_explain_nodes):
        """Test batch explanation endpoint."""
        mock_explain_nodes.return_value = [
            {'node_id': 1, 'prediction': 0.8},
            {'node_id': 2, 'prediction': 0.3}
        ]
        
        request_data = {'node_ids': [1, 2]}
        
        response = self.client.post('/explain/batch',
                                  data=json.dumps(request_data),
                                  content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'results' in data
        assert len(data['results']) == 2
    
    def test_not_found_endpoint(self):
        """Test 404 handling."""
        response = self.client.get('/nonexistent')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'not found' in data['error'].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
