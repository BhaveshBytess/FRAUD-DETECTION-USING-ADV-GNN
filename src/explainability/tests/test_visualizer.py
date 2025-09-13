"""
Unit tests for Phase C — Visualizer & Report generation.

Tests visualization functions, HTML report generation, and batch processing.
"""

import pytest
import torch
import numpy as np
import networkx as nx
import tempfile
import os
from unittest.mock import patch, Mock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.explainability.visualizer import (
    visualize_subgraph,
    explain_report,
    explain_batch_to_html,
    create_feature_importance_plot,
    _dict_to_networkx,
    _generate_report_html
)


class TestVisualizer:
    """Test visualization functions."""
    
    def setup_method(self):
        """Setup test data."""
        # Create test graph
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])
        
        # Test masks
        self.masks = {
            'edge_mask': torch.tensor([0.9, 0.7, 0.3, 0.8]),
            'node_feat_mask': torch.tensor([0.5, 0.6, 0.4, 0.7])
        }
        
        # Node metadata
        self.node_meta = {
            'labels': {0: 'suspicious', 1: 'normal', 2: 'normal', 3: 'flagged'},
            'features': {0: [1.0, 2.0], 1: [0.5, 1.5], 2: [0.8, 1.2], 3: [1.2, 1.8]}
        }
        
        # Create temp directory for outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dict_to_networkx(self):
        """Test conversion from dictionary to NetworkX graph."""
        graph_dict = {
            'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]])
        }
        
        G = _dict_to_networkx(graph_dict)
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 3
        assert G.has_edge(0, 1)
        assert G.has_edge(1, 2)
        assert G.has_edge(2, 0)
    
    def test_visualize_subgraph_static(self):
        """Test static visualization generation."""
        output_path = os.path.join(self.temp_dir, "test_vis")
        
        results = visualize_subgraph(
            G=self.G,
            masks=self.masks,
            node_meta=self.node_meta,
            target_node=0,
            top_k=3,
            output_path=output_path,
            interactive=False  # Skip interactive for testing
        )
        
        # Check that static visualization was created
        assert 'static' in results
        assert os.path.exists(results['static'])
        assert results['static'].endswith('.png')
    
    def test_visualize_subgraph_with_dict_input(self):
        """Test visualization with dictionary input."""
        graph_dict = {
            'edge_index': torch.tensor([[0, 1, 2, 0], [1, 2, 3, 3]])
        }
        
        output_path = os.path.join(self.temp_dir, "test_dict_vis")
        
        results = visualize_subgraph(
            G=graph_dict,
            masks=self.masks,
            node_meta=self.node_meta,
            target_node=0,
            top_k=2,
            output_path=output_path,
            interactive=False
        )
        
        assert 'static' in results
        assert os.path.exists(results['static'])
    
    @patch('src.explainability.visualizer.HAS_PYVIS', True)
    def test_visualize_subgraph_interactive(self):
        """Test interactive visualization (mocked)."""
        with patch('src.explainability.visualizer.Network') as mock_network:
            mock_net = Mock()
            mock_network.return_value = mock_net
            
            output_path = os.path.join(self.temp_dir, "test_interactive")
            
            results = visualize_subgraph(
                G=self.G,
                masks=self.masks,
                node_meta=self.node_meta,
                target_node=0,
                interactive=True,
                output_path=output_path
            )
            
            # Check that pyvis methods were called
            mock_net.add_node.assert_called()
            mock_net.add_edge.assert_called()
            mock_net.save_graph.assert_called()


class TestReportGeneration:
    """Test HTML report generation."""
    
    def setup_method(self):
        """Setup test data."""
        self.temp_dir = tempfile.mkdtemp()
        
        self.masks = {
            'edge_mask': torch.tensor([0.9, 0.7, 0.3, 0.8]),
            'explanation_type': 'gnn_explainer'
        }
        
        self.top_features = [
            ('transaction_amount', 0.85),
            ('num_connections', 0.72),
            ('time_since_last', -0.65),
            ('account_age', 0.48),
            ('location_risk', 0.91)
        ]
    
    def teardown_method(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_explain_report_generation(self):
        """Test HTML report generation."""
        output_path = os.path.join(self.temp_dir, "test_report.html")
        
        generated_path = explain_report(
            node_id=12345,
            pred_prob=0.87,
            masks=self.masks,
            top_features=self.top_features,
            explanation_text="This transaction is flagged due to high amount and suspicious connections.",
            output_path=output_path
        )
        
        assert generated_path == output_path
        assert os.path.exists(generated_path)
        
        # Check HTML content
        with open(generated_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "Node 12345" in content
        assert "87.0% fraud probability" in content
        assert "transaction_amount" in content
        assert "0.850" in content
        assert "high-risk" in content  # Should be high risk due to >0.5 prob
    
    def test_generate_report_html_content(self):
        """Test HTML content generation."""
        html_content = _generate_report_html(
            node_id=123,
            pred_prob=0.25,  # Low risk
            masks=self.masks,
            top_features=self.top_features,
            explanation_text="Test explanation text."
        )
        
        assert "Node 123" in html_content
        assert "25.0% fraud probability" in html_content
        assert "low-risk" in html_content  # Should be low risk
        assert "Test explanation text." in html_content
        assert "transaction_amount" in html_content
        
        # Check that JSON data is included
        assert '"node_id": 123' in html_content
    
    def test_explain_batch_to_html(self):
        """Test batch report generation."""
        explanations = [
            {
                'node_id': 1,
                'prediction': 0.9,
                'masks': self.masks,
                'top_features': self.top_features[:3],
                'explanation_text': 'High risk transaction 1.'
            },
            {
                'node_id': 2,
                'prediction': 0.3,
                'masks': self.masks,
                'top_features': self.top_features[:2],
                'explanation_text': 'Low risk transaction 2.'
            }
        ]
        
        generated_files = explain_batch_to_html(
            explanations=explanations,
            out_dir=self.temp_dir
        )
        
        assert len(generated_files) == 2
        for file_path in generated_files:
            assert os.path.exists(file_path)
            assert file_path.endswith('.html')
        
        # Check content of first file
        with open(generated_files[0], 'r', encoding='utf-8') as f:
            content = f.read()
        assert "Node 1" in content
        assert "90.0% fraud probability" in content
    
    def test_explain_batch_to_html_with_missing_data(self):
        """Test batch processing with incomplete data."""
        explanations = [
            {
                'node_id': 1,
                # Missing some fields
                'prediction': 0.7
            },
            {
                # Minimal data
                'node_id': 2
            }
        ]
        
        generated_files = explain_batch_to_html(
            explanations=explanations,
            out_dir=self.temp_dir
        )
        
        # Should still generate files with default values
        assert len(generated_files) == 2
        for file_path in generated_files:
            assert os.path.exists(file_path)


class TestFeatureImportancePlot:
    """Test feature importance plotting."""
    
    def setup_method(self):
        """Setup test data."""
        self.temp_dir = tempfile.mkdtemp()
        
        self.top_features = [
            ('feature_1', 0.85),
            ('feature_2', 0.72),
            ('feature_3', -0.65),
            ('feature_4', 0.48),
            ('feature_5', -0.91),
            ('feature_6', 0.33)
        ]
    
    def teardown_method(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_feature_importance_plot(self):
        """Test feature importance plot creation."""
        output_path = os.path.join(self.temp_dir, "feature_plot.png")
        
        generated_path = create_feature_importance_plot(
            top_features=self.top_features,
            output_path=output_path
        )
        
        assert generated_path == output_path
        assert os.path.exists(generated_path)
        assert generated_path.endswith('.png')
    
    def test_create_feature_importance_plot_auto_name(self):
        """Test plot creation with automatic naming."""
        # Change to temp directory for auto-generated files
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            generated_path = create_feature_importance_plot(
                top_features=self.top_features
            )
            
            assert os.path.exists(generated_path)
            assert 'feature_importance_' in generated_path
            assert generated_path.endswith('.png')
        finally:
            os.chdir(original_cwd)
    
    def test_create_feature_importance_plot_empty_features(self):
        """Test plot creation with empty features list."""
        with pytest.raises(ValueError, match="No features provided"):
            create_feature_importance_plot(
                top_features=[],
                output_path=os.path.join(self.temp_dir, "empty.png")
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
