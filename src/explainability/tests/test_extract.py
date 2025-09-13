"""
Unit tests for Phase A — Subgraph extraction utilities.

Following Stage 10 Reference §Phase A validation requirements:
- extracted subgraph has target node present
- node count ≤ max_nodes
- deterministic for same seed
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data, HeteroData

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.explainability.extract_subgraph import (
    extract_khop_subgraph,
    extract_hetero_subgraph,
    SubgraphExtractor
)


class TestExtractKhopSubgraph:
    """Test homogeneous k-hop subgraph extraction."""
    
    def setup_method(self):
        """Create test graph."""
        # Create a simple graph: 0-1-2-3-4 (chain) + 5-6 (isolated)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 5, 6],
            [1, 0, 2, 1, 3, 2, 4, 3, 6, 5]
        ])
        self.edge_index = edge_index
        self.num_nodes = 7
    
    def test_target_node_present(self):
        """Test that target node is always in extracted subgraph."""
        target_node = 2
        
        subset, sub_edge_index, mapping, edge_mask = extract_khop_subgraph(
            node_id=target_node,
            num_hops=2,
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            seed=42
        )
        
        # Target node should be in subset
        assert target_node in subset, f"Target node {target_node} not in subset {subset}"
        
        # If relabeled, target should map to a valid index
        if len(mapping) > 0 and len(mapping) > target_node:
            target_new_id = mapping[target_node]
            # Only check if mapping is valid (not -1)
            if target_new_id != -1:
                assert 0 <= target_new_id < len(subset), f"Invalid mapping for target node"
    
    def test_max_nodes_constraint(self):
        """Test that subgraph respects max_nodes limit."""
        target_node = 2
        max_nodes = 3
        
        subset, sub_edge_index, mapping, edge_mask = extract_khop_subgraph(
            node_id=target_node,
            num_hops=5,  # Large hop count to trigger sampling
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            max_nodes=max_nodes,
            seed=42
        )
        
        assert len(subset) <= max_nodes, f"Subgraph has {len(subset)} nodes, exceeds max {max_nodes}"
        assert target_node in subset, "Target node must be preserved even with sampling"
    
    def test_deterministic_extraction(self):
        """Test that extraction is deterministic with same seed."""
        target_node = 1
        num_hops = 2
        seed = 123
        
        # Extract twice with same seed
        result1 = extract_khop_subgraph(
            node_id=target_node,
            num_hops=num_hops,
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            seed=seed
        )
        
        result2 = extract_khop_subgraph(
            node_id=target_node,
            num_hops=num_hops,
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            seed=seed
        )
        
        # Results should be identical
        assert torch.equal(result1[0], result2[0]), "Subsets should be identical"
        assert torch.equal(result1[1], result2[1]), "Edge indices should be identical"
        assert torch.equal(result1[2], result2[2]), "Mappings should be identical"
        assert torch.equal(result1[3], result2[3]), "Edge masks should be identical"
    
    def test_different_seeds_different_results(self):
        """Test that different seeds can produce different results when sampling."""
        target_node = 2
        max_nodes = 2  # Force sampling
        
        result1 = extract_khop_subgraph(
            node_id=target_node,
            num_hops=3,
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            max_nodes=max_nodes,
            seed=1
        )
        
        result2 = extract_khop_subgraph(
            node_id=target_node,
            num_hops=3,
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            max_nodes=max_nodes,
            seed=2
        )
        
        # Both should have target node but may have different other nodes
        assert target_node in result1[0] and target_node in result2[0]
        assert len(result1[0]) <= max_nodes and len(result2[0]) <= max_nodes
    
    def test_edge_consistency(self):
        """Test that extracted edges are consistent with node subset."""
        target_node = 1
        
        subset, sub_edge_index, mapping, edge_mask = extract_khop_subgraph(
            node_id=target_node,
            num_hops=2,
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            relabel_nodes=True,
            seed=42
        )
        
        # All edge endpoints should be valid indices in relabeled graph
        assert sub_edge_index.min() >= 0, "Edge indices should be non-negative"
        assert sub_edge_index.max() < len(subset), "Edge indices should be within subset range"
        
        # Original edges should correspond to subset nodes
        original_edges = self.edge_index[:, edge_mask]
        for i in range(original_edges.size(1)):
            src, dst = original_edges[0, i], original_edges[1, i]
            assert src in subset and dst in subset, f"Edge ({src}, {dst}) nodes not in subset"


class TestExtractHeteroSubgraph:
    """Test heterogeneous subgraph extraction."""
    
    def setup_method(self):
        """Create test heterogeneous graph."""
        self.hetero_data = HeteroData()
        
        # Node types: user, item
        self.hetero_data['user'].num_nodes = 4
        self.hetero_data['item'].num_nodes = 3
        
        # Edge types
        self.hetero_data['user', 'buys', 'item'].edge_index = torch.tensor([
            [0, 1, 2, 3, 0],
            [0, 1, 1, 2, 2]
        ])
        
        self.hetero_data['item', 'bought_by', 'user'].edge_index = torch.tensor([
            [0, 1, 1, 2, 2],
            [0, 1, 2, 3, 0]
        ])
    
    def test_hetero_target_node_present(self):
        """Test target node is present in heterogeneous extraction."""
        target_node = 1
        target_type = 'user'
        
        subset_dict, sub_edge_index_dict, mapping_dict = extract_hetero_subgraph(
            node_id=target_node,
            node_type=target_type,
            num_hops=2,
            hetero_data=self.hetero_data,
            seed=42
        )
        
        assert target_type in subset_dict, f"Target type {target_type} not in subset_dict"
        assert target_node in subset_dict[target_type], f"Target node {target_node} not in subset"
    
    def test_hetero_max_nodes(self):
        """Test max nodes constraint for heterogeneous graphs."""
        target_node = 0
        target_type = 'user'
        max_nodes = 2
        
        subset_dict, sub_edge_index_dict, mapping_dict = extract_hetero_subgraph(
            node_id=target_node,
            node_type=target_type,
            num_hops=3,
            hetero_data=self.hetero_data,
            max_nodes=max_nodes,
            seed=42
        )
        
        for ntype, subset in subset_dict.items():
            assert len(subset) <= max_nodes, f"Node type {ntype} has {len(subset)} nodes, exceeds max {max_nodes}"
        
        # Target should still be present
        assert target_node in subset_dict[target_type], "Target node should be preserved"


class TestSubgraphExtractor:
    """Test unified SubgraphExtractor class."""
    
    def setup_method(self):
        """Setup test data."""
        # Homogeneous graph
        self.homo_data = Data()
        self.homo_data.edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        self.homo_data.x = torch.randn(3, 5)
        self.homo_data.num_nodes = 3
        
        # Heterogeneous graph (simplified)
        self.hetero_data = HeteroData()
        self.hetero_data['node'].x = torch.randn(3, 5)
        self.hetero_data['node'].num_nodes = 3
        self.hetero_data['node', 'edge', 'node'].edge_index = torch.tensor([[0, 1], [1, 2]])
    
    def test_extractor_homo(self):
        """Test extractor with homogeneous graph."""
        extractor = SubgraphExtractor(max_nodes=10, seed=42)
        
        result = extractor.extract(
            graph_data=self.homo_data,
            node_id=1,
            num_hops=1
        )
        
        assert result['type'] == 'homo'
        assert 'subset' in result
        assert 'edge_index' in result
        assert result['target_node'] == 1
        assert 1 in result['subset']
    
    def test_extractor_hetero(self):
        """Test extractor with heterogeneous graph."""
        extractor = SubgraphExtractor(max_nodes=10, seed=42)
        
        result = extractor.extract(
            graph_data=self.hetero_data,
            node_id=1,
            node_type='node',
            num_hops=1
        )
        
        assert result['type'] == 'hetero'
        assert 'subset_dict' in result
        assert 'edge_index_dict' in result
        assert result['target_node'] == 1
        assert result['target_type'] == 'node'
    
    def test_get_subgraph_data_homo(self):
        """Test creating subgraph data object for homogeneous graph."""
        extractor = SubgraphExtractor(max_nodes=10, seed=42)
        
        result = extractor.extract(
            graph_data=self.homo_data,
            node_id=1,
            num_hops=1
        )
        
        sub_data = extractor.get_subgraph_data(self.homo_data, result)
        
        assert isinstance(sub_data, Data)
        assert hasattr(sub_data, 'edge_index')
        assert hasattr(sub_data, 'x')
        assert sub_data.num_nodes == len(result['subset'])
    
    def test_extractor_validation_error(self):
        """Test that extractor raises error for missing node_type in hetero graph."""
        extractor = SubgraphExtractor()
        
        with pytest.raises(ValueError, match="node_type required"):
            extractor.extract(
                graph_data=self.hetero_data,
                node_id=1,
                num_hops=1
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
