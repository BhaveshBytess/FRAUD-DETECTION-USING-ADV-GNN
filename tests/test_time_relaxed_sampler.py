# tests/test_time_relaxed_sampler.py
"""
Unit tests for time-relaxed sampling per §PHASE_A.5 - MANDATORY per APPENDIX
Tests must run in CI in lite-mode and be fast (<30s)
"""

import torch
import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sampling.cpu_fallback import (
    sample_time_relaxed_neighbors, 
    TemporalGraph,
    get_neighbors_in_time_range,
    sample_candidates
)
from sampling.utils import (
    build_temporal_adjacency,
    validate_temporal_constraints,
    create_temporal_masks
)

class TestTimeRelaxedSampler:
    """Test suite for time-relaxed neighbor sampling per §PHASE_A.5"""
    
    def setup_method(self):
        """Create synthetic temporal graph for testing"""
        # Create simple temporal graph:
        # Node 0 -> [1 (t=10), 2 (t=8), 3 (t=5)]
        # Node 1 -> [2 (t=9), 3 (t=7)]  
        # Node 2 -> [3 (t=6)]
        # Node 3 -> []
        
        self.num_nodes = 4
        self.indptr = torch.tensor([0, 3, 5, 6, 6])  # pointers
        self.indices = torch.tensor([1, 2, 3, 2, 3, 3])  # neighbors
        self.timestamps = torch.tensor([10.0, 8.0, 5.0, 9.0, 7.0, 6.0])  # descending per node
        
        self.temporal_graph = TemporalGraph(
            indptr=self.indptr,
            indices=self.indices, 
            timestamps=self.timestamps,
            num_nodes=self.num_nodes,
            num_edges=6
        )
    
    def test_time_window_filtering(self):
        """Test time_window_filtering per §PHASE_A.5: all sampled edges within time window"""
        seed_nodes = torch.tensor([0])
        t_eval = torch.tensor([10.0])  # evaluation at t=10
        delta_t = 3.0  # time window: [7, 10]
        
        # Sample with 1-hop
        result = sample_time_relaxed_neighbors(
            node_ids=seed_nodes,
            t_eval=t_eval,
            depth=1,
            fanouts=[2],  # sample 2 neighbors max
            delta_t=delta_t,
            temporal_graph=self.temporal_graph
        )
        
        # Validate time constraints
        validation = validate_temporal_constraints(
            temporal_graph=self.temporal_graph,
            seed_nodes=seed_nodes,
            t_eval=t_eval,
            delta_t=delta_t,
            sampled_subgraph=result
        )
        
        assert validation['time_window_filtering'], "Time window filtering failed"
        
        # Check specific edges - should include nodes 1 (t=10) and 2 (t=8) but exclude 3 (t=5)
        sampled_node_ids = list(result.node_mapping.keys())
        assert 0 in sampled_node_ids, "Seed node should be included"
        assert 1 in sampled_node_ids, "Node 1 (t=10) should be included"
        assert 2 in sampled_node_ids, "Node 2 (t=8) should be included"
        # Node 3 (t=5) should be excluded as it's outside [7, 10] window
    
    def test_no_leakage(self):
        """Test no_leakage per §PHASE_A.5: training mask edges don't leak future info"""
        seed_nodes = torch.tensor([0])
        t_eval = torch.tensor([8.0])  # evaluate at t=8 (earlier than some edges)
        delta_t = 2.0  # time window: [6, 8]
        
        result = sample_time_relaxed_neighbors(
            node_ids=seed_nodes,
            t_eval=t_eval,
            depth=1,
            fanouts=[5],  # sample more to test filtering
            delta_t=delta_t,
            temporal_graph=self.temporal_graph
        )
        
        validation = validate_temporal_constraints(
            temporal_graph=self.temporal_graph,
            seed_nodes=seed_nodes,
            t_eval=t_eval,
            delta_t=delta_t,
            sampled_subgraph=result
        )
        
        assert validation['no_leakage'], "Future information leakage detected"
        
        # Should exclude node 1 (t=10 > 8) but include node 2 (t=8)
        sampled_node_ids = list(result.node_mapping.keys())
        assert 2 in sampled_node_ids, "Node 2 (t=8) should be included"
        # Node 1 should be excluded due to future timestamp
    
    def test_frontier_size(self):
        """Test frontier_size per §PHASE_A.5: frontier growth within bounds"""
        seed_nodes = torch.tensor([0, 1])  # multiple seeds
        t_eval = torch.tensor([10.0, 10.0])
        delta_t = 10.0  # large window to include all
        fanouts = [2, 1]  # 2-hop sampling
        
        result = sample_time_relaxed_neighbors(
            node_ids=seed_nodes,
            t_eval=t_eval,
            depth=2,
            fanouts=fanouts,
            delta_t=delta_t,
            temporal_graph=self.temporal_graph
        )
        
        validation = validate_temporal_constraints(
            temporal_graph=self.temporal_graph,
            seed_nodes=seed_nodes,
            t_eval=t_eval,
            delta_t=delta_t,
            sampled_subgraph=result
        )
        
        assert validation['frontier_size'], "Frontier size exceeded bounds"
        
        # Total sampled nodes should be reasonable
        assert result.num_nodes <= 10, f"Too many nodes sampled: {result.num_nodes}"
    
    def test_get_neighbors_in_time_range(self):
        """Test time range extraction utility"""
        # Test data: neighbors [1, 2, 3] with timestamps [10, 8, 5] (descending)
        neighbors = torch.tensor([1, 2, 3])
        timestamps = torch.tensor([10.0, 8.0, 5.0])
        
        # Test case 1: window [7, 10] should return [1, 2]
        result1 = get_neighbors_in_time_range(neighbors, timestamps, t_end=10.0, t_start=7.0)
        expected1 = torch.tensor([1, 2])
        assert torch.equal(result1, expected1), f"Expected {expected1}, got {result1}"
        
        # Test case 2: window [8, 8] should return [2] only
        result2 = get_neighbors_in_time_range(neighbors, timestamps, t_end=8.0, t_start=8.0)
        expected2 = torch.tensor([2])
        assert torch.equal(result2, expected2), f"Expected {expected2}, got {result2}"
        
        # Test case 3: window [3, 4] should return empty
        result3 = get_neighbors_in_time_range(neighbors, timestamps, t_end=4.0, t_start=3.0)
        assert len(result3) == 0, f"Expected empty, got {result3}"
    
    def test_sample_candidates(self):
        """Test candidate sampling strategies"""
        candidates = torch.tensor([1, 2, 3, 4, 5])
        
        # Test recency strategy (should take first k)
        result_recency = sample_candidates(candidates, k=3, strategy='recency')
        expected_recency = torch.tensor([1, 2, 3])
        assert torch.equal(result_recency, expected_recency), "Recency sampling failed"
        
        # Test random strategy (should return k elements)
        torch.manual_seed(42)  # for reproducibility
        result_random = sample_candidates(candidates, k=3, strategy='random')
        assert len(result_random) == 3, "Random sampling wrong size"
        assert all(x in candidates for x in result_random), "Random sampling invalid elements"
        
        # Test k >= candidates length (should return all)
        result_all = sample_candidates(candidates, k=10, strategy='recency')
        assert torch.equal(result_all, candidates), "Should return all candidates when k >= length"
    
    def test_monotonic_timestamps(self):
        """Test that timestamps are properly ordered per node"""
        # Validate our test temporal graph
        validation = validate_temporal_constraints(
            temporal_graph=self.temporal_graph,
            seed_nodes=torch.tensor([0]),
            t_eval=torch.tensor([10.0]),
            delta_t=5.0,
            sampled_subgraph=sample_time_relaxed_neighbors(
                node_ids=torch.tensor([0]),
                t_eval=torch.tensor([10.0]),
                depth=1,
                fanouts=[5],
                delta_t=5.0,
                temporal_graph=self.temporal_graph
            )
        )
        
        assert validation['monotonic_timestamps'], "Timestamps not properly ordered"
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        seed_nodes = torch.tensor([0])
        t_eval = torch.tensor([10.0])
        
        # Test empty result (very restrictive time window)
        result_empty = sample_time_relaxed_neighbors(
            node_ids=seed_nodes,
            t_eval=torch.tensor([1.0]),  # very early time
            depth=1,
            fanouts=[5],
            delta_t=0.1,  # tiny window
            temporal_graph=self.temporal_graph
        )
        
        # Should still include seed node
        assert result_empty.num_nodes >= 1, "Should include at least seed nodes"
        assert 0 in result_empty.node_mapping, "Seed node missing from mapping"
        
        # Test zero delta_t (strict temporal constraint)
        result_strict = sample_time_relaxed_neighbors(
            node_ids=seed_nodes,
            t_eval=torch.tensor([10.0]),
            depth=1,
            fanouts=[5],
            delta_t=0.0,  # no relaxation
            temporal_graph=self.temporal_graph
        )
        
        # Should only include edges exactly at t=10
        validation_strict = validate_temporal_constraints(
            temporal_graph=self.temporal_graph,
            seed_nodes=seed_nodes,
            t_eval=torch.tensor([10.0]),
            delta_t=0.0,
            sampled_subgraph=result_strict
        )
        assert validation_strict['time_window_filtering'], "Strict temporal filtering failed"

def test_temporal_graph_validation():
    """Test TemporalGraph validation per §PHASE_A.1"""
    # Valid temporal graph
    valid_graph = TemporalGraph(
        indptr=torch.tensor([0, 2, 3]),
        indices=torch.tensor([1, 2, 0]),
        timestamps=torch.tensor([10.0, 5.0, 8.0]),
        num_nodes=2,
        num_edges=3
    )
    
    # Should not raise exception
    assert valid_graph.num_nodes == 2
    assert valid_graph.num_edges == 3

if __name__ == "__main__":
    # Run tests directly for development
    test_suite = TestTimeRelaxedSampler()
    test_suite.setup_method()
    
    print("Running time-relaxed sampler tests...")
    test_suite.test_time_window_filtering()
    print("✓ time_window_filtering test passed")
    
    test_suite.test_no_leakage()
    print("✓ no_leakage test passed")
    
    test_suite.test_frontier_size()
    print("✓ frontier_size test passed")
    
    test_suite.test_get_neighbors_in_time_range()
    print("✓ get_neighbors_in_time_range test passed")
    
    test_suite.test_sample_candidates()
    print("✓ sample_candidates test passed")
    
    test_suite.test_monotonic_timestamps()
    print("✓ monotonic_timestamps test passed")
    
    test_suite.test_edge_cases()
    print("✓ edge_cases test passed")
    
    test_temporal_graph_validation()
    print("✓ temporal_graph_validation test passed")
    
    print("All Phase A tests passed! ✓")
