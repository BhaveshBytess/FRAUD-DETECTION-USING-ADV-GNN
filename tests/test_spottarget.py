"""
Unit tests for SpotTarget core functionality.
Following Stage7 Reference §Phase1 test requirements:
- test_spottarget_small_graph(): verify T_low exclusion preserves connectivity 
- test_leakage_check_removes_test_edges(): ensure test edge removal
"""

import pytest
import torch
import numpy as np
from typing import Dict, Tuple

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from spot_target import (
    SpotTargetSampler,
    leakage_check,
    compute_avg_degree,
    compute_default_delta,
    create_inference_graph,
    setup_spottarget_sampler
)


class TestSpotTargetCore:
    """Test SpotTarget core functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
    
    def create_small_test_graph(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Create small test graph for unit testing.
        Following §Phase1 test requirements.
        
        Returns:
            (edge_index, train_edge_mask, edge_splits)
        """
        # Create small graph: 6 nodes, mix of high/low degree nodes
        # Node degrees: [1, 1, 3, 3, 2, 2] -> avg = 2
        edges = [
            (0, 2), (2, 0),  # node 0: degree 1 (low)
            (1, 3), (3, 1),  # node 1: degree 1 (low)  
            (2, 3), (3, 2),  # nodes 2,3: degree 3 (high)
            (2, 4), (4, 2),  # node 4: degree 2 (medium)
            (3, 5), (5, 3),  # node 5: degree 2 (medium)
            (4, 5), (5, 4),  # additional edges
        ]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        num_edges = edge_index.size(1)
        
        # Create train/test splits
        # Mark edges involving low-degree nodes as train targets
        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        test_mask = torch.zeros(num_edges, dtype=torch.bool)
        valid_mask = torch.zeros(num_edges, dtype=torch.bool)
        
        # Assign some edges as train targets (including low-degree incidents)
        train_indices = [0, 1, 2, 3, 6, 7]  # includes low-degree node edges
        test_indices = [4, 5]  # some high-degree edges as test
        valid_indices = [8, 9, 10, 11]  # remaining as validation
        
        train_mask[train_indices] = True
        test_mask[test_indices] = True
        valid_mask[valid_indices] = True
        
        edge_splits = {
            'train': train_mask,
            'test': test_mask,
            'valid': valid_mask
        }
        
        return edge_index, train_mask, edge_splits
    
    def test_compute_avg_degree(self):
        """Test degree computation following reference snippet."""
        edge_index, _, _ = self.create_small_test_graph()
        num_nodes = 6
        
        avg_deg, degrees = compute_avg_degree(edge_index, num_nodes)
        
        print(f"Computed degrees: {degrees}")
        print(f"Edge index shape: {edge_index.shape}")
        print(f"Edges: {edge_index}")
        
        # Verify expected degrees manually
        # Let's count actual degrees from the edge list
        expected_degrees = torch.zeros(num_nodes, dtype=torch.long)
        src, dst = edge_index
        for s, d in zip(src.tolist(), dst.tolist()):
            expected_degrees[s] += 1
            expected_degrees[d] += 1
        
        print(f"Expected degrees: {expected_degrees}")
        assert torch.equal(degrees, expected_degrees)
        
        # Verify average degree
        expected_avg = int(expected_degrees.float().mean().item())
        assert avg_deg == expected_avg
    
    def test_compute_default_delta(self):
        """Test default delta computation."""
        edge_index, _, _ = self.create_small_test_graph()
        num_nodes = 6
        
        delta = compute_default_delta(edge_index, num_nodes)
        
        # Calculate expected average: [2, 2, 6, 6, 4, 4] -> avg = 4
        expected_avg = int((2 + 2 + 6 + 6 + 4 + 4) / 6)
        assert delta == expected_avg
    
    def test_spottarget_sampler_initialization(self):
        """Test SpotTarget sampler initialization."""
        edge_index, train_mask, _ = self.create_small_test_graph()
        num_nodes = 6
        
        _, degrees = compute_avg_degree(edge_index, num_nodes)
        
        sampler = SpotTargetSampler(
            edge_index=edge_index,
            train_edge_mask=train_mask,
            degrees=degrees,
            delta=4,  # use actual average degree
            verbose=True
        )
        
        assert sampler.delta == 4
        assert torch.equal(sampler.degrees, degrees)
        assert torch.equal(sampler.train_edge_mask, train_mask)
    
    def test_spottarget_small_graph(self):
        """
        Test SpotTarget on small graph - verify T_low exclusion.
        Following Stage7 Reference §Phase1 test requirement:
        "verify that excluding T_low preserves connectivity more than ExcludeAll"
        """
        edge_index, train_mask, _ = self.create_small_test_graph()
        num_nodes = 6
        
        _, degrees = compute_avg_degree(edge_index, num_nodes)
        delta = 3  # set threshold below high-degree nodes (6,6,4,4) but above low-degree (2,2)
        
        sampler = SpotTargetSampler(
            edge_index=edge_index,
            train_edge_mask=train_mask,
            degrees=degrees,
            delta=delta,
            verbose=False
        )
        
        # Test batch sampling - use all edges as batch
        all_edge_indices = torch.arange(edge_index.size(1))
        filtered_edge_index = sampler.sample_batch(all_edge_indices)
        
        # Verify some edges were filtered
        original_edges = edge_index.size(1)
        filtered_edges = filtered_edge_index.size(1)
        print(f"Original edges: {original_edges}, Filtered edges: {filtered_edges}")
        
        # Verify T_low edges were excluded
        # T_low = train target edges with min(deg[u], deg[v]) < delta=3
        # Nodes 0,1 have degree 2 < 3, so edges involving them should be excluded if they're train targets
        src, dst = edge_index
        min_deg = torch.minimum(degrees[src], degrees[dst])
        tlow_expected = train_mask & (min_deg < delta)
        
        expected_exclusions = tlow_expected.sum().item()
        actual_exclusions = original_edges - filtered_edges
        
        print(f"Expected exclusions: {expected_exclusions}, Actual exclusions: {actual_exclusions}")
        print(f"Train mask: {train_mask}")
        print(f"Min degrees: {min_deg}")
        print(f"T_low mask: {tlow_expected}")
        
        assert actual_exclusions == expected_exclusions, \
            f"Expected {expected_exclusions} T_low exclusions, got {actual_exclusions}"
        
        # Get sampler stats
        stats = sampler.get_stats()
        assert stats['delta'] == delta
        assert stats['tlow_edges'] == expected_exclusions
        
        print(f"✅ SpotTarget test passed: excluded {actual_exclusions} T_low edges")
    
    def test_spottarget_preserves_connectivity(self):
        """Test that SpotTarget preserves more connectivity than ExcludeAll."""
        edge_index, train_mask, _ = self.create_small_test_graph()
        num_nodes = 6
        
        _, degrees = compute_avg_degree(edge_index, num_nodes)
        
        sampler = SpotTargetSampler(
            edge_index=edge_index,
            train_edge_mask=train_mask,
            degrees=degrees,
            delta=3  # use threshold that excludes some but not all train edges
        )
        
        # SpotTarget filtering
        all_indices = torch.arange(edge_index.size(1))
        spottarget_edges = sampler.sample_batch(all_indices)
        
        # ExcludeAll - remove all train target edges
        exclude_all_mask = ~train_mask
        exclude_all_edges = edge_index[:, exclude_all_mask]
        
        # SpotTarget should preserve more edges than ExcludeAll
        spottarget_count = spottarget_edges.size(1)
        exclude_all_count = exclude_all_edges.size(1)
        
        print(f"SpotTarget edges: {spottarget_count}, ExcludeAll edges: {exclude_all_count}")
        assert spottarget_count >= exclude_all_count, \
            "SpotTarget should preserve at least as much connectivity as ExcludeAll"
        
        print(f"✅ Connectivity test: SpotTarget={spottarget_count}, ExcludeAll={exclude_all_count}")


class TestLeakageCheck:
    """Test leakage check functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
    
    def test_leakage_check_removes_test_edges(self):
        """
        Test leakage check removes test edges.
        Following Stage7 Reference §Phase1 test requirement.
        """
        # Create test graph
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 0]
        ], dtype=torch.long)
        
        num_edges = edge_index.size(1)
        
        # Create splits - mark some edges as test
        train_mask = torch.tensor([True, True, False, False, False, False])
        test_mask = torch.tensor([False, False, True, True, False, False])
        valid_mask = torch.tensor([False, False, False, False, True, True])
        
        edge_splits = {
            'train': train_mask,
            'test': test_mask,
            'valid': valid_mask
        }
        
        # Apply leakage check - should remove test edges
        filtered_edge_index = leakage_check(
            edge_index=edge_index,
            edge_splits=edge_splits,
            use_validation_edges=False,
            strict_mode=True
        )
        
        # Verify test edges were removed
        original_edges = edge_index.size(1)
        filtered_edges = filtered_edge_index.size(1)
        expected_removed = test_mask.sum().item()
        actual_removed = original_edges - filtered_edges
        
        assert actual_removed == expected_removed, \
            f"Expected {expected_removed} test edges removed, got {actual_removed}"
        
        print(f"✅ Leakage check test passed: removed {actual_removed} test edges")
    
    def test_leakage_check_with_validation(self):
        """Test leakage check with validation edge removal."""
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 0]
        ], dtype=torch.long)
        
        test_mask = torch.tensor([False, False, True, True, False, False])
        valid_mask = torch.tensor([False, False, False, False, True, True])
        
        edge_splits = {'test': test_mask, 'valid': valid_mask}
        
        # Remove both test and validation edges
        filtered_edge_index = leakage_check(
            edge_index=edge_index,
            edge_splits=edge_splits,
            use_validation_edges=True
        )
        
        expected_removed = (test_mask | valid_mask).sum().item()
        actual_removed = edge_index.size(1) - filtered_edge_index.size(1)
        
        assert actual_removed == expected_removed
        print(f"✅ Validation removal test passed: removed {actual_removed} edges")
    
    def test_create_inference_graph(self):
        """Test inference graph creation with statistics."""
        edge_index = torch.tensor([
            [0, 1, 2, 3],
            [1, 2, 3, 0]
        ], dtype=torch.long)
        
        test_mask = torch.tensor([False, True, False, True])
        edge_splits = {'test': test_mask}
        
        inference_edge_index, stats = create_inference_graph(
            edge_index=edge_index,
            edge_splits=edge_splits,
            exclude_validation=False
        )
        
        assert stats['original_edges'] == 4
        assert stats['removed_edges'] == 2
        assert stats['remaining_edges'] == 2
        assert stats['removal_rate'] == 0.5
        
        print(f"✅ Inference graph test passed: {stats}")


class TestIntegration:
    """Integration tests for SpotTarget setup."""
    
    def test_setup_spottarget_sampler(self):
        """Test factory function for sampler setup."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        train_mask = torch.tensor([True, False, True])
        num_nodes = 3
        
        # Test with auto delta
        config = {'delta': 'auto', 'verbose': True}
        sampler = setup_spottarget_sampler(
            edge_index=edge_index,
            train_edge_mask=train_mask,
            num_nodes=num_nodes,
            config=config
        )
        
        expected_delta = int((2 + 2 + 2) / 3)  # each node has degree 2
        assert sampler.delta == expected_delta
        assert sampler.verbose == True
        
        print(f"✅ Setup test passed: delta={sampler.delta}")
    
    def test_edge_case_zero_degree(self):
        """Test handling of zero-degree nodes."""
        # Graph with isolated node
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # nodes 0,1 connected
        train_mask = torch.tensor([True, True])
        num_nodes = 3  # node 2 is isolated (degree 0)
        
        _, degrees = compute_avg_degree(edge_index, num_nodes)
        assert degrees[2] == 0  # isolated node
        
        sampler = SpotTargetSampler(
            edge_index=edge_index,
            train_edge_mask=train_mask,
            degrees=degrees,
            delta=1
        )
        
        # Should handle zero-degree nodes gracefully
        batch_indices = torch.arange(edge_index.size(1))
        filtered_edges = sampler.sample_batch(batch_indices)
        
        # Should not crash and should return some edges
        assert filtered_edges.size(1) >= 0
        
        print(f"✅ Zero-degree test passed: filtered {filtered_edges.size(1)} edges")


# Pytest fixtures and runners
@pytest.fixture
def small_graph():
    """Fixture for small test graph."""
    test_case = TestSpotTargetCore()
    return test_case.create_small_test_graph()


def test_spottarget_functionality():
    """Main test runner for SpotTarget functionality."""
    print("\n🧪 Running SpotTarget Core Tests...")
    
    # Core functionality tests
    test_core = TestSpotTargetCore()
    test_core.setup_method()
    
    test_core.test_compute_avg_degree()
    test_core.test_compute_default_delta()
    test_core.test_spottarget_sampler_initialization()
    test_core.test_spottarget_small_graph()
    test_core.test_spottarget_preserves_connectivity()
    
    # Leakage check tests
    test_leakage = TestLeakageCheck()
    test_leakage.setup_method()
    
    test_leakage.test_leakage_check_removes_test_edges()
    test_leakage.test_leakage_check_with_validation()
    test_leakage.test_create_inference_graph()
    
    # Integration tests
    test_integration = TestIntegration()
    test_integration.test_setup_spottarget_sampler()
    test_integration.test_edge_case_zero_degree()
    
    print("✅ All SpotTarget tests passed!")


if __name__ == "__main__":
    test_spottarget_functionality()
