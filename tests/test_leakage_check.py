"""
Dedicated tests for leakage check functionality.
Following Stage7 Reference §Phase1: test_leakage_check_removes_test_edges
"""

import pytest
import torch
import numpy as np
from typing import Dict

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from spot_target import leakage_check, create_inference_graph


class TestLeakageCheckDetailed:
    """Detailed tests for leakage check functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_no_test_edges_provided(self):
        """Test leakage check when no test edges are in splits."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        edge_splits = {'train': torch.tensor([True, False, True])}
        
        # Should handle gracefully when no test edges
        filtered_edge_index = leakage_check(
            edge_index=edge_index,
            edge_splits=edge_splits,
            strict_mode=False  # Don't raise warning
        )
        
        # Should return original edge index
        assert torch.equal(filtered_edge_index, edge_index)
    
    def test_empty_edge_splits(self):
        """Test leakage check with empty edge splits."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_splits = {}
        
        filtered_edge_index = leakage_check(
            edge_index=edge_index,
            edge_splits=edge_splits,
            strict_mode=False
        )
        
        assert torch.equal(filtered_edge_index, edge_index)
    
    def test_all_edges_are_test(self):
        """Test edge case where all edges are test edges."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        edge_splits = {
            'test': torch.tensor([True, True, True])
        }
        
        filtered_edge_index = leakage_check(
            edge_index=edge_index,
            edge_splits=edge_splits
        )
        
        # Should remove all edges
        assert filtered_edge_index.size(1) == 0
    
    def test_mixed_edge_types(self):
        """Test leakage check with mixed train/test/valid edges."""
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 0]
        ], dtype=torch.long)
        
        edge_splits = {
            'train': torch.tensor([True, True, False, False, False, False, False, False]),
            'test': torch.tensor([False, False, True, True, False, False, False, False]),
            'valid': torch.tensor([False, False, False, False, True, True, False, False]),
            'other': torch.tensor([False, False, False, False, False, False, True, True])
        }
        
        # Remove only test edges
        filtered_edge_index = leakage_check(
            edge_index=edge_index,
            edge_splits=edge_splits,
            use_validation_edges=False
        )
        
        expected_remaining = 8 - 2  # remove 2 test edges
        assert filtered_edge_index.size(1) == expected_remaining
        
        # Remove test and validation edges
        filtered_edge_index_val = leakage_check(
            edge_index=edge_index,
            edge_splits=edge_splits,
            use_validation_edges=True
        )
        
        expected_remaining_val = 8 - 2 - 2  # remove 2 test + 2 valid
        assert filtered_edge_index_val.size(1) == expected_remaining_val
    
    def test_device_consistency(self):
        """Test leakage check maintains device consistency."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long).to(device)
        edge_splits = {
            'test': torch.tensor([False, True, False]).to(device)
        }
        
        filtered_edge_index = leakage_check(
            edge_index=edge_index,
            edge_splits=edge_splits
        )
        
        assert filtered_edge_index.device == device
        assert filtered_edge_index.size(1) == 2  # removed 1 test edge
    
    def test_inference_graph_creation_comprehensive(self):
        """Comprehensive test of inference graph creation."""
        # Create larger test graph
        num_edges = 10
        edge_index = torch.stack([
            torch.arange(num_edges),
            torch.roll(torch.arange(num_edges), 1)
        ])
        
        # Create realistic splits (60% train, 20% valid, 20% test)
        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        valid_mask = torch.zeros(num_edges, dtype=torch.bool)
        test_mask = torch.zeros(num_edges, dtype=torch.bool)
        
        train_mask[:6] = True
        valid_mask[6:8] = True
        test_mask[8:] = True
        
        edge_splits = {
            'train': train_mask,
            'valid': valid_mask,
            'test': test_mask
        }
        
        # Test without validation removal
        inference_edge_index, stats = create_inference_graph(
            edge_index=edge_index,
            edge_splits=edge_splits,
            exclude_validation=False
        )
        
        assert stats['original_edges'] == 10
        assert stats['removed_edges'] == 2  # only test edges
        assert stats['remaining_edges'] == 8
        assert stats['removal_rate'] == 0.2
        
        # Test with validation removal
        inference_edge_index_val, stats_val = create_inference_graph(
            edge_index=edge_index,
            edge_splits=edge_splits,
            exclude_validation=True
        )
        
        assert stats_val['removed_edges'] == 4  # test + valid edges
        assert stats_val['remaining_edges'] == 6
        assert stats_val['removal_rate'] == 0.4
    
    def test_p3_leakage_prevention(self):
        """
        Test P3 (test-time leakage) prevention.
        Following Stage7 Reference: SpotTarget prevents fake performance gains.
        """
        # Simulate scenario where test edges would boost performance
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4],  # sources
            [1, 2, 3, 4, 0]   # targets
        ], dtype=torch.long)
        
        # Mark some edges that would create shortcuts to test targets
        edge_splits = {
            'train': torch.tensor([True, True, False, False, False]),
            'test': torch.tensor([False, False, True, True, True])  # 3 test edges
        }
        
        # Without leakage check - would include test edges (P3 leakage)
        original_edges = edge_index.size(1)
        
        # With leakage check - removes test edges
        safe_edge_index = leakage_check(
            edge_index=edge_index,
            edge_splits=edge_splits
        )
        
        # Verify test edges were removed to prevent P3 leakage
        removed_edges = original_edges - safe_edge_index.size(1)
        expected_test_edges = edge_splits['test'].sum().item()
        
        assert removed_edges == expected_test_edges, \
            f"P3 prevention failed: expected {expected_test_edges} removed, got {removed_edges}"
        
        print(f"✅ P3 leakage prevention: removed {removed_edges} test edges")
    
    def test_leakage_detection_strict_mode(self):
        """Test strict mode leakage detection."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        
        # No test edges - should trigger warning in strict mode
        edge_splits = {'train': torch.tensor([True, True])}
        
        # Should not raise exception but might log warning
        filtered_edge_index = leakage_check(
            edge_index=edge_index,
            edge_splits=edge_splits,
            strict_mode=True
        )
        
        assert torch.equal(filtered_edge_index, edge_index)


def test_leakage_check_comprehensive():
    """Main test runner for leakage check functionality."""
    print("\n🧪 Running Leakage Check Tests...")
    
    test_leakage = TestLeakageCheckDetailed()
    test_leakage.setup_method()
    
    # Run all leakage check tests
    test_leakage.test_no_test_edges_provided()
    test_leakage.test_empty_edge_splits()
    test_leakage.test_all_edges_are_test()
    test_leakage.test_mixed_edge_types()
    test_leakage.test_device_consistency()
    test_leakage.test_inference_graph_creation_comprehensive()
    test_leakage.test_p3_leakage_prevention()
    test_leakage.test_leakage_detection_strict_mode()
    
    print("✅ All leakage check tests passed!")


if __name__ == "__main__":
    test_leakage_check_comprehensive()
