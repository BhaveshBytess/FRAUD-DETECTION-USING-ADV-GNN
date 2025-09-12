# test_phase_a_integration.py - Quick integration test for Phase A
"""
Test Phase A integration with existing data loading per §PHASE_A.5
"""

import sys
import os
sys.path.append('src')

import torch
from sampling.temporal_data_loader import prepare_stage6_data, debug_temporal_data
from sampling.cpu_fallback import sample_time_relaxed_neighbors
from sampling.utils import validate_temporal_constraints

def test_phase_a_integration():
    """Test complete Phase A integration"""
    print("Testing Phase A integration...")
    
    # Prepare Stage 6 data
    hetero_data, temporal_graph, train_loader, val_loader, test_loader = prepare_stage6_data(
        data_path="data/sample/sample_hetero.pt",
        config={'batch_size': 32, 'train_ratio': 0.7, 'val_ratio': 0.15}
    )
    
    # Debug temporal data
    debug_temporal_data(hetero_data, temporal_graph)
    
    # Test temporal sampling with real data
    print("\nTesting temporal sampling on real data...")
    
    # Get a small batch from train loader
    train_batch = next(iter(train_loader))
    seed_nodes, t_eval_batch, labels = train_batch
    
    # Sample only first few nodes for testing
    test_seeds = seed_nodes[:2]
    # Use larger t_eval values that are compatible with edge timestamps (1-6)
    test_t_eval = torch.tensor([6.0, 5.0])  # Use timestamps that can see the edges
    
    print(f"Test seeds: {test_seeds}")
    print(f"Test t_eval: {test_t_eval}")
    
    # Perform time-relaxed sampling
    sampled_subgraph = sample_time_relaxed_neighbors(
        node_ids=test_seeds,
        t_eval=test_t_eval,
        depth=2,
        fanouts=[5, 3],
        delta_t=10.0,  # larger window to include edges
        temporal_graph=temporal_graph,
        strategy='recency'
    )
    
    print(f"Sampled subgraph: {sampled_subgraph.num_nodes} nodes, {sampled_subgraph.num_edges} edges")
    
    # Validate temporal constraints
    validation_results = validate_temporal_constraints(
        temporal_graph=temporal_graph,
        seed_nodes=test_seeds,
        t_eval=test_t_eval,
        delta_t=10.0,  # same large window
        sampled_subgraph=sampled_subgraph
    )
    
    print(f"Validation results: {validation_results}")
    
    # Check that all validations pass
    all_passed = all(validation_results.values())
    print(f"All temporal constraints satisfied: {all_passed}")
    
    if all_passed:
        print("✓ Phase A integration test PASSED")
        return True
    else:
        print("✗ Phase A integration test FAILED")
        return False

if __name__ == "__main__":
    success = test_phase_a_integration()
    if success:
        print("\nPhase A implementation complete and validated! Ready for Phase B.")
    else:
        print("\nPhase A integration failed. Please check implementation.")
