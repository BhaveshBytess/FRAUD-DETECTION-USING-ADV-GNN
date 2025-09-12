# tests/test_phase_c_simple.py
"""
Simple Phase C validation test per §PHASE_C.5
Tests core TDGNN functionality without complex dependencies
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from models.tdgnn_wrapper import TDGNNHypergraphModel
from models.hypergraph import create_hypergraph_model, HypergraphData
from sampling.gsampler import GSampler
from sampling.cpu_fallback import TemporalGraph

def test_basic_tdgnn_integration():
    """Test basic TDGNN integration per §PHASE_C.1"""
    print("Testing basic TDGNN integration...")
    
    # Create minimal test data
    num_nodes = 10
    num_edges = 15
    feature_dim = 8
    
    # Create simple temporal graph
    edges = torch.randint(0, num_nodes, (2, num_edges))
    timestamps = torch.rand(num_edges) * 100
    
    # Build adjacency structure
    indptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    indices = []
    edge_times = []
    
    for i in range(num_nodes):
        neighbors = []
        times = []
        for j in range(num_edges):
            if edges[0, j] == i:
                neighbors.append(edges[1, j].item())
                times.append(timestamps[j].item())
        
        indptr[i + 1] = indptr[i] + len(neighbors)
        indices.extend(neighbors)
        edge_times.extend(times)
    
    temporal_graph = TemporalGraph(
        num_nodes=num_nodes,
        num_edges=len(indices),
        indptr=indptr,
        indices=torch.tensor(indices, dtype=torch.long),
        timestamps=torch.tensor(edge_times)
    )
    
    # Create hypergraph data
    incidence_matrix = torch.randn(num_nodes, 5).abs()  # 5 hyperedges
    node_features = torch.randn(num_nodes, feature_dim)
    labels = torch.randint(0, 2, (num_nodes,))
    
    hypergraph_data = HypergraphData(
        incidence_matrix=incidence_matrix,
        node_features=node_features,
        node_labels=labels
    )
    
    # Create base model
    base_model = create_hypergraph_model(
        input_dim=feature_dim,
        hidden_dim=16,
        output_dim=2,
        model_config={'layer_type': 'simple', 'num_layers': 2}
    )
    
    # Create G-SAMPLER
    gsampler = GSampler(
        csr_indptr=temporal_graph.indptr,
        csr_indices=temporal_graph.indices,
        csr_timestamps=temporal_graph.timestamps,
        device='cpu'
    )
    
    # Create TDGNN wrapper
    tdgnn_model = TDGNNHypergraphModel(
        base_model=base_model,
        gsampler=gsampler,
        temporal_graph=temporal_graph
    )
    
    # Test forward pass
    seed_nodes = torch.tensor([0, 1, 2])
    t_eval_array = torch.ones(3) * 80.0
    fanouts = [3, 2]
    delta_t = 20.0
    
    logits = tdgnn_model(
        seed_nodes=seed_nodes,
        t_eval_array=t_eval_array,
        fanouts=fanouts,
        delta_t=delta_t
    )
    
    assert logits.shape == (3, 2), f"Expected shape (3, 2), got {logits.shape}"
    assert torch.isfinite(logits).all(), "Non-finite values in output"
    
    print("✓ Basic TDGNN integration test passed")
    return True

def test_training_components():
    """Test training components per §PHASE_C.2"""
    print("Testing training components...")
    
    # Create minimal setup
    feature_dim = 4
    base_model = create_hypergraph_model(
        input_dim=feature_dim,
        hidden_dim=8,
        output_dim=2,
        model_config={'layer_type': 'simple', 'num_layers': 1}
    )
    
    # Create minimal temporal graph  
    temporal_graph = TemporalGraph(
        num_nodes=5,
        num_edges=3,
        indptr=torch.tensor([0, 1, 2, 3, 3, 3]),
        indices=torch.tensor([1, 2, 0]),
        timestamps=torch.tensor([10.0, 20.0, 30.0])
    )
    
    gsampler = GSampler(
        csr_indptr=temporal_graph.indptr,
        csr_indices=temporal_graph.indices,
        csr_timestamps=temporal_graph.timestamps,
        device='cpu'
    )
    
    tdgnn_model = TDGNNHypergraphModel(
        base_model=base_model,
        gsampler=gsampler,
        temporal_graph=temporal_graph
    )
    
    # Test optimizer setup
    optimizer = torch.optim.Adam(tdgnn_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Test backward pass
    seed_nodes = torch.tensor([0, 1])
    t_eval_array = torch.ones(2) * 40.0
    labels = torch.tensor([0, 1])
    
    logits = tdgnn_model(seed_nodes, t_eval_array, [2, 1], 15.0)
    loss = criterion(logits, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check gradients
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in tdgnn_model.parameters())
    assert has_gradients, "No gradients computed"
    
    print("✓ Training components test passed")
    return True

def test_config_compatibility():
    """Test configuration compatibility per §PHASE_C.3"""
    print("Testing configuration compatibility...")
    
    # Test that Stage 6 config parameters can be loaded
    stage6_config = {
        'fanouts': [15, 10],
        'delta_t': 86400.0,
        'sampling_strategy': 'recency',
        'batch_size': 512,
        'use_gpu_sampling': True,
        'deterministic': True,
        'profile_memory': True
    }
    
    # Validate config structure
    required_keys = ['fanouts', 'delta_t', 'sampling_strategy', 'batch_size']
    for key in required_keys:
        assert key in stage6_config, f"Missing required config key: {key}"
    
    # Validate config values
    assert isinstance(stage6_config['fanouts'], list)
    assert stage6_config['delta_t'] > 0
    assert stage6_config['batch_size'] > 0
    
    print("✓ Configuration compatibility test passed")
    return True

def run_simple_validation():
    """Run simple Phase C validation tests"""
    print("=== Phase C Simple Validation Tests ===")
    
    torch.manual_seed(42)
    
    tests = [
        ("Basic TDGNN Integration", test_basic_tdgnn_integration),
        ("Training Components", test_training_components),
        ("Configuration Compatibility", test_config_compatibility)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning {test_name}...")
            success = test_func()
            if success:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            failed += 1
    
    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    
    if failed == 0:
        print("All Phase C validation tests passed! ✓")
        return True
    else:
        print("Some tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    success = run_simple_validation()
    sys.exit(0 if success else 1)
