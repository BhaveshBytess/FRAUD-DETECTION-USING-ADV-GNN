"""
Unit tests for Phase 2: PhenomNN Layer Implementation

Tests the mathematical correctness and implementation of:
- PhenomNNSimpleLayer (Equation 25)
- PhenomNNLayer (Equation 22)  
- Energy-based updates and convergence
- Gradient flow and numerical stability

Following the PhenomNN paper specifications exactly.
"""

import torch
import numpy as np
import pytest
from typing import Dict, List, Tuple
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.models.hypergraph.hypergraph_data import HypergraphData
    from src.models.hypergraph.phenomnn import PhenomNNSimpleLayer, PhenomNNLayer, PhenomNNBlock
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)


def create_test_hypergraph_data(n_nodes=6, n_hyperedges=4, input_dim=8) -> HypergraphData:
    """Create test hypergraph data for layer testing."""
    # Ensure incidence matrix matches n_nodes
    if n_nodes == 4:
        # 4-node case
        incidence_matrix = torch.tensor([
            [1, 0, 1, 0],  # node 0: hyperedges 0, 2
            [1, 1, 0, 0],  # node 1: hyperedges 0, 1
            [0, 1, 1, 0],  # node 2: hyperedges 1, 2
            [0, 1, 0, 1]   # node 3: hyperedges 1, 3
        ], dtype=torch.float)
    elif n_nodes == 5:
        # 5-node case
        incidence_matrix = torch.tensor([
            [1, 0, 1, 0],  # node 0: hyperedges 0, 2
            [1, 1, 0, 0],  # node 1: hyperedges 0, 1
            [0, 1, 1, 0],  # node 2: hyperedges 1, 2
            [0, 1, 0, 1],  # node 3: hyperedges 1, 3
            [0, 0, 1, 1]   # node 4: hyperedges 2, 3
        ], dtype=torch.float)
    elif n_nodes == 2:
        # 2-node case
        incidence_matrix = torch.tensor([
            [1, 0],  # node 0: hyperedge 0
            [1, 1]   # node 1: hyperedges 0, 1
        ], dtype=torch.float)
        n_hyperedges = 2  # Override for 2-node case
    else:
        # Default 6-node case
        incidence_matrix = torch.tensor([
            [1, 0, 1, 0],  # node 0: hyperedges 0, 2
            [1, 1, 0, 0],  # node 1: hyperedges 0, 1
            [0, 1, 1, 0],  # node 2: hyperedges 1, 2
            [0, 1, 0, 1],  # node 3: hyperedges 1, 3
            [0, 0, 1, 1],  # node 4: hyperedges 2, 3
            [0, 0, 0, 1]   # node 5: hyperedge 3
        ], dtype=torch.float)
    
    node_features = torch.randn(n_nodes, input_dim)
    node_labels = torch.randint(0, 2, (n_nodes,))
    
    return HypergraphData(incidence_matrix, node_features, node_labels=node_labels)


def test_phenomnn_simple_layer_creation():
    """Test PhenomNNSimpleLayer creation and basic properties."""
    input_dim, hidden_dim = 8, 16
    layer = PhenomNNSimpleLayer(input_dim, hidden_dim)
    
    # Test basic properties
    assert layer.input_dim == input_dim
    assert layer.hidden_dim == hidden_dim
    assert layer.lambda0 == 1.0  # Default values
    assert layer.lambda1 == 1.0
    assert layer.alpha == 0.1
    
    # Test learnable parameters
    assert layer.feature_transform.in_features == input_dim
    assert layer.feature_transform.out_features == hidden_dim


def test_phenomnn_simple_layer_forward():
    """Test PhenomNNSimpleLayer forward pass implementing Equation 25."""
    hg_data = create_test_hypergraph_data(n_nodes=6, input_dim=8)
    layer = PhenomNNSimpleLayer(input_dim=8, hidden_dim=12, num_iterations=3)
    
    # Forward pass
    Y_output, iteration_info = layer(hg_data)
    
    # Test output shape
    assert Y_output.shape == (6, 12)  # (n_nodes, hidden_dim)
    
    # Test non-negative outputs (ReLU activation)
    assert torch.all(Y_output >= 0)
    
    # Test iteration info
    assert 'convergence_history' in iteration_info
    assert 'converged' in iteration_info
    assert 'final_iteration' in iteration_info
    assert len(iteration_info['convergence_history']) <= 3  # max iterations


def test_phenomnn_layer_forward():
    """Test full PhenomNNLayer forward pass implementing Equation 22."""
    hg_data = create_test_hypergraph_data(n_nodes=6, input_dim=8)
    layer = PhenomNNLayer(input_dim=8, hidden_dim=12, num_iterations=5)
    
    # Forward pass
    Y_output, iteration_info = layer(hg_data)
    
    # Test output shape
    assert Y_output.shape == (6, 12)
    
    # Test non-negative outputs
    assert torch.all(Y_output >= 0)
    
    # Test iteration info includes energy tracking
    assert 'energy_history' in iteration_info
    assert 'alpha_history' in iteration_info
    assert len(iteration_info['energy_history']) <= 5


def test_equation_25_mathematical_correctness():
    """Test mathematical correctness of Equation 25 implementation."""
    hg_data = create_test_hypergraph_data(n_nodes=4, input_dim=6)
    layer = PhenomNNSimpleLayer(
        input_dim=6, 
        hidden_dim=6,  # Same dim to avoid dimension issues
        num_iterations=1,  # Single iteration for testing
        alpha=0.5,  # Larger step for observable changes
        lambda0=1.0,
        lambda1=1.0
    )
    
    # Get initial state
    X = hg_data.X
    
    # Manual computation of Equation 25 components
    with torch.no_grad():
        # f(X;W) computation
        f_X = layer.feature_transform(X)
        
        # Get matrices
        D_tilde = hg_data.get_preconditioner(layer.lambda0, layer.lambda1)
        expansion_matrix = layer.lambda0 * hg_data.AC + layer.lambda1 * hg_data.AS_bar
        
        # Initialize Y
        Y_init = f_X.clone()
        
        # Manual update: Y^(t+1) = ReLU((1-α)Y^(t) + αD̃^(-1)[(λ0*AC + λ1*AS_bar)*Y^(t) + f(X;W)])
        D_tilde_inv = torch.inverse(D_tilde + 1e-6 * torch.eye(D_tilde.shape[0]))
        expansion_term = expansion_matrix @ Y_init
        update_term = expansion_term + f_X
        preconditioned_update = D_tilde_inv @ update_term
        Y_manual = torch.relu((1 - layer.alpha) * Y_init + layer.alpha * preconditioned_update)
    
    # Layer computation
    Y_layer, _ = layer(hg_data, Y_init=Y_init)
    
    # Should match within numerical precision
    assert torch.allclose(Y_layer, Y_manual, atol=1e-5), "Equation 25 implementation mismatch"


def test_convergence_behavior():
    """Test convergence behavior and iteration tracking."""
    hg_data = create_test_hypergraph_data(n_nodes=5, input_dim=4)
    
    # Layer with tight convergence threshold
    layer = PhenomNNSimpleLayer(
        input_dim=4,
        hidden_dim=4,
        num_iterations=20,
        convergence_threshold=1e-4,
        alpha=0.1
    )
    
    Y_output, iteration_info = layer(hg_data)
    
    # Test convergence tracking
    convergence_history = iteration_info['convergence_history']
    assert len(convergence_history) > 0
    
    # Convergence should generally decrease (not strict due to ReLU)
    if len(convergence_history) > 1:
        # At least shouldn't increase dramatically
        assert convergence_history[-1] < convergence_history[0] * 10
    
    # If converged, final change should be below threshold
    if iteration_info['converged']:
        assert convergence_history[-1] < 1e-4


def test_learnable_parameters():
    """Test learnable parameters in PhenomNNLayer."""
    hg_data = create_test_hypergraph_data(n_nodes=4, input_dim=6)
    layer = PhenomNNLayer(input_dim=6, hidden_dim=8, adaptive_alpha=True)
    
    # Check learnable parameters exist
    param_names = [name for name, _ in layer.named_parameters()]
    assert 'clique_weight' in param_names
    assert 'star_weight' in param_names
    assert 'alpha_param' in param_names
    
    # Check parameter values are reasonable
    assert 0 <= layer.clique_weight.item() <= 10
    assert 0 <= layer.star_weight.item() <= 10
    assert 0.01 <= layer.alpha_param.item() <= 0.5


def test_gradient_flow():
    """Test gradient flow through PhenomNN layers."""
    hg_data = create_test_hypergraph_data(n_nodes=4, input_dim=6)
    layer = PhenomNNSimpleLayer(input_dim=6, hidden_dim=8, num_iterations=2)
    
    # Create dummy loss
    Y_output, _ = layer(hg_data)
    loss = Y_output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist and are reasonable
    feature_grad = layer.feature_transform.weight.grad
    assert feature_grad is not None
    assert torch.isfinite(feature_grad).all()
    assert feature_grad.norm() > 0  # Non-zero gradients


def test_numerical_stability():
    """Test numerical stability with edge cases."""
    # Test with very small hypergraph
    small_hg = create_test_hypergraph_data(n_nodes=2, n_hyperedges=1, input_dim=4)
    layer = PhenomNNSimpleLayer(input_dim=4, hidden_dim=6)
    
    Y_output, iteration_info = layer(small_hg)
    
    # Should not produce NaN or Inf
    assert torch.isfinite(Y_output).all()
    assert not torch.isnan(Y_output).any()
    
    # Test with larger step size
    aggressive_layer = PhenomNNSimpleLayer(
        input_dim=4, 
        hidden_dim=6, 
        alpha=0.9,  # Very large step size
        num_iterations=1
    )
    
    Y_aggressive, _ = aggressive_layer(small_hg)
    assert torch.isfinite(Y_aggressive).all()


def test_phenomnn_block():
    """Test PhenomNNBlock with multiple layers."""
    hg_data = create_test_hypergraph_data(n_nodes=5, input_dim=8)
    
    # Test simple block
    block = PhenomNNBlock(
        input_dim=8,
        hidden_dim=12,
        num_layers=2,
        layer_type='simple',
        num_iterations=2
    )
    
    Y_output, all_info = block(hg_data)
    
    # Test output shape
    assert Y_output.shape == (5, 12)
    
    # Test iteration info for all layers
    assert 'layer_0' in all_info
    assert 'layer_1' in all_info
    
    # Test residual connection effect (output shouldn't be exactly layer output)
    layer_only = PhenomNNSimpleLayer(input_dim=8, hidden_dim=12, num_iterations=2)
    Y_layer_only, _ = layer_only(hg_data)
    
    # With residual, output should be different
    assert not torch.allclose(Y_output, Y_layer_only)


def test_energy_computation():
    """Test energy computation in full PhenomNNLayer."""
    hg_data = create_test_hypergraph_data(n_nodes=4, input_dim=6)
    layer = PhenomNNLayer(input_dim=6, hidden_dim=6, num_iterations=3)
    
    Y_output, iteration_info = layer(hg_data)
    
    # Energy should be tracked
    energy_history = iteration_info['energy_history']
    assert len(energy_history) > 0
    
    # Energy should be positive (it's a sum of squared norms and traces)
    assert all(e >= 0 for e in energy_history)


def test_different_lambda_values():
    """Test different expansion weight combinations."""
    hg_data = create_test_hypergraph_data(n_nodes=4, input_dim=6)
    
    # Test different lambda combinations
    configs = [
        (1.0, 0.0),  # Clique only
        (0.0, 1.0),  # Star only  
        (1.0, 1.0),  # Balanced
        (2.0, 0.5)   # Clique-heavy
    ]
    
    outputs = []
    for lambda0, lambda1 in configs:
        layer = PhenomNNSimpleLayer(
            input_dim=6, 
            hidden_dim=6,
            lambda0=lambda0,
            lambda1=lambda1,
            num_iterations=2
        )
        Y, _ = layer(hg_data)
        outputs.append(Y)
    
    # Different lambda values should produce different outputs
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            assert not torch.allclose(outputs[i], outputs[j], atol=1e-4), f"Outputs {i} and {j} are too similar"


def test_device_handling():
    """Test proper device handling."""
    hg_data = create_test_hypergraph_data(n_nodes=4, input_dim=6)
    layer = PhenomNNSimpleLayer(input_dim=6, hidden_dim=8)
    
    # Test CPU
    Y_cpu, _ = layer(hg_data)
    assert Y_cpu.device == torch.device('cpu')
    
    # Test CUDA (if available)
    if torch.cuda.is_available():
        hg_data_cuda = hg_data.to(torch.device('cuda'))
        layer_cuda = layer.to(torch.device('cuda'))
        
        Y_cuda, _ = layer_cuda(hg_data_cuda)
        assert Y_cuda.device.type == 'cuda'


if __name__ == '__main__':
    # Run tests manually for debugging
    print("Running Phase 2 PhenomNN Layer Tests...")
    
    test_phenomnn_simple_layer_creation()
    print("✓ PhenomNNSimpleLayer creation test passed")
    
    test_phenomnn_simple_layer_forward()
    print("✓ PhenomNNSimpleLayer forward test passed")
    
    test_phenomnn_layer_forward()
    print("✓ PhenomNNLayer forward test passed")
    
    test_equation_25_mathematical_correctness()
    print("✓ Equation 25 mathematical correctness test passed")
    
    test_convergence_behavior()
    print("✓ Convergence behavior test passed")
    
    test_learnable_parameters()
    print("✓ Learnable parameters test passed")
    
    test_gradient_flow()
    print("✓ Gradient flow test passed")
    
    test_numerical_stability()
    print("✓ Numerical stability test passed")
    
    test_phenomnn_block()
    print("✓ PhenomNNBlock test passed")
    
    test_energy_computation()
    print("✓ Energy computation test passed")
    
    test_different_lambda_values()
    print("✓ Different lambda values test passed")
    
    print("\n🎉 All Phase 2 tests passed! PhenomNN layers are working correctly.")
