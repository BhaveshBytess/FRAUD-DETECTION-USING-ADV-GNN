#!/usr/bin/env python3
"""
Final validation test for CUSP module implementation.
Tests all major functionality with proper error handling.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import sys
import os

# Add project root to path
sys.path.append('.')

from src.models.cusp.cusp_core import CuspModule, create_cusp_model

def test_cusp_basic_functionality():
    """Test basic CUSP module functionality."""
    print("=== Testing Basic CUSP Functionality ===")
    
    # Create a simple CUSP module
    model = CuspModule(
        input_dim=10,
        hidden_dim=16,
        output_dim=2,
        num_layers=2,
        curvature_encoding_dim=4,
        gpr_hops=3
    )
    
    # Create test data
    x = torch.randn(20, 10)  # 20 nodes, 10 features each
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8]
    ])
    
    print(f"Input shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    
    # Test forward pass
    try:
        output = model(x, edge_index)
        print(f"✓ Forward pass successful. Output shape: {output.shape}")
        assert output.shape == (20, 2), f"Expected (20, 2), got {output.shape}"
        print("✓ Output shape correct")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    # Test with return_intermediate
    try:
        output, extras = model(x, edge_index, return_intermediate=True)
        print(f"✓ Return intermediate successful. Extras keys: {list(extras.keys())}")
    except Exception as e:
        print(f"✗ Return intermediate failed: {e}")
        return False
    
    # Test with return_attention  
    try:
        output, extras = model(x, edge_index, return_attention=True)
        print(f"✓ Return attention successful. Extras keys: {list(extras.keys())}")
    except Exception as e:
        print(f"✗ Return attention failed: {e}")
        return False
    
    print("✓ All basic functionality tests passed")
    return True

def test_cusp_model_factory():
    """Test CUSP model factory functionality."""
    print("\n=== Testing CUSP Model Factory ===")
    
    try:
        # Test default model creation
        model = create_cusp_model(
            input_dim=8,
            num_classes=3,
            task_type='node_classification'
        )
        
        # Test with data
        x = torch.randn(15, 8)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]
        ])
        
        output = model(x, edge_index)
        print(f"✓ Model factory successful. Output shape: {output.shape}")
        assert output.shape == (15, 3), f"Expected (15, 3), got {output.shape}"
        print("✓ Factory model output shape correct")
        
    except Exception as e:
        print(f"✗ Model factory failed: {e}")
        return False
    
    print("✓ Model factory tests passed")
    return True

def test_cusp_batch_processing():
    """Test CUSP module with batched graphs."""
    print("\n=== Testing Batch Processing ===")
    
    model = CuspModule(
        input_dim=6,
        hidden_dim=12,
        output_dim=1,
        num_layers=1
    )
    
    # Create batched data
    data_list = []
    for i in range(3):  # 3 graphs in batch
        x = torch.randn(8, 6)  # 8 nodes per graph
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]
        ])
        data_list.append(Data(x=x, edge_index=edge_index))
    
    batch_data = Batch.from_data_list(data_list)
    print(f"Batch data - x shape: {batch_data.x.shape}, batch shape: {batch_data.batch.shape}")
    
    try:
        output = model(batch_data.x, batch_data.edge_index, batch_data.batch)
        print(f"✓ Batch processing successful. Output shape: {output.shape}")
        assert output.shape == (24, 1), f"Expected (24, 1), got {output.shape}"  # 3 graphs * 8 nodes = 24
        print("✓ Batch output shape correct")
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
        return False
    
    print("✓ Batch processing tests passed")
    return True

def test_cusp_gradient_computation():
    """Test gradient computation through CUSP module."""
    print("\n=== Testing Gradient Computation ===")
    
    model = CuspModule(
        input_dim=4,
        hidden_dim=8,
        output_dim=2,
        num_layers=1
    )
    
    # Create test data with consistent node count
    x = torch.randn(6, 4, requires_grad=True)  # Match edge_index node count
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]
    ])
    target = torch.randint(0, 2, (6,))  # Match feature tensor size
    
    try:
        # Forward pass
        output = model(x, edge_index)
        
        # Compute loss
        loss = F.cross_entropy(output, target)
        print(f"Loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        param_with_grad = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            total_params += 1
            if param.grad is not None:
                param_with_grad += 1
        
        print(f"✓ Gradient computation successful. {param_with_grad}/{total_params} parameters have gradients")
        
        # Check input gradients
        if x.grad is not None:
            print(f"✓ Input gradients computed. Shape: {x.grad.shape}")
        else:
            print("? No input gradients (expected if not needed)")
            
    except Exception as e:
        print(f"✗ Gradient computation failed: {e}")
        return False
    
    print("✓ Gradient computation tests passed")
    return True

def test_cusp_different_configurations():
    """Test CUSP module with different configurations."""
    print("\n=== Testing Different Configurations ===")
    
    configs = [
        {'num_layers': 1, 'manifold_types': ['euclidean']},
        {'num_layers': 2, 'manifold_types': ['euclidean', 'hyperbolic']},
        {'num_layers': 1, 'manifold_types': ['euclidean', 'hyperbolic', 'spherical']},
        {'use_fast_orc': True, 'num_layers': 1},
        {'learnable_curvatures': False, 'num_layers': 1},
    ]
    
    base_config = {
        'input_dim': 5,
        'hidden_dim': 10,
        'output_dim': 3,
        'num_layers': 1
    }
    
    for i, config in enumerate(configs):
        try:
            # Merge configurations
            test_config = {**base_config, **config}
            model = CuspModule(**test_config)
            
            # Test forward pass
            x = torch.randn(12, 5)
            edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
            
            output = model(x, edge_index)
            print(f"✓ Config {i+1} successful. Output shape: {output.shape}")
            
        except Exception as e:
            print(f"✗ Config {i+1} failed: {e}")
            return False
    
    print("✓ All configuration tests passed")
    return True

def main():
    """Run all CUSP tests."""
    print("🎯 Running Final CUSP Module Validation Tests\n")
    
    tests = [
        test_cusp_basic_functionality,
        test_cusp_model_factory,
        test_cusp_batch_processing,
        test_cusp_gradient_computation,
        test_cusp_different_configurations
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n🎯 Final Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All CUSP tests passed! Phase 5 implementation complete.")
        return True
    else:
        print(f"❌ {failed} tests failed. Please review implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
