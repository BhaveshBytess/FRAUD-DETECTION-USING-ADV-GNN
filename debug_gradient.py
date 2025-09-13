#!/usr/bin/env python3
"""
Debug gradient computation issue
"""

import torch
import torch.nn.functional as F
import sys
sys.path.append('.')

from src.models.cusp.cusp_core import CuspModule

def debug_gradient_issue():
    """Debug the gradient computation dimension mismatch."""
    print("=== Debugging Gradient Computation Issue ===")
    
    # Create simple test case
    model = CuspModule(
        input_dim=4, 
        hidden_dim=8, 
        output_dim=2, 
        num_layers=1, 
        pooling_strategy='none',
        use_fast_orc=True  # Use fast (zeros) to avoid ORC issues
    )
    
    # Create test data with exactly 6 nodes (to match the error)
    x = torch.randn(6, 4, requires_grad=True)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]
    ])
    target = torch.randint(0, 2, (6,))
    
    print(f"Input shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Number of nodes inferred from edge_index: {edge_index.max().item() + 1}")
    
    try:
        # Forward pass
        output = model(x, edge_index)
        print(f"Forward pass successful. Output shape: {output.shape}")
        
        # Compute loss
        loss = F.cross_entropy(output, target)
        print(f"Loss computed: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        print("✓ Gradient computation successful!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in gradient computation: {e}")
        return False

if __name__ == "__main__":
    success = debug_gradient_issue()
    print(f"\nDebugging result: {'SUCCESS' if success else 'FAILED'}")
