"""
Simple test script for Stage 4 temporal models
Tests the core functionality without full training loop
"""

import torch
import pandas as pd
import numpy as np
from src.temporal_utils import load_temporal_ellipticpp
from src.models.temporal import create_temporal_model
import os

def test_temporal_implementation():
    """Test temporal implementation components."""
    print("=== Stage 4 Temporal Model Test ===")
    
    # Test 1: Load temporal data
    print("\n1. Testing temporal data loading...")
    try:
        temporal_data = load_temporal_ellipticpp(
            'data/ellipticpp', 
            window_size=2, 
            add_temporal_feats=True
        )
        print(f"✓ Temporal data loaded successfully")
        print(f"  - Enhanced features shape: {temporal_data['enhanced_features'].shape}")
        print(f"  - Number of windows: {len(temporal_data['windows'])}")
        print(f"  - Time steps range: {temporal_data['time_step_range']}")
    except Exception as e:
        print(f"✗ Error loading temporal data: {e}")
        return False
    
    # Test 2: Create models
    print("\n2. Testing temporal models...")
    input_dim = temporal_data['enhanced_features'].shape[1]
    
    # Test LSTM
    try:
        lstm_config = {
            'hidden_dim': 64,
            'num_layers': 1,
            'dropout': 0.2,
            'bidirectional': True,
            'use_attention': True
        }
        lstm_model = create_temporal_model('lstm', input_dim, lstm_config)
        print(f"✓ LSTM model created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    except Exception as e:
        print(f"✗ Error creating LSTM model: {e}")
        return False
    
    # Test 3: Model forward pass
    print("\n3. Testing model forward pass...")
    try:
        batch_size = 2
        seq_len = 10
        test_input = torch.randn(batch_size, seq_len, input_dim)
        test_lengths = torch.tensor([10, 8])
        
        with torch.no_grad():
            output = lstm_model(test_input, test_lengths)
            
        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {test_input.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Output logits: {output}")
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        return False
    
    # Test 4: Data splits
    print("\n4. Testing temporal splits...")
    try:
        splits = temporal_data['temporal_splits']
        train_mask = splits['train_mask']
        val_mask = splits['val_mask']
        test_mask = splits['test_mask']
        
        print(f"✓ Temporal splits working")
        print(f"  - Train samples: {train_mask.sum().item()}")
        print(f"  - Validation samples: {val_mask.sum().item()}")
        print(f"  - Test samples: {test_mask.sum().item()}")
        print(f"  - Total samples: {len(train_mask)}")
    except Exception as e:
        print(f"✗ Error with temporal splits: {e}")
        return False
    
    # Test 5: Load labels
    print("\n5. Testing label loading...")
    try:
        labels_file = 'data/ellipticpp/txs_classes.csv'
        if os.path.exists(labels_file):
            labels_data = pd.read_csv(labels_file)
            print(f"✓ Labels loaded successfully")
            print(f"  - Label file shape: {labels_data.shape}")
            print(f"  - Class distribution:")
            print(f"    {labels_data['class'].value_counts()}")
        else:
            print(f"✗ Labels file not found: {labels_file}")
            return False
    except Exception as e:
        print(f"✗ Error loading labels: {e}")
        return False
    
    print("\n=== All tests passed! ===")
    print("Stage 4 temporal implementation is working correctly.")
    print("\nNext steps:")
    print("- Run full training with: python train_temporal.py")
    print("- Compare results with Stage 3 HAN baseline")
    print("- Experiment with different temporal models (GRU, TGAN)")
    
    return True

if __name__ == "__main__":
    success = test_temporal_implementation()
    if success:
        print("\n🎉 Stage 4 temporal models ready for training!")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
