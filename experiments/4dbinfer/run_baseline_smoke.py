#!/usr/bin/env python3
"""
Baseline Smoke Test for 4DBInfer Framework
Validates that the framework components can load and basic functionality works
"""

import sys
import os
import traceback
from datetime import datetime

# Add the 4DBInfer repo to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'multi-table-benchmark'))

def test_basic_imports():
    """Test that basic 4DBInfer modules can be imported"""
    try:
        # Test basic package import
        import dbinfer_bench as dbb
        print("✓ dbinfer_bench imported successfully")
        
        # Test solution imports
        from dbinfer.solutions import get_gml_solution_choice, get_gml_solution_class
        print("✓ GML solution registry imported successfully")
        
        # Test available solutions
        choices = get_gml_solution_choice()
        print(f"✓ Available GML solutions: {list(choices.__members__.keys())}")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_sage_baseline():
    """Test SAGE baseline model loading (our reference baseline)"""
    try:
        from dbinfer.solutions import get_gml_solution_class
        
        # Get SAGE solution class (our baseline reference)
        sage_class = get_gml_solution_class('sage')
        print(f"✓ SAGE solution class loaded: {sage_class}")
        
        # Test config class
        config_class = sage_class.config_class
        print(f"✓ SAGE config class: {config_class}")
        
        # Create default config
        default_config = config_class(
            lr=0.01,
            batch_size=256,
            eval_batch_size=256,
            fanouts=[10, 10],
            hid_size=64,
            dropout=0.1
        )
        print(f"✓ SAGE default config created successfully")
        
        return True
    except Exception as e:
        print(f"❌ SAGE baseline test failed: {e}")
        traceback.print_exc()
        return False

def test_synthetic_dataset():
    """Test synthetic dataset creation for smoke test"""
    try:
        import torch
        import numpy as np
        from collections import namedtuple
        
        # Create minimal synthetic dataset structure
        GraphDataset = namedtuple('GraphDataset', ['metadata', 'graph_tasks'])
        Task = namedtuple('Task', ['train_set', 'validation_set', 'test_set'])
        
        # Synthetic task sets
        train_set = {'default': torch.randn(100, 64)}
        val_set = {'default': torch.randn(20, 64)}
        test_set = {'default': torch.randn(20, 64)}
        
        task = Task(train_set=train_set, validation_set=val_set, test_set=test_set)
        
        dataset = GraphDataset(
            metadata={'name': 'synthetic_smoke_test'},
            graph_tasks={'smoke_test': task}
        )
        
        print(f"✓ Synthetic dataset created: {len(train_set['default'])} train samples")
        return True, dataset
        
    except Exception as e:
        print(f"❌ Synthetic dataset creation failed: {e}")
        return False, None

def run_baseline_smoke_test():
    """Run complete baseline smoke test"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"experiments/4dbinfer/bootstrap/{timestamp}_baseline_smoke.log"
    
    print("="*60)
    print("4DBInfer Baseline Smoke Test")
    print("="*60)
    
    results = {
        'timestamp': timestamp,
        'imports': False,
        'sage_baseline': False,
        'synthetic_data': False,
        'overall_success': False
    }
    
    # Test 1: Basic imports
    print("\n1. Testing basic 4DBInfer imports...")
    results['imports'] = test_basic_imports()
    
    # Test 2: SAGE baseline
    print("\n2. Testing SAGE baseline model...")
    results['sage_baseline'] = test_sage_baseline()
    
    # Test 3: Synthetic dataset
    print("\n3. Testing synthetic dataset creation...")
    results['synthetic_data'], _ = test_synthetic_dataset()
    
    # Overall result
    results['overall_success'] = all([
        results['imports'],
        results['sage_baseline'], 
        results['synthetic_data']
    ])
    
    print("\n" + "="*60)
    if results['overall_success']:
        print("✅ BASELINE SMOKE TEST PASSED")
        print("4DBInfer framework is functional for integration")
    else:
        print("❌ BASELINE SMOKE TEST FAILED")
        print("Some framework components are not working")
    
    print("="*60)
    
    return results

if __name__ == "__main__":
    results = run_baseline_smoke_test()
    
    # Print final summary
    print(f"\nTest Results Summary:")
    for test, passed in results.items():
        if test != 'timestamp':
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test}: {status}")
