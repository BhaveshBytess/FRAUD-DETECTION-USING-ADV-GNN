#!/usr/bin/env python3
"""
Quick test of the ablation study framework.
Tests just a few lambda weight combinations.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from ablation_study import AblationStudy


def main():
    """Run a quick ablation test."""
    print("🧪 Running Quick Ablation Test...")
    
    # Initialize ablation study
    base_config = 'configs/hypergraph.yaml'
    output_dir = 'experiments/ablation_test'
    
    study = AblationStudy(base_config, output_dir)
    
    # Test just 2 lambda combinations quickly
    lambda_values = [0.5, 1.0]
    results = []
    
    print(f"\n🔬 Testing {len(lambda_values)}x{len(lambda_values)} = {len(lambda_values)**2} lambda combinations...")
    
    for lambda0 in lambda_values:
        for lambda1 in lambda_values:
            modifications = {
                'lambda0_init': float(lambda0),
                'lambda1_init': float(lambda1)
            }
            
            config_path = study.create_config_variant(modifications)
            experiment_name = f"test_l0_{lambda0}_l1_{lambda1}"
            
            print(f"\n--- Testing lambda0={lambda0}, lambda1={lambda1} ---")
            
            result = study.run_single_experiment(
                config_path, 
                experiment_name, 
                epochs=10,  # Quick test with few epochs
                sample=300   # Small sample
            )
            
            result.update({
                'lambda0': lambda0,
                'lambda1': lambda1,
                'ablation_type': 'lambda_test'
            })
            
            results.append(result)
            study.results.append(result)
            
            print(f"✅ Completed: AUC={result['final_test_auc']:.4f}")
    
    # Print summary
    print(f"\n📊 Quick Test Summary:")
    print("=" * 60)
    for result in results:
        status = "✅" if result['status'] == 'success' else "❌"
        print(f"{status} λ0={result['lambda0']}, λ1={result['lambda1']}: "
              f"AUC={result['final_test_auc']:.4f}, F1={result['final_test_f1']:.4f}")
    
    # Find best result
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        best_result = max(successful_results, key=lambda x: x['final_test_auc'])
        print(f"\n🏆 Best Result: λ0={best_result['lambda0']}, λ1={best_result['lambda1']}")
        print(f"   AUC={best_result['final_test_auc']:.4f}, F1={best_result['final_test_f1']:.4f}")
    else:
        print("\n❌ No successful experiments!")
    
    print(f"\n🎯 Quick ablation test completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
