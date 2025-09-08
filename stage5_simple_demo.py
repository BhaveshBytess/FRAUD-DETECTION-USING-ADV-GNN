#!/usr/bin/env python3
"""
Stage 5 Simple Demo Script

A simplified demonstration of Stage 5 architectures without external dependencies.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# Import our models
from models.advanced.graph_transformer import create_graph_transformer
from models.advanced.hetero_graph_transformer import create_heterogeneous_graph_transformer
from models.advanced.temporal_graph_transformer import create_temporal_graph_transformer
from models.advanced.evaluation import ModelBenchmark


def run_simple_demo():
    """Run a simple demonstration of Stage 5 models."""
    
    print("🎯 Stage 5 Simple Demo")
    print("=" * 40)
    print("Demonstrating advanced transformer architectures for fraud detection")
    print()
    
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cpu')
    
    # Parameters
    batch_size = 100
    input_dim = 186
    num_samples = 1000
    
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Input dimension: {input_dim}")
    print(f"Number of samples: {num_samples}")
    print()
    
    # Generate synthetic data
    print("📊 Generating synthetic fraud detection data...")
    
    # Node features
    x = torch.randn(num_samples, input_dim)
    
    # Graph structure
    edge_index = torch.randint(0, num_samples, (2, 2000))
    
    # Labels (fraud detection)
    labels = torch.randint(0, 2, (num_samples,))
    fraud_rate = labels.float().mean()
    print(f"Fraud rate: {fraud_rate:.1%}")
    
    # Heterogeneous data
    x_dict = {'transaction': x}
    edge_index_dict = {'transaction__to__transaction': edge_index}
    
    test_data = {
        'x': x,
        'edge_index': edge_index,
        'x_dict': x_dict,
        'edge_index_dict': edge_index_dict
    }
    
    print("✓ Data generated")
    print()
    
    # Initialize benchmark
    print("🔧 Setting up evaluation framework...")
    benchmark = ModelBenchmark("experiments/stage5_simple_demo")
    print("✓ Benchmark initialized")
    print()
    
    # Create and register models
    print("🏗️ Creating Stage 5 models...")
    
    # 1. Graph Transformer
    print("  Creating Graph Transformer...")
    gt_config = {
        'hidden_dim': 64,  # Smaller for demo
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.1,
        'use_positional_encoding': True
    }
    
    try:
        graph_transformer = create_graph_transformer(input_dim, gt_config)
        benchmark.register_model("Graph_Transformer", graph_transformer, "graph", 5, gt_config)
        
        # Count parameters
        total_params = sum(p.numel() for p in graph_transformer.parameters())
        print(f"    ✓ Graph Transformer ({total_params:,} parameters)")
    except Exception as e:
        print(f"    ❌ Graph Transformer failed: {e}")
    
    # 2. Heterogeneous Graph Transformer
    print("  Creating Heterogeneous Graph Transformer...")
    hgt_config = {
        'hidden_dim': 64,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.1
    }
    
    try:
        input_dims = {'transaction': input_dim}
        hetero_gt = create_heterogeneous_graph_transformer(input_dims, hgt_config)
        benchmark.register_model("Hetero_Graph_Transformer", hetero_gt, "hetero_graph", 5, hgt_config)
        
        total_params = sum(p.numel() for p in hetero_gt.parameters())
        print(f"    ✓ Heterogeneous Graph Transformer ({total_params:,} parameters)")
    except Exception as e:
        print(f"    ❌ Heterogeneous Graph Transformer failed: {e}")
    
    # 3. Temporal Graph Transformer
    print("  Creating Temporal Graph Transformer...")
    tgt_config = {
        'hidden_dim': 64,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.1,
        'prediction_mode': 'node'
    }
    
    try:
        temporal_gt = create_temporal_graph_transformer(input_dim, tgt_config)
        benchmark.register_model("Temporal_Graph_Transformer", temporal_gt, "temporal_graph", 5, tgt_config)
        
        total_params = sum(p.numel() for p in temporal_gt.parameters())
        print(f"    ✓ Temporal Graph Transformer ({total_params:,} parameters)")
    except Exception as e:
        print(f"    ❌ Temporal Graph Transformer failed: {e}")
    
    print()
    
    # Run evaluation
    print("📈 Running model evaluation...")
    start_time = time.time()
    
    try:
        results = benchmark.evaluate_all_models(test_data, labels, device)
        
        evaluation_time = time.time() - start_time
        print(f"✓ Evaluation completed in {evaluation_time:.2f} seconds")
        print()
        
        # Generate reports
        print("📊 Generating comparison report...")
        report_df = benchmark.generate_comparison_report()
        
        # Display results
        print("🏆 Demo Results:")
        print("-" * 50)
        
        # Filter successful models
        successful_models = report_df[report_df['AUC'] > 0]
        
        if len(successful_models) > 0:
            display_cols = ['Model', 'Stage', 'AUC', 'F1', 'Parameters', 'Inference_Time(s)']
            print(successful_models[display_cols].to_string(index=False))
            
            # Best model
            best_model = successful_models.loc[successful_models['AUC'].idxmax()]
            print(f"\n🥇 Best Model: {best_model['Model']}")
            print(f"   AUC: {best_model['AUC']:.4f}")
            print(f"   F1: {best_model['F1']:.4f}")
            print(f"   Parameters: {best_model['Parameters']:,}")
            
        else:
            print("No successful model evaluations")
        
        print()
        
        # Generate visualizations
        print("📊 Generating visualizations...")
        try:
            benchmark.generate_visualizations()
            print("✓ Visualizations saved")
        except Exception as e:
            print(f"⚠️ Visualization generation failed: {e}")
        
        # Save results
        print("💾 Saving detailed results...")
        benchmark.save_detailed_results()
        print("✓ Results saved")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("✅ Stage 5 Simple Demo Completed!")
    print(f"📁 Results saved to: experiments/stage5_simple_demo")
    print()
    print("🚀 Next Steps:")
    print("1. Install PyTorch Geometric for full functionality")
    print("2. Run with real data: python stage5_main.py --mode benchmark")
    print("3. Train individual models: python stage5_main.py --mode train --model graph_transformer")
    print("4. Explore configuration files in configs/stage5/")


if __name__ == "__main__":
    try:
        run_simple_demo()
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
