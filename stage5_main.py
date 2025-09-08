#!/usr/bin/env python3
"""
Stage 5 Main Execution Script

This is the main entry point for running Stage 5 advanced fraud detection architectures.
It provides a comprehensive interface for training, evaluation, and benchmarking.

Usage Examples:
    # Run full benchmark
    python stage5_main.py --mode benchmark --config configs/stage5_benchmark.yaml
    
    # Train specific model
    python stage5_main.py --mode train --model graph_transformer --config configs/stage5/graph_transformer.yaml
    
    # Evaluate trained models
    python stage5_main.py --mode evaluate --checkpoint experiments/stage5/checkpoints/best_model.pt
    
    # Quick demo
    python stage5_main.py --mode demo
"""

import argparse
import sys
import os
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# Import project modules
from config import load_config
from utils import set_seed, setup_logging
from run_stage5_benchmark import Stage5BenchmarkRunner, create_default_benchmark_config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 5 Advanced Fraud Detection Architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode benchmark                    # Run full benchmark
  %(prog)s --mode train --model graph_transformer
  %(prog)s --mode demo                         # Quick demonstration
  %(prog)s --mode evaluate --checkpoint path/to/model.pt
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['benchmark', 'train', 'evaluate', 'demo'],
        default='demo',
        help='Execution mode (default: demo)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--model',
        choices=['graph_transformer', 'hetero_graph_transformer', 'temporal_graph_transformer', 'ensemble'],
        help='Model to train (for train mode)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (for evaluate mode)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/stage5',
        help='Output directory (default: experiments/stage5)'
    )
    
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Device to use (default: auto)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode with reduced settings for testing'
    )
    
    return parser.parse_args()


def run_benchmark_mode(args):
    """Run comprehensive benchmark mode."""
    print("🚀 Starting Stage 5 Comprehensive Benchmark Mode")
    print("=" * 60)
    
    # Setup config
    if args.config is None:
        config_path = 'configs/stage5_benchmark.yaml'
        if not Path(config_path).exists():
            create_default_benchmark_config(config_path)
        args.config = config_path
    
    # Modify config for quick mode
    if args.quick:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Reduce settings for quick testing
        config['data']['sample_size'] = 5000
        config['training']['epochs'] = 3
        config['models']['graph_transformer']['num_layers'] = 2
        config['models']['hetero_graph_transformer']['num_layers'] = 2
        config['models']['temporal_graph_transformer']['num_layers'] = 2
        
        # Save modified config
        quick_config_path = f"{args.config.split('.')[0]}_quick.yaml"
        with open(quick_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        args.config = quick_config_path
        print(f"Quick mode enabled - using {quick_config_path}")
    
    # Run benchmark
    runner = Stage5BenchmarkRunner(args.config)
    runner.run_full_benchmark()
    
    print("\n✅ Benchmark completed successfully!")


def run_train_mode(args):
    """Run single model training mode."""
    print(f"🏋️ Training {args.model} model")
    print("=" * 40)
    
    if args.model is None:
        raise ValueError("Model name required for train mode")
    
    # Setup config
    if args.config is None:
        config_path = f'configs/stage5/{args.model}.yaml'
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        args.config = config_path
    
    # Import training modules
    from models.advanced.training import Stage5Trainer
    from load_ellipticpp import load_ellipticpp_data
    from torch.utils.data import DataLoader
    
    # Load data
    print("Loading data...")
    data = load_ellipticpp_data(
        data_path='data/ellipticpp',
        sample_size=10000 if args.quick else None,
        test_size=0.2,
        val_size=0.1
    )
    
    train_loader = DataLoader(data['train_dataset'], batch_size=128, shuffle=True)
    val_loader = DataLoader(data['val_dataset'], batch_size=128, shuffle=False)
    
    # Create trainer
    trainer = Stage5Trainer(
        config_path=args.config,
        output_dir=args.output_dir,
        use_wandb=False
    )
    
    # Train model
    trainer.train(train_loader, val_loader, data['input_dim'])
    
    print(f"✅ {args.model} training completed!")


def run_evaluate_mode(args):
    """Run model evaluation mode."""
    print("📊 Evaluating trained model")
    print("=" * 30)
    
    if args.checkpoint is None:
        raise ValueError("Checkpoint path required for evaluate mode")
    
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # TODO: Implement evaluation logic
    print(f"Loading checkpoint: {args.checkpoint}")
    print("Evaluation functionality coming soon...")


def run_demo_mode(args):
    """Run demonstration mode with synthetic data."""
    print("🎯 Running Stage 5 Demo Mode")
    print("=" * 40)
    print("This demonstrates Stage 5 architectures with synthetic data")
    
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    
    # Import models
    from models.advanced.graph_transformer import create_graph_transformer
    from models.advanced.hetero_graph_transformer import create_heterogeneous_graph_transformer
    from models.advanced.temporal_graph_transformer import create_temporal_graph_transformer
    from models.advanced.evaluation import ModelBenchmark
    
    # Setup synthetic data
    print("Generating synthetic data...")
    batch_size = 100
    input_dim = 186
    num_samples = 1000
    
    # Create synthetic features
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    x = torch.randn(num_samples, input_dim)
    edge_index = torch.randint(0, num_samples, (2, 2000))
    labels = torch.randint(0, 2, (num_samples,))
    
    # Create heterogeneous data
    x_dict = {'transaction': x}
    edge_index_dict = {'transaction__to__transaction': edge_index}
    
    test_data = {
        'x': x,
        'edge_index': edge_index,
        'x_dict': x_dict,
        'edge_index_dict': edge_index_dict
    }
    
    # Setup device
    device = torch.device('cpu')  # Use CPU for demo
    print(f"Using device: {device}")
    
    # Initialize benchmark
    benchmark = ModelBenchmark("experiments/stage5_demo")
    
    # Create and register models
    print("\nCreating Stage 5 models...")
    
    # 1. Graph Transformer
    gt_config = {
        'hidden_dim': 128,
        'num_layers': 3,
        'num_heads': 4,
        'dropout': 0.1
    }
    graph_transformer = create_graph_transformer(input_dim, gt_config)
    benchmark.register_model("Graph_Transformer", graph_transformer, "graph", 5, gt_config)
    print("✓ Graph Transformer created")
    
    # 2. Heterogeneous Graph Transformer
    hgt_config = {
        'hidden_dim': 128,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.1
    }
    input_dims = {'transaction': input_dim}
    hetero_gt = create_heterogeneous_graph_transformer(input_dims, hgt_config)
    benchmark.register_model("Hetero_Graph_Transformer", hetero_gt, "hetero_graph", 5, hgt_config)
    print("✓ Heterogeneous Graph Transformer created")
    
    # 3. Temporal Graph Transformer
    tgt_config = {
        'hidden_dim': 128,
        'num_layers': 3,
        'num_heads': 4,
        'dropout': 0.1,
        'prediction_mode': 'node'
    }
    temporal_gt = create_temporal_graph_transformer(input_dim, tgt_config)
    benchmark.register_model("Temporal_Graph_Transformer", temporal_gt, "temporal_graph", 5, tgt_config)
    print("✓ Temporal Graph Transformer created")
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = benchmark.evaluate_all_models(test_data, labels, device)
    
    # Generate reports
    print("\nGenerating reports...")
    report_df = benchmark.generate_comparison_report()
    benchmark.generate_visualizations()
    benchmark.save_detailed_results()
    
    # Display results
    print("\n📊 Demo Results:")
    print(report_df[['Model', 'Stage', 'AUC', 'F1', 'Parameters']].to_string(index=False))
    
    print(f"\n✅ Demo completed successfully!")
    print(f"📁 Results saved to: experiments/stage5_demo")
    
    # Show next steps
    print("\n🚀 Next Steps:")
    print("1. Run full benchmark: python stage5_main.py --mode benchmark")
    print("2. Train specific model: python stage5_main.py --mode train --model graph_transformer")
    print("3. See benchmark config: configs/stage5_benchmark.yaml")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup
    set_seed(args.seed)
    setup_logging(verbose=args.verbose)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("🤖 Stage 5 Advanced Fraud Detection Architectures")
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    if args.quick:
        print("⚡ Quick mode enabled")
    print()
    
    start_time = time.time()
    
    try:
        # Route to appropriate mode
        if args.mode == 'benchmark':
            run_benchmark_mode(args)
        elif args.mode == 'train':
            run_train_mode(args)
        elif args.mode == 'evaluate':
            run_evaluate_mode(args)
        elif args.mode == 'demo':
            run_demo_mode(args)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
        
        # Success
        elapsed_time = time.time() - start_time
        print(f"\n🎉 Stage 5 execution completed successfully!")
        print(f"⏱️ Total time: {elapsed_time:.2f} seconds")
        
    except KeyboardInterrupt:
        print("\n⚠️ Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
