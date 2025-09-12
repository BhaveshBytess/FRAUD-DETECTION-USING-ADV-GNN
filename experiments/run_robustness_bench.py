"""
Robustness Benchmarking Experiment
Following Stage7 Spot Target And Robustness Reference §Phase5

Benchmark DropEdge, RGNN, adversarial training robustness modules.
Test performance overhead and defensive effectiveness.
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from robustness import (
    DropEdge,
    RGNNWrapper, 
    AdversarialEdgeTrainer,
    create_robust_model,
    benchmark_dropedge_determinism,
    benchmark_rgnn_overhead
)
from spot_target import SpotTargetSampler, compute_avg_degree
from training_wrapper import SpotTargetTrainer
from imbalance import ImbalanceHandler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttackableGNN(nn.Module):
    """
    Simple GNN that can be attacked for robustness testing.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Feature transformation layers
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Message passing layers (simplified as MLPs)
        self.message_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.message_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))
        
        # Output layer
        self.classifier = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with simplified message passing."""
        # Encode features
        h = self.feature_encoder(x)
        
        # Simplified message passing (can be extended)
        for layer in self.message_layers:
            h = layer(h)
        
        # Classification
        logits = self.classifier(h)
        return logits


def create_adversarial_dataset(
    num_nodes: int = 1000,
    num_features: int = 64,
    num_classes: int = 2,
    edge_prob: float = 0.02,
    noise_level: float = 0.1,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    Create dataset suitable for adversarial attacks.
    
    Args:
        num_nodes: Number of nodes
        num_features: Feature dimension
        num_classes: Number of classes
        edge_prob: Edge probability
        noise_level: Noise level in features
        seed: Random seed
        
    Returns:
        dataset: Graph dataset dict
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create structured features with some noise
    base_features = torch.randn(num_nodes, num_features // 2)
    noise_features = torch.randn(num_nodes, num_features // 2) * noise_level
    x = torch.cat([base_features, noise_features], dim=1)
    
    # Create graph with community structure
    edges = []
    
    # Create communities
    community_size = num_nodes // num_classes
    
    for c in range(num_classes):
        start_node = c * community_size
        end_node = min((c + 1) * community_size, num_nodes)
        
        # Intra-community edges (higher probability)
        for i in range(start_node, end_node):
            for j in range(i + 1, end_node):
                if torch.rand(1).item() < edge_prob * 3:  # Higher intra-community connectivity
                    edges.extend([(i, j), (j, i)])
        
        # Inter-community edges (lower probability)
        for i in range(start_node, end_node):
            for j in range(num_nodes):
                if j < start_node or j >= end_node:  # Different community
                    if torch.rand(1).item() < edge_prob * 0.3:  # Lower inter-community connectivity
                        edges.extend([(i, j), (j, i)])
    
    # Ensure minimum connectivity
    if len(edges) < num_nodes:
        for i in range(num_nodes - 1):
            edges.extend([(i, i + 1), (i + 1, i)])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    # Create labels based on community structure
    labels = torch.zeros(num_nodes, dtype=torch.long)
    for c in range(num_classes):
        start_node = c * community_size
        end_node = min((c + 1) * community_size, num_nodes)
        labels[start_node:end_node] = c
    
    # Add some label noise
    noise_indices = torch.randperm(num_nodes)[:int(0.1 * num_nodes)]
    labels[noise_indices] = torch.randint(0, num_classes, (len(noise_indices),))
    
    # Create masks
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    indices = torch.randperm(num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    # Create edge splits
    num_edges = edge_index.size(1)
    edge_indices = torch.randperm(num_edges)
    
    train_edge_size = int(0.7 * num_edges)
    val_edge_size = int(0.15 * num_edges)
    
    edge_splits = {
        'train': torch.zeros(num_edges, dtype=torch.bool),
        'valid': torch.zeros(num_edges, dtype=torch.bool),
        'test': torch.zeros(num_edges, dtype=torch.bool)
    }
    
    edge_splits['train'][edge_indices[:train_edge_size]] = True
    edge_splits['valid'][edge_indices[train_edge_size:train_edge_size + val_edge_size]] = True
    edge_splits['test'][edge_indices[train_edge_size + val_edge_size:]] = True
    
    dataset = {
        'x': x,
        'edge_index': edge_index,
        'y': labels,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'edge_splits': edge_splits,
        'num_nodes': num_nodes,
        'num_features': num_features,
        'num_classes': num_classes
    }
    
    return dataset


def generate_edge_attack(
    edge_index: torch.Tensor,
    attack_ratio: float = 0.1,
    attack_type: str = 'random'
) -> torch.Tensor:
    """
    Generate adversarial edge perturbations.
    
    Args:
        edge_index: Original edge indices
        attack_ratio: Fraction of edges to perturb
        attack_type: Type of attack ('random', 'targeted')
        
    Returns:
        attacked_edge_index: Perturbed edge indices
    """
    num_edges = edge_index.size(1)
    num_attacks = int(attack_ratio * num_edges)
    
    # Clone original edges
    attacked_edges = edge_index.clone()
    
    if attack_type == 'random':
        # Random edge removal
        remove_indices = torch.randperm(num_edges)[:num_attacks // 2]
        mask = torch.ones(num_edges, dtype=torch.bool)
        mask[remove_indices] = False
        attacked_edges = attacked_edges[:, mask]
        
        # Random edge addition
        max_node = edge_index.max().item()
        new_edges = []
        for _ in range(num_attacks // 2):
            src = torch.randint(0, max_node + 1, (1,)).item()
            dst = torch.randint(0, max_node + 1, (1,)).item()
            if src != dst:
                new_edges.extend([[src, dst], [dst, src]])
        
        if new_edges:
            new_edge_tensor = torch.tensor(new_edges, dtype=torch.long).t()
            attacked_edges = torch.cat([attacked_edges, new_edge_tensor], dim=1)
    
    elif attack_type == 'targeted':
        # More sophisticated attack (could be implemented)
        pass
    
    return attacked_edges


def benchmark_robustness_modules(
    dataset: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    """
    Benchmark individual robustness modules.
    
    Args:
        dataset: Graph dataset
        config: Configuration dict
        device: Computation device
        
    Returns:
        benchmark_results: Performance metrics for each module
    """
    # Move dataset to device
    for key, value in dataset.items():
        if torch.is_tensor(value):
            dataset[key] = value.to(device)
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if torch.is_tensor(subvalue):
                    dataset[key][subkey] = subvalue.to(device)
    
    results = {}
    
    # Benchmark 1: DropEdge determinism and overhead
    logger.info("Benchmarking DropEdge...")
    
    dropedge_config = {
        'dropedge_p': config.get('dropedge_p', 0.1),
        'deterministic': True
    }
    
    # Test determinism
    determinism_results = benchmark_dropedge_determinism(
        dataset['edge_index'], 
        dropedge_config['dropedge_p'],
        num_trials=10
    )
    
    # Test overhead
    dropedge_module = DropEdge(p_drop=dropedge_config['dropedge_p'], training=True)
    
    # Time baseline
    start_time = time.time()
    for _ in range(100):
        _ = dataset['edge_index']
    baseline_time = time.time() - start_time
    
    # Time with DropEdge
    start_time = time.time()
    for _ in range(100):
        _ = dropedge_module(dataset['edge_index'])
    dropedge_time = time.time() - start_time
    
    dropedge_overhead = dropedge_time / baseline_time if baseline_time > 0 else float('inf')
    
    results['dropedge'] = {
        'determinism': determinism_results,
        'overhead_ratio': dropedge_overhead,
        'config': dropedge_config
    }
    
    # Benchmark 2: RGNN Wrapper overhead
    logger.info("Benchmarking RGNN Wrapper...")
    
    base_model = AttackableGNN(
        input_dim=dataset['num_features'],
        hidden_dim=config['hidden_dim'],
        output_dim=dataset['num_classes']
    ).to(device)
    
    rgnn_config = {
        'attention_gating': True,
        'spectral_norm': True,
        'attention_dim': config['hidden_dim'] // 2
    }
    
    rgnn_overhead = benchmark_rgnn_overhead(
        base_model,
        dataset['x'],
        dataset['edge_index'],
        rgnn_config,
        num_trials=50
    )
    
    results['rgnn'] = {
        'overhead': rgnn_overhead,
        'config': rgnn_config
    }
    
    # Benchmark 3: Adversarial Training effectiveness
    logger.info("Benchmarking Adversarial Training...")
    
    # Test model without adversarial training
    clean_model = AttackableGNN(
        input_dim=dataset['num_features'],
        hidden_dim=config['hidden_dim'],
        output_dim=dataset['num_classes']
    ).to(device)
    
    # Test model with adversarial training
    robust_model = create_robust_model(clean_model, config).to(device)
    
    # Generate attacks
    clean_edges = dataset['edge_index']
    attacked_edges = generate_edge_attack(clean_edges, attack_ratio=0.1, attack_type='random')
    
    # Quick evaluation
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate clean model on clean data
    clean_model.eval()
    with torch.no_grad():
        clean_logits = clean_model(dataset['x'], clean_edges)
        clean_acc_clean = (clean_logits[dataset['test_mask']].argmax(dim=1) == 
                          dataset['y'][dataset['test_mask']]).float().mean().item()
    
    # Evaluate clean model on attacked data
    with torch.no_grad():
        attacked_logits = clean_model(dataset['x'], attacked_edges)
        clean_acc_attacked = (attacked_logits[dataset['test_mask']].argmax(dim=1) == 
                             dataset['y'][dataset['test_mask']]).float().mean().item()
    
    # Evaluate robust model on attacked data (if available)
    try:
        robust_model.eval()
        with torch.no_grad():
            robust_logits = robust_model(dataset['x'], attacked_edges)
            robust_acc_attacked = (robust_logits[dataset['test_mask']].argmax(dim=1) == 
                                  dataset['y'][dataset['test_mask']]).float().mean().item()
    except Exception as e:
        logger.warning(f"Robust model evaluation failed: {e}")
        robust_acc_attacked = 0.0
    
    # Calculate robustness metrics
    accuracy_drop = clean_acc_clean - clean_acc_attacked
    robustness_improvement = robust_acc_attacked - clean_acc_attacked if robust_acc_attacked > 0 else 0.0
    
    results['adversarial'] = {
        'clean_acc_clean_data': clean_acc_clean,
        'clean_acc_attacked_data': clean_acc_attacked,
        'robust_acc_attacked_data': robust_acc_attacked,
        'accuracy_drop': accuracy_drop,
        'robustness_improvement': robustness_improvement,
        'attack_ratio': 0.1
    }
    
    return results


def run_integration_benchmark(
    dataset: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    """
    Benchmark integrated robustness system with SpotTarget.
    
    Args:
        dataset: Graph dataset
        config: Configuration dict
        device: Computation device
        
    Returns:
        integration_results: End-to-end performance metrics
    """
    logger.info("Running integration benchmark...")
    
    # Create configurations for comparison
    configurations = {
        'baseline': {
            'use_spottarget': False,
            'use_dropedge': False,
            'use_rgnn': False,
            'use_adversarial': False
        },
        'spottarget_only': {
            'use_spottarget': True,
            'use_dropedge': False,
            'use_rgnn': False,
            'use_adversarial': False,
            'delta': 'auto'
        },
        'robustness_only': {
            'use_spottarget': False,
            'use_dropedge': True,
            'use_rgnn': True,
            'use_adversarial': False,
            'dropedge_p': 0.1
        },
        'full_integration': {
            'use_spottarget': True,
            'use_dropedge': True,
            'use_rgnn': True,
            'use_adversarial': False,  # May be unstable
            'delta': 'auto',
            'dropedge_p': 0.1
        }
    }
    
    integration_results = {}
    
    for config_name, config_params in configurations.items():
        logger.info(f"Testing configuration: {config_name}")
        
        try:
            # Create model
            base_model = AttackableGNN(
                input_dim=dataset['num_features'],
                hidden_dim=config['hidden_dim'],
                output_dim=dataset['num_classes']
            )
            
            # Apply robustness modifications
            if config_params.get('use_rgnn', False):
                robustness_config = {
                    'use_rgnn': True,
                    'attention_gating': True,
                    'spectral_norm': True,
                    'attention_dim': config['hidden_dim'] // 2
                }
                model = create_robust_model(base_model, robustness_config)
            else:
                model = base_model
            
            model = model.to(device)
            
            # Setup training
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
            criterion = nn.CrossEntropyLoss()
            
            # Setup SpotTarget if enabled
            spottarget_sampler = None
            if config_params.get('use_spottarget', False):
                avg_deg, degrees = compute_avg_degree(dataset['edge_index'], dataset['num_nodes'])
                delta = avg_deg if config_params['delta'] == 'auto' else config_params['delta']
                
                spottarget_sampler = SpotTargetSampler(
                    edge_index=dataset['edge_index'],
                    train_edge_mask=dataset['edge_splits']['train'],
                    degrees=degrees,
                    delta=delta,
                    verbose=False
                )
            
            # Setup DropEdge if enabled
            dropedge_module = None
            if config_params.get('use_dropedge', False):
                dropedge_module = DropEdge(
                    p_drop=config_params['dropedge_p'],
                    training=True
                )
            
            # Quick training loop
            model.train()
            training_start = time.time()
            
            for epoch in range(config.get('quick_epochs', 5)):
                optimizer.zero_grad()
                
                # Get edge index for this epoch
                current_edge_index = dataset['edge_index']
                
                # Apply SpotTarget filtering
                if spottarget_sampler is not None:
                    # Simulate batch filtering
                    batch_indices = torch.arange(current_edge_index.size(1))
                    current_edge_index = spottarget_sampler.sample_batch(batch_indices)
                
                # Apply DropEdge
                if dropedge_module is not None:
                    current_edge_index = dropedge_module(current_edge_index)
                
                # Forward pass
                logits = model(dataset['x'], current_edge_index)
                loss = criterion(logits[dataset['train_mask']], dataset['y'][dataset['train_mask']])
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            training_time = time.time() - training_start
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                # Clean evaluation
                eval_edge_index = dataset['edge_index']
                if spottarget_sampler is not None:
                    # Use inference-time edge filtering (no T_low edges)
                    batch_indices = torch.arange(eval_edge_index.size(1))
                    eval_edge_index = spottarget_sampler.sample_batch(batch_indices)
                
                clean_logits = model(dataset['x'], eval_edge_index)
                clean_acc = (clean_logits[dataset['test_mask']].argmax(dim=1) == 
                           dataset['y'][dataset['test_mask']]).float().mean().item()
                
                # Attacked evaluation
                attacked_edge_index = generate_edge_attack(eval_edge_index, attack_ratio=0.1)
                attacked_logits = model(dataset['x'], attacked_edge_index)
                attacked_acc = (attacked_logits[dataset['test_mask']].argmax(dim=1) == 
                              dataset['y'][dataset['test_mask']]).float().mean().item()
            
            # Compile results
            integration_results[config_name] = {
                'config': config_params,
                'training_time': training_time,
                'clean_accuracy': clean_acc,
                'attacked_accuracy': attacked_acc,
                'robustness': attacked_acc / clean_acc if clean_acc > 0 else 0.0,
                'success': True
            }
            
            # SpotTarget specific metrics
            if spottarget_sampler is not None:
                stats = spottarget_sampler.get_stats()
                integration_results[config_name]['spottarget_stats'] = stats
            
            logger.info(f"{config_name}: Clean Acc={clean_acc:.4f}, "
                       f"Attacked Acc={attacked_acc:.4f}, Time={training_time:.2f}s")
        
        except Exception as e:
            logger.error(f"Configuration {config_name} failed: {e}")
            integration_results[config_name] = {
                'config': config_params,
                'success': False,
                'error': str(e)
            }
    
    return integration_results


def run_robustness_benchmark(
    config: Dict[str, Any],
    save_dir: str = 'experiments/stage7'
) -> Dict[str, Any]:
    """
    Run complete robustness benchmarking study.
    Following Stage7 Reference §Phase5: benchmark robustness modules.
    
    Args:
        config: Experiment configuration
        save_dir: Directory to save results
        
    Returns:
        benchmark_results: Complete benchmark results
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create adversarial dataset
    logger.info("Creating adversarial dataset...")
    dataset = create_adversarial_dataset(
        num_nodes=config.get('num_nodes', 500),
        num_features=config.get('num_features', 32),
        num_classes=config.get('num_classes', 2),
        edge_prob=config.get('edge_prob', 0.03),
        noise_level=config.get('noise_level', 0.1),
        seed=config.get('seed', 42)
    )
    
    logger.info(f"Dataset created: {dataset['num_nodes']} nodes, {dataset['edge_index'].size(1)} edges")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Run module benchmarks
    logger.info("\n" + "="*50)
    logger.info("ROBUSTNESS MODULE BENCHMARKS")
    logger.info("="*50)
    
    module_results = benchmark_robustness_modules(dataset, config, device)
    
    # Run integration benchmarks
    logger.info("\n" + "="*50)
    logger.info("INTEGRATION BENCHMARKS")
    logger.info("="*50)
    
    integration_results = run_integration_benchmark(dataset, config, device)
    
    # Compile final results
    benchmark_results = {
        'config': config,
        'dataset_info': {
            'num_nodes': dataset['num_nodes'],
            'num_edges': dataset['edge_index'].size(1),
            'num_features': dataset['num_features'],
            'num_classes': dataset['num_classes']
        },
        'module_benchmarks': module_results,
        'integration_benchmarks': integration_results,
        'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
    }
    
    # Save results
    results_file = os.path.join(save_dir, f"robustness_benchmark_{benchmark_results['timestamp']}.json")
    with open(results_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("ROBUSTNESS BENCHMARK SUMMARY")
    logger.info("="*60)
    
    # Module summary
    logger.info("\nModule Performance:")
    if 'dropedge' in module_results:
        dropedge = module_results['dropedge']
        logger.info(f"  DropEdge: {dropedge['overhead_ratio']:.2f}x overhead, "
                   f"deterministic={dropedge['determinism']['is_deterministic']}")
    
    if 'rgnn' in module_results:
        rgnn = module_results['rgnn']
        logger.info(f"  RGNN: {rgnn['overhead']['overhead_ratio']:.2f}x overhead")
    
    if 'adversarial' in module_results:
        adv = module_results['adversarial']
        logger.info(f"  Adversarial: {adv['accuracy_drop']:.4f} acc drop, "
                   f"{adv['robustness_improvement']:.4f} improvement")
    
    # Integration summary
    logger.info("\nIntegration Performance:")
    logger.info(f"{'Config':<20} {'Clean Acc':<12} {'Attacked Acc':<15} {'Robustness':<12} {'Time (s)':<10}")
    logger.info("-" * 75)
    
    for config_name, result in integration_results.items():
        if result.get('success', False):
            logger.info(f"{config_name:<20} {result['clean_accuracy']:<12.4f} "
                       f"{result['attacked_accuracy']:<15.4f} {result['robustness']:<12.4f} "
                       f"{result['training_time']:<10.2f}")
    
    return benchmark_results


def main():
    """Main robustness benchmark script."""
    parser = argparse.ArgumentParser(description='Robustness Benchmarking')
    parser.add_argument('--quick_epochs', type=int, default=5, help='Quick training epochs')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--save_dir', type=str, default='experiments/stage7', help='Save directory')
    
    # Quick mode for testing
    parser.add_argument('--quick', action='store_true', help='Quick mode (smaller dataset)')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'quick_epochs': args.quick_epochs,
        'hidden_dim': args.hidden_dim,
        'learning_rate': args.learning_rate,
        'seed': args.seed,
        'verbose': args.verbose,
        
        # Dataset config
        'num_nodes': 200 if args.quick else 500,
        'num_features': 16 if args.quick else 32,
        'num_classes': 2,
        'edge_prob': 0.05,
        'noise_level': 0.1,
        
        # Robustness config
        'dropedge_p': 0.1,
        'use_rgnn': True,
        'attention_gating': True,
        'spectral_norm': True
    }
    
    logger.info("Starting robustness benchmarking...")
    logger.info(f"Configuration: {config}")
    
    # Run benchmark
    results = run_robustness_benchmark(config, args.save_dir)
    
    logger.info("Robustness benchmarking completed!")


if __name__ == "__main__":
    main()
