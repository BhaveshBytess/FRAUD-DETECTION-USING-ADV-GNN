"""
SpotTarget Ablation Experiment
Following Stage7 Spot Target And Robustness Reference §Phase5

Sweep δ values and record metrics to reproduce U-shaped sensitivity curve.
Tests δ ∈ {0, avg_deg/2, avg_deg, avg_deg*2, +∞} as specified in reference.
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, List, Any, Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from spot_target import (
    SpotTargetSampler, 
    compute_avg_degree, 
    compute_default_delta,
    leakage_check,
    setup_spottarget_sampler
)
from training_wrapper import (
    SpotTargetTrainer,
    train_epoch_with_spottarget,
    validate_with_leakage_check
)
from robustness import DropEdge, create_robust_model
from imbalance import ImbalanceHandler, analyze_class_imbalance

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleGNN(nn.Module):
    """Simple GNN for ablation experiments."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Simple MLP layers (can be extended to proper GNN layers)
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """Simple forward pass (can be extended for proper message passing)."""
        return self.layers(x)


def create_synthetic_dataset(
    num_nodes: int = 1000,
    num_features: int = 64, 
    num_classes: int = 2,
    edge_prob: float = 0.02,
    imbalance_ratio: float = 3.0,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    Create synthetic dataset for ablation experiments.
    
    Args:
        num_nodes: Number of nodes
        num_features: Feature dimension
        num_classes: Number of classes
        edge_prob: Edge connection probability
        imbalance_ratio: Class imbalance ratio
        seed: Random seed
        
    Returns:
        dataset: Dict with graph data
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create node features
    x = torch.randn(num_nodes, num_features)
    
    # Create random graph
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if torch.rand(1).item() < edge_prob:
                edges.extend([(i, j), (j, i)])  # Undirected
    
    if len(edges) == 0:
        # Ensure at least some edges
        edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    # Create imbalanced labels
    minority_size = int(num_nodes / (1 + imbalance_ratio))
    majority_size = num_nodes - minority_size
    
    labels = torch.cat([
        torch.zeros(majority_size, dtype=torch.long),  # Majority class
        torch.ones(minority_size, dtype=torch.long)    # Minority class
    ])
    
    # Shuffle labels
    perm = torch.randperm(num_nodes)
    labels = labels[perm]
    
    # Create train/val/test splits
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    indices = torch.randperm(num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    # Create edge splits for SpotTarget
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


def run_single_experiment(
    dataset: Dict[str, torch.Tensor],
    delta: Optional[int],
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, float]:
    """
    Run single ablation experiment with given delta.
    
    Args:
        dataset: Dataset dict
        delta: SpotTarget delta parameter (None for no filtering)
        config: Experiment configuration
        device: Computation device
        
    Returns:
        metrics: Experiment results
    """
    # Move dataset to device
    for key, value in dataset.items():
        if torch.is_tensor(value):
            dataset[key] = value.to(device)
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if torch.is_tensor(subvalue):
                    dataset[key][subkey] = subvalue.to(device)
    
    # Create model
    model = SimpleGNN(
        input_dim=dataset['num_features'],
        hidden_dim=config['hidden_dim'],
        output_dim=dataset['num_classes'],
        num_layers=config['num_layers']
    ).to(device)
    
    # Apply robustness wrappers if enabled
    if config.get('use_robust_model', False):
        model = create_robust_model(model, config)
    
    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Setup imbalance handling
    if config.get('use_imbalance_handling', False):
        imbalance_handler = ImbalanceHandler(config)
        criterion = imbalance_handler.compute_loss_function(
            dataset['y'][dataset['train_mask']], 
            dataset['num_classes']
        )
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Setup SpotTarget if delta is provided
    if delta is not None:
        # Create SpotTarget sampler
        _, degrees = compute_avg_degree(dataset['edge_index'], dataset['num_nodes'])
        sampler = SpotTargetSampler(
            edge_index=dataset['edge_index'],
            train_edge_mask=dataset['edge_splits']['train'],
            degrees=degrees,
            delta=delta,
            verbose=config.get('verbose', False)
        )
        
        sampler_stats = sampler.get_stats()
        logger.info(f"SpotTarget δ={delta}: {sampler_stats['exclusion_rate']:.3f} exclusion rate")
    else:
        sampler = None
        sampler_stats = {'exclusion_rate': 0.0, 'tlow_edges': 0}
    
    # Training loop
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        
        if sampler is not None:
            # Use SpotTarget training
            # Create mock data loader (single batch)
            batch_data = {
                'x': dataset['x'],
                'edge_index': dataset['edge_index'],
                'y': dataset['y'],
                'train_mask': dataset['train_mask'],
                'batch_edge_indices': torch.arange(dataset['edge_index'].size(1))
            }
            
            # Apply SpotTarget filtering
            filtered_batch = sampler.sample_batch(batch_data['batch_edge_indices'])
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(dataset['x'], filtered_batch)
            
            if isinstance(criterion, nn.Module):
                loss = criterion(logits[dataset['train_mask']], dataset['y'][dataset['train_mask']])
            else:
                loss = criterion(logits[dataset['train_mask']], dataset['y'][dataset['train_mask']])
            
            loss.backward()
            optimizer.step()
        else:
            # Standard training without SpotTarget
            optimizer.zero_grad()
            logits = model(dataset['x'], dataset['edge_index'])
            
            if isinstance(criterion, nn.Module):
                loss = criterion(logits[dataset['train_mask']], dataset['y'][dataset['train_mask']])
            else:
                loss = criterion(logits[dataset['train_mask']], dataset['y'][dataset['train_mask']])
            
            loss.backward()
            optimizer.step()
        
        # Compute training accuracy
        with torch.no_grad():
            pred = logits[dataset['train_mask']].argmax(dim=1)
            train_acc = (pred == dataset['y'][dataset['train_mask']]).float().mean().item()
        
        train_losses.append(loss.item())
        train_accuracies.append(train_acc)
        
        # Validation
        model.eval()
        with torch.no_grad():
            # Use leakage-safe inference graph
            inference_edge_index = leakage_check(
                dataset['edge_index'],
                dataset['edge_splits'],
                use_validation_edges=False
            )
            
            val_logits = model(dataset['x'], inference_edge_index)
            val_loss = F.cross_entropy(
                val_logits[dataset['val_mask']], 
                dataset['y'][dataset['val_mask']]
            )
            
            val_pred = val_logits[dataset['val_mask']].argmax(dim=1)
            val_acc = (val_pred == dataset['y'][dataset['val_mask']]).float().mean().item()
        
        val_losses.append(val_loss.item())
        val_accuracies.append(val_acc)
        
        if epoch % 10 == 0 or epoch == config['num_epochs'] - 1:
            logger.info(f"Epoch {epoch}: Train Loss={loss:.4f}, Train Acc={train_acc:.4f}, "
                       f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(dataset['x'], inference_edge_index)
        test_pred = test_logits[dataset['test_mask']].argmax(dim=1)
        test_acc = (test_pred == dataset['y'][dataset['test_mask']]).float().mean().item()
        
        # Per-class accuracy for imbalanced evaluation
        test_labels = dataset['y'][dataset['test_mask']]
        per_class_acc = []
        for class_idx in range(dataset['num_classes']):
            class_mask = test_labels == class_idx
            if class_mask.sum() > 0:
                class_acc = (test_pred[class_mask] == test_labels[class_mask]).float().mean().item()
                per_class_acc.append(class_acc)
            else:
                per_class_acc.append(0.0)
    
    # Compile results
    results = {
        'delta': delta if delta is not None else 'None',
        'final_train_loss': train_losses[-1],
        'final_train_acc': train_accuracies[-1],
        'final_val_loss': val_losses[-1],
        'final_val_acc': val_accuracies[-1],
        'test_acc': test_acc,
        'per_class_acc': per_class_acc,
        'avg_per_class_acc': sum(per_class_acc) / len(per_class_acc),
        'exclusion_rate': sampler_stats['exclusion_rate'] if sampler else 0.0,
        'excluded_edges': sampler_stats['tlow_edges'] if sampler else 0
    }
    
    return results


def run_spottarget_ablation(
    config: Dict[str, Any],
    save_dir: str = 'experiments/stage7'
) -> Dict[str, Any]:
    """
    Run complete SpotTarget ablation study.
    Following Stage7 Reference §Phase5: sweep δ values for U-shaped sensitivity curve.
    
    Args:
        config: Experiment configuration
        save_dir: Directory to save results
        
    Returns:
        ablation_results: Complete ablation results
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create dataset
    logger.info("Creating synthetic dataset...")
    dataset = create_synthetic_dataset(
        num_nodes=config.get('num_nodes', 500),
        num_features=config.get('num_features', 32),
        num_classes=config.get('num_classes', 2),
        edge_prob=config.get('edge_prob', 0.03),
        imbalance_ratio=config.get('imbalance_ratio', 2.0),
        seed=config.get('seed', 42)
    )
    
    # Analyze class imbalance
    imbalance_analysis = analyze_class_imbalance(
        dataset['y'],
        dataset['train_mask'],
        dataset['val_mask'],
        dataset['test_mask']
    )
    
    logger.info(f"Dataset created: {dataset['num_nodes']} nodes, {dataset['edge_index'].size(1)} edges")
    logger.info(f"Class distribution: {imbalance_analysis['train']['class_counts']}")
    
    # Compute average degree for delta values
    avg_deg, _ = compute_avg_degree(dataset['edge_index'], dataset['num_nodes'])
    logger.info(f"Average degree: {avg_deg}")
    
    # Define delta values following Stage7 Reference §Phase5
    delta_values = [
        0,                    # ExcludeAll equivalent
        avg_deg // 2,         # Low threshold
        avg_deg,              # Default (average degree)
        avg_deg * 2,          # High threshold
        None                  # ExcludeNone (no SpotTarget)
    ]
    
    # Add infinity (very large delta) equivalent
    delta_values.insert(-1, avg_deg * 10)  # Very high threshold ~ infinity
    
    logger.info(f"Testing delta values: {delta_values}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Run experiments
    all_results = []
    
    for i, delta in enumerate(delta_values):
        logger.info(f"\n{'='*50}")
        logger.info(f"Experiment {i+1}/{len(delta_values)}: δ = {delta}")
        logger.info(f"{'='*50}")
        
        try:
            results = run_single_experiment(dataset, delta, config, device)
            results['experiment_id'] = i
            results['success'] = True
            all_results.append(results)
            
            logger.info(f"Results: Test Acc={results['test_acc']:.4f}, "
                       f"Avg Per-Class Acc={results['avg_per_class_acc']:.4f}")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            all_results.append({
                'delta': delta,
                'experiment_id': i,
                'success': False,
                'error': str(e)
            })
    
    # Compile final results
    ablation_results = {
        'config': config,
        'dataset_info': {
            'num_nodes': dataset['num_nodes'],
            'num_edges': dataset['edge_index'].size(1),
            'avg_degree': avg_deg,
            'imbalance_analysis': imbalance_analysis
        },
        'delta_values': delta_values,
        'results': all_results,
        'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
    }
    
    # Save results
    results_file = os.path.join(save_dir, f"spottarget_ablation_{ablation_results['timestamp']}.json")
    with open(results_file, 'w') as f:
        json.dump(ablation_results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    # Print summary
    successful_results = [r for r in all_results if r.get('success', False)]
    if successful_results:
        logger.info("\n" + "="*60)
        logger.info("ABLATION SUMMARY")
        logger.info("="*60)
        logger.info(f"{'Delta':<10} {'Test Acc':<10} {'Per-Class Acc':<15} {'Exclusion Rate':<15}")
        logger.info("-" * 60)
        
        for result in successful_results:
            delta_str = str(result['delta'])[:8]
            logger.info(f"{delta_str:<10} {result['test_acc']:<10.4f} "
                       f"{result['avg_per_class_acc']:<15.4f} {result['exclusion_rate']:<15.4f}")
    
    return ablation_results


def main():
    """Main ablation script."""
    parser = argparse.ArgumentParser(description='SpotTarget Ablation Study')
    parser.add_argument('--dataset', type=str, default='synthetic', help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--save_dir', type=str, default='experiments/stage7', help='Save directory')
    
    # Quick mode for testing
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer epochs/nodes)')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'num_epochs': 10 if args.quick else args.epochs,
        'hidden_dim': args.hidden_dim,
        'num_layers': 2,
        'learning_rate': args.learning_rate,
        'seed': args.seed,
        'verbose': args.verbose,
        
        # Dataset config
        'num_nodes': 200 if args.quick else 500,
        'num_features': 16 if args.quick else 32,
        'num_classes': 2,
        'edge_prob': 0.05,
        'imbalance_ratio': 2.0,
        
        # Additional options
        'use_robust_model': False,
        'use_imbalance_handling': False
    }
    
    logger.info("Starting SpotTarget ablation study...")
    logger.info(f"Configuration: {config}")
    
    # Run ablation
    results = run_spottarget_ablation(config, args.save_dir)
    
    logger.info("Ablation study completed!")


if __name__ == "__main__":
    main()
