#!/usr/bin/env python3
"""
Stage 12 Robustness Runner
Tests hHGTN robustness against adversarial attacks and distribution shifts.

Usage:
    python robustness_runner.py --scenario edge_flips --config ../configs/robustness_configs.yaml
"""

import argparse
import json
import random
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import hashlib
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device():
    """Get appropriate device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleHHGTN(nn.Module):
    """Simplified hHGTN for robustness testing."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, **config):
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        if config.get('use_cusp', True):
            self.cusp_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        if config.get('use_hypergraph', True):
            self.hypergraph_conv = nn.Linear(hidden_dim, hidden_dim)
        
        if config.get('use_memory', True):
            self.memory_module = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index=None):
        h = self.embedding(x)
        h = F.relu(h)
        
        if self.config.get('use_cusp', True) and hasattr(self, 'cusp_attention'):
            h_reshaped = h.unsqueeze(0)
            attn_out, _ = self.cusp_attention(h_reshaped, h_reshaped, h_reshaped)
            h = attn_out.squeeze(0)
        
        if self.config.get('use_hypergraph', True) and hasattr(self, 'hypergraph_conv'):
            h = self.hypergraph_conv(h)
            h = F.relu(h)
        
        if self.config.get('use_memory', True) and hasattr(self, 'memory_module'):
            h_seq = h.unsqueeze(0)
            h_mem, _ = self.memory_module(h_seq)
            h = h_mem.squeeze(0)
        
        # Apply defenses
        defense = self.config.get('use_rg_defense', 'none')
        if defense == 'dropedge':
            h = self.dropout(h)
        elif defense == 'spectral':
            h = F.normalize(h, p=2, dim=-1)
        
        h = self.dropout(h)
        return self.classifier(h)


def create_synthetic_data(n_nodes=1000, n_features=64):
    """Create synthetic graph data."""
    x = torch.randn(n_nodes, n_features)
    y = torch.randint(0, 2, (n_nodes,))
    
    # Create edge index (simple random graph)
    n_edges = min(5000, n_nodes * 3)  # Limit edges for speed
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    
    # Create masks
    perm = torch.randperm(n_nodes)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool) 
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    
    train_mask[perm[:int(0.6 * n_nodes)]] = True
    val_mask[perm[int(0.6 * n_nodes):int(0.8 * n_nodes)]] = True
    test_mask[perm[int(0.8 * n_nodes):]] = True
    
    return {
        'x': x,
        'y': y,
        'edge_index': edge_index,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask
    }


def apply_edge_flips(data, flip_rate=0.01, target_strategy='high_degree'):
    """Apply adversarial edge flips."""
    edge_index = data['edge_index'].clone()
    n_edges = edge_index.shape[1]
    n_flips = int(n_edges * flip_rate)
    
    if target_strategy == 'high_degree':
        # Calculate node degrees
        degrees = torch.zeros(data['x'].shape[0])
        for i in range(data['x'].shape[0]):
            degrees[i] = (edge_index[0] == i).sum() + (edge_index[1] == i).sum()
        
        # Target high-degree nodes
        high_degree_nodes = torch.topk(degrees, k=min(100, len(degrees)//2)).indices
        
        # Flip edges involving high-degree nodes
        target_edges = []
        for node in high_degree_nodes:
            node_edges = torch.where((edge_index[0] == node) | (edge_index[1] == node))[0]
            target_edges.extend(node_edges.tolist())
        
        if target_edges:
            flip_indices = random.sample(target_edges, min(n_flips, len(target_edges)))
        else:
            flip_indices = random.sample(range(n_edges), n_flips)
    else:  # random
        flip_indices = random.sample(range(n_edges), n_flips)
    
    # Apply flips by removing edges and adding random ones
    mask = torch.ones(n_edges, dtype=torch.bool)
    mask[flip_indices] = False
    edge_index = edge_index[:, mask]
    
    # Add random new edges
    new_edges = torch.randint(0, data['x'].shape[0], (2, n_flips))
    edge_index = torch.cat([edge_index, new_edges], dim=1)
    
    data_perturbed = data.copy()
    data_perturbed['edge_index'] = edge_index
    return data_perturbed


def apply_feature_drift(data, shift_magnitude=0.1):
    """Apply feature drift to node features."""
    x_perturbed = data['x'].clone()
    n_features = x_perturbed.shape[1]
    
    # Add Gaussian noise to a subset of features
    affected_features = int(n_features * 0.5)  # Affect 50% of features
    feature_indices = random.sample(range(n_features), affected_features)
    
    for feat_idx in feature_indices:
        noise = torch.randn_like(x_perturbed[:, feat_idx]) * shift_magnitude
        x_perturbed[:, feat_idx] += noise
    
    data_perturbed = data.copy()
    data_perturbed['x'] = x_perturbed
    return data_perturbed


def apply_temporal_shift(data, shift_window=0.2):
    """Apply temporal distribution shift."""
    # Simulate temporal shift by altering the test set distribution
    test_indices = torch.where(data['test_mask'])[0]
    shift_size = int(len(test_indices) * shift_window)
    
    if shift_size > 0:
        # Select subset of test nodes to shift
        shift_indices = test_indices[:shift_size]
        
        # Apply systematic bias to these nodes
        data_perturbed = data.copy()
        x_shifted = data_perturbed['x'].clone()
        
        # Systematic shift: increase feature values for shifted nodes
        x_shifted[shift_indices] += 0.5
        
        # Change some labels to simulate new fraud patterns
        y_shifted = data_perturbed['y'].clone()
        # Flip 20% of shifted labels
        flip_size = max(1, shift_size // 5)
        flip_indices = shift_indices[:flip_size]
        y_shifted[flip_indices] = 1 - y_shifted[flip_indices]
        
        data_perturbed['x'] = x_shifted
        data_perturbed['y'] = y_shifted
        return data_perturbed
    
    return data


def train_and_evaluate(model, data, epochs=5, lr=0.001, device='cpu'):
    """Train model and return evaluation metrics."""
    model = model.to(device)
    data_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data_gpu['x'])
        loss = criterion(out[data_gpu['train_mask']], data_gpu['y'][data_gpu['train_mask']])
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        out = model(data_gpu['x'])
        test_out = out[data_gpu['test_mask']]
        test_y = data_gpu['y'][data_gpu['test_mask']]
        
        pred = torch.argmax(test_out, dim=1)
        correct = (pred == test_y).float()
        accuracy = correct.mean().item()
        
        # Calculate detailed metrics
        tp = ((pred == 1) & (test_y == 1)).sum().item()
        fp = ((pred == 1) & (test_y == 0)).sum().item()
        fn = ((pred == 0) & (test_y == 1)).sum().item()
        tn = ((pred == 0) & (test_y == 0)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Simple AUC approximation
        probs = F.softmax(test_out, dim=1)[:, 1]
        sorted_probs, indices = torch.sort(probs, descending=True)
        sorted_labels = test_y[indices]
        auc = torch.mean(sorted_labels[:len(sorted_labels)//2].float()).item()
        auc = max(0.5, min(1.0, auc))
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }


class RobustnessRunner:
    """Runs robustness experiments."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.results_dir = Path(self.config['output']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[RobustnessRunner] Initialized with config: {config_path}")
    
    def _load_config(self):
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def run_edge_flips_scenario(self):
        """Run adversarial edge flips scenario."""
        print(f"[RobustnessRunner] Running edge flips scenario")
        results = []
        
        for seed in self.config['experiment']['seeds']:
            set_seed(seed)
            data = create_synthetic_data(n_nodes=1000)
            
            for defense_name, defense_config in self.config['defenses'].items():
                # Train baseline model
                model_config = {**self.config['model_config'], **defense_config}
                hidden_dim = model_config.pop('hidden_dim', 64)  # Remove to avoid duplicate
                model = SimpleHHGTN(input_dim=64, hidden_dim=hidden_dim, output_dim=2, **model_config)
                
                baseline_metrics = train_and_evaluate(model, data, 
                                                    epochs=self.config['experiment']['epochs_baseline'])
                
                for flip_rate in self.config['edge_flips']['perturbation_rates']:
                    # Apply edge flips
                    data_perturbed = apply_edge_flips(data, flip_rate)
                    
                    # Evaluate on perturbed data
                    perturbed_metrics = train_and_evaluate(model, data_perturbed,
                                                         epochs=self.config['experiment']['epochs_perturbed'])
                    
                    # Calculate metric drops
                    result = {
                        'scenario': 'edge_flips',
                        'seed': seed,
                        'defense': defense_name,
                        'perturbation_rate': flip_rate,
                        'baseline_accuracy': baseline_metrics['accuracy'],
                        'baseline_f1': baseline_metrics['f1'],
                        'baseline_auc': baseline_metrics['auc'],
                        'perturbed_accuracy': perturbed_metrics['accuracy'],
                        'perturbed_f1': perturbed_metrics['f1'],
                        'perturbed_auc': perturbed_metrics['auc'],
                        'accuracy_drop': baseline_metrics['accuracy'] - perturbed_metrics['accuracy'],
                        'f1_drop': baseline_metrics['f1'] - perturbed_metrics['f1'],
                        'auc_drop': baseline_metrics['auc'] - perturbed_metrics['auc']
                    }
                    results.append(result)
                    
                    print(f"[RobustnessRunner] {defense_name} @ {flip_rate*100:.1f}% flips: F1 drop = {result['f1_drop']:.3f}")
        
        # Save results
        output_file = self.results_dir / 'edge_flips_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_feature_drift_scenario(self):
        """Run feature drift scenario."""
        print(f"[RobustnessRunner] Running feature drift scenario")
        results = []
        
        for seed in self.config['experiment']['seeds']:
            set_seed(seed)
            data = create_synthetic_data(n_nodes=1000)
            
            for defense_name, defense_config in self.config['defenses'].items():
                model_config = {**self.config['model_config'], **defense_config}
                hidden_dim = model_config.pop('hidden_dim', 64)  # Remove to avoid duplicate  
                model = SimpleHHGTN(input_dim=64, hidden_dim=hidden_dim, output_dim=2, **model_config)
                
                baseline_metrics = train_and_evaluate(model, data,
                                                    epochs=self.config['experiment']['epochs_baseline'])
                
                for shift_mag in self.config['feature_drift']['shift_magnitudes']:
                    data_perturbed = apply_feature_drift(data, shift_mag)
                    perturbed_metrics = train_and_evaluate(model, data_perturbed,
                                                         epochs=self.config['experiment']['epochs_perturbed'])
                    
                    result = {
                        'scenario': 'feature_drift',
                        'seed': seed,
                        'defense': defense_name,
                        'shift_magnitude': shift_mag,
                        'baseline_accuracy': baseline_metrics['accuracy'],
                        'baseline_f1': baseline_metrics['f1'],
                        'baseline_auc': baseline_metrics['auc'],
                        'perturbed_accuracy': perturbed_metrics['accuracy'],
                        'perturbed_f1': perturbed_metrics['f1'],
                        'perturbed_auc': perturbed_metrics['auc'],
                        'accuracy_drop': baseline_metrics['accuracy'] - perturbed_metrics['accuracy'],
                        'f1_drop': baseline_metrics['f1'] - perturbed_metrics['f1'],
                        'auc_drop': baseline_metrics['auc'] - perturbed_metrics['auc']
                    }
                    results.append(result)
                    
                    print(f"[RobustnessRunner] {defense_name} @ drift {shift_mag}: F1 drop = {result['f1_drop']:.3f}")
        
        output_file = self.results_dir / 'feature_drift_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_temporal_shift_scenario(self):
        """Run temporal distribution shift scenario."""
        print(f"[RobustnessRunner] Running temporal shift scenario")
        results = []
        
        for seed in self.config['experiment']['seeds']:
            set_seed(seed)
            data = create_synthetic_data(n_nodes=1000)
            
            for defense_name, defense_config in self.config['defenses'].items():
                model_config = {**self.config['model_config'], **defense_config}
                hidden_dim = model_config.pop('hidden_dim', 64)  # Remove to avoid duplicate
                model = SimpleHHGTN(input_dim=64, hidden_dim=hidden_dim, output_dim=2, **model_config)
                
                baseline_metrics = train_and_evaluate(model, data,
                                                    epochs=self.config['experiment']['epochs_baseline'])
                
                data_perturbed = apply_temporal_shift(data, self.config['temporal_shift']['shift_window'])
                perturbed_metrics = train_and_evaluate(model, data_perturbed,
                                                     epochs=self.config['experiment']['epochs_perturbed'])
                
                result = {
                    'scenario': 'temporal_shift',
                    'seed': seed,
                    'defense': defense_name,
                    'shift_window': self.config['temporal_shift']['shift_window'],
                    'baseline_accuracy': baseline_metrics['accuracy'],
                    'baseline_f1': baseline_metrics['f1'],
                    'baseline_auc': baseline_metrics['auc'],
                    'perturbed_accuracy': perturbed_metrics['accuracy'],
                    'perturbed_f1': perturbed_metrics['f1'],
                    'perturbed_auc': perturbed_metrics['auc'],
                    'accuracy_drop': baseline_metrics['accuracy'] - perturbed_metrics['accuracy'],
                    'f1_drop': baseline_metrics['f1'] - perturbed_metrics['f1'],
                    'auc_drop': baseline_metrics['auc'] - perturbed_metrics['auc']
                }
                results.append(result)
                
                print(f"[RobustnessRunner] {defense_name} @ temporal shift: F1 drop = {result['f1_drop']:.3f}")
        
        output_file = self.results_dir / 'temporal_shift_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_all_scenarios(self):
        """Run all robustness scenarios."""
        print(f"[RobustnessRunner] Running all robustness scenarios")
        
        edge_results = self.run_edge_flips_scenario()
        drift_results = self.run_feature_drift_scenario()
        temporal_results = self.run_temporal_shift_scenario()
        
        # Create summary
        summary = {
            'edge_flips': {
                'total_tests': len(edge_results),
                'avg_f1_drop': np.mean([r['f1_drop'] for r in edge_results]),
                'max_f1_drop': np.max([r['f1_drop'] for r in edge_results])
            },
            'feature_drift': {
                'total_tests': len(drift_results),
                'avg_f1_drop': np.mean([r['f1_drop'] for r in drift_results]),
                'max_f1_drop': np.max([r['f1_drop'] for r in drift_results])
            },
            'temporal_shift': {
                'total_tests': len(temporal_results),
                'avg_f1_drop': np.mean([r['f1_drop'] for r in temporal_results]),
                'max_f1_drop': np.max([r['f1_drop'] for r in temporal_results])
            }
        }
        
        summary_file = self.results_dir / self.config['output']['summary_file']
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[RobustnessRunner] All scenarios completed. Summary saved to {summary_file}")
        return summary


def main():
    parser = argparse.ArgumentParser(description="Run robustness experiments")
    parser.add_argument("--scenario", choices=["edge_flips", "feature_drift", "temporal_shift", "all"],
                       default="all", help="Robustness scenario to run")
    parser.add_argument("--config", required=True, help="Path to robustness config")
    
    args = parser.parse_args()
    
    runner = RobustnessRunner(args.config)
    
    if args.scenario == "edge_flips":
        runner.run_edge_flips_scenario()
    elif args.scenario == "feature_drift":
        runner.run_feature_drift_scenario()
    elif args.scenario == "temporal_shift":
        runner.run_temporal_shift_scenario()
    else:  # all
        runner.run_all_scenarios()
    
    print(f"\n=== Robustness Testing Complete ===")


if __name__ == "__main__":
    main()
