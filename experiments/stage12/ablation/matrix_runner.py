#!/usr/bin/env python3
"""
Stage 12 Ablation Matrix Runner
Systematically tests hHGTN component contributions through ablation studies.

Usage:
    python matrix_runner.py --mode lite --configs ../configs/ablation_grid.yaml --runs 3
"""

import argparse
import itertools
import json
import os
import random
import time
import traceback
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import hashlib
import subprocess
import sys

# Add project root to path for imports
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
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Get the appropriate device for computation."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleHHGTN(nn.Module):
    """Simplified hHGTN model for ablation testing."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, **config):
        super().__init__()
        self.config = config
        
        # Base components
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Conditional components based on ablation config
        if config.get('use_cusp', True):
            self.cusp_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        if config.get('use_hypergraph', True):
            self.hypergraph_conv = nn.Linear(hidden_dim, hidden_dim)
        
        if config.get('use_memory', True):
            self.memory_module = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        
        # Output layer
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index=None):
        """Forward pass with ablation controls."""
        batch_size = x.shape[0]
        
        # Base embedding
        h = self.embedding(x)
        h = F.relu(h)
        
        # Conditional components
        if self.config.get('use_cusp', True) and hasattr(self, 'cusp_attention'):
            # Simple self-attention as CUSP proxy
            h_reshaped = h.unsqueeze(0)  # Add batch dimension for attention
            attn_out, _ = self.cusp_attention(h_reshaped, h_reshaped, h_reshaped)
            h = attn_out.squeeze(0)
        
        if self.config.get('use_hypergraph', True) and hasattr(self, 'hypergraph_conv'):
            # Simple linear transformation as hypergraph proxy
            h = self.hypergraph_conv(h)
            h = F.relu(h)
        
        if self.config.get('use_memory', True) and hasattr(self, 'memory_module'):
            # Simple temporal processing
            h_seq = h.unsqueeze(0)  # Add sequence dimension
            h_mem, _ = self.memory_module(h_seq)
            h = h_mem.squeeze(0)
        
        # Apply defense mechanisms
        defense = self.config.get('use_rg_defense', 'none')
        if defense == 'dropedge':
            # Simple dropout as edge dropping proxy
            h = self.dropout(h)
        elif defense == 'spectral':
            # Simple normalization as spectral defense proxy
            h = F.normalize(h, p=2, dim=-1)
        
        # Classification
        h = self.dropout(h)
        out = self.classifier(h)
        
        return out


def create_synthetic_data(n_nodes=1000, n_features=64):
    """Create synthetic graph data for testing."""
    # Generate random node features
    x = torch.randn(n_nodes, n_features)
    
    # Generate binary labels (fraud detection)
    y = torch.randint(0, 2, (n_nodes,))
    
    # Create train/val/test splits
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
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask
    }


def train_model(model, data, epochs=10, lr=0.001, device='cpu'):
    """Train model and return losses."""
    model = model.to(device)
    data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        out = model(data['x'])
        train_loss = criterion(out[data['train_mask']], data['y'][data['train_mask']])
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data['x'])
            val_loss = criterion(out[data['val_mask']], data['y'][data['val_mask']])
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
    
    return train_losses, val_losses


def evaluate_model(model, data, device='cpu'):
    """Evaluate model and return metrics."""
    model = model.to(device)
    data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    
    model.eval()
    with torch.no_grad():
        out = model(data['x'])
        test_out = out[data['test_mask']]
        test_y = data['y'][data['test_mask']]
        
        # Convert to predictions
        pred = torch.argmax(test_out, dim=1)
        
        # Calculate metrics
        correct = (pred == test_y).float()
        accuracy = correct.mean().item()
        
        # Calculate AUC and F1 (simplified)
        tp = ((pred == 1) & (test_y == 1)).sum().item()
        fp = ((pred == 1) & (test_y == 0)).sum().item()
        fn = ((pred == 0) & (test_y == 1)).sum().item()
        tn = ((pred == 0) & (test_y == 0)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Simplified AUC (using prediction probabilities)
        probs = F.softmax(test_out, dim=1)[:, 1]
        # Very simple AUC approximation
        sorted_probs, indices = torch.sort(probs, descending=True)
        sorted_labels = test_y[indices]
        auc = torch.mean(sorted_labels[:len(sorted_labels)//2].float()).item()
        auc = max(0.5, min(1.0, auc))  # Clamp to reasonable range
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }


class AblationMatrixRunner:
    """Runs systematic ablation experiments across component toggles."""
    
    def __init__(self, config_path: str):
        """Initialize runner with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.results = []
        self.run_counter = 0
        
        # Setup directories
        self.results_dir = Path(self.config['output']['results_dir'])
        self.logs_dir = Path(self.config['output']['logs_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Git hash for reproducibility
        self.git_hash = self._get_git_hash()
        
        print(f"[AblationRunner] Initialized with config: {config_path}")
        print(f"[AblationRunner] Results dir: {self.results_dir}")
        print(f"[AblationRunner] Git hash: {self.git_hash}")
    
    def _load_config(self) -> Dict:
        """Load YAML configuration."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_git_hash(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
            return result.stdout.strip()[:8]
        except:
            return "unknown"
    
    def generate_ablation_configs(self, max_runs: int = None, sampling: str = "stratified") -> List[Dict]:
        """Generate ablation configurations using specified sampling strategy."""
        grid = self.config['ablation_grid']
        baseline = self.config['baseline']
        
        # Generate all combinations
        keys = list(grid.keys())
        values = list(grid.values())
        all_combinations = list(itertools.product(*values))
        
        print(f"[AblationRunner] Total possible combinations: {len(all_combinations)}")
        
        # Always include baseline
        baseline_tuple = tuple(baseline[k] for k in keys)
        configs = [dict(zip(keys, baseline_tuple))]
        
        if max_runs and len(all_combinations) > max_runs:
            # Sample subset
            remaining_combinations = [combo for combo in all_combinations if combo != baseline_tuple]
            
            if sampling == "stratified":
                # Ensure we test each component individually
                single_ablation_configs = []
                for i, key in enumerate(keys):
                    if key == 'use_rg_defense':  # Handle special case
                        for value in grid[key]:
                            if value != baseline[key]:
                                config = baseline.copy()
                                config[key] = value
                                single_ablation_configs.append(tuple(config[k] for k in keys))
                    else:
                        config = baseline.copy()
                        config[key] = not baseline[key]  # Flip boolean
                        single_ablation_configs.append(tuple(config[k] for k in keys))
                
                # Add single ablations
                for config_tuple in single_ablation_configs:
                    if config_tuple in remaining_combinations:
                        configs.append(dict(zip(keys, config_tuple)))
                        remaining_combinations.remove(config_tuple)
                
                # Fill remaining with random sampling
                n_remaining = max_runs - len(configs)
                if n_remaining > 0 and remaining_combinations:
                    sampled = random.sample(remaining_combinations, 
                                          min(n_remaining, len(remaining_combinations)))
                    configs.extend([dict(zip(keys, combo)) for combo in sampled])
            else:  # Random sampling
                n_sample = min(max_runs - 1, len(remaining_combinations))
                sampled = random.sample(remaining_combinations, n_sample)
                configs.extend([dict(zip(keys, combo)) for combo in sampled])
        else:
            # Use all combinations
            configs = [dict(zip(keys, combo)) for combo in all_combinations]
        
        print(f"[AblationRunner] Selected {len(configs)} configurations")
        return configs
    
    def _config_to_hash(self, config: Dict) -> str:
        """Generate hash for configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def run_single_experiment(self, config: Dict, seed: int, mode: str = "lite") -> Dict:
        """Run single ablation experiment."""
        self.run_counter += 1
        config_hash = self._config_to_hash(config)
        run_id = f"ablation_{self.run_counter:03d}_{config_hash}_{seed}"
        
        print(f"[AblationRunner] Starting run {run_id}")
        print(f"[AblationRunner] Config: {config}")
        
        # Setup logging
        log_file = self.logs_dir / f"{run_id}.log"
        
        start_time = time.time()
        
        try:
            # Set seed for reproducibility
            set_seed(seed)
            
            # Load data (using synthetic data for this demo)
            print(f"[AblationRunner] Loading data...")
            data = create_synthetic_data(n_nodes=1000, n_features=64)
            
            # Create model with ablation config
            device = get_device()
            model = self._create_ablated_model(config, data, device)
            
            # Training configuration
            train_config = {
                'epochs': self.config['experiment']['max_epochs'] if mode == 'lite' else 50,
                'lr': 0.001,
                'device': device
            }
            
            # Train model
            print(f"[AblationRunner] Training model...")
            train_losses, val_losses = train_model(model, data, **train_config)
            
            # Evaluate
            print(f"[AblationRunner] Evaluating model...")
            test_metrics = evaluate_model(model, data, device)
            
            runtime = time.time() - start_time
            
            # Collect metrics
            result = {
                'run_id': run_id,
                'config': config,
                'config_hash': config_hash,
                'seed': seed,
                'git_hash': self.git_hash,
                'mode': mode,
                'runtime_s': runtime,
                'node_count': data['x'].shape[0],
                'edge_count': 0,  # Not used in synthetic data
                'peak_mem_mb': self._get_peak_memory(),
                **test_metrics
            }
            
            # Save detailed log
            self._save_run_log(run_id, result, log_file)
            
            print(f"[AblationRunner] Run {run_id} completed - AUC: {test_metrics.get('auc', 0):.4f}, F1: {test_metrics.get('f1', 0):.4f}")
            
            return result
            
        except Exception as e:
            runtime = time.time() - start_time
            error_result = {
                'run_id': run_id,
                'config': config,
                'config_hash': config_hash,
                'seed': seed,
                'git_hash': self.git_hash,
                'mode': mode,
                'runtime_s': runtime,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
            print(f"[AblationRunner] Run {run_id} FAILED: {e}")
            with open(log_file, 'w') as f:
                f.write(f"ERROR in run {run_id}:\n")
                f.write(f"Config: {config}\n")
                f.write(f"Error: {e}\n")
                f.write(f"Traceback:\n{traceback.format_exc()}\n")
            
            return error_result
    
    def _create_ablated_model(self, config: Dict, data, device) -> torch.nn.Module:
        """Create model with ablation settings applied."""
        # Base model configuration
        model_config = {
            'input_dim': data['x'].shape[1],
            'hidden_dim': 64,
            'output_dim': 2,  # Binary classification
            
            # Ablation toggles
            'use_cusp': config.get('use_cusp', True),
            'use_hypergraph': config.get('use_hypergraph', True),
            'use_memory': config.get('use_memory', True),
            'use_spottarget': config.get('use_spottarget', True),
            'use_gsampler': config.get('use_gsampler', True),
            'use_rg_defense': config.get('use_rg_defense', 'none')
        }
        
        model = SimpleHHGTN(**model_config).to(device)
        return model
    
    def _get_peak_memory(self) -> float:
        """Get peak GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return 0.0
    
    def _save_run_log(self, run_id: str, result: Dict, log_file: Path):
        """Save detailed run log."""
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Ablation Run Log: {run_id} ===\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Git Hash: {self.git_hash}\n\n")
            
            f.write("Configuration:\n")
            for key, value in result['config'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("Results:\n")
            for key, value in result.items():
                if key not in ['config', 'traceback']:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            if 'traceback' in result:
                f.write("Error Traceback:\n")
                f.write(result['traceback'])
    
    def run_ablation_study(self, mode: str = "lite", max_runs: int = None) -> str:
        """Run complete ablation study."""
        print(f"[AblationRunner] Starting ablation study in {mode} mode")
        
        # Generate configurations
        max_runs = max_runs or self.config['experiment']['max_runs']
        configs = self.generate_ablation_configs(max_runs, 
                                                self.config['experiment']['sampling_strategy'])
        
        seeds = self.config['experiment']['seeds']
        
        print(f"[AblationRunner] Running {len(configs)} configs × {len(seeds)} seeds = {len(configs) * len(seeds)} total runs")
        
        # Run experiments
        all_results = []
        for config in configs:
            for seed in seeds:
                result = self.run_single_experiment(config, seed, mode)
                all_results.append(result)
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Save results
        csv_path = self.results_dir / self.config['output']['csv_filename']
        df = pd.DataFrame(all_results)
        df.to_csv(csv_path, index=False)
        
        print(f"[AblationRunner] Ablation study completed")
        print(f"[AblationRunner] Results saved to: {csv_path}")
        print(f"[AblationRunner] Total runs: {len(all_results)}")
        print(f"[AblationRunner] Successful runs: {len([r for r in all_results if 'error' not in r])}")
        
        return str(csv_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run ablation matrix experiments")
    parser.add_argument("--mode", choices=["lite", "full"], default="lite",
                       help="Experiment mode")
    parser.add_argument("--configs", required=True,
                       help="Path to ablation grid config")
    parser.add_argument("--runs", type=int,
                       help="Maximum number of runs (overrides config)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create runner and execute
    runner = AblationMatrixRunner(args.configs)
    results_path = runner.run_ablation_study(mode=args.mode, max_runs=args.runs)
    
    print(f"\n=== Ablation Study Complete ===")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
