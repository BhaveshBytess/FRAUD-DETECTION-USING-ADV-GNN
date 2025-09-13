#!/usr/bin/env python3
"""
Stage 12 Scalability Runner
Tests hHGTN performance across different graph sizes and configurations.

Usage:
    python scalability_runner.py --config ../configs/scalability_configs.yaml
"""

import argparse
import json
import os
import random
import time
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
import matplotlib.pyplot as plt
import psutil
import gc


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
    """Simplified hHGTN model for scalability testing."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, **config):
        super().__init__()
        self.config = config
        
        # Base components
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Conditional components based on config
        if config.get('use_cusp', True):
            self.cusp_attention = nn.MultiheadAttention(hidden_dim, num_heads=config.get('num_heads', 4), batch_first=True)
        
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
        
        # Classification
        h = self.dropout(h)
        out = self.classifier(h)
        
        return out


def create_synthetic_data(n_nodes=1000, n_features=64, edge_density=0.1):
    """Create synthetic graph data for scalability testing."""
    # Generate random node features
    x = torch.randn(n_nodes, n_features)
    
    # Generate binary labels (fraud detection)
    y = torch.randint(0, 2, (n_nodes,))
    
    # Generate edges based on density
    n_edges = int(n_nodes * (n_nodes - 1) * edge_density / 2)
    edge_list = []
    
    for _ in range(n_edges):
        src = random.randint(0, n_nodes - 1)
        dst = random.randint(0, n_nodes - 1)
        if src != dst:
            edge_list.append([src, dst])
            edge_list.append([dst, src])  # Undirected
    
    edge_index = torch.tensor(edge_list).t().contiguous() if edge_list else torch.zeros((2, 0), dtype=torch.long)
    
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
        'edge_index': edge_index,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'num_nodes': n_nodes,
        'num_edges': edge_index.shape[1]
    }


class MemoryMonitor:
    """Monitor memory usage during training."""
    
    def __init__(self):
        self.peak_memory = 0
        self.gpu_peak_memory = 0
        
    def update(self):
        """Update memory usage statistics."""
        # CPU memory
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)
        
        # GPU memory
        if torch.cuda.is_available():
            current_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            self.gpu_peak_memory = max(self.gpu_peak_memory, current_gpu_memory)
    
    def reset(self):
        """Reset monitoring."""
        self.peak_memory = 0
        self.gpu_peak_memory = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def train_epoch(model, data, optimizer, criterion, device, monitor):
    """Train for one epoch and return metrics."""
    model.train()
    model = model.to(device)
    data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    
    start_time = time.time()
    
    optimizer.zero_grad()
    monitor.update()
    
    out = model(data['x'])
    loss = criterion(out[data['train_mask']], data['y'][data['train_mask']])
    
    monitor.update()
    loss.backward()
    monitor.update()
    
    optimizer.step()
    monitor.update()
    
    epoch_time = time.time() - start_time
    throughput = data['num_nodes'] / epoch_time  # nodes per second
    
    return {
        'epoch_time': epoch_time,
        'throughput': throughput,
        'loss': loss.item()
    }


class ScalabilityRunner:
    """Runs scalability experiments across different graph sizes."""
    
    def __init__(self, config_path: str):
        """Initialize runner with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.results = []
        
        # Setup directories
        self.results_dir = Path(self.config['output']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Git hash for reproducibility
        self.git_hash = self._get_git_hash()
        
        print(f"[ScalabilityRunner] Initialized with config: {config_path}")
        print(f"[ScalabilityRunner] Results dir: {self.results_dir}")
    
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
    
    def run_scalability_test(self, n_nodes: int, batch_size: int, fanout: int, seed: int) -> Dict:
        """Run single scalability test."""
        test_id = f"scale_{n_nodes}n_{batch_size}b_{fanout}f_{seed}s"
        print(f"[ScalabilityRunner] Running test {test_id}")
        
        # Set seed
        set_seed(seed)
        
        # Create data
        data = create_synthetic_data(n_nodes=n_nodes, n_features=64)
        
        # Create model
        device = get_device()
        model_config = self.config['scalability_tests']['model_config'].copy()
        model_config['batch_size'] = batch_size
        model_config['fanout'] = fanout
        
        hidden_dim = model_config.pop('hidden_dim', 64)  # Remove to avoid duplicate
        model = SimpleHHGTN(input_dim=64, hidden_dim=hidden_dim, output_dim=2, **model_config)
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        monitor = MemoryMonitor()
        
        # Training loop
        epochs = self.config['experiment']['epochs_per_test']
        epoch_metrics = []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            monitor.reset()
            epoch_result = train_epoch(model, data, optimizer, criterion, device, monitor)
            epoch_result['peak_cpu_memory_mb'] = monitor.peak_memory
            epoch_result['peak_gpu_memory_mb'] = monitor.gpu_peak_memory
            epoch_metrics.append(epoch_result)
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        total_time = time.time() - start_time
        
        # Aggregate metrics
        avg_epoch_time = np.mean([m['epoch_time'] for m in epoch_metrics])
        avg_throughput = np.mean([m['throughput'] for m in epoch_metrics])
        peak_cpu_memory = max([m['peak_cpu_memory_mb'] for m in epoch_metrics])
        peak_gpu_memory = max([m['peak_gpu_memory_mb'] for m in epoch_metrics])
        
        result = {
            'test_id': test_id,
            'n_nodes': n_nodes,
            'n_edges': data['num_edges'],
            'batch_size': batch_size,
            'fanout': fanout,
            'seed': seed,
            'epochs': epochs,
            'total_time_s': total_time,
            'avg_epoch_time_s': avg_epoch_time,
            'avg_throughput_nodes_per_s': avg_throughput,
            'peak_cpu_memory_mb': peak_cpu_memory,
            'peak_gpu_memory_mb': peak_gpu_memory,
            'git_hash': self.git_hash,
            'device': str(device)
        }
        
        print(f"[ScalabilityRunner] Test {test_id} completed - "
              f"Avg epoch: {avg_epoch_time:.3f}s, Throughput: {avg_throughput:.1f} nodes/s, "
              f"Peak mem: {peak_cpu_memory:.1f}MB")
        
        return result
    
    def run_full_scalability_study(self) -> str:
        """Run complete scalability study."""
        print(f"[ScalabilityRunner] Starting scalability study")
        
        config = self.config['scalability_tests']
        node_counts = config['node_counts']
        batch_sizes = config['batch_sizes']
        fanouts = config['fanouts']
        seeds = self.config['experiment']['seeds']
        
        # Generate test combinations
        test_combinations = []
        for n_nodes in node_counts:
            for batch_size in batch_sizes:
                for fanout in fanouts:
                    for seed in seeds:
                        test_combinations.append((n_nodes, batch_size, fanout, seed))
        
        print(f"[ScalabilityRunner] Running {len(test_combinations)} test combinations")
        
        all_results = []
        for i, (n_nodes, batch_size, fanout, seed) in enumerate(test_combinations):
            try:
                result = self.run_scalability_test(n_nodes, batch_size, fanout, seed)
                all_results.append(result)
                
                print(f"[ScalabilityRunner] Progress: {i+1}/{len(test_combinations)} tests completed")
                
            except Exception as e:
                print(f"[ScalabilityRunner] Test failed: {e}")
                error_result = {
                    'test_id': f"scale_{n_nodes}n_{batch_size}b_{fanout}f_{seed}s",
                    'n_nodes': n_nodes,
                    'batch_size': batch_size,
                    'fanout': fanout,
                    'seed': seed,
                    'error': str(e)
                }
                all_results.append(error_result)
        
        # Save results
        csv_path = self.results_dir / self.config['output']['csv_filename']
        df = pd.DataFrame(all_results)
        df.to_csv(csv_path, index=False)
        
        print(f"[ScalabilityRunner] Scalability study completed")
        print(f"[ScalabilityRunner] Results saved to: {csv_path}")
        print(f"[ScalabilityRunner] Total tests: {len(all_results)}")
        print(f"[ScalabilityRunner] Successful tests: {len([r for r in all_results if 'error' not in r])}")
        
        return str(csv_path)


def create_scalability_plots(results_csv: str, output_dir: Path):
    """Create scalability visualization plots."""
    df = pd.read_csv(results_csv)
    
    # Filter successful runs
    df_success = df[df['error'].isna()] if 'error' in df.columns else df
    
    if len(df_success) == 0:
        print("[ScalabilityPlots] No successful runs to plot")
        return
    
    plt.style.use('default')
    
    # Plot 1: Runtime vs Node Count
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Group by node count and plot averages
    node_groups = df_success.groupby('n_nodes').agg({
        'avg_epoch_time_s': ['mean', 'std'],
        'peak_cpu_memory_mb': ['mean', 'std']
    }).round(4)
    
    node_counts = node_groups.index
    epoch_times_mean = node_groups[('avg_epoch_time_s', 'mean')]
    epoch_times_std = node_groups[('avg_epoch_time_s', 'std')]
    memory_mean = node_groups[('peak_cpu_memory_mb', 'mean')]
    memory_std = node_groups[('peak_cpu_memory_mb', 'std')]
    
    ax1.errorbar(node_counts, epoch_times_mean, yerr=epoch_times_std, 
                marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Average Epoch Time (seconds)')
    ax1.set_title('Scalability: Runtime vs Graph Size')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    ax2.errorbar(node_counts, memory_mean, yerr=memory_std,
                marker='s', capsize=5, capthick=2, linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Peak Memory Usage (MB)')
    ax2.set_title('Scalability: Memory vs Graph Size')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scalability_runtime_memory.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Throughput vs Node Count
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    throughput_groups = df_success.groupby('n_nodes')['avg_throughput_nodes_per_s'].agg(['mean', 'std'])
    
    ax.errorbar(throughput_groups.index, throughput_groups['mean'], 
               yerr=throughput_groups['std'], marker='d', capsize=5, 
               capthick=2, linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Throughput (nodes/second)')
    ax.set_title('Scalability: Throughput vs Graph Size')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.savefig(output_dir / 'scalability_throughput.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[ScalabilityPlots] Plots saved to {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run scalability experiments")
    parser.add_argument("--config", required=True, help="Path to scalability config")
    
    args = parser.parse_args()
    
    # Create runner and execute
    runner = ScalabilityRunner(args.config)
    results_path = runner.run_full_scalability_study()
    
    # Create plots
    create_scalability_plots(results_path, runner.results_dir)
    
    print(f"\n=== Scalability Study Complete ===")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
