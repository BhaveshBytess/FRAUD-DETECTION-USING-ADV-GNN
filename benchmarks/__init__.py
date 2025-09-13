"""
4DBInfer-style Benchmarking Framework for GNN Models
====================================================

This module implements a comprehensive benchmarking framework inspired by the 4DBInfer methodology
for systematically evaluating Graph Neural Network models on fraud detection tasks.

The framework provides:
1. Standardized metrics collection (runtime, memory, throughput, accuracy, PR-AUC, Recall@k)
2. Unified model evaluation under identical conditions
3. Scalability testing across different graph sizes
4. Ablation studies for model components
5. Automated report generation with charts and interpretations

Based on 4DBInfer: A 4D Benchmarking Toolbox for Graph-Centric Predictive Modeling on Relational DBs
"""

import os
import sys
import time
import json
import logging
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import psutil
import GPUtil
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import Config
from src.metrics import compute_metrics
from src.utils import set_seed, setup_logging


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""
    
    # Model selection
    models_to_benchmark: List[str] = None
    
    # Data configuration
    dataset_path: str = "data/ellipticpp.pt"
    test_sizes: List[int] = None  # For scalability testing
    
    # Training configuration
    num_epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 0.001
    
    # Benchmarking configuration
    num_runs: int = 3  # Number of runs for averaging
    warmup_runs: int = 1  # Warmup runs to exclude from timing
    measure_memory: bool = True
    measure_runtime: bool = True
    measure_throughput: bool = True
    
    # Output configuration
    output_dir: str = "benchmarks/results"
    save_detailed_logs: bool = True
    generate_charts: bool = True
    
    # Hardware configuration
    device: str = "auto"  # auto, cpu, cuda
    max_memory_gb: float = 8.0  # Memory limit for experiments
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.models_to_benchmark is None:
            self.models_to_benchmark = [
                "gcn", "graphsage", "gat", "rgcn", "tgn", 
                "hgnn", "tdgnn", "cusp", "hhgtn"
            ]
        
        if self.test_sizes is None:
            self.test_sizes = [1000, 5000, 10000, 25000, 50000]
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    
    # Model identification
    model_name: str
    run_id: int
    dataset_size: int
    
    # Performance metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    recall_at_k: Dict[int, float] = None
    
    # Runtime metrics (in seconds)
    training_time: float = 0.0
    inference_time: float = 0.0
    total_time: float = 0.0
    throughput_samples_per_sec: float = 0.0
    
    # Memory metrics (in MB)
    peak_gpu_memory: float = 0.0
    peak_cpu_memory: float = 0.0
    model_size_mb: float = 0.0
    
    # System metrics
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    
    # Additional info
    num_parameters: int = 0
    num_trainable_parameters: int = 0
    convergence_epoch: int = -1
    final_loss: float = 0.0
    
    def __post_init__(self):
        """Initialize nested dictionaries."""
        if self.recall_at_k is None:
            self.recall_at_k = {}


class MemoryProfiler:
    """Memory usage profiler for benchmarking."""
    
    def __init__(self):
        self.peak_memory = 0
        self.monitoring = False
        
    def start(self):
        """Start memory monitoring."""
        tracemalloc.start()
        self.monitoring = True
        
    def stop(self) -> float:
        """Stop monitoring and return peak memory in MB."""
        if not self.monitoring:
            return 0.0
            
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.monitoring = False
        return peak / 1024 / 1024  # Convert to MB
    
    def get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def get_system_memory(self) -> float:
        """Get current system memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024


class TimeProfiler:
    """Time profiler for benchmarking."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        
    def start(self):
        """Start timing."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.start_time = time.time()
        
    def stop(self) -> float:
        """Stop timing and return elapsed time in seconds."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.end_time = time.time()
        return self.end_time - self.start_time


def get_model_size(model: nn.Module) -> Tuple[float, int, int]:
    """
    Get model size and parameter counts.
    
    Returns:
        Tuple of (size_mb, total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size (4 bytes per float32 parameter)
    size_mb = total_params * 4 / 1024 / 1024
    
    return size_mb, total_params, trainable_params


def compute_recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k_values: List[int]) -> Dict[int, float]:
    """
    Compute Recall@k for different k values.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores/probabilities
        k_values: List of k values to compute recall for
        
    Returns:
        Dictionary mapping k to Recall@k value
    """
    recall_at_k = {}
    
    # Sort by scores in descending order
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_labels = y_true[sorted_indices]
    
    total_positives = np.sum(y_true)
    
    if total_positives == 0:
        return {k: 0.0 for k in k_values}
    
    for k in k_values:
        if k > len(sorted_labels):
            k_actual = len(sorted_labels)
        else:
            k_actual = k
            
        top_k_labels = sorted_labels[:k_actual]
        recall_at_k[k] = np.sum(top_k_labels) / total_positives
    
    return recall_at_k


def get_system_info() -> Dict[str, Any]:
    """Get system information for benchmarking context."""
    info = {
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / 1024**3,
        'python_version': sys.version,
        'torch_version': torch.__version__,
    }
    
    if torch.cuda.is_available():
        info['cuda_available'] = True
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        
        # GPU memory info
        try:
            gpus = GPUtil.getGPUs()
            info['gpu_memory_gb'] = [gpu.memoryTotal / 1024 for gpu in gpus]
        except:
            info['gpu_memory_gb'] = []
    else:
        info['cuda_available'] = False
    
    return info


def setup_device(config: BenchmarkConfig) -> torch.device:
    """Setup computation device based on configuration."""
    if config.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config.device)
    
    logging.info(f"Using device: {device}")
    
    # Set memory fraction if using CUDA
    if device.type == "cuda" and config.max_memory_gb > 0:
        try:
            # Set memory fraction (rough estimate)
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
            memory_fraction = min(config.max_memory_gb / total_memory, 1.0)
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            logging.info(f"Set CUDA memory fraction to {memory_fraction:.2f}")
        except Exception as e:
            logging.warning(f"Could not set memory fraction: {e}")
    
    return device


if __name__ == "__main__":
    # Test the framework components
    logging.basicConfig(level=logging.INFO)
    
    # Test system info
    system_info = get_system_info()
    print("System Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Test memory profiler
    profiler = MemoryProfiler()
    profiler.start()
    
    # Simulate some memory usage
    test_tensor = torch.randn(1000, 1000)
    peak_memory = profiler.stop()
    
    print(f"\nMemory test:")
    print(f"  Peak memory: {peak_memory:.2f} MB")
    print(f"  GPU memory: {profiler.get_gpu_memory():.2f} MB")
    print(f"  System memory: {profiler.get_system_memory():.2f} MB")
    
    # Test time profiler
    timer = TimeProfiler()
    timer.start()
    time.sleep(0.1)  # Simulate work
    elapsed = timer.stop()
    
    print(f"\nTime test:")
    print(f"  Elapsed time: {elapsed:.3f} seconds")
