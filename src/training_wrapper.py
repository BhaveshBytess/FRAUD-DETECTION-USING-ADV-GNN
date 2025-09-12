"""
Training Wrapper Integration for SpotTarget
Following Stage7 Spot Target And Robustness Reference §Phase2

Implements train_epoch_with_spottarget that automatically applies SpotTarget filtering 
on each mini-batch before model.forward with minimal changes to model API.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Callable, Tuple, List
import logging
import time
from dataclasses import dataclass

from spot_target import SpotTargetSampler, leakage_check, load_stage7_config

logger = logging.getLogger(__name__)


@dataclass
class TrainingStats:
    """Statistics for training epoch."""
    epoch_loss: float
    epoch_accuracy: float
    edges_filtered_total: int
    edges_original_total: int
    filtering_overhead_ms: float
    batch_count: int


class SpotTargetTrainingWrapper:
    """
    Training wrapper that integrates SpotTarget filtering into training loops.
    Following Stage7 Reference §Phase2: wrapper ensures minimal changes to model API.
    """
    
    def __init__(
        self,
        model: nn.Module,
        sampler: SpotTargetSampler,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize training wrapper.
        
        Args:
            model: The GNN model to train
            sampler: SpotTarget sampler for edge filtering
            config: Stage7 configuration
        """
        self.model = model
        self.sampler = sampler
        self.config = config or load_stage7_config()
        self.verbose = self.config.get('verbose', False)
        
        # Statistics tracking
        self.reset_stats()
    
    def reset_stats(self):
        """Reset training statistics."""
        self.total_filtered_edges = 0
        self.total_original_edges = 0
        self.total_filtering_time = 0.0
        self.batch_count = 0
    
    def apply_spottarget_to_batch(
        self, 
        batch_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply SpotTarget filtering to a mini-batch.
        Following Stage7 Reference §Phase2: operate on mini-batch subgraph only.
        
        Args:
            batch_data: Dict containing 'edge_index', 'batch_edge_indices', etc.
            
        Returns:
            filtered_batch_data: Batch with filtered edge_index
        """
        start_time = time.time()
        
        # Extract batch edge indices
        batch_edge_indices = batch_data.get('batch_edge_indices')
        if batch_edge_indices is None:
            # If no specific batch indices, use all edges
            batch_edge_indices = torch.arange(batch_data['edge_index'].size(1))
        
        # Apply SpotTarget filtering
        original_count = len(batch_edge_indices)
        filtered_edge_index = self.sampler.sample_batch(batch_edge_indices)
        filtered_count = filtered_edge_index.size(1)
        
        # Update statistics
        self.total_original_edges += original_count
        self.total_filtered_edges += (original_count - filtered_count)
        self.batch_count += 1
        
        # Create filtered batch data
        filtered_batch = batch_data.copy()
        filtered_batch['edge_index'] = filtered_edge_index
        
        # Track timing
        filtering_time = (time.time() - start_time) * 1000  # ms
        self.total_filtering_time += filtering_time
        
        if self.verbose:
            logger.info(f"Batch {self.batch_count}: filtered {original_count - filtered_count}/{original_count} edges ({filtering_time:.2f}ms)")
        
        return filtered_batch
    
    def train_step(
        self,
        batch_data: Dict[str, torch.Tensor],
        optimizer: optim.Optimizer,
        criterion: Callable,
        **forward_kwargs
    ) -> Tuple[float, float]:
        """
        Single training step with SpotTarget filtering.
        
        Args:
            batch_data: Batch data dict
            optimizer: Optimizer
            criterion: Loss function
            **forward_kwargs: Additional args for model.forward
            
        Returns:
            (loss, accuracy): Training metrics for this batch
        """
        # Apply SpotTarget filtering
        filtered_batch = self.apply_spottarget_to_batch(batch_data)
        
        # Forward pass with filtered edges
        self.model.train()
        logits = self.model(
            x=filtered_batch['x'],
            edge_index=filtered_batch['edge_index'],
            **forward_kwargs
        )
        
        # Compute loss on training targets only
        train_mask = filtered_batch.get('train_mask')
        if train_mask is not None:
            loss = criterion(logits[train_mask], filtered_batch['y'][train_mask])
            
            # Compute accuracy
            with torch.no_grad():
                pred = logits[train_mask].argmax(dim=1)
                accuracy = (pred == filtered_batch['y'][train_mask]).float().mean().item()
        else:
            # No mask - use all nodes
            loss = criterion(logits, filtered_batch['y'])
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                accuracy = (pred == filtered_batch['y']).float().mean().item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item(), accuracy


def train_epoch_with_spottarget(
    model: nn.Module,
    train_loader: Any,  # DataLoader or similar
    optimizer: optim.Optimizer,
    criterion: Callable,
    sampler: SpotTargetSampler,
    device: torch.device = None,
    config: Optional[Dict[str, Any]] = None
) -> TrainingStats:
    """
    Train one epoch with SpotTarget filtering.
    Following Stage7 Reference §Phase2: automatic SpotTarget filtering per mini-batch.
    
    Args:
        model: GNN model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        sampler: SpotTarget sampler
        device: Device for computation
        config: Stage7 configuration
        
    Returns:
        TrainingStats: Epoch training statistics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    wrapper = SpotTargetTrainingWrapper(model, sampler, config)
    
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    num_batches = 0
    
    model.train()
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        if hasattr(batch, 'to'):
            batch = batch.to(device)
        elif isinstance(batch, dict):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Training step with SpotTarget
        batch_loss, batch_accuracy = wrapper.train_step(batch, optimizer, criterion)
        
        epoch_loss += batch_loss
        epoch_accuracy += batch_accuracy
        num_batches += 1
    
    # Compute epoch statistics
    epoch_time = (time.time() - start_time) * 1000  # ms
    
    stats = TrainingStats(
        epoch_loss=epoch_loss / max(num_batches, 1),
        epoch_accuracy=epoch_accuracy / max(num_batches, 1),
        edges_filtered_total=wrapper.total_filtered_edges,
        edges_original_total=wrapper.total_original_edges,
        filtering_overhead_ms=wrapper.total_filtering_time,
        batch_count=wrapper.batch_count
    )
    
    if wrapper.verbose:
        logger.info(f"Epoch complete: loss={stats.epoch_loss:.4f}, acc={stats.epoch_accuracy:.4f}")
        logger.info(f"Filtering: {stats.edges_filtered_total}/{stats.edges_original_total} edges excluded")
        logger.info(f"Overhead: {stats.filtering_overhead_ms:.2f}ms ({stats.filtering_overhead_ms/epoch_time*100:.1f}% of epoch)")
    
    return stats


def validate_with_leakage_check(
    model: nn.Module,
    val_loader: Any,
    criterion: Callable,
    edge_splits: Dict[str, torch.Tensor],
    original_edge_index: torch.Tensor,
    device: torch.device = None,
    exclude_validation: bool = False
) -> Tuple[float, float]:
    """
    Validation with leakage-safe inference graph.
    Following Stage7 Reference §Phase2: ensure inference uses leakage_check.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        edge_splits: Edge split masks for leakage check
        original_edge_index: Original edge index before filtering
        device: Computation device
        exclude_validation: Whether to exclude validation edges too
        
    Returns:
        (val_loss, val_accuracy): Validation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create leakage-safe inference graph
    inference_edge_index = leakage_check(
        edge_index=original_edge_index,
        edge_splits=edge_splits,
        use_validation_edges=exclude_validation,
        strict_mode=True
    )
    
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            if hasattr(batch, 'to'):
                batch = batch.to(device)
            elif isinstance(batch, dict):
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Use leakage-safe edge index for inference
            batch_inference = batch.copy() if isinstance(batch, dict) else batch
            if isinstance(batch_inference, dict):
                batch_inference['edge_index'] = inference_edge_index
            
            # Forward pass
            logits = model(
                x=batch_inference['x'], 
                edge_index=batch_inference['edge_index']
            )
            
            # Compute validation metrics
            val_mask = batch.get('val_mask')
            if val_mask is not None:
                loss = criterion(logits[val_mask], batch['y'][val_mask])
                pred = logits[val_mask].argmax(dim=1)
                accuracy = (pred == batch['y'][val_mask]).float().mean().item()
            else:
                loss = criterion(logits, batch['y'])
                pred = logits.argmax(dim=1)
                accuracy = (pred == batch['y']).float().mean().item()
            
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_accuracy = total_accuracy / max(num_batches, 1)
    
    logger.info(f"Validation (leakage-safe): loss={avg_loss:.4f}, acc={avg_accuracy:.4f}")
    
    return avg_loss, avg_accuracy


class SpotTargetTrainer:
    """
    Complete trainer class with SpotTarget integration.
    Following Stage7 Reference §Phase2: drop-in wrapper for existing training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        edge_index: torch.Tensor,
        edge_splits: Dict[str, torch.Tensor],
        num_nodes: int,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize SpotTarget trainer.
        
        Args:
            model: GNN model
            edge_index: Full edge index
            edge_splits: Train/test/val edge masks
            num_nodes: Number of nodes
            config: Stage7 configuration
        """
        self.model = model
        self.edge_index = edge_index
        self.edge_splits = edge_splits
        self.num_nodes = num_nodes
        self.config = config or load_stage7_config()
        
        # Initialize SpotTarget sampler
        from spot_target import setup_spottarget_sampler
        self.sampler = setup_spottarget_sampler(
            edge_index=edge_index,
            train_edge_mask=edge_splits['train'],
            num_nodes=num_nodes,
            config=self.config
        )
        
        logger.info(f"SpotTarget trainer initialized with δ={self.sampler.delta}")
    
    def train_epoch(
        self,
        train_loader: Any,
        optimizer: optim.Optimizer,
        criterion: Callable,
        device: torch.device = None
    ) -> TrainingStats:
        """Train one epoch with SpotTarget."""
        return train_epoch_with_spottarget(
            model=self.model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            sampler=self.sampler,
            device=device,
            config=self.config
        )
    
    def validate(
        self,
        val_loader: Any,
        criterion: Callable,
        device: torch.device = None,
        exclude_validation: bool = False
    ) -> Tuple[float, float]:
        """Validate with leakage-safe inference."""
        return validate_with_leakage_check(
            model=self.model,
            val_loader=val_loader,
            criterion=criterion,
            edge_splits=self.edge_splits,
            original_edge_index=self.edge_index,
            device=device,
            exclude_validation=exclude_validation
        )
    
    def get_sampler_stats(self) -> Dict[str, Any]:
        """Get SpotTarget sampler statistics."""
        return self.sampler.get_stats()


# Utility functions for integration with existing pipelines
def create_spottarget_trainer(
    model: nn.Module,
    graph_data: Dict[str, torch.Tensor],
    config_path: Optional[str] = None
) -> SpotTargetTrainer:
    """
    Factory function to create SpotTarget trainer from graph data.
    
    Args:
        model: GNN model
        graph_data: Dict with 'edge_index', 'edge_splits', 'num_nodes'
        config_path: Optional path to config file
        
    Returns:
        SpotTargetTrainer instance
    """
    config = load_stage7_config() if config_path is None else None
    
    return SpotTargetTrainer(
        model=model,
        edge_index=graph_data['edge_index'],
        edge_splits=graph_data['edge_splits'],
        num_nodes=graph_data['num_nodes'],
        config=config
    )


def benchmark_filtering_overhead(
    sampler: SpotTargetSampler,
    batch_sizes: List[int] = [32, 64, 128, 256, 512],
    num_trials: int = 10
) -> Dict[str, List[float]]:
    """
    Benchmark SpotTarget filtering overhead.
    Following Stage7 Reference §Phase2: overhead should be O(|B|) and negligible.
    
    Args:
        sampler: SpotTarget sampler
        batch_sizes: Batch sizes to test
        num_trials: Number of trials per batch size
        
    Returns:
        timing_results: Dict with timing statistics
    """
    results = {'batch_sizes': batch_sizes, 'mean_times': [], 'std_times': []}
    
    total_edges = sampler.edge_index.size(1)
    
    for batch_size in batch_sizes:
        times = []
        
        for _ in range(num_trials):
            # Random batch indices
            batch_indices = torch.randperm(total_edges)[:batch_size]
            
            # Time filtering
            start_time = time.time()
            _ = sampler.sample_batch(batch_indices)
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)  # ms
        
        mean_time = sum(times) / len(times)
        std_time = (sum([(t - mean_time)**2 for t in times]) / len(times))**0.5
        
        results['mean_times'].append(mean_time)
        results['std_times'].append(std_time)
        
        logger.info(f"Batch size {batch_size}: {mean_time:.3f}±{std_time:.3f}ms")
    
    return results
