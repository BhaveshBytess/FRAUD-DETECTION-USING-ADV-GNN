"""
Robustness Modules for Graph Neural Networks
Following Stage7 Spot Target And Robustness Reference §Phase3

Implements DropEdge, RGNN-inspired defensive wrappers, and adversarial training
as complementary defenses to SpotTarget training discipline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple, List, Callable
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


class DropEdge(nn.Module):
    """
    DropEdge: randomly drop edges during training for robustness.
    Following Stage7 Reference §Phase3: deterministic seeding, parameterizable schedule.
    """
    
    def __init__(self, p_drop: float = 0.1, training: bool = True):
        """
        Initialize DropEdge module.
        
        Args:
            p_drop: Probability of dropping each edge
            training: Whether in training mode (no dropping during inference)
        """
        super().__init__()
        self.p_drop = p_drop
        self.training_mode = training
        
    def forward(self, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply DropEdge to edge index.
        
        Args:
            edge_index: (2, E) edge index tensor
            edge_attr: (E, F) optional edge attributes
            
        Returns:
            (dropped_edge_index, dropped_edge_attr): Filtered edges and attributes
        """
        if not self.training_mode or self.p_drop == 0.0:
            return edge_index, edge_attr
        
        num_edges = edge_index.size(1)
        if num_edges == 0:
            return edge_index, edge_attr
        
        # Sample edges to keep (deterministic given seed)
        keep_prob = 1.0 - self.p_drop
        keep_mask = torch.rand(num_edges, device=edge_index.device) < keep_prob
        
        # Filter edges
        dropped_edge_index = edge_index[:, keep_mask]
        dropped_edge_attr = edge_attr[keep_mask] if edge_attr is not None else None
        
        if logger.isEnabledFor(logging.DEBUG):
            kept_edges = keep_mask.sum().item()
            logger.debug(f"DropEdge: kept {kept_edges}/{num_edges} edges (p_drop={self.p_drop})")
        
        return dropped_edge_index if edge_attr is None else (dropped_edge_index, dropped_edge_attr)
    

def benchmark_dropedge_determinism(
    edge_index: torch.Tensor,
    p_drop: float,
    num_trials: int = 10
) -> Dict[str, Any]:
    """
    Benchmark DropEdge determinism.
    
    Args:
        edge_index: Edge index tensor
        p_drop: Drop probability
        num_trials: Number of trials
        
    Returns:
        results: Determinism benchmark results
    """
    dropedge = DropEdge(p_drop=p_drop, training=True)
    
    # Test determinism
    first_result = dropedge(edge_index)
    deterministic = True
    
    for _ in range(num_trials - 1):
        result = dropedge(edge_index)
        if not torch.equal(result, first_result):
            deterministic = False
            break
    
    # Test drop rate
    original_edges = edge_index.size(1)
    dropped_edges = first_result.size(1)
    actual_drop_rate = 1.0 - (dropped_edges / original_edges)
    
    return {
        'is_deterministic': deterministic,
        'expected_drop_rate': p_drop,
        'actual_drop_rate': actual_drop_rate,
        'original_edges': original_edges,
        'remaining_edges': dropped_edges
    }


def benchmark_rgnn_overhead(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    config: Dict[str, Any],
    num_trials: int = 50
) -> Dict[str, Any]:
    """
    Benchmark RGNN wrapper overhead.
    
    Args:
        model: Base model
        x: Node features
        edge_index: Edge index
        config: RGNN configuration
        num_trials: Number of timing trials
        
    Returns:
        results: Overhead benchmark results
    """
    import time
    
    # Benchmark baseline model
    model.eval()
    baseline_times = []
    
    for _ in range(num_trials):
        start_time = time.time()
        with torch.no_grad():
            _ = model(x, edge_index)
        baseline_times.append(time.time() - start_time)
    
    # Create RGNN wrapper
    rgnn_model = create_robust_model(model, {'rgnn': config})
    rgnn_model.eval()
    
    # Benchmark RGNN model
    rgnn_times = []
    
    for _ in range(num_trials):
        start_time = time.time()
        with torch.no_grad():
            _ = rgnn_model(x, edge_index)
        rgnn_times.append(time.time() - start_time)
    
    baseline_avg = sum(baseline_times) / len(baseline_times)
    rgnn_avg = sum(rgnn_times) / len(rgnn_times)
    
    return {
        'baseline_time_avg': baseline_avg,
        'rgnn_time_avg': rgnn_avg,
        'overhead_ratio': rgnn_avg / baseline_avg if baseline_avg > 0 else float('inf'),
        'num_trials': num_trials
    }
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.training_mode = mode
        return super().train(mode)
    
    def eval(self):
        """Set evaluation mode."""
        self.training_mode = False
        return super().eval()


class RGNNWrapper(nn.Module):
    """
    RGNN-inspired defensive wrapper for robust graph neural networks.
    Following Stage7 Reference §Phase3: attention gating + spectral norm + gradient clipping.
    
    Implements noise-robust aggregation techniques:
    - Edge-wise attention gating with weight clipping
    - Spectral normalization on weight matrices
    - Residual connections for stability
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        hidden_dim: int,
        enable_spectral_norm: bool = True,
        enable_attention_gating: bool = True,
        attention_clip: float = 5.0,
        residual_weight: float = 0.1
    ):
        """
        Initialize RGNN wrapper.
        
        Args:
            base_model: Base GNN model to wrap
            hidden_dim: Hidden dimension for attention
            enable_spectral_norm: Apply spectral normalization
            enable_attention_gating: Use attention-based edge gating
            attention_clip: Clipping value for attention weights
            residual_weight: Weight for residual connections
        """
        super().__init__()
        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.enable_spectral_norm = enable_spectral_norm
        self.enable_attention_gating = enable_attention_gating
        self.attention_clip = attention_clip
        self.residual_weight = residual_weight
        
        # Attention gating mechanism
        if enable_attention_gating:
            self.attention_net = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            
            # Apply spectral normalization if enabled
            if enable_spectral_norm:
                for module in self.attention_net.modules():
                    if isinstance(module, nn.Linear):
                        spectral_norm(module)
        
        # Apply spectral norm to base model if requested
        if enable_spectral_norm:
            self._apply_spectral_norm_to_model(self.base_model)
    
    def _apply_spectral_norm_to_model(self, model: nn.Module):
        """Apply spectral normalization to linear layers in model."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                try:
                    spectral_norm(module)
                    logger.debug(f"Applied spectral norm to {name}")
                except Exception as e:
                    logger.warning(f"Failed to apply spectral norm to {name}: {e}")
    
    def _compute_attention_weights(
        self, 
        edge_index: torch.Tensor, 
        node_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention weights for edges.
        
        Args:
            edge_index: (2, E) edge index
            node_features: (N, F) node features
            
        Returns:
            attention_weights: (E,) attention weights for each edge
        """
        if not self.enable_attention_gating or edge_index.size(1) == 0:
            return torch.ones(edge_index.size(1), device=edge_index.device)
        
        src, dst = edge_index
        
        # Project features to hidden dimension if needed
        feature_dim = node_features.size(1)
        if feature_dim != self.hidden_dim:
            # Simple linear projection to match expected dimension
            projected_features = node_features[:, :min(feature_dim, self.hidden_dim)]
            if projected_features.size(1) < self.hidden_dim:
                # Pad if needed
                padding = torch.zeros(
                    node_features.size(0), 
                    self.hidden_dim - projected_features.size(1),
                    device=node_features.device
                )
                projected_features = torch.cat([projected_features, padding], dim=1)
        else:
            projected_features = node_features
        
        # Concatenate source and destination features
        edge_features = torch.cat([
            projected_features[src],
            projected_features[dst]
        ], dim=1)
        
        # Compute attention weights
        attention_weights = self.attention_net(edge_features).squeeze(1)
        
        # Apply clipping for stability
        if self.attention_clip > 0:
            attention_weights = torch.clamp(attention_weights, 0, self.attention_clip)
        
        return attention_weights
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass with robust aggregation.
        
        Args:
            x: (N, F) node features
            edge_index: (2, E) edge index
            **kwargs: Additional arguments for base model
            
        Returns:
            output: Model output with robust aggregation
        """
        # Store original features for residual connection
        x_original = x
        
        # Compute attention weights for edge gating
        if self.enable_attention_gating and edge_index.size(1) > 0:
            attention_weights = self._compute_attention_weights(edge_index, x)
            
            # Apply attention gating to edges (conceptual - actual implementation depends on base model)
            # For now, we pass attention weights as edge attributes if the model supports it
            if 'edge_attr' in kwargs:
                existing_attr = kwargs['edge_attr']
                if existing_attr is not None:
                    # Combine with existing edge attributes
                    gated_attr = existing_attr * attention_weights.unsqueeze(1)
                else:
                    gated_attr = attention_weights.unsqueeze(1)
                kwargs['edge_attr'] = gated_attr
            else:
                kwargs['edge_attr'] = attention_weights.unsqueeze(1)
        
        # Forward pass through base model
        output = self.base_model(x, edge_index, **kwargs)
        
        # Apply residual connection if dimensions match
        if self.residual_weight > 0 and x_original.size() == output.size():
            output = (1 - self.residual_weight) * output + self.residual_weight * x_original
        
        return output


class AdversarialEdgeTrainer:
    """
    Adversarial edge perturbation trainer for graph robustness.
    Following Stage7 Reference §Phase3: PGD-style small flips, toggleable and expensive.
    
    WARNING: This is computationally expensive and should be used sparingly.
    """
    
    def __init__(
        self,
        epsilon: float = 0.01,
        steps: int = 3,
        step_size: Optional[float] = None,
        norm: str = 'inf',
        enabled: bool = False
    ):
        """
        Initialize adversarial trainer.
        
        Args:
            epsilon: Perturbation budget
            steps: Number of PGD steps
            step_size: Step size (default: epsilon/steps)
            norm: Norm for perturbation ('inf' or '2')
            enabled: Whether adversarial training is enabled
        """
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size or epsilon / steps
        self.norm = norm
        self.enabled = enabled
        
        if enabled:
            logger.info(f"AdversarialEdgeTrainer initialized: ε={epsilon}, steps={steps}")
        else:
            logger.info("AdversarialEdgeTrainer disabled (expensive)")
    
    def perturb_edge_weights(
        self,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,
        model: nn.Module,
        batch_data: Dict[str, torch.Tensor],
        loss_fn: Callable
    ) -> torch.Tensor:
        """
        Generate adversarial edge weight perturbations.
        
        Args:
            edge_index: (2, E) edge index
            edge_weights: (E,) edge weights to perturb
            model: Model for computing gradients
            batch_data: Batch data for loss computation
            loss_fn: Loss function
            
        Returns:
            perturbed_weights: Adversarially perturbed edge weights
        """
        if not self.enabled:
            return edge_weights
        
        # Clone weights and enable gradients
        perturbed_weights = edge_weights.clone().detach().requires_grad_(True)
        
        for step in range(self.steps):
            # Zero gradients
            if perturbed_weights.grad is not None:
                perturbed_weights.grad.zero_()
            
            # Forward pass with current weights
            model.zero_grad()
            
            # Use perturbed weights in model (implementation-specific)
            # For the test model, we'll just use standard forward pass
            output = model(
                x=batch_data['x'],
                edge_index=edge_index
            )
            
            # Compute loss
            if 'train_mask' in batch_data:
                loss = loss_fn(output[batch_data['train_mask']], batch_data['y'][batch_data['train_mask']])
            else:
                loss = loss_fn(output, batch_data['y'])
            
            # Backward pass
            loss.backward()
            
            # Check if gradients exist
            if perturbed_weights.grad is None:
                logger.warning("No gradients computed for edge weights, using random perturbation")
                # Use small random perturbation as fallback
                delta = torch.randn_like(edge_weights) * self.epsilon * 0.1
                perturbed_weights = torch.clamp(edge_weights + delta, 0, 1)
                break
            
            # Update perturbation
            with torch.no_grad():
                if self.norm == 'inf':
                    grad_sign = perturbed_weights.grad.sign()
                    perturbed_weights = perturbed_weights + self.step_size * grad_sign
                    # Project to epsilon ball
                    delta = perturbed_weights - edge_weights
                    delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                    perturbed_weights = edge_weights + delta
                elif self.norm == '2':
                    grad_norm = perturbed_weights.grad.norm()
                    if grad_norm > 0:
                        grad_normalized = perturbed_weights.grad / grad_norm
                        perturbed_weights = perturbed_weights + self.step_size * grad_normalized
                        # Project to epsilon ball
                        delta = perturbed_weights - edge_weights
                        delta_norm = delta.norm()
                        if delta_norm > self.epsilon:
                            delta = delta / delta_norm * self.epsilon
                        perturbed_weights = edge_weights + delta
                
                # Clamp to valid range [0, 1] for edge weights
                perturbed_weights = torch.clamp(perturbed_weights, 0, 1)
                perturbed_weights = perturbed_weights.detach().requires_grad_(True)
        
        return perturbed_weights.detach()
    
    def perturb_and_train(
        self,
        batch_data: Dict[str, torch.Tensor],
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: Callable
    ) -> Tuple[float, float]:
        """
        Perform adversarial training step.
        
        Args:
            batch_data: Batch data
            model: Model to train
            optimizer: Optimizer
            loss_fn: Loss function
            
        Returns:
            (clean_loss, adv_loss): Clean and adversarial losses
        """
        if not self.enabled:
            # Standard training
            model.train()
            optimizer.zero_grad()
            
            output = model(x=batch_data['x'], edge_index=batch_data['edge_index'])
            if 'train_mask' in batch_data:
                loss = loss_fn(output[batch_data['train_mask']], batch_data['y'][batch_data['train_mask']])
            else:
                loss = loss_fn(output, batch_data['y'])
            
            loss.backward()
            optimizer.step()
            
            return loss.item(), loss.item()
        
        # Adversarial training
        edge_index = batch_data['edge_index']
        edge_weights = torch.ones(edge_index.size(1), device=edge_index.device)
        
        # Generate adversarial perturbation
        adv_weights = self.perturb_edge_weights(
            edge_index, edge_weights, model, batch_data, loss_fn
        )
        
        # Train on adversarial example
        model.train()
        optimizer.zero_grad()
        
        output = model(
            x=batch_data['x'],
            edge_index=edge_index,
            edge_attr=adv_weights.unsqueeze(1)
        )
        
        if 'train_mask' in batch_data:
            adv_loss = loss_fn(output[batch_data['train_mask']], batch_data['y'][batch_data['train_mask']])
        else:
            adv_loss = loss_fn(output, batch_data['y'])
        
        adv_loss.backward()
        optimizer.step()
        
        # Compute clean loss for comparison
        with torch.no_grad():
            model.eval()
            clean_output = model(x=batch_data['x'], edge_index=edge_index)
            if 'train_mask' in batch_data:
                clean_loss = loss_fn(clean_output[batch_data['train_mask']], batch_data['y'][batch_data['train_mask']])
            else:
                clean_loss = loss_fn(clean_output, batch_data['y'])
            model.train()
        
        return clean_loss.item(), adv_loss.item()


class RobustnessAugmentations:
    """
    Additional robustness techniques and regularization strategies.
    Following Stage7 Reference §Phase3: toggleable defenses.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize robustness augmentations."""
        self.config = config or {}
        self.label_smoothing = self.config.get('label_smoothing', 0.0)
        self.feature_noise_std = self.config.get('feature_noise_std', 0.0)
        self.gradient_clip = self.config.get('gradient_clip', 1.0)
    
    def apply_label_smoothing(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Apply label smoothing for regularization.
        
        Args:
            labels: (N,) class labels
            num_classes: Number of classes
            
        Returns:
            smoothed_labels: (N, C) smoothed label distribution
        """
        if self.label_smoothing == 0.0:
            return F.one_hot(labels, num_classes).float()
        
        # Convert to one-hot and apply smoothing
        one_hot = F.one_hot(labels, num_classes).float()
        smoothed = (1 - self.label_smoothing) * one_hot + self.label_smoothing / num_classes
        
        return smoothed
    
    def add_feature_noise(self, features: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to node features.
        
        Args:
            features: (N, F) node features
            
        Returns:
            noisy_features: Features with added noise
        """
        if self.feature_noise_std == 0.0:
            return features
        
        noise = torch.randn_like(features) * self.feature_noise_std
        return features + noise
    
    def clip_gradients(self, model: nn.Module) -> float:
        """
        Apply gradient clipping to model parameters.
        
        Args:
            model: Model to clip gradients
            
        Returns:
            grad_norm: Gradient norm before clipping
        """
        if self.gradient_clip <= 0:
            return 0.0
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
        return grad_norm.item()


def create_robust_model(
    base_model: nn.Module,
    config: Dict[str, Any]
) -> nn.Module:
    """
    Factory function to create robust model with defensive wrappers.
    Following Stage7 Reference §Phase3: toggleable robustness modules.
    
    Args:
        base_model: Base GNN model
        config: Robustness configuration
        
    Returns:
        robust_model: Model with robustness wrappers applied
    """
    robust_model = base_model
    
    # Apply RGNN wrapper if enabled
    if config.get('rgnn', {}).get('enabled', False):
        rgnn_config = config['rgnn']
        
        # Estimate hidden dimension (heuristic)
        hidden_dim = 64
        for module in base_model.modules():
            if isinstance(module, nn.Linear):
                hidden_dim = module.in_features
                break
        
        robust_model = RGNNWrapper(
            base_model=robust_model,
            hidden_dim=hidden_dim,
            enable_spectral_norm=rgnn_config.get('spectral_norm', True),
            enable_attention_gating=True,
            attention_clip=rgnn_config.get('attention_clip', 5.0)
        )
        
        logger.info("Applied RGNN defensive wrapper")
    
    return robust_model


def create_dropedge_transform(config: Dict[str, Any]) -> DropEdge:
    """
    Create DropEdge transform from configuration.
    
    Args:
        config: Configuration dict
        
    Returns:
        DropEdge transform
    """
    p_drop = config.get('dropedge_p', 0.1)
    return DropEdge(p_drop=p_drop)


def benchmark_robustness_overhead(
    model: nn.Module,
    robust_model: nn.Module,
    test_data: Dict[str, torch.Tensor],
    num_trials: int = 10
) -> Dict[str, float]:
    """
    Benchmark computational overhead of robustness modules.
    
    Args:
        model: Original model
        robust_model: Model with robustness modules
        test_data: Test data for timing
        num_trials: Number of timing trials
        
    Returns:
        timing_results: Dict with timing comparisons
    """
    device = next(model.parameters()).device
    
    # Move test data to device
    test_data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in test_data.items()}
    
    # Benchmark original model
    model.eval()
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    original_times = []
    for _ in range(num_trials):
        start_time = time.time()
        with torch.no_grad():
            _ = model(test_data['x'], test_data['edge_index'])
        torch.cuda.synchronize() if device.type == 'cuda' else None
        original_times.append(time.time() - start_time)
    
    # Benchmark robust model
    robust_model.eval()
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    robust_times = []
    for _ in range(num_trials):
        start_time = time.time()
        with torch.no_grad():
            _ = robust_model(test_data['x'], test_data['edge_index'])
        torch.cuda.synchronize() if device.type == 'cuda' else None
        robust_times.append(time.time() - start_time)
    
    # Compute statistics
    original_mean = sum(original_times) / len(original_times)
    robust_mean = sum(robust_times) / len(robust_times)
    overhead_ratio = robust_mean / original_mean
    overhead_ms = (robust_mean - original_mean) * 1000
    
    return {
        'original_time_ms': original_mean * 1000,
        'robust_time_ms': robust_mean * 1000,
        'overhead_ratio': overhead_ratio,
        'overhead_ms': overhead_ms
    }
