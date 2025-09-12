"""
Imbalance Handling for Graph Neural Networks
Following Stage7 Spot Target And Robustness Reference §Phase4

Implements class_weighted_loss, focal_loss, and GraphSMOTE helpers
for handling class imbalance in graph classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from collections import Counter

logger = logging.getLogger(__name__)


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: Optional[int] = None,
    method: str = 'inverse_frequency'
) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        labels: (N,) class labels
        num_classes: Number of classes (inferred if None)
        method: Weighting method ('inverse_frequency', 'balanced', 'sqrt_inverse')
        
    Returns:
        class_weights: (C,) weight for each class
    """
    if num_classes is None:
        num_classes = int(labels.max().item()) + 1
    
    # Count class frequencies
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    total_samples = labels.size(0)
    
    if method == 'inverse_frequency':
        # Weight inversely proportional to frequency
        weights = total_samples / (num_classes * class_counts)
    elif method == 'balanced':
        # sklearn-style balanced weights
        weights = total_samples / (num_classes * class_counts)
    elif method == 'sqrt_inverse':
        # Square root of inverse frequency (less aggressive)
        freq_weights = total_samples / (num_classes * class_counts)
        weights = torch.sqrt(freq_weights)
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Handle zero counts
    weights[torch.isinf(weights)] = 0.0
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    logger.info(f"Class weights computed ({method}): {weights.tolist()}")
    
    return weights


def class_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute class-weighted cross-entropy loss.
    
    Args:
        logits: (N, C) model predictions
        labels: (N,) ground truth labels
        class_weights: (C,) weights for each class
        reduction: Loss reduction method
        
    Returns:
        loss: Weighted cross-entropy loss
    """
    if class_weights is None:
        # Compute weights automatically
        num_classes = logits.size(1)
        class_weights = compute_class_weights(labels, num_classes)
    
    # Move weights to same device as logits
    class_weights = class_weights.to(logits.device)
    
    # Compute weighted cross-entropy
    loss = F.cross_entropy(logits, labels, weight=class_weights, reduction=reduction)
    
    return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Following Stage7 Reference §Phase4: focal_loss implementation.
    
    Focal Loss = -α(1-p_t)^γ * log(p_t)
    where p_t is the model's estimated probability for the true class.
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: (C,) class weights or None for uniform
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Loss reduction ('mean', 'sum', 'none')
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: (N, C) model predictions
            labels: (N,) ground truth labels
            
        Returns:
            loss: Focal loss value
        """
        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(
            logits, labels, 
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        # Compute probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get probabilities for true classes
        labels_one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
        pt = (probs * labels_one_hot).sum(dim=1)
        
        # Compute focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha[labels]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class GraphSMOTE:
    """
    Graph-based SMOTE for synthetic minority class oversampling.
    Following Stage7 Reference §Phase4: careful to ensure no label leakage.
    
    WARNING: This is a simplified implementation. Be very careful about 
    label leakage when using synthetic samples.
    """
    
    def __init__(
        self,
        k_neighbors: int = 5,
        sampling_strategy: str = 'minority',
        random_state: Optional[int] = None
    ):
        """
        Initialize GraphSMOTE.
        
        Args:
            k_neighbors: Number of neighbors for interpolation
            sampling_strategy: Strategy for sampling ('minority', 'all', dict)
            random_state: Random seed for reproducibility
        """
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
    
    def _find_graph_neighbors(
        self,
        edge_index: torch.Tensor,
        node_features: torch.Tensor,
        target_node: int,
        k: int
    ) -> List[int]:
        """
        Find k nearest neighbors in graph structure.
        
        Args:
            edge_index: (2, E) edge index
            node_features: (N, F) node features
            target_node: Target node index
            k: Number of neighbors to find
            
        Returns:
            neighbor_indices: List of neighbor node indices
        """
        # Get direct neighbors from graph
        src, dst = edge_index
        direct_neighbors = dst[src == target_node].tolist()
        
        if len(direct_neighbors) >= k:
            # Randomly sample k direct neighbors
            return np.random.choice(direct_neighbors, k, replace=False).tolist()
        else:
            # Use feature similarity for additional neighbors
            target_features = node_features[target_node]
            
            # Compute similarities with all other nodes
            similarities = F.cosine_similarity(
                target_features.unsqueeze(0),
                node_features,
                dim=1
            )
            
            # Exclude self and sort by similarity
            similarities[target_node] = -float('inf')
            _, indices = similarities.topk(k)
            
            return indices.tolist()
    
    def _generate_synthetic_sample(
        self,
        node_features: torch.Tensor,
        source_idx: int,
        neighbor_idx: int,
        alpha: float = None
    ) -> torch.Tensor:
        """
        Generate synthetic sample between source and neighbor.
        
        Args:
            node_features: (N, F) node features
            source_idx: Source node index
            neighbor_idx: Neighbor node index
            alpha: Interpolation factor (random if None)
            
        Returns:
            synthetic_features: (F,) synthetic node features
        """
        if alpha is None:
            alpha = np.random.random()
        
        source_features = node_features[source_idx]
        neighbor_features = node_features[neighbor_idx]
        
        # Linear interpolation
        synthetic_features = (1 - alpha) * source_features + alpha * neighbor_features
        
        return synthetic_features
    
    def fit_resample(
        self,
        node_features: torch.Tensor,
        labels: torch.Tensor,
        edge_index: torch.Tensor,
        train_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate synthetic samples for minority classes.
        
        Args:
            node_features: (N, F) node features
            labels: (N,) node labels
            edge_index: (2, E) edge index
            train_mask: (N,) training mask
            
        Returns:
            (augmented_features, augmented_labels, augmented_mask): 
            Augmented data with synthetic samples
        """
        # Only consider training nodes
        train_indices = torch.where(train_mask)[0]
        train_labels = labels[train_mask]
        
        # Count class frequencies
        class_counts = torch.bincount(train_labels)
        num_classes = len(class_counts)
        
        # Determine target counts
        if self.sampling_strategy == 'minority':
            max_count = class_counts.max().item()
            target_counts = {i: max_count for i in range(num_classes)}
        else:
            # Use existing counts (no oversampling)
            target_counts = {i: class_counts[i].item() for i in range(num_classes)}
        
        # Collect synthetic samples
        synthetic_features = []
        synthetic_labels = []
        
        for class_idx in range(num_classes):
            current_count = class_counts[class_idx].item()
            target_count = target_counts[class_idx]
            
            if target_count > current_count:
                # Need to generate synthetic samples
                n_synthetic = target_count - current_count
                class_nodes = train_indices[train_labels == class_idx]
                
                for _ in range(n_synthetic):
                    # Randomly select a minority class node
                    source_idx = np.random.choice(class_nodes.tolist())
                    
                    # Find neighbors
                    neighbors = self._find_graph_neighbors(
                        edge_index, node_features, source_idx, self.k_neighbors
                    )
                    
                    # Filter neighbors to same class to avoid label leakage
                    same_class_neighbors = [
                        n for n in neighbors 
                        if n < len(labels) and labels[n] == class_idx
                    ]
                    
                    if same_class_neighbors:
                        neighbor_idx = np.random.choice(same_class_neighbors)
                    else:
                        # Fallback: use any neighbor (less ideal)
                        neighbor_idx = np.random.choice(neighbors)
                        logger.warning(f"No same-class neighbors found for node {source_idx}")
                    
                    # Generate synthetic sample
                    synthetic_sample = self._generate_synthetic_sample(
                        node_features, source_idx, neighbor_idx
                    )
                    
                    synthetic_features.append(synthetic_sample)
                    synthetic_labels.append(class_idx)
        
        # Combine original and synthetic data
        if synthetic_features:
            synthetic_features = torch.stack(synthetic_features)
            synthetic_labels = torch.tensor(synthetic_labels, dtype=labels.dtype)
            
            augmented_features = torch.cat([node_features, synthetic_features], dim=0)
            augmented_labels = torch.cat([labels, synthetic_labels], dim=0)
            
            # Extend masks
            n_synthetic = len(synthetic_labels)
            synthetic_train_mask = torch.ones(n_synthetic, dtype=torch.bool)
            augmented_mask = torch.cat([train_mask, synthetic_train_mask], dim=0)
            
            logger.info(f"Generated {n_synthetic} synthetic samples using GraphSMOTE")
            
            return augmented_features, augmented_labels, augmented_mask
        else:
            # No synthetic samples needed
            return node_features, labels, train_mask


class ImbalanceHandler:
    """
    Unified handler for class imbalance in graph learning.
    Following Stage7 Reference §Phase4: comprehensive imbalance handling.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize imbalance handler.
        
        Args:
            config: Configuration dict with imbalance handling options
        """
        self.config = config or {}
        
        # Loss function configuration
        self.use_focal_loss = self.config.get('use_focal_loss', False)
        self.focal_gamma = self.config.get('focal_gamma', 2.0)
        self.class_weights_method = self.config.get('class_weights', 'auto')
        
        # SMOTE configuration
        self.use_smote = self.config.get('use_smote', False)
        self.smote_k = self.config.get('smote_k_neighbors', 5)
        
        # Cache for computed weights
        self._class_weights = None
        self._focal_loss = None
    
    def compute_loss_function(
        self,
        labels: torch.Tensor,
        num_classes: Optional[int] = None
    ) -> nn.Module:
        """
        Create appropriate loss function for imbalanced data.
        
        Args:
            labels: Training labels for weight computation
            num_classes: Number of classes
            
        Returns:
            loss_function: Configured loss function
        """
        if num_classes is None:
            num_classes = int(labels.max().item()) + 1
        
        # Compute class weights if needed
        if self.class_weights_method == 'auto':
            self._class_weights = compute_class_weights(labels, num_classes)
        elif isinstance(self.class_weights_method, (list, tuple)):
            self._class_weights = torch.tensor(self.class_weights_method, dtype=torch.float)
        else:
            self._class_weights = None
        
        # Create loss function
        if self.use_focal_loss:
            self._focal_loss = FocalLoss(
                alpha=self._class_weights,
                gamma=self.focal_gamma,
                reduction='mean'
            )
            return self._focal_loss
        else:
            # Use class-weighted cross-entropy
            def weighted_ce_loss(logits, targets):
                return class_weighted_loss(logits, targets, self._class_weights)
            
            return weighted_ce_loss
    
    def apply_smote_augmentation(
        self,
        node_features: torch.Tensor,
        labels: torch.Tensor,
        edge_index: torch.Tensor,
        train_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply GraphSMOTE augmentation if enabled.
        
        Args:
            node_features: (N, F) node features
            labels: (N,) labels
            edge_index: (2, E) edge index
            train_mask: (N,) training mask
            
        Returns:
            (features, labels, mask): Possibly augmented data
        """
        if not self.use_smote:
            return node_features, labels, train_mask
        
        smote = GraphSMOTE(
            k_neighbors=self.smote_k,
            random_state=self.config.get('seed', 42)
        )
        
        return smote.fit_resample(node_features, labels, edge_index, train_mask)
    
    def get_class_distribution(self, labels: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, Any]:
        """
        Analyze class distribution in dataset.
        
        Args:
            labels: Class labels
            mask: Optional mask to consider subset
            
        Returns:
            distribution_stats: Class distribution statistics
        """
        if mask is not None:
            labels = labels[mask]
        
        class_counts = torch.bincount(labels)
        total = len(labels)
        
        distribution = {
            'class_counts': class_counts.tolist(),
            'class_proportions': (class_counts.float() / total).tolist(),
            'total_samples': total,
            'num_classes': len(class_counts),
            'imbalance_ratio': class_counts.max().item() / class_counts.min().item()
        }
        
        return distribution


def create_imbalance_handler(config: Dict[str, Any]) -> ImbalanceHandler:
    """
    Factory function to create imbalance handler.
    
    Args:
        config: Configuration dict
        
    Returns:
        ImbalanceHandler instance
    """
    return ImbalanceHandler(config)


def analyze_class_imbalance(
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor
) -> Dict[str, Dict[str, Any]]:
    """
    Comprehensive analysis of class imbalance across splits.
    
    Args:
        labels: All labels
        train_mask: Training mask
        val_mask: Validation mask  
        test_mask: Test mask
        
    Returns:
        imbalance_analysis: Analysis for each split
    """
    handler = ImbalanceHandler()
    
    analysis = {
        'train': handler.get_class_distribution(labels, train_mask),
        'val': handler.get_class_distribution(labels, val_mask),
        'test': handler.get_class_distribution(labels, test_mask),
        'overall': handler.get_class_distribution(labels)
    }
    
    # Add recommendations
    train_imbalance = analysis['train']['imbalance_ratio']
    if train_imbalance > 10:
        analysis['recommendations'] = [
            'High imbalance detected (ratio > 10)',
            'Consider using focal loss or class weights',
            'GraphSMOTE may help with synthetic oversampling'
        ]
    elif train_imbalance > 3:
        analysis['recommendations'] = [
            'Moderate imbalance detected',
            'Class weights recommended'
        ]
    else:
        analysis['recommendations'] = ['Dataset appears balanced']
    
    return analysis
