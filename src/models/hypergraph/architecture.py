"""
Hypergraph Neural Network Architecture
=====================================

Complete multi-layer hypergraph neural network implementation combining
PhenomNN layers with proper initialization, forward pass, and integration
with the existing fraud detection pipeline.

Classes:
    HypergraphNN: Multi-layer hypergraph neural network
    HypergraphConfig: Configuration dataclass for hyperparameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Union
import logging

from .hypergraph_data import HypergraphData
from .phenomnn import PhenomNNSimpleLayer, PhenomNNLayer

logger = logging.getLogger(__name__)


@dataclass
class HypergraphConfig:
    """Configuration for hypergraph neural networks."""
    
    # Architecture parameters
    layer_type: str = 'full'  # 'simple' or 'full'
    num_layers: int = 3
    hidden_dim: int = 64
    dropout: float = 0.2
    use_residual: bool = True
    use_batch_norm: bool = False
    
    # PhenomNN layer parameters
    lambda0_init: float = 1.0  # Initial clique expansion weight
    lambda1_init: float = 1.0  # Initial star expansion weight
    alpha_init: float = 0.1    # Initial step size
    max_iterations: int = 10   # Max iterations for convergence
    convergence_threshold: float = 1e-4
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 5e-4
    
    # Fraud-specific parameters
    fraud_pattern_types: List[str] = None  # Will be set to default if None
    
    def __post_init__(self):
        """Set default fraud pattern types if not provided."""
        if self.fraud_pattern_types is None:
            self.fraud_pattern_types = [
                'transaction_patterns',
                'temporal_patterns', 
                'amount_patterns',
                'behavioral_patterns'
            ]


class HypergraphNN(nn.Module):
    """
    Multi-layer Hypergraph Neural Network for Fraud Detection
    
    Combines multiple PhenomNN layers with residual connections,
    normalization, and dropout for robust fraud detection.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (typically 2 for binary classification)
        config: HypergraphConfig object with hyperparameters
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        config: Optional[HypergraphConfig] = None
    ):
        super().__init__()
        
        self.config = config or HypergraphConfig()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = self.config.num_layers
        
        # PhenomNN layers (no separate input projection)
        self.hypergraph_layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            # First layer takes input_dim, subsequent layers take hidden_dim
            layer_input_dim = input_dim if i == 0 else hidden_dim
            
            if self.config.layer_type == 'simple':
                layer = PhenomNNSimpleLayer(
                    input_dim=layer_input_dim,
                    hidden_dim=hidden_dim,
                    lambda0=self.config.lambda0_init,
                    lambda1=self.config.lambda1_init,
                    alpha=self.config.alpha_init,
                    num_iterations=self.config.max_iterations,
                    convergence_threshold=self.config.convergence_threshold
                )
            elif self.config.layer_type == 'full':
                layer = PhenomNNLayer(
                    input_dim=layer_input_dim,
                    hidden_dim=hidden_dim,
                    lambda0=self.config.lambda0_init,
                    lambda1=self.config.lambda1_init,
                    alpha=self.config.alpha_init,
                    num_iterations=self.config.max_iterations,
                    convergence_threshold=self.config.convergence_threshold
                )
            else:
                raise ValueError(f"Unknown layer_type: {self.config.layer_type}")
                
            self.hypergraph_layers.append(layer)
        
        # Normalization layers (optional)
        if self.config.use_batch_norm:
            self.norm_layers = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(self.num_layers)
            ])
        else:
            self.norm_layers = None
            
        # Dropout
        self.dropout = nn.Dropout(self.config.dropout)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Initialize parameters
        self._init_parameters()
        
        logger.info(f"Created HypergraphNN: {input_dim}→{hidden_dim}→{output_dim}, "
                   f"{self.num_layers} {self.config.layer_type} layers")
    
    def _init_parameters(self):
        """Initialize network parameters."""
        # Xavier initialization for output layer
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        logger.debug("Initialized HypergraphNN parameters")
    
    def forward(
        self, 
        hypergraph_data: HypergraphData,
        node_features: torch.Tensor,
        return_attention: bool = False,
        return_convergence_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass through hypergraph neural network.
        
        Args:
            hypergraph_data: HypergraphData object with matrices
            node_features: Node features [num_nodes, input_dim]
            return_attention: Whether to return attention weights
            return_convergence_info: Whether to return convergence information
            
        Returns:
            output: Network output [num_nodes, output_dim]
            info: Dictionary with attention/convergence info (if requested)
        """
        batch_size = node_features.size(0)
        
        # Start with input features directly
        x = node_features  # [num_nodes, input_dim]
        
        # Store information for analysis
        layer_outputs = []
        convergence_info = {}
        attention_weights = []
        
        # Pass through hypergraph layers
        for i, layer in enumerate(self.hypergraph_layers):
            # Store input for residual connection
            residual = x if self.config.use_residual and x.size(-1) == self.hidden_dim else None
            
            # Apply hypergraph layer
            # Create a copy of hypergraph_data with current features
            layer_hypergraph_data = HypergraphData(
                incidence_matrix=hypergraph_data.B,
                node_features=x,
                hyperedge_features=hypergraph_data.U,
                node_labels=hypergraph_data.y
            )
            
            if return_convergence_info:
                layer_output, layer_info = layer(
                    layer_hypergraph_data, return_convergence_info=True
                )
                convergence_info[f'layer_{i}'] = layer_info
            else:
                layer_output, _ = layer(layer_hypergraph_data)
            
            # Apply normalization
            if self.norm_layers is not None:
                layer_output = self.norm_layers[i](layer_output)
            
            # Apply activation and dropout
            layer_output = F.relu(layer_output)
            layer_output = self.dropout(layer_output)
            
            # Residual connection (only if dimensions match)
            if residual is not None and residual.size(-1) == layer_output.size(-1):
                x = layer_output + residual
            else:
                x = layer_output
                
            layer_outputs.append(x.clone())
        
        # Output projection
        output = self.output_proj(x)  # [num_nodes, output_dim]
        
        # Prepare return values
        if return_attention or return_convergence_info:
            info = {
                'layer_outputs': layer_outputs,
                'convergence_info': convergence_info if return_convergence_info else None,
                'attention_weights': attention_weights if return_attention else None
            }
            return output, info
        else:
            return output
    
    def get_layer_parameters(self) -> Dict[str, Dict[str, float]]:
        """Get learnable parameters from each layer."""
        params = {}
        for i, layer in enumerate(self.hypergraph_layers):
            layer_params = {}
            if hasattr(layer, 'lambda0'):
                if hasattr(layer.lambda0, 'item'):  # learnable parameter
                    layer_params['lambda0'] = layer.lambda0.item()
                else:  # fixed parameter
                    layer_params['lambda0'] = layer.lambda0
            if hasattr(layer, 'lambda1'):
                if hasattr(layer.lambda1, 'item'):
                    layer_params['lambda1'] = layer.lambda1.item()
                else:
                    layer_params['lambda1'] = layer.lambda1
            if hasattr(layer, 'alpha'):
                if hasattr(layer.alpha, 'item'):
                    layer_params['alpha'] = layer.alpha.item()
                else:
                    layer_params['alpha'] = layer.alpha
            params[f'layer_{i}'] = layer_params
        return params
    
    def count_parameters(self) -> Tuple[int, int]:
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def get_hypergraph_stats(self, hypergraph_data: HypergraphData) -> Dict[str, Any]:
        """Get statistics about the hypergraph structure."""
        hyperedge_sizes = hypergraph_data.B.sum(dim=0).float()
        node_degrees = hypergraph_data.B.sum(dim=1).float()
        
        # Handle empty hypergraph case
        if hypergraph_data.n_hyperedges == 0:
            stats = {
                'num_nodes': hypergraph_data.n_nodes,
                'num_hyperedges': 0,
                'avg_hyperedge_size': 0.0,
                'max_hyperedge_size': 0.0,
                'min_hyperedge_size': 0.0,
                'avg_node_degree': 0.0,
                'max_node_degree': 0.0,
                'density': 0.0
            }
        else:
            stats = {
                'num_nodes': hypergraph_data.n_nodes,
                'num_hyperedges': hypergraph_data.n_hyperedges,
                'avg_hyperedge_size': hyperedge_sizes.mean().item(),
                'max_hyperedge_size': hyperedge_sizes.max().item(),
                'min_hyperedge_size': hyperedge_sizes.min().item(),
                'avg_node_degree': node_degrees.mean().item(),
                'max_node_degree': node_degrees.max().item(),
                'density': (hypergraph_data.B.sum() / 
                           (hypergraph_data.n_nodes * hypergraph_data.n_hyperedges)).item()
            }
        return stats


class HypergraphClassifier(nn.Module):
    """
    Complete hypergraph-based fraud detection classifier.
    
    Combines hypergraph construction, neural network processing,
    and classification into a single end-to-end model.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int = 2,
        config: Optional[HypergraphConfig] = None
    ):
        super().__init__()
        
        self.config = config or HypergraphConfig()
        self.hypergraph_nn = HypergraphNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            config=self.config
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(
        self,
        hypergraph_data: HypergraphData,
        node_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional loss computation.
        
        Args:
            hypergraph_data: HypergraphData object
            node_features: Node features [num_nodes, input_dim]
            labels: Ground truth labels [num_nodes] (optional)
            
        Returns:
            logits: Class predictions [num_nodes, num_classes]
            loss: Cross-entropy loss (if labels provided)
        """
        logits = self.hypergraph_nn(hypergraph_data, node_features)
        
        if labels is not None:
            loss = self.criterion(logits, labels)
            return logits, loss
        else:
            return logits
    
    def predict(
        self,
        hypergraph_data: HypergraphData,
        node_features: torch.Tensor
    ) -> torch.Tensor:
        """Predict class labels."""
        self.eval()
        with torch.no_grad():
            logits = self(hypergraph_data, node_features)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def predict_proba(
        self,
        hypergraph_data: HypergraphData,
        node_features: torch.Tensor
    ) -> torch.Tensor:
        """Predict class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self(hypergraph_data, node_features)
            probabilities = F.softmax(logits, dim=1)
        return probabilities


def create_hypergraph_model(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    model_config: Dict[str, Any]
) -> HypergraphNN:
    """
    Factory function to create hypergraph models from config dictionary.
    
    This function provides compatibility with the existing training pipeline
    by accepting the same config format as other models.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension  
        output_dim: Output dimension
        model_config: Dictionary with model configuration
        
    Returns:
        Configured HypergraphNN model
    """
    # Convert config dict to HypergraphConfig
    config = HypergraphConfig(
        layer_type=model_config.get('layer_type', 'full'),
        num_layers=model_config.get('num_layers', 3),
        hidden_dim=hidden_dim,
        dropout=model_config.get('dropout', 0.2),
        use_residual=model_config.get('use_residual', True),
        use_batch_norm=model_config.get('use_batch_norm', False),
        lambda0_init=model_config.get('lambda0_init', 1.0),
        lambda1_init=model_config.get('lambda1_init', 1.0),
        alpha_init=model_config.get('alpha_init', 0.1),
        max_iterations=model_config.get('max_iterations', 10),
        convergence_threshold=model_config.get('convergence_threshold', 1e-4)
    )
    
    model = HypergraphNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        config=config
    )
    
    total_params, trainable_params = model.count_parameters()
    logger.info(f"Created hypergraph model: {total_params:,} total params, "
               f"{trainable_params:,} trainable")
    
    return model
