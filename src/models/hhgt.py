"""
hHGTN (Heterogeneous Hypergraph Temporal Graph Network) - Stage 9 Integration

This module implements the full hHGTN pipeline combining all previous stages:
- Stage 3: Heterogeneous layers (HGT/R-GCN)  
- Stage 4: Temporal memory (TGN)
- Stage 5: Hypergraph layers (PhenomNN)
- Stage 6: Efficient sampling (TDGNN + GSampler)
- Stage 7: Training discipline (SpotTarget + Robustness)
- Stage 8: Manifold embeddings (CUSP)

The model supports modular toggling of components via configuration flags.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union
import logging
from pathlib import Path
import yaml

# Import component adapters for Stage 9 integration
from .hhgt_adapters import (
    PhenomAdapter, HeteroAdapter, TGNAdapter, CuspAdapter,
    SpotTargetAdapter, RobustnessAdapter, TDGNNAdapter
)

logger = logging.getLogger(__name__)


class hHGTNConfig:
    """Configuration class for hHGTN model with all component toggles."""
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        # Default configuration
        self.config = {
            'model': {
                'name': 'hHGTN',
                'use_hypergraph': True,
                'use_hetero': True, 
                'use_memory': True,
                'use_cusp': True,
                'use_tdgnn': True,
                'use_gsampler': True,
                'use_spottarget': True,
                'use_robustness': True,
                'hidden_dim': 128,
                'num_layers': 3,
                'num_heads': 8,
                'dropout': 0.1
            }
        }
        
        # Load from file if provided
        if config_path:
            self.load_config(config_path)
            
        # Override with kwargs
        self.update_config(**kwargs)
    
    def load_config(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
        self._deep_update(self.config, file_config)
    
    def update_config(self, **kwargs):
        """Update configuration with keyword arguments."""
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested keys like 'model.hidden_dim'
                keys = key.split('.')
                config_section = self.config
                for k in keys[:-1]:
                    if k not in config_section:
                        config_section[k] = {}
                    config_section = config_section[k]
                config_section[keys[-1]] = value
            else:
                self.config[key] = value
    
    def _deep_update(self, base_dict, update_dict):
        """Recursively update nested dictionary."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


class hHGTN(nn.Module):
    """
    Full hHGTN model integrating all components with modular toggles.
    
    Forward Pass Pipeline:
    1. Optional sampling (GSampler/TDGNN)
    2. Optional SpotTarget filtering  
    3. CUSP embeddings → augment features
    4. Hypergraph/PhenomNN layers
    5. Heterogeneous layers (HGT/R-GCN)
    6. TGN memory updates
    7. RGNN/DropEdge defenses
    8. Final classifier
    """
    
    def __init__(self, 
                 node_types: Dict[str, int],
                 edge_types: Dict[Tuple[str, str, str], int], 
                 config: Union[hHGTNConfig, str, Dict] = None,
                 **kwargs):
        super().__init__()
        
        # Initialize configuration
        if isinstance(config, str):
            self.config = hHGTNConfig(config_path=config, **kwargs)
        elif isinstance(config, dict):
            self.config = hHGTNConfig(**config, **kwargs)
        elif isinstance(config, hHGTNConfig):
            self.config = config
            self.config.update_config(**kwargs)
        else:
            self.config = hHGTNConfig(**kwargs)
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = self.config.get('model.hidden_dim', 128)
        
        # Initialize components based on config flags
        self._build_components()
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.get('model.dropout', 0.1)),
            nn.Linear(self.hidden_dim // 2, 2)  # Binary classification
        )
        
        # Memory state for checkpointing
        self.memory_state = {}
        
        logger.info(f"Initialized hHGTN with components: {self._get_active_components()}")
    
    def _build_components(self):
        """Build model components based on configuration flags."""
        
        # 1. CUSP Manifold Embeddings (Stage 8)
        if self.config.get('model.use_cusp', False):
            self.cusp_module = CuspAdapter(
                node_types=self.node_types,
                edge_types=self.edge_types,
                config=self.config.get('model.cusp', {})
            )
        else:
            self.cusp_module = None
            
        # 2. Hypergraph Layers (Stage 5 - PhenomNN)
        if self.config.get('model.use_hypergraph', False):
            self.hypergraph_module = PhenomAdapter(
                node_types=self.node_types,
                edge_types=self.edge_types, 
                hidden_dim=self.hidden_dim,
                config=self.config.get('model.hypergraph', {})
            )
        else:
            self.hypergraph_module = None
            
        # 3. Heterogeneous Layers (Stage 3 - HGT/R-GCN)
        if self.config.get('model.use_hetero', False):
            self.hetero_module = HeteroAdapter(
                node_types=self.node_types,
                edge_types=self.edge_types,
                hidden_dim=self.hidden_dim,
                num_layers=self.config.get('model.num_layers', 3),
                num_heads=self.config.get('model.num_heads', 8)
            )
        else:
            self.hetero_module = None
            
        # 4. Temporal Memory (Stage 4 - TGN)
        if self.config.get('model.use_memory', False):
            self.memory_module = TGNAdapter(
                node_types=self.node_types,
                config=self.config.get('model.memory', {})
            )
        else:
            self.memory_module = None
            
        # 5. Robustness Defenses (Stage 7 - RGNN/DropEdge)
        if self.config.get('model.use_robustness', False):
            self.robustness_module = RobustnessAdapter(
                hidden_dim=self.hidden_dim,
                config=self.config.get('model.robustness', {})
            )
        else:
            self.robustness_module = None
            
        # 6. SpotTarget Training Discipline (Stage 7)
        if self.config.get('model.use_spottarget', False):
            self.spottarget_module = SpotTargetAdapter(
                config=self.config.get('training.spottarget', {})
            )
        else:
            self.spottarget_module = None
            
        # 7. TDGNN Sampling (Stage 6)
        if self.config.get('model.use_tdgnn', False):
            self.tdgnn_module = TDGNNAdapter(
                config=self.config.get('sampling.tdgnn', {})
            )
        else:
            self.tdgnn_module = None
    
    def forward(self, 
                batch,
                return_attention: bool = False,
                return_memory: bool = False) -> Dict[str, Any]:
        """
        Forward pass through hHGTN pipeline.
        
        Args:
            batch: Heterogeneous graph batch
            return_attention: Whether to return attention weights
            return_memory: Whether to return memory states
            
        Returns:
            Dictionary containing logits, embeddings, and optional components
        """
        
        # Initialize results dictionary
        results = {
            'logits': None,
            'embeddings': None,
            'attention_weights': None if return_attention else None,
            'memory_state': None if return_memory else None,
            'component_outputs': {}
        }
        
        # Step 1: Optional TDGNN Sampling
        if self.tdgnn_module is not None and self.training:
            batch = self.tdgnn_module.sample_batch(batch)
            results['component_outputs']['tdgnn_sampling'] = True
        
        # Step 2: Optional SpotTarget Filtering  
        if self.spottarget_module is not None and self.training:
            batch = self.spottarget_module.filter_batch(batch)
            results['component_outputs']['spottarget_filtering'] = True
        
        # Step 3: CUSP Embeddings → Augment Features
        x = batch.x_dict.copy()  # Node features by type
        edge_index = batch.edge_index_dict  # Edge indices by type
        
        if self.cusp_module is not None:
            cusp_embeddings = self.cusp_module(batch)
            # Augment node features with CUSP embeddings
            for node_type in x.keys():
                if node_type in cusp_embeddings:
                    x[node_type] = torch.cat([x[node_type], cusp_embeddings[node_type]], dim=-1)
            results['component_outputs']['cusp_embeddings'] = cusp_embeddings
        
        # Step 4: Hypergraph/PhenomNN Layers
        if self.hypergraph_module is not None:
            # Create batch-like object for hypergraph module
            hypergraph_batch = type('Batch', (), {'x_dict': x})()
            hypergraph_out = self.hypergraph_module(hypergraph_batch)
            x = hypergraph_out['embeddings']
            results['component_outputs']['hypergraph_output'] = hypergraph_out
        
        # Step 5: Heterogeneous Layers (HGT/R-GCN)
        if self.hetero_module is not None:
            hetero_out = self.hetero_module(x, edge_index)
            x = hetero_out['embeddings']
            if return_attention and 'attention_weights' in hetero_out:
                results['attention_weights'] = hetero_out['attention_weights']
            results['component_outputs']['hetero_output'] = hetero_out
        
        # Step 6: TGN Memory Updates
        if self.memory_module is not None:
            memory_out = self.memory_module.update_memory(batch, x)
            x = memory_out['embeddings']
            if return_memory:
                results['memory_state'] = memory_out['memory_state']
            results['component_outputs']['memory_output'] = memory_out
        
        # Step 7: RGNN/DropEdge Defenses
        if self.robustness_module is not None:
            x = self.robustness_module.apply_defenses(x, edge_index)
            results['component_outputs']['robustness_applied'] = True
        
        # Step 8: Final Classification
        # For node-level tasks, get target node embeddings
        if hasattr(batch, 'target_nodes'):
            target_embeddings = self._get_target_embeddings(x, batch.target_nodes)
        else:
            # For graph-level tasks, pool all embeddings
            target_embeddings = self._pool_embeddings(x)
        
        # Apply classifier
        logits = self.classifier(target_embeddings)
        
        results['logits'] = logits
        results['embeddings'] = target_embeddings
        
        return results
    
    def _get_target_embeddings(self, x_dict: Dict[str, torch.Tensor], 
                             target_nodes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract embeddings for target nodes."""
        target_embeddings = []
        for node_type, node_indices in target_nodes.items():
            if node_type in x_dict:
                target_embeddings.append(x_dict[node_type][node_indices])
        
        if len(target_embeddings) == 0:
            raise ValueError("No target node embeddings found")
        elif len(target_embeddings) == 1:
            return target_embeddings[0]
        else:
            # Concatenate embeddings from different node types
            return torch.cat(target_embeddings, dim=-1)
    
    def _pool_embeddings(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Pool embeddings across all nodes for graph-level tasks."""
        pooled_embeddings = []
        for node_type, embeddings in x_dict.items():
            # Use mean pooling for each node type
            pooled = torch.mean(embeddings, dim=0, keepdim=True)
            pooled_embeddings.append(pooled)
        
        if len(pooled_embeddings) == 0:
            raise ValueError("No node embeddings found for pooling")
        
        # Concatenate pooled embeddings from all node types
        return torch.cat(pooled_embeddings, dim=-1)
    
    def _get_active_components(self) -> list:
        """Get list of active components for logging."""
        components = []
        if self.config.get('model.use_cusp', False):
            components.append('CUSP')
        if self.config.get('model.use_hypergraph', False):
            components.append('Hypergraph')
        if self.config.get('model.use_hetero', False):
            components.append('Heterogeneous')
        if self.config.get('model.use_memory', False):
            components.append('Memory')
        if self.config.get('model.use_robustness', False):
            components.append('Robustness')
        if self.config.get('model.use_spottarget', False):
            components.append('SpotTarget')
        if self.config.get('model.use_tdgnn', False):
            components.append('TDGNN')
        return components
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Dict = None, 
                       metrics: Dict = None):
        """Save model checkpoint with all components."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': self.config.config,
            'node_types': self.node_types,
            'edge_types': self.edge_types,
            'memory_state': self.memory_state,
            'metrics': metrics or {}
        }
        
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
            
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = False):
        """Load model checkpoint and restore all components."""
        checkpoint = torch.load(path, map_location='cpu')
        
        # Load model state
        self.load_state_dict(checkpoint['model_state_dict'])
        self.memory_state = checkpoint.get('memory_state', {})
        
        # Restore configuration
        if 'config' in checkpoint:
            self.config.config = checkpoint['config']
        
        logger.info(f"Checkpoint loaded from {path}, epoch {checkpoint.get('epoch', 'unknown')}")
        
        return checkpoint


def create_hhgt_model(node_types: Dict[str, int],
                     edge_types: Dict[Tuple[str, str, str], int],
                     config_path: str = None,
                     config: Union[hHGTNConfig, str, Dict] = None,
                     **kwargs) -> hHGTN:
    """
    Factory function to create hHGTN model with configuration.
    
    Args:
        node_types: Dictionary mapping node type names to feature dimensions
        edge_types: Dictionary mapping edge type tuples to feature dimensions  
        config_path: Path to YAML configuration file
        config: Configuration object, dict, or path
        **kwargs: Additional configuration overrides
        
    Returns:
        Configured hHGTN model
    """
    
    # Use provided config or config_path
    if config is not None:
        model_config = config
    elif config_path is not None:
        model_config = config_path
    else:
        # Use default config
        default_config_path = Path(__file__).parent.parent.parent / "configs" / "stage9.yaml"
        model_config = str(default_config_path)
    
    # Create and return model
    model = hHGTN(
        node_types=node_types,
        edge_types=edge_types, 
        config=model_config,
        **kwargs
    )
    
    return model


# Utility functions for model creation and management

def get_model_size(model: hHGTN) -> Dict[str, int]:
    """Get model parameter counts by component."""
    sizes = {}
    total_params = 0
    
    for name, module in model.named_children():
        if module is not None:
            params = sum(p.numel() for p in module.parameters())
            sizes[name] = params
            total_params += params
    
    sizes['total'] = total_params
    return sizes


def print_model_summary(model: hHGTN):
    """Print detailed model summary."""
    print("=" * 80)
    print(f"hHGTN Model Summary")
    print("=" * 80)
    
    # Active components
    active_components = model._get_active_components()
    print(f"Active Components: {', '.join(active_components)}")
    print(f"Hidden Dimension: {model.hidden_dim}")
    print()
    
    # Parameter counts
    sizes = get_model_size(model)
    print("Parameter Counts:")
    for component, size in sizes.items():
        if component != 'total':
            print(f"  {component:15s}: {size:8,d} parameters")
    print(f"  {'Total':15s}: {sizes['total']:8,d} parameters")
    print()
    
    # Configuration summary
    print("Configuration Summary:")
    model_config = model.config.get('model', {})
    for key, value in model_config.items():
        if key.startswith('use_'):
            component = key.replace('use_', '').upper()
            status = "✓" if value else "✗"
            print(f"  {status} {component}")
    
    print("=" * 80)
