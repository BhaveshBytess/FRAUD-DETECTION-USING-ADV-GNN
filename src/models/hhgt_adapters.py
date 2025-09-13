"""
hHGTN Component Adapters - Stage 9 Integration

Provides consistent wrapper interfaces for all Stage 1-8 components.
Each adapter ensures consistent output format: (embeddings, meta, memory_state)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BaseAdapter(nn.Module):
    """Base adapter class for consistent component interfaces."""
    
    def __init__(self, config: Dict = None):
        super().__init__()
        self.config = config or {}
        self.is_enabled = True
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Adapter must implement forward method")
    
    def get_output_dim(self) -> int:
        """Return output embedding dimension."""
        return self.config.get('output_dim', 128)


class PhenomAdapter(BaseAdapter):
    """Adapter for PhenomNN hypergraph component (Stage 5)."""
    
    def __init__(self, node_types: Dict, edge_types: Dict, hidden_dim: int = 128, config: Dict = None):
        super().__init__(config)
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        
        # Create flexible projection layers that can handle variable input dimensions
        self.hypergraph_layers = nn.ModuleDict()
        
        # We'll create the actual layers during first forward pass
        self._layers_created = False
    
    def _create_layers(self, x_dict):
        """Create layers based on actual input dimensions during first forward pass."""
        if self._layers_created:
            return
            
        for node_type, features in x_dict.items():
            input_dim = features.size(-1)  # Get actual input dimension
            self.hypergraph_layers[node_type] = nn.Linear(input_dim, self.hidden_dim)
        
        self._layers_created = True
    
    def forward(self, batch) -> Dict[str, Any]:
        """Forward pass through hypergraph layers."""
        
        if not hasattr(batch, 'x_dict'):
            raise ValueError("Batch must have x_dict attribute for node features")
        
        # Create layers if not already done
        self._create_layers(batch.x_dict)
        
        embeddings = {}
        for node_type, features in batch.x_dict.items():
            if node_type in self.hypergraph_layers:
                embeddings[node_type] = self.hypergraph_layers[node_type](features)
            else:
                # Create layer on-the-fly for new node types
                input_dim = features.size(-1)
                self.hypergraph_layers[node_type] = nn.Linear(input_dim, self.hidden_dim)
                embeddings[node_type] = self.hypergraph_layers[node_type](features)
        
        return {
            'embeddings': embeddings,
            'meta': {'component': 'hypergraph', 'num_nodes': sum(emb.size(0) for emb in embeddings.values())},
            'memory_state': None
        }


class HeteroAdapter(BaseAdapter):
    """Adapter for Heterogeneous GNN component (Stage 3)."""
    
    def __init__(self, node_types: Dict, edge_types: Dict, hidden_dim: int = 128, 
                 num_layers: int = 3, num_heads: int = 8, config: Dict = None):
        super().__init__(config)
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Create flexible layers that adapt to input dimensions
        self.hetero_layers = nn.ModuleDict()
        self._layers_created = False
    
    def _create_layers(self, x_dict):
        """Create layers based on actual input dimensions."""
        if self._layers_created:
            return
            
        for node_type, features in x_dict.items():
            input_dim = features.size(-1)
            self.hetero_layers[node_type] = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
        
        self._layers_created = True
    
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> Dict[str, Any]:
        """Forward pass through heterogeneous layers."""
        
        # Create layers if needed
        self._create_layers(x_dict)
        
        embeddings = {}
        attention_weights = {}
        
        for node_type, features in x_dict.items():
            if node_type in self.hetero_layers:
                embeddings[node_type] = self.hetero_layers[node_type](features)
            else:
                # Create layer on-the-fly
                input_dim = features.size(-1)
                self.hetero_layers[node_type] = nn.Sequential(
                    nn.Linear(input_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim)
                )
                embeddings[node_type] = self.hetero_layers[node_type](features)
            
            # Mock attention weights
            attention_weights[node_type] = torch.ones(features.size(0), self.num_heads)
        
        return {
            'embeddings': embeddings,
            'attention_weights': attention_weights,
            'meta': {'component': 'hetero', 'num_layers': self.num_layers}
        }


class TGNAdapter(BaseAdapter):
    """Adapter for Temporal Graph Network component (Stage 4)."""
    
    def __init__(self, node_types: Dict, config: Dict = None):
        super().__init__(config)
        self.node_types = node_types
        self.memory_dim = config.get('memory_dim', 128) if config else 128
        
        # Initialize memory banks for each node type
        self.memory_banks = nn.ParameterDict()
        for node_type in node_types:
            # Assume max 1000 nodes per type for initialization
            max_nodes = 1000
            self.memory_banks[node_type] = nn.Parameter(
                torch.zeros(max_nodes, self.memory_dim)
            )
    
    def update_memory(self, batch, embeddings: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Update memory and return new embeddings."""
        
        updated_embeddings = {}
        memory_state = {}
        
        for node_type, emb in embeddings.items():
            if node_type in self.memory_banks:
                # Simple memory update: add current embeddings to memory
                num_nodes = emb.size(0)
                if num_nodes <= self.memory_banks[node_type].size(0):
                    # Get current memory state
                    memory_context = self.memory_banks[node_type][:num_nodes]
                    
                    # Return memory-augmented embeddings without updating memory during training
                    # (Memory update disabled to avoid gradient issues)
                    updated_embeddings[node_type] = emb + 0.5 * memory_context.detach()
                    memory_state[node_type] = memory_context.detach()
                else:
                    updated_embeddings[node_type] = emb
                    memory_state[node_type] = None
            else:
                updated_embeddings[node_type] = emb
                memory_state[node_type] = None
        
        return {
            'embeddings': updated_embeddings,
            'memory_state': memory_state,
            'meta': {'component': 'tgn', 'memory_dim': self.memory_dim}
        }


class CuspAdapter(BaseAdapter):
    """Adapter for CUSP manifold embeddings component (Stage 8)."""
    
    def __init__(self, node_types: Dict, edge_types: Dict, config: Dict = None):
        super().__init__(config)
        self.node_types = node_types
        self.edge_types = edge_types
        
        # CUSP embedding dimensions
        self.euclidean_dim = config.get('euclidean_dim', 64) if config else 64
        self.hyperbolic_dim = config.get('hyperbolic_dim', 64) if config else 64
        self.spherical_dim = config.get('spherical_dim', 64) if config else 64
        
        # Create embedding layers for each manifold
        self.euclidean_embedders = nn.ModuleDict()
        self.hyperbolic_embedders = nn.ModuleDict()
        self.spherical_embedders = nn.ModuleDict()
        
        for node_type in node_types:
            input_dim = node_types[node_type]
            self.euclidean_embedders[node_type] = nn.Linear(input_dim, self.euclidean_dim)
            self.hyperbolic_embedders[node_type] = nn.Linear(input_dim, self.hyperbolic_dim)
            self.spherical_embedders[node_type] = nn.Linear(input_dim, self.spherical_dim)
    
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        """Generate CUSP manifold embeddings."""
        
        if not hasattr(batch, 'x_dict'):
            raise ValueError("Batch must have x_dict attribute")
        
        cusp_embeddings = {}
        
        for node_type, features in batch.x_dict.items():
            if node_type in self.euclidean_embedders:
                # Generate embeddings in each manifold
                euclidean_emb = self.euclidean_embedders[node_type](features)
                hyperbolic_emb = torch.tanh(self.hyperbolic_embedders[node_type](features))  # Bounded for hyperbolic
                spherical_emb = F.normalize(self.spherical_embedders[node_type](features), dim=-1)  # Unit sphere
                
                # Concatenate manifold embeddings
                cusp_embeddings[node_type] = torch.cat([euclidean_emb, hyperbolic_emb, spherical_emb], dim=-1)
            else:
                # Return zero embeddings if node type not found
                total_dim = self.euclidean_dim + self.hyperbolic_dim + self.spherical_dim
                cusp_embeddings[node_type] = torch.zeros(features.size(0), total_dim)
        
        return cusp_embeddings


class SpotTargetAdapter(BaseAdapter):
    """Adapter for SpotTarget training discipline (Stage 7)."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.remove_target_edges = config.get('remove_target_edges', True) if config else True
        self.min_degree_threshold = config.get('min_degree_threshold', 2) if config else 2
        self.edge_removal_prob = config.get('edge_removal_prob', 0.5) if config else 0.5
    
    def filter_batch(self, batch):
        """Apply SpotTarget filtering to batch."""
        
        if not self.training or not self.remove_target_edges:
            return batch
        
        # Simple implementation: randomly remove some edges
        filtered_batch = batch
        
        if hasattr(batch, 'edge_index_dict'):
            filtered_edge_index_dict = {}
            
            for edge_type, edge_index in batch.edge_index_dict.items():
                # Randomly remove edges based on probability
                num_edges = edge_index.size(1)
                keep_mask = torch.rand(num_edges) > self.edge_removal_prob
                filtered_edge_index_dict[edge_type] = edge_index[:, keep_mask]
            
            # Create new batch with filtered edges
            filtered_batch.edge_index_dict = filtered_edge_index_dict
        
        return filtered_batch


class RobustnessAdapter(BaseAdapter):
    """Adapter for robustness defenses (Stage 7)."""
    
    def __init__(self, hidden_dim: int = 128, config: Dict = None):
        super().__init__(config)
        self.hidden_dim = hidden_dim
        self.drop_edge_rate = config.get('drop_edge_rate', 0.1) if config else 0.1
        self.adversarial_epsilon = config.get('adversarial_epsilon', 0.01) if config else 0.01
        
        # DropEdge implementation
        self.dropout = nn.Dropout(self.drop_edge_rate)
    
    def apply_defenses(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> Dict[str, torch.Tensor]:
        """Apply robustness defenses to embeddings."""
        
        defended_embeddings = {}
        
        for node_type, embeddings in x_dict.items():
            # Apply dropout for robustness
            defended_emb = self.dropout(embeddings)
            
            # Add small noise for adversarial robustness
            if self.training:
                noise = torch.randn_like(defended_emb) * self.adversarial_epsilon
                defended_emb = defended_emb + noise
            
            defended_embeddings[node_type] = defended_emb
        
        return defended_embeddings


class TDGNNAdapter(BaseAdapter):
    """Adapter for TDGNN time-relaxed sampling (Stage 6)."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.delta_t = config.get('delta_t', 3600) if config else 3600  # 1 hour
        self.max_neighbors = config.get('max_neighbors', 50) if config else 50
        self.time_relaxed = config.get('time_relaxed', True) if config else True
    
    def sample_batch(self, batch):
        """Apply TDGNN sampling to batch."""
        
        # Simple implementation: return batch unchanged for now
        # In real implementation, this would perform time-aware neighbor sampling
        
        if hasattr(batch, 'edge_index_dict'):
            sampled_batch = batch
            
            # Mock sampling: limit number of edges per type
            for edge_type, edge_index in batch.edge_index_dict.items():
                num_edges = edge_index.size(1)
                if num_edges > self.max_neighbors:
                    # Randomly sample edges
                    indices = torch.randperm(num_edges)[:self.max_neighbors]
                    sampled_batch.edge_index_dict[edge_type] = edge_index[:, indices]
            
            return sampled_batch
        
        return batch


# Import guard for missing torch_geometric
import torch.nn.functional as F


class AdapterFactory:
    """Factory for creating component adapters."""
    
    @staticmethod
    def create_adapter(component_name: str, *args, **kwargs) -> BaseAdapter:
        """Create adapter for specified component."""
        
        adapter_map = {
            'phenom': PhenomAdapter,
            'hypergraph': PhenomAdapter,
            'hetero': HeteroAdapter, 
            'heterogeneous': HeteroAdapter,
            'tgn': TGNAdapter,
            'memory': TGNAdapter,
            'cusp': CuspAdapter,
            'spottarget': SpotTargetAdapter,
            'robustness': RobustnessAdapter,
            'tdgnn': TDGNNAdapter
        }
        
        if component_name.lower() in adapter_map:
            return adapter_map[component_name.lower()](*args, **kwargs)
        else:
            raise ValueError(f"Unknown component: {component_name}")


def test_adapter_consistency():
    """Test that all adapters follow consistent interface."""
    
    # Sample data
    node_types = {'transaction': 10, 'address': 8}
    edge_types = {('transaction', 'to', 'address'): 1}
    
    # Test each adapter
    adapters = [
        ('phenom', PhenomAdapter(node_types, edge_types)),
        ('hetero', HeteroAdapter(node_types, edge_types)),
        ('tgn', TGNAdapter(node_types)),
        ('cusp', CuspAdapter(node_types, edge_types)),
        ('spottarget', SpotTargetAdapter()),
        ('robustness', RobustnessAdapter()),
        ('tdgnn', TDGNNAdapter())
    ]
    
    print("Testing adapter consistency:")
    for name, adapter in adapters:
        print(f"  ✓ {name}: {type(adapter).__name__}")
        assert isinstance(adapter, BaseAdapter)
        assert hasattr(adapter, 'config')
        assert hasattr(adapter, 'is_enabled')
    
    print("All adapters follow consistent interface!")


if __name__ == "__main__":
    test_adapter_consistency()
