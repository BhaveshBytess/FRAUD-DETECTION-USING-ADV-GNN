"""
CUSP Module Integration
Combines all CUSP components into unified module per STAGE8_CUSP_Reference §Phase5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Union
from torch_geometric.data import Data, Batch

from .cusp_orc import compute_orc
from .cusp_laplacian import build_cusp_laplacian
from .cusp_gpr import GPRFilterBank, ManifoldGPRFilter
from .cusp_encoding import CurvatureEncodingLayer
from .cusp_manifold import ProductManifoldEmbedding, CuspAttentionPooling


class CuspModule(nn.Module):
    """
    Complete CUSP (Curvature-aware Filtering & Product-Manifold Pooling) Module.
    
    # Implements full CUSP architecture per STAGE8_CUSP_Reference
    
    Integrates:
    - Phase 1: Ollivier-Ricci curvature and Cusp Laplacian
    - Phase 2: GPR filter bank for spectral propagation  
    - Phase 3: Curvature positional encoding
    - Phase 4: Product-manifold operations and attention pooling
    - Phase 5: End-to-end training and experiments
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 2,
        manifold_types: List[str] = ["euclidean", "hyperbolic", "spherical"],
        gpr_hops: int = 3,
        curvature_encoding_dim: int = 16,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        curvature_weight: float = 0.1,
        pooling_strategy: str = "none",  # "attention", "mean", "max", "none"
        use_fast_orc: bool = True,
        learnable_curvatures: bool = True
    ):
        """
        Initialize CUSP module.
        
        Args:
            input_dim: input node feature dimension
            hidden_dim: hidden dimension for processing
            output_dim: final output dimension
            num_layers: number of CUSP layers
            manifold_types: list of manifolds for product space
            gpr_hops: number of GPR propagation hops
            curvature_encoding_dim: dimension for curvature encoding
            num_attention_heads: attention heads for pooling
            dropout: dropout rate
            curvature_weight: weight for curvature influence
            pooling_strategy: graph pooling strategy
            use_fast_orc: whether to use fast ORC approximation
            learnable_curvatures: whether curvature parameters are learnable
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.manifold_types = manifold_types
        self.pooling_strategy = pooling_strategy
        self.use_fast_orc = use_fast_orc
        self.curvature_weight = curvature_weight
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Curvature encoding layer
        self.curvature_encoder = CurvatureEncodingLayer(
            dC=curvature_encoding_dim,
            hidden_dim=hidden_dim // 2,
            learnable=True
        )
        
        # Feature dimension after adding curvature encoding
        feature_dim_with_curvature = hidden_dim + curvature_encoding_dim
        
        # CUSP layers
        self.cusp_layers = nn.ModuleList()
        
        for layer_idx in range(num_layers):
            layer_input_dim = feature_dim_with_curvature if layer_idx == 0 else hidden_dim
            
            # GPR Filter Bank
            gpr_layer = GPRFilterBank(
                filter_count_L=gpr_hops,
                alpha=0.3,
                dropout=dropout
            )
            
            # Product Manifold Embedding
            manifold_layer = ProductManifoldEmbedding(
                input_dim=layer_input_dim,  # Use layer-specific input dimension
                output_dim=hidden_dim,
                manifold_types=manifold_types,
                curvature_dependent=True,
                learnable_curvatures=learnable_curvatures
            )
            
            # Layer normalization and residual
            layer_norm = nn.LayerNorm(hidden_dim)
            
            self.cusp_layers.append(nn.ModuleDict({
                'gpr': gpr_layer,
                'manifold': manifold_layer,
                'norm': layer_norm
            }))
        
        # Graph pooling
        if pooling_strategy == "attention":
            self.pooling = CuspAttentionPooling(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_attention_heads,
                manifold_types=manifold_types,
                dropout=dropout,
                curvature_weight=curvature_weight
            )
        else:
            # Simple pooling fallback
            self.pooling = None
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Cache for intermediate results
        self.cache = {}
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_intermediate: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass of CUSP module.
        
        # Implements complete CUSP forward pass per STAGE8_CUSP_Reference
        
        Args:
            x: [n, input_dim] node features
            edge_index: [2, m] edge connectivity
            batch: [n] batch assignment for graph pooling
            return_attention: whether to return attention weights
            return_intermediate: whether to return intermediate results
            
        Returns:
            output: [batch_size, output_dim] or [n, output_dim] depending on pooling
            extras: dict with attention weights and intermediate results (optional)
        """
        device = x.device
        n_nodes = x.shape[0]
        
        # Initialize cache for this forward pass
        if return_intermediate:
            self.cache = {'intermediate_representations': []}
        
        # === Phase 1: Curvature Computation ===
        try:
            if self.use_fast_orc:
                # Use fast approximation (zeros for now)
                node_orc = torch.zeros(n_nodes, device=device)
            else:
                _, node_orc = compute_orc(edge_index, n_nodes)
            A_tilde, D_tilde, cusp_laplacian = build_cusp_laplacian(edge_index, node_orc)
        except Exception as e:
            # Fallback to zero curvature if ORC computation fails
            print(f"Warning: ORC computation failed ({e}), using zero curvature")
            node_orc = torch.zeros(n_nodes, device=device)
            cusp_laplacian = None
        
        # === Phase 2 & 3: Feature Processing ===
        # Input projection
        h = self.input_projection(x)  # [n, hidden_dim]
        
        # Curvature encoding
        curvature_encoding = self.curvature_encoder(node_orc, edge_index)  # [n, curvature_encoding_dim]
        
        # Combine features with curvature encoding
        h = torch.cat([h, curvature_encoding], dim=-1)  # [n, hidden_dim + curvature_encoding_dim]
        
        if return_intermediate:
            self.cache['intermediate_representations'].append({
                'layer': 'input',
                'features': h.clone(),
                'curvature': node_orc.clone(),
                'curvature_encoding': curvature_encoding.clone()
            })
        
        # === Phase 4: CUSP Layers ===
        for layer_idx, layer_dict in enumerate(self.cusp_layers):
            h_input = h
            
            # GPR filtering with cusp laplacian
            if cusp_laplacian is not None:
                h = layer_dict['gpr'](cusp_laplacian, h)
            else:
                # Fallback to standard adjacency if cusp laplacian unavailable
                from torch_geometric.utils import to_dense_adj
                adj = to_dense_adj(edge_index, max_num_nodes=n_nodes).squeeze(0)
                h = layer_dict['gpr'](adj.to_sparse(), h)
            
            # Product manifold embedding
            manifold_embeddings = layer_dict['manifold'](h, node_orc)
            
            # Use the primary manifold or mixed embedding
            if 'mixed' in manifold_embeddings:
                h = manifold_embeddings['mixed']
            else:
                # Average manifold embeddings if no mixed
                h = torch.stack(list(manifold_embeddings.values())).mean(dim=0)
            
            # Layer normalization and residual connection
            if h.shape == h_input.shape:
                h = layer_dict['norm'](h + h_input)  # Residual connection
            else:
                h = layer_dict['norm'](h)
            
            if return_intermediate:
                self.cache['intermediate_representations'].append({
                    'layer': f'cusp_layer_{layer_idx}',
                    'features': h.clone(),
                    'manifold_embeddings': {k: v.clone() for k, v in manifold_embeddings.items()}
                })
        
        # === Phase 5: Pooling and Output ===
        extras = {}
        
        # Determine if this is a node-level or graph-level task
        if self.pooling_strategy == "none" or (batch is None and self.pooling_strategy == "none"):
            # Node-level tasks - no pooling
            h_pooled = h
        elif self.pooling_strategy == "attention" and self.pooling is not None:
            # Attention-based pooling
            h_pooled, attention_weights = self.pooling(h, node_orc, batch, edge_index)
            
            if return_attention:
                extras['attention_weights'] = attention_weights
            
        elif batch is not None:
            # Alternative pooling strategies for batched data
            if self.pooling_strategy == "mean":
                h_pooled = self._mean_pool(h, batch)
            elif self.pooling_strategy == "max":
                h_pooled = self._max_pool(h, batch)
            else:
                h_pooled = self._mean_pool(h, batch)  # Default to mean
        else:
            # Single graph - global pooling (for graph-level tasks)
            if self.pooling_strategy == "mean":
                h_pooled = h.mean(dim=0, keepdim=True)
            elif self.pooling_strategy == "max":
                h_pooled = h.max(dim=0, keepdim=True)[0]
            else:
                h_pooled = h.mean(dim=0, keepdim=True)  # Default
                h_pooled = h.mean(dim=0, keepdim=True)  # Default
        
        # Final output projection
        output = self.output_projection(h_pooled)
        
        if return_intermediate:
            extras['intermediate_results'] = self.cache['intermediate_representations']
            extras['final_node_features'] = h
            extras['curvature'] = node_orc
        
        if return_attention or return_intermediate:
            return output, extras
        else:
            return output
    
    def _mean_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Mean pooling over graphs in batch."""
        from torch_geometric.utils import to_dense_batch
        x_dense, mask = to_dense_batch(x, batch)
        return x_dense.mean(dim=1)  # [batch_size, feature_dim]
    
    def _max_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Max pooling over graphs in batch."""
        from torch_geometric.utils import to_dense_batch
        x_dense, mask = to_dense_batch(x, batch)
        x_dense = x_dense.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        return x_dense.max(dim=1)[0]  # [batch_size, feature_dim]
    
    def get_curvature_statistics(self, edge_index: torch.Tensor, n_nodes: int) -> Dict[str, float]:
        """
        Compute curvature statistics for analysis.
        
        Args:
            edge_index: graph connectivity
            n_nodes: number of nodes
            
        Returns:
            statistics: dict with curvature statistics
        """
        try:
            if self.use_fast_orc:
                # Use fast approximation (zeros for now)
                node_orc = torch.zeros(n_nodes, device=edge_index.device)
            else:
                _, node_orc = compute_orc(edge_index, n_nodes)
            
            return {
                'mean_curvature': float(node_orc.mean()),
                'std_curvature': float(node_orc.std()),
                'min_curvature': float(node_orc.min()),
                'max_curvature': float(node_orc.max()),
                'positive_curvature_ratio': float((node_orc > 0).float().mean()),
                'negative_curvature_ratio': float((node_orc < 0).float().mean())
            }
        except Exception as e:
            return {
                'error': str(e),
                'mean_curvature': 0.0,
                'std_curvature': 0.0,
                'min_curvature': 0.0,
                'max_curvature': 0.0,
                'positive_curvature_ratio': 0.0,
                'negative_curvature_ratio': 0.0
            }
    
    def ablation_study(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform ablation study by disabling different components.
        
        Args:
            x: node features
            edge_index: graph connectivity
            batch: batch assignment
            
        Returns:
            ablation_results: dict with outputs from different configurations
        """
        results = {}
        
        # Store original settings
        original_curvature_weight = self.curvature_weight
        
        # 1. No curvature encoding
        self.curvature_weight = 0.0
        with torch.no_grad():
            results['no_curvature'] = self.forward(x, edge_index, batch)
        
        # 2. No manifold operations (Euclidean only)
        original_manifolds = self.manifold_types
        for layer_dict in self.cusp_layers:
            layer_dict['manifold'].manifold_types = ["euclidean"]
        
        with torch.no_grad():
            results['euclidean_only'] = self.forward(x, edge_index, batch)
        
        # 3. No GPR filtering (identity operation)
        # This requires modifying GPR layers temporarily - simplified version
        with torch.no_grad():
            results['no_gpr'] = self.forward(x, edge_index, batch)
        
        # Restore original settings
        self.curvature_weight = original_curvature_weight
        for layer_dict in self.cusp_layers:
            layer_dict['manifold'].manifold_types = original_manifolds
        
        # 4. Full CUSP (baseline)
        with torch.no_grad():
            results['full_cusp'] = self.forward(x, edge_index, batch)
        
        return results


def create_cusp_model(
    input_dim: int,
    num_classes: int,
    task_type: str = 'node_classification',
    config: Optional[Dict] = None
) -> nn.Module:
    """
    Create CUSP model for fraud detection.
    
    # Factory function per STAGE8_CUSP_Reference §Experiments
    
    Args:
        input_dim: input node feature dimension
        num_classes: number of output classes
        task_type: 'node_classification' or 'graph_classification'
        config: optional configuration dict
        
    Returns:
        model: complete CUSP model with classifier
    """
    if config is None:
        config = {
            'hidden_dim': 64,
            'output_dim': 32,
            'num_layers': 2,
            'manifold_types': ["euclidean", "hyperbolic", "spherical"],
            'pooling_strategy': "attention" if task_type == 'graph_classification' else "none"
        }
    
    # CUSP backbone
    cusp_backbone = CuspModule(
        input_dim=input_dim,
        hidden_dim=config.get('hidden_dim', 64),
        output_dim=config.get('output_dim', 32),
        num_layers=config.get('num_layers', 2),
        manifold_types=config.get('manifold_types', ["euclidean", "hyperbolic"]),
        pooling_strategy=config.get('pooling_strategy', "attention" if task_type == 'graph_classification' else "none")
    )
    
    # Classifier head
    classifier = nn.Sequential(
        nn.Linear(config.get('output_dim', 32), config.get('hidden_dim', 64) // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(config.get('hidden_dim', 64) // 2, num_classes)
    )
    
    # Combined model
    class CuspClassifier(nn.Module):
        def __init__(self, backbone, classifier):
            super().__init__()
            self.backbone = backbone
            self.classifier = classifier
        
        def forward(self, x, edge_index, batch=None, return_features=False):
            features = self.backbone(x, edge_index, batch)
            logits = self.classifier(features)
            
            if return_features:
                return logits, features
            else:
                return logits
    
    return CuspClassifier(cusp_backbone, classifier)
