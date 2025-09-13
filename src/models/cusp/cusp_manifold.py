"""
Product-Manifold Operations and Cusp Pooling for CUSP module
Implements manifold utilities and attention-based pooling per STAGE8_CUSP_Reference §Phase4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List, Union
from torch_geometric.utils import to_dense_batch


class ManifoldUtils:
    """
    Utility functions for product-manifold operations.
    
    # Implements manifold operations per STAGE8_CUSP_Reference §Phase4
    
    Supports Euclidean, hyperbolic, and spherical manifold operations
    for curvature-aware graph embedding.
    """
    
    @staticmethod
    def euclidean_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Euclidean distance in flat space."""
        return torch.norm(x - y, dim=-1, keepdim=True)
    
    @staticmethod
    def hyperbolic_distance(
        x: torch.Tensor, 
        y: torch.Tensor, 
        curvature: float = -1.0,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Hyperbolic distance in Poincaré ball model.
        
        # Implements hyperbolic geometry per §Phase4
        
        Args:
            x, y: points in Poincaré ball (norm < 1)
            curvature: negative curvature parameter
            eps: numerical stability
            
        Returns:
            Hyperbolic distance between points
        """
        K = abs(curvature)
        sqrt_K = math.sqrt(K)
        
        # Ensure points are in Poincaré ball
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        y_norm = torch.norm(y, dim=-1, keepdim=True)
        
        x_proj = x / torch.clamp(x_norm / (1 - eps), min=1.0)
        y_proj = y / torch.clamp(y_norm / (1 - eps), min=1.0)
        
        # Simplified hyperbolic distance (avoid numerical issues with acosh)
        diff_norm_sq = torch.sum((x_proj - y_proj) ** 2, dim=-1, keepdim=True)
        x_norm_sq = torch.sum(x_proj ** 2, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y_proj ** 2, dim=-1, keepdim=True)
        
        # Use artanh formula which is more stable
        numerator = 2 * diff_norm_sq
        denominator = (1 - x_norm_sq) * (1 - y_norm_sq) + eps
        
        ratio = numerator / denominator
        
        # Use atanh approximation for stability: atanh(x) ≈ x + x^3/3 for small x
        ratio_clamped = torch.clamp(ratio, 0, 0.99)  # Ensure in valid range
        distance = (2 / sqrt_K) * torch.atanh(torch.sqrt(ratio_clamped) + eps)
        
        return torch.clamp(distance, min=0.0, max=10.0)  # Ensure non-negative
    
    @staticmethod
    def spherical_distance(
        x: torch.Tensor, 
        y: torch.Tensor, 
        radius: float = 1.0,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Spherical distance on sphere surface.
        
        # Implements spherical geometry per §Phase4
        
        Args:
            x, y: points on sphere (normalized)
            radius: sphere radius
            eps: numerical stability
            
        Returns:
            Great circle distance between points
        """
        # Normalize to sphere surface
        x_norm = F.normalize(x, dim=-1, eps=eps)
        y_norm = F.normalize(y, dim=-1, eps=eps)
        
        # Dot product for angle
        cos_angle = torch.sum(x_norm * y_norm, dim=-1, keepdim=True)
        cos_angle = torch.clamp(cos_angle, -1 + eps, 1 - eps)
        
        # Great circle distance
        angle = torch.acos(cos_angle)
        distance = radius * angle
        
        return distance
    
    @staticmethod
    def project_to_manifold(
        x: torch.Tensor,
        manifold_type: str,
        curvature: float = 0.0,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Project embeddings to specified manifold.
        
        # Implements manifold projection per §Phase4
        
        Args:
            x: input embeddings
            manifold_type: "euclidean", "hyperbolic", or "spherical"
            curvature: manifold curvature parameter
            eps: numerical stability
            
        Returns:
            Projected embeddings on manifold
        """
        if manifold_type == "euclidean":
            return x  # No projection needed
        
        elif manifold_type == "hyperbolic":
            # Project to Poincaré ball with stricter bounds
            norm = torch.norm(x, dim=-1, keepdim=True)
            max_norm = 1.0 - eps * 10  # More conservative bound
            # Use safe division and scaling
            scale = torch.where(norm > max_norm, max_norm / torch.clamp(norm, min=eps), torch.ones_like(norm))
            return x * scale
        
        elif manifold_type == "spherical":
            # Project to unit sphere
            return F.normalize(x, dim=-1, eps=eps)
        
        else:
            raise ValueError(f"Unknown manifold type: {manifold_type}")


class CuspAttentionPooling(nn.Module):
    """
    Cusp attention pooling with manifold-aware operations.
    
    # Implements attention pooling per STAGE8_CUSP_Reference §Phase4 Equations 6-8
    
    Combines curvature information with attention mechanisms
    for graph-level representations.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        manifold_types: List[str] = ["euclidean", "hyperbolic", "spherical"],
        dropout: float = 0.1,
        curvature_weight: float = 0.1
    ):
        """
        Initialize cusp attention pooling.
        
        Args:
            input_dim: input feature dimension
            hidden_dim: hidden attention dimension
            num_heads: number of attention heads
            manifold_types: list of manifolds for product space
            dropout: dropout probability
            curvature_weight: weight for curvature information
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.manifold_types = manifold_types
        self.curvature_weight = curvature_weight
        
        # Ensure num_heads divides input_dim
        if input_dim % num_heads != 0:
            # Adjust num_heads to be a divisor of input_dim
            valid_heads = [h for h in [1, 2, 4, 8, 16] if input_dim % h == 0]
            num_heads = valid_heads[-1] if valid_heads else 1
        
        # Multi-head attention components
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in manifold_types
        ])
        
        # Manifold-specific projections
        self.manifold_projections = nn.ModuleList([
            nn.Linear(input_dim, input_dim)
            for _ in manifold_types
        ])
        
        # Curvature integration
        self.curvature_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * len(manifold_types), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.manifold_utils = ManifoldUtils()
    
    def forward(
        self,
        x: torch.Tensor,
        node_curvature: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for cusp attention pooling.
        
        # Implements Equations 6-8 from STAGE8_CUSP_Reference §Phase4
        
        Args:
            x: [n, d] node features
            node_curvature: [n] node curvature values
            batch: [n] batch assignment for graph pooling
            edge_index: [2, m] edge connectivity (optional)
            
        Returns:
            pooled_graph: [batch_size, d] graph-level representation
            attention_weights: dict of attention weights per manifold
        """
        device = x.device
        n_nodes, input_dim = x.shape
        
        # Convert to dense batches if batch indices provided
        if batch is not None:
            x_dense, mask = to_dense_batch(x, batch)  # [batch_size, max_nodes, d]
            curvature_dense, _ = to_dense_batch(node_curvature.unsqueeze(-1), batch)
            curvature_dense = curvature_dense.squeeze(-1)  # [batch_size, max_nodes]
        else:
            # Single graph case
            x_dense = x.unsqueeze(0)  # [1, n_nodes, d]
            curvature_dense = node_curvature.unsqueeze(0)  # [1, n_nodes]
            mask = torch.ones(1, n_nodes, device=device, dtype=torch.bool)
        
        batch_size, max_nodes, _ = x_dense.shape
        
        # Curvature-aware weighting
        curvature_weights = self.curvature_mlp(curvature_dense.unsqueeze(-1))  # [batch_size, max_nodes, d]
        x_weighted = x_dense * (1 + self.curvature_weight * curvature_weights)
        
        # Process each manifold
        manifold_outputs = []
        attention_weights = {}
        
        for i, (manifold_type, attention_layer, projection) in enumerate(
            zip(self.manifold_types, self.attention_layers, self.manifold_projections)
        ):
            # Project to manifold
            x_manifold = projection(x_weighted)
            x_manifold = self.manifold_utils.project_to_manifold(x_manifold, manifold_type)
            
            # Self-attention within manifold
            # Use x_manifold as query, key, and value
            attn_output, attn_weights = attention_layer(
                x_manifold, x_manifold, x_manifold,
                key_padding_mask=~mask,
                need_weights=True
            )
            
            # Manifold-aware pooling
            if manifold_type == "euclidean":
                # Standard attention pooling
                pooled = self._attention_pool(attn_output, attn_weights, mask)
            
            elif manifold_type == "hyperbolic":
                # Hyperbolic centroid
                pooled = self._hyperbolic_pool(attn_output, attn_weights, mask)
            
            elif manifold_type == "spherical":
                # Spherical centroid
                pooled = self._spherical_pool(attn_output, attn_weights, mask)
            
            manifold_outputs.append(pooled)
            attention_weights[f"{manifold_type}_attention"] = attn_weights.mean(dim=1)  # Average over heads
        
        # Fuse manifold representations
        concatenated = torch.cat(manifold_outputs, dim=-1)  # [batch_size, d * num_manifolds]
        pooled_graph = self.fusion(concatenated)  # [batch_size, d]
        
        return pooled_graph, attention_weights
    
    def _attention_pool(
        self,
        features: torch.Tensor,
        attention_weights: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Standard attention-weighted pooling."""
        batch_size, max_nodes, dim = features.shape
        
        # Average attention weights across heads and queries to get node importance
        # attention_weights: [batch_size, num_heads, max_nodes, max_nodes]
        # We want to pool over the second dimension (max_nodes) to get graph representation
        
        # Take mean across heads and average over queries (rows) to get node importance
        pooling_weights = attention_weights.mean(dim=1)  # [batch_size, max_nodes, max_nodes]
        pooling_weights = pooling_weights.mean(dim=1, keepdim=True)  # [batch_size, 1, max_nodes]
        
        # Apply mask
        pooling_weights = pooling_weights.masked_fill(~mask.unsqueeze(1), 0)
        
        # Normalize weights
        pooling_weights = F.softmax(pooling_weights, dim=-1)
        
        # Weighted sum: [batch_size, 1, max_nodes] @ [batch_size, max_nodes, dim] -> [batch_size, 1, dim]
        pooled = torch.bmm(pooling_weights, features).squeeze(1)  # [batch_size, dim]
        
        return pooled
    
    def _hyperbolic_pool(
        self,
        features: torch.Tensor,
        attention_weights: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """Hyperbolic centroid pooling."""
        batch_size, max_nodes, dim = features.shape
        
        # Compute attention weights
        pooling_weights = attention_weights.mean(dim=1).mean(dim=1)  # [batch_size, max_nodes]
        pooling_weights = pooling_weights.masked_fill(~mask, 0)
        pooling_weights = F.softmax(pooling_weights, dim=-1)
        
        # Hyperbolic centroid (approximation)
        centroids = []
        
        for b in range(batch_size):
            valid_mask = mask[b]
            if not valid_mask.any():
                centroids.append(torch.zeros(dim, device=features.device))
                continue
            
            valid_features = features[b, valid_mask]  # [valid_nodes, dim]
            valid_weights = pooling_weights[b, valid_mask]  # [valid_nodes]
            
            # Weighted average in tangent space (approximation)
            weighted_sum = torch.sum(valid_weights.unsqueeze(-1) * valid_features, dim=0)
            
            # Project back to Poincaré ball
            centroid = self.manifold_utils.project_to_manifold(
                weighted_sum.unsqueeze(0), "hyperbolic"
            ).squeeze(0)
            
            centroids.append(centroid)
        
        return torch.stack(centroids)
    
    def _spherical_pool(
        self,
        features: torch.Tensor,
        attention_weights: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Spherical centroid pooling."""
        batch_size, max_nodes, dim = features.shape
        
        # Compute attention weights
        pooling_weights = attention_weights.mean(dim=1).mean(dim=1)  # [batch_size, max_nodes]
        pooling_weights = pooling_weights.masked_fill(~mask, 0)
        pooling_weights = F.softmax(pooling_weights, dim=-1)
        
        # Spherical centroid
        centroids = []
        
        for b in range(batch_size):
            valid_mask = mask[b]
            if not valid_mask.any():
                centroids.append(torch.zeros(dim, device=features.device))
                continue
            
            valid_features = features[b, valid_mask]  # [valid_nodes, dim]
            valid_weights = pooling_weights[b, valid_mask]  # [valid_nodes]
            
            # Weighted average on sphere
            weighted_sum = torch.sum(valid_weights.unsqueeze(-1) * valid_features, dim=0)
            
            # Project to sphere
            centroid = F.normalize(weighted_sum, dim=-1)
            
            centroids.append(centroid)
        
        return torch.stack(centroids)


class ProductManifoldEmbedding(nn.Module):
    """
    Product manifold embedding layer.
    
    # Implements product-manifold operations per STAGE8_CUSP_Reference §Phase4
    
    Creates embeddings in product space of multiple manifolds
    with curvature-dependent mixing.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        manifold_types: List[str] = ["euclidean", "hyperbolic", "spherical"],
        curvature_dependent: bool = True,
        learnable_curvatures: bool = True
    ):
        """
        Initialize product manifold embedding.
        
        Args:
            input_dim: input feature dimension
            output_dim: output embedding dimension per manifold
            manifold_types: list of manifold types
            curvature_dependent: whether to use curvature for mixing
            learnable_curvatures: whether curvatures are learnable
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.manifold_types = manifold_types
        self.curvature_dependent = curvature_dependent
        
        # Embedding layers for each manifold
        self.manifold_embeddings = nn.ModuleList([
            nn.Linear(input_dim, output_dim)
            for _ in manifold_types
        ])
        
        # Learnable curvature parameters
        if learnable_curvatures:
            self.curvature_params = nn.ParameterDict({
                f"{manifold}_curvature": nn.Parameter(torch.tensor(0.0))
                for manifold in manifold_types
            })
        else:
            # Use individual buffers for each curvature
            self.register_buffer("euclidean_curvature", torch.tensor(0.0))
            self.register_buffer("hyperbolic_curvature", torch.tensor(-1.0))
            self.register_buffer("spherical_curvature", torch.tensor(1.0))
            
            # For compatibility, create a property to mimic the dict interface
            self.curvature_params = {
                "euclidean_curvature": self.euclidean_curvature,
                "hyperbolic_curvature": self.hyperbolic_curvature,
                "spherical_curvature": self.spherical_curvature
            }
        
        # Curvature-dependent mixing weights
        if curvature_dependent:
            self.mixing_network = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, len(manifold_types)),
                nn.Softmax(dim=-1)
            )
        
        self.manifold_utils = ManifoldUtils()
    
    def forward(
        self,
        x: torch.Tensor,
        node_curvature: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for product manifold embedding.
        
        Args:
            x: [n, input_dim] input features
            node_curvature: [n] node curvature values (optional)
            
        Returns:
            manifold_embeddings: dict of embeddings per manifold
        """
        embeddings = {}
        
        # Generate embeddings for each manifold
        for i, (manifold_type, embedding_layer) in enumerate(
            zip(self.manifold_types, self.manifold_embeddings)
        ):
            # Linear transformation
            h = embedding_layer(x)
            
            # Get curvature parameter
            curvature = self.curvature_params.get(
                f"{manifold_type}_curvature",
                torch.tensor(0.0, device=x.device)
            )
            
            # Project to manifold
            h_manifold = self.manifold_utils.project_to_manifold(
                h, manifold_type, curvature.item()
            )
            
            embeddings[manifold_type] = h_manifold
        
        # Curvature-dependent mixing (if enabled and curvature provided)
        if self.curvature_dependent and node_curvature is not None:
            mixing_weights = self.mixing_network(node_curvature.unsqueeze(-1))  # [n, num_manifolds]
            
            # Weighted combination of manifold embeddings
            mixed_embedding = torch.zeros_like(embeddings[self.manifold_types[0]])
            
            for i, manifold_type in enumerate(self.manifold_types):
                weight = mixing_weights[:, i:i+1]  # [n, 1]
                mixed_embedding += weight * embeddings[manifold_type]
            
            embeddings["mixed"] = mixed_embedding
        
        return embeddings


def validate_product_manifold_operations(
    embeddings: Dict[str, torch.Tensor],
    expected_manifolds: List[str],
    expected_shape: Tuple[int, int],
    eps: float = 1e-6
) -> bool:
    """
    Validate product manifold operations.
    
    # Implements validation per STAGE8_CUSP_Reference §Phase4
    
    Args:
        embeddings: dict of manifold embeddings
        expected_manifolds: expected manifold types
        expected_shape: expected (n_nodes, dim) shape
        eps: numerical tolerance
        
    Returns:
        is_valid: True if validation passes
    """
    try:
        # Check all expected manifolds present
        for manifold in expected_manifolds:
            if manifold not in embeddings:
                print(f"Missing manifold: {manifold}")
                return False
        
        # Check shapes and properties
        for manifold, embedding in embeddings.items():
            # Shape check
            if embedding.shape != expected_shape:
                print(f"Wrong shape for {manifold}: expected {expected_shape}, got {embedding.shape}")
                return False
            
            # Finite values check
            if not torch.all(torch.isfinite(embedding)):
                print(f"Non-finite values in {manifold} embedding")
                return False
            
            # Check for diversity (not all zeros)
            if torch.allclose(embedding, torch.zeros_like(embedding), atol=eps):
                print(f"Diversity validation failed: all {manifold} encodings are zero")
                return False
            
            # Manifold-specific constraints
            if manifold == "hyperbolic":
                # Check Poincaré ball constraint with tolerance
                norms = torch.norm(embedding, dim=-1)
                if torch.any(norms >= 1.0 - eps):
                    print(f"Hyperbolic embedding violates Poincaré ball constraint: max norm {torch.max(norms)}")
                    return False
            
            elif manifold == "spherical":
                # Check unit sphere constraint
                norms = torch.norm(embedding, dim=-1)
                if not torch.allclose(norms, torch.ones_like(norms), atol=eps):
                    print(f"Spherical embedding violates unit sphere constraint")
                    return False
        
        print("Product manifold operations validation passed")
        return True
        
    except Exception as e:
        print(f"Product manifold validation failed with exception: {e}")
        return False
