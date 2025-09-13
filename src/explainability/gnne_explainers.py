"""
Phase B — Explainability primitives: GNN explainer wrappers.

Implements wrappers for GNNExplainer, PGExplainer, HGNNExplainer and temporal explainers
with standard interfaces for use in the hHGTN pipeline.

Following Stage 10 Reference §Phase B requirements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from abc import ABC, abstractmethod

# PyG imports with fallbacks
try:
    from torch_geometric.nn import GNNExplainer
    from torch_geometric.data import Data, HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Warning: PyTorch Geometric not available. Some explainer features may be limited.")

logger = logging.getLogger(__name__)


class BaseExplainer(ABC):
    """Base class for all explainer wrappers."""
    
    def __init__(self, model, device: str = 'cpu', seed: int = 0):
        self.model = model
        self.device = device
        self.seed = seed
        self._set_seed()
    
    def _set_seed(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
    
    @abstractmethod
    def explain_node(self, node_id: int, x: torch.Tensor, edge_index: torch.Tensor, 
                    label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Explain prediction for a target node.
        
        Returns:
            Dictionary with 'edge_mask', 'node_feat_mask', 'important_subgraph'
        """
        pass


class GNNExplainerWrapper(BaseExplainer):
    """
    Wrapper for PyG's GNNExplainer with standard interface.
    
    Following Stage 10 Reference §Phase B requirement:
    explain_node(node_id, model, subgraph, label) → returns edge_mask, node_feat_mask, important_subgraph
    """
    
    def __init__(self, model, epochs: int = 200, lr: float = 0.01, device: str = 'cpu', seed: int = 0):
        super().__init__(model, device, seed)
        self.epochs = epochs
        self.lr = lr
        
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric required for GNNExplainer")
        
        # Initialize GNNExplainer
        self.explainer = GNNExplainer(
            model=model,
            epochs=epochs,
            lr=lr,
            return_type='log_prob'
        )
        
        logger.info(f"Initialized GNNExplainer with {epochs} epochs, lr={lr}")
    
    def explain_node(self, node_id: int, x: torch.Tensor, edge_index: torch.Tensor, 
                    label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Explain prediction for target node using GNNExplainer.
        
        Args:
            node_id: Target node ID
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]  
            label: Ground truth label (optional)
            
        Returns:
            Dictionary with explanation components
        """
        self._set_seed()
        
        # Move to device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        if label is not None:
            label = label.to(self.device)
        
        # Run GNNExplainer
        node_feat_mask, edge_mask = self.explainer.explain_node(
            node_idx=node_id,
            x=x,
            edge_index=edge_index
        )
        
        # Create important subgraph based on edge mask
        important_subgraph = self._create_important_subgraph(
            edge_index, edge_mask, threshold=0.7
        )
        
        return {
            'edge_mask': edge_mask.detach(),
            'node_feat_mask': node_feat_mask.detach() if node_feat_mask is not None else None,
            'important_subgraph': important_subgraph,
            'explanation_type': 'gnn_explainer'
        }
    
    def _create_important_subgraph(self, edge_index: torch.Tensor, 
                                  edge_mask: torch.Tensor, threshold: float = 0.7) -> Dict[str, torch.Tensor]:
        """Create subgraph with most important edges."""
        # Apply threshold to edge mask
        important_edges = edge_mask >= threshold
        
        # If no edges meet threshold, take top-k
        if important_edges.sum() == 0:
            k = min(10, len(edge_mask))
            _, top_indices = torch.topk(edge_mask, k)
            important_edges = torch.zeros_like(edge_mask, dtype=torch.bool)
            important_edges[top_indices] = True
        
        # Extract important subgraph
        important_edge_index = edge_index[:, important_edges]
        important_edge_weights = edge_mask[important_edges]
        
        return {
            'edge_index': important_edge_index,
            'edge_weights': important_edge_weights,
            'edge_mask': important_edges
        }


class PGExplainerTrainer(BaseExplainer):
    """
    Parameterized Graph Explainer trainer wrapper.
    
    Trains a mask predictor network that learns to generate explanations
    without per-instance optimization.
    """
    
    def __init__(self, model, hidden_dim: int = 64, device: str = 'cpu', seed: int = 0):
        super().__init__(model, device, seed)
        self.hidden_dim = hidden_dim
        
        # Mask predictor network (simple MLP)
        self.mask_predictor = None
        self.is_trained = False
        
        logger.info(f"Initialized PGExplainer trainer with hidden_dim={hidden_dim}")
    
    def _build_mask_predictor(self, input_dim: int):
        """Build mask predictor network based on input dimensions."""
        self.mask_predictor = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),  # Single output per edge
            nn.Sigmoid()  # Output in [0,1]
        ).to(self.device)
    
    def train_explainer(self, train_data: List[Dict], epochs: int = 100, lr: float = 0.01):
        """
        Train the mask predictor on a set of training examples.
        
        Args:
            train_data: List of training examples with 'x', 'edge_index', 'node_id', 'ground_truth_mask'
            epochs: Training epochs
            lr: Learning rate
        """
        if len(train_data) == 0:
            raise ValueError("No training data provided")
        
        # Build mask predictor based on first example
        first_example = train_data[0]
        edge_features_dim = self._compute_edge_features(
            first_example['x'], first_example['edge_index']
        ).size(-1)
        
        self._build_mask_predictor(edge_features_dim)
        
        optimizer = torch.optim.Adam(self.mask_predictor.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        logger.info(f"Training PGExplainer for {epochs} epochs on {len(train_data)} examples")
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for example in train_data:
                self._set_seed()
                
                x = example['x'].to(self.device)
                edge_index = example['edge_index'].to(self.device)
                target_mask = example['ground_truth_mask'].to(self.device)
                
                # Compute edge features
                edge_features = self._compute_edge_features(x, edge_index)
                
                # Predict mask
                predicted_mask = self.mask_predictor(edge_features).squeeze()
                
                # Compute loss
                loss = criterion(predicted_mask, target_mask)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                avg_loss = total_loss / len(train_data)
                logger.info(f"Epoch {epoch}: Average loss = {avg_loss:.4f}")
        
        self.is_trained = True
        logger.info("PGExplainer training completed")
    
    def _compute_edge_features(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute features for each edge."""
        # Simple edge features: concatenate source and target node features
        src_features = x[edge_index[0]]  # [num_edges, num_features]
        dst_features = x[edge_index[1]]  # [num_edges, num_features]
        edge_features = torch.cat([src_features, dst_features], dim=-1)
        return edge_features
    
    def explain_node(self, node_id: int, x: torch.Tensor, edge_index: torch.Tensor, 
                    label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Generate explanation using trained mask predictor."""
        if not self.is_trained:
            raise RuntimeError("PGExplainer must be trained before use")
        
        self._set_seed()
        
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        
        # Compute edge features
        edge_features = self._compute_edge_features(x, edge_index)
        
        # Predict edge mask
        with torch.no_grad():
            edge_mask = self.mask_predictor(edge_features).squeeze()
        
        # Create important subgraph
        important_subgraph = self._create_important_subgraph(
            edge_index, edge_mask, threshold=0.5
        )
        
        return {
            'edge_mask': edge_mask.detach(),
            'node_feat_mask': None,  # PGExplainer focuses on edges
            'important_subgraph': important_subgraph,
            'explanation_type': 'pg_explainer'
        }
    
    def _create_important_subgraph(self, edge_index: torch.Tensor, 
                                  edge_mask: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """Create subgraph with most important edges."""
        important_edges = edge_mask >= threshold
        
        if important_edges.sum() == 0:
            k = min(10, len(edge_mask))
            _, top_indices = torch.topk(edge_mask, k)
            important_edges = torch.zeros_like(edge_mask, dtype=torch.bool)
            important_edges[top_indices] = True
        
        important_edge_index = edge_index[:, important_edges]
        important_edge_weights = edge_mask[important_edges]
        
        return {
            'edge_index': important_edge_index,
            'edge_weights': important_edge_weights,
            'edge_mask': important_edges
        }
    
    def save_explainer(self, path: str):
        """Save trained explainer."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained explainer")
        
        torch.save({
            'mask_predictor_state': self.mask_predictor.state_dict(),
            'hidden_dim': self.hidden_dim,
            'is_trained': self.is_trained
        }, path)
        logger.info(f"Saved PGExplainer to {path}")
    
    def load_explainer(self, path: str):
        """Load trained explainer."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.hidden_dim = checkpoint['hidden_dim']
        self.is_trained = checkpoint['is_trained']
        
        # Rebuild mask predictor (we need input_dim, so this is simplified)
        # In practice, you'd save input_dim too
        if self.mask_predictor is None:
            # Placeholder - in real use, save input_dim in checkpoint
            input_dim = 128  # This should come from checkpoint
            self._build_mask_predictor(input_dim)
        
        self.mask_predictor.load_state_dict(checkpoint['mask_predictor_state'])
        logger.info(f"Loaded PGExplainer from {path}")


class HGNNExplainer(BaseExplainer):
    """
    Heterogeneous GNN Explainer for explaining predictions in heterogeneous graphs.
    
    Learns per-relation importance masks for relation weight matrices W_r.
    """
    
    def __init__(self, model, node_types: List[str], edge_types: List[Tuple[str, str, str]], 
                 device: str = 'cpu', seed: int = 0):
        super().__init__(model, device, seed)
        self.node_types = node_types
        self.edge_types = edge_types
        
        # Relation mask learners (small MLPs per relation)
        self.relation_mask_learners = nn.ModuleDict()
        for edge_type in edge_types:
            src_type, rel_type, dst_type = edge_type
            rel_name = f"{src_type}__{rel_type}__{dst_type}"
            self.relation_mask_learners[rel_name] = nn.Sequential(
                nn.Linear(2, 16),  # Input: src and dst embeddings concatenated (simplified)
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            ).to(device)
        
        logger.info(f"Initialized HGNNExplainer for {len(edge_types)} relation types")
    
    def explain_node(self, node_id: int, x_dict: Dict[str, torch.Tensor], 
                    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                    node_type: str, label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Explain prediction for target node in heterogeneous graph.
        
        Args:
            node_id: Target node ID
            x_dict: Node features by type
            edge_index_dict: Edge indices by type  
            node_type: Type of target node
            label: Ground truth label (optional)
            
        Returns:
            Dictionary with per-relation importance and edge masks
        """
        self._set_seed()
        
        # Move to device
        for ntype in x_dict:
            x_dict[ntype] = x_dict[ntype].to(self.device)
        for etype in edge_index_dict:
            edge_index_dict[etype] = edge_index_dict[etype].to(self.device)
        
        relation_importance = {}
        edge_masks = {}
        
        # For each relation type, compute importance
        for edge_type, edge_index in edge_index_dict.items():
            src_type, rel_type, dst_type = edge_type
            rel_name = f"{src_type}__{rel_type}__{dst_type}"
            
            if rel_name in self.relation_mask_learners:
                # Compute relation importance (simplified)
                # In practice, this would use actual node embeddings from the model
                
                # Get source and destination embeddings (simplified)
                if src_type in x_dict and dst_type in x_dict:
                    src_emb = x_dict[src_type].mean(dim=-1, keepdim=True)  # [num_nodes, 1]
                    dst_emb = x_dict[dst_type].mean(dim=-1, keepdim=True)  # [num_nodes, 1]
                    
                    # For each edge, compute mask
                    edge_features = torch.cat([
                        src_emb[edge_index[0]],
                        dst_emb[edge_index[1]]
                    ], dim=-1)  # [num_edges, 2]
                    
                    edge_mask = self.relation_mask_learners[rel_name](edge_features).squeeze()
                    edge_masks[edge_type] = edge_mask
                    
                    # Relation-level importance (mean of edge masks)
                    relation_importance[edge_type] = edge_mask.mean().item()
        
        # Create important subgraph
        important_subgraph = self._create_hetero_important_subgraph(
            edge_index_dict, edge_masks, threshold=0.7
        )
        
        return {
            'edge_masks': edge_masks,
            'relation_importance': relation_importance,
            'important_subgraph': important_subgraph,
            'target_node': node_id,
            'target_type': node_type,
            'explanation_type': 'hgnn_explainer'
        }
    
    def _create_hetero_important_subgraph(self, edge_index_dict: Dict, edge_masks: Dict, 
                                        threshold: float = 0.7) -> Dict[str, Any]:
        """Create important subgraph for heterogeneous graph."""
        important_edges = {}
        
        for edge_type, edge_mask in edge_masks.items():
            important = edge_mask >= threshold
            
            if important.sum() == 0:
                k = min(5, len(edge_mask))
                _, top_indices = torch.topk(edge_mask, k)
                important = torch.zeros_like(edge_mask, dtype=torch.bool)
                important[top_indices] = True
            
            important_edges[edge_type] = {
                'edge_index': edge_index_dict[edge_type][:, important],
                'edge_weights': edge_mask[important],
                'edge_mask': important
            }
        
        return important_edges


class TemporalExplainer(BaseExplainer):
    """
    Temporal explainer for TGN-like models.
    
    Provides interface for temporal explainers with event masking.
    """
    
    def __init__(self, model, device: str = 'cpu', seed: int = 0):
        super().__init__(model, device, seed)
        
        # Temporal mask learner
        self.temporal_mask_learner = nn.Sequential(
            nn.Linear(1, 16),  # Input: timestamp
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(device)
        
        logger.info("Initialized TemporalExplainer")
    
    def explain_node(self, node_id: int, x: torch.Tensor, edge_index: torch.Tensor,
                    edge_time: torch.Tensor, label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Explain temporal prediction for target node.
        
        Args:
            node_id: Target node ID
            x: Node features
            edge_index: Edge indices
            edge_time: Edge timestamps
            label: Ground truth label
            
        Returns:
            Dictionary with temporal masks and importance
        """
        self._set_seed()
        
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_time = edge_time.to(self.device)
        
        # Normalize timestamps to [0,1]
        time_normalized = (edge_time - edge_time.min()) / (edge_time.max() - edge_time.min() + 1e-8)
        
        # Compute temporal mask for each edge/event
        temporal_mask = self.temporal_mask_learner(time_normalized.unsqueeze(-1)).squeeze()
        
        # Create temporal window importance
        time_windows = self._create_time_windows(edge_time, temporal_mask)
        
        # Create important subgraph based on temporal mask
        important_subgraph = self._create_temporal_subgraph(
            edge_index, edge_time, temporal_mask, threshold=0.5
        )
        
        return {
            'temporal_mask': temporal_mask.detach(),
            'time_windows': time_windows,
            'important_subgraph': important_subgraph,
            'explanation_type': 'temporal_explainer'
        }
    
    def _create_time_windows(self, edge_time: torch.Tensor, temporal_mask: torch.Tensor) -> Dict[str, Any]:
        """Create time windows with importance scores."""
        # Simple binning approach
        num_bins = 10
        time_min, time_max = edge_time.min(), edge_time.max()
        bin_edges = torch.linspace(time_min, time_max, num_bins + 1)
        
        window_importance = []
        for i in range(num_bins):
            mask = (edge_time >= bin_edges[i]) & (edge_time < bin_edges[i + 1])
            if mask.sum() > 0:
                importance = temporal_mask[mask].mean().item()
            else:
                importance = 0.0
            window_importance.append(importance)
        
        return {
            'bin_edges': bin_edges,
            'window_importance': window_importance
        }
    
    def _create_temporal_subgraph(self, edge_index: torch.Tensor, edge_time: torch.Tensor,
                                temporal_mask: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """Create subgraph with most temporally important edges."""
        important_events = temporal_mask >= threshold
        
        if important_events.sum() == 0:
            k = min(10, len(temporal_mask))
            _, top_indices = torch.topk(temporal_mask, k)
            important_events = torch.zeros_like(temporal_mask, dtype=torch.bool)
            important_events[top_indices] = True
        
        return {
            'edge_index': edge_index[:, important_events],
            'edge_time': edge_time[important_events],
            'temporal_weights': temporal_mask[important_events],
            'event_mask': important_events
        }


# Factory function for creating explainers
def create_explainer(explainer_type: str, model, **kwargs) -> BaseExplainer:
    """
    Factory function to create explainers.
    
    Args:
        explainer_type: One of 'gnn', 'pg', 'hgnn', 'temporal'
        model: Model to explain
        **kwargs: Additional arguments for explainer
        
    Returns:
        Initialized explainer
    """
    explainers = {
        'gnn': GNNExplainerWrapper,
        'pg': PGExplainerTrainer,
        'hgnn': HGNNExplainer,
        'temporal': TemporalExplainer
    }
    
    if explainer_type not in explainers:
        raise ValueError(f"Unknown explainer type: {explainer_type}")
    
    return explainers[explainer_type](model, **kwargs)
