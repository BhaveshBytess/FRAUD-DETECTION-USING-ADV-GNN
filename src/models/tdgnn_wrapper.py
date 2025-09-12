# src/models/tdgnn_wrapper.py
"""
TDGNN wrapper per §PHASE_C - thin wrapper to accept sampled subgraph
Integrates temporal sampling with existing hypergraph models from Stage 5
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import logging

from sampling.gsampler import GSampler, SubgraphBatch
from sampling.temporal_data_loader import TemporalGraph
from models.hypergraph import HypergraphNN
from data_utils import build_hypergraph_data

logger = logging.getLogger(__name__)

class TDGNNHypergraphModel(nn.Module):
    """
    TDGNN wrapper that combines temporal sampling with hypergraph models per §PHASE_C.1
    """
    
    def __init__(
        self,
        base_model: HypergraphNN,
        gsampler: GSampler,
        temporal_graph: TemporalGraph,
        hypergraph_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize TDGNN wrapper
        
        Args:
            base_model: Base hypergraph model from Stage 5
            gsampler: G-SAMPLER instance for temporal sampling
            temporal_graph: Temporal graph structure
            hypergraph_config: Configuration for hypergraph construction
        """
        super().__init__()
        
        self.base_model = base_model
        self.gsampler = gsampler
        self.temporal_graph = temporal_graph
        self.hypergraph_config = hypergraph_config or {}
        
        # Store original device
        self.device = next(base_model.parameters()).device
        
        logger.info(f"TDGNN wrapper initialized with {type(base_model).__name__} on {self.device}")
    
    def forward(
        self, 
        seed_nodes: torch.Tensor,
        t_eval_array: torch.Tensor,
        fanouts: list,
        delta_t: float,
        return_subgraph: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with temporal sampling per §PHASE_C.1 data flow
        
        Args:
            seed_nodes: seed nodes for sampling (batch_size)
            t_eval_array: evaluation timestamps per seed (batch_size)
            fanouts: neighbor fanouts per hop
            delta_t: time relaxation parameter
            return_subgraph: whether to return subgraph info for debugging
            
        Returns:
            logits: model predictions for seed nodes
        """
        # Step 1: Temporal sampling per §PHASE_C.1
        subgraph_batch = self.gsampler.sample_time_relaxed(
            seed_nodes=seed_nodes,
            t_eval_array=t_eval_array,
            fanouts=fanouts,
            delta_t=delta_t,
            strategy='recency'
        )
        
        # Step 2: Convert subgraph to hypergraph representation per §PHASE_C.1
        if subgraph_batch.num_nodes == 0:
            # Handle empty subgraph
            batch_size = len(seed_nodes)
            num_classes = getattr(self.base_model, 'num_classes', 2)
            return torch.zeros(batch_size, num_classes, device=self.device)
        
        # Create dummy HeteroData for hypergraph construction
        # This is a simplified version - in production, would extract features properly
        hypergraph_data = self._create_hypergraph_from_subgraph(subgraph_batch)
        
        # Step 3: Forward pass through base hypergraph model per §PHASE_C.1
        full_logits = self.base_model(hypergraph_data, hypergraph_data.X)
        
        # Step 4: Extract predictions for seed nodes only
        seed_logits = self._extract_seed_predictions(full_logits, subgraph_batch, seed_nodes)
        
        if return_subgraph:
            return seed_logits, subgraph_batch
        else:
            return seed_logits
    
    def _create_hypergraph_from_subgraph(self, subgraph_batch: SubgraphBatch):
        """Create hypergraph data structure from sampled subgraph"""
        # For now, create a minimal hypergraph structure
        # In production, this would properly extract node features and construct hyperedges
        
        num_nodes = subgraph_batch.num_nodes
        feature_dim = getattr(self.base_model, 'input_dim', 64)
        
        # Create dummy node features (in production, extract from original graph)
        node_features = torch.randn(num_nodes, feature_dim, device=self.device)
        
        # Create simple incidence matrix from subgraph edges
        num_edges = subgraph_batch.num_edges
        if num_edges > 0:
            # Convert edge list to hyperedges (each edge becomes a 2-node hyperedge)
            incidence_matrix = torch.zeros(num_nodes, num_edges, device=self.device)
            
            # Build incidence matrix from CSR structure
            edge_idx = 0
            for u in range(num_nodes):
                if u < len(subgraph_batch.sub_indptr) - 1:
                    start_idx = subgraph_batch.sub_indptr[u]
                    end_idx = subgraph_batch.sub_indptr[u + 1]
                    
                    for neighbor_idx in range(start_idx, min(end_idx, len(subgraph_batch.sub_indices))):
                        if edge_idx < num_edges:
                            v = subgraph_batch.sub_indices[neighbor_idx]
                            if v < num_nodes:  # Safety check
                                incidence_matrix[u, edge_idx] = 1.0
                                incidence_matrix[v, edge_idx] = 1.0
                                edge_idx += 1
        else:
            # No edges - create minimal structure
            incidence_matrix = torch.zeros(num_nodes, max(1, num_nodes), device=self.device)
            # Add self-loops as hyperedges
            for i in range(num_nodes):
                if i < incidence_matrix.size(1):
                    incidence_matrix[i, i] = 1.0
        
        # Create hypergraph data object compatible with Stage 5 models
        from models.hypergraph.hypergraph_data import HypergraphData
        
        hypergraph_data = HypergraphData(
            incidence_matrix=incidence_matrix,
            node_features=node_features
        )
        
        return hypergraph_data
    
    def _extract_seed_predictions(
        self, 
        full_logits: torch.Tensor, 
        subgraph_batch: SubgraphBatch, 
        seed_nodes: torch.Tensor
    ) -> torch.Tensor:
        """Extract predictions for seed nodes from full subgraph predictions"""
        batch_size = len(seed_nodes)
        num_classes = full_logits.size(-1)
        
        # Initialize output tensor
        seed_logits = torch.zeros(batch_size, num_classes, device=self.device)
        
        # Map seed nodes to their positions in subgraph
        for i, seed_node in enumerate(seed_nodes):
            seed_node_id = seed_node.item()
            if seed_node_id in subgraph_batch.node_mapping:
                subgraph_idx = subgraph_batch.node_mapping[seed_node_id]
                if subgraph_idx < full_logits.size(0):
                    seed_logits[i] = full_logits[subgraph_idx]
                else:
                    logger.warning(f"Subgraph index {subgraph_idx} out of bounds for logits {full_logits.shape}")
            else:
                logger.warning(f"Seed node {seed_node_id} not found in subgraph mapping")
        
        return seed_logits
    
    def get_sampling_stats(self) -> Dict[str, Any]:
        """Get sampling statistics per §PHASE_C.3"""
        return {
            'gsampler_memory': self.gsampler.get_memory_stats(),
            'temporal_graph_nodes': self.temporal_graph.num_nodes,
            'temporal_graph_edges': self.temporal_graph.num_edges,
        }

def train_epoch(
    model: TDGNNHypergraphModel, 
    gsampler: GSampler, 
    train_seed_loader, 
    optimizer, 
    criterion, 
    cfg: Dict[str, Any]
) -> Dict[str, float]:
    """
    Training epoch with TDGNN + G-SAMPLER per §PHASE_C.2 exact modifications
    
    Args:
        model: TDGNN wrapper model
        gsampler: G-SAMPLER instance (already part of model)
        train_seed_loader: temporal data loader
        optimizer: PyTorch optimizer
        criterion: loss function
        cfg: configuration dictionary
        
    Returns:
        training metrics
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Get sampling parameters from config per §PHASE_C.2
    fanouts = cfg.get('fanouts', [15, 10])
    delta_t = cfg.get('delta_t', 86400.0)  # 1 day default
    
    for batch_idx, (seed_nodes, t_evals, labels) in enumerate(train_seed_loader):
        # Move to device
        seed_nodes = seed_nodes.to(model.device)
        t_evals = t_evals.to(model.device)
        labels = labels.to(model.device)
        
        # Debug prints per APPENDIX
        if batch_idx == 0:
            print(f"[TDGNN Training] batch_size={len(seed_nodes)} fanouts={fanouts} delta_t={delta_t}")
        
        # Forward pass with temporal sampling per §PHASE_C.2
        try:
            logits = model(
                seed_nodes=seed_nodes,
                t_eval_array=t_evals,
                fanouts=fanouts,
                delta_t=delta_t
            )
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Backward pass per §PHASE_C.2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Memory check per §PHASE_C.3
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"[MEM] batch={batch_idx} allocated={allocated:.3f}GB reserved={reserved:.3f}GB")
            
        except Exception as e:
            logger.error(f"Training error at batch {batch_idx}: {e}")
            continue
    
    avg_loss = total_loss / max(num_batches, 1)
    
    return {
        'train_loss': avg_loss,
        'num_batches': num_batches
    }

def evaluate_model(
    model: TDGNNHypergraphModel,
    eval_loader,
    criterion,
    cfg: Dict[str, Any],
    split_name: str = 'val'
) -> Dict[str, float]:
    """
    Evaluate model with temporal sampling per §PHASE_C.2
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    num_batches = 0
    
    # Get sampling parameters
    fanouts = cfg.get('fanouts', [15, 10])
    delta_t = cfg.get('delta_t', 86400.0)
    
    with torch.no_grad():
        for seed_nodes, t_evals, labels in eval_loader:
            # Move to device
            seed_nodes = seed_nodes.to(model.device)
            t_evals = t_evals.to(model.device)
            labels = labels.to(model.device)
            
            # Forward pass
            try:
                logits = model(
                    seed_nodes=seed_nodes,
                    t_eval_array=t_evals,
                    fanouts=fanouts,
                    delta_t=delta_t
                )
                
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                # Collect predictions
                predictions = torch.softmax(logits, dim=1)
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                
                num_batches += 1
                
            except Exception as e:
                logger.error(f"Evaluation error: {e}")
                continue
    
    # Compute metrics
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        
        # Combine all predictions and labels
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute basic metrics (can be extended)
        predicted_classes = torch.argmax(all_predictions, dim=1)
        accuracy = (predicted_classes == all_labels).float().mean().item()
    else:
        avg_loss = float('inf')
        accuracy = 0.0
    
    return {
        f'{split_name}_loss': avg_loss,
        f'{split_name}_accuracy': accuracy,
        f'{split_name}_num_samples': len(all_labels) if num_batches > 0 else 0
    }
