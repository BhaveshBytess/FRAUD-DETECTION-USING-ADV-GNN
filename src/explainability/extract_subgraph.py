"""
Phase A — Subgraph extraction utilities for explainability.

Implements k-hop subgraph extraction with heterogeneous node/edge support
and deterministic sampling for reproducible explanations.

Following Stage 10 Reference §Phase A requirements.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import random
from torch_geometric.utils import k_hop_subgraph, to_undirected
from torch_geometric.data import HeteroData, Data
import logging

logger = logging.getLogger(__name__)


def extract_khop_subgraph(
    node_id: Union[int, torch.Tensor],
    num_hops: int,
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
    relabel_nodes: bool = True,
    max_nodes: int = 2000,
    seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract k-hop subgraph around target node with deterministic sampling.
    
    Args:
        node_id: Target node ID to extract subgraph around
        num_hops: Number of hops to include
        edge_index: Edge index tensor [2, num_edges]
        num_nodes: Total number of nodes in graph
        relabel_nodes: Whether to relabel nodes starting from 0
        max_nodes: Maximum nodes in subgraph (sample if exceeded)
        seed: Random seed for deterministic sampling
        
    Returns:
        subset: Node indices in subgraph
        sub_edge_index: Edge index for subgraph
        mapping: Mapping from original to subgraph node IDs
        edge_mask: Boolean mask for edges in original graph
    """
    # Set seed for deterministic behavior
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Convert node_id to tensor if needed
    if isinstance(node_id, int):
        node_id = torch.tensor([node_id])
    elif not isinstance(node_id, torch.Tensor):
        node_id = torch.tensor([node_id])
    
    # Extract k-hop subgraph using PyG utility
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_id,
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=relabel_nodes,
        num_nodes=num_nodes
    )
    
    # Apply deterministic sampling if subgraph too large
    if len(subset) > max_nodes:
        logger.warning(f"Subgraph has {len(subset)} nodes, sampling to {max_nodes}")
        
        # Ensure target node is always included
        target_idx = torch.where(subset == node_id[0])[0]
        if len(target_idx) == 0:
            raise ValueError(f"Target node {node_id[0]} not found in subgraph")
        
        # Sample remaining nodes deterministically
        remaining_budget = max_nodes - 1  # Reserve 1 for target
        other_nodes = subset[subset != node_id[0]]
        
        if len(other_nodes) > remaining_budget:
            # Sample by node degree (prefer high-degree nodes)
            degrees = torch.zeros(len(other_nodes))
            for i, node in enumerate(other_nodes):
                degrees[i] = (sub_edge_index == node).sum()
            
            # Sort by degree and take top nodes deterministically
            _, sorted_indices = torch.sort(degrees, descending=True)
            sampled_other = other_nodes[sorted_indices[:remaining_budget]]
            
            # Combine target + sampled nodes
            subset = torch.cat([node_id, sampled_other])
        
        # Re-extract subgraph with sampled nodes
        edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
        for i in range(edge_index.size(1)):
            if edge_index[0, i] in subset and edge_index[1, i] in subset:
                edge_mask[i] = True
        
        sub_edge_index = edge_index[:, edge_mask]
        
        if relabel_nodes:
            # Create proper mapping for relabeling
            max_node_id = max(subset.max().item(), edge_index.max().item()) if len(subset) > 0 else 0
            mapping = torch.full((max_node_id + 1,), -1, dtype=torch.long)
            mapping[subset] = torch.arange(len(subset))
            sub_edge_index = mapping[sub_edge_index]
    
    logger.info(f"Extracted {num_hops}-hop subgraph: {len(subset)} nodes, {sub_edge_index.size(1)} edges")
    
    return subset, sub_edge_index, mapping, edge_mask


def extract_hetero_subgraph(
    node_id: Union[int, str],
    node_type: str,
    num_hops: int,
    hetero_data: HeteroData,
    max_nodes: int = 2000,
    seed: int = 0
) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Extract k-hop subgraph from heterogeneous graph.
    
    Args:
        node_id: Target node ID
        node_type: Type of target node
        num_hops: Number of hops
        hetero_data: Heterogeneous graph data
        max_nodes: Maximum nodes per type
        seed: Random seed
        
    Returns:
        subset_dict: Node indices per type
        sub_edge_index_dict: Edge indices per edge type
        mapping_dict: Node mappings per type
    """
    # Set seed for deterministic behavior
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    subset_dict = {}
    sub_edge_index_dict = {}
    mapping_dict = {}
    
    # Start with target node
    if node_type not in hetero_data.node_types:
        raise ValueError(f"Node type {node_type} not found in graph")
    
    # Initialize with target node
    current_nodes = {node_type: torch.tensor([node_id])}
    
    # Expand k hops
    for hop in range(num_hops):
        next_nodes = {}
        
        # For each edge type, find neighbors
        for edge_type in hetero_data.edge_types:
            src_type, rel_type, dst_type = edge_type
            edge_index = hetero_data[edge_type].edge_index
            
            # Find outgoing neighbors
            if src_type in current_nodes:
                src_nodes = current_nodes[src_type]
                # Find edges starting from current nodes
                mask = torch.isin(edge_index[0], src_nodes)
                dst_nodes = edge_index[1, mask].unique()
                
                if dst_type not in next_nodes:
                    next_nodes[dst_type] = dst_nodes
                else:
                    next_nodes[dst_type] = torch.cat([next_nodes[dst_type], dst_nodes]).unique()
        
        # Add new nodes to current set
        for ntype, nodes in next_nodes.items():
            if ntype not in current_nodes:
                current_nodes[ntype] = nodes
            else:
                current_nodes[ntype] = torch.cat([current_nodes[ntype], nodes]).unique()
    
    # Apply sampling if needed
    for ntype, nodes in current_nodes.items():
        if len(nodes) > max_nodes:
            # Always keep target node if it's this type
            if ntype == node_type:
                target_mask = nodes == node_id
                other_nodes = nodes[~target_mask]
                sampled_others = other_nodes[torch.randperm(len(other_nodes))[:max_nodes-1]]
                current_nodes[ntype] = torch.cat([torch.tensor([node_id]), sampled_others])
            else:
                current_nodes[ntype] = nodes[torch.randperm(len(nodes))[:max_nodes]]
    
    # Extract edges for subgraph
    for edge_type in hetero_data.edge_types:
        src_type, rel_type, dst_type = edge_type
        edge_index = hetero_data[edge_type].edge_index
        
        if src_type in current_nodes and dst_type in current_nodes:
            src_nodes = current_nodes[src_type]
            dst_nodes = current_nodes[dst_type]
            
            # Find edges within subgraph
            src_mask = torch.isin(edge_index[0], src_nodes)
            dst_mask = torch.isin(edge_index[1], dst_nodes)
            edge_mask = src_mask & dst_mask
            
            if edge_mask.sum() > 0:
                sub_edge_index_dict[edge_type] = edge_index[:, edge_mask]
    
    # Create mappings
    for ntype, nodes in current_nodes.items():
        mapping = torch.zeros(nodes.max().item() + 1, dtype=torch.long)
        mapping[nodes] = torch.arange(len(nodes))
        mapping_dict[ntype] = mapping
        subset_dict[ntype] = nodes
    
    # Relabel edges
    for edge_type, edge_index in sub_edge_index_dict.items():
        src_type, rel_type, dst_type = edge_type
        if src_type in mapping_dict and dst_type in mapping_dict:
            sub_edge_index_dict[edge_type] = torch.stack([
                mapping_dict[src_type][edge_index[0]],
                mapping_dict[dst_type][edge_index[1]]
            ])
    
    logger.info(f"Extracted hetero subgraph: {sum(len(nodes) for nodes in subset_dict.values())} total nodes")
    
    return subset_dict, sub_edge_index_dict, mapping_dict


class SubgraphExtractor:
    """
    Unified subgraph extractor supporting both homogeneous and heterogeneous graphs.
    """
    
    def __init__(self, max_nodes: int = 2000, seed: int = 0):
        self.max_nodes = max_nodes
        self.seed = seed
    
    def extract(
        self,
        graph_data: Union[Data, HeteroData],
        node_id: Union[int, str],
        node_type: Optional[str] = None,
        num_hops: int = 3
    ) -> Dict[str, Any]:
        """
        Extract subgraph from either homogeneous or heterogeneous graph.
        
        Args:
            graph_data: PyG Data or HeteroData object
            node_id: Target node ID
            node_type: Node type (required for hetero graphs)
            num_hops: Number of hops
            
        Returns:
            Dictionary with subgraph data and mappings
        """
        if isinstance(graph_data, HeteroData):
            if node_type is None:
                raise ValueError("node_type required for heterogeneous graphs")
            
            subset_dict, sub_edge_index_dict, mapping_dict = extract_hetero_subgraph(
                node_id=node_id,
                node_type=node_type,
                num_hops=num_hops,
                hetero_data=graph_data,
                max_nodes=self.max_nodes,
                seed=self.seed
            )
            
            return {
                'type': 'hetero',
                'subset_dict': subset_dict,
                'edge_index_dict': sub_edge_index_dict,
                'mapping_dict': mapping_dict,
                'target_node': node_id,
                'target_type': node_type
            }
        
        elif isinstance(graph_data, Data):
            subset, sub_edge_index, mapping, edge_mask = extract_khop_subgraph(
                node_id=node_id,
                num_hops=num_hops,
                edge_index=graph_data.edge_index,
                num_nodes=graph_data.num_nodes,
                max_nodes=self.max_nodes,
                seed=self.seed
            )
            
            return {
                'type': 'homo',
                'subset': subset,
                'edge_index': sub_edge_index,
                'mapping': mapping,
                'edge_mask': edge_mask,
                'target_node': node_id
            }
        
        else:
            raise ValueError(f"Unsupported graph type: {type(graph_data)}")
    
    def get_subgraph_data(self, graph_data: Union[Data, HeteroData], extraction_result: Dict[str, Any]) -> Union[Data, HeteroData]:
        """
        Create new PyG data object with extracted subgraph.
        
        Args:
            graph_data: Original graph data
            extraction_result: Result from extract() method
            
        Returns:
            New PyG data object with subgraph
        """
        if extraction_result['type'] == 'hetero':
            # Create new HeteroData object
            sub_data = HeteroData()
            
            # Copy node features
            for ntype, subset in extraction_result['subset_dict'].items():
                if hasattr(graph_data[ntype], 'x') and graph_data[ntype].x is not None:
                    sub_data[ntype].x = graph_data[ntype].x[subset]
                sub_data[ntype].num_nodes = len(subset)
            
            # Copy edge indices and features
            for edge_type, edge_index in extraction_result['edge_index_dict'].items():
                sub_data[edge_type].edge_index = edge_index
                if hasattr(graph_data[edge_type], 'edge_attr') and graph_data[edge_type].edge_attr is not None:
                    # Map edge attributes (this is simplified)
                    sub_data[edge_type].edge_attr = graph_data[edge_type].edge_attr[:edge_index.size(1)]
            
            return sub_data
        
        else:
            # Create new Data object
            subset = extraction_result['subset']
            edge_index = extraction_result['edge_index']
            
            sub_data = Data()
            sub_data.edge_index = edge_index
            sub_data.num_nodes = len(subset)
            
            if hasattr(graph_data, 'x') and graph_data.x is not None:
                sub_data.x = graph_data.x[subset]
            
            if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
                edge_mask = extraction_result['edge_mask']
                sub_data.edge_attr = graph_data.edge_attr[edge_mask]
            
            return sub_data
