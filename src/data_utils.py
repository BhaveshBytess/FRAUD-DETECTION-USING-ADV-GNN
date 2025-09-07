# src/data_utils.py
import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import HeteroData

def load_csv_nodes_edges(nodes_csv, edges_csv):
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)
    return nodes, edges

def build_hetero_data(nodes_df, edges_df,
                      node_type_col='type', node_id_col='id',
                      src_col='src', dst_col='dst', edge_type_col='etype',
                      time_col='time', label_col='label'):
    """
    Minimal converter: returns a PyG HeteroData object.
    Assumes nodes_df has: id, type, features... ; edges_df has: src, dst, etype, time, features...
    """
    data = HeteroData()
    
    # Create a mapping from original node ID to a new integer index for each node type
    node_mappings = {}
    for nt in nodes_df[node_type_col].unique():
        nt_nodes = nodes_df[nodes_df[node_type_col] == nt]
        node_mappings[nt] = {old_id: new_id for new_id, old_id in enumerate(nt_nodes[node_id_col])}

    # Build node collections by type
    for nt, mapping in node_mappings.items():
        nt_df = nodes_df[nodes_df[node_type_col] == nt]
        data[nt].num_nodes = len(nt_df)
        
        # Store features if present
        if 'feature_0' in nt_df.columns:
            feats = torch.tensor(nt_df.filter(regex='^feature_').values, dtype=torch.float)
            data[nt].x = feats

    # Build edges per relation
    for rel in edges_df[edge_type_col].unique():
        rel_df = edges_df[edges_df[edge_type_col] == rel]
        src_type, _, dst_type = rel.partition('->')
        
        # Map original source and destination IDs to the new integer indices
        src_indices = rel_df[src_col].map(node_mappings[src_type]).values
        dst_indices = rel_df[dst_col].map(node_mappings[dst_type]).values
        
        edge_index = torch.tensor(np.vstack([src_indices, dst_indices]), dtype=torch.long)
        data[(src_type, rel, dst_type)].edge_index = edge_index
        
        # Attach timestamps if present
        if time_col in rel_df.columns:
            data[(src_type, rel, dst_type)].time = torch.tensor(rel_df[time_col].values, dtype=torch.float)

    return data
