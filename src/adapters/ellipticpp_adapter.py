# src/adapters/ellipticpp_adapter.py
import os
import pandas as pd

def inspect_files(path):
    files = os.listdir(path)
    return files

def load_ellipticpp_raw(path):
    """
    Read all files found in the Elliptic++ folder into pandas DataFrames.
    The actual filenames may vary; detect by keywords e.g., 'nodes', 'edges', 'features', 'labels'.
    """
    data = {}
    for fname in os.listdir(path):
        if fname.endswith('.csv'):
            df = pd.read_csv(os.path.join(path, fname))
            data[fname] = df
        elif fname.endswith('.json'):
            data[fname] = pd.read_json(os.path.join(path, fname))
        # add other filetypes if present
    return data

def map_to_canonical(data_dict):
    """
    Map raw data frames to canonical objects:
      - nodes_df: columns [orig_id, node_type(optional), feature_*...]
      - edges_df: columns [src_orig, dst_orig, edge_type, timestamp(optional), edge_feature_*...]
      - labels_df: mapping orig_id -> label (if separate)
    Implement rules specific to Elliptic++ file naming/columns here.
    """
    # heuristic mapping example (customize based on actual files)
    nodes_df = None
    edges_df = None
    labels_df = None
    for fname, df in data_dict.items():
        n = fname.lower()
        if 'node' in n or 'address' in n or 'wallet' in n:
            nodes_df = df
            # A reasonable assumption is that the first column is the ID
            nodes_df.rename(columns={nodes_df.columns[0]: 'orig_id'}, inplace=True)
        if 'edge' in n or 'graph' in n:
            edges_df = df
            # Assuming src and dst are the first two columns
            edges_df.rename(columns={edges_df.columns[0]: 'src', edges_df.columns[1]: 'dst'}, inplace=True)
        if 'label' in n or 'class' in n:
            labels_df = df

    # Add a default node_type if not present
    if nodes_df is not None and 'type' not in nodes_df.columns:
        nodes_df['type'] = 'transaction'
        
    if edges_df is not None and 'etype' not in edges_df.columns:
        edges_df['etype'] = 'transaction->transaction'


    # Additional mapping logic here...
    return nodes_df, edges_df, labels_df
