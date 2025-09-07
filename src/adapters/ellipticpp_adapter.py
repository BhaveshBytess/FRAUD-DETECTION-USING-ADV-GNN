# src/adapters/ellipticpp_adapter.py
import os
import pandas as pd

def inspect_files(path):
    files = os.listdir(path)
    return files

def load_ellipticpp_raw(path):
    """
    Read all .csv files found in the Elliptic++ folder into pandas DataFrames.
    """
    data = {}
    for fname in os.listdir(path):
        if fname.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(path, fname))
                data[fname] = df
            except Exception as e:
                print(f"Could not read {fname}: {e}")
    return data

def map_to_canonical(data_dict):
    """
    Map raw data frames from Elliptic++ to canonical objects:
      - nodes_df: Combined dataframe for all nodes (transactions).
      - edges_df: Combined dataframe for all edges.
      - labels_df: Combined dataframe for all labels.
    """
    # --- Process Nodes (Features) ---
    tx_features_df = data_dict.get('txs_features.csv')
    if tx_features_df is None: raise FileNotFoundError("txs_features.csv not found.")
    
    # The first column is the original ID. The rest are features.
    cols = tx_features_df.columns
    rename_dict = {cols[0]: 'orig_id'}
    rename_dict.update({cols[i]: f'feature_{i-1}' for i in range(1, len(cols))})
    
    tx_features_df.rename(columns=rename_dict, inplace=True)
    tx_features_df['type'] = 'transaction'
    nodes_df = tx_features_df

    # --- Process Labels ---
    tx_labels_df = data_dict.get('txs_classes.csv')
    if tx_labels_df is None: raise FileNotFoundError("txs_classes.csv not found.")
    tx_labels_df.rename(columns={tx_labels_df.columns[0]: 'orig_id', tx_labels_df.columns[1]: 'label'}, inplace=True)
    labels_df = tx_labels_df

    # --- Process Edges ---
    # For this simple GCN baseline, we are not using edges.
    edges_df = pd.DataFrame(columns=['src', 'dst', 'etype'])

    return nodes_df, edges_df, labels_df
