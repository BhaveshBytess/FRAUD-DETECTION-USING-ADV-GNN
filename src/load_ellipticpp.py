# src/load_ellipticpp.py
import argparse
import torch
import pandas as pd
from adapters.ellipticpp_adapter import load_ellipticpp_raw, map_to_canonical
from data_utils import build_hetero_data

def main(path, out=None, sample_n=None):
    raw = load_ellipticpp_raw(path)
    nodes_df, edges_df, labels_df = map_to_canonical(raw)

    # Optionally sample small subset for quick testing
    if sample_n:
        # We must sample the LABELED nodes to have a useful sample
        labeled_node_ids = labels_df['orig_id'].unique()
        
        # Filter nodes_df to only include nodes that have labels
        nodes_df = nodes_df[nodes_df['orig_id'].isin(labeled_node_ids)]

        # Further sample if needed
        if len(nodes_df) > sample_n:
            nodes_df = nodes_df.sample(n=sample_n, random_state=42)

        # filter labels to match the sampled nodes
        labels_df = labels_df[labels_df['orig_id'].isin(nodes_df['orig_id'])]

    
    # The column 'orig_id' from the adapter is the 'id' our builder expects
    data = build_hetero_data(nodes_df, edges_df, node_id_col='orig_id', labels_df=labels_df)
    print("HeteroData created with node types:", data.node_types)
    if out:
        torch.save(data, out)
        print("Saved to", out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to data/ellipticpp/")
    parser.add_argument("--out", default=None)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    main(args.path, args.out, args.sample)
