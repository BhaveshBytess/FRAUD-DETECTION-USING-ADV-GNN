# src/load_ellipticpp.py
import argparse
import torch
from src.adapters.ellipticpp_adapter import load_ellipticpp_raw, map_to_canonical
from src.data_utils import build_hetero_data

def main(path, out=None, sample_n=None):
    raw = load_ellipticpp_raw(path)
    nodes_df, edges_df, labels_df = map_to_canonical(raw)
    # Optionally sample small subset for quick testing
    if sample_n:
        nodes_df = nodes_df.head(sample_n)
        # filter edges involving these nodes
        edges_df = edges_df[edges_df['src'].isin(nodes_df['orig_id']) | edges_df['dst'].isin(nodes_df['orig_id'])]
    
    # The column 'orig_id' from the adapter is the 'id' our builder expects
    data = build_hetero_data(nodes_df, edges_df, node_id_col='orig_id')
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
