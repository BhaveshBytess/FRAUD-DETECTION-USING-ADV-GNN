# src/load_elliptic.py
import argparse
import torch
from src.data_utils import load_csv_nodes_edges, build_hetero_data

def main(nodes_csv, edges_csv, out_file=None):
    nodes, edges = load_csv_nodes_edges(nodes_csv, edges_csv)
    data = build_hetero_data(nodes, edges)
    print("HeteroData created:")
    for key in data.metadata():
        print(key)
    if out_file:
        torch.save(data, out_file)
        print("Saved to", out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    main(args.nodes, args.edges, args.out)
