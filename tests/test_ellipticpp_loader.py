# tests/test_ellipticpp_loader.py
import os
import pandas as pd
from src.adapters.ellipticpp_adapter import load_ellipticpp_raw, map_to_canonical
from src.data_utils import build_hetero_data

def test_ellipticpp_files_exist():
    path = os.path.join('data','ellipticpp_sample')
    # Create dummy files for testing if they don't exist
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, 'nodes.csv'), 'w') as f:
        f.write('id,feature_0\n1,0.1\n2,0.2')
    with open(os.path.join(path, 'edges.csv'), 'w') as f:
        f.write('src,dst\n1,2')
    with open(os.path.join(path, 'labels.csv'), 'w') as f:
        f.write('id,label\n1,0')

    files = os.listdir(path)
    assert len(files) >= 3

def test_ellipticpp_load_sample():
    # load small sample from sample subfolder for CI or use sample flag
    raw = load_ellipticpp_raw(os.path.join('data','ellipticpp_sample'))
    nodes_df, edges_df, labels_df = map_to_canonical(raw)
    data = build_hetero_data(nodes_df.head(100), edges_df.head(200), node_id_col='orig_id')
    assert len(data.node_types) >= 1
    assert len(data.edge_types) >= 1
