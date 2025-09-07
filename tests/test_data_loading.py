# tests/test_data_loading.py
from src.data_utils import load_csv_nodes_edges, build_hetero_data
import os

def test_sample_load():
    nodes_csv = os.path.join('data','sample','nodes.csv')
    edges_csv = os.path.join('data','sample','edges.csv')
    nodes, edges = load_csv_nodes_edges(nodes_csv, edges_csv)
    data = build_hetero_data(nodes, edges)
    # smoke asserts
    assert len(data.node_types) > 0
    # check at least one relation
    assert len(data.edge_types) > 0
