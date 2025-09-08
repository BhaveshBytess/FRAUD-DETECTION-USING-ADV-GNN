# tests/test_baseline_pipeline.py
import torch
import os
from src.train_baseline import load_data
from src.models.gcn_baseline import SimpleGCN

def test_forward_small():
    # This test requires the sample data file created in Stage 0
    sample_data_path = 'data/ellipticpp.pt'
    if not os.path.exists(sample_data_path):
        # If the main data file doesn't exist, maybe the sample one does.
        # This is a fallback for a slightly different setup.
        sample_data_path = 'data/ellipticpp_sample/sample_hetero.pt'
        if not os.path.exists(sample_data_path):
            # As a last resort, try to generate it from the sample CSVs
            # This assumes the Stage 0 setup is complete.
            # We'll just use the final expected file for now.
            sample_data_path = 'data/ellipticpp.pt'


    assert os.path.exists(sample_data_path), "Sample data file not found. Run Stage 0 first."

    # Load the data and convert to homogeneous for the GCN test
    data = load_data(sample_data_path, sample_n=100)
    
    x = data.x
    edge_index = data.edge_index
    
    model = SimpleGCN(in_dim=x.size(1), hidden_dim=8, out_dim=1)
    logits = model(x, edge_index)
    
    assert logits.shape[0] == x.shape[0]
    assert logits.shape[1] == 1
