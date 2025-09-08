# tests/test_gcn_pipeline.py
import pytest
import torch
import os
import sys
import tempfile
from torch_geometric.data import Data

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.gcn_baseline import SimpleGCN
from metrics import compute_metrics

def create_sample_data():
    """Create a small sample dataset for testing."""
    num_nodes = 100
    num_features = 16
    
    # Create random features and labels
    x = torch.randn(num_nodes, num_features)
    y = torch.randint(0, 2, (num_nodes,))
    edge_index = torch.randint(0, num_nodes, (2, 200))
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[:60] = True
    val_mask[60:80] = True
    test_mask[80:] = True
    
    data = Data(x=x, edge_index=edge_index, y=y, 
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data

def test_gcn_forward_shapes():
    """Test that GCN model produces correct output shapes."""
    data = create_sample_data()
    model = SimpleGCN(in_dim=data.x.size(1), hidden_dim=8, out_dim=1)
    
    # Test forward pass
    logits = model(data.x, data.edge_index)
    
    # Check output shape - our model outputs (N, 1) which gets squeezed in training
    assert logits.shape[0] == data.x.shape[0], f"Expected {data.x.shape[0]} outputs, got {logits.shape[0]}"
    assert logits.shape[1] == 1, f"Expected output dim 1, got shape {logits.shape}"

def test_gcn_training_smoke():
    """Smoke test for GCN training (few iterations)."""
    data = create_sample_data()
    model = SimpleGCN(in_dim=data.x.size(1), hidden_dim=8, out_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Training for a few steps
    model.train()
    initial_loss = None
    
    for epoch in range(3):
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index).squeeze()  # Squeeze to get 1D output
        loss = criterion(logits[data.train_mask], data.y[data.train_mask].float())
        
        if epoch == 0:
            initial_loss = loss.item()
        
        loss.backward()
        optimizer.step()
    
    # Check that loss is finite
    assert torch.isfinite(loss), "Loss should be finite"
    assert loss.item() > 0, "Loss should be positive"

def test_compute_metrics():
    """Test the metrics computation function."""
    # Create dummy predictions and ground truth
    y_true = torch.tensor([0, 0, 1, 1, 1]).numpy()
    y_score = torch.tensor([0.1, 0.3, 0.7, 0.8, 0.9]).numpy()
    
    metrics = compute_metrics(y_true, y_score)
    
    # Check that all expected metrics are present
    expected_keys = ['auc', 'pr_auc', 'f1', 'recall', 'precision']
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"
        assert isinstance(metrics[key], (int, float)), f"Metric {key} should be numeric"
        assert not torch.isnan(torch.tensor(metrics[key])), f"Metric {key} should not be NaN"

def test_gcn_eval_mode():
    """Test GCN model in evaluation mode."""
    data = create_sample_data()
    model = SimpleGCN(in_dim=data.x.size(1), hidden_dim=8, out_dim=1)
    
    # Set to eval mode
    model.eval()
    
    with torch.no_grad():
        logits1 = model(data.x, data.edge_index)
        logits2 = model(data.x, data.edge_index)
    
    # In eval mode without dropout, outputs should be deterministic
    assert torch.allclose(logits1, logits2), "Eval mode should give deterministic outputs"

def test_model_save_load():
    """Test saving and loading model checkpoints."""
    data = create_sample_data()
    model = SimpleGCN(in_dim=data.x.size(1), hidden_dim=8, out_dim=1)
    
    # Get initial prediction
    model.eval()
    with torch.no_grad():
        initial_output = model(data.x, data.edge_index)
    
    # Save model
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        torch.save(model.state_dict(), f.name)
        temp_path = f.name
    
    try:
        # Create new model and load weights
        new_model = SimpleGCN(in_dim=data.x.size(1), hidden_dim=8, out_dim=1)
        new_model.load_state_dict(torch.load(temp_path))
        new_model.eval()
        
        # Compare outputs
        with torch.no_grad():
            loaded_output = new_model(data.x, data.edge_index)
        
        assert torch.allclose(initial_output, loaded_output), "Loaded model should give same outputs"
        
    finally:
        # Clean up
        os.unlink(temp_path)

if __name__ == "__main__":
    # Run tests when script is executed directly
    test_gcn_forward_shapes()
    test_gcn_training_smoke()
    test_compute_metrics() 
    test_gcn_eval_mode()
    test_model_save_load()
    print("All tests passed!")
