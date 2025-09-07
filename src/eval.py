# src/eval.py
import argparse
import torch
import json
from src.models.gcn_baseline import SimpleGCN
from src.train_baseline import load_data
from src.metrics import compute_metrics

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data = load_data(args.data_path, sample_n=args.sample)
    x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    test_mask = data.test_mask.to(device)

    # Load model
    # Note: We need to know the model's dimensions. We can infer this from the data,
    # but for simplicity, we'll hardcode or pass them as args.
    # A better approach would be to save model config with the checkpoint.
    in_dim = data.x.shape[1]
    # Assuming the output is a single logit for binary classification
    out_dim = 1 
    # This hidden_dim should match the one used during training.
    # We'll add it as an argument.
    model = SimpleGCN(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=out_dim).to(device)
    
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(x, edge_index).squeeze()
        
        valid_test_mask = test_mask & ~torch.isnan(y)
        if valid_test_mask.sum() > 0:
            probs = torch.sigmoid(logits[valid_test_mask]).cpu().numpy()
            true_labels = y[valid_test_mask].cpu().numpy()
            
            metrics = compute_metrics(true_labels, probs)
            
            print("Evaluation Metrics:")
            print(json.dumps(metrics, indent=4))
        else:
            print("No valid test labels found for evaluation.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--data_path', default='data/ellipticpp.pt', help='Path to the data file')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension of the GCN model')
    parser.add_argument('--sample', type=int, default=None, help='Subsample N nodes for quick evaluation (lite mode)')
    args = parser.parse_args()
    evaluate(args)
