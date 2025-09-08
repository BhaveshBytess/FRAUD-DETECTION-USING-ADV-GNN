# src/eval.py
import argparse
import torch
import json
from models.gcn_baseline import SimpleGCN
from models.graphsage_baseline import SimpleGraphSAGE
from models.rgcn_baseline import SimpleRGCN
from train_baseline import load_data
from metrics import compute_metrics

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = load_data(args.data_path, model_name=args.model, sample_n=args.sample)
    
    if args.model in ['gcn', 'graphsage']:
        data = data.to(device)
        x, edge_index, y = data.x, data.edge_index, data.y
        test_mask = data.test_mask
        
        if args.model == 'gcn':
            model = SimpleGCN(in_dim=x.size(1), hidden_dim=args.hidden_dim, out_dim=1).to(device)
        else: # graphsage
            model = SimpleGraphSAGE(in_dim=x.size(1), hidden_dim=args.hidden_dim, out_dim=1).to(device)
        
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.eval()

        with torch.no_grad():
            logits = model(x, edge_index).squeeze()
            probs = torch.sigmoid(logits[test_mask]).cpu().numpy()
            true_labels = y[test_mask].cpu().numpy()
            metrics = compute_metrics(true_labels, probs)
    
    elif args.model == 'rgcn':
        data = data.to(device)
        
        tx_data = data['transaction']

        # Ensure masks exist, recreating if necessary for evaluation context
        if not hasattr(tx_data, 'test_mask') or tx_data.test_mask is None:
            num_tx_nodes = tx_data.num_nodes
            perm = torch.randperm(num_tx_nodes)
            # Create dummy masks, only test_mask is actually used but good practice
            tx_data.train_mask = torch.zeros(num_tx_nodes, dtype=torch.bool); tx_data.train_mask[perm[:int(0.7*num_tx_nodes)]] = True
            tx_data.val_mask = torch.zeros(num_tx_nodes, dtype=torch.bool); tx_data.val_mask[perm[int(0.7*num_tx_nodes):int(0.85*num_tx_nodes)]] = True
            tx_data.test_mask = torch.zeros(num_tx_nodes, dtype=torch.bool); tx_data.test_mask[perm[int(0.85*num_tx_nodes):]] = True
        
        # --- Pre-filter data on transaction nodes ---
        known_mask = tx_data.y != 3
        
        y = tx_data.y[known_mask].clone()
        y[y == 1] = 0  # licit
        y[y == 2] = 1  # illicit
        
        test_mask = tx_data.test_mask[known_mask]

        # --- Convert to homogeneous ---
        homo_data = data.to_homogeneous()
        x = homo_data.x
        edge_index = homo_data.edge_index
        x[torch.isnan(x)] = 0

        # --- Handle edge_type ---
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
            edge_type = torch.empty(0, dtype=torch.long).to(device)
        else:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=device)
            offset = 0
            for i, store in enumerate(data.edge_stores):
                if hasattr(store, 'edge_index') and store.edge_index is not None:
                    num_edges = store.edge_index.size(1)
                    edge_type[offset : offset + num_edges] = i
                    offset += num_edges
        
        # --- Identify transaction nodes in the homogeneous graph ---
        tx_node_type_index = data.node_types.index('transaction')
        tx_mask_homo = (homo_data.node_type == tx_node_type_index)
        
        model = SimpleRGCN(in_dim=x.size(1), hidden_dim=args.hidden_dim, out_dim=1, num_relations=len(data.edge_types)).to(device)
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.eval()

        with torch.no_grad():
            logits = model(x, edge_index, edge_type).squeeze()
            
            # Filter logits for transaction nodes with known labels
            test_logits_all = logits[tx_mask_homo]
            test_logits_known = test_logits_all[known_mask]

            probs = torch.sigmoid(test_logits_known[test_mask]).cpu().numpy()
            true_labels = y[test_mask].cpu().numpy()
            metrics = compute_metrics(true_labels, probs)

    print("Evaluation Metrics:")
    print(json.dumps(metrics, indent=4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--data_path', default='data/ellipticpp/ellipticpp.pt', help='Path to the data file')
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'graphsage', 'rgcn'])
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension of the model (must match training)')
    parser.add_argument('--sample', type=int, default=None, help='Subsample N nodes for quick evaluation (lite mode)')
    args = parser.parse_args()
    evaluate(args)
