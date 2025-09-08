# src/train_baseline.py
import argparse, torch, json, os
from torch import optim
from torch_geometric.data import Data
from models.gcn_baseline import SimpleGCN
from models.graphsage_baseline import SimpleGraphSAGE
from models.rgcn_baseline import SimpleRGCN
from metrics import compute_metrics
import yaml

def load_data(data_path, model_name, sample_n=None):
    """
    Loads the HeteroData object and prepares it for training.
    - For GCN/GraphSAGE, it creates a homogeneous graph of transaction nodes.
    - For RGCN, it returns the full HeteroData object.
    - It also filters for known labels and remaps them to binary.
    """
    data = torch.load(data_path, weights_only=False)
    tx_data = data['transaction']

    # --- Label Preprocessing & Mask Creation ---
    known_mask = tx_data.y != 3

    # Ensure masks exist on the transaction data store
    if not hasattr(tx_data, 'train_mask') or tx_data.train_mask is None:
        num_tx_nodes = tx_data.num_nodes
        perm = torch.randperm(num_tx_nodes)
        train_mask = torch.zeros(num_tx_nodes, dtype=torch.bool); train_mask[perm[:int(0.7*num_tx_nodes)]] = True
        val_mask = torch.zeros(num_tx_nodes, dtype=torch.bool); val_mask[perm[int(0.7*num_tx_nodes):int(0.85*num_tx_nodes)]] = True
        test_mask = torch.zeros(num_tx_nodes, dtype=torch.bool); test_mask[perm[int(0.85*num_tx_nodes):]] = True
        tx_data.train_mask = train_mask
        tx_data.val_mask = val_mask
        tx_data.test_mask = test_mask
    
    y = tx_data.y[known_mask].clone()
    y[y == 1] = 0  # licit
    y[y == 2] = 1  # illicit
    
    x = tx_data.x[known_mask]
    x[torch.isnan(x)] = 0 # Impute NaNs

    if model_name in ['gcn', 'graphsage']:
        homo_data = Data(x=x, edge_index=torch.empty((2, 0), dtype=torch.long), y=y)
        
        homo_data.train_mask = tx_data.train_mask[known_mask]
        homo_data.val_mask = tx_data.val_mask[known_mask]
        homo_data.test_mask = tx_data.test_mask[known_mask]

        if sample_n:
            num_nodes_to_sample = min(sample_n, homo_data.num_nodes)
            perm = torch.randperm(homo_data.num_nodes)[:num_nodes_to_sample]
            
            sampled_data = Data(x=homo_data.x[perm], edge_index=torch.empty((2, 0), dtype=torch.long), y=homo_data.y[perm])
            sampled_data.train_mask = homo_data.train_mask[perm]
            sampled_data.val_mask = homo_data.val_mask[perm]
            sampled_data.test_mask = homo_data.test_mask[perm]
            return sampled_data
        return homo_data

    elif model_name == 'rgcn':
        # For RGCN, we need the full graph but with filtered nodes
        # This is more complex; for now, we'll pass the full data and handle it in train
        return data

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                setattr(args, key, value)

    data = load_data(args.data_path, model_name=args.model, sample_n=args.sample)
    
    if args.model in ['gcn', 'graphsage']:
        data = data.to(device)
        x, edge_index, y = data.x, data.edge_index, data.y
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
        
        if args.model == 'gcn':
            model = SimpleGCN(in_dim=x.size(1), hidden_dim=args.hidden_dim, out_dim=1).to(device)
        else: # graphsage
            model = SimpleGraphSAGE(in_dim=x.size(1), hidden_dim=args.hidden_dim, out_dim=1).to(device)
        
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        print(f"Starting training for {args.model}...")
        for epoch in range(args.epochs):
            model.train()
            logits = model(x, edge_index).squeeze()
            loss = torch.nn.BCEWithLogitsLoss()(logits[train_mask], y[train_mask].float())
            opt.zero_grad(); loss.backward(); opt.step()
            
            model.eval()
            with torch.no_grad():
                val_logits = model(x, edge_index).squeeze()
                val_probs = torch.sigmoid(val_logits[val_mask]).cpu().numpy()
                val_true = y[val_mask].cpu().numpy()
                metrics = compute_metrics(val_true, val_probs)
                print(f"Epoch {epoch} loss {loss.item():.4f} val_auc {metrics['auc']:.4f}")

    elif args.model == 'rgcn':
        # RGCN requires HeteroData
        data = data.to(device)
        
        # --- Pre-filter data on transaction nodes ---
        tx_data = data['transaction']
        known_mask = tx_data.y != 3
        
        y = tx_data.y[known_mask].clone()
        y[y == 1] = 0  # licit
        y[y == 2] = 1  # illicit
        
        # These masks are now correctly sized for the filtered 'y'
        train_mask = tx_data.train_mask[known_mask]
        val_mask = tx_data.val_mask[known_mask]
        test_mask = tx_data.test_mask[known_mask]

        # --- Convert to homogeneous ---
        print("Warning: RGCN baseline is simplified. It trains on all node types but only evaluates on transactions.")
        homo_data = data.to_homogeneous()
        x = homo_data.x
        edge_index = homo_data.edge_index
        x[torch.isnan(x)] = 0

        # --- Handle edge_type ---
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
            edge_type = torch.empty(0, dtype=torch.long).to(device)
        else:
            # Recreate edge_type from scratch based on data.edge_stores
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=device)
            offset = 0
            for i, store in enumerate(data.edge_stores):
                if hasattr(store, 'edge_index'):
                    num_edges = store.edge_index.size(1)
                    edge_type[offset : offset + num_edges] = i
                    offset += num_edges
        
        # --- Identify transaction nodes in the homogeneous graph ---
        tx_node_type_index = data.node_types.index('transaction')
        tx_mask_homo = (homo_data.node_type == tx_node_type_index)

        model = SimpleRGCN(in_dim=x.size(1), hidden_dim=args.hidden_dim, out_dim=1, num_relations=len(data.edge_types)).to(device)
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        print(f"Starting training for {args.model}...")
        for epoch in range(args.epochs):
            model.train()
            logits = model(x, edge_index, edge_type).squeeze()
            
            # Get logits for transaction nodes that have known labels
            tx_logits_all = logits[tx_mask_homo]
            tx_logits_known = tx_logits_all[known_mask]

            loss = torch.nn.BCEWithLogitsLoss()(tx_logits_known[train_mask], y[train_mask].float())
            opt.zero_grad(); loss.backward(); opt.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(x, edge_index, edge_type).squeeze()[tx_mask_homo][known_mask]
                val_probs = torch.sigmoid(val_logits[val_mask]).cpu().numpy()
                val_true = y[val_mask].cpu().numpy()
                metrics = compute_metrics(val_true, val_probs)
                print(f"Epoch {epoch} loss {loss.item():.4f} val_auc {metrics['auc']:.4f}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'ckpt.pth'))
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        if args.model in ['gcn', 'graphsage']:
            test_logits = model(x, edge_index).squeeze()
            test_probs = torch.sigmoid(test_logits[test_mask]).cpu().numpy()
            test_true = y[test_mask].cpu().numpy()
        elif args.model == 'rgcn':
            test_logits = model(x, edge_index, edge_type).squeeze()[tx_mask_homo][known_mask]
            test_probs = torch.sigmoid(test_logits[test_mask]).cpu().numpy()
            test_true = y[test_mask].cpu().numpy()
            
        final_metrics = compute_metrics(test_true, test_probs)
        print("Final Test Metrics:", final_metrics)
        with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
            json.dump(final_metrics, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--data_path', type=str, default='data/ellipticpp/ellipticpp.pt')
    parser.add_argument('--out_dir', default='experiments/baseline/lite')
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'graphsage', 'rgcn'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample', type=int, default=None)  # lite mode
    args = parser.parse_args()
    train(args)
