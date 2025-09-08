# src/train_baseline.py
import argparse, torch, json, os
from torch import optim
from torch_geometric.data import Data
from models.gcn_baseline import SimpleGCN
from models.graphsage_baseline import SimpleGraphSAGE
from models.rgcn_baseline import SimpleRGCN
# from models.hgt_baseline import SimpleHGT  # Temporarily disabled due to import issues
from models.han_baseline import SimpleHAN
from metrics import compute_metrics
from utils import set_seed
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

    elif model_name in ['rgcn', 'hgt', 'han']:
        # For heterogeneous models, we need the full HeteroData structure
        # Apply sampling to HeteroData if requested
        if sample_n and model_name in ['hgt', 'han']:
            # Sample transaction nodes and keep the hetero structure
            num_tx_nodes = tx_data.num_nodes
            if sample_n < num_tx_nodes:
                # Sample transaction nodes
                perm = torch.randperm(num_tx_nodes)[:sample_n]
                
                # Create a new HeteroData with sampled transaction nodes
                sampled_data = data.clone()
                sampled_data['transaction'].x = tx_data.x[perm]
                sampled_data['transaction'].y = tx_data.y[perm]
                sampled_data['transaction'].train_mask = tx_data.train_mask[perm] if hasattr(tx_data, 'train_mask') else None
                sampled_data['transaction'].val_mask = tx_data.val_mask[perm] if hasattr(tx_data, 'val_mask') else None
                sampled_data['transaction'].test_mask = tx_data.test_mask[perm] if hasattr(tx_data, 'test_mask') else None
                
                # Filter edges to only include sampled transaction nodes
                for edge_type in data.edge_types:
                    if edge_type[0] == 'transaction' or edge_type[2] == 'transaction':
                        edge_index = data[edge_type].edge_index
                        if edge_type[0] == 'transaction' and edge_type[2] == 'transaction':
                            # Both source and target are transactions
                            mask = torch.isin(edge_index[0], perm) & torch.isin(edge_index[1], perm)
                        elif edge_type[0] == 'transaction':
                            # Source is transaction
                            mask = torch.isin(edge_index[0], perm)
                        else:
                            # Target is transaction
                            mask = torch.isin(edge_index[1], perm)
                        
                        sampled_data[edge_type].edge_index = edge_index[:, mask]
                
                return sampled_data
        
        return data
        # For RGCN, we need the full graph but with filtered nodes
        # This is more complex; for now, we'll pass the full data and handle it in train
        return data

def train(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                if value is not None:  # Don't override with None values
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

    elif args.model in ['hgt', 'han']:
        # HGT/HAN require HeteroData
        data = data.to(device)
        
        # --- Pre-filter data on transaction nodes ---
        tx_data = data['transaction']
        known_mask = tx_data.y != 3
        
        y = tx_data.y[known_mask].clone()
        y[y == 1] = 0  # licit
        y[y == 2] = 1  # illicit
        
        # These masks are now correctly sized for the filtered 'y'
        train_mask = tx_data.train_mask[known_mask] if hasattr(tx_data, 'train_mask') else None
        val_mask = tx_data.val_mask[known_mask] if hasattr(tx_data, 'val_mask') else None
        test_mask = tx_data.test_mask[known_mask] if hasattr(tx_data, 'test_mask') else None
        
        # Create masks if they don't exist
        if train_mask is None:
            num_known = known_mask.sum().item()
            perm = torch.randperm(num_known)
            train_mask = torch.zeros(num_known, dtype=torch.bool, device=device)
            val_mask = torch.zeros(num_known, dtype=torch.bool, device=device)
            test_mask = torch.zeros(num_known, dtype=torch.bool, device=device)
            train_mask[perm[:int(0.7*num_known)]] = True
            val_mask[perm[int(0.7*num_known):int(0.85*num_known)]] = True
            test_mask[perm[int(0.85*num_known):]] = True

        # Prepare HeteroData for HGT/HAN
        x_dict = {}
        edge_index_dict = {}
        
        print(f"Available node types: {data.node_types}")
        print(f"Available edge types: {data.edge_types}")
        
        for node_type in data.node_types:
            node_data = data[node_type]
            print(f"Node type '{node_type}' - has features: {hasattr(node_data, 'x')}")
            
            if hasattr(node_data, 'x') and node_data.x is not None:
                if node_type == 'transaction':
                    # Use filtered transaction data
                    x_dict[node_type] = node_data.x[known_mask]
                    # Handle NaN values
                    x_dict[node_type] = torch.nan_to_num(x_dict[node_type], nan=0.0)
                    print(f"Transaction features shape: {x_dict[node_type].shape}")
                else:
                    x_dict[node_type] = node_data.x
                    # Handle NaN values
                    x_dict[node_type] = torch.nan_to_num(x_dict[node_type], nan=0.0)
                    print(f"{node_type} features shape: {x_dict[node_type].shape}")
            else:
                print(f"Warning: {node_type} has no features, creating dummy features")
                # Create dummy features for nodes without features
                num_nodes = node_data.num_nodes if hasattr(node_data, 'num_nodes') else 100
                x_dict[node_type] = torch.zeros((num_nodes, 16), device=device)  # 16-dim dummy features
        
        for edge_type in data.edge_types:
            if hasattr(data[edge_type], 'edge_index') and data[edge_type].edge_index is not None:
                edge_index_dict[edge_type] = data[edge_type].edge_index
                print(f"Edge type '{edge_type}' shape: {edge_index_dict[edge_type].shape}")
            else:
                print(f"Warning: {edge_type} has no edges")
        
        # Initialize model
        metadata = (data.node_types, data.edge_types)
        if args.model == 'hgt':
            # TODO: Fix HGT model import issues
            raise NotImplementedError("HGT model temporarily disabled due to import issues")
            # model = SimpleHGT(metadata=metadata, hidden_dim=args.hidden_dim, out_dim=1).to(device)
        else:  # han
            model = SimpleHAN(metadata=metadata, hidden_dim=args.hidden_dim, out_dim=1).to(device)
        
        # Update input dimensions based on actual data
        model.update_input_dims(x_dict)
        model = model.to(device)
        
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        print(f"Starting training for {args.model}...")
        for epoch in range(args.epochs):
            model.train()
            logits = model(x_dict, edge_index_dict)
            loss = torch.nn.BCEWithLogitsLoss()(logits[train_mask], y[train_mask].float())
            opt.zero_grad(); loss.backward(); opt.step()
            
            model.eval()
            with torch.no_grad():
                val_logits = model(x_dict, edge_index_dict)
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
        elif args.model in ['hgt', 'han']:
            # Heterogeneous models - use full x_dict and edge_index_dict
            test_logits = model(x_dict, edge_index_dict).squeeze()
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
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'graphsage', 'rgcn', 'hgt', 'han'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample', type=int, default=None)  # lite mode
    parser.add_argument('--seed', type=int, default=42)  # reproducibility
    args = parser.parse_args()
    train(args)
