# src/train_baseline.py
import argparse, torch, json, os
from torch import optim
# This is a placeholder for the actual data loading function.
# We will need to adapt this to work with our HeteroData object.
from src.data_utils import build_hetero_data
from src.models.gcn_baseline import SimpleGCN
from src.models.rgcn_baseline import SimpleRGCN
from src.metrics import compute_metrics
import yaml

def load_data(data_path, sample_n=None):
    """
    A placeholder function to load data.
    This will need to be adapted to convert the HeteroData object
    into a homogeneous graph for the GCN baseline.
    """
    data = torch.load(data_path, weights_only=False)
    
    # For the simple GCN baseline, we'll use only the 'transaction' nodes
    # and their features. This is a simplification.
    if 'transaction' in data.node_types:
        # Create a homogeneous graph from the transaction nodes
        # and the edges that connect them.
        
        # Get the transaction nodes
        x = data['transaction'].x
        
        if hasattr(data['transaction'], 'y'):
            y = data['transaction'].y
        else:
            print("Warning: 'y' attribute not found for 'transaction' nodes. Creating dummy labels.")
            y = torch.randint(0, 2, (x.shape[0],))
        
        # Get the edges between transaction nodes
        # This assumes a 'transaction__to__transaction' edge type, which may not exist.
        # We will need to adapt this based on the actual edge types.
        # For now, we'll create a dummy edge_index.
        # A more robust solution would be to use a specific edge type or
        # create a homogeneous graph from the heterogeneous one.
        
        # Find an edge type that connects transactions, or just use all edges
        # and map them to a single node space.
        
        # Let's find the first edge type that connects to 'transaction'
        edge_type_key = None
        for key in data.edge_types:
            if 'transaction' in key:
                edge_type_key = key
                break
        
        if edge_type_key:
            edge_index = data[edge_type_key].edge_index
        else:
            # If no direct transaction edges, we can't proceed with this simple logic.
            # For now, let's create a dummy edge index for shape purposes.
            # This will need to be fixed.
            print("Warning: Could not find a suitable edge type for the GCN baseline. Using dummy edges.")
            edge_index = torch.randint(0, x.shape[0], (2, 100))


        # Create masks if they don't exist
        num_nodes = x.shape[0]
        if 'train_mask' not in data['transaction']:
            # Create a simple time-based split if 'time' is available
            if 'time' in data['transaction']:
                time = data['transaction'].time
                sorted_time_indices = torch.argsort(time)
                train_end = int(num_nodes * 0.7)
                val_end = int(num_nodes * 0.85)
                
                train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                train_mask[sorted_time_indices[:train_end]] = True
                
                val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                val_mask[sorted_time_indices[train_end:val_end]] = True

                test_mask = torch.zeros(num_nodes, dtype=torch.bool)
                test_mask[sorted_time_indices[val_end:]] = True
            else: # fallback to random split
                perm = torch.randperm(num_nodes)
                train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                train_mask[perm[:int(0.7*num_nodes)]] = True
                val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                val_mask[perm[int(0.7*num_nodes):int(0.85*num_nodes)]] = True
                test_mask = torch.zeros(num_nodes, dtype=torch.bool)
                test_mask[perm[int(0.85*num_nodes):]] = True
        else:
            train_mask = data['transaction'].train_mask
            val_mask = data['transaction'].val_mask
            test_mask = data['transaction'].test_mask

        # Create a simple Data object for the GCN
        from torch_geometric.data import Data
        homo_data = Data(x=x, edge_index=edge_index, y=y, 
                         train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        
        if sample_n:
            # Sub-sampling logic needs to be carefully implemented to preserve graph structure
            # For now, we will just use the first N nodes and their induced subgraph.
            # This is a simplification and may not be the best approach.
            from torch_geometric.utils import k_hop_subgraph
            subset = torch.arange(min(sample_n, homo_data.num_nodes))
            subset, edge_index, _, _ = k_hop_subgraph(subset, 2, homo_data.edge_index, relabel_nodes=True)
            
            homo_data.x = homo_data.x[subset]
            homo_data.y = homo_data.y[subset]
            homo_data.edge_index = edge_index
            homo_data.train_mask = homo_data.train_mask[subset]
            homo_data.val_mask = homo_data.val_mask[subset]
            homo_data.test_mask = homo_data.test_mask[subset]
            homo_data.num_nodes = subset.size(0)


        return homo_data
    else:
        raise ValueError("The provided data object does not have 'transaction' nodes.")


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            # Override args with config values
            for key, value in config.items():
                setattr(args, key, value)

    data = load_data(args.data_path, sample_n=args.sample)
    
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    y = data.y.to(device)
    train_mask, val_mask, test_mask = data.train_mask.to(device), data.val_mask.to(device), data.test_mask.to(device)
    
    model = SimpleGCN(in_dim=x.size(1), hidden_dim=args.hidden_dim, out_dim=1).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        logits = model(x, edge_index).squeeze()
        
        # Handle potential NaN or missing labels in y
        valid_mask = train_mask & ~torch.isnan(y)
        if valid_mask.sum() == 0:
            print(f"Epoch {epoch}: No valid training labels, skipping.")
            continue

        loss = torch.nn.BCEWithLogitsLoss()(logits[valid_mask], y[valid_mask].float())
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # eval
        model.eval()
        with torch.no_grad():
            val_logits = model(x, edge_index).squeeze()
            
            valid_val_mask = val_mask & ~torch.isnan(y)
            if valid_val_mask.sum() > 0:
                val_probs = torch.sigmoid(val_logits[valid_val_mask]).cpu().numpy()
                val_true = y[valid_val_mask].cpu().numpy()
                metrics = compute_metrics(val_true, val_probs)
                print(f"Epoch {epoch} loss {loss.item():.4f} val_auc {metrics['auc']:.4f}")
            else:
                print(f"Epoch {epoch} loss {loss.item():.4f} (no validation labels)")


    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'ckpt.pth'))
    
    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        test_logits = model(x, edge_index).squeeze()
        valid_test_mask = test_mask & ~torch.isnan(y)
        if valid_test_mask.sum() > 0:
            test_probs = torch.sigmoid(test_logits[valid_test_mask]).cpu().numpy()
            test_true = y[valid_test_mask].cpu().numpy()
            final_metrics = compute_metrics(test_true, test_probs)
            print("Final Test Metrics:", final_metrics)
            with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
                json.dump(final_metrics, f)
        else:
            print("No valid test labels for final evaluation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--data_path', type=str, default='data/ellipticpp.pt')
    parser.add_argument('--out_dir', default='experiments/baseline/lite')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample', type=int, default=None)  # lite mode
    args = parser.parse_args()
    train(args)
