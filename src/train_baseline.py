# src/train_baseline.py
import argparse, torch, json, os
from torch import optim
from torch_geometric.data import Data
from models.gcn_baseline import SimpleGCN
from models.graphsage_baseline import SimpleGraphSAGE
from models.rgcn_baseline import SimpleRGCN
# from models.hgt_baseline import SimpleHGT  # Temporarily disabled due to import issues
from models.han_baseline import SimpleHAN
from models.hypergraph import create_hypergraph_model
from data_utils import build_hypergraph_data, create_hypergraph_masks
from metrics import compute_metrics
from utils import set_seed
import yaml
import logging

logger = logging.getLogger(__name__)

def load_data(data_path, model_name, sample_n=None):
    """
    Loads the HeteroData object and prepares it for training.
    - For GCN/GraphSAGE, it creates a homogeneous graph of transaction nodes.
    - For RGCN, it returns the full HeteroData object.
    - For hypergraph models, it creates hypergraph data structure.
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

    elif model_name == 'hypergraph':
        # For hypergraph models, convert to hypergraph representation
        logger.info("Converting HeteroData to hypergraph representation...")
        
        # Apply sampling to HeteroData first if requested
        if sample_n:
            num_tx_nodes = tx_data.num_nodes
            if sample_n < num_tx_nodes:
                # Sample transaction nodes
                perm = torch.randperm(num_tx_nodes)[:sample_n]
                
                # Create new HeteroData with sampled transaction nodes
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
                
                data = sampled_data
        
        # Build hypergraph data with error handling
        try:
            hypergraph_data, node_features, labels = build_hypergraph_data(data)
        except Exception as e:
            logger.warning(f"Hypergraph construction failed: {e}")
            logger.info("Falling back to simple hypergraph construction...")
            
            # Fallback: Create simple hypergraph manually
            tx_data = data['transaction']
            known_mask = tx_data.y != 3
            labels = tx_data.y[known_mask].clone()
            labels[labels == 1] = 0  # licit
            labels[labels == 2] = 1  # illicit
            
            node_features = tx_data.x[known_mask]
            node_features[torch.isnan(node_features)] = 0
            
            n_nodes = labels.size(0)
            
            # Create simple random hyperedges for fallback
            import random
            random.seed(42)
            n_hyperedges = max(1, min(10, n_nodes // 5))
            
            from models.hypergraph import HypergraphData
            incidence_matrix = torch.zeros((n_nodes, n_hyperedges), dtype=torch.float)
            
            for he_idx in range(n_hyperedges):
                # Each hyperedge connects 2-4 random nodes
                size = random.randint(2, min(4, n_nodes))
                nodes = random.sample(range(n_nodes), size)
                for node_idx in nodes:
                    incidence_matrix[node_idx, he_idx] = 1.0
            
            hypergraph_data = HypergraphData(
                incidence_matrix=incidence_matrix,
                node_features=node_features,
                hyperedge_features=None,
                node_labels=labels
            )
        
        # Create train/val/test masks for hypergraph
        train_mask, val_mask, test_mask = create_hypergraph_masks(
            num_nodes=labels.size(0),
            seed=42
        )
        
        return {
            'hypergraph_data': hypergraph_data,
            'node_features': node_features,
            'labels': labels,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask
        }

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
    
    # Initialize model to None for safety
    model = None
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                if value is not None:  # Don't override with None values
                    setattr(args, key, value)

    data = load_data(args.data_path, model_name=args.model, sample_n=args.sample)
    
    print(f"DEBUG: args.model = '{args.model}'")
    print(f"DEBUG: Available model types: gcn, graphsage, hypergraph, rgcn, hgt, han")

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

    elif args.model == 'hypergraph':
        print("DEBUG: Entering hypergraph model branch")
        # Hypergraph neural network training
        data_dict = data  # data is already a dictionary with hypergraph components
        hypergraph_data = data_dict['hypergraph_data'].to(device)
        node_features = data_dict['node_features'].to(device)
        y = data_dict['labels'].to(device)
        train_mask = data_dict['train_mask'].to(device)
        val_mask = data_dict['val_mask'].to(device)
        test_mask = data_dict['test_mask'].to(device)
        
        # Ensure all dimensions are consistent
        n_nodes = min(hypergraph_data.B.shape[0], node_features.shape[0], y.shape[0])
        
        # Truncate all tensors to consistent size
        if hypergraph_data.B.shape[0] != n_nodes:
            print(f"Adjusting hypergraph dimensions from {hypergraph_data.B.shape[0]} to {n_nodes}")
            hypergraph_data.B = hypergraph_data.B[:n_nodes, :]
        
        node_features = node_features[:n_nodes, :]
        y = y[:n_nodes]
        train_mask = train_mask[:n_nodes]
        val_mask = val_mask[:n_nodes] 
        test_mask = test_mask[:n_nodes]
        
        # Update hypergraph_data with consistent dimensions
        hypergraph_data.X = node_features
        hypergraph_data.y = y
        
        # If we have empty hypergraphs, create simple meaningful hyperedges
        if hypergraph_data.B.shape[1] == 0:
            print("Creating fallback hyperedges...")
            n_nodes = n_nodes
            # Create K-NN hyperedges based on feature similarity
            with torch.no_grad():
                # Compute pairwise distances
                features_norm = torch.nn.functional.normalize(node_features, p=2, dim=1)
                similarity = torch.mm(features_norm, features_norm.t())
                
                # Create hyperedges from top-k similar nodes
                k = min(5, n_nodes - 1)
                _, top_k_indices = torch.topk(similarity, k + 1, dim=1)  # +1 because self is included
                
                # Create hyperedges from each node and its k nearest neighbors
                hyperedges = []
                for i in range(n_nodes):
                    neighbors = top_k_indices[i, 1:]  # exclude self
                    hyperedge = torch.zeros(n_nodes)
                    hyperedge[i] = 1.0
                    hyperedge[neighbors] = 1.0
                    hyperedges.append(hyperedge)
                
                # Stack into incidence matrix
                B_fallback = torch.stack(hyperedges, dim=1).to(device)
                hypergraph_data.B = B_fallback
                print(f"Created {B_fallback.shape[1]} fallback hyperedges")
                
                # Update statistics - will be computed after model creation
                print(f"Hypergraph construction completed: {y.size(0)} labeled nodes")
        
        # Create hypergraph model
        model_config = {
            'layer_type': getattr(args, 'layer_type', 'full'),
            'num_layers': int(getattr(args, 'num_layers', 3)),
            'dropout': float(getattr(args, 'dropout', 0.2)),
            'use_residual': bool(getattr(args, 'use_residual', True)),
            'use_batch_norm': bool(getattr(args, 'use_batch_norm', False)),
            'lambda0_init': float(getattr(args, 'lambda0_init', 1.0)),
            'lambda1_init': float(getattr(args, 'lambda1_init', 1.0)),
            'alpha_init': float(getattr(args, 'alpha_init', 0.1)),
            'max_iterations': int(getattr(args, 'max_iterations', 10)),
            'convergence_threshold': float(getattr(args, 'convergence_threshold', 1e-4))
        }
        
        model = create_hypergraph_model(
            input_dim=node_features.size(1),
            hidden_dim=args.hidden_dim,
            output_dim=2,  # Binary classification
            model_config=model_config
        ).to(device)
        
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        print(f"Starting training for {args.model}...")
        print(f"Model parameters: {model.count_parameters()}")
        print(f"Hypergraph stats: {model.get_hypergraph_stats(hypergraph_data)}")
        
        for epoch in range(args.epochs):
            model.train()
            logits = model(hypergraph_data, node_features)
            loss = torch.nn.CrossEntropyLoss()(logits[train_mask], y[train_mask])
            opt.zero_grad(); loss.backward(); opt.step()
            
            model.eval()
            with torch.no_grad():
                val_logits = model(hypergraph_data, node_features)
                val_probs = torch.softmax(val_logits[val_mask], dim=1)[:, 1].cpu().numpy()  # Get positive class probability
                val_true = y[val_mask].cpu().numpy()
                metrics = compute_metrics(val_true, val_probs)
                
                # Log layer parameters for monitoring
                if epoch % 5 == 0:
                    layer_params = model.get_layer_parameters()
                    logger.info(f"Epoch {epoch} layer parameters: {layer_params}")
                
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
    
    else:
        print(f"ERROR: Unknown model type '{args.model}'. Supported models: gcn, graphsage, hypergraph, rgcn, hgt, han")
        return {'error': f'Unknown model type: {args.model}', 'auc': 0.0, 'pr_auc': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

    # Ensure model exists before saving
    if model is None:
        print("ERROR: Model was not properly initialized!")
        return {'error': 'Model not initialized', 'auc': 0.0, 'pr_auc': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'ckpt.pth'))
    
    # Final evaluation
    if model is None:
        print("ERROR: Model was not properly initialized for final evaluation!")
        return {'error': 'Model not initialized', 'auc': 0.0, 'pr_auc': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        
    model.eval()
    with torch.no_grad():
        if args.model in ['gcn', 'graphsage']:
            test_logits = model(x, edge_index).squeeze()
            test_probs = torch.sigmoid(test_logits[test_mask]).cpu().numpy()
            test_true = y[test_mask].cpu().numpy()
        elif args.model == 'hypergraph':
            test_logits = model(hypergraph_data, node_features)
            test_probs = torch.softmax(test_logits[test_mask], dim=1)[:, 1].cpu().numpy()
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
        
        return final_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--data_path', type=str, default='data/ellipticpp/ellipticpp.pt')
    parser.add_argument('--out_dir', default='experiments/baseline/lite')
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'graphsage', 'rgcn', 'hgt', 'han', 'hypergraph'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample', type=int, default=None)  # lite mode
    parser.add_argument('--seed', type=int, default=42)  # reproducibility
    
    # Hypergraph-specific arguments
    parser.add_argument('--layer_type', type=str, default='full', choices=['simple', 'full'], 
                       help='PhenomNN layer type')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of hypergraph layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--use_residual', action='store_true', default=True,
                       help='Use residual connections')
    parser.add_argument('--use_batch_norm', action='store_true', default=False,
                       help='Use batch normalization')
    parser.add_argument('--lambda0_init', type=float, default=1.0,
                       help='Initial clique expansion weight')
    parser.add_argument('--lambda1_init', type=float, default=1.0,
                       help='Initial star expansion weight')
    parser.add_argument('--alpha_init', type=float, default=0.1,
                       help='Initial step size for PhenomNN')
    parser.add_argument('--max_iterations', type=int, default=10,
                       help='Max iterations for PhenomNN convergence')
    parser.add_argument('--convergence_threshold', type=float, default=1e-4,
                       help='Convergence threshold for PhenomNN')
    
    args = parser.parse_args()
    train(args)
