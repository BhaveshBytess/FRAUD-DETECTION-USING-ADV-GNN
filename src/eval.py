# src/eval.py
import argparse
import torch
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from models.gcn_baseline import SimpleGCN
from models.graphsage_baseline import SimpleGraphSAGE
from models.rgcn_baseline import SimpleRGCN
from models.hgt_baseline import SimpleHGT
from models.han_baseline import SimpleHAN
from models.temporal import create_temporal_model
from train_baseline import load_data
from temporal_utils import load_temporal_ellipticpp
from metrics import compute_metrics

def temporal_evaluate_model(
    model: torch.nn.Module,
    temporal_data: Dict,
    labels: torch.Tensor,
    device: torch.device,
    time_aware: bool = True
) -> Dict[str, Any]:
    """
    Evaluate temporal models with time-aware metrics.
    
    Args:
        model: Trained temporal model
        temporal_data: Temporal data dictionary from load_temporal_ellipticpp
        labels: Binary labels (0=non-fraud, 1=fraud)
        device: Device to run evaluation on
        time_aware: Whether to perform time-aware evaluation
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    # Get test split
    test_mask = temporal_data['temporal_splits']['test_mask']
    test_features = temporal_data['enhanced_features'][test_mask]
    test_labels = labels[test_mask]
    test_time_steps = temporal_data['time_steps'][test_mask]
    
    all_predictions = []
    all_true_labels = []
    time_step_metrics = {}
    
    if time_aware:
        # Evaluate per time step for temporal analysis
        unique_times = torch.unique(test_time_steps).sort()[0]
        
        with torch.no_grad():
            for time_step in unique_times:
                time_mask = (test_time_steps == time_step)
                if time_mask.sum() == 0:
                    continue
                    
                step_features = test_features[time_mask].to(device)
                step_labels = test_labels[time_mask]
                
                # Create sequence batch
                if len(step_features) > 0:
                    # Use all features as a single sequence
                    seq_input = step_features.unsqueeze(0)  # (1, seq_len, features)
                    seq_length = torch.tensor([len(step_features)])
                    
                    # Get model predictions
                    logits = model(seq_input, seq_length)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    
                    # Use single prediction for all transactions in time step
                    step_predictions = np.full(len(step_labels), probs[0])
                    step_true = step_labels.cpu().numpy()
                    
                    all_predictions.extend(step_predictions)
                    all_true_labels.extend(step_true)
                    
                    # Compute time step metrics
                    if len(np.unique(step_true)) > 1:  # Avoid single-class issues
                        step_metrics = compute_metrics(step_true, step_predictions)
                        time_step_metrics[f'time_{time_step.item()}'] = step_metrics
    else:
        # Standard evaluation without time awareness
        with torch.no_grad():
            # Simple batched evaluation
            batch_size = 32
            for i in range(0, len(test_features), batch_size):
                batch_features = test_features[i:i+batch_size].to(device)
                batch_labels = test_labels[i:i+batch_size]
                
                # Create sequence batches
                for j in range(len(batch_features)):
                    seq_input = batch_features[j:j+1].unsqueeze(0)  # (1, 1, features)
                    seq_length = torch.tensor([1])
                    
                    logits = model(seq_input, seq_length)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    
                    all_predictions.extend(probs)
                    all_true_labels.append(batch_labels[j].item())
    
    # Compute overall metrics
    overall_metrics = compute_metrics(
        np.array(all_true_labels), 
        np.array(all_predictions)
    )
    
    # Compute temporal-specific metrics
    temporal_metrics = {}
    if time_aware and time_step_metrics:
        # Average metrics across time steps
        time_aucs = [metrics['auc'] for metrics in time_step_metrics.values() 
                    if 'auc' in metrics and not np.isnan(metrics['auc'])]
        time_f1s = [metrics['f1'] for metrics in time_step_metrics.values() 
                   if 'f1' in metrics and not np.isnan(metrics['f1'])]
        
        if time_aucs:
            temporal_metrics['temporal_auc_mean'] = np.mean(time_aucs)
            temporal_metrics['temporal_auc_std'] = np.std(time_aucs)
            temporal_metrics['temporal_stability'] = 1.0 - (np.std(time_aucs) / np.mean(time_aucs))
        
        if time_f1s:
            temporal_metrics['temporal_f1_mean'] = np.mean(time_f1s)
            temporal_metrics['temporal_f1_std'] = np.std(time_f1s)
    
    return {
        'overall_metrics': overall_metrics,
        'temporal_metrics': temporal_metrics,
        'time_step_metrics': time_step_metrics,
        'evaluation_info': {
            'total_test_samples': len(all_true_labels),
            'unique_time_steps': len(time_step_metrics) if time_aware else 0,
            'time_aware_evaluation': time_aware
        }
    }


def evaluate_temporal_model(args):
    """
    Evaluate temporal models (LSTM, GRU, TGAN).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load temporal data
    temporal_data = load_temporal_ellipticpp(
        args.data_path.replace('ellipticpp.pt', ''),  # Use directory path
        window_size=args.window_size,
        add_temporal_feats=True
    )
    
    # Load labels
    labels_file = args.data_path.replace('ellipticpp.pt', 'txs_classes.csv')
    labels_data = pd.read_csv(labels_file)
    
    # Create labels tensor
    tx_ids = temporal_data['tx_ids']
    labels_dict = dict(zip(labels_data['txId'], labels_data['class']))
    labels = torch.tensor([labels_dict.get(tx_id.item(), 0) for tx_id in tx_ids], dtype=torch.long)
    
    # Convert to binary (1=illicit, 0=others)
    binary_labels = (labels == 1).long()
    
    # Load model configuration
    model_config = {
        'hidden_dim': args.hidden_dim,
        'num_layers': 2,
        'dropout': 0.3,
        'bidirectional': True,
        'use_attention': True
    }
    
    if args.model == 'tgan':
        model_config.update({
            'temporal_dim': 64,
            'graph_dim': 64,
            'num_heads': 4,
            'use_temporal': True,
            'use_graph': True
        })
    
    # Create and load model
    input_dim = temporal_data['enhanced_features'].shape[1]
    model = create_temporal_model(args.model, input_dim, model_config)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model
    results = temporal_evaluate_model(
        model, temporal_data, binary_labels, device, 
        time_aware=args.temporal_aware
    )
    
    return results


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Handle temporal models separately
    if args.model in ['lstm', 'gru', 'tgan']:
        results = evaluate_temporal_model(args)
        print("Temporal Model Evaluation Results:")
        print(json.dumps(results, indent=4, default=str))
        return
    
    # Original evaluation code for graph models
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

    elif args.model in ['hgt', 'han']:
        data = data.to(device)
        
        # For heterogeneous models, we need x_dict and edge_index_dict
        x_dict = {}
        for node_type in data.node_types:
            if hasattr(data[node_type], 'x'):
                x_dict[node_type] = data[node_type].x
                x_dict[node_type][torch.isnan(x_dict[node_type])] = 0
        
        edge_index_dict = {}
        for edge_type in data.edge_types:
            edge_store = data[edge_type]
            if hasattr(edge_store, 'edge_index') and edge_store.edge_index is not None:
                edge_index_dict[edge_type] = edge_store.edge_index
        
        # Get transaction data for evaluation
        tx_data = data['transaction']
        
        # Ensure test mask exists
        if not hasattr(tx_data, 'test_mask') or tx_data.test_mask is None:
            num_tx_nodes = tx_data.num_nodes
            perm = torch.randperm(num_tx_nodes)
            tx_data.test_mask = torch.zeros(num_tx_nodes, dtype=torch.bool)
            tx_data.test_mask[perm[int(0.85*num_tx_nodes):]] = True
        
        # Filter known labels (exclude class 3 - unknown)
        known_mask = tx_data.y != 3
        y = tx_data.y[known_mask].clone()
        y[y == 1] = 0  # licit
        y[y == 2] = 1  # illicit
        test_mask = tx_data.test_mask[known_mask]
        
        # Initialize model
        if args.model == 'hgt':
            model = SimpleHGT(
                node_types=data.node_types,
                edge_types=data.edge_types,
                in_dim=args.hidden_dim,
                hidden_dim=args.hidden_dim,
                out_dim=1
            ).to(device)
        else:  # han
            model = SimpleHAN(
                node_types=data.node_types,
                edge_types=data.edge_types,
                in_dim=args.hidden_dim,
                hidden_dim=args.hidden_dim,
                out_dim=1
            ).to(device)
        
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.eval()
        
        with torch.no_grad():
            logits = model(x_dict, edge_index_dict).squeeze()
            probs = torch.sigmoid(logits[test_mask]).cpu().numpy()
            true_labels = y[test_mask].cpu().numpy()
            metrics = compute_metrics(true_labels, probs)

    print("Evaluation Metrics:")
    print(json.dumps(metrics, indent=4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--data_path', default='data/ellipticpp/ellipticpp.pt', help='Path to the data file')
    parser.add_argument('--model', type=str, default='gcn', 
                       choices=['gcn', 'graphsage', 'rgcn', 'hgt', 'han', 'lstm', 'gru', 'tgan'])
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension of the model (must match training)')
    parser.add_argument('--sample', type=int, default=None, help='Subsample N nodes for quick evaluation (lite mode)')
    
    # Temporal model specific arguments
    parser.add_argument('--window_size', type=int, default=3, help='Temporal window size for temporal models')
    parser.add_argument('--temporal_aware', action='store_true', default=True, 
                       help='Use time-aware evaluation for temporal models')
    
    args = parser.parse_args()
    evaluate(args)
