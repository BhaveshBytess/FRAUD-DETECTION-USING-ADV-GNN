# src/train_tdgnn.py
"""
TDGNN + G-SAMPLER training script per §PHASE_C.2 exact modifications
Implements temporal GNN training with efficient sampling for scalable fraud detection
"""

import argparse
import torch
import json
import os
import yaml
import logging
from torch import optim
from typing import Dict, Any, Optional

# TDGNN imports
from models.tdgnn_wrapper import TDGNNHypergraphModel, train_epoch, evaluate_model
from models.hypergraph import create_hypergraph_model
from sampling.gsampler import GSampler
from sampling.temporal_data_loader import TemporalGraphDataLoader
from data_utils import build_hypergraph_data, create_hypergraph_masks
from metrics import compute_metrics
from utils import set_seed

logger = logging.getLogger(__name__)

def create_stage6_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create Stage 6 TDGNN configuration per §PHASE_C.3 exact specifications"""
    stage6_config = base_config.copy()
    
    # TDGNN-specific parameters per §PHASE_C.3
    stage6_config.update({
        # Temporal sampling parameters
        'fanouts': [15, 10],  # Neighbor sampling fanouts per hop
        'delta_t': 86400.0,   # Time relaxation (1 day in seconds)
        'sampling_strategy': 'recency',  # Temporal sampling strategy
        
        # G-SAMPLER parameters per §PHASE_B.3
        'num_workers': 4,
        'prefetch_factor': 2,
        'pin_memory': True,
        'use_gpu_sampling': True,  # Will fallback to CPU if CUDA unavailable
        
        # Training parameters per §PHASE_C.2
        'batch_size': 512,
        'accumulate_grad_batches': 1,
        'gradient_clip_val': 1.0,
        
        # Checkpointing per §PHASE_C.3
        'save_every': 5,
        'validate_every': 1,
        'early_stopping_patience': 10,
        
        # Reproducibility per §PHASE_C.3
        'deterministic': True,
        'benchmark': False,
        
        # Profiling per §PHASE_C.3
        'profile_memory': True,
        'profile_sampling': True,
    })
    
    return stage6_config

def load_temporal_data(data_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Load and prepare temporal graph data per §PHASE_C.1"""
    logger.info(f"Loading temporal data from {data_path}")
    
    # Load base HeteroData
    hetero_data = torch.load(data_path, weights_only=False)
    
    # Create temporal graph structure
    temporal_loader = TemporalGraphDataLoader()
    temporal_graph = temporal_loader.create_temporal_graph(hetero_data)
    
    logger.info(f"Temporal graph: {temporal_graph.num_nodes} nodes, {temporal_graph.num_edges} edges")
    
    # Build hypergraph representation for base model
    try:
        hypergraph_data, node_features, labels = build_hypergraph_data(hetero_data)
        logger.info("Successfully built hypergraph representation")
    except Exception as e:
        logger.warning(f"Hypergraph construction failed: {e}, using fallback")
        # Create fallback hypergraph
        tx_data = hetero_data['transaction']
        known_mask = tx_data.y != 3
        
        labels = tx_data.y[known_mask].clone()
        labels[labels == 1] = 0  # licit
        labels[labels == 2] = 1  # illicit
        
        node_features = tx_data.x[known_mask]
        node_features[torch.isnan(node_features)] = 0
        
        # Create simple incidence matrix
        n_nodes = labels.size(0)
        n_hyperedges = max(1, n_nodes // 10)
        
        from models.hypergraph import HypergraphData
        incidence_matrix = torch.zeros((n_nodes, n_hyperedges), dtype=torch.float)
        
        # Create random hyperedges
        import random
        random.seed(42)
        for he_idx in range(n_hyperedges):
            size = random.randint(3, min(8, n_nodes))
            nodes = random.sample(range(n_nodes), size)
            for node_idx in nodes:
                incidence_matrix[node_idx, he_idx] = 1.0
        
        hypergraph_data = HypergraphData(
            incidence_matrix=incidence_matrix,
            node_features=node_features,
            hyperedge_features=None,
            node_labels=labels
        )
    
    # Create train/val/test masks
    train_mask, val_mask, test_mask = create_hypergraph_masks(
        num_nodes=labels.size(0),
        seed=config.get('seed', 42)
    )
    
    # Create temporal data loaders per §PHASE_C.1
    train_seeds = torch.where(train_mask)[0]
    val_seeds = torch.where(val_mask)[0]
    test_seeds = torch.where(test_mask)[0]
    
    # Generate evaluation timestamps (simplified - in practice would use actual timestamps)
    train_t_evals = torch.ones(len(train_seeds), dtype=torch.float32) * temporal_graph.max_time
    val_t_evals = torch.ones(len(val_seeds), dtype=torch.float32) * temporal_graph.max_time
    test_t_evals = torch.ones(len(test_seeds), dtype=torch.float32) * temporal_graph.max_time
    
    # Create data loaders
    from torch.utils.data import DataLoader, TensorDataset
    
    batch_size = config.get('batch_size', 512)
    
    train_dataset = TensorDataset(train_seeds, train_t_evals, labels[train_seeds])
    val_dataset = TensorDataset(val_seeds, val_t_evals, labels[val_seeds])
    test_dataset = TensorDataset(test_seeds, test_t_evals, labels[test_seeds])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=config.get('num_workers', 4),
                            pin_memory=config.get('pin_memory', True))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=config.get('num_workers', 4),
                          pin_memory=config.get('pin_memory', True))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=config.get('num_workers', 4),
                           pin_memory=config.get('pin_memory', True))
    
    return {
        'temporal_graph': temporal_graph,
        'hypergraph_data': hypergraph_data,
        'node_features': node_features,
        'labels': labels,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask
    }

def create_tdgnn_model(
    temporal_graph,
    hypergraph_data,
    node_features,
    config: Dict[str, Any],
    device: torch.device
) -> TDGNNHypergraphModel:
    """Create TDGNN + G-SAMPLER model per §PHASE_C.1"""
    
    # Create base hypergraph model (from Stage 5)
    hypergraph_config = {
        'layer_type': config.get('layer_type', 'full'),
        'num_layers': config.get('num_layers', 3),
        'dropout': config.get('dropout', 0.2),
        'use_residual': config.get('use_residual', True),
        'use_batch_norm': config.get('use_batch_norm', False),
        'lambda0_init': config.get('lambda0_init', 1.0),
        'lambda1_init': config.get('lambda1_init', 1.0),
        'alpha_init': config.get('alpha_init', 0.1),
        'max_iterations': config.get('max_iterations', 10),
        'convergence_threshold': config.get('convergence_threshold', 1e-4)
    }
    
    base_model = create_hypergraph_model(
        input_dim=node_features.size(1),
        hidden_dim=config.get('hidden_dim', 128),
        output_dim=2,  # Binary classification
        model_config=hypergraph_config
    )
    
    # Create G-SAMPLER per §PHASE_B.3
    gsampler = GSampler(
        temporal_graph=temporal_graph,
        use_gpu=config.get('use_gpu_sampling', True),
        device=device
    )
    
    # Create TDGNN wrapper per §PHASE_C.1
    tdgnn_model = TDGNNHypergraphModel(
        base_model=base_model,
        gsampler=gsampler,
        temporal_graph=temporal_graph,
        hypergraph_config=hypergraph_config
    )
    
    return tdgnn_model

def train_tdgnn(args):
    """Main TDGNN training function per §PHASE_C.2 exact modifications"""
    
    # Set seed for reproducibility per §PHASE_C.3
    set_seed(args.seed)
    
    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                if value is not None:
                    setattr(args, key, value)
    else:
        config = {}
    
    # Create Stage 6 configuration per §PHASE_C.3
    config = create_stage6_config(vars(args))
    
    # Configure deterministic operations per §PHASE_C.3
    if config.get('deterministic', True):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load temporal data per §PHASE_C.1
    data_dict = load_temporal_data(args.data_path, config)
    temporal_graph = data_dict['temporal_graph']
    hypergraph_data = data_dict['hypergraph_data'].to(device)
    node_features = data_dict['node_features'].to(device)
    labels = data_dict['labels'].to(device)
    
    # Create TDGNN model per §PHASE_C.1
    model = create_tdgnn_model(
        temporal_graph, hypergraph_data, node_features, config, device
    ).to(device)
    
    # Setup optimizer and loss function per §PHASE_C.2
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    logger.info(f"TDGNN model created with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"G-SAMPLER statistics: {model.get_sampling_stats()}")
    
    # Training loop per §PHASE_C.2\n    best_val_auc = 0.0\n    patience_counter = 0\n    early_stopping_patience = config.get('early_stopping_patience', 10)\n    \n    # Create output directory\n    os.makedirs(args.out_dir, exist_ok=True)\n    \n    # Training metrics tracking\n    training_history = {\n        'train_loss': [],\n        'val_loss': [],\n        'val_auc': [],\n        'val_accuracy': []\n    }\n    \n    logger.info(\"Starting TDGNN training...\")\n    \n    for epoch in range(args.epochs):\n        # Training per §PHASE_C.2\n        train_metrics = train_epoch(\n            model=model,\n            gsampler=model.gsampler,\n            train_seed_loader=data_dict['train_loader'],\n            optimizer=optimizer,\n            criterion=criterion,\n            cfg=config\n        )\n        \n        # Validation per §PHASE_C.2\n        if epoch % config.get('validate_every', 1) == 0:\n            val_metrics = evaluate_model(\n                model=model,\n                eval_loader=data_dict['val_loader'],\n                criterion=criterion,\n                cfg=config,\n                split_name='val'\n            )\n            \n            # Update training history\n            training_history['train_loss'].append(train_metrics['train_loss'])\n            training_history['val_loss'].append(val_metrics['val_loss'])\n            training_history['val_accuracy'].append(val_metrics['val_accuracy'])\n            \n            # Compute validation AUC for early stopping\n            current_val_auc = val_metrics.get('val_auc', val_metrics['val_accuracy'])  # Fallback to accuracy\n            training_history['val_auc'].append(current_val_auc)\n            \n            # Learning rate scheduling\n            scheduler.step(current_val_auc)\n            \n            # Early stopping check per §PHASE_C.3\n            if current_val_auc > best_val_auc:\n                best_val_auc = current_val_auc\n                patience_counter = 0\n                \n                # Save best model per §PHASE_C.3\n                torch.save({\n                    'epoch': epoch,\n                    'model_state_dict': model.state_dict(),\n                    'optimizer_state_dict': optimizer.state_dict(),\n                    'best_val_auc': best_val_auc,\n                    'config': config\n                }, os.path.join(args.out_dir, 'best_model.pth'))\n                \n            else:\n                patience_counter += 1\n                \n            logger.info(\n                f\"Epoch {epoch:3d} | \"\n                f\"Train Loss: {train_metrics['train_loss']:.4f} | \"\n                f\"Val Loss: {val_metrics['val_loss']:.4f} | \"\n                f\"Val Acc: {val_metrics['val_accuracy']:.4f} | \"\n                f\"Best Val AUC: {best_val_auc:.4f} | \"\n                f\"Patience: {patience_counter}/{early_stopping_patience}\"\n            )\n            \n            # Memory profiling per §PHASE_C.3\n            if config.get('profile_memory', True) and torch.cuda.is_available():\n                allocated = torch.cuda.memory_allocated() / 1e9\n                reserved = torch.cuda.memory_reserved() / 1e9\n                logger.info(f\"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB\")\n            \n            # Early stopping per §PHASE_C.3\n            if patience_counter >= early_stopping_patience:\n                logger.info(f\"Early stopping triggered after {epoch + 1} epochs\")\n                break\n        \n        # Checkpointing per §PHASE_C.3\n        if epoch % config.get('save_every', 5) == 0:\n            torch.save({\n                'epoch': epoch,\n                'model_state_dict': model.state_dict(),\n                'optimizer_state_dict': optimizer.state_dict(),\n                'training_history': training_history,\n                'config': config\n            }, os.path.join(args.out_dir, f'checkpoint_epoch_{epoch}.pth'))\n    \n    # Load best model for final evaluation\n    best_checkpoint = torch.load(os.path.join(args.out_dir, 'best_model.pth'))\n    model.load_state_dict(best_checkpoint['model_state_dict'])\n    \n    # Final test evaluation per §PHASE_C.2\n    logger.info(\"Performing final test evaluation...\")\n    test_metrics = evaluate_model(\n        model=model,\n        eval_loader=data_dict['test_loader'],\n        criterion=criterion,\n        cfg=config,\n        split_name='test'\n    )\n    \n    # Compute detailed test metrics using existing metrics function\n    model.eval()\n    all_test_probs = []\n    all_test_labels = []\n    \n    with torch.no_grad():\n        for seed_nodes, t_evals, labels_batch in data_dict['test_loader']:\n            seed_nodes = seed_nodes.to(device)\n            t_evals = t_evals.to(device)\n            labels_batch = labels_batch.to(device)\n            \n            logits = model(\n                seed_nodes=seed_nodes,\n                t_eval_array=t_evals,\n                fanouts=config['fanouts'],\n                delta_t=config['delta_t']\n            )\n            \n            probs = torch.softmax(logits, dim=1)[:, 1]  # Positive class probability\n            all_test_probs.append(probs.cpu())\n            all_test_labels.append(labels_batch.cpu())\n    \n    # Compute final metrics\n    all_test_probs = torch.cat(all_test_probs, dim=0).numpy()\n    all_test_labels = torch.cat(all_test_labels, dim=0).numpy()\n    \n    final_metrics = compute_metrics(all_test_labels, all_test_probs)\n    final_metrics.update(test_metrics)  # Add test loss and accuracy\n    \n    logger.info(\"Final Test Metrics:\")\n    for metric_name, metric_value in final_metrics.items():\n        logger.info(f\"  {metric_name}: {metric_value:.4f}\")\n    \n    # Save final results per §PHASE_C.3\n    results = {\n        'final_metrics': final_metrics,\n        'training_history': training_history,\n        'config': config,\n        'best_epoch': best_checkpoint['epoch'],\n        'total_epochs': epoch + 1\n    }\n    \n    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:\n        json.dump(results, f, indent=2)\n    \n    # Save sampling statistics per §PHASE_C.3\n    sampling_stats = model.get_sampling_stats()\n    with open(os.path.join(args.out_dir, 'sampling_stats.json'), 'w') as f:\n        json.dump(sampling_stats, f, indent=2)\n    \n    logger.info(f\"Training completed! Results saved to {args.out_dir}\")\n    \n    return final_metrics\n\nif __name__ == '__main__':\n    parser = argparse.ArgumentParser(description='TDGNN + G-SAMPLER Training per §PHASE_C')\n    \n    # Basic arguments\n    parser.add_argument('--config', type=str, help='Path to YAML config file')\n    parser.add_argument('--data_path', type=str, default='data/ellipticpp/ellipticpp.pt', \n                       help='Path to temporal graph data')\n    parser.add_argument('--out_dir', default='experiments/stage6_tdgnn', \n                       help='Output directory for results')\n    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],\n                       help='Device to use for training')\n    parser.add_argument('--seed', type=int, default=42, help='Random seed')\n    \n    # Training arguments\n    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')\n    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')\n    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')\n    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')\n    \n    # TDGNN-specific arguments per §PHASE_C.3\n    parser.add_argument('--fanouts', nargs='+', type=int, default=[15, 10],\n                       help='Neighbor sampling fanouts per hop')\n    parser.add_argument('--delta_t', type=float, default=86400.0,\n                       help='Time relaxation parameter (seconds)')\n    parser.add_argument('--sampling_strategy', default='recency', \n                       choices=['recency', 'uniform'],\n                       help='Temporal sampling strategy')\n    parser.add_argument('--batch_size', type=int, default=512, \n                       help='Training batch size')\n    \n    # Hypergraph model arguments\n    parser.add_argument('--layer_type', type=str, default='full', choices=['simple', 'full'])\n    parser.add_argument('--num_layers', type=int, default=3)\n    parser.add_argument('--dropout', type=float, default=0.2)\n    parser.add_argument('--use_residual', action='store_true', default=True)\n    parser.add_argument('--use_batch_norm', action='store_true', default=False)\n    parser.add_argument('--lambda0_init', type=float, default=1.0)\n    parser.add_argument('--lambda1_init', type=float, default=1.0)\n    parser.add_argument('--alpha_init', type=float, default=0.1)\n    parser.add_argument('--max_iterations', type=int, default=10)\n    parser.add_argument('--convergence_threshold', type=float, default=1e-4)\n    \n    args = parser.parse_args()\n    \n    # Setup logging\n    logging.basicConfig(\n        level=logging.INFO,\n        format='%(asctime)s - %(levelname)s - %(message)s',\n        handlers=[\n            logging.FileHandler(os.path.join(args.out_dir, 'training.log') if args.out_dir else 'training.log'),\n            logging.StreamHandler()\n        ]\n    )\n    \n    train_tdgnn(args)
