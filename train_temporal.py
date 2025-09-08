"""
Temporal Model Training Pipeline for Stage 4

This module implements training pipeline for temporal fraud detection models.
Handles temporal data loading, sequential validation, and proper evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import os
import yaml
import argparse
from tqdm import tqdm
import logging
from datetime import datetime

# Import our modules
from src.temporal_utils import load_temporal_ellipticpp, TemporalDataProcessor
from src.models.temporal import create_temporal_model, TemporalDataLoader
from src.metrics import compute_metrics
from src.eval import evaluate_model
from src.utils import set_seed, create_output_dir, save_model_checkpoint


class TemporalDataset(Dataset):
    """
    PyTorch Dataset for temporal fraud detection.
    """
    
    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        time_steps: torch.Tensor,
        window_size: int = 3,
        max_seq_len: int = 100
    ):
        self.features = features
        self.labels = labels
        self.time_steps = time_steps
        self.window_size = window_size
        self.max_seq_len = max_seq_len
        
        # Create temporal sequences
        self.sequences = self._create_sequences()
    
    def _create_sequences(self) -> List[Dict]:
        """Create temporal sequences from the data."""
        sequences = []
        
        # Get unique time steps
        unique_times = torch.unique(self.time_steps).sort()[0]
        
        # Create sliding windows
        for i in range(len(unique_times) - self.window_size + 1):
            window_times = unique_times[i:i + self.window_size]
            
            # Get data for this window
            window_mask = torch.isin(self.time_steps, window_times)
            window_features = self.features[window_mask]
            window_labels = self.labels[window_mask]
            window_time_steps = self.time_steps[window_mask]
            
            if len(window_features) > 0:
                # Truncate if too long
                if len(window_features) > self.max_seq_len:
                    indices = torch.randperm(len(window_features))[:self.max_seq_len]
                    window_features = window_features[indices]
                    window_labels = window_labels[indices]
                    window_time_steps = window_time_steps[indices]
                
                sequences.append({
                    'features': window_features,
                    'labels': window_labels,
                    'time_steps': window_time_steps,
                    'length': len(window_features),
                    'window_times': window_times
                })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.sequences[idx]


def collate_temporal_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for temporal batches with padding.
    """
    # Find max sequence length in batch
    max_len = max(item['length'] for item in batch)
    
    batch_features = []
    batch_labels = []
    batch_lengths = []
    
    for item in batch:
        features = item['features']
        labels = item['labels']
        length = item['length']
        
        # Pad sequences
        if length < max_len:
            feat_padding = torch.zeros(max_len - length, features.shape[1])
            features = torch.cat([features, feat_padding], dim=0)
            
            label_padding = torch.zeros(max_len - length, dtype=labels.dtype)
            labels = torch.cat([labels, label_padding], dim=0)
        
        batch_features.append(features)
        batch_labels.append(labels)
        batch_lengths.append(length)
    
    return {
        'features': torch.stack(batch_features),
        'labels': torch.stack(batch_labels),
        'lengths': torch.tensor(batch_lengths),
        'max_length': max_len
    }


class TemporalTrainer:
    """
    Trainer class for temporal fraud detection models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Set up optimizer and loss
        self.optimizer = self._create_optimizer()
        self.criterion = self._create_criterion()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_auc': [], 'val_auc': [],
            'train_f1': [], 'val_f1': []
        }
        
        # Set up logging
        self.logger = self._setup_logger()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')
        lr = optimizer_config.get('lr', 0.001)
        weight_decay = optimizer_config.get('weight_decay', 1e-5)
        
        if optimizer_type.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            momentum = optimizer_config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_criterion(self) -> nn.Module:
        """Create loss criterion."""
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'cross_entropy')
        
        if loss_type == 'cross_entropy':
            # Handle class imbalance if specified
            class_weights = loss_config.get('class_weights', None)
            if class_weights:
                weight = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
                return nn.CrossEntropyLoss(weight=weight)
            else:
                return nn.CrossEntropyLoss()
        elif loss_type == 'focal':
            # Implement focal loss for imbalanced data
            return FocalLoss(alpha=loss_config.get('alpha', 1.0), gamma=loss_config.get('gamma', 2.0))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_config = self.config.get('scheduler', {})
        if not scheduler_config.get('enabled', False):
            return None
        
        scheduler_type = scheduler_config.get('type', 'step')
        
        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.5)
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('training', {}).get('epochs', 100)
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=scheduler_config.get('patience', 5),
                factor=scheduler_config.get('factor', 0.5)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger."""
        logger = logging.getLogger('temporal_trainer')
        logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        return logger
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get predictions for each sequence in batch
            batch_size, max_len, _ = features.shape
            batch_logits = []
            batch_targets = []
            
            for i in range(batch_size):
                seq_len = lengths[i].item()
                seq_features = features[i, :seq_len].unsqueeze(0)  # (1, seq_len, features)
                seq_labels = labels[i, :seq_len]  # (seq_len,)
                
                # Get model predictions
                logits = self.model(seq_features, torch.tensor([seq_len]))  # (1, num_classes)
                
                # Use majority vote for sequence label
                seq_target = torch.mode(seq_labels)[0].unsqueeze(0)
                
                batch_logits.append(logits)
                batch_targets.append(seq_target)
            
            # Concatenate predictions
            logits = torch.cat(batch_logits, dim=0)  # (batch_size, num_classes)
            targets = torch.cat(batch_targets, dim=0)  # (batch_size,)
            
            # Compute loss
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('training', {}).get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            predictions = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            true_labels = targets.detach().cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(true_labels)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = compute_metrics(np.array(all_labels), np.array(all_predictions))
        
        return avg_loss, metrics
    
    def validate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc='Validating'):
                # Move to device
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                
                # Process sequences
                batch_size, max_len, _ = features.shape
                batch_logits = []
                batch_targets = []
                
                for i in range(batch_size):
                    seq_len = lengths[i].item()
                    seq_features = features[i, :seq_len].unsqueeze(0)
                    seq_labels = labels[i, :seq_len]
                    
                    # Get predictions
                    logits = self.model(seq_features, torch.tensor([seq_len]))
                    seq_target = torch.mode(seq_labels)[0].unsqueeze(0)
                    
                    batch_logits.append(logits)
                    batch_targets.append(seq_target)
                
                # Concatenate and compute loss
                logits = torch.cat(batch_logits, dim=0)
                targets = torch.cat(batch_targets, dim=0)
                loss = self.criterion(logits, targets)
                
                total_loss += loss.item()
                predictions = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                true_labels = targets.cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(true_labels)
        
        avg_loss = total_loss / len(loader)
        metrics = compute_metrics(np.array(all_labels), np.array(all_predictions))
        
        return avg_loss, metrics
    
    def train(self) -> Dict[str, Any]:
        """Full training loop."""
        training_config = self.config.get('training', {})
        epochs = training_config.get('epochs', 100)
        early_stopping_patience = training_config.get('early_stopping_patience', 10)
        
        self.logger.info(f"Starting temporal model training for {epochs} epochs")
        
        best_val_score = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate(self.val_loader)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['auc'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Train AUC: {train_metrics['auc']:.4f}, Val AUC: {val_metrics['auc']:.4f}, "
                f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}"
            )
            
            # Save training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_auc'].append(train_metrics['auc'])
            self.training_history['val_auc'].append(val_metrics['auc'])
            self.training_history['train_f1'].append(train_metrics['f1'])
            self.training_history['val_f1'].append(val_metrics['f1'])
            
            # Check for best model
            current_score = val_metrics['auc']
            if current_score > best_val_score:
                best_val_score = current_score
                patience_counter = 0
                
                # Save best model
                self.save_checkpoint(epoch, best=True)
                self.logger.info(f"New best model saved with AUC: {best_val_score:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        # Final evaluation on test set
        self.logger.info("Evaluating on test set...")
        test_loss, test_metrics = self.validate(self.test_loader)
        
        self.logger.info(
            f"Final Test Results - Loss: {test_loss:.4f}, "
            f"AUC: {test_metrics['auc']:.4f}, F1: {test_metrics['f1']:.4f}, "
            f"Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}"
        )
        
        return {
            'training_history': self.training_history,
            'best_val_auc': best_val_score,
            'test_metrics': test_metrics,
            'total_epochs': self.current_epoch + 1
        }
    
    def save_checkpoint(self, epoch: int, best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filename = 'best_model.pt' if best else f'checkpoint_epoch_{epoch}.pt'
        output_dir = self.config.get('output_dir', 'experiments/temporal')
        os.makedirs(output_dir, exist_ok=True)
        
        torch.save(checkpoint, os.path.join(output_dir, filename))


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def create_temporal_dataloaders(
    temporal_data: Dict,
    labels_data: pd.DataFrame,
    config: Dict,
    device: torch.device
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create temporal data loaders with proper train/val/test splits.
    """
    # Load enhanced features and temporal info
    features = temporal_data['enhanced_features']
    time_steps = temporal_data['time_steps']
    
    # Get labels (assuming txs_classes.csv has 'txId' and 'class' columns)
    tx_ids = temporal_data['tx_ids']
    
    # Match labels with transaction IDs
    labels_dict = dict(zip(labels_data['txId'], labels_data['class']))
    labels = torch.tensor([labels_dict.get(tx_id.item(), 0) for tx_id in tx_ids], dtype=torch.long)
    
    # Get temporal splits
    train_mask = temporal_data['temporal_splits']['train_mask']
    val_mask = temporal_data['temporal_splits']['val_mask']
    test_mask = temporal_data['temporal_splits']['test_mask']
    
    # Create datasets
    dataset_config = config.get('dataset', {})
    window_size = dataset_config.get('window_size', 3)
    max_seq_len = dataset_config.get('max_seq_len', 100)
    
    train_dataset = TemporalDataset(
        features[train_mask], labels[train_mask], time_steps[train_mask],
        window_size=window_size, max_seq_len=max_seq_len
    )
    
    val_dataset = TemporalDataset(
        features[val_mask], labels[val_mask], time_steps[val_mask],
        window_size=window_size, max_seq_len=max_seq_len
    )
    
    test_dataset = TemporalDataset(
        features[test_mask], labels[test_mask], time_steps[test_mask],
        window_size=window_size, max_seq_len=max_seq_len
    )
    
    # Create data loaders
    batch_size = config.get('training', {}).get('batch_size', 32)
    num_workers = config.get('training', {}).get('num_workers', 0)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_temporal_batch,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_temporal_batch,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_temporal_batch,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train temporal fraud detection models')
    parser.add_argument('--config', type=str, default='configs/temporal.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model_type', type=str, default='lstm',
                       choices=['lstm', 'gru', 'tgan'],
                       help='Type of temporal model to train')
    parser.add_argument('--data_dir', type=str, default='data/ellipticpp',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='experiments/temporal',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'model': {
                'type': args.model_type,
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.3,
                'bidirectional': True,
                'use_attention': True
            },
            'dataset': {
                'window_size': 3,
                'max_seq_len': 100,
                'add_temporal_feats': True
            },
            'training': {
                'epochs': 50,
                'batch_size': 16,  # Smaller batch for memory efficiency
                'lr': 0.001,
                'weight_decay': 1e-5,
                'grad_clip': 1.0,
                'early_stopping_patience': 10
            },
            'optimizer': {
                'type': 'adam',
                'lr': 0.001,
                'weight_decay': 1e-5
            },
            'scheduler': {
                'enabled': True,
                'type': 'plateau',
                'patience': 5,
                'factor': 0.5
            },
            'loss': {
                'type': 'cross_entropy',
                'class_weights': [1.0, 3.0]  # Give more weight to fraud class
            }
        }
    
    config['output_dir'] = args.output_dir
    
    # Create output directory
    create_output_dir(args.output_dir)
    
    # Load temporal data
    print("Loading temporal data...")
    temporal_data = load_temporal_ellipticpp(
        args.data_dir,
        window_size=config['dataset']['window_size'],
        add_temporal_feats=config['dataset']['add_temporal_feats']
    )
    
    # Load labels
    labels_file = os.path.join(args.data_dir, 'txs_classes.csv')
    labels_data = pd.read_csv(labels_file)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_temporal_dataloaders(
        temporal_data, labels_data, config, device
    )
    
    # Create model
    print(f"Creating {args.model_type} model...")
    input_dim = temporal_data['enhanced_features'].shape[1]
    model = create_temporal_model(args.model_type, input_dim, config['model'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = TemporalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )
    
    # Train model
    print("Starting training...")
    results = trainer.train()
    
    # Save results
    results_file = os.path.join(args.output_dir, 'training_results.yaml')
    with open(results_file, 'w') as f:
        yaml.dump(results, f)
    
    print(f"Training completed! Results saved to {results_file}")
    print(f"Best validation AUC: {results['best_val_auc']:.4f}")
    print(f"Test AUC: {results['test_metrics']['auc']:.4f}")


if __name__ == "__main__":
    main()
