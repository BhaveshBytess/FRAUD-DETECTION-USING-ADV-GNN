"""
Stage 5 Advanced Training Pipeline

Comprehensive training system for all Stage 5 advanced architectures including:
- Graph Transformer
- Heterogeneous Graph Transformer  
- Temporal Graph Transformer
- Advanced Ensemble Methods

Features:
- Unified training interface
- Advanced optimization techniques
- Comprehensive monitoring and logging
- Model checkpointing and recovery
- Performance tracking and visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import time
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score
import wandb
from tqdm import tqdm
import warnings

# Import models
from .graph_transformer import GraphTransformer, create_graph_transformer
from .hetero_graph_transformer import HeterogeneousGraphTransformer, create_heterogeneous_graph_transformer
from .temporal_graph_transformer import TemporalGraphTransformer, create_temporal_graph_transformer
from .ensemble import AdaptiveEnsemble, CrossValidationEnsemble

# Import utilities
from ...config import load_config
from ...metrics import compute_fraud_detection_metrics
from ...utils import set_seed, EarlyStopping, LearningRateScheduler


class Stage5Trainer:
    """
    Comprehensive trainer for Stage 5 advanced architectures.
    """
    
    def __init__(
        self,
        config_path: str,
        output_dir: str = "experiments/stage5",
        use_wandb: bool = False,
        wandb_project: str = "fraud-detection-stage5"
    ):
        """
        Initialize Stage 5 trainer.
        
        Args:
            config_path: Path to configuration file
            output_dir: Output directory for experiments
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
        """
        self.config = load_config(config_path)
        self.output_dir = Path(output_dir)
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_score = 0.0
        self.training_history = []
        
        # Set random seed
        set_seed(self.config.get('seed', 42))
        
        print(f"Stage 5 Trainer initialized")
        print(f"Config: {config_path}")
        print(f"Output: {self.output_dir}")
        print(f"Device: {self.device}")
    
    def setup_model(self, input_dim: int, input_dims: Optional[Dict[str, int]] = None):
        """
        Setup model based on configuration.
        
        Args:
            input_dim: Input feature dimension
            input_dims: Input dimensions for heterogeneous models
        """
        model_name = self.config['model']['name']
        model_config = self.config['model']
        
        print(f"Setting up model: {model_name}")
        
        if model_name == 'graph_transformer':
            self.model = create_graph_transformer(input_dim, model_config)
            
        elif model_name == 'hetero_graph_transformer':
            if input_dims is None:
                raise ValueError("input_dims required for heterogeneous model")
            self.model = create_heterogeneous_graph_transformer(input_dims, model_config)
            
        elif model_name == 'temporal_graph_transformer':
            self.model = create_temporal_graph_transformer(input_dim, model_config)
            
        elif model_name == 'adaptive_ensemble':
            # For ensemble, we'll set it up separately
            self._setup_ensemble_model(input_dim, input_dims)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Move to device
        if self.model is not None:
            self.model = self.model.to(self.device)
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
    
    def _setup_ensemble_model(self, input_dim: int, input_dims: Optional[Dict[str, int]]):
        """Setup ensemble model with base models."""
        ensemble_config = self.config['model']
        base_models = []
        
        # Create base models
        for base_config in ensemble_config.get('base_models', []):
            if base_config['name'] == 'graph_transformer':
                base_model = create_graph_transformer(input_dim, base_config)
            elif base_config['name'] == 'hetero_graph_transformer':
                base_model = create_heterogeneous_graph_transformer(input_dims, base_config)
            elif base_config['name'] == 'temporal_graph_transformer':
                base_model = create_temporal_graph_transformer(input_dim, base_config)
            else:
                continue
                
            base_models.append(base_model)
        
        # Create ensemble
        self.model = AdaptiveEnsemble(base_models, ensemble_config)
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        optimizer_config = self.config['optimizer']
        scheduler_config = self.config.get('scheduler', {})
        
        # Optimizer
        if optimizer_config['name'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0.0)
            )
        elif optimizer_config['name'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['name']}")
        
        # Scheduler
        if scheduler_config.get('name') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_config.get('name') == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        
        print(f"Optimizer: {optimizer_config['name']}")
        print(f"Scheduler: {scheduler_config.get('name', 'None')}")
    
    def setup_loss_function(self):
        """Setup loss function."""
        loss_config = self.config['loss']
        
        if loss_config['name'] == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_config['name'] == 'focal':
            alpha = loss_config.get('alpha', 1.0)
            gamma = loss_config.get('gamma', 2.0)
            self.criterion = self._focal_loss(alpha, gamma)
        elif loss_config['name'] == 'weighted_cross_entropy':
            weights = torch.tensor(loss_config['weights']).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            raise ValueError(f"Unknown loss function: {loss_config['name']}")
        
        print(f"Loss function: {loss_config['name']}")
    
    def _focal_loss(self, alpha: float, gamma: float):
        """Create focal loss function."""
        def focal_loss(logits, targets):
            ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, targets)
            pt = torch.exp(-ce_loss)
            focal_loss = alpha * (1 - pt) ** gamma * ce_loss
            return focal_loss.mean()
        return focal_loss
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            
            try:
                outputs = self.model(**batch)
                
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('output'))
                else:
                    logits = outputs
                
                # Compute loss
                loss = self.criterion(logits, batch['labels'])
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config['training'].get('grad_clip'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )
                
                self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                total_samples += len(batch['labels'])
                
                # Predictions for metrics
                probs = torch.softmax(logits, dim=-1)
                predictions = probs[:, 1].detach().cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Avg Loss': f"{total_loss / (batch_idx + 1):.4f}"
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        
        try:
            train_auc = roc_auc_score(all_labels, all_predictions)
            train_f1 = f1_score(all_labels, (np.array(all_predictions) > 0.5).astype(int))
        except:
            train_auc = 0.0
            train_f1 = 0.0
        
        metrics = {
            'loss': avg_loss,
            'auc': train_auc,
            'f1': train_f1,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    def validate_epoch(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                try:
                    outputs = self.model(**batch)
                    
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits', outputs.get('output'))
                    else:
                        logits = outputs
                    
                    # Compute loss
                    loss = self.criterion(logits, batch['labels'])
                    total_loss += loss.item()
                    
                    # Predictions for metrics
                    probs = torch.softmax(logits, dim=-1)
                    predictions = probs[:, 1].cpu().numpy()
                    labels = batch['labels'].cpu().numpy()
                    
                    all_predictions.extend(predictions)
                    all_labels.extend(labels)
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        # Compute validation metrics
        avg_loss = total_loss / len(val_loader)
        
        try:
            val_auc = roc_auc_score(all_labels, all_predictions)
            val_f1 = f1_score(all_labels, (np.array(all_predictions) > 0.5).astype(int))
        except:
            val_auc = 0.0
            val_f1 = 0.0
        
        metrics = {
            'loss': avg_loss,
            'auc': val_auc,
            'f1': val_f1
        }
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        input_dim: int,
        input_dims: Optional[Dict[str, int]] = None
    ):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            input_dim: Input feature dimension
            input_dims: Input dimensions for heterogeneous models
        """
        print("🚀 Starting Stage 5 Training")
        print("=" * 50)
        
        # Setup model and training components
        self.setup_model(input_dim, input_dims)
        self.setup_optimizer()
        self.setup_loss_function()
        
        # Initialize W&B if requested
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                config=self.config,
                name=f"{self.config['model']['name']}_{int(time.time())}"
            )
            wandb.watch(self.model)
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config['training'].get('early_stopping_patience', 20),
            min_delta=self.config['training'].get('early_stopping_min_delta', 1e-4)
        )
        
        # Training loop
        epochs = self.config['training']['epochs']
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 30)
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['auc'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_auc': train_metrics['auc'],
                'train_f1': train_metrics['f1'],
                'val_loss': val_metrics['loss'],
                'val_auc': val_metrics['auc'],
                'val_f1': val_metrics['f1'],
                'lr': train_metrics['lr']
            }
            
            self.training_history.append(epoch_metrics)
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}")
            
            # W&B logging
            if self.use_wandb:
                wandb.log(epoch_metrics)
            
            # Save best model
            if val_metrics['auc'] > self.best_score:
                self.best_score = val_metrics['auc']
                self.save_checkpoint(epoch, is_best=True)
                print(f"✓ New best AUC: {self.best_score:.4f}")
            
            # Regular checkpoint
            if epoch % self.config['training'].get('save_every', 10) == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping check
            if early_stopping(val_metrics['auc']):
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Save final results
        self.save_training_history()
        self.generate_training_plots()
        
        print("\n✅ Training completed!")
        print(f"Best validation AUC: {self.best_score:.4f}")
        
        if self.use_wandb:
            wandb.finish()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_score': self.best_score,
            'config': self.config
        }
        
        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def save_training_history(self):
        """Save training history."""
        history_df = pd.DataFrame(self.training_history)
        history_path = self.logs_dir / "training_history.csv"
        history_df.to_csv(history_path, index=False)
        
        # Also save as JSON
        history_json_path = self.logs_dir / "training_history.json"
        with open(history_json_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"Training history saved to {history_path}")
    
    def generate_training_plots(self):
        """Generate training visualization plots."""
        if not self.training_history:
            return
        
        df = pd.DataFrame(self.training_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # AUC plot
        axes[0, 1].plot(df['epoch'], df['train_auc'], label='Train AUC', color='blue')
        axes[0, 1].plot(df['epoch'], df['val_auc'], label='Val AUC', color='red')
        axes[0, 1].set_title('AUC Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 plot
        axes[1, 0].plot(df['epoch'], df['train_f1'], label='Train F1', color='blue')
        axes[1, 0].plot(df['epoch'], df['val_f1'], label='Val F1', color='red')
        axes[1, 0].set_title('F1 Score Over Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot
        axes[1, 1].plot(df['epoch'], df['lr'], color='green')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = self.logs_dir / "training_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to {plot_path}")


def train_stage5_model(config_path: str, train_loader: DataLoader, val_loader: DataLoader):
    """
    Convenience function to train a Stage 5 model.
    
    Args:
        config_path: Path to model configuration
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    trainer = Stage5Trainer(config_path)
    
    # Get input dimensions from data
    sample_batch = next(iter(train_loader))
    
    if 'x' in sample_batch:
        input_dim = sample_batch['x'].shape[-1]
    elif 'features' in sample_batch:
        input_dim = sample_batch['features'].shape[-1]
    else:
        raise ValueError("Cannot determine input dimension from data")
    
    # For heterogeneous models
    input_dims = None
    if 'x_dict' in sample_batch:
        input_dims = {key: tensor.shape[-1] for key, tensor in sample_batch['x_dict'].items()}
    
    # Train model
    trainer.train(train_loader, val_loader, input_dim, input_dims)
    
    return trainer


if __name__ == "__main__":
    # Example usage
    config_path = "configs/stage5/graph_transformer.yaml"
    
    # Create dummy data loaders for testing
    from torch.utils.data import TensorDataset
    
    # Dummy data
    batch_size = 32
    input_dim = 186
    num_samples = 1000
    
    x = torch.randn(num_samples, input_dim)
    edge_index = torch.randint(0, num_samples, (2, 2000))
    labels = torch.randint(0, 2, (num_samples,))
    
    dataset = TensorDataset(x, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Train model
    trainer = train_stage5_model(config_path, train_loader, val_loader)
