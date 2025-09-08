# src/temporal_utils.py
"""
Temporal utilities for handling time-series aspects of fraud detection.
Supports temporal feature extraction, windowing, and time-aware data splits.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data, HeteroData

class TemporalDataProcessor:
    """
    Handles temporal aspects of the Elliptic++ dataset including:
    - Time step extraction and normalization
    - Temporal windowing for sequence modeling
    - Time-aware train/val/test splits
    - Temporal feature engineering
    """
    
    def __init__(self, window_size: int = 5, overlap: float = 0.5):
        """
        Initialize temporal processor.
        
        Args:
            window_size: Number of time steps in each temporal window
            overlap: Overlap fraction between consecutive windows (0.0 to 1.0)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.time_steps = None
        self.temporal_features = None
        
    def extract_temporal_features(self, data_path: str) -> Dict[str, torch.Tensor]:
        """
        Extract temporal features from Elliptic++ transaction data.
        
        Args:
            data_path: Path to the ellipticpp data directory
            
        Returns:
            Dictionary containing temporal features and metadata
        """
        # Load transaction features
        tx_features_df = pd.read_csv(f"{data_path}/txs_features.csv")
        
        # Extract time steps
        time_steps = tx_features_df['Time step'].values
        tx_ids = tx_features_df['txId'].values
        
        # Extract all other features (excluding txId and Time step)
        feature_cols = [col for col in tx_features_df.columns if col not in ['txId', 'Time step']]
        features = tx_features_df[feature_cols].values
        
        # Convert to tensors
        temporal_data = {
            'time_steps': torch.from_numpy(time_steps).long(),
            'tx_ids': torch.from_numpy(tx_ids).long(), 
            'features': torch.from_numpy(features).float(),
            'feature_names': feature_cols,
            'num_time_steps': len(np.unique(time_steps)),
            'time_step_range': (time_steps.min(), time_steps.max())
        }
        
        # Store for later use
        self.time_steps = temporal_data['time_steps']
        self.temporal_features = temporal_data['features']
        
        print(f"Extracted temporal features:")
        print(f"  - Time steps: {temporal_data['time_step_range'][0]} to {temporal_data['time_step_range'][1]}")
        print(f"  - Total transactions: {len(tx_ids)}")
        print(f"  - Feature dimensions: {features.shape[1]}")
        print(f"  - Transactions per time step (avg): {len(tx_ids) / temporal_data['num_time_steps']:.1f}")
        
        return temporal_data
    
    def create_temporal_windows(self, data: torch.Tensor, time_steps: torch.Tensor) -> List[Dict]:
        """
        Create sliding temporal windows for sequence modeling.
        
        Args:
            data: Transaction features [num_transactions, num_features]
            time_steps: Time step for each transaction [num_transactions]
            
        Returns:
            List of temporal windows, each containing sequences of transactions
        """
        windows = []
        unique_time_steps = torch.unique(time_steps)
        
        # Calculate step size based on overlap
        step_size = max(1, int(self.window_size * (1 - self.overlap)))
        
        for start_idx in range(0, len(unique_time_steps) - self.window_size + 1, step_size):
            window_time_steps = unique_time_steps[start_idx:start_idx + self.window_size]
            
            # Get all transactions in this temporal window
            window_mask = torch.isin(time_steps, window_time_steps)
            window_data = data[window_mask]
            window_times = time_steps[window_mask]
            
            # Sort by time step within window
            sort_indices = torch.argsort(window_times)
            window_data = window_data[sort_indices]
            window_times = window_times[sort_indices]
            
            window_info = {
                'data': window_data,
                'time_steps': window_times,
                'window_range': (window_time_steps[0].item(), window_time_steps[-1].item()),
                'num_transactions': len(window_data)
            }
            windows.append(window_info)
            
        print(f"Created {len(windows)} temporal windows:")
        print(f"  - Window size: {self.window_size} time steps")
        print(f"  - Overlap: {self.overlap}")
        print(f"  - Average transactions per window: {np.mean([w['num_transactions'] for w in windows]):.1f}")
        
        return windows
    
    def temporal_train_val_test_split(self, time_steps: torch.Tensor, 
                                    train_ratio: float = 0.7, 
                                    val_ratio: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create temporal train/val/test splits based on time steps.
        Early time steps for training, later for validation and testing.
        
        Args:
            time_steps: Time step for each transaction
            train_ratio: Fraction of time steps for training
            val_ratio: Fraction of time steps for validation
            
        Returns:
            Tuple of (train_mask, val_mask, test_mask)
        """
        unique_time_steps = torch.unique(time_steps).sort()[0]
        num_time_steps = len(unique_time_steps)
        
        # Calculate split points
        train_split = int(num_time_steps * train_ratio)
        val_split = int(num_time_steps * (train_ratio + val_ratio))
        
        # Define time step ranges
        train_time_steps = unique_time_steps[:train_split]
        val_time_steps = unique_time_steps[train_split:val_split]
        test_time_steps = unique_time_steps[val_split:]
        
        # Create masks
        train_mask = torch.isin(time_steps, train_time_steps)
        val_mask = torch.isin(time_steps, val_time_steps)
        test_mask = torch.isin(time_steps, test_time_steps)
        
        print(f"Temporal split created:")
        print(f"  - Train: time steps {train_time_steps[0]}-{train_time_steps[-1]} ({train_mask.sum()} transactions)")
        print(f"  - Val: time steps {val_time_steps[0]}-{val_time_steps[-1]} ({val_mask.sum()} transactions)")
        print(f"  - Test: time steps {test_time_steps[0]}-{test_time_steps[-1]} ({test_mask.sum()} transactions)")
        
        return train_mask, val_mask, test_mask
    
    def add_temporal_features(self, features: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """
        Add engineered temporal features to the existing feature set.
        
        Args:
            features: Original transaction features [num_transactions, num_features]
            time_steps: Time step for each transaction [num_transactions]
            
        Returns:
            Enhanced features with temporal information
        """
        # Normalize time steps to [0, 1] range
        min_time = time_steps.float().min()
        max_time = time_steps.float().max()
        normalized_time = (time_steps.float() - min_time) / (max_time - min_time)
        
        # Create cyclical time features (assuming time steps represent regular intervals)
        time_sin = torch.sin(2 * np.pi * normalized_time)
        time_cos = torch.cos(2 * np.pi * normalized_time)
        
        # Time step as direct feature
        time_feature = time_steps.float().unsqueeze(1)
        normalized_time_feature = normalized_time.unsqueeze(1)
        time_sin_feature = time_sin.unsqueeze(1)
        time_cos_feature = time_cos.unsqueeze(1)
        
        # Concatenate all temporal features
        temporal_features = torch.cat([
            time_feature,
            normalized_time_feature, 
            time_sin_feature,
            time_cos_feature
        ], dim=1)
        
        # Combine with original features
        enhanced_features = torch.cat([features, temporal_features], dim=1)
        
        print(f"Added temporal features:")
        print(f"  - Original features: {features.shape[1]}")
        print(f"  - Temporal features: {temporal_features.shape[1]}")
        print(f"  - Total features: {enhanced_features.shape[1]}")
        
        return enhanced_features

def load_temporal_ellipticpp(data_path: str, window_size: int = 5, 
                           add_temporal_feats: bool = True) -> Dict:
    """
    Load Elliptic++ data with temporal processing.
    
    Args:
        data_path: Path to ellipticpp data directory
        window_size: Size of temporal windows
        add_temporal_feats: Whether to add engineered temporal features
        
    Returns:
        Dictionary containing processed temporal data
    """
    processor = TemporalDataProcessor(window_size=window_size)
    
    # Extract temporal features
    temporal_data = processor.extract_temporal_features(data_path)
    
    # Add temporal features if requested
    if add_temporal_feats:
        enhanced_features = processor.add_temporal_features(
            temporal_data['features'], 
            temporal_data['time_steps']
        )
        temporal_data['enhanced_features'] = enhanced_features
    
    # Create temporal windows
    windows = processor.create_temporal_windows(
        temporal_data['features'], 
        temporal_data['time_steps']
    )
    temporal_data['windows'] = windows
    
    # Create temporal splits
    train_mask, val_mask, test_mask = processor.temporal_train_val_test_split(
        temporal_data['time_steps']
    )
    temporal_data['temporal_splits'] = {
        'train_mask': train_mask,
        'val_mask': val_mask, 
        'test_mask': test_mask
    }
    
    return temporal_data
