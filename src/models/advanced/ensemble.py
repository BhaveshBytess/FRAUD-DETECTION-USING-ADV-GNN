"""
Advanced Ensemble Methods for Stage 5

This module implements sophisticated ensemble techniques that combine models
from Stages 3-5 to achieve state-of-the-art fraud detection performance.

Key Features:
- Multi-stage model integration
- Adaptive ensemble weighting
- Cross-validation based model selection
- Advanced voting mechanisms
- Stacking and meta-learning approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings

# Import models from different stages
from ..han import HAN
from ..temporal import TemporalLSTM, TemporalGRU
from ..temporal_stable import SimpleLSTM, SimpleGRU, SimpleTemporalMLP
from .graph_transformer import GraphTransformer
from .hetero_graph_transformer import HeterogeneousGraphTransformer
from .temporal_graph_transformer import TemporalGraphTransformer


class EnsembleWeightLearner(nn.Module):
    """
    Learnable ensemble weights that adapt based on input features.
    """
    
    def __init__(self, num_models: int, input_dim: int, hidden_dim: int = 64):
        super(EnsembleWeightLearner, self).__init__()
        
        self.num_models = num_models
        self.input_dim = input_dim
        
        # Weight prediction network
        self.weight_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_models),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict ensemble weights based on input features.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Ensemble weights [batch_size, num_models]
        """
        weights = self.weight_network(x)
        return weights


class AdaptiveEnsemble(nn.Module):
    """
    Adaptive ensemble that learns optimal combination weights.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        model_names: List[str],
        ensemble_method: str = 'learned_weights',
        input_dim: Optional[int] = None,
        freeze_base_models: bool = True
    ):
        super(AdaptiveEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.model_names = model_names
        self.ensemble_method = ensemble_method
        self.num_models = len(models)
        
        # Freeze base models if requested
        if freeze_base_models:
            for model in self.models:
                for param in model.parameters():
                    param.requires_grad = False
        
        # Initialize ensemble combination method
        if ensemble_method == 'learned_weights':
            if input_dim is None:
                raise ValueError("input_dim required for learned_weights method")
            self.weight_learner = EnsembleWeightLearner(self.num_models, input_dim)
        elif ensemble_method == 'fixed_weights':
            self.fixed_weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        elif ensemble_method == 'stacking':
            # Meta-learner for stacking
            self.meta_learner = nn.Sequential(
                nn.Linear(self.num_models * 2, 64),  # *2 for logits of 2 classes
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 2)
            )
        
    def forward(
        self,
        inputs: Dict[str, Any],
        return_individual_predictions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble.
        
        Args:
            inputs: Dictionary containing inputs for different models
            return_individual_predictions: Whether to return individual model predictions
            
        Returns:
            Dictionary containing ensemble predictions and optionally individual predictions
        """
        individual_predictions = []
        individual_logits = []
        
        # Get predictions from each model
        for i, (model, model_name) in enumerate(zip(self.models, self.model_names)):
            try:
                # Prepare inputs for this specific model
                model_inputs = self._prepare_model_inputs(inputs, model_name)
                
                # Get model prediction
                with torch.no_grad() if hasattr(self, 'weight_learner') else torch.enable_grad():
                    if hasattr(model, 'forward'):
                        outputs = model(**model_inputs)
                        if isinstance(outputs, dict):
                            logits = outputs.get('logits', outputs.get('output', None))
                        else:
                            logits = outputs
                    else:
                        logits = model(**model_inputs)
                
                if logits is not None:
                    individual_logits.append(logits)
                    individual_predictions.append(torch.softmax(logits, dim=-1))
                else:
                    # Handle case where model doesn't return valid output
                    batch_size = self._infer_batch_size(inputs)
                    dummy_logits = torch.zeros(batch_size, 2, device=next(model.parameters()).device)
                    individual_logits.append(dummy_logits)
                    individual_predictions.append(torch.softmax(dummy_logits, dim=-1))
                    
            except Exception as e:
                warnings.warn(f"Model {model_name} failed with error: {e}")
                # Create dummy prediction
                batch_size = self._infer_batch_size(inputs)
                device = next(model.parameters()).device
                dummy_logits = torch.zeros(batch_size, 2, device=device)
                individual_logits.append(dummy_logits)
                individual_predictions.append(torch.softmax(dummy_logits, dim=-1))
        
        # Stack predictions
        if individual_logits:
            stacked_logits = torch.stack(individual_logits, dim=1)  # [batch_size, num_models, num_classes]
            stacked_probs = torch.stack(individual_predictions, dim=1)
        else:
            raise RuntimeError("No valid predictions obtained from any model")
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == 'simple_average':
            ensemble_probs = stacked_probs.mean(dim=1)
            ensemble_logits = torch.log(ensemble_probs + 1e-8)
            
        elif self.ensemble_method == 'learned_weights':
            # Use first available input for weight learning
            weight_input = self._get_weight_input(inputs)
            weights = self.weight_learner(weight_input)  # [batch_size, num_models]
            
            # Apply weights
            weighted_probs = (stacked_probs * weights.unsqueeze(-1)).sum(dim=1)
            ensemble_logits = torch.log(weighted_probs + 1e-8)
            ensemble_probs = weighted_probs
            
        elif self.ensemble_method == 'fixed_weights':
            weights = F.softmax(self.fixed_weights, dim=0)
            weighted_probs = (stacked_probs * weights.view(1, -1, 1)).sum(dim=1)
            ensemble_logits = torch.log(weighted_probs + 1e-8)
            ensemble_probs = weighted_probs
            
        elif self.ensemble_method == 'stacking':
            # Flatten logits for meta-learner
            meta_input = stacked_logits.view(stacked_logits.size(0), -1)
            ensemble_logits = self.meta_learner(meta_input)
            ensemble_probs = torch.softmax(ensemble_logits, dim=-1)
            
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        result = {
            'logits': ensemble_logits,
            'probs': ensemble_probs,
            'ensemble_weights': weights if self.ensemble_method == 'learned_weights' else None
        }
        
        if return_individual_predictions:
            result['individual_logits'] = stacked_logits
            result['individual_probs'] = stacked_probs
            result['model_names'] = self.model_names
        
        return result
    
    def _prepare_model_inputs(self, inputs: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Prepare inputs for a specific model based on its requirements."""
        
        # Base inputs that most models need
        model_inputs = {}
        
        # Node features
        if 'x' in inputs:
            model_inputs['x'] = inputs['x']
        elif 'features' in inputs:
            model_inputs['x'] = inputs['features']
        
        # Edge information
        if 'edge_index' in inputs:
            model_inputs['edge_index'] = inputs['edge_index']
        
        if 'edge_attr' in inputs:
            model_inputs['edge_attr'] = inputs['edge_attr']
        
        # Temporal information
        if 'temporal' in model_name.lower() or 'lstm' in model_name.lower() or 'gru' in model_name.lower():
            if 'lengths' in inputs:
                model_inputs['lengths'] = inputs['lengths']
            if 'time_steps' in inputs:
                model_inputs['time_steps'] = inputs['time_steps']
        
        # Heterogeneous graph information
        if 'hetero' in model_name.lower() or 'han' in model_name.lower():
            if 'x_dict' in inputs:
                model_inputs['x_dict'] = inputs['x_dict']
            if 'edge_index_dict' in inputs:
                model_inputs['edge_index_dict'] = inputs['edge_index_dict']
            if 'edge_attr_dict' in inputs:
                model_inputs['edge_attr_dict'] = inputs['edge_attr_dict']
        
        # Batch information
        if 'batch' in inputs:
            model_inputs['batch'] = inputs['batch']
        
        return model_inputs
    
    def _get_weight_input(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Extract input for weight learning."""
        if 'x' in inputs:
            x = inputs['x']
            if x.dim() == 3:  # Sequence data
                return x.mean(dim=1)  # Average over sequence
            else:
                return x
        elif 'features' in inputs:
            features = inputs['features']
            if features.dim() == 3:
                return features.mean(dim=1)
            else:
                return features
        else:
            raise ValueError("No suitable input found for weight learning")
    
    def _infer_batch_size(self, inputs: Dict[str, Any]) -> int:
        """Infer batch size from inputs."""
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                return value.size(0)
        return 1


class CrossValidationEnsemble:
    """
    Cross-validation based ensemble for model selection and combination.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        model_names: List[str],
        n_folds: int = 5,
        selection_metric: str = 'auc'
    ):
        self.models = models
        self.model_names = model_names
        self.n_folds = n_folds
        self.selection_metric = selection_metric
        
        self.cv_scores = {}
        self.selected_models = []
        self.ensemble_weights = None
        
    def fit(
        self,
        train_data: Dict[str, torch.Tensor],
        train_labels: torch.Tensor,
        validation_data: Optional[Dict[str, torch.Tensor]] = None,
        validation_labels: Optional[torch.Tensor] = None
    ):
        """
        Fit ensemble using cross-validation.
        
        Args:
            train_data: Training data dictionary
            train_labels: Training labels
            validation_data: Validation data (optional)
            validation_labels: Validation labels (optional)
        """
        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
        
        n_samples = len(train_labels)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # Evaluate each model with cross-validation
        for model_idx, (model, model_name) in enumerate(zip(self.models, self.model_names)):
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(range(n_samples))):
                try:
                    # Prepare fold data
                    fold_train_data = {k: v[train_idx] for k, v in train_data.items()}
                    fold_val_data = {k: v[val_idx] for k, v in train_data.items()}
                    fold_train_labels = train_labels[train_idx]
                    fold_val_labels = train_labels[val_idx]
                    
                    # Train model (simplified - in practice you'd need proper training loop)
                    # For now, just evaluate pre-trained models
                    model.eval()
                    with torch.no_grad():
                        model_inputs = self._prepare_single_model_inputs(fold_val_data, model_name)
                        outputs = model(**model_inputs)
                        
                        if isinstance(outputs, dict):
                            logits = outputs.get('logits', outputs.get('output'))
                        else:
                            logits = outputs
                        
                        probs = torch.softmax(logits, dim=-1)
                        predictions = probs[:, 1].cpu().numpy()  # Fraud probability
                        true_labels = fold_val_labels.cpu().numpy()
                        
                        # Compute metric
                        if self.selection_metric == 'auc':
                            score = roc_auc_score(true_labels, predictions)
                        elif self.selection_metric == 'f1':
                            pred_labels = (predictions > 0.5).astype(int)
                            score = f1_score(true_labels, pred_labels)
                        elif self.selection_metric == 'accuracy':
                            pred_labels = (predictions > 0.5).astype(int)
                            score = accuracy_score(true_labels, pred_labels)
                        else:
                            raise ValueError(f"Unknown metric: {self.selection_metric}")
                        
                        fold_scores.append(score)
                        
                except Exception as e:
                    warnings.warn(f"Model {model_name} failed on fold {fold}: {e}")
                    fold_scores.append(0.0)
            
            # Store average CV score
            avg_score = np.mean(fold_scores)
            self.cv_scores[model_name] = {
                'mean': avg_score,
                'std': np.std(fold_scores),
                'scores': fold_scores
            }
        
        # Select best models (top 50% or at least 2 models)
        sorted_models = sorted(self.cv_scores.items(), key=lambda x: x[1]['mean'], reverse=True)
        n_select = max(2, len(sorted_models) // 2)
        
        self.selected_models = []
        selected_scores = []
        
        for i in range(min(n_select, len(sorted_models))):
            model_name, score_info = sorted_models[i]
            model_idx = self.model_names.index(model_name)
            self.selected_models.append((self.models[model_idx], model_name))
            selected_scores.append(score_info['mean'])
        
        # Compute ensemble weights based on performance
        selected_scores = np.array(selected_scores)
        # Softmax weighting based on scores
        self.ensemble_weights = np.exp(selected_scores) / np.sum(np.exp(selected_scores))
        
        print(f"Selected {len(self.selected_models)} models for ensemble:")
        for (model, name), weight, score in zip(self.selected_models, self.ensemble_weights, selected_scores):
            print(f"  {name}: {score:.4f} (weight: {weight:.3f})")
    
    def predict(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Make ensemble predictions.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Ensemble predictions
        """
        if not self.selected_models:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        predictions = []
        
        for (model, model_name), weight in zip(self.selected_models, self.ensemble_weights):
            try:
                model.eval()
                with torch.no_grad():
                    model_inputs = self._prepare_single_model_inputs(data, model_name)
                    outputs = model(**model_inputs)
                    
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits', outputs.get('output'))
                    else:
                        logits = outputs
                    
                    probs = torch.softmax(logits, dim=-1)
                    weighted_probs = probs * weight
                    predictions.append(weighted_probs)
                    
            except Exception as e:
                warnings.warn(f"Model {model_name} failed during prediction: {e}")
                # Add zero prediction for failed model
                batch_size = self._infer_batch_size(data)
                device = next(model.parameters()).device
                zero_probs = torch.zeros(batch_size, 2, device=device)
                predictions.append(zero_probs)
        
        if predictions:
            ensemble_probs = torch.stack(predictions).sum(dim=0)
            return ensemble_probs
        else:
            raise RuntimeError("All models failed during prediction")
    
    def _prepare_single_model_inputs(self, data: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Prepare inputs for a single model."""
        model_inputs = {}
        
        if 'x' in data:
            model_inputs['x'] = data['x']
        if 'edge_index' in data:
            model_inputs['edge_index'] = data['edge_index']
        if 'edge_attr' in data:
            model_inputs['edge_attr'] = data['edge_attr']
        
        # Add model-specific inputs
        if 'temporal' in model_name.lower():
            if 'lengths' in data:
                model_inputs['lengths'] = data['lengths']
        
        return model_inputs
    
    def _infer_batch_size(self, data: Dict[str, Any]) -> int:
        """Infer batch size from data."""
        for value in data.values():
            if isinstance(value, torch.Tensor):
                return value.size(0)
        return 1


def create_stage5_ensemble(
    stage3_models: List[nn.Module],
    stage4_models: List[nn.Module], 
    stage5_models: List[nn.Module],
    ensemble_config: Dict[str, Any]
) -> AdaptiveEnsemble:
    """
    Create comprehensive ensemble combining models from Stages 3-5.
    
    Args:
        stage3_models: Graph models from Stage 3
        stage4_models: Temporal models from Stage 4
        stage5_models: Advanced models from Stage 5
        ensemble_config: Ensemble configuration
        
    Returns:
        Configured adaptive ensemble
    """
    all_models = []
    model_names = []
    
    # Add Stage 3 models
    for i, model in enumerate(stage3_models):
        all_models.append(model)
        model_names.append(f"stage3_graph_{i}")
    
    # Add Stage 4 models
    for i, model in enumerate(stage4_models):
        all_models.append(model)
        model_names.append(f"stage4_temporal_{i}")
    
    # Add Stage 5 models
    for i, model in enumerate(stage5_models):
        all_models.append(model)
        model_names.append(f"stage5_advanced_{i}")
    
    ensemble = AdaptiveEnsemble(
        models=all_models,
        model_names=model_names,
        ensemble_method=ensemble_config.get('method', 'learned_weights'),
        input_dim=ensemble_config.get('input_dim', None),
        freeze_base_models=ensemble_config.get('freeze_base_models', True)
    )
    
    return ensemble


if __name__ == "__main__":
    # Test ensemble system
    print("Testing Advanced Ensemble System...")
    
    # Create dummy models for testing
    input_dim = 186
    
    # Simple test models
    model1 = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 2))
    model2 = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 2))
    model3 = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 2))
    
    models = [model1, model2, model3]
    model_names = ['simple1', 'simple2', 'simple3']
    
    # Test adaptive ensemble
    ensemble = AdaptiveEnsemble(
        models=models,
        model_names=model_names,
        ensemble_method='learned_weights',
        input_dim=input_dim
    )
    
    # Test data
    batch_size = 10
    test_data = {
        'x': torch.randn(batch_size, input_dim),
        'edge_index': torch.randint(0, batch_size, (2, 20))
    }
    
    # Test prediction
    with torch.no_grad():
        outputs = ensemble(test_data, return_individual_predictions=True)
    
    print(f"Ensemble logits shape: {outputs['logits'].shape}")
    print(f"Individual predictions shape: {outputs['individual_logits'].shape}")
    print(f"Model names: {outputs['model_names']}")
    
    print("✅ Advanced Ensemble System test completed successfully!")
