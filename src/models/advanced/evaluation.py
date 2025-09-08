"""
Stage 5 Comprehensive Evaluation Framework

This module provides a complete evaluation system for all Stage 5 advanced architectures,
comparing them against Stage 3-4 baselines and establishing new benchmarks.

Key Features:
- Multi-model comparison framework
- Advanced metrics and analysis
- Performance profiling and optimization
- Comprehensive visualization and reporting
- Statistical significance testing
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, 
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from scipy import stats
import warnings

# Import models
from ..han import HAN
from ..temporal_stable import SimpleLSTM, SimpleGRU, SimpleTemporalMLP
from .graph_transformer import GraphTransformer, create_graph_transformer
from .hetero_graph_transformer import HeterogeneousGraphTransformer, create_heterogeneous_graph_transformer
from .temporal_graph_transformer import TemporalGraphTransformer, create_temporal_graph_transformer
from .ensemble import AdaptiveEnsemble, CrossValidationEnsemble


class ModelBenchmark:
    """
    Comprehensive benchmark for comparing multiple fraud detection models.
    """
    
    def __init__(self, output_dir: str = "experiments/stage5/benchmark"):
        self.output_dir = output_dir
        self.models = {}
        self.results = {}
        self.timing_results = {}
        self.memory_usage = {}
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        
    def register_model(
        self, 
        name: str, 
        model: nn.Module, 
        model_type: str,
        stage: int,
        config: Optional[Dict] = None
    ):
        """
        Register a model for benchmarking.
        
        Args:
            name: Model name
            model: Model instance
            model_type: Type of model (graph, temporal, hetero, etc.)
            stage: Stage number (3, 4, or 5)
            config: Model configuration
        """
        self.models[name] = {
            'model': model,
            'type': model_type,
            'stage': stage,
            'config': config or {},
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        print(f"Registered model '{name}' (Stage {stage}, {model_type})")
        print(f"  Parameters: {self.models[name]['parameters']:,}")
        print(f"  Trainable: {self.models[name]['trainable_parameters']:,}")
    
    def evaluate_model(
        self,
        name: str,
        test_data: Dict[str, torch.Tensor],
        test_labels: torch.Tensor,
        device: torch.device
    ) -> Dict[str, Any]:
        """
        Evaluate a single model comprehensively.
        
        Args:
            name: Model name
            test_data: Test data dictionary
            test_labels: True labels
            device: Compute device
            
        Returns:
            Comprehensive evaluation results
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not registered")
        
        model_info = self.models[name]
        model = model_info['model'].to(device)
        model.eval()
        
        print(f"\nEvaluating {name}...")
        
        # Timing and memory tracking
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        start_time = time.time()
        
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated()
        
        predictions = []
        logits_list = []
        
        try:
            with torch.no_grad():
                # Prepare model inputs based on model type
                model_inputs = self._prepare_model_inputs(test_data, model_info['type'])
                
                # Forward pass
                outputs = model(**model_inputs)
                
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('output', None))
                else:
                    logits = outputs
                
                if logits is not None:
                    probs = torch.softmax(logits, dim=-1)
                    predictions = probs[:, 1].cpu().numpy()  # Fraud probabilities
                    logits_list = logits.cpu().numpy()
                else:
                    raise ValueError("Model did not return valid logits")
                    
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            return {
                'error': str(e),
                'success': False
            }
        
        # Timing and memory
        end_time = time.time()
        inference_time = end_time - start_time
        
        if device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = peak_memory - start_memory
        else:
            memory_used = 0
        
        # Convert labels to numpy
        true_labels = test_labels.cpu().numpy()
        
        # Compute metrics
        try:
            metrics = self._compute_comprehensive_metrics(true_labels, predictions, logits_list)
            
            # Add timing and memory info
            metrics.update({
                'inference_time': inference_time,
                'memory_usage_bytes': memory_used,
                'memory_usage_mb': memory_used / (1024 * 1024),
                'samples_per_second': len(predictions) / inference_time,
                'success': True
            })
            
            # Store results
            self.results[name] = metrics
            self.timing_results[name] = inference_time
            self.memory_usage[name] = memory_used
            
            print(f"✓ {name} evaluation completed")
            print(f"  AUC: {metrics['auc']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  Time: {inference_time:.2f}s")
            
            return metrics
            
        except Exception as e:
            print(f"Error computing metrics for {name}: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def evaluate_all_models(
        self,
        test_data: Dict[str, torch.Tensor],
        test_labels: torch.Tensor,
        device: torch.device
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all registered models.
        
        Args:
            test_data: Test data dictionary
            test_labels: True labels
            device: Compute device
            
        Returns:
            Results for all models
        """
        print("Starting comprehensive model evaluation...")
        print(f"Test set size: {len(test_labels)}")
        print(f"Device: {device}")
        
        all_results = {}
        
        for name in self.models.keys():
            try:
                result = self.evaluate_model(name, test_data, test_labels, device)
                all_results[name] = result
            except Exception as e:
                print(f"Failed to evaluate {name}: {e}")
                all_results[name] = {'error': str(e), 'success': False}
        
        return all_results
    
    def _prepare_model_inputs(self, data: Dict[str, torch.Tensor], model_type: str) -> Dict[str, torch.Tensor]:
        """Prepare inputs for different model types."""
        inputs = {}
        
        # Basic inputs
        if 'x' in data:
            inputs['x'] = data['x']
        elif 'features' in data:
            inputs['x'] = data['features']
        
        # Graph inputs
        if model_type in ['graph', 'hetero_graph', 'temporal_graph']:
            if 'edge_index' in data:
                inputs['edge_index'] = data['edge_index']
            if 'edge_attr' in data:
                inputs['edge_attr'] = data['edge_attr']
        
        # Temporal inputs
        if model_type in ['temporal', 'temporal_graph']:
            if 'lengths' in data:
                inputs['lengths'] = data['lengths']
            if 'time_steps' in data:
                inputs['time_steps'] = data['time_steps']
        
        # Heterogeneous inputs
        if model_type == 'hetero_graph':
            if 'x_dict' in data:
                inputs['x_dict'] = data['x_dict']
            if 'edge_index_dict' in data:
                inputs['edge_index_dict'] = data['edge_index_dict']
        
        # Batch info
        if 'batch' in data:
            inputs['batch'] = data['batch']
        
        return inputs
    
    def _compute_comprehensive_metrics(
        self, 
        true_labels: np.ndarray, 
        predictions: np.ndarray,
        logits: np.ndarray
    ) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        
        # Binary predictions
        binary_preds = (predictions > 0.5).astype(int)
        
        # Basic metrics
        metrics = {
            'auc': roc_auc_score(true_labels, predictions),
            'f1': f1_score(true_labels, binary_preds),
            'precision': precision_score(true_labels, binary_preds),
            'recall': recall_score(true_labels, binary_preds),
            'accuracy': accuracy_score(true_labels, binary_preds)
        }
        
        # Additional metrics
        try:
            # Confusion matrix
            cm = confusion_matrix(true_labels, binary_preds)
            tn, fp, fn, tp = cm.ravel()
            
            metrics.update({
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
                'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0.0
            })
            
            # Precision-Recall AUC
            precision_curve, recall_curve, _ = precision_recall_curve(true_labels, predictions)
            metrics['pr_auc'] = np.trapz(precision_curve, recall_curve)
            
        except Exception as e:
            warnings.warn(f"Error computing extended metrics: {e}")
        
        return metrics
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate comprehensive comparison report."""
        
        if not self.results:
            raise ValueError("No evaluation results available")
        
        # Create comparison dataframe
        rows = []
        
        for name, result in self.results.items():
            if not result.get('success', False):
                continue
                
            model_info = self.models[name]
            
            row = {
                'Model': name,
                'Stage': model_info['stage'],
                'Type': model_info['type'],
                'Parameters': model_info['parameters'],
                'AUC': result.get('auc', 0.0),
                'F1': result.get('f1', 0.0),
                'Precision': result.get('precision', 0.0),
                'Recall': result.get('recall', 0.0),
                'Accuracy': result.get('accuracy', 0.0),
                'PR-AUC': result.get('pr_auc', 0.0),
                'Inference_Time(s)': result.get('inference_time', 0.0),
                'Memory_Usage(MB)': result.get('memory_usage_mb', 0.0),
                'Samples/sec': result.get('samples_per_second', 0.0)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by AUC
        df = df.sort_values('AUC', ascending=False)
        
        # Save report
        report_path = f"{self.output_dir}/model_comparison_report.csv"
        df.to_csv(report_path, index=False)
        print(f"Comparison report saved to: {report_path}")
        
        return df
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations."""
        
        if not self.results:
            print("No results to visualize")
            return
        
        # Set up plotting
        plt.style.use('seaborn-v0_8')
        fig_size = (15, 12)
        
        # 1. Performance comparison
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        
        # Extract data for successful models
        successful_models = {name: result for name, result in self.results.items() 
                           if result.get('success', False)}
        
        if not successful_models:
            print("No successful models to visualize")
            return
        
        names = list(successful_models.keys())
        aucs = [successful_models[name]['auc'] for name in names]
        f1s = [successful_models[name]['f1'] for name in names]
        precisions = [successful_models[name]['precision'] for name in names]
        recalls = [successful_models[name]['recall'] for name in names]
        
        # AUC comparison
        axes[0, 0].bar(names, aucs, color='skyblue')
        axes[0, 0].set_title('AUC Comparison')
        axes[0, 0].set_ylabel('AUC Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1 comparison
        axes[0, 1].bar(names, f1s, color='lightgreen')
        axes[0, 1].set_title('F1 Score Comparison')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Precision vs Recall
        axes[1, 0].scatter(recalls, precisions, s=100, alpha=0.7)
        for i, name in enumerate(names):
            axes[1, 0].annotate(name, (recalls[i], precisions[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision vs Recall')
        
        # Performance vs Efficiency
        times = [successful_models[name]['inference_time'] for name in names]
        axes[1, 1].scatter(times, aucs, s=100, alpha=0.7)
        for i, name in enumerate(names):
            axes[1, 1].annotate(name, (times[i], aucs[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Inference Time (s)')
        axes[1, 1].set_ylabel('AUC Score')
        axes[1, 1].set_title('Performance vs Speed')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Stage comparison
        stage_performance = {}
        for name, result in successful_models.items():
            stage = self.models[name]['stage']
            if stage not in stage_performance:
                stage_performance[stage] = []
            stage_performance[stage].append(result['auc'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        stages = sorted(stage_performance.keys())
        stage_aucs = [np.mean(stage_performance[stage]) for stage in stages]
        stage_stds = [np.std(stage_performance[stage]) for stage in stages]
        
        ax.bar([f"Stage {stage}" for stage in stages], stage_aucs, 
               yerr=stage_stds, capsize=5, alpha=0.7)
        ax.set_title('Average Performance by Stage')
        ax.set_ylabel('Average AUC Score')
        
        plt.savefig(f"{self.output_dir}/stage_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {self.output_dir}")
    
    def save_detailed_results(self):
        """Save detailed results to JSON."""
        
        detailed_results = {
            'models': self.models,
            'results': self.results,
            'timing': self.timing_results,
            'memory': self.memory_usage,
            'summary': {
                'total_models': len(self.models),
                'successful_evaluations': sum(1 for r in self.results.values() if r.get('success', False)),
                'best_model': max(
                    (name for name, result in self.results.items() if result.get('success', False)),
                    key=lambda name: self.results[name]['auc'],
                    default=None
                ),
                'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        detailed_results = recursive_convert(detailed_results)
        
        # Save to JSON
        results_path = f"{self.output_dir}/detailed_results.json"
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"Detailed results saved to: {results_path}")


def run_stage5_benchmark():
    """
    Run comprehensive benchmark of all Stage 5 models.
    """
    print("🚀 Starting Stage 5 Comprehensive Benchmark")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = ModelBenchmark("experiments/stage5/benchmark")
    
    # Create test models (simplified for demonstration)
    input_dim = 186
    
    # Graph Transformer
    gt_config = {
        'hidden_dim': 128,
        'num_layers': 3,
        'num_heads': 4,
        'dropout': 0.1
    }
    graph_transformer = create_graph_transformer(input_dim, gt_config)
    benchmark.register_model("Graph_Transformer", graph_transformer, "graph", 5, gt_config)
    
    # Heterogeneous Graph Transformer
    hgt_config = {
        'hidden_dim': 128,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.1
    }
    input_dims = {'transaction': input_dim}
    hetero_gt = create_heterogeneous_graph_transformer(input_dims, hgt_config)
    benchmark.register_model("Hetero_Graph_Transformer", hetero_gt, "hetero_graph", 5, hgt_config)
    
    # Temporal Graph Transformer
    tgt_config = {
        'hidden_dim': 128,
        'num_layers': 3,
        'num_heads': 4,
        'dropout': 0.1,
        'prediction_mode': 'node'
    }
    temporal_gt = create_temporal_graph_transformer(input_dim, tgt_config)
    benchmark.register_model("Temporal_Graph_Transformer", temporal_gt, "temporal_graph", 5, tgt_config)
    
    # Create test data
    batch_size = 100
    test_data = {
        'x': torch.randn(batch_size, input_dim),
        'edge_index': torch.randint(0, batch_size, (2, 200)),
        'x_dict': {'transaction': torch.randn(batch_size, input_dim)},
        'edge_index_dict': {'transaction__to__transaction': torch.randint(0, batch_size, (2, 200))}
    }
    test_labels = torch.randint(0, 2, (batch_size,))
    
    device = torch.device('cpu')  # Use CPU for testing
    
    # Run evaluation
    results = benchmark.evaluate_all_models(test_data, test_labels, device)
    
    # Generate reports
    report_df = benchmark.generate_comparison_report()
    print("\n📊 Model Comparison Report:")
    print(report_df.to_string(index=False))
    
    # Generate visualizations
    benchmark.generate_visualizations()
    
    # Save detailed results
    benchmark.save_detailed_results()
    
    print("\n✅ Stage 5 Benchmark Completed!")
    print(f"Results saved to: {benchmark.output_dir}")
    
    return benchmark


if __name__ == "__main__":
    benchmark = run_stage5_benchmark()
