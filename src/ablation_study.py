#!/usr/bin/env python3
"""
Ablation Study for Hypergraph Neural Networks (PhenomNN)
Stage 5: Comprehensive analysis and comparison of hypergraph configurations.
"""

import os
import sys
import json
import yaml
import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from train_baseline import train as train_main
from config import Config
from utils import set_seed


class AblationStudy:
    """
    Comprehensive ablation study for hypergraph neural networks.
    
    Studies the impact of:
    1. Lambda weights (lambda0, lambda1) 
    2. Number of iterations
    3. Convergence thresholds
    4. Comparison with baseline models
    """
    
    def __init__(self, base_config_path: str, output_dir: str = "experiments/ablation"):
        """
        Initialize ablation study.
        
        Args:
            base_config_path: Path to base hypergraph config
            output_dir: Directory to save results
        """
        self.base_config_path = base_config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base configuration
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
            
        self.results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_config_variant(self, modifications: Dict[str, Any]) -> str:
        """
        Create a configuration variant with modifications.
        
        Args:
            modifications: Dictionary of config modifications
            
        Returns:
            Path to the created config file
        """
        import copy
        config = copy.deepcopy(self.base_config)
        
        # Apply modifications
        for key, value in modifications.items():
            if '.' in key:
                # Nested key like 'model.lambda0'
                keys = key.split('.')
                current = config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    elif not isinstance(current[k], dict):
                        # Convert to dict if it's not already
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                config[key] = value
        
        # Save config variant
        config_name = f"config_{self.experiment_id}_{len(self.results)}.yaml"
        config_path = self.output_dir / config_name
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        return str(config_path)
    
    def run_single_experiment(self, config_path: str, experiment_name: str, 
                            epochs: int = 50, sample: int = 500) -> Dict[str, Any]:
        """
        Run a single training experiment.
        
        Args:
            config_path: Path to config file
            experiment_name: Name of the experiment
            epochs: Number of training epochs
            sample: Sample size for data
            
        Returns:
            Dictionary with experiment results
        """
        print(f"\n=== Running experiment: {experiment_name} ===")
        
        # Set seed for reproducibility
        set_seed(42)
        
        try:
            # Create arguments for train_main
            output_dir = self.output_dir
            class Args:
                def __init__(self):
                    self.config = config_path
                    self.data_path = 'data/ellipticpp/ellipticpp.pt'
                    self.out_dir = str(output_dir / f"exp_{experiment_name}")
                    self.model = 'hypergraph'  # Force hypergraph model for ablation
                    self.epochs = epochs
                    self.lr = 1e-3
                    self.hidden_dim = 128
                    self.weight_decay = 1e-5
                    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    self.sample = sample
                    self.seed = 42
                    # Hypergraph specific
                    self.layer_type = 'full'
                    self.num_layers = 3
                    self.dropout = 0.2
                    self.use_residual = True
                    self.use_batch_norm = False
                    self.lambda0_init = 1.0
                    self.lambda1_init = 1.0
                    self.alpha_init = 0.1
                    self.max_iterations = 10
                    self.convergence_threshold = 1e-4
                    
            args = Args()
            
            # Create output directory
            os.makedirs(args.out_dir, exist_ok=True)
            
            # Run training
            results = train_main(args)
            
            # Extract key metrics
            result = {
                'experiment_name': experiment_name,
                'config_path': config_path,
                'final_test_auc': results.get('auc', 0.0) if results else 0.0,
                'final_test_pr_auc': results.get('pr_auc', 0.0) if results else 0.0,
                'final_test_f1': results.get('f1', 0.0) if results else 0.0,
                'final_test_precision': results.get('precision', 0.0) if results else 0.0,
                'final_test_recall': results.get('recall', 0.0) if results else 0.0,
                'status': 'success' if results and not results.get('error') else 'failed',
                'error': results.get('error', None) if results else None
            }
            
            print(f"Results: AUC={result['final_test_auc']:.4f}, "
                  f"PR-AUC={result['final_test_pr_auc']:.4f}, "
                  f"F1={result['final_test_f1']:.4f}")
            
            return result
            
        except Exception as e:
            print(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'experiment_name': experiment_name,
                'config_path': config_path,
                'final_test_auc': 0.0,
                'final_test_pr_auc': 0.0,
                'final_test_f1': 0.0,
                'final_test_precision': 0.0,
                'final_test_recall': 0.0,
                'status': 'failed',
                'error': str(e)
            }
    
    def lambda_weight_ablation(self, epochs: int = 30, sample: int = 500) -> List[Dict[str, Any]]:
        """
        Study the impact of lambda0 and lambda1 weights.
        
        Args:
            epochs: Number of training epochs
            sample: Sample size
            
        Returns:
            List of experiment results
        """
        print("\n🔬 Running Lambda Weight Ablation Study...")
        
        # Test different lambda combinations (reduced for testing)
        lambda_values = [0.5, 1.0, 2.0]
        results = []
        
        for lambda0, lambda1 in itertools.product(lambda_values, lambda_values):
            modifications = {
                'lambda0_init': float(lambda0),
                'lambda1_init': float(lambda1)
            }
            
            config_path = self.create_config_variant(modifications)
            experiment_name = f"lambda_ablation_l0_{lambda0}_l1_{lambda1}"
            
            result = self.run_single_experiment(config_path, experiment_name, epochs, sample)
            result.update({
                'lambda0': lambda0,
                'lambda1': lambda1,
                'ablation_type': 'lambda_weights'
            })
            
            results.append(result)
            self.results.append(result)
            
        return results
    
    def iteration_ablation(self, epochs: int = 30, sample: int = 500) -> List[Dict[str, Any]]:
        """
        Study the impact of number of iterations.
        
        Args:
            epochs: Number of training epochs
            sample: Sample size
            
        Returns:
            List of experiment results
        """
        print("\n🔬 Running Iteration Count Ablation Study...")
        
        # Test different iteration counts
        iteration_values = [5, 10, 20, 50, 100]
        results = []
        
        for num_iterations in iteration_values:
            modifications = {
                'max_iterations': int(num_iterations)
            }
            
            config_path = self.create_config_variant(modifications)
            experiment_name = f"iteration_ablation_iter_{num_iterations}"
            
            result = self.run_single_experiment(config_path, experiment_name, epochs, sample)
            result.update({
                'num_iterations': num_iterations,
                'ablation_type': 'num_iterations'
            })
            
            results.append(result)
            self.results.append(result)
            
        return results
    
    def convergence_threshold_ablation(self, epochs: int = 30, sample: int = 500) -> List[Dict[str, Any]]:
        """
        Study the impact of convergence threshold.
        
        Args:
            epochs: Number of training epochs
            sample: Sample size
            
        Returns:
            List of experiment results
        """
        print("\n🔬 Running Convergence Threshold Ablation Study...")
        
        # Test different convergence thresholds
        threshold_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        results = []
        
        for threshold in threshold_values:
            modifications = {
                'convergence_threshold': float(threshold)
            }
            
            config_path = self.create_config_variant(modifications)
            experiment_name = f"convergence_ablation_thresh_{threshold:.0e}"
            
            result = self.run_single_experiment(config_path, experiment_name, epochs, sample)
            result.update({
                'convergence_threshold': threshold,
                'ablation_type': 'convergence_threshold'
            })
            
            results.append(result)
            self.results.append(result)
            
        return results
    
    def baseline_comparison(self, epochs: int = 50, sample: int = 500) -> List[Dict[str, Any]]:
        """
        Compare hypergraph model with baseline models.
        
        Args:
            epochs: Number of training epochs
            sample: Sample size
            
        Returns:
            List of experiment results
        """
        print("\n🔬 Running Baseline Model Comparison...")
        
        baseline_configs = {
            'gcn': 'configs/baseline.yaml',
            'graphsage': 'configs/baseline.yaml', 
            'han': 'configs/baseline.yaml'
        }
        
        results = []
        
        # First run optimal hypergraph config
        optimal_config_path = self.create_config_variant({})  # Use base config
        result = self.run_single_experiment(optimal_config_path, "hypergraph_optimal", epochs, sample)
        result.update({
            'model_type': 'hypergraph',
            'ablation_type': 'baseline_comparison'
        })
        results.append(result)
        self.results.append(result)
        
        # Run baseline models
        for model_name, config_path in baseline_configs.items():
            if os.path.exists(config_path):
                # Modify baseline config to use specific model
                with open(config_path, 'r') as f:
                    baseline_config = yaml.safe_load(f)
                
                baseline_config['model']['name'] = model_name
                
                # Save modified config
                modified_config_path = self.output_dir / f"baseline_{model_name}_{self.experiment_id}.yaml"
                with open(modified_config_path, 'w') as f:
                    yaml.dump(baseline_config, f, default_flow_style=False)
                
                result = self.run_single_experiment(str(modified_config_path), f"baseline_{model_name}", epochs, sample)
                result.update({
                    'model_type': model_name,
                    'ablation_type': 'baseline_comparison'
                })
                results.append(result)
                self.results.append(result)
        
        return results
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report with visualizations."""
        print("\n📊 Generating Analysis Report...")
        
        if not self.results:
            print("No results to analyze!")
            return
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Create report directory
        report_dir = self.output_dir / f"report_{self.experiment_id}"
        report_dir.mkdir(exist_ok=True)
        
        # Save raw results
        df.to_csv(report_dir / "raw_results.csv", index=False)
        
        # Generate visualizations
        self._create_visualizations(df, report_dir)
        
        # Generate text report
        self._create_text_report(df, report_dir)
        
        print(f"📈 Analysis report saved to: {report_dir}")
    
    def _create_visualizations(self, df: pd.DataFrame, report_dir: Path):
        """Create visualization plots."""
        plt.style.use('seaborn-v0_8')
        
        # 1. Lambda weight heatmap
        lambda_results = df[df['ablation_type'] == 'lambda_weights']
        if not lambda_results.empty:
            pivot_auc = lambda_results.pivot(index='lambda0', columns='lambda1', values='final_test_auc')
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_auc, annot=True, fmt='.3f', cmap='viridis')
            plt.title('AUC Performance vs Lambda Weights')
            plt.xlabel('Lambda1 (Star Weight)')
            plt.ylabel('Lambda0 (Clique Weight)')
            plt.tight_layout()
            plt.savefig(report_dir / "lambda_weights_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Iteration count analysis
        iteration_results = df[df['ablation_type'] == 'num_iterations']
        if not iteration_results.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(iteration_results['num_iterations'], iteration_results['final_test_auc'], 'o-', linewidth=2, markersize=8)
            plt.xlabel('Number of Iterations')
            plt.ylabel('Test AUC')
            plt.title('Performance vs Number of Iterations')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(report_dir / "iteration_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Baseline comparison
        baseline_results = df[df['ablation_type'] == 'baseline_comparison']
        if not baseline_results.empty:
            plt.figure(figsize=(12, 6))
            
            metrics = ['final_test_auc', 'final_test_pr_auc', 'final_test_f1']
            metric_names = ['AUC', 'PR-AUC', 'F1']
            
            x = np.arange(len(baseline_results))
            width = 0.25
            
            for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                plt.bar(x + i*width, baseline_results[metric], width, label=name, alpha=0.8)
            
            plt.xlabel('Models')
            plt.ylabel('Performance')
            plt.title('Baseline Model Comparison')
            plt.xticks(x + width, baseline_results['model_type'], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(report_dir / "baseline_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_text_report(self, df: pd.DataFrame, report_dir: Path):
        """Create comprehensive text report."""
        report_path = report_dir / "analysis_report.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# Hypergraph Neural Network Ablation Study Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Experiment ID:** {self.experiment_id}\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"- Total experiments: {len(df)}\n")
            f.write(f"- Successful experiments: {len(df[df['status'] == 'success'])}\n")
            f.write(f"- Failed experiments: {len(df[df['status'] == 'failed'])}\n\n")
            
            # Best performing configurations
            successful_df = df[df['status'] == 'success']
            if not successful_df.empty:
                best_auc = successful_df.loc[successful_df['final_test_auc'].idxmax()]
                best_f1 = successful_df.loc[successful_df['final_test_f1'].idxmax()]
                
                f.write("## Best Performing Configurations\n\n")
                f.write(f"### Best AUC: {best_auc['final_test_auc']:.4f}\n")
                f.write(f"- Experiment: {best_auc['experiment_name']}\n")
                if 'lambda0' in best_auc:
                    f.write(f"- Lambda0: {best_auc['lambda0']}, Lambda1: {best_auc['lambda1']}\n")
                f.write(f"- F1: {best_auc['final_test_f1']:.4f}\n\n")
                
                f.write(f"### Best F1: {best_f1['final_test_f1']:.4f}\n")
                f.write(f"- Experiment: {best_f1['experiment_name']}\n")
                if 'lambda0' in best_f1:
                    f.write(f"- Lambda0: {best_f1['lambda0']}, Lambda1: {best_f1['lambda1']}\n")
                f.write(f"- AUC: {best_f1['final_test_auc']:.4f}\n\n")
            
            # Ablation study results
            for ablation_type in df['ablation_type'].unique():
                if pd.isna(ablation_type):
                    continue
                    
                subset = df[df['ablation_type'] == ablation_type]
                f.write(f"## {ablation_type.replace('_', ' ').title()} Results\n\n")
                
                for _, row in subset.iterrows():
                    f.write(f"- **{row['experiment_name']}**: ")
                    f.write(f"AUC={row['final_test_auc']:.4f}, ")
                    f.write(f"F1={row['final_test_f1']:.4f}")
                    if row['status'] == 'failed':
                        f.write(f" (FAILED: {row['error']})")
                    f.write("\n")
                f.write("\n")
        
        print(f"📝 Text report saved to: {report_path}")


def main():
    """Main function for running ablation studies."""
    parser = argparse.ArgumentParser(description="Hypergraph Neural Network Ablation Study")
    parser.add_argument('--config', type=str, default='configs/hypergraph.yaml',
                        help='Base configuration file')
    parser.add_argument('--output-dir', type=str, default='experiments/ablation',
                        help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--sample', type=int, default=500,
                        help='Sample size for experiments')
    parser.add_argument('--studies', nargs='+', 
                        choices=['lambda', 'iterations', 'convergence', 'baseline', 'all'],
                        default=['all'],
                        help='Which ablation studies to run')
    
    args = parser.parse_args()
    
    # Initialize ablation study
    study = AblationStudy(args.config, args.output_dir)
    
    # Run selected studies
    if 'all' in args.studies:
        studies_to_run = ['lambda', 'iterations', 'convergence', 'baseline']
    else:
        studies_to_run = args.studies
    
    print(f"🚀 Starting Ablation Study with {len(studies_to_run)} study types...")
    
    try:
        if 'lambda' in studies_to_run:
            study.lambda_weight_ablation(args.epochs, args.sample)
        
        if 'iterations' in studies_to_run:
            study.iteration_ablation(args.epochs, args.sample)
        
        if 'convergence' in studies_to_run:
            study.convergence_threshold_ablation(args.epochs, args.sample)
        
        if 'baseline' in studies_to_run:
            study.baseline_comparison(args.epochs, args.sample)
        
        # Generate final analysis report
        study.generate_analysis_report()
        
        print("\n✅ Ablation study completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Ablation study failed: {e}")
        raise


if __name__ == "__main__":
    main()
