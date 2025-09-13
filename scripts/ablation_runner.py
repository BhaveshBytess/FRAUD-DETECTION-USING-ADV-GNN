#!/usr/bin/env python3
"""
hHGTN Ablation Study Runner - Stage 9 Integration

Systematic ablation studies to understand component contributions.
"""

import argparse
import logging
import os
import sys
import yaml
import itertools
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.train_hhgt import hHGTNTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AblationRunner:
    """Runner for systematic ablation studies."""
    
    def __init__(self, base_config_path: str, output_dir: str = "experiments/ablations"):
        self.base_config_path = Path(base_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base configuration
        with open(self.base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Component toggles for ablation
        self.component_flags = [
            'use_hypergraph',
            'use_hetero', 
            'use_memory',
            'use_cusp',
            'use_tdgnn',
            'use_gsampler',
            'use_spottarget',
            'use_robustness'
        ]
        
        self.results = []
        
        logger.info(f"Ablation runner initialized with {len(self.component_flags)} components")
    
    def run_single_ablation(self, experiment_name: str, component_config: Dict[str, bool]) -> Dict[str, Any]:
        """Run single ablation experiment."""
        
        logger.info(f"Running ablation: {experiment_name}")
        logger.info(f"Component config: {component_config}")
        
        # Create experiment config
        config = self.base_config.copy()
        
        # Update component toggles
        for component, enabled in component_config.items():
            config['model'][component] = enabled
        
        # Set experiment name and directory
        config['experiment']['name'] = experiment_name
        config['experiment']['save_dir'] = str(self.output_dir)
        
        # Force lite mode for ablations
        config['training']['mode'] = 'lite'
        config['training']['epochs'] = 5  # Short ablation runs
        config['training']['batch_size'] = 16
        
        try:
            # Initialize and run trainer
            trainer = hHGTNTrainer(config)
            
            # Mock graph schema
            node_types = {
                'transaction': 16,
                'address': 12,
                'user': 8
            }
            edge_types = {
                ('transaction', 'to', 'address'): 1,
                ('address', 'owns', 'user'): 1,
                ('user', 'makes', 'transaction'): 1
            }
            
            trainer.setup_model(node_types, edge_types)
            trainer.setup_data()
            
            # Run training
            trainer.train()
            
            # Extract final metrics
            final_metrics = {
                'experiment_name': experiment_name,
                'best_auc': trainer.best_metric,
                'components_enabled': sum(component_config.values()),
                'total_parameters': sum(p.numel() for p in trainer.model.parameters()),
                **component_config
            }
            
            logger.info(f"Ablation {experiment_name} completed. Best AUC: {trainer.best_metric:.4f}")
            return final_metrics
            
        except Exception as e:
            logger.error(f"Ablation {experiment_name} failed: {e}")
            return {
                'experiment_name': experiment_name,
                'best_auc': 0.0,
                'components_enabled': sum(component_config.values()),
                'total_parameters': 0,
                'error': str(e),
                **component_config
            }
    
    def run_component_ablation(self):
        """Run ablation study removing one component at a time."""
        
        logger.info("Starting component ablation study...")
        
        # Baseline: all components enabled
        baseline_config = {flag: True for flag in self.component_flags}
        baseline_result = self.run_single_ablation("baseline_all", baseline_config)
        self.results.append(baseline_result)
        
        # Ablate each component individually
        for component in self.component_flags:
            config = baseline_config.copy()
            config[component] = False  # Disable this component
            
            experiment_name = f"ablate_{component.replace('use_', '')}"
            result = self.run_single_ablation(experiment_name, config)
            self.results.append(result)
        
        # Minimal: only heterogeneous layers
        minimal_config = {flag: False for flag in self.component_flags}
        minimal_config['use_hetero'] = True  # Keep basic functionality
        minimal_result = self.run_single_ablation("minimal_hetero", minimal_config)
        self.results.append(minimal_result)
    
    def run_additive_ablation(self):
        """Run ablation study adding one component at a time."""
        
        logger.info("Starting additive ablation study...")
        
        # Start with minimal configuration
        base_config = {flag: False for flag in self.component_flags}
        base_config['use_hetero'] = True  # Always keep basic heterogeneous layers
        
        # Add components one by one
        for i, component in enumerate(self.component_flags):
            if component == 'use_hetero':
                continue  # Already included in base
                
            config = base_config.copy()
            
            # Enable this component and all previous ones
            for j in range(i + 1):
                if self.component_flags[j] != 'use_hetero':
                    config[self.component_flags[j]] = True
            
            experiment_name = f"additive_{i+1}_{component.replace('use_', '')}"
            result = self.run_single_ablation(experiment_name, config)
            self.results.append(result)
    
    def run_pairwise_ablation(self):
        """Run ablation study with pairs of components."""
        
        logger.info("Starting pairwise ablation study...")
        
        # Test important component pairs
        important_pairs = [
            ('use_hetero', 'use_memory'),       # Hetero + Temporal
            ('use_hetero', 'use_hypergraph'),   # Hetero + Hypergraph
            ('use_cusp', 'use_hypergraph'),     # CUSP + Hypergraph
            ('use_spottarget', 'use_robustness'), # Training discipline
            ('use_tdgnn', 'use_gsampler'),      # Sampling methods
        ]
        
        for pair in important_pairs:
            config = {flag: False for flag in self.component_flags}
            config[pair[0]] = True
            config[pair[1]] = True
            
            pair_name = f"pair_{pair[0].replace('use_', '')}_{pair[1].replace('use_', '')}"
            result = self.run_single_ablation(pair_name, config)
            self.results.append(result)
    
    def run_full_ablation_study(self):
        """Run comprehensive ablation study."""
        
        logger.info("Starting full ablation study...")
        
        # Run different ablation strategies
        self.run_component_ablation()
        self.run_additive_ablation()
        self.run_pairwise_ablation()
        
        # Save results
        self.save_results()
        self.analyze_results()
    
    def save_results(self):
        """Save ablation results to CSV and YAML."""
        
        # Save to CSV for easy analysis
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / 'ablation_results.csv'
        df.to_csv(csv_path, index=False)
        
        # Save to YAML for detailed inspection
        yaml_path = self.output_dir / 'ablation_results.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(self.results, f, default_flow_style=False)
        
        logger.info(f"Ablation results saved to {csv_path} and {yaml_path}")
    
    def analyze_results(self):
        """Analyze and summarize ablation results."""
        
        if not self.results:
            logger.warning("No results to analyze")
            return
        
        df = pd.DataFrame(self.results)
        
        # Remove error cases for analysis
        clean_df = df[df['best_auc'] > 0].copy()
        
        if len(clean_df) == 0:
            logger.warning("No successful experiments for analysis")
            return
        
        logger.info("\n" + "="*60)
        logger.info("ABLATION STUDY ANALYSIS")
        logger.info("="*60)
        
        # Find best and worst performers
        best_idx = clean_df['best_auc'].idxmax()
        worst_idx = clean_df['best_auc'].idxmin()
        
        best_experiment = clean_df.loc[best_idx]
        worst_experiment = clean_df.loc[worst_idx]
        
        logger.info(f"\n🏆 BEST PERFORMER:")
        logger.info(f"  Experiment: {best_experiment['experiment_name']}")
        logger.info(f"  AUC: {best_experiment['best_auc']:.4f}")
        logger.info(f"  Components: {best_experiment['components_enabled']}")
        logger.info(f"  Parameters: {best_experiment['total_parameters']:,}")
        
        logger.info(f"\n📉 WORST PERFORMER:")
        logger.info(f"  Experiment: {worst_experiment['experiment_name']}")
        logger.info(f"  AUC: {worst_experiment['best_auc']:.4f}")
        logger.info(f"  Components: {worst_experiment['components_enabled']}")
        
        # Component importance analysis
        logger.info(f"\n🧪 COMPONENT IMPORTANCE:")
        
        # Find baseline performance
        baseline_rows = clean_df[clean_df['experiment_name'] == 'baseline_all']
        if len(baseline_rows) > 0:
            baseline_auc = baseline_rows['best_auc'].iloc[0]
            
            # Calculate drop for each component ablation
            component_importance = {}
            for component in self.component_flags:
                ablation_name = f"ablate_{component.replace('use_', '')}"
                ablation_rows = clean_df[clean_df['experiment_name'] == ablation_name]
                
                if len(ablation_rows) > 0:
                    ablation_auc = ablation_rows['best_auc'].iloc[0]
                    importance = baseline_auc - ablation_auc
                    component_importance[component] = importance
            
            # Sort by importance
            sorted_components = sorted(component_importance.items(), key=lambda x: x[1], reverse=True)
            
            for component, importance in sorted_components:
                component_name = component.replace('use_', '').upper()
                logger.info(f"  {component_name:12s}: {importance:+.4f} AUC drop when removed")
        
        # Performance vs complexity analysis
        logger.info(f"\n⚖️ PERFORMANCE vs COMPLEXITY:")
        
        # Calculate efficiency score (AUC / log(parameters))
        import numpy as np
        clean_df['efficiency'] = clean_df['best_auc'] / np.log10(clean_df['total_parameters'] + 1)
        clean_df_sorted = clean_df.sort_values('efficiency', ascending=False)
        
        logger.info("  Top 3 most efficient configurations:")
        for i in range(min(3, len(clean_df_sorted))):
            row = clean_df_sorted.iloc[i]
            logger.info(f"    {i+1}. {row['experiment_name']:20s} - AUC: {row['best_auc']:.4f}, "
                       f"Params: {row['total_parameters']:6,d}, Efficiency: {row['efficiency']:.4f}")
        
        # Summary statistics
        logger.info(f"\n📊 SUMMARY STATISTICS:")
        logger.info(f"  Total experiments: {len(clean_df)}")
        logger.info(f"  Mean AUC: {clean_df['best_auc'].mean():.4f} ± {clean_df['best_auc'].std():.4f}")
        logger.info(f"  AUC range: {clean_df['best_auc'].min():.4f} - {clean_df['best_auc'].max():.4f}")
        logger.info(f"  Parameter range: {clean_df['total_parameters'].min():,} - {clean_df['total_parameters'].max():,}")
        
        logger.info("="*60)
        
        # Save analysis summary
        analysis_path = self.output_dir / 'ablation_analysis.txt'
        with open(analysis_path, 'w') as f:
            f.write("hHGTN Ablation Study Analysis Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Best Performer: {best_experiment['experiment_name']} (AUC: {best_experiment['best_auc']:.4f})\n")
            f.write(f"Worst Performer: {worst_experiment['experiment_name']} (AUC: {worst_experiment['best_auc']:.4f})\n\n")
            f.write("Component Importance (AUC drop when removed):\n")
            for component, importance in sorted_components:
                component_name = component.replace('use_', '')
                f.write(f"  {component_name}: {importance:+.4f}\n")
        
        logger.info(f"Analysis summary saved to {analysis_path}")


def main():
    """Main ablation runner entry point."""
    
    parser = argparse.ArgumentParser(description='Run hHGTN ablation studies')
    parser.add_argument('--config', type=str, default='configs/stage9.yaml', 
                       help='Base configuration file path')
    parser.add_argument('--output-dir', type=str, default='experiments/ablations',
                       help='Output directory for ablation results')
    parser.add_argument('--study-type', type=str, choices=['component', 'additive', 'pairwise', 'full'],
                       default='full', help='Type of ablation study to run')
    
    args = parser.parse_args()
    
    # Verify config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    # Initialize ablation runner
    runner = AblationRunner(str(config_path), args.output_dir)
    
    # Run specified study type
    if args.study_type == 'component':
        runner.run_component_ablation()
    elif args.study_type == 'additive':
        runner.run_additive_ablation()
    elif args.study_type == 'pairwise':
        runner.run_pairwise_ablation()
    else:  # full
        runner.run_full_ablation_study()
    
    logger.info("Ablation study completed!")


if __name__ == "__main__":
    main()
