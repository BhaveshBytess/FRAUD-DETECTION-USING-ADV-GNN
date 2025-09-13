#!/usr/bin/env python3
"""
Ablation Analysis Script for Stage 12
Analyzes systematic ablation results and computes effect sizes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import ast
from scipy import stats

def parse_config(config_str):
    """Parse configuration string to dictionary."""
    try:
        return ast.literal_eval(config_str)
    except:
        return {}

def compute_effect_sizes(df, baseline_config):
    """Compute effect sizes (Cohen's d) for each component ablation."""
    baseline_mask = df['config'] == str(baseline_config)
    baseline_metrics = df[baseline_mask]
    
    if len(baseline_metrics) == 0:
        print("Warning: No baseline configuration found")
        return {}
    
    baseline_auc = baseline_metrics['auc'].mean()
    baseline_f1 = baseline_metrics['f1'].mean()
    baseline_auc_std = baseline_metrics['auc'].std()
    baseline_f1_std = baseline_metrics['f1'].std()
    
    effects = {}
    
    # Analyze single component ablations
    components = ['use_cusp', 'use_hypergraph', 'use_memory', 'use_spottarget', 'use_gsampler']
    
    for component in components:
        # Find configurations with only this component ablated
        component_off_configs = []
        for _, row in df.iterrows():
            config = parse_config(row['config'])
            if config.get(component) == False:  # Component is turned off
                # Check if all other components match baseline
                other_components_match = True
                for other_comp in components:
                    if other_comp != component:
                        if config.get(other_comp) != baseline_config.get(other_comp):
                            other_components_match = False
                            break
                
                if other_components_match and config.get('use_rg_defense') == baseline_config.get('use_rg_defense'):
                    component_off_configs.append(row)
        
        if component_off_configs:
            component_df = pd.DataFrame(component_off_configs)
            component_auc = component_df['auc'].mean()
            component_f1 = component_df['f1'].mean()
            component_auc_std = component_df['auc'].std()
            component_f1_std = component_df['f1'].std()
            
            # Cohen's d = (mean1 - mean2) / pooled_std
            pooled_auc_std = np.sqrt((baseline_auc_std**2 + component_auc_std**2) / 2)
            pooled_f1_std = np.sqrt((baseline_f1_std**2 + component_f1_std**2) / 2)
            
            auc_effect = (baseline_auc - component_auc) / pooled_auc_std if pooled_auc_std > 0 else 0
            f1_effect = (baseline_f1 - component_f1) / pooled_f1_std if pooled_f1_std > 0 else 0
            
            effects[component] = {
                'auc_effect': auc_effect,
                'f1_effect': f1_effect,
                'auc_delta': baseline_auc - component_auc,
                'f1_delta': baseline_f1 - component_f1,
                'n_runs': len(component_df)
            }
    
    return effects, {'baseline_auc': baseline_auc, 'baseline_f1': baseline_f1}

def analyze_defense_mechanisms(df):
    """Analyze the effect of different defense mechanisms."""
    defense_results = {}
    
    for defense in ['none', 'dropedge', 'spectral']:
        defense_runs = df[df['config'].str.contains(f"'use_rg_defense': '{defense}'")]
        if len(defense_runs) > 0:
            defense_results[defense] = {
                'auc_mean': defense_runs['auc'].mean(),
                'auc_std': defense_runs['auc'].std(),
                'f1_mean': defense_runs['f1'].mean(),
                'f1_std': defense_runs['f1'].std(),
                'n_runs': len(defense_runs)
            }
    
    return defense_results

def create_summary_plots(df, effects, baseline_metrics, defense_results, output_dir):
    """Create summary visualizations."""
    plt.style.use('default')
    
    # Plot 1: Component Effect Sizes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    components = list(effects.keys())
    auc_effects = [effects[comp]['auc_effect'] for comp in components]
    f1_effects = [effects[comp]['f1_effect'] for comp in components]
    
    x = np.arange(len(components))
    
    ax1.bar(x, auc_effects, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Components')
    ax1.set_ylabel('Effect Size (Cohen\'s d)')
    ax1.set_title('AUC Effect Sizes for Component Ablations')
    ax1.set_xticks(x)
    ax1.set_xticklabels([comp.replace('use_', '') for comp in components], rotation=45)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(x, f1_effects, alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Components')
    ax2.set_ylabel('Effect Size (Cohen\'s d)')
    ax2.set_title('F1 Effect Sizes for Component Ablations')
    ax2.set_xticks(x)
    ax2.set_xticklabels([comp.replace('use_', '') for comp in components], rotation=45)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'component_effects.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Defense Mechanisms Comparison
    if defense_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        defenses = list(defense_results.keys())
        auc_means = [defense_results[d]['auc_mean'] for d in defenses]
        auc_stds = [defense_results[d]['auc_std'] for d in defenses]
        f1_means = [defense_results[d]['f1_mean'] for d in defenses]
        f1_stds = [defense_results[d]['f1_std'] for d in defenses]
        
        ax1.bar(defenses, auc_means, yerr=auc_stds, alpha=0.7, capsize=5, color='lightgreen')
        ax1.set_ylabel('AUC')
        ax1.set_title('Defense Mechanisms - AUC Performance')
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(defenses, f1_means, yerr=f1_stds, alpha=0.7, capsize=5, color='orange')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Defense Mechanisms - F1 Performance')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'defense_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Runtime vs Performance
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    scatter = ax.scatter(df['runtime_s'], df['auc'], c=df['f1'], cmap='viridis', alpha=0.6, s=60)
    ax.set_xlabel('Runtime (seconds)')
    ax.set_ylabel('AUC')
    ax.set_title('Runtime vs Performance (color = F1 score)')
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, label='F1 Score')
    plt.savefig(output_dir / 'runtime_vs_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Analysis] Plots saved to {output_dir}")

def generate_latex_table(effects, baseline_metrics, defense_results):
    """Generate LaTeX-ready table."""
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{hHGTN Component Ablation Study Results}",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Component & AUC Effect & F1 Effect & AUC $\\Delta$ & F1 $\\Delta$ & Runs \\\\",
        "\\hline"
    ]
    
    # Add baseline
    latex_lines.append(f"Baseline (All) & - & - & {baseline_metrics['baseline_auc']:.3f} & {baseline_metrics['baseline_f1']:.3f} & - \\\\")
    
    # Add component ablations
    for comp, data in effects.items():
        comp_name = comp.replace('use_', '').replace('_', ' ').title()
        latex_lines.append(
            f"{comp_name} & {data['auc_effect']:.3f} & {data['f1_effect']:.3f} & "
            f"{data['auc_delta']:.3f} & {data['f1_delta']:.3f} & {data['n_runs']} \\\\"
        )
    
    latex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_lines)

def main():
    """Main analysis function."""
    # Load results
    results_path = Path('results/ablation_table.csv')
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        return
    
    df = pd.read_csv(results_path)
    print(f"[Analysis] Loaded {len(df)} ablation results")
    
    # Define baseline configuration
    baseline_config = {
        'use_cusp': True,
        'use_hypergraph': True,
        'use_memory': True,
        'use_spottarget': True,
        'use_gsampler': True,
        'use_rg_defense': 'none'
    }
    
    # Compute effect sizes
    effects, baseline_metrics = compute_effect_sizes(df, baseline_config)
    
    # Analyze defense mechanisms
    defense_results = analyze_defense_mechanisms(df)
    
    # Create output directory
    output_dir = Path('../ablation_summary')
    output_dir.mkdir(exist_ok=True)
    
    # Generate summary report
    summary = {
        'baseline_metrics': baseline_metrics,
        'component_effects': effects,
        'defense_results': defense_results,
        'total_runs': len(df),
        'successful_runs': len(df[df['error'].isna()]) if 'error' in df.columns else len(df)
    }
    
    # Save summary JSON
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(effects, baseline_metrics, defense_results)
    with open(output_dir / 'ablation_table.tex', 'w') as f:
        f.write(latex_table)
    
    # Create plots
    create_summary_plots(df, effects, baseline_metrics, defense_results, output_dir)
    
    # Print summary
    print("\n=== Ablation Study Summary ===")
    print(f"Baseline AUC: {baseline_metrics['baseline_auc']:.4f}")
    print(f"Baseline F1: {baseline_metrics['baseline_f1']:.4f}")
    print("\nComponent Effects (Effect Size):")
    for comp, data in effects.items():
        impact = "Negative" if data['auc_effect'] > 0 else "Positive" if data['auc_effect'] < 0 else "Neutral"
        print(f"  {comp}: AUC Δ={data['auc_delta']:.4f} (d={data['auc_effect']:.3f}) - {impact}")
    
    print("\nDefense Mechanisms:")
    for defense, data in defense_results.items():
        print(f"  {defense}: AUC={data['auc_mean']:.4f}±{data['auc_std']:.4f}, F1={data['f1_mean']:.4f}±{data['f1_std']:.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"LaTeX table: {output_dir}/ablation_table.tex")

if __name__ == "__main__":
    main()
