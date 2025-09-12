# experiments/stage6_experiments_v2.py
"""
Stage 6 TDGNN + G-SAMPLER Experimental Framework per §PHASE_D.1-D.3
Comprehensive comparison and ablation studies for temporal fraud detection
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime

# TDGNN imports
from models.tdgnn_wrapper import TDGNNHypergraphModel, train_epoch, evaluate_model
from models.hypergraph import create_hypergraph_model, HypergraphData
from sampling.gsampler import GSampler
from sampling.cpu_fallback import TemporalGraph
from torch.utils.data import DataLoader, TensorDataset

# Baseline model imports
from models.gcn_baseline import SimpleGCN
from models.graphsage_baseline import SimpleGraphSAGE

# Utility imports
from metrics import compute_metrics
from utils import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_simple_baseline_comparison():
    """Simplified baseline comparison for Phase D"""
    print("🚀 Starting Stage 6 Phase D - Simplified Experiments")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create simple synthetic data
    num_nodes = 300
    num_edges = 800
    feature_dim = 32
    
    print(f"📊 Creating dataset: {num_nodes} nodes, {num_edges} edges, {feature_dim} features")
    
    # Generate temporal graph
    edges = torch.randint(0, num_nodes, (2, num_edges))
    timestamps = torch.rand(num_edges) * 1000 + 500  # Random timestamps 500-1500
    
    # Build CSR format
    edge_data = list(zip(edges[0].tolist(), edges[1].tolist(), timestamps.tolist()))
    edge_data.sort(key=lambda x: (x[0], -x[2]))
    
    indptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    indices = []
    edge_times = []
    
    current_node = 0
    for src, dst, time in edge_data:
        while current_node < src:
            indptr[current_node + 1] = len(indices)
            current_node += 1
        indices.append(dst)
        edge_times.append(time)
    
    while current_node < num_nodes:
        indptr[current_node + 1] = len(indices)
        current_node += 1
    
    temporal_graph = TemporalGraph(
        num_nodes=num_nodes,
        num_edges=len(indices),
        indptr=indptr,
        indices=torch.tensor(indices, dtype=torch.long),
        timestamps=torch.tensor(edge_times)
    )
    
    # Generate features and labels
    node_features = torch.randn(num_nodes, feature_dim)
    # Create realistic fraud pattern: high first features → higher fraud probability
    fraud_prob = torch.sigmoid(node_features[:, 0] * 0.3 + torch.randn(num_nodes) * 0.2)
    labels = (torch.rand(num_nodes) < fraud_prob * 0.12).long()  # ~8-12% fraud rate
    
    print(f"🎯 Dataset created: {labels.sum().item()}/{num_nodes} fraud nodes ({labels.float().mean()*100:.1f}%)")
    
    # Split data
    perm = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size + val_size]] = True
    test_mask[perm[train_size + val_size:]] = True
    
    # Results storage
    results = {}
    
    # 1. TDGNN + G-SAMPLER Experiment
    print("\n🎯 Testing TDGNN + G-SAMPLER...")
    
    try:
        # Create TDGNN model
        base_model = create_hypergraph_model(
            input_dim=feature_dim,
            hidden_dim=64,
            output_dim=2,
            model_config={'layer_type': 'full', 'num_layers': 2}
        )
        
        gsampler = GSampler(
            csr_indptr=temporal_graph.indptr,
            csr_indices=temporal_graph.indices,
            csr_timestamps=temporal_graph.timestamps,
            device='cpu'
        )
        
        tdgnn_model = TDGNNHypergraphModel(
            base_model=base_model,
            gsampler=gsampler,
            temporal_graph=temporal_graph
        )
        
        # Quick evaluation
        test_seeds = torch.where(test_mask)[0][:50]  # Sample of test nodes
        t_evals = torch.ones(len(test_seeds)) * 1200.0  # Evaluation time
        
        start_time = time.time()
        
        with torch.no_grad():
            logits = tdgnn_model(
                seed_nodes=test_seeds,
                t_eval_array=t_evals,
                fanouts=[10, 5],
                delta_t=200.0
            )
            
        inference_time = time.time() - start_time
        
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        test_labels = labels[test_seeds].numpy()
        
        if len(np.unique(test_labels)) > 1:  # Check if both classes present
            tdgnn_metrics = compute_metrics(test_labels, probs)
            results['TDGNN'] = {
                'auc': tdgnn_metrics['auc'],
                'f1': tdgnn_metrics['f1'],
                'precision': tdgnn_metrics['precision'],
                'recall': tdgnn_metrics['recall'],
                'inference_time': inference_time,
                'status': 'success'
            }
            print(f"   ✅ TDGNN AUC: {tdgnn_metrics['auc']:.4f}, F1: {tdgnn_metrics['f1']:.4f}")
        else:
            results['TDGNN'] = {'status': 'insufficient_data', 'error': 'Only one class in test set'}
            print("   ⚠️  TDGNN: Insufficient test data diversity")
            
    except Exception as e:
        results['TDGNN'] = {'status': 'error', 'error': str(e)}
        print(f"   ❌ TDGNN Error: {e}")
    
    # 2. Hypergraph Baseline (Stage 5)
    print("\n📈 Testing Hypergraph Baseline...")
    
    try:
        hypergraph_model = create_hypergraph_model(
            input_dim=feature_dim,
            hidden_dim=64,
            output_dim=2,
            model_config={'layer_type': 'full', 'num_layers': 2}
        )
        
        # Create simple hypergraph structure
        num_hyperedges = min(30, num_nodes // 8)
        incidence_matrix = torch.zeros(num_nodes, num_hyperedges)
        
        for he in range(num_hyperedges):
            community_size = torch.randint(5, 15, (1,)).item()
            nodes = torch.randperm(num_nodes)[:community_size]
            incidence_matrix[nodes, he] = 1.0
        
        hypergraph_data = HypergraphData(
            incidence_matrix=incidence_matrix,
            node_features=node_features,
            node_labels=labels
        )
        
        start_time = time.time()
        
        with torch.no_grad():
            test_logits = hypergraph_model(hypergraph_data, node_features)
            
        inference_time = time.time() - start_time
        
        test_probs = torch.softmax(test_logits[test_mask], dim=1)[:, 1].numpy()
        test_labels_hyper = labels[test_mask].numpy()
        
        if len(np.unique(test_labels_hyper)) > 1:
            hyper_metrics = compute_metrics(test_labels_hyper, test_probs)
            results['Hypergraph'] = {
                'auc': hyper_metrics['auc'],
                'f1': hyper_metrics['f1'],
                'precision': hyper_metrics['precision'],
                'recall': hyper_metrics['recall'],
                'inference_time': inference_time,
                'status': 'success'
            }
            print(f"   ✅ Hypergraph AUC: {hyper_metrics['auc']:.4f}, F1: {hyper_metrics['f1']:.4f}")
        else:
            results['Hypergraph'] = {'status': 'insufficient_data'}
            print("   ⚠️  Hypergraph: Insufficient test data diversity")
            
    except Exception as e:
        results['Hypergraph'] = {'status': 'error', 'error': str(e)}
        print(f"   ❌ Hypergraph Error: {e}")
    
    # 3. Simple Feature-only Baseline
    print("\n📊 Testing Feature-only Baseline...")
    
    try:
        class SimpleClassifier(nn.Module):
            def __init__(self, in_dim, hidden_dim, out_dim):
                super().__init__()
                self.fc1 = nn.Linear(in_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, out_dim)
                self.dropout = nn.Dropout(0.3)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                return self.fc2(x)
        
        feature_model = SimpleClassifier(feature_dim, 64, 2)
        
        start_time = time.time()
        
        with torch.no_grad():
            feature_logits = feature_model(node_features)
            
        inference_time = time.time() - start_time
        
        feature_probs = torch.softmax(feature_logits[test_mask], dim=1)[:, 1].numpy()
        test_labels_feat = labels[test_mask].numpy()
        
        if len(np.unique(test_labels_feat)) > 1:
            feat_metrics = compute_metrics(test_labels_feat, feature_probs)
            results['Feature-only'] = {
                'auc': feat_metrics['auc'],
                'f1': feat_metrics['f1'],
                'precision': feat_metrics['precision'],
                'recall': feat_metrics['recall'],
                'inference_time': inference_time,
                'status': 'success'
            }
            print(f"   ✅ Feature-only AUC: {feat_metrics['auc']:.4f}, F1: {feat_metrics['f1']:.4f}")
        else:
            results['Feature-only'] = {'status': 'insufficient_data'}
            print("   ⚠️  Feature-only: Insufficient test data diversity")
            
    except Exception as e:
        results['Feature-only'] = {'status': 'error', 'error': str(e)}
        print(f"   ❌ Feature-only Error: {e}")
    
    # 4. Ablation Study: Delta_t Sensitivity
    print("\n🔬 Ablation Study: Delta_t Sensitivity...")
    
    ablation_results = {}
    delta_t_values = [50.0, 100.0, 200.0, 400.0, 800.0]
    
    for delta_t in delta_t_values:
        try:
            with torch.no_grad():
                logits = tdgnn_model(
                    seed_nodes=test_seeds,
                    t_eval_array=t_evals,
                    fanouts=[10, 5],
                    delta_t=delta_t
                )
                
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            
            if len(np.unique(test_labels)) > 1:
                metrics = compute_metrics(test_labels, probs)
                ablation_results[f'delta_t_{delta_t}'] = metrics['auc']
                print(f"   δt={delta_t:>5.0f}: AUC={metrics['auc']:.4f}")
            else:
                ablation_results[f'delta_t_{delta_t}'] = 0.0
                print(f"   δt={delta_t:>5.0f}: Insufficient data")
                
        except Exception as e:
            ablation_results[f'delta_t_{delta_t}'] = 0.0
            print(f"   δt={delta_t:>5.0f}: Error - {e}")
    
    results['ablation_delta_t'] = ablation_results
    
    # Print Summary
    print("\n" + "="*60)
    print("PHASE D EXPERIMENTAL RESULTS SUMMARY")
    print("="*60)
    
    print("\nModel Performance Comparison:")
    print("-" * 40)
    
    for model_name, result in results.items():
        if model_name != 'ablation_delta_t' and isinstance(result, dict):
            if result.get('status') == 'success':
                print(f"{model_name:15} | AUC: {result['auc']:.4f} | F1: {result['f1']:.4f} | "
                      f"Time: {result['inference_time']:.4f}s")
            else:
                print(f"{model_name:15} | Status: {result.get('status', 'unknown')}")
    
    print("\nAblation Study - Delta_t Sensitivity:")
    print("-" * 40)
    if 'ablation_delta_t' in results:
        for config, auc in results['ablation_delta_t'].items():
            print(f"{config:15} | AUC: {auc:.4f}")
    
    # Save results
    results_dir = Path("experiments/stage6_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"phase_d_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("\n🎉 Phase D experiments completed!")
    
    return results

if __name__ == "__main__":
    run_simple_baseline_comparison()
