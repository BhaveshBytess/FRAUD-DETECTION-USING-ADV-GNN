# demo_stage6_tdgnn.py
"""
Stage 6 TDGNN + G-SAMPLER Demonstration
Shows complete temporal graph neural network with efficient sampling
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
from models.tdgnn_wrapper import TDGNNHypergraphModel, train_epoch, evaluate_model
from models.hypergraph import create_hypergraph_model, HypergraphData
from sampling.gsampler import GSampler
from sampling.cpu_fallback import TemporalGraph
from torch.utils.data import DataLoader, TensorDataset

def create_demo_data():
    """Create demonstration temporal graph data"""
    print("🔧 Creating demonstration temporal graph...")
    
    # Create realistic temporal graph structure
    num_nodes = 100
    num_edges = 200
    feature_dim = 32
    
    # Generate temporal edges with realistic patterns
    edges = torch.randint(0, num_nodes, (2, num_edges))
    timestamps = torch.cumsum(torch.rand(num_edges) * 10, dim=0)  # Increasing timestamps
    
    # Build CSR adjacency structure
    indptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    indices = []
    edge_times = []
    
    # Sort edges by source node for CSR format
    edge_data = list(zip(edges[0].tolist(), edges[1].tolist(), timestamps.tolist()))
    edge_data.sort(key=lambda x: x[0])
    
    current_node = 0
    for src, dst, time in edge_data:
        while current_node < src:
            indptr[current_node + 1] = len(indices)
            current_node += 1
        
        indices.append(dst)
        edge_times.append(time)
    
    # Fill remaining indptr entries
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
    
    # Create hypergraph data for base model
    num_hyperedges = 20
    incidence_matrix = torch.zeros(num_nodes, num_hyperedges)
    
    # Create meaningful hyperedges (communities)
    for he in range(num_hyperedges):
        # Each hyperedge connects 5-10 nodes
        size = torch.randint(5, 11, (1,)).item()
        nodes = torch.randperm(num_nodes)[:size]
        incidence_matrix[nodes, he] = 1.0
    
    # Create node features and labels for fraud detection
    node_features = torch.randn(num_nodes, feature_dim)
    
    # Generate realistic labels (fraud detection scenario)
    fraud_ratio = 0.1  # 10% fraudulent
    num_fraud = int(num_nodes * fraud_ratio)
    labels = torch.zeros(num_nodes, dtype=torch.long)
    fraud_nodes = torch.randperm(num_nodes)[:num_fraud]
    labels[fraud_nodes] = 1
    
    hypergraph_data = HypergraphData(
        incidence_matrix=incidence_matrix,
        node_features=node_features,
        node_labels=labels
    )
    
    print(f"✅ Created temporal graph: {num_nodes} nodes, {len(indices)} edges")
    print(f"✅ Created hypergraph: {num_hyperedges} hyperedges, {fraud_ratio*100:.1f}% fraud")
    
    return temporal_graph, hypergraph_data, node_features, labels

def create_demo_model(temporal_graph, node_features):
    """Create TDGNN + G-SAMPLER model"""
    print("🔧 Creating TDGNN + G-SAMPLER model...")
    
    feature_dim = node_features.size(1)
    hidden_dim = 64
    
    # Create base hypergraph model (Stage 5)
    base_model = create_hypergraph_model(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=2,  # Binary fraud detection
        model_config={
            'layer_type': 'full',
            'num_layers': 3,
            'dropout': 0.2,
            'use_residual': True
        }
    )
    
    # Create G-SAMPLER (Stage 6)
    gsampler = GSampler(
        csr_indptr=temporal_graph.indptr,
        csr_indices=temporal_graph.indices,
        csr_timestamps=temporal_graph.timestamps,
        device='cpu'  # Use CPU for demo
    )
    
    # Create TDGNN wrapper (Stage 6)
    tdgnn_model = TDGNNHypergraphModel(
        base_model=base_model,
        gsampler=gsampler,
        temporal_graph=temporal_graph
    )
    
    print(f"✅ TDGNN model created with {sum(p.numel() for p in tdgnn_model.parameters())} parameters")
    
    return tdgnn_model

def demo_temporal_sampling(tdgnn_model, labels):
    """Demonstrate temporal sampling capabilities"""
    print("🔍 Demonstrating temporal sampling...")
    
    # Sample some nodes for demonstration
    num_samples = 10
    seed_nodes = torch.randint(0, len(labels), (num_samples,))
    
    # Different evaluation times
    current_time = 1000.0
    t_eval_array = torch.ones(num_samples) * current_time
    
    # Different sampling configurations
    configs = [
        {"fanouts": [5, 3], "delta_t": 100.0, "name": "Conservative"},
        {"fanouts": [15, 10], "delta_t": 200.0, "name": "Balanced"},
        {"fanouts": [25, 15], "delta_t": 500.0, "name": "Aggressive"}
    ]
    
    for config in configs:
        print(f"  📊 {config['name']} sampling (fanouts={config['fanouts']}, δt={config['delta_t']})...")
        
        with torch.no_grad():
            logits = tdgnn_model(
                seed_nodes=seed_nodes,
                t_eval_array=t_eval_array,
                fanouts=config['fanouts'],
                delta_t=config['delta_t']
            )
            
            predictions = torch.softmax(logits, dim=1)
            fraud_probs = predictions[:, 1]  # Fraud probability
            
            print(f"      📈 Avg fraud probability: {fraud_probs.mean():.3f}")
            print(f"      📈 Prediction entropy: {-(predictions * torch.log(predictions + 1e-8)).sum(1).mean():.3f}")
    
    print("✅ Temporal sampling demonstration complete")

def demo_training_loop(tdgnn_model, labels):
    """Demonstrate training loop"""
    print("🎯 Demonstrating TDGNN training...")
    
    # Create train/val split
    num_nodes = len(labels)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    perm = torch.randperm(num_nodes)
    train_size = int(0.7 * num_nodes)
    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size + int(0.2 * num_nodes)]] = True
    
    # Create data loaders
    train_seeds = torch.where(train_mask)[0]
    val_seeds = torch.where(val_mask)[0]
    
    current_time = 1000.0
    train_t_evals = torch.ones(len(train_seeds)) * current_time
    val_t_evals = torch.ones(len(val_seeds)) * current_time
    
    train_dataset = TensorDataset(train_seeds, train_t_evals, labels[train_seeds])
    val_dataset = TensorDataset(val_seeds, val_t_evals, labels[val_seeds])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Setup training
    optimizer = torch.optim.Adam(tdgnn_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    config = {
        'fanouts': [15, 10],
        'delta_t': 200.0
    }
    
    print(f"  📚 Training set: {len(train_seeds)} nodes")
    print(f"  📚 Validation set: {len(val_seeds)} nodes")
    
    # Run a few training epochs
    for epoch in range(5):
        # Training
        train_metrics = train_epoch(
            model=tdgnn_model,
            gsampler=tdgnn_model.gsampler,
            train_seed_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            cfg=config
        )
        
        # Validation
        val_metrics = evaluate_model(
            model=tdgnn_model,
            eval_loader=val_loader,
            criterion=criterion,
            cfg=config,
            split_name='val'
        )
        
        print(f"  📊 Epoch {epoch+1}: Train Loss={train_metrics['train_loss']:.4f}, "
              f"Val Acc={val_metrics['val_accuracy']:.4f}")
    
    print("✅ Training demonstration complete")

def main():
    """Run complete Stage 6 TDGNN demonstration"""
    print("🚀 Stage 6 TDGNN + G-SAMPLER Demonstration")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create demonstration data
    temporal_graph, hypergraph_data, node_features, labels = create_demo_data()
    
    # Create TDGNN model
    tdgnn_model = create_demo_model(temporal_graph, node_features)
    
    # Demonstrate temporal sampling
    demo_temporal_sampling(tdgnn_model, labels)
    
    # Demonstrate training
    demo_training_loop(tdgnn_model, labels)
    
    # Show model statistics
    print("\n📊 Final Model Statistics:")
    stats = tdgnn_model.get_sampling_stats()
    print(f"  🔧 Temporal graph nodes: {stats['temporal_graph_nodes']}")
    print(f"  🔧 Temporal graph edges: {stats['temporal_graph_edges']}")
    print(f"  🔧 G-SAMPLER memory: {stats['gsampler_memory']}")
    
    print("\n🎉 Stage 6 TDGNN + G-SAMPLER demonstration complete!")
    print("✅ All components working: temporal sampling ✓, hypergraph models ✓, training ✓")

if __name__ == "__main__":
    main()
