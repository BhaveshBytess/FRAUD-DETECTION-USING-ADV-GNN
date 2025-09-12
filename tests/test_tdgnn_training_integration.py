# tests/test_tdgnn_training_integration.py
"""
Test suite for Phase C: TDGNN + G-SAMPLER training integration per §PHASE_C.5
Validates complete training pipeline with temporal sampling and hypergraph models
"""

import pytest
import torch
import tempfile
import os
import yaml
from pathlib import Path

# Import TDGNN components
from models.tdgnn_wrapper import TDGNNHypergraphModel, train_epoch, evaluate_model
from models.hypergraph import create_hypergraph_model, HypergraphData
from sampling.gsampler import GSampler
from sampling.temporal_data_loader import TemporalGraphDataLoader
from sampling.cpu_fallback import TemporalGraph
from torch.utils.data import DataLoader, TensorDataset

class TestTDGNNTrainingIntegration:
    """Test TDGNN training integration per §PHASE_C.5"""
    
    @pytest.fixture
    def small_temporal_graph(self):
        """Create small temporal graph for testing"""
        # Create simple temporal graph
        num_nodes = 20
        num_edges = 40
        
        # Generate edges with timestamps
        edges = torch.randint(0, num_nodes, (2, num_edges))
        timestamps = torch.rand(num_edges) * 1000  # Random timestamps 0-1000
        
        # Create adjacency lists
        indptr = torch.zeros(num_nodes + 1, dtype=torch.long)
        indices = []
        edge_times = []
        
        for i in range(num_nodes):
            neighbors = []
            times = []
            for j in range(num_edges):
                if edges[0, j] == i:
                    neighbors.append(edges[1, j].item())
                    times.append(timestamps[j].item())
            
            indptr[i + 1] = indptr[i] + len(neighbors)
            indices.extend(neighbors)
            edge_times.extend(times)
        
        temporal_graph = TemporalGraph(
            num_nodes=num_nodes,
            num_edges=num_edges,
            indptr=torch.tensor(indptr),
            indices=torch.tensor(indices, dtype=torch.long),
            timestamps=torch.tensor(edge_times),
            max_time=1000.0
        )
        
        return temporal_graph
    
    @pytest.fixture
    def small_hypergraph_data(self):
        """Create small hypergraph data for testing"""
        num_nodes = 20
        num_hyperedges = 10
        feature_dim = 16
        
        # Create incidence matrix
        incidence_matrix = torch.zeros(num_nodes, num_hyperedges)
        for he in range(num_hyperedges):
            # Each hyperedge connects 3-5 nodes
            size = torch.randint(3, 6, (1,)).item()
            nodes = torch.randperm(num_nodes)[:size]
            incidence_matrix[nodes, he] = 1.0
        
        # Create node features and labels
        node_features = torch.randn(num_nodes, feature_dim)
        labels = torch.randint(0, 2, (num_nodes,))
        
        hypergraph_data = HypergraphData(
            incidence_matrix=incidence_matrix,
            node_features=node_features,
            hyperedge_features=None,
            node_labels=labels
        )
        
        return hypergraph_data, node_features, labels
    
    @pytest.fixture  
    def tdgnn_model_setup(self, small_temporal_graph, small_hypergraph_data):
        """Setup TDGNN model for testing"""
        hypergraph_data, node_features, labels = small_hypergraph_data
        
        # Create base hypergraph model
        base_model = create_hypergraph_model(
            input_dim=node_features.size(1),
            hidden_dim=32,
            output_dim=2,
            model_config={'layer_type': 'simple', 'num_layers': 2}
        )
        
        # Create G-SAMPLER  
        gsampler = GSampler(
            temporal_graph=small_temporal_graph,
            use_gpu=False,  # Use CPU for testing
            device=torch.device('cpu')
        )
        
        # Create TDGNN wrapper
        tdgnn_model = TDGNNHypergraphModel(
            base_model=base_model,
            gsampler=gsampler,
            temporal_graph=small_temporal_graph
        )
        
        return tdgnn_model, hypergraph_data, node_features, labels
    
    def test_tdgnn_forward_pass(self, tdgnn_model_setup):
        """Test TDGNN forward pass per §PHASE_C.1"""
        tdgnn_model, hypergraph_data, node_features, labels = tdgnn_model_setup
        
        # Test forward pass
        batch_size = 5
        seed_nodes = torch.randint(0, node_features.size(0), (batch_size,))
        t_eval_array = torch.ones(batch_size) * 800.0  # Evaluation time
        fanouts = [5, 3]
        delta_t = 100.0
        
        # Forward pass
        logits = tdgnn_model(
            seed_nodes=seed_nodes,
            t_eval_array=t_eval_array,
            fanouts=fanouts,
            delta_t=delta_t
        )
        
        # Validate output shape
        assert logits.shape == (batch_size, 2), f"Expected shape ({batch_size}, 2), got {logits.shape}"
        assert torch.isfinite(logits).all(), "Forward pass produced non-finite values"
        
        print("✓ TDGNN forward pass test passed")
    
    def test_tdgnn_backward_pass(self, tdgnn_model_setup):
        """Test TDGNN backward pass per §PHASE_C.2"""
        tdgnn_model, hypergraph_data, node_features, labels = tdgnn_model_setup
        
        # Setup for backward pass
        optimizer = torch.optim.Adam(tdgnn_model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        batch_size = 5
        seed_nodes = torch.randint(0, node_features.size(0), (batch_size,))
        t_eval_array = torch.ones(batch_size) * 800.0
        batch_labels = torch.randint(0, 2, (batch_size,))
        fanouts = [5, 3]
        delta_t = 100.0
        
        # Forward pass
        logits = tdgnn_model(
            seed_nodes=seed_nodes,
            t_eval_array=t_eval_array,
            fanouts=fanouts,
            delta_t=delta_t
        )
        
        # Backward pass
        loss = criterion(logits, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check gradients were computed
        has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 
                          for p in tdgnn_model.parameters())
        assert has_gradients, "No gradients were computed during backward pass"
        
        print("✓ TDGNN backward pass test passed")
    
    def test_training_epoch_integration(self, tdgnn_model_setup):
        """Test complete training epoch per §PHASE_C.2"""
        tdgnn_model, hypergraph_data, node_features, labels = tdgnn_model_setup
        
        # Create data loader
        num_samples = 15
        seed_nodes = torch.randint(0, node_features.size(0), (num_samples,))
        t_evals = torch.ones(num_samples) * 800.0
        batch_labels = torch.randint(0, 2, (num_samples,))
        
        dataset = TensorDataset(seed_nodes, t_evals, batch_labels)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
        
        # Setup training components
        optimizer = torch.optim.Adam(tdgnn_model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        cfg = {
            'fanouts': [5, 3],
            'delta_t': 100.0
        }
        
        # Run training epoch
        train_metrics = train_epoch(
            model=tdgnn_model,
            gsampler=tdgnn_model.gsampler,
            train_seed_loader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            cfg=cfg
        )
        
        # Validate training metrics
        assert 'train_loss' in train_metrics, "Training metrics missing 'train_loss'"
        assert 'num_batches' in train_metrics, "Training metrics missing 'num_batches'"
        assert train_metrics['num_batches'] > 0, "No batches were processed"
        assert torch.isfinite(torch.tensor(train_metrics['train_loss'])), "Training loss is not finite"
        
        print(f"✓ Training epoch test passed - Loss: {train_metrics['train_loss']:.4f}")
    
    def test_evaluation_integration(self, tdgnn_model_setup):
        """Test model evaluation per §PHASE_C.2"""
        tdgnn_model, hypergraph_data, node_features, labels = tdgnn_model_setup
        
        # Create evaluation data
        num_samples = 10
        seed_nodes = torch.randint(0, node_features.size(0), (num_samples,))
        t_evals = torch.ones(num_samples) * 800.0
        batch_labels = torch.randint(0, 2, (num_samples,))
        
        dataset = TensorDataset(seed_nodes, t_evals, batch_labels)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
        
        criterion = torch.nn.CrossEntropyLoss()
        cfg = {
            'fanouts': [5, 3],
            'delta_t': 100.0
        }
        
        # Run evaluation
        eval_metrics = evaluate_model(
            model=tdgnn_model,
            eval_loader=dataloader,
            criterion=criterion,
            cfg=cfg,
            split_name='test'
        )
        
        # Validate evaluation metrics
        assert 'test_loss' in eval_metrics, "Evaluation metrics missing 'test_loss'"
        assert 'test_accuracy' in eval_metrics, "Evaluation metrics missing 'test_accuracy'"
        assert 'test_num_samples' in eval_metrics, "Evaluation metrics missing 'test_num_samples'"
        
        assert eval_metrics['test_num_samples'] == num_samples, "Sample count mismatch"
        assert 0 <= eval_metrics['test_accuracy'] <= 1, "Accuracy out of valid range"
        
        print(f"✓ Evaluation test passed - Accuracy: {eval_metrics['test_accuracy']:.4f}")
    
    def test_empty_subgraph_handling(self, tdgnn_model_setup):
        """Test handling of empty subgraphs per §PHASE_C.1"""
        tdgnn_model, hypergraph_data, node_features, labels = tdgnn_model_setup
        
        # Create restrictive sampling that will likely produce empty subgraphs
        batch_size = 3
        seed_nodes = torch.randint(0, node_features.size(0), (batch_size,))
        t_eval_array = torch.zeros(batch_size)  # Very early time, likely empty neighborhoods
        fanouts = [1, 1]  # Small fanouts
        delta_t = 1.0  # Very restrictive time window
        
        # Forward pass should handle empty subgraphs gracefully
        logits = tdgnn_model(
            seed_nodes=seed_nodes,
            t_eval_array=t_eval_array,
            fanouts=fanouts,
            delta_t=delta_t
        )
        
        # Should return zero logits for empty subgraphs
        assert logits.shape == (batch_size, 2), f"Expected shape ({batch_size}, 2), got {logits.shape}"
        assert torch.isfinite(logits).all(), "Empty subgraph handling produced non-finite values"
        
        print("✓ Empty subgraph handling test passed")
    
    def test_sampling_statistics(self, tdgnn_model_setup):
        """Test sampling statistics per §PHASE_C.3"""
        tdgnn_model, hypergraph_data, node_features, labels = tdgnn_model_setup
        
        # Get sampling statistics
        stats = tdgnn_model.get_sampling_stats()
        
        # Validate statistics structure
        assert isinstance(stats, dict), "Sampling stats should be a dictionary"
        assert 'gsampler_memory' in stats, "Missing G-SAMPLER memory stats"
        assert 'temporal_graph_nodes' in stats, "Missing temporal graph node count"
        assert 'temporal_graph_edges' in stats, "Missing temporal graph edge count"
        
        # Validate values
        assert stats['temporal_graph_nodes'] > 0, "Temporal graph should have nodes"
        assert stats['temporal_graph_edges'] > 0, "Temporal graph should have edges"
        
        print(f"✓ Sampling statistics test passed - {stats}")
    
    def test_config_integration(self):
        """Test configuration loading per §PHASE_C.3"""
        
        # Test config structure
        config_path = Path(__file__).parent.parent / "configs" / "stage6_tdgnn.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required config sections
            required_sections = [
                'fanouts', 'delta_t', 'sampling_strategy',
                'use_gpu_sampling', 'batch_size', 'epochs',
                'layer_type', 'num_layers', 'hidden_dim'
            ]
            
            for section in required_sections:
                assert section in config, f"Missing required config section: {section}"
            
            # Validate config values
            assert isinstance(config['fanouts'], list), "fanouts should be a list"
            assert len(config['fanouts']) > 0, "fanouts should not be empty"
            assert config['delta_t'] > 0, "delta_t should be positive"
            assert config['batch_size'] > 0, "batch_size should be positive"
            assert config['epochs'] > 0, "epochs should be positive"
            
            print("✓ Configuration integration test passed")
        else:
            print("⚠ Configuration file not found, skipping config test")
    
    def test_gpu_cpu_consistency(self, small_temporal_graph, small_hypergraph_data):
        """Test GPU-CPU consistency for TDGNN per §PHASE_C.5"""
        hypergraph_data, node_features, labels = small_hypergraph_data
        
        # Test both GPU and CPU if CUDA available
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
        
        results = {}
        
        for device in devices:
            # Create model on device
            base_model = create_hypergraph_model(
                input_dim=node_features.size(1),
                hidden_dim=32,
                output_dim=2,
                model_config={'layer_type': 'simple', 'num_layers': 2}
            ).to(device)
            
            gsampler = GSampler(
                temporal_graph=small_temporal_graph,
                use_gpu=(device.type == 'cuda'),
                device=device
            )
            
            tdgnn_model = TDGNNHypergraphModel(
                base_model=base_model,
                gsampler=gsampler,
                temporal_graph=small_temporal_graph
            ).to(device)
            
            # Run forward pass
            batch_size = 3
            seed_nodes = torch.randint(0, node_features.size(0), (batch_size,)).to(device)
            t_eval_array = torch.ones(batch_size).to(device) * 800.0
            fanouts = [5, 3]
            delta_t = 100.0
            
            with torch.no_grad():
                logits = tdgnn_model(
                    seed_nodes=seed_nodes,
                    t_eval_array=t_eval_array,
                    fanouts=fanouts,
                    delta_t=delta_t
                )
            
            results[device.type] = logits.cpu()
        
        # Compare results if both devices tested\n        if len(results) > 1:\n            cpu_result = results['cpu']\n            gpu_result = results['cuda']\n            \n            # Check shapes match\n            assert cpu_result.shape == gpu_result.shape, \"GPU-CPU shape mismatch\"\n            \n            # Check for approximate equality (allowing for numerical differences)\n            max_diff = torch.abs(cpu_result - gpu_result).max().item()\n            assert max_diff < 1e-3, f\"GPU-CPU difference too large: {max_diff}\"\n            \n            print(f\"✓ GPU-CPU consistency test passed - Max diff: {max_diff:.6f}\")\n        else:\n            print(\"✓ Single device test passed\")\n\ndef run_integration_tests():\n    \"\"\"Run all Phase C integration tests\"\"\"\n    import sys\n    \n    # Configure test environment\n    torch.manual_seed(42)\n    if torch.cuda.is_available():\n        torch.cuda.manual_seed(42)\n    \n    test_class = TestTDGNNTrainingIntegration()\n    \n    # Create fixtures\n    temporal_graph = test_class.small_temporal_graph()\n    hypergraph_data = test_class.small_hypergraph_data()\n    model_setup = test_class.tdgnn_model_setup(temporal_graph, hypergraph_data)\n    \n    tests = [\n        (\"Forward Pass\", lambda: test_class.test_tdgnn_forward_pass(model_setup)),\n        (\"Backward Pass\", lambda: test_class.test_tdgnn_backward_pass(model_setup)),\n        (\"Training Epoch\", lambda: test_class.test_training_epoch_integration(model_setup)),\n        (\"Evaluation\", lambda: test_class.test_evaluation_integration(model_setup)),\n        (\"Empty Subgraph\", lambda: test_class.test_empty_subgraph_handling(model_setup)),\n        (\"Sampling Stats\", lambda: test_class.test_sampling_statistics(model_setup)),\n        (\"Configuration\", lambda: test_class.test_config_integration()),\n        (\"GPU-CPU Consistency\", lambda: test_class.test_gpu_cpu_consistency(temporal_graph, hypergraph_data)),\n    ]\n    \n    passed = 0\n    failed = 0\n    \n    print(\"\\n=== Phase C Integration Tests ===\")\n    \n    for test_name, test_func in tests:\n        try:\n            print(f\"\\nRunning {test_name} test...\")\n            test_func()\n            passed += 1\n        except Exception as e:\n            print(f\"✗ {test_name} test failed: {e}\")\n            failed += 1\n    \n    print(f\"\\n=== Results: {passed} passed, {failed} failed ===\")\n    \n    if failed > 0:\n        print(\"Some tests failed. Please check the implementation.\")\n        sys.exit(1)\n    else:\n        print(\"All Phase C integration tests passed! ✓\")\n        sys.exit(0)\n\nif __name__ == \"__main__\":\n    run_integration_tests()
