"""
Unit tests for 4DBInfer hHGTN adapter
Stage 11: Systematic Benchmarking Integration
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add the adapter to path (look in parent directory)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from hhgt_adapter import (
    HHGTSolutionConfig, 
    HeteroHHGT, 
    HHGT, 
    HHGTSolution,
    DGLToHypergraphAdapter
)

class TestHHGTAdapter(unittest.TestCase):
    """Test suite for hHGTN 4DBInfer adapter"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = HHGTSolutionConfig(
            hid_size=128,
            batch_size=32,
            num_layers=2,
            dropout=0.1,
            spot_target_enabled=True,
            cusp_enabled=True,
            trd_enabled=True,
            memory_enabled=True
        )
        
        self.node_in_size_dict = {"user": 64, "item": 64, "transaction": 32}
        self.edge_in_size_dict = {("user", "interacts", "item"): 16}
        self.out_size = 256
        
    def test_hhgt_solution_config(self):
        """Test HHGTSolutionConfig initialization and validation"""
        config = HHGTSolutionConfig()
        
        # Test default values
        self.assertEqual(config.hid_size, 256)
        self.assertEqual(config.batch_size, 256)
        self.assertTrue(config.spot_target_enabled)
        self.assertTrue(config.cusp_enabled)
        self.assertTrue(config.trd_enabled)
        self.assertTrue(config.memory_enabled)
        
        # Test custom values
        custom_config = HHGTSolutionConfig(
            hid_size=512,
            spot_target_enabled=False,
            cusp_enabled=False
        )
        self.assertEqual(custom_config.hid_size, 512)
        self.assertFalse(custom_config.spot_target_enabled)
        self.assertFalse(custom_config.cusp_enabled)
        
    def test_dgl_to_hypergraph_adapter(self):
        """Test DGL to hypergraph conversion adapter"""
        adapter = DGLToHypergraphAdapter(self.config)
        
        # Test fallback hypergraph data creation
        hypergraph_data = adapter._create_fallback_hypergraph_data()
        
        self.assertIn('node_features', hypergraph_data)
        self.assertIn('hyperedge_features', hypergraph_data)
        self.assertIn('node_hyperedge_adj', hypergraph_data)
        
        # Check shapes
        expected_batch_size = self.config.batch_size
        expected_num_hyperedges = self.config.num_hyperedges
        expected_feature_dim = self.config.hid_size
        
        self.assertEqual(hypergraph_data['node_features'].shape[0], expected_batch_size)
        self.assertEqual(hypergraph_data['hyperedge_features'].shape[1], expected_num_hyperedges)
        self.assertEqual(hypergraph_data['node_features'].shape[2], expected_feature_dim)
        
    def test_hetero_hhgt_initialization(self):
        """Test HeteroHHGT model initialization"""
        model = HeteroHHGT(
            graph_config=None,  # Mock graph config
            solution_config=self.config,
            node_in_size_dict=self.node_in_size_dict,
            edge_in_size_dict=self.edge_in_size_dict,
            out_size=self.out_size,
            num_layers=self.config.num_layers
        )
        
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.out_size, self.out_size)
        self.assertEqual(model.num_layers, self.config.num_layers)
        
        # Test ablation components are initialized correctly
        self.assertIsNotNone(model.spot_target)  # Should be initialized
        self.assertIsNotNone(model.cusp_embeddings)
        self.assertIsNotNone(model.trd_sampler)
        self.assertIsNotNone(model.memory_module)
        
    def test_hetero_hhgt_forward_pass(self):
        """Test HeteroHHGT forward pass with synthetic data"""
        model = HeteroHHGT(
            graph_config=None,
            solution_config=self.config,
            node_in_size_dict=self.node_in_size_dict,
            edge_in_size_dict=self.edge_in_size_dict,
            out_size=self.out_size,
            num_layers=self.config.num_layers
        )
        
        # Create synthetic input data
        batch_size = 4
        num_nodes = 100
        feature_dim = 64
        
        # Mock DGL format inputs
        mfgs = [None, None]  # Placeholder for DGL message flow graphs
        X_node_dict = {
            "user": torch.randn(batch_size, num_nodes, feature_dim),
            "item": torch.randn(batch_size, num_nodes, feature_dim)
        }
        X_edge_dicts = [
            {("user", "interacts", "item"): torch.randn(batch_size, 20, 16)},
            {("user", "interacts", "item"): torch.randn(batch_size, 20, 16)}
        ]
        
        # Run forward pass
        output = model.forward(mfgs, X_node_dict, X_edge_dicts)
        
        # Validate output format
        self.assertIsInstance(output, dict)
        self.assertIn("default", output)
        self.assertEqual(output["default"].shape[0], batch_size)
        self.assertEqual(output["default"].shape[1], self.out_size)
        
    def test_ablation_controls(self):
        """Test ablation component on/off controls"""
        # Test with all ablation components disabled
        config_disabled = HHGTSolutionConfig(
            spot_target_enabled=False,
            cusp_enabled=False,
            trd_enabled=False,
            memory_enabled=False
        )
        
        model_disabled = HeteroHHGT(
            graph_config=None,
            solution_config=config_disabled,
            node_in_size_dict=self.node_in_size_dict,
            edge_in_size_dict=self.edge_in_size_dict,
            out_size=self.out_size,
            num_layers=2
        )
        
        # Check that ablation components are None when disabled
        self.assertIsNone(model_disabled.spot_target)
        self.assertIsNone(model_disabled.cusp_embeddings)
        self.assertIsNone(model_disabled.trd_sampler)
        self.assertIsNone(model_disabled.memory_module)
        
        # Test forward pass still works
        mfgs = [None, None]
        X_node_dict = {"default": torch.randn(2, 50, 64)}
        X_edge_dicts = [{}, {}]
        
        output = model_disabled.forward(mfgs, X_node_dict, X_edge_dicts)
        self.assertIsInstance(output, dict)
        self.assertIn("default", output)
        
    def test_hhgt_solution_initialization(self):
        """Test HHGTSolution wrapper initialization"""
        solution = HHGTSolution(self.config, None)
        
        self.assertEqual(solution.name, "hhgt")
        self.assertEqual(solution.config_class, HHGTSolutionConfig)
        self.assertIsInstance(solution.model, HHGT)
        
        # Test ablation config retrieval
        ablation_config = solution.get_ablation_config()
        self.assertIsInstance(ablation_config, dict)
        self.assertIn('spot_target_enabled', ablation_config)
        self.assertIn('cusp_enabled', ablation_config)
        self.assertIn('trd_enabled', ablation_config)
        self.assertIn('memory_enabled', ablation_config)
        
        # Check values match config
        self.assertEqual(ablation_config['spot_target_enabled'], self.config.spot_target_enabled)
        self.assertEqual(ablation_config['cusp_enabled'], self.config.cusp_enabled)
        
    def test_adapter_error_handling(self):
        """Test adapter error handling and fallback behavior"""
        adapter = DGLToHypergraphAdapter(self.config)
        
        # Test with invalid inputs (should fall back gracefully)
        try:
            result = adapter.convert_mfgs_to_hypergraph(None, {}, [])
            # Should not raise exception and return valid hypergraph data
            self.assertIsInstance(result, dict)
            self.assertIn('node_features', result)
        except Exception as e:
            self.fail(f"Adapter should handle errors gracefully: {e}")
            
    def test_temporal_feature_extraction(self):
        """Test temporal feature extraction for Memory/TGN components"""
        adapter = DGLToHypergraphAdapter(self.config)
        
        # Create mock node features with temporal information
        node_feat_dict = {
            "user": {
                "features": torch.randn(10, 64),
                "timestamp": torch.randint(0, 1000, (10,)),
                "creation_time": torch.randint(0, 1000, (10,))
            },
            "transaction": {
                "amount": torch.randn(20, 1),
                "time_of_day": torch.randint(0, 24, (20,))
            }
        }
        
        edge_feat_dicts = [
            {("user", "makes", "transaction"): {"timestamp": torch.randint(0, 1000, (15,))}}
        ]
        
        temporal_features = adapter.extract_temporal_features(node_feat_dict, edge_feat_dicts)
        
        # Should extract timestamp-related features
        self.assertIsInstance(temporal_features, dict)
        # Check that temporal features were found
        temporal_keys = [k for k in temporal_features.keys() if 'time' in k.lower()]
        self.assertGreater(len(temporal_keys), 0, "Should extract at least one temporal feature")

    def test_synthetic_data_flow(self):
        """Test end-to-end data flow with synthetic data"""
        solution = HHGTSolution(self.config, None)
        
        # Test that we can create and configure the solution
        self.assertIsNotNone(solution.model)
        
        # Test configuration serialization (important for checkpointing)
        config_dict = self.config.dict()
        self.assertIn('spot_target_enabled', config_dict)
        self.assertIn('hid_size', config_dict)
        self.assertIn('fanouts', config_dict)
        
        print("✓ All adapter tests passed!")


class TestAblationMatrix(unittest.TestCase):
    """Test ablation study configurations"""
    
    def test_ablation_combinations(self):
        """Test different ablation combinations"""
        ablation_configs = [
            # All enabled (baseline)
            {"spot_target_enabled": True, "cusp_enabled": True, "trd_enabled": True, "memory_enabled": True},
            # Individual ablations
            {"spot_target_enabled": False, "cusp_enabled": True, "trd_enabled": True, "memory_enabled": True},
            {"spot_target_enabled": True, "cusp_enabled": False, "trd_enabled": True, "memory_enabled": True},
            {"spot_target_enabled": True, "cusp_enabled": True, "trd_enabled": False, "memory_enabled": True},
            {"spot_target_enabled": True, "cusp_enabled": True, "trd_enabled": True, "memory_enabled": False},
            # All disabled
            {"spot_target_enabled": False, "cusp_enabled": False, "trd_enabled": False, "memory_enabled": False},
        ]
        
        for i, ablation in enumerate(ablation_configs):
            with self.subTest(ablation_id=i):
                config = HHGTSolutionConfig(**ablation)
                solution = HHGTSolution(config, None)
                
                # Verify configuration is applied
                actual_ablation = solution.get_ablation_config()
                for key, expected_value in ablation.items():
                    self.assertEqual(actual_ablation[key], expected_value, 
                                   f"Ablation {key} mismatch in config {i}")
                    
        print(f"✓ Tested {len(ablation_configs)} ablation configurations")


if __name__ == "__main__":
    # Run the test suite
    print("Running hHGTN 4DBInfer Adapter Tests...")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestHHGTAdapter))
    suite.addTests(loader.loadTestsFromTestCase(TestAblationMatrix))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed! Adapter is ready for integration.")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
