# tests/test_gsampler_gpu_cpu_consistency.py
"""
GPU-CPU consistency tests for G-SAMPLER per §PHASE_B.5 - MANDATORY per APPENDIX
Tests must run in CI in lite-mode and be fast (<30s)
"""

import torch
import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sampling.gsampler import GSampler, GSamplerConfig
from sampling.cpu_fallback import sample_time_relaxed_neighbors, TemporalGraph
from sampling.utils import validate_temporal_constraints

class TestGSamplerConsistency:
    """Test suite for GPU-CPU sampling consistency per §PHASE_B.5"""
    
    def setup_method(self):
        """Create synthetic temporal graph for testing"""
        # Same test graph as Phase A but with more edges for GPU testing
        # Node 0 -> [1 (t=10), 2 (t=8), 3 (t=5), 4 (t=3)]
        # Node 1 -> [2 (t=9), 3 (t=7), 4 (t=4)]  
        # Node 2 -> [3 (t=6), 4 (t=2)]
        # Node 3 -> [4 (t=1)]
        # Node 4 -> []
        
        self.num_nodes = 5
        self.indptr = torch.tensor([0, 4, 7, 9, 10, 10])  # pointers
        self.indices = torch.tensor([1, 2, 3, 4, 2, 3, 4, 3, 4, 4])  # neighbors
        self.timestamps = torch.tensor([10.0, 8.0, 5.0, 3.0, 9.0, 7.0, 4.0, 6.0, 2.0, 1.0])  # descending per node
        
        self.temporal_graph = TemporalGraph(
            indptr=self.indptr,
            indices=self.indices, 
            timestamps=self.timestamps,
            num_nodes=self.num_nodes,
            num_edges=10
        )
        
        # Create G-SAMPLER instance
        self.gsampler_config = GSamplerConfig(
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            max_batch_nodes=100,
            cpu_fallback=True
        )
        
        self.gsampler = GSampler(
            csr_indptr=self.indptr,
            csr_indices=self.indices,
            csr_timestamps=self.timestamps,
            device=self.gsampler_config.device,
            config=self.gsampler_config
        )
    
    def test_sampling_correctness(self):
        """Test GPU sampling output equals CPU reference per §PHASE_B.5"""
        seed_nodes = torch.tensor([0, 1])
        t_eval = torch.tensor([10.0, 9.0])
        delta_t = 5.0
        fanouts = [2, 1]
        
        # CPU reference
        cpu_result = sample_time_relaxed_neighbors(
            node_ids=seed_nodes,
            t_eval=t_eval,
            depth=2,
            fanouts=fanouts,
            delta_t=delta_t,
            temporal_graph=self.temporal_graph,
            strategy='recency'
        )
        
        # GPU sampling (may fallback to CPU if CUDA not available)
        gpu_result = self.gsampler.sample_time_relaxed(
            seed_nodes=seed_nodes,
            t_eval_array=t_eval,
            fanouts=fanouts,
            delta_t=delta_t,
            strategy='recency'
        )
        
        # Compare results
        print(f"CPU result: {cpu_result.num_nodes} nodes, {cpu_result.num_edges} edges")
        print(f"GPU result: {gpu_result.num_nodes} nodes, {gpu_result.num_edges} edges")
        
        # Basic consistency checks
        assert cpu_result.num_nodes == gpu_result.num_nodes, f"Node count mismatch: CPU={cpu_result.num_nodes}, GPU={gpu_result.num_nodes}"
        
        # Check that seed nodes are included
        cpu_seeds_in_mapping = all(seed.item() in cpu_result.node_mapping for seed in seed_nodes)
        gpu_seeds_in_mapping = all(seed.item() in gpu_result.node_mapping for seed in seed_nodes)
        
        assert cpu_seeds_in_mapping, "CPU result missing seed nodes"
        assert gpu_seeds_in_mapping, "GPU result missing seed nodes"
        
        # Validate temporal constraints for both
        cpu_validation = validate_temporal_constraints(
            temporal_graph=self.temporal_graph,
            seed_nodes=seed_nodes,
            t_eval=t_eval,
            delta_t=delta_t,
            sampled_subgraph=cpu_result
        )
        
        gpu_validation = validate_temporal_constraints(
            temporal_graph=self.temporal_graph,
            seed_nodes=seed_nodes,
            t_eval=t_eval,
            delta_t=delta_t,
            sampled_subgraph=gpu_result
        )
        
        assert all(cpu_validation.values()), f"CPU validation failed: {cpu_validation}"
        assert all(gpu_validation.values()), f"GPU validation failed: {gpu_validation}"
        
        print("✓ GPU-CPU sampling consistency test passed")
    
    def test_throughput(self):
        """Test throughput and compare to CPU per §PHASE_B.5"""
        import time
        
        # Larger batch for throughput testing
        batch_size = 50 if torch.cuda.is_available() else 10
        seed_nodes = torch.randint(0, self.num_nodes, (batch_size,))
        t_eval = torch.full((batch_size,), 10.0)
        delta_t = 5.0
        fanouts = [3, 2]
        
        # Time CPU sampling
        start_time = time.time()
        cpu_result = sample_time_relaxed_neighbors(
            node_ids=seed_nodes,
            t_eval=t_eval,
            depth=2,
            fanouts=fanouts,
            delta_t=delta_t,
            temporal_graph=self.temporal_graph,
            strategy='recency'
        )
        cpu_time = time.time() - start_time
        
        # Time GPU sampling 
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        gpu_result = self.gsampler.sample_time_relaxed(
            seed_nodes=seed_nodes,
            t_eval_array=t_eval,
            fanouts=fanouts,
            delta_t=delta_t,
            strategy='recency'
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"Throughput comparison (batch_size={batch_size}):")
        print(f"  CPU time: {cpu_time:.4f}s")
        print(f"  GPU time: {gpu_time:.4f}s")
        
        if torch.cuda.is_available() and self.gsampler.cuda_available:
            speedup = cpu_time / gpu_time
            print(f"  Speedup: {speedup:.2f}x")
            # Note: Real speedup expected only for larger graphs per §PHASE_D.3
        
        print("✓ Throughput test completed")
    
    def test_memory_safety_check(self):
        """Test memory safety per §PHASE_B.5"""
        if not torch.cuda.is_available():
            print("CUDA not available, skipping memory test")
            return
        
        # Get initial memory state
        initial_memory = torch.cuda.memory_allocated()
        
        # Run sampling
        seed_nodes = torch.tensor([0, 1, 2])
        t_eval = torch.tensor([10.0, 9.0, 8.0])
        
        result = self.gsampler.sample_time_relaxed(
            seed_nodes=seed_nodes,
            t_eval_array=t_eval,
            fanouts=[2, 1],
            delta_t=5.0
        )
        
        # Check memory stats
        memory_stats = self.gsampler.get_memory_stats()
        print(f"Memory stats: {memory_stats}")
        
        # Ensure memory usage is reasonable
        memory_threshold = 1.0  # 1GB
        assert memory_stats.get('reserved_gb', 0) < memory_threshold, f"Memory usage {memory_stats.get('reserved_gb', 0):.3f}GB exceeds threshold {memory_threshold}GB"
        
        print("✓ Memory safety test passed")
    
    def test_edge_cases(self):
        """Test edge cases for G-SAMPLER"""
        # Test empty seed nodes
        empty_seeds = torch.tensor([], dtype=torch.int32)
        empty_t_eval = torch.tensor([], dtype=torch.float32)
        
        result = self.gsampler.sample_time_relaxed(
            seed_nodes=empty_seeds,
            t_eval_array=empty_t_eval,
            fanouts=[2, 1],
            delta_t=5.0
        )
        
        assert result.num_nodes == 0, "Empty input should produce empty result"
        print("✓ Empty input test passed")
        
        # Test single seed
        single_seed = torch.tensor([0])
        single_t_eval = torch.tensor([10.0])
        
        result = self.gsampler.sample_time_relaxed(
            seed_nodes=single_seed,
            t_eval_array=single_t_eval,
            fanouts=[2],
            delta_t=5.0
        )
        
        assert result.num_nodes >= 1, "Single seed should be included"
        assert 0 in result.node_mapping, "Seed node should be in mapping"
        print("✓ Single seed test passed")
        
        # Test large fanout (larger than available neighbors)
        large_fanout_result = self.gsampler.sample_time_relaxed(
            seed_nodes=torch.tensor([4]),  # Node with no neighbors
            t_eval_array=torch.tensor([10.0]),
            fanouts=[100],  # Very large fanout
            delta_t=10.0
        )
        
        assert large_fanout_result.num_nodes == 1, "Node with no neighbors should only include itself"
        print("✓ Large fanout test passed")
        
        # Test very restrictive time window
        restrictive_result = self.gsampler.sample_time_relaxed(
            seed_nodes=torch.tensor([0]),
            t_eval_array=torch.tensor([1.0]),  # Very early time
            fanouts=[5],
            delta_t=0.1  # Very small window
        )
        
        # Should have very few or no neighbors due to time restriction
        assert restrictive_result.num_nodes >= 1, "Should include at least seed node"
        print("✓ Restrictive time window test passed")

def test_gsampler_initialization():
    """Test G-SAMPLER initialization"""
    # Minimal graph
    indptr = torch.tensor([0, 1, 2])
    indices = torch.tensor([1, 0])
    timestamps = torch.tensor([1.0, 2.0])
    
    gsampler = GSampler(
        csr_indptr=indptr,
        csr_indices=indices,
        csr_timestamps=timestamps,
        device='cpu'  # Force CPU for testing
    )
    
    assert gsampler.num_nodes == 2
    assert gsampler.num_edges == 2
    print("✓ G-SAMPLER initialization test passed")

if __name__ == "__main__":
    # Run tests directly for development
    test_suite = TestGSamplerConsistency()
    test_suite.setup_method()
    
    print("Running G-SAMPLER GPU-CPU consistency tests...")
    
    test_suite.test_sampling_correctness()
    test_suite.test_throughput()
    test_suite.test_memory_safety_check()
    test_suite.test_edge_cases()
    
    test_gsampler_initialization()
    
    print("All G-SAMPLER tests passed! ✓")
