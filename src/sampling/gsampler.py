# src/sampling/gsampler.py
"""
G-SAMPLER Python wrapper per §PHASE_B.3
GPU-native, high-throughput sampling implementation
"""

import torch
import torch.nn as nn
import numpy as np
import ctypes
import os
from typing import Optional, List, Tuple, Dict, Any
import logging
from dataclasses import dataclass

from .cpu_fallback import SubgraphBatch, TemporalGraph

logger = logging.getLogger(__name__)

@dataclass
class GSamplerConfig:
    """Configuration for G-SAMPLER per §PHASE_B.4"""
    device: str = 'cuda:0'
    max_batch_nodes: int = 10000
    memory_pool_size: str = "1GB"
    stream_overlap: bool = True
    cpu_fallback: bool = True
    warp_size: int = 32
    max_fanout: int = 100

class GSampler:
    """
    GPU-based graph sampler per §PHASE_B.3
    Implements exact G-SAMPLER algorithms from reference
    """
    
    def __init__(
        self, 
        csr_indptr: torch.Tensor, 
        csr_indices: torch.Tensor, 
        csr_timestamps: torch.Tensor, 
        device: str = 'cuda:0',
        config: Optional[GSamplerConfig] = None
    ):
        """
        Initialize G-SAMPLER with compressed graph arrays per §PHASE_B.3
        
        Args:
            csr_indptr: CSR pointer array (n+1)
            csr_indices: neighbor indices (nnz)
            csr_timestamps: edge timestamps aligned with indices (nnz)
            device: CUDA device string
            config: G-SAMPLER configuration
        """
        self.device = torch.device(device)
        self.config = config or GSamplerConfig()
        
        # Validate inputs
        assert len(csr_indptr) == len(csr_indices) + 1 or len(csr_indptr) > 0
        assert len(csr_indices) == len(csr_timestamps)
        
        # Copy data to GPU per §PHASE_B.3
        self.indptr = csr_indptr.to(self.device, dtype=torch.int32)
        self.indices = csr_indices.to(self.device, dtype=torch.int32)
        self.timestamps = csr_timestamps.to(self.device, dtype=torch.float32)
        
        self.num_nodes = len(self.indptr) - 1
        self.num_edges = len(self.indices)
        
        # Load CUDA kernels
        self._load_cuda_kernels()
        
        # Pre-allocate memory pools per §PHASE_B.4
        self._init_memory_pools()
        
        logger.info(f"G-SAMPLER initialized: {self.num_nodes} nodes, {self.num_edges} edges on {device}")
        
    def _load_cuda_kernels(self):
        """Load compiled CUDA kernels for GPU sampling"""
        try:
            kernel_lib_path = os.path.join(
                os.path.dirname(__file__), 
                'kernels', 
                'libgsampler_kernels.so'
            )
            
            if not os.path.exists(kernel_lib_path):
                logger.warning(f"CUDA kernels not found at {kernel_lib_path}")
                self.cuda_available = False
                return
            
            self.cuda_lib = ctypes.CDLL(kernel_lib_path)
            
            # Define function signatures
            self.cuda_lib.launch_time_window_cutoff.argtypes = [
                ctypes.POINTER(ctypes.c_int32),  # indptr
                ctypes.POINTER(ctypes.c_int32),  # indices  
                ctypes.POINTER(ctypes.c_float),  # timestamps
                ctypes.POINTER(ctypes.c_int32),  # seed_nodes
                ctypes.POINTER(ctypes.c_float),  # t_eval_array
                ctypes.c_float,                  # delta_t
                ctypes.POINTER(ctypes.c_int32),  # start_indices
                ctypes.POINTER(ctypes.c_int32),  # end_indices
                ctypes.c_int32                   # num_seeds
            ]
            
            self.cuda_lib.launch_sample_k_from_range.argtypes = [
                ctypes.POINTER(ctypes.c_int32),  # indices
                ctypes.POINTER(ctypes.c_int32),  # start_indices
                ctypes.POINTER(ctypes.c_int32),  # end_indices
                ctypes.POINTER(ctypes.c_int32),  # fanouts
                ctypes.c_int32,                  # hop
                ctypes.POINTER(ctypes.c_int32),  # sampled_indices
                ctypes.POINTER(ctypes.c_int32),  # sampled_counts
                ctypes.c_int32,                  # num_seeds
                ctypes.POINTER(ctypes.c_uint32)  # random_states
            ]
            
            self.cuda_available = True
            logger.info("CUDA kernels loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load CUDA kernels: {e}")
            self.cuda_available = False
    
    def _init_memory_pools(self):
        """Initialize GPU memory pools per §PHASE_B.4"""
        max_batch = self.config.max_batch_nodes
        max_fanout = self.config.max_fanout
        
        # Pre-allocate buffers for largest expected batch
        self.start_indices_buffer = torch.zeros(max_batch, dtype=torch.int32, device=self.device)
        self.end_indices_buffer = torch.zeros(max_batch, dtype=torch.int32, device=self.device)
        self.sampled_indices_buffer = torch.zeros(max_batch * max_fanout, dtype=torch.int32, device=self.device)
        self.sampled_counts_buffer = torch.zeros(max_batch, dtype=torch.int32, device=self.device)
        self.node_mapping_buffer = torch.full((self.num_nodes,), -1, dtype=torch.int32, device=self.device)
        
        logger.info(f"GPU memory pools initialized for max_batch={max_batch}")
        
    def sample_time_relaxed(
        self, 
        seed_nodes: torch.Tensor, 
        t_eval_array: torch.Tensor, 
        fanouts: List[int], 
        delta_t: float, 
        strategy: str = 'recency'
    ) -> SubgraphBatch:
        """
        GPU time-relaxed sampling per §PHASE_B.3 API
        
        Args:
            seed_nodes: seed nodes for sampling (batch_size)
            t_eval_array: evaluation timestamps per seed (batch_size) 
            fanouts: neighbor fanouts per hop [f1, f2, ...]
            delta_t: time relaxation parameter
            strategy: sampling strategy ('recency' or 'random')
            
        Returns:
            SubgraphBatch with GPU-resident tensors
        """
        # Debug prints per APPENDIX requirements
        print(f"[GSAMPLER] Starting GPU sampling: seeds={len(seed_nodes)}, delta_t={delta_t}")
        
        # Check GPU availability and fallback if needed
        if not self.cuda_available or not torch.cuda.is_available():
            return self._cpu_fallback_sampling(seed_nodes, t_eval_array, fanouts, delta_t, strategy)
        
        # Memory safety check per §PHASE_B.4
        if len(seed_nodes) > self.config.max_batch_nodes:
            logger.warning(f"Batch size {len(seed_nodes)} exceeds max {self.config.max_batch_nodes}, using CPU fallback")
            return self._cpu_fallback_sampling(seed_nodes, t_eval_array, fanouts, delta_t, strategy)
        
        try:
            return self._gpu_sample_time_relaxed(seed_nodes, t_eval_array, fanouts, delta_t, strategy)
        except Exception as e:
            logger.error(f"GPU sampling failed: {e}")
            if self.config.cpu_fallback:
                logger.info("Falling back to CPU sampling")
                return self._cpu_fallback_sampling(seed_nodes, t_eval_array, fanouts, delta_t, strategy)
            else:
                raise
    
    def _gpu_sample_time_relaxed(
        self, 
        seed_nodes: torch.Tensor, 
        t_eval_array: torch.Tensor, 
        fanouts: List[int], 
        delta_t: float, 
        strategy: str
    ) -> SubgraphBatch:
        """Core GPU sampling implementation per §PHASE_B.2 kernel sequence"""
        
        # Ensure tensors are on GPU
        seed_nodes = seed_nodes.to(self.device, dtype=torch.int32)
        t_eval_array = t_eval_array.to(self.device, dtype=torch.float32)
        
        num_seeds = len(seed_nodes)
        depth = len(fanouts)
        
        # Track all sampled nodes across hops
        all_sampled_nodes = set(seed_nodes.cpu().tolist())
        current_frontier = seed_nodes.clone()
        
        # Multi-hop sampling per §PHASE_B.2 iterative approach
        for hop in range(depth):
            if len(current_frontier) == 0:
                break
                
            fanout = fanouts[hop]
            
            # Step 1: Time window cutoff kernel per §PHASE_B.2
            start_indices = self.start_indices_buffer[:len(current_frontier)]
            end_indices = self.end_indices_buffer[:len(current_frontier)]
            
            self._call_time_window_cutoff_kernel(
                current_frontier, t_eval_array[:len(current_frontier)], 
                delta_t, start_indices, end_indices
            )
            
            # Step 2: Sample k from range kernel per §PHASE_B.2
            sampled_indices = self.sampled_indices_buffer[:len(current_frontier) * fanout]
            sampled_counts = self.sampled_counts_buffer[:len(current_frontier)]
            
            self._call_sample_k_from_range_kernel(
                start_indices, end_indices, fanout, hop,
                sampled_indices, sampled_counts
            )
            
            # Collect new frontier nodes
            new_frontier_nodes = []
            offset = 0
            for i in range(len(current_frontier)):
                count = sampled_counts[i].item()
                for j in range(count):
                    neighbor_id = sampled_indices[offset + j].item()
                    if neighbor_id not in all_sampled_nodes:
                        new_frontier_nodes.append(neighbor_id)
                        all_sampled_nodes.add(neighbor_id)
                offset += fanout
            
            current_frontier = torch.tensor(new_frontier_nodes, device=self.device, dtype=torch.int32)
            
            # Debug prints per APPENDIX
            print(f"[GSAMPLER] hop={hop} frontier_size={len(current_frontier)} total_sampled={len(all_sampled_nodes)}")
        
        # Step 3: Compact subgraph kernel per §PHASE_B.2
        sampled_node_list = sorted(list(all_sampled_nodes))
        subgraph_batch = self._build_subgraph_batch(sampled_node_list, seed_nodes)
        
        # Debug memory usage per APPENDIX
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            print(f"[MEM] allocated={allocated:.3f}GB reserved={reserved:.3f}GB")
        
        return subgraph_batch
    
    def _call_time_window_cutoff_kernel(
        self, 
        seed_nodes: torch.Tensor, 
        t_eval_array: torch.Tensor, 
        delta_t: float,
        start_indices: torch.Tensor, 
        end_indices: torch.Tensor
    ):
        """Call CUDA time window cutoff kernel"""
        num_seeds = len(seed_nodes)
        
        # Get data pointers 
        indptr_ptr = self.indptr.data_ptr()
        indices_ptr = self.indices.data_ptr()
        timestamps_ptr = self.timestamps.data_ptr()
        seeds_ptr = seed_nodes.data_ptr()
        t_eval_ptr = t_eval_array.data_ptr()
        start_ptr = start_indices.data_ptr()
        end_ptr = end_indices.data_ptr()
        
        # Call CUDA kernel
        self.cuda_lib.launch_time_window_cutoff(
            ctypes.cast(indptr_ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(indices_ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(timestamps_ptr, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(seeds_ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(t_eval_ptr, ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(delta_t),
            ctypes.cast(start_ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(end_ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(num_seeds)
        )
    
    def _call_sample_k_from_range_kernel(
        self,
        start_indices: torch.Tensor,
        end_indices: torch.Tensor, 
        fanout: int,
        hop: int,
        sampled_indices: torch.Tensor,
        sampled_counts: torch.Tensor
    ):
        """Call CUDA sample k from range kernel"""
        num_seeds = len(start_indices)
        fanouts_tensor = torch.tensor([fanout], device=self.device, dtype=torch.int32)
        
        # Simple random states (can be improved)
        random_states = torch.randint(0, 2**32-1, (num_seeds,), device=self.device, dtype=torch.uint32)
        
        # Get data pointers
        indices_ptr = self.indices.data_ptr()
        start_ptr = start_indices.data_ptr()
        end_ptr = end_indices.data_ptr()
        fanouts_ptr = fanouts_tensor.data_ptr()
        sampled_ptr = sampled_indices.data_ptr()
        counts_ptr = sampled_counts.data_ptr()
        random_ptr = random_states.data_ptr()
        
        # Call CUDA kernel
        self.cuda_lib.launch_sample_k_from_range(
            ctypes.cast(indices_ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(start_ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(end_ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(fanouts_ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(hop),
            ctypes.cast(sampled_ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(counts_ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(num_seeds),
            ctypes.cast(random_ptr, ctypes.POINTER(ctypes.c_uint32))
        )
    
    def _build_subgraph_batch(
        self, 
        sampled_node_list: List[int], 
        seed_nodes: torch.Tensor
    ) -> SubgraphBatch:
        """Build final subgraph batch from sampled nodes per §PHASE_B.2"""
        num_sampled = len(sampled_node_list)
        
        # Create node mapping
        node_mapping = {}
        for new_id, orig_id in enumerate(sampled_node_list):
            node_mapping[orig_id] = new_id
        
        # Build subgraph edges (simplified CPU version for now)
        # TODO: Use compact_subgraph_kernel for full GPU implementation
        sub_edges = []
        sub_timestamps = []
        
        for orig_u in sampled_node_list:
            start_idx = self.indptr[orig_u].item()
            end_idx = self.indptr[orig_u + 1].item()
            
            for edge_idx in range(start_idx, end_idx):
                orig_v = self.indices[edge_idx].item()
                if orig_v in node_mapping:
                    new_u = node_mapping[orig_u]
                    new_v = node_mapping[orig_v]
                    sub_edges.append((new_u, new_v))
                    sub_timestamps.append(self.timestamps[edge_idx].item())
        
        # Build CSR structure on GPU
        sub_indptr = torch.zeros(num_sampled + 1, dtype=torch.int32, device=self.device)
        if sub_edges:
            sub_indices = torch.tensor([e[1] for e in sub_edges], dtype=torch.int32, device=self.device)
            sub_timestamps_tensor = torch.tensor(sub_timestamps, dtype=torch.float32, device=self.device)
            
            # Count edges per node
            for u, v in sub_edges:
                sub_indptr[u + 1] += 1
            
            # Cumulative sum
            torch.cumsum(sub_indptr, dim=0, out=sub_indptr)
        else:
            sub_indices = torch.tensor([], dtype=torch.int32, device=self.device)
            sub_timestamps_tensor = torch.tensor([], dtype=torch.float32, device=self.device)
        
        # Create train mask for seed nodes
        train_mask = torch.zeros(num_sampled, dtype=torch.bool, device=self.device)
        for orig_seed in seed_nodes:
            orig_seed_item = orig_seed.item()
            if orig_seed_item in node_mapping:
                train_mask[node_mapping[orig_seed_item]] = True
        
        return SubgraphBatch(
            seed_nodes=seed_nodes,
            sub_indptr=sub_indptr,
            sub_indices=sub_indices,
            sub_timestamps=sub_timestamps_tensor,
            node_mapping=node_mapping,
            train_mask=train_mask,
            num_nodes=num_sampled,
            num_edges=len(sub_edges)
        )
    
    def _cpu_fallback_sampling(
        self, 
        seed_nodes: torch.Tensor, 
        t_eval_array: torch.Tensor, 
        fanouts: List[int], 
        delta_t: float, 
        strategy: str
    ) -> SubgraphBatch:
        """CPU fallback using existing implementation per §PHASE_B.4"""
        from .cpu_fallback import sample_time_relaxed_neighbors
        
        # Convert to CPU temporal graph
        temporal_graph = TemporalGraph(
            indptr=self.indptr.cpu(),
            indices=self.indices.cpu(),
            timestamps=self.timestamps.cpu(),
            num_nodes=self.num_nodes,
            num_edges=self.num_edges
        )
        
        # Call CPU implementation
        return sample_time_relaxed_neighbors(
            node_ids=seed_nodes.cpu(),
            t_eval=t_eval_array.cpu(),
            depth=len(fanouts),
            fanouts=fanouts,
            delta_t=delta_t,
            temporal_graph=temporal_graph,
            strategy=strategy
        )
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get GPU memory statistics per §PHASE_B.4"""
        if not torch.cuda.is_available():
            return {}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1e9
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated,
            'utilization': allocated / reserved if reserved > 0 else 0.0
        }
