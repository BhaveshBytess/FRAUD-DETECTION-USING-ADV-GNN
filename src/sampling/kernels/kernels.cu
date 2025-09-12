// src/sampling/kernels/kernels.cu
// CUDA kernels for G-SAMPLER GPU implementation per §PHASE_B.2
// Implements exact GPU sampler primitives from STAGE6_TDGNN_GSAMPLER_REFERENCE.md

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

// Warp-level binary search for time window cutoff per §PHASE_B.2
__device__ int warp_binary_search(
    const float* timestamps, 
    int start_idx, 
    int end_idx, 
    float target_time, 
    bool find_lower_bound
) {
    int left = start_idx;
    int right = end_idx - 1;
    int result = find_lower_bound ? end_idx : start_idx - 1;
    
    // Cooperative binary search within warp
    while (left <= right) {
        int mid = (left + right) / 2;
        float mid_val = timestamps[mid];
        
        if (find_lower_bound) {
            // Find first index where timestamp >= target_time
            if (mid_val >= target_time) {
                result = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {
            // Find last index where timestamp <= target_time
            if (mid_val <= target_time) {
                result = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    
    return result;
}

/**
 * Kernel 1: time_window_cutoff_kernel per §PHASE_B.2
 * Input: indptr, indices, timestamps, seed_nodes, t_eval_array, delta_t
 * Output: start_idx, end_idx per seed neighbor list
 */
__global__ void time_window_cutoff_kernel(
    const int* indptr,           // CSR pointer array (n+1)
    const int* indices,          // neighbor indices (nnz)
    const float* timestamps,     // edge timestamps aligned with indices (nnz)
    const int* seed_nodes,       // seed nodes for sampling (batch_size)
    const float* t_eval_array,   // evaluation times per seed (batch_size)
    float delta_t,               // time relaxation window
    int* start_indices,          // output: start index in filtered range (batch_size)
    int* end_indices,            // output: end index in filtered range (batch_size)
    int num_seeds
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Each warp processes one seed node
    if (warp_id >= num_seeds) return;
    
    int seed_node = seed_nodes[warp_id];
    float t_eval = t_eval_array[warp_id];
    float t_start = t_eval - delta_t;
    float t_end = t_eval;
    
    // Get neighbor range for this seed
    int neighbor_start = indptr[seed_node];
    int neighbor_end = indptr[seed_node + 1];
    
    if (neighbor_start >= neighbor_end) {
        // No neighbors
        if (lane_id == 0) {
            start_indices[warp_id] = neighbor_start;
            end_indices[warp_id] = neighbor_start;
        }
        return;
    }
    
    // Warp-level binary search for time window bounds
    // Assuming timestamps are sorted descending per node per §PHASE_A.2
    int valid_start = -1, valid_end = -1;
    
    if (lane_id == 0) {
        // Find first index where timestamp <= t_end (upper bound)
        valid_start = neighbor_start;
        while (valid_start < neighbor_end && timestamps[valid_start] > t_end) {
            valid_start++;
        }
        
        // Find last index where timestamp >= t_start (lower bound)  
        valid_end = neighbor_end - 1;
        while (valid_end >= neighbor_start && timestamps[valid_end] < t_start) {
            valid_end--;
        }
        
        // Ensure valid range
        if (valid_start > valid_end) {
            valid_start = neighbor_start;
            valid_end = neighbor_start - 1;  // Empty range
        } else {
            valid_end = valid_end + 1;  // Convert to exclusive end
        }
        
        start_indices[warp_id] = valid_start;
        end_indices[warp_id] = valid_end;
    }
}

/**
 * Kernel 2: sample_k_from_range_kernel per §PHASE_B.2
 * Input: index ranges, k (fanout), strategy
 * Output: sampled neighbor indices
 */
__global__ void sample_k_from_range_kernel(
    const int* indices,          // neighbor indices array
    const int* start_indices,    // start of valid range per seed
    const int* end_indices,      // end of valid range per seed
    const int* fanouts,          // fanout per hop (max_depth)
    int hop,                     // current hop index
    int* sampled_indices,        // output: sampled neighbor indices
    int* sampled_counts,         // output: number sampled per seed
    int num_seeds,
    unsigned int* random_states  // random states for sampling
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_seeds) return;
    
    int start_idx = start_indices[tid];
    int end_idx = end_indices[tid];
    int k = fanouts[hop];
    int range_size = end_idx - start_idx;
    
    if (range_size <= 0) {
        sampled_counts[tid] = 0;
        return;
    }
    
    int actual_k = min(k, range_size);
    sampled_counts[tid] = actual_k;
    
    // Calculate output offset for this seed
    int output_offset = tid * k;  // Assume pre-allocated max space
    
    if (actual_k == range_size) {
        // Take all candidates (recency strategy default per §PHASE_A.2)
        for (int i = 0; i < actual_k; i++) {
            sampled_indices[output_offset + i] = indices[start_idx + i];
        }
    } else {
        // Sample top-k by recency (already sorted) or random
        // For now, implement recency (top-k)
        for (int i = 0; i < actual_k; i++) {
            sampled_indices[output_offset + i] = indices[start_idx + i];
        }
        
        // TODO: Add random sampling strategy if needed
        // Can use reservoir sampling with random_states[tid]
    }
}

/**
 * Kernel 3: compact_subgraph_kernel per §PHASE_B.2
 * Build subgraph CSR for union of sampled nodes
 */
__global__ void compact_subgraph_kernel(
    const int* original_indptr,     // original CSR pointers
    const int* original_indices,    // original neighbor indices
    const float* original_timestamps, // original edge timestamps
    const int* sampled_node_list,   // list of all sampled nodes
    const int* node_mapping,        // original_id -> new_id mapping
    int num_sampled_nodes,
    int* sub_indptr,               // output: subgraph CSR pointers
    int* sub_indices,              // output: subgraph neighbor indices
    float* sub_timestamps,         // output: subgraph edge timestamps
    int max_edges
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_sampled_nodes) return;
    
    int orig_node_id = sampled_node_list[tid];
    int new_node_id = tid;  // Sequential mapping
    
    // Count edges for this node that stay within subgraph
    int orig_start = original_indptr[orig_node_id];
    int orig_end = original_indptr[orig_node_id + 1];
    int edge_count = 0;
    
    // First pass: count valid edges
    for (int i = orig_start; i < orig_end; i++) {
        int neighbor_id = original_indices[i];
        if (node_mapping[neighbor_id] != -1) {  // Neighbor is in subgraph
            edge_count++;
        }
    }
    
    // Store edge count (will be converted to pointers later)
    sub_indptr[new_node_id + 1] = edge_count;
    
    // Second pass: copy valid edges (if we have space)
    int edge_offset = atomicAdd(&sub_indptr[0], edge_count);  // Atomic counter
    if (edge_offset + edge_count <= max_edges) {
        int write_idx = edge_offset;
        for (int i = orig_start; i < orig_end; i++) {
            int neighbor_id = original_indices[i];
            int new_neighbor_id = node_mapping[neighbor_id];
            if (new_neighbor_id != -1) {
                sub_indices[write_idx] = new_neighbor_id;
                sub_timestamps[write_idx] = original_timestamps[i];
                write_idx++;
            }
        }
    }
}

/**
 * Kernel 4: parallel_map_kernel per §PHASE_B.2
 * Map subgraph node ids to contiguous ids in batch
 */
__global__ void parallel_map_kernel(
    const int* original_ids,      // original node IDs
    int* node_mapping,           // output: original_id -> new_id mapping (global array)
    int* reverse_mapping,        // output: new_id -> original_id mapping
    int num_nodes,
    int max_node_id
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_nodes) return;
    
    int orig_id = original_ids[tid];
    int new_id = tid;
    
    // Create bidirectional mapping
    if (orig_id < max_node_id) {
        node_mapping[orig_id] = new_id;
    }
    reverse_mapping[new_id] = orig_id;
}

// Host function declarations for Python interface
extern "C" {
    void launch_time_window_cutoff(
        const int* indptr, const int* indices, const float* timestamps,
        const int* seed_nodes, const float* t_eval_array, float delta_t,
        int* start_indices, int* end_indices, int num_seeds
    );
    
    void launch_sample_k_from_range(
        const int* indices, const int* start_indices, const int* end_indices,
        const int* fanouts, int hop, int* sampled_indices, int* sampled_counts,
        int num_seeds, unsigned int* random_states
    );
    
    void launch_compact_subgraph(
        const int* original_indptr, const int* original_indices, const float* original_timestamps,
        const int* sampled_node_list, const int* node_mapping, int num_sampled_nodes,
        int* sub_indptr, int* sub_indices, float* sub_timestamps, int max_edges
    );
    
    void launch_parallel_map(
        const int* original_ids, int* node_mapping, int* reverse_mapping,
        int num_nodes, int max_node_id
    );
}

// Host function implementations
void launch_time_window_cutoff(
    const int* indptr, const int* indices, const float* timestamps,
    const int* seed_nodes, const float* t_eval_array, float delta_t,
    int* start_indices, int* end_indices, int num_seeds
) {
    // Use warp-centric launch: 32 threads per seed (one warp per seed)
    int threads_per_block = 256;
    int num_warps = (num_seeds + 7) / 8;  // 8 warps per block
    int num_blocks = (num_warps * 32 + threads_per_block - 1) / threads_per_block;
    
    time_window_cutoff_kernel<<<num_blocks, threads_per_block>>>(
        indptr, indices, timestamps, seed_nodes, t_eval_array, delta_t,
        start_indices, end_indices, num_seeds
    );
    
    cudaDeviceSynchronize();
}

void launch_sample_k_from_range(
    const int* indices, const int* start_indices, const int* end_indices,
    const int* fanouts, int hop, int* sampled_indices, int* sampled_counts,
    int num_seeds, unsigned int* random_states
) {
    int threads_per_block = 256;
    int num_blocks = (num_seeds + threads_per_block - 1) / threads_per_block;
    
    sample_k_from_range_kernel<<<num_blocks, threads_per_block>>>(
        indices, start_indices, end_indices, fanouts, hop,
        sampled_indices, sampled_counts, num_seeds, random_states
    );
    
    cudaDeviceSynchronize();
}

void launch_compact_subgraph(
    const int* original_indptr, const int* original_indices, const float* original_timestamps,
    const int* sampled_node_list, const int* node_mapping, int num_sampled_nodes,
    int* sub_indptr, int* sub_indices, float* sub_timestamps, int max_edges
) {
    int threads_per_block = 256;
    int num_blocks = (num_sampled_nodes + threads_per_block - 1) / threads_per_block;
    
    compact_subgraph_kernel<<<num_blocks, threads_per_block>>>(
        original_indptr, original_indices, original_timestamps,
        sampled_node_list, node_mapping, num_sampled_nodes,
        sub_indptr, sub_indices, sub_timestamps, max_edges
    );
    
    cudaDeviceSynchronize();
}

void launch_parallel_map(
    const int* original_ids, int* node_mapping, int* reverse_mapping,
    int num_nodes, int max_node_id
) {
    int threads_per_block = 256;
    int num_blocks = (num_nodes + threads_per_block - 1) / threads_per_block;
    
    parallel_map_kernel<<<num_blocks, threads_per_block>>>(
        original_ids, node_mapping, reverse_mapping, num_nodes, max_node_id
    );
    
    cudaDeviceSynchronize();
}
