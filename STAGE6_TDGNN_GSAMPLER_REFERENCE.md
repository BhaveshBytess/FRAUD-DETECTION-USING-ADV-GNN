# STAGE6\_TDGNN\_GSAMPLER\_REFERENCE.md

**Stage 6 — Unified Reference (TDGNN + G-SAMPLER)**
**Project:** hHGTN — Heterogeneous Hypergraph Temporal Network
**Purpose:** Add temporal, leakage-safe, and GPU-scalable sampling to the hypergraph fraud model. This document is the canonical, agent-facing reference. **The agent MUST consult this file before every code edit or design decision.**

---

## QUICK RULES (Planner-mode directives)

1. **Before every action**: open this file and quote the relevant section(s) you are following (e.g., “I will implement Time-Relaxed Neighbor Sampling per §PHASE\_A.2 and use the GPU sampler wrapper per §PHASE\_B.3”).
2. **Phase-first**: Implement exactly one phase at a time (Phase A → B → C → D → E). Do not start Phase N+1 until Phase N validation tests pass.
3. **Keep reproducibility**: include seeds, save intermediate artifacts, and store metrics in `experiments/stage6/<run-id>/`.
4. **No hardcoded data**: read config values from `configs/stage6.yaml`.
5. **If unsure**: default to conservative numerical values (α=0.05, sample size small, use sparse ops).

---

## TABLE OF CONTENTS

* PHASE A — TDGNN (Time-Relaxed Sampling & Temporal GNN changes)
* PHASE B — G-SAMPLER (GPU Sampling Infrastructure)
* PHASE C — Integration: TDGNN + G-SAMPLER into training harness
* PHASE D — Experiments, Ablations & Benchmarks
* PHASE E — Production / Deployment guidance and checklists
* APPENDIX: Code skeletons, config file, tests, debug prints

---

# PHASE A — TDGNN (Time-Relaxed Sampling for Timestamped Directed GNNs)

**Goal:** Implement time-relaxed neighborhood sampling that (a) respects causality (no future leakage), (b) relaxes strict time windows for robustness (Δt window), and (c) integrates with temporal message passing (TGN-like memory module if needed).

### A.1 — Concepts & constraints

* **Graph elements**: edges are timestamped directed transactions (src → dst at time t).
* **Causality rule**: when training/evaluating at time `t_eval`, the model must not use edges with timestamp > `t_eval`.
* **Time-relaxed window Δt**: instead of strict `< t_eval`, allow `[t_eval - Δt_relax, t_eval]` during training sampling to increase robustness; ensure label leakage is prevented if Δt would include edges directly connected to the target transaction that reveal label.
* **SpotTarget discipline**: When a target transaction's label could be leaked via immediate future edges, exclude those edges from training split (see Stage5 SpotTarget practice).

### A.2 — Time-Relaxed Neighbor Sampling (algorithm)

**Function:** `sample_time_relaxed_neighbors(node_ids, t_eval, depth, fanouts, delta_t)`

Pseudocode (must be implemented exactly as shown, with vectorized ops where possible):

```python
def sample_time_relaxed_neighbors(node_ids, t_eval, depth, fanouts, delta_t, Batches):
    # node_ids: seed nodes to sample neighborhoods for (could be transaction nodes)
    # t_eval: evaluation timestamp (scalar or per-seed array)
    # depth: number of hops
    # fanouts: [f1, f2, ...] neighbors per hop
    # delta_t: time relaxation value (>=0)
    # Batches: data structure mapping node -> out edges with timestamps (sorted descending or ascending)
    sampled_nodes = set(node_ids)
    frontier = set(node_ids)
    for hop in range(depth):
        next_frontier = set()
        f = fanouts[hop]
        for u in frontier:
            # Candidate neighbors: neighbors v with timestamp <= t_eval(u) and timestamp >= t_eval(u) - delta_t
            # Use Batches[u] which is pre-sorted by timestamp decreasing; binary-search cut-off indices
            candidates = get_neighbors_in_time_range(Batches[u], t_end=t_eval[u], t_start=t_eval[u]-delta_t)
            # If len(candidates) <= f: take all; else sample = top-k by recency if prefer recency else random sample
            neighbors = sample_candidates(candidates, k=f, strategy='recency_or_random')
            next_frontier.update(neighbors)
        sampled_nodes.update(next_frontier)
        frontier = next_frontier
    return induced_subgraph(sampled_nodes)
```

**Implementation notes:**

* Pre-sort adjacency lists by timestamp for each node to allow O(log n) time-window extraction (binary search).
* `t_eval` may be per-seed: when training on mini-batches, maintain array of `t_eval` indexed by seed node.
* If directed graph: sample outgoing or incoming neighbors according to model design. For fraud detection, sample both incoming and outgoing to detect both sending and receiving behaviors.
* `strategy`: prefer `recency` (most recent edges within Δt). If degrees are skewed, use prioritized reservoir sampling for fairness.

### A.3 — TDGNN model hooks

* **Memory module (optional)**: if implementing TGN-style temporal memory, provide API:

  * `memory.update(node_ids, messages, timestamps)`
  * `memory.query(node_ids, timestamp)` returns last-state before timestamp.
* **Layer interface**: The GNN layer should accept `t_eval` and `mask_future=True` flags and must rely on sampled subgraph already filtered by timestamps.

### A.4 — Hyperparameters (defaults)

* `Δt_relax = 3600*24` (1 day) — conservative default for transaction data; tune later.
* `fanouts = [15, 10]` for 2-hop sampling initial tests.
* `depth = 2`
* `sampling_strategy = 'recency'`
* `alpha_time = 0.05` (if we apply any time-dependent scaling)

### A.5 — Tests (TDGNN-specific)

* `test_time_window_filtering`: sample with `t_eval`, assert all sampled\_edge.timestamps <= t\_eval and >= t\_eval - delta\_t.
* `test_no_leakage`: for seeded target transaction, ensure training mask edges are not included.
* `test_frontier_size`: ensure frontier size <= sum(fanouts).

---

# PHASE B — G-SAMPLER (GPU-Based Graph Sampling)

**Goal:** Implement GPU-native, high-throughput sampling building blocks as described in the GSampler paper and expose a Python wrapper compatible with PyG/DGL training loops.

### B.1 — High-level design

* **Kernel model**: Warp-centric parallel kernels that sample neighbors per node in parallel.
* **Data layout**: graph adjacency stored in CSR-like arrays on GPU: `indptr` (n+1), `indices` (nnz), `timestamps` (aligned), `edge_attrs` optional.
* **APIs (Python wrapper)**:

  * `GSampler(graph_gpu, strategy, fanouts, sample_seed, device)` returns `SubgraphBatch(seed_nodes, sub_indptr, sub_indices, mapping, edge_timestamps)`
  * `sample_time_relaxed(seed_nodes, t_eval_array, fanouts, delta_t)` — integrates TDGNN sampling directly into GPU kernels (preferred).

### B.2 — Kernel primitives to implement (order matters)

1. **time\_window\_cutoff\_kernel**:

   * Input: `indptr, indices, timestamps, seed_nodes, t_eval_array, delta_t`
   * Output: start\_idx, end\_idx per seed neighbor list (index ranges within indices where timestamps in `[t_eval-delta_t, t_eval]`)
   * Implementation: binary-search-per-list in parallel (use warp-level binary search)
2. **sample\_k\_from\_range\_kernel**:

   * Input: index ranges, k (fanout), strategy (top-k by recency or reservoir/random)
   * Output: sampled neighbor indices
   * Implementation: if `k` >= range length: copy all; else warp-scan & preferential sampling
3. **compact\_subgraph\_kernel**:

   * Build subgraph CSR for the union of sampled nodes; write sub\_indptr, sub\_indices
4. **parallel\_map\_kernel**:

   * Map subgraph node ids to contiguous ids in batch (reverse mapping for loss/metrics)

### B.3 — Python wrapper and API

```python
class GSampler:
    def __init__(self, csr_indptr, csr_indices, csr_timestamps, device='cuda:0'):
        # copy compressed arrays to device
    def sample_time_relaxed(self, seed_nodes, t_eval_array, fanouts, delta_t, strategy='recency'):
        # 1) run time_window_cutoff_kernel
        # 2) run sample_k_from_range_kernel (per-hop iterative)
        # 3) run compact_subgraph_kernel
        # 4) return SubgraphBatch object with tensors on GPU
```

### B.4 — Important GPU engineering notes

* **Memory pool**: pre-allocate buffers for largest expected batch to avoid repeated allocations.
* **Stream overlap**: sample on stream 1 while compute runs on stream 0 to overlap.
* **Atomic ops**: reduce atomics by warp-reduce and prefix-sum aggregation.
* **Fallbacks**: if GPU memory short, perform sampling in CPU streamed mode (`cpu_streaming=True`).

### B.5 — Tests and performance checks

* `test_sampling_correctness`: validate GPU sampling output equals CPU reference for small graphs.
* `test_throughput`: measure batches/sec and compare to CPU sampler.
* `memory_safety_check`: ensure `torch.cuda.memory_reserved()` below threshold during sampling.

---

# PHASE C — Integration: TDGNN + G-SAMPLER in training harness

**Goal:** Combine Phase A + B so training receives mini-batches that are **time-safe** and **GPU-resident**, then forward/backprop without extra CPU<->GPU copies.

### C.1 — Data flow (exact)

1. Seed nodes selected (e.g., batch of labeled transaction nodes).
2. Agent computes `t_eval_array` for each seed (the transaction timestamps).
3. Call `GSampler.sample_time_relaxed(seed_nodes, t_eval_array, fanouts, delta_t)`.
4. Receive `SubgraphBatch` on GPU with node features, edge timestamps, edge attributes.
5. Convert subgraph into PyG/DGL input:

   * If using hypergraph pipeline from Stage5, build induced hypergraph incidence submatrix B\_sub (if necessary) from sampled nodes on GPU.
6. Forward pass: `Y = HypergraphNN(subgraph_hypergraph_data)` with `t_eval` awareness if needed.
7. Loss computed on seeds; backward; optimizer.step().

### C.2 — Modifications to training loop (must match exactly)

* Replace existing CPU sampler with:

```python
for epoch in epochs:
    for seed_nodes, t_eval_array, labels in train_loader:
        sub = gsampler.sample_time_relaxed(seed_nodes, t_eval_array, fanouts, delta_t)
        logits = model(sub.hypergraph_data)    # model expects data on GPU
        loss = criterion(logits[seed_mask], labels)
        loss.backward(); optimizer.step(); optimizer.zero_grad()
```

* Ensure `train_loader` yields seed nodes + timestamps only. All heavy subgraph creation is done inside `GSampler` on GPU.

### C.3 — Checkpointing & reproducibility

* Save `configs/stage6.yaml` with sampling seeds, fanouts, delta\_t, random seeds.
* When resuming, ensure the same random seed and same snapshot of the graph is used.

---

# PHASE D — Experiments, Ablations & Benchmarks

**Goal:** Validate correctness, reproduce expected speedups, and verify accuracy improvements on fraud detection.

### D.1 — Core experiments

* Compare four pipelines:

  1. Baseline GCN/RGCN on clique/star expansion (Stage4 baseline).
  2. Stage5 PhenomNN hypergraph model w/o sampling (full-batch small graph).
  3. Stage5 + TDGNN sampling but CPU sampler.
  4. Stage5 + TDGNN + GSampler (GPU time-relaxed sampling).

Measure:

* AUC, PR-AUC, F1 (fraud is imbalanced — focus on recall @ low FPR and PR-AUC).
* Throughput (batches/sec), latency per batch.
* VRAM usage.

### D.2 — Ablations (must include)

* `Δt_relax`: {0 (strict), 1 hour, 1 day, 7 days}
* Sampling fanouts: vary `[10,5]`, `[20,10]`, `[50,25]`
* Strategies: `recency`, `random`, `importance` (edge-weighted)
* GSampler kernel variants: warp-centric vs naive parallel — compare occupancy.

### D.3 — Expected baselines from papers (for sanity)

* GSampler claims average speedup 1.14–32.7× (hardware dependent). Aim for *≥10×* vs CPU sampler for medium graphs.
* TDGNN indicates improved robustness and generalization via time-relaxed sampling; expect recall/F1 improvements if Δt tuned.

### D.4 — Reporting

* Save per-run metrics to `experiments/stage6/<run-id>/metrics.json`
* Produce plots: `throughput_vs_batchsize.png`, `auc_vs_delta_t.png`, `memory_vs_model.png`.

---

# PHASE E — Production / Deployment & Guardrails

### E.1 — Production checklist

* Use pre-allocated memory pools and pinned memory for host<->device transfers.
* Limit per-batch VRAM budget and fallback to streamed CPU+GPU hybrid when exceeded.
* Expose config knobs `MAX_BATCH_NODES`, `GPU_MEM_LIMIT_PERCENT`.

### E.2 — Safety checks (fraud ML guardrails)

* Validate no test-time future edges are used.
* Log for each flagged transaction the highest-contributing sampled edges (for explainability).
* Keep an audit log of sampling seeds and timestamps per training epoch for traceability.

---

# APPENDIX — Code skeletons & files to create (copy-paste ready)

**Required file structure**

```
src/
  sampling/
    __init__.py
    gsampler.py            # Python wrapper
    kernels/               
      kernels.cu           # CUDA kernels (time window cutoff, sample_k, compact)
      build.sh             # compilation script for kernels
    cpu_fallback.py
    utils.py               # sparse helpers
configs/
  stage6.yaml
src/models/
  hypergraph/             # (from Stage5)
  tdgnn_wrapper.py        # thin wrapper to accept sampled subgraph
notebooks/
  stage6_experiments.ipynb
experiments/
  stage6/
```

**Minimal `gsampler.py` skeleton**

```python
# src/sampling/gsampler.py
import torch
class GSampler:
    def __init__(self, indptr, indices, timestamps, device='cuda:0'):
        # copy to device, create torch tensors
        self.device = device
        self.indptr = indptr.to(device)
        self.indices = indices.to(device)
        self.timestamps = timestamps.to(device)

    def sample_time_relaxed(self, seed_nodes, t_eval_array, fanouts, delta_t, strategy='recency'):
        # 1) call kernel: time_window_cutoff_kernel(...)
        # 2) for hop in H: call sample_k_from_range_kernel(...)
        # 3) compact subgraph and return SubgraphBatch
        raise NotImplementedError("Implement with CUDA kernels per reference")
```

**Minimal `tdgnn_wrapper.py`**

```python
# src/models/tdgnn_wrapper.py
def train_epoch(model, gsampler, train_seed_loader, optimizer, criterion, cfg):
    model.train()
    for seed_nodes, t_evals, labels in train_seed_loader:
        sub = gsampler.sample_time_relaxed(seed_nodes, t_evals, cfg.fanouts, cfg.delta_t)
        logits = model(sub.hypergraph_data)
        loss = criterion(logits[sub.train_mask], labels)
        loss.backward(); optimizer.step(); optimizer.zero_grad()
```

---

# UNIT TESTS (MANDATORY — agent must add)

* `tests/test_time_relaxed_sampler.py`
* `tests/test_gsampler_gpu_cpu_consistency.py`
* `tests/test_integration_tdgnn_gsampler.py` (small synthetic graph, train 1 epoch no error)

Each test must run in CI in lite-mode and be fast (<30s).

---

# DEBUG PRINTS (must be included in dev mode)

```python
print(f"[Stage6] seed_nodes={len(seed_nodes)} t_eval_min={t_eval_array.min()} max={t_eval_array.max()}")
print(f"[GSAMPLER] sampled_nodes={sub.num_nodes} sampled_edges={sub.num_edges}")
print(f"[TDGNN] max_frontier={max_frontier_size} avg_frontier={avg_frontier_size}")
torch.cuda.synchronize()
print(f"[MEM] allocated={torch.cuda.memory_allocated()/1e9:.3f}GB reserved={torch.cuda.memory_reserved()/1e9:.3f}GB")
```

---

# AGENT COMMUNICATION PROTOCOL (MANDATORY)

* **Before coding**: Post a short plan message in chat:
  Example:

  > “Begin Phase A.2: implementing `sample_time_relaxed_neighbors` using adjacency binary-search per seed (§PHASE\_A.2). Plan: 1) implement sorted per-node adjacency with timestamps, 2) implement CPU reference function and unit test, 3) adapt to GPU via GSampler wrapper.”
  > Include referenced section in brackets (e.g., `§PHASE_A.2`).

* **During coding**: Post progress updates referencing sections used and immediate blockers.

* **After phase**: Run unit tests; post results; if passed, declare phase complete and ask approval to continue.

---

# CONFIG TEMPLATE (`configs/stage6.yaml`)

```yaml
stage6:
  delta_t: 86400        # 1 day in seconds
  fanouts: [15, 10]
  depth: 2
  sampling_strategy: recency
  alpha_time: 0.05
  gs_gpu_device: cuda:0
  gs_max_batch_nodes: 10000
  seed: 42
```

---

# FINAL NOTES (agent must obey)

* This document is authoritative for **Stage 6**. **Every code change** must be justified by referencing a section here.
* If a research detail is ambiguous, favor numerical safety and add a TODO comment linking the paper section (e.g., `# TODO: verify per Prop 5.1`).
* Preserve reproducibility: save every run’s `configs/stage6.yaml` and random seeds in experiment folder.

---

