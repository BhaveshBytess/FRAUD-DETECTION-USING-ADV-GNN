# STAGE 8 — CUSP Reference (Planner Mode)

**Project**: hHGTN — Stage 8 (CUSP: Curvature- and Spectrum-aware Product-manifold Filtering)

**Purpose of this reference:**
This single, unified reference file contains everything an agent or developer needs to implement CUSP ("Cusp Filtering + Curvature Encoding + Cusp Pooling") from the ICLR 2025 paper. It is written in planner/cheat-sheet style so an automated agent (or you) can follow it step-by-step.

> NOTE: This reference assumes Stage 6 (TDGNN + gSampler integration) and Stage 5 (PhenomNN hypergraph) artifacts are available. CUSP is a plug-in spectral/positional module that can be used as a backbone or head inside the hHGTN pipeline.

---

## Quick overview (one-paragraph)

CUSP introduces a curvature-informed Laplacian ("Cusp Laplacian") based on Ollivier–Ricci curvature (ORC), uses a GPR-based filter bank (multi-filter spectral propagation) to extract features across different parts of the spectrum, encodes curvature as positional features (functional curvature encoding), and fuses mixed-curvature signals in a product-manifold via an attention-driven pooling called *Cusp Pooling*. The module outputs a node embedding ζ(x) that augments or replaces standard GNN features.

---

## High-level Phases (what to implement, in order)

1. **Phase 1 — ORC & Cusp Laplacian**

   * Compute Ollivier–Ricci curvature (ORC) per edge and per node.
   * Build curvature-weighted adjacency Â and degree D̃, then the Cusp Laplacian L̃ = D̃ − Â and normalized Ã\_n = D̃^{−1/2} Â D̃^{−1/2}.
   * Provide robust numerical fallbacks (clipping, normalization) and sparse representations.

2. **Phase 2 — Cusp Filtering (GPR filter bank)**

   * Implement GPRGNN-style propagation on the curvature-normalized adjacency Ã\_n.
   * Build an L-filter bank (L small, e.g., 5–25) where each filter is a learnable polynomial / GPR weight vector.
   * For each manifold component q of the product manifold P, perform manifold-aware propagation (stereographic/möbius operations) and collect filter outputs.

3. **Phase 3 — Functional Curvature Encoding**

   * Produce curvature positional encodings Φ(x) (harmonic analysis based) per node.
   * Concatenate/augment curvature encodings with filter outputs to form component encodings.

4. **Phase 4 — Product Manifold Construction & Cusp Pooling**

   * Heuristically estimate product-manifold signature (curvature components H/S/E and their dimensions d(q)).
   * Lift Euclidean features into manifold components via exponential map and perform manifold-local operations.
   * Use hierarchical attention (Cusp Pooling) to compute attentional weights β\_q across components and attentively concatenate to final embedding ζ(x).

5. **Phase 5 — Integration, Ablations & Tests**

   * Integrate outputs into downstream head (node classification / link prediction / as positional features for hHGTN).
   * Implement ablations (replace Cusp Laplacian with standard Laplacian, single filter vs filter bank, remove curvature encoding/pooling) and reproduce paper checks.

---

## Core mathematical building blocks (implementation checklist)

* **Ollivier–Ricci curvature (ORC)** per edge (Wasserstein-1 between neighbor distributions). Provide function `compute_orc(graph, delta=0.2)` that returns edge\_curvatures and node\_curvatures.

* **Cusp Laplacian (Definition)**: implement weight w̄\_{xy} = exp( -1 / (1 - κ̃(x,y)) ) and define:

  * Ã\_{xy} = w̄\_{xy} \* A\_{xy}
  * D̃\_{xx} = ∑*y Ã*{xy}
  * L̃ = D̃ - Ã
  * Normalized adjacency: Ã\_n = D̃^{-1/2} Ã D̃^{-1/2}

* **GPR propagation** per manifold component (matrix form):

  * H^{(0)} = exp\_{κ(q)}^0( f\_θ(F) )  (lift base features to manifold)
  * H^{(l)} = Ã\_n ⊠\_{κ(q)} H^{(l-1)}  (κ-left-matrix multiplication)
  * Z^{(L)}*{M*{κ(q),d(q)}} = κ(q) ⊕*{l∈{0..L}} γ\_l ⊗*{κ(q)} H^{(l)}
  * Concatenate components with attentional concatenation to form Z^{(L)}\_{PdM}

* **Functional curvature encoding (Φ)**: harmonic analysis transform on ORC per node; implement `curvature_positional_encoding(orc_values, dC)`.

* **Cusp Pooling (attention)**:

  * Compute centroid µ(L) in tangent / log map for each component (Eq. 6).
  * Compute relative importance τ\_q and softmax weights β\_q (Eq. 7).
  * Attentional concatenation (Eq. 8) and final weighted sum across filter bank.

---

## Practical implementation plan (developer-friendly)

### File layout (suggested)

```
src/models/cusp/
├─ __init__.py
├─ cusp_core.py         # high-level CUSP class (CuspModule)
├─ cusp_orc.py          # ORC computation utilities
├─ cusp_laplacian.py    # build A_tilde, D_tilde, L_tilde, normalizations
├─ cusp_gpr.py          # GPR propagation & filter bank implementation
├─ cusp_manifold.py     # product-manifold utilities: exp, log, mobius ops
├─ cusp_encoding.py     # curvature positional encoding functions
├─ cusp_pooling.py      # Cusp Pooling (attention code)
└─ tests/
   ├─ test_orc.py
   ├─ test_laplacian.py
   ├─ test_gpr.py
   └─ test_integration.py
```

### Public API (what the agent should expose)

* `CuspModule(config)` — main module with `forward(node_features, edge_index, batch=None)` returning node embeddings.
* `compute_orc(graph, delta=0.2)` — returns edge\_orc, node\_orc
* `build_cusp_laplacian(edge_index, edge_orc)` — returns A\_tilde (sparse), D\_tilde (diag), A\_tilde\_n (sparse normalized)
* `gpr_filter_bank(A_tilde_n, X, L, gamma)` — returns list of filter outputs
* `estimate_product_manifold_signature(node_orc, dM, thresholds)` — returns list of components (κ, d)
* `curvature_positional_encoding(node_orc, dC)`
* `cusp_pooling(component_encodings, θ)` — returns pooled embedding

---

## Pseudocode: top-level forward pass

```py
def forward(X, edge_index):
    edge_orc, node_orc = compute_orc(graph)
    A_tilde, D_tilde, A_tilde_n = build_cusp_laplacian(edge_index, edge_orc)

    # per-component propagation
    components = estimate_product_manifold_signature(node_orc, dM)
    component_encodings = []
    for (kappa, dim) in components:
        H0 = exp_map(kappa, f_theta(X))
        H_filters = gpr_propagate(A_tilde_n, H0, L, gamma)
        Zk = combine_filters_on_manifold(H_filters, kappa)
        component_encodings.append(Zk)

    curvature_enc = curvature_positional_encoding(node_orc, dC)
    pooled = cusp_pooling(component_encodings, curvature_enc)
    final = sum_{l} epsilon_l * pooled_l  # weighted sum across filters
    return final  # (n_nodes x (dM + dC))
```

---

## Numerical & complexity notes (important)

* **ORC** computation is the heaviest step: requires computing local Wasserstein-1 distances between neighbor distributions. Use efficient approximations or GPU-accelerated OT libraries where possible; cache neighborhood distributions.
* **Sparse-friendly**: Keep Ã and Ã\_n in sparse (CSR/COO) format and leverage sparse-dense matmul.
* **Stability**: clip κ̃ values into \[-1+ε, 1-ε] for weight formula w̄ = exp(-1 / (1 - κ̃)). Add ε to denominators to avoid blow-ups.
* **Memory**: GPR with L filters needs L sparse matmuls; choose L between 5..25 depending on VRAM.

---

## Hyperparameters (paper defaults & grid hints)

* `L` (num filters): {5,10,15,20,25}
* `δ` (neighborhood mass param for ORC): {0.2,0.5,0.7}
* `α` (GPR init / propagation): {0.1,0.3,0.5,0.9}
* `dC` (curvature embedding dim): {8,16,32,64}
* `dM` (product manifold dim): {32,48,64,128,256}
* `dropout`: {0.2,0.3,0.5}
* `lr`: {1e-4,4e-3,1e-3,1e-2}
* `weight_decay`: {0,1e-4,5e-4}

---

## Validation checklist & unit tests (must pass)

**Phase 1**

* [ ] `compute_orc` returns finite values and bounded in \[-1,1] after normalization.
* [ ] `build_cusp_laplacian` produces A\_tilde with non-negative entries and D\_tilde diagonal > 0.

**Phase 2**

* [ ] GPR outputs shape matches (n, dM).
* [ ] Filters are stable and numerically finite for test graphs.

**Phase 3 & 4**

* [ ] `estimate_product_manifold_signature` partitions dims and sum(d(q)) == dM.
* [ ] `cusp_pooling` outputs shape (n, dM + dC).

**Integration**

* [ ] Replacing standard Laplacian with Cusp Laplacian yields non-negative improvement on heterophilic datasets in small smoke tests (paper sees \~1.5% avg).

---

## Ablations to run (automated experiment scripts)

* CUSPlap vs standard Laplacian
* filter bank (L>1) vs single filter
* with/without curvature positional encoding
* with/without Cusp Pooling (attention)
* vary L, dC, dM grid

---

## Integration notes for hHGTN

* Use CUSP as a **feature extractor** producing ζ(x). Then either:

  * Concatenate ζ(x) to node features before PhenomNN/PhenomNNSimple layers, or
  * Replace early GPR/GCN layers with CUSP module and feed hypergraph-specific features as base X.
* For link prediction: use pooled mixed-curvature embeddings and compute dot/MLP scores.

---

## Developer tips & warnings

* Precompute and persist ORC values for large graphs; recompute only when graph topology changes.
* When using Colab/Kaggle: test on small subgraphs first; ORC scales poorly on very dense graphs.
* Use sparse libraries (PyTorch sparse, torch-scatter where helpful) and consider approximate OT solvers for ORC.

---

## References & appendix pointers (paper sections to consult during implementation)

* Definition 1: Cusp Laplacian (Section 4.1)
* Cusp Filtering: GPR-based filter bank and propagation (Section 4.2)
* Functional Curvature Encoding (Section 4.3)
* Cusp Pooling: Eqns (6)-(8) and attention rollout (Section 4.4)
* Appendix: ORC details, product-manifold heuristics, GPR background, and hyperparameter tables.

---

*End of Stage 8 reference.*


