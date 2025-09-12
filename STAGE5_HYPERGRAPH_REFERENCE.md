# STAGE 5 REFERENCE: PhenomNN-Based Hypergraph Implementation

**PROJECT**: hHGTN (Heterogeneous Hypergraph Temporal Network)  
**STAGE**: 5 - Higher-Order / Hypergraph Modeling  
**PAPER**: "From Hypergraph Energy Functions to Hypergraph Neural Networks" (PhenomNN)  
**OBJECTIVE**: Model transactions as hyperedges linking multiple entities (user, merchant, device, IP)

---

## MANDATORY IMPLEMENTATION SEQUENCE

### PHASE 1: Core Hypergraph Infrastructure
**Priority**: CRITICAL - All subsequent phases depend on this

#### 1.1 Hypergraph Data Structure
```python
class HypergraphData:
    def __init__(self, incidence_matrix, node_features, hyperedge_features=None):
        self.B = incidence_matrix  # n x m binary matrix (nodes x hyperedges)
        self.X = node_features     # n x d node features  
        self.U = hyperedge_features # m x d edge features (optional)
```

#### 1.2 Required Matrix Computations (FROM PAPER)
- **Hyperedge degree matrix**: `DH = diag(B.sum(dim=0))` 
- **Clique expansion**: `AC = B @ inv(DH) @ B.T`, `DC = diag(AC.sum(dim=1))`
- **Star expansion**: `AS_bar = B @ inv(DH) @ B.T`, `DS_bar = diag(AS_bar.sum(dim=1))`

#### 1.3 Fraud-Specific Hyperedge Construction
```python
def construct_fraud_hyperedges(transactions_df):
    hyperedges = []
    # Multi-entity transaction hyperedges
    for _, txn in transactions_df.iterrows():
        entities = [f"user_{txn['user_id']}", f"merchant_{txn['merchant_id']}", 
                   f"device_{txn['device_id']}", f"ip_{txn['ip_address']}"]
        hyperedges.append(entities)
    return hyperedges
```

### PHASE 2: PhenomNN Layer Implementation
**Priority**: HIGH - Core innovation from paper

#### 2.1 Energy-Based Update Equations
**EQUATION 25 (Simplified)**: 
```
Y^(t+1) = ReLU((1-α)Y^(t) + αD̃^(-1)[(λ0*AC + λ1*AS_bar)*Y^(t) + f(X;W)])
```

**EQUATION 22 (General)**:
```
Y^(t+1) = ReLU((1-α)Y^(t) + αD̃^(-1)[f(X;W) + λ0*Ỹ_C^(t) + λ1*(L̄_S*Y^(t) + Ỹ_S^(t))])
```

#### 2.2 Critical Components
- **Preconditioner**: `D̃ = λ0*DC + λ1*DS_bar + I`
- **Step Size**: `α = 0.1` (default from paper)
- **Expansion Weights**: `λ0 = λ1 = 1.0` (balanced clique+star)

### PHASE 3: Model Architecture Integration
**Priority**: MEDIUM - Connects to existing pipeline

#### 3.1 Multi-Layer Architecture
```python
class HypergraphNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, lambda0=1.0, lambda1=1.0):
        # Stack multiple PhenomNN layers
        # Add classification head
```

#### 3.2 Data Loader Modification
- Extend existing loaders to construct hypergraphs
- Maintain compatibility with current pipeline
- Add hypergraph collation function

### PHASE 4: Training Integration
**Priority**: MEDIUM - Required for end-to-end training

#### 4.1 Modified Training Loop
```python
def train_with_hypergraph(model, dataloader, optimizer, criterion):
    # Check for hypergraph_data in batch
    # Forward pass with hypergraph layers
    # Standard backward pass
```

### PHASE 5: Ablation & Evaluation
**Priority**: LOW - For optimization and analysis

#### 5.1 Expansion Weight Ablation (Table 5 from paper)
- `{λ0: 0, λ1: 1}` - Star expansion only
- `{λ0: 1, λ1: 0}` - Clique expansion only  
- `{λ0: 1, λ1: 1}` - Combined (best performance)

---

## CRITICAL IMPLEMENTATION CONSTRAINTS

### Mathematical Accuracy
- **MUST** implement Equations 22 and 25 exactly as specified
- **MUST** compute degree matrices correctly (DC, DH, DS_bar)
- **MUST** respect convergence conditions from Propositions 5.1, 5.2

### Code Structure Requirements
```
src/models/hypergraph/
├── __init__.py
├── phenomnn.py          # Core layer implementations (Eq 22, 25)
├── hypergraph_data.py   # HypergraphData class
├── construction.py      # Fraud hyperedge construction
└── utils.py            # Matrix computation helpers

experiments/stage5/
├── ablation_hypergraph.py
├── benchmark_hypergraph.py
└── integration_test.py
```

### Performance Targets (From Paper)
- **Accuracy**: 2-5% improvement over GCN baseline
- **Memory**: 2-3x GCN overhead (acceptable)
- **Training Speed**: Comparable to GCN with 2-3 layers

---

## FRAUD DETECTION SPECIFIC REQUIREMENTS

### Hyperedge Types for Fraud
1. **Transaction Hyperedges**: Link [user, merchant, device, IP] per transaction
2. **Temporal Hyperedges**: Connect entities active in same time window
3. **Amount Pattern Hyperedges**: Group similar transaction amounts
4. **Behavioral Hyperedges**: Connect entities with similar activity patterns

### Expected Benefits
- Detect multi-entity collusion patterns
- Capture higher-order fraud rings
- Model complex entity interactions beyond pairwise relationships

---

## TESTING & VALIDATION CHECKLIST

### Phase 1 Validation
- [ ] Incidence matrix B has correct dimensions (n_nodes × n_hyperedges)
- [ ] Degree matrices sum correctly
- [ ] Hyperedge construction produces valid entity groupings

### Phase 2 Validation  
- [ ] Layer outputs have correct dimensions
- [ ] Gradient flow works through energy-based updates
- [ ] ReLU activation applied correctly

### Phase 3 Validation
- [ ] Multi-layer architecture trains without errors
- [ ] Memory usage within acceptable bounds
- [ ] Integration with existing data loaders works

### Phase 4 Validation
- [ ] Training loop handles both standard and hypergraph data
- [ ] Loss decreases over training epochs
- [ ] Evaluation metrics show improvement over baseline

### Phase 5 Validation
- [ ] Ablation studies reproduce paper findings
- [ ] Combined expansion (λ0=λ1=1) performs best
- [ ] Fraud detection metrics improve over Stage 4 results

---

## ERROR HANDLING & DEBUGGING

### Common Issues
1. **Singular Matrix Error**: Check DH has no zero diagonal elements
2. **Memory Overflow**: Use sparse matrices for large hypergraphs
3. **Gradient Explosion**: Verify step size α < convergence bound
4. **Dimension Mismatch**: Ensure compatibility matrices H0, H1 have correct size

### Debug Commands
```python
# Check hypergraph structure
print(f"Nodes: {B.shape[0]}, Hyperedges: {B.shape[1]}")
print(f"Max hyperedge size: {B.sum(dim=0).max()}")

# Verify degree matrices
assert torch.all(DH.diag() > 0), "Empty hyperedges detected"
assert torch.allclose(DC.diag(), AC.sum(dim=1)), "Degree matrix mismatch"
```

---

## PAPER REFERENCE POINTS

### Key Equations
- **Energy Function**: Equation (1) - General form
- **Simplified Energy**: Equation (9) - Practical implementation
- **Update Rules**: Equations (22, 25) - Core layer computations
- **Convergence**: Propositions (5.1, 5.2) - Step size bounds

### Experimental Results
- **Table 1**: Performance on citation networks
- **Table 2**: Results on visual datasets  
- **Table 5**: Expansion weight ablation
- **Section 7**: Time/space complexity analysis

### Implementation Notes
- **Section 5.1**: Proximal gradient descent derivation
- **Section 6**: Connection to existing GNN layers
- **Algorithm 1**: Overall training procedure