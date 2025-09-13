# hHGTN Integration Plan for 4DBInfer Framework

**Document**: `experiments/4dbinfer/hhgt_integration_plan.md`  
**Date**: September 13, 2025 (Updated after Phase A completion)  
**Purpose**: Comprehensive model interface mapping for hHGTN → 4DBInfer integration

## 1. Model Interface Analysis (COMPLETED ✅)

### 1.1 Required Interface (from `dbinfer/solutions/base_gml_solution.py`)

**Base Class**: `BaseGMLSolution` 
**Pattern**: All GNN models must inherit from `BaseGMLSolution` and implement:

```python
class BaseGMLSolution(GraphMLSolution):
    config_class = <SolutionConfig>  # REQUIRED
    name = "<model_name>"           # REQUIRED - CLI identifier
    
    def __init__(self, solution_config: BaseGNNSolutionConfig, data_config: GraphDatasetConfig)
    def create_model(self) -> nn.Module  # REQUIRED - returns BaseGNN subclass
    def fit(self, dataset, task_name, ckpt_path, device) -> FitSummary
    def evaluate(self, item_set_dict, graph, feat_store, device) -> float
    def checkpoint(self, ckpt_path: Path) -> None
    def load_from_checkpoint(self, ckpt_path: Path) -> None
```

### 1.2 Model Registration Pattern (from `dbinfer/solutions/sage.py`)

```python
@gml_solution  # REQUIRED decorator for registration
class HHGTSolution(BaseGMLSolution):
    config_class = HHGTSolutionConfig  # Links to config class
    name = "hhgt"                      # CLI command identifier
    
    def create_model(self):
        return HHGT(self.solution_config, self.data_config)
```

### 1.3 Config Class Pattern (from `BaseGNNSolutionConfig`)

```python
class HHGTSolutionConfig(BaseGNNSolutionConfig):
    # Inherited from BaseGNNSolutionConfig:
    lr: float = 0.01                    # Learning rate
    batch_size: int = 256               # Training batch size
    eval_batch_size: int = 256          # Evaluation batch size
    fanouts: List[int] = [10, 10]       # Neighbor sampling fanouts
    epochs: int = 200                   # Training epochs
    
    # hHGTN-specific parameters:
    hid_size: int = 64                  # Hidden dimension
    dropout: float = 0.1                # Dropout rate
    
    # Ablation control flags:
    spot_target_enabled: bool = True    # SpotTarget attention mechanism
    cusp_enabled: bool = True           # CUSP structural embeddings  
    trd_enabled: bool = True            # TRD/G-Sampler temporal sampling
    memory_enabled: bool = True         # Memory/TGN temporal modeling
```
```python
class HHGTSolution(BaseGMLSolution):
    config_class = HHGTSolutionConfig
    name = "hhgt"
    
    def create_model(self) -> nn.Module:
        return HHGT(self.solution_config, self.data_config)
```

#### B. Model Level (`BaseGNN` inheritance):
```python
class HHGT(BaseGNN):
    def create_gnn(
        self,
        node_feat_size_dict: Dict[str, int],
        edge_feat_size_dict: Dict[str, int], 
        seed_feat_size: int,
        out_size: Optional[int],
    ) -> nn.Module:
        # Return HeteroHHGT wrapper
```

## 2. Data Format Mapping

### Input Data Flow:
```
minibatch (gb.MiniBatch) 
  ↓
node_feat_dict: Dict[NType, Dict[str, torch.Tensor]]
edge_feat_dicts: List[Dict[EType, Dict[str, torch.Tensor]]]
input_node_id_dict: Dict[NType, torch.Tensor]
mfgs: List[DGLGraph] (message flow graphs)
seed_feat_dict: Dict[str, Dict[str, torch.Tensor]]
```

### Key Data Transformations Needed:
1. **Hypergraph Construction**: Convert DGL heterogeneous graphs to hypergraph format
2. **Temporal Features**: Extract timestamps from TIMESTAMP_FEATURE_NAME
3. **Node/Edge Features**: Map to hHGTN input format
4. **Seed Context**: Handle seed feature encoding

## 3. Configuration Schema

### Required Config Classes:
```python
class HHGTSolutionConfig(BaseGNNSolutionConfig):
    hid_size: int
    dropout: float
    # hHGTN specific configs
    spot_target_enabled: bool = True
    cusp_enabled: bool = True
    trd_enabled: bool = True
    memory_enabled: bool = True
    num_hyperedges: int = 64
    # DGL adapter configs
    conv: HHGTConvConfig = HHGTConvConfig()
```

## 4. Forward Pass Signature

### Expected Input/Output:
```python
def forward(
    self,
    mfgs: List[DGLGraph],           # Message flow graphs  
    node_feat_dict: Dict[str, Dict[str, torch.Tensor]],
    input_node_id_dict: Dict[str, torch.Tensor],
    edge_feat_dicts: List[Dict[str, Dict[str, torch.Tensor]]],
    seed_feat_dict: Dict[str, Dict[str, torch.Tensor]],
    seed_lookup_idx: torch.Tensor
) -> torch.Tensor:
    # Returns prediction logits
```

## 5. Integration Points for hHGTN Features

### A. SpotTarget Integration:
- Map from `seed_feat_dict` and `seed_lookup_idx`
- Configure via `spot_target_enabled` config flag

### B. CUSP Embeddings:
- Extract from `node_feat_dict` structural features  
- Enable/disable via `cusp_enabled` config

### C. TRD/G-Sampler:
- Use DGL subgraph sampling from `mfgs`
- Control via `trd_enabled` config

### D. Memory/TGN Components:
- Leverage temporal features from TIMESTAMP_FEATURE_NAME
- Configure via `memory_enabled` config

## 6. Adapter Strategy

### Core Adaptation Layer:
```python
class DGLToHypergraphAdapter:
    def convert_mfgs_to_hypergraph(self, mfgs):
        # Convert DGL message flow graphs to hypergraph representation
        
    def extract_temporal_features(self, node_feat_dict, edge_feat_dicts):
        # Extract timestamp information for TGN components
        
    def prepare_hhgt_input(self, minibatch_data):
        # Transform 4DBInfer format to hHGTN format
```

## 7. Dataset Format Requirements

From `dbinfer_bench.graph_dataset`:
- `DBBGraphDataset`: Contains graph, features, tasks
- `gb.sampling_graph.SamplingGraph`: DGL GraphBolt sampling graph
- `gb.FeatureStore`: Feature storage backend
- `gb.ItemSetDict`: Training/validation/test splits

### Key Attributes to Map:
- `data_config.graph.ntypes`: Node types
- `data_config.graph.etypes`: Edge types  
- `data_config.task.seed_type`: Target node type
- `data_config.task.target_type`: Prediction target type

## 8. Metrics & Evaluation

### Required Metrics Interface:
```python
def evaluate(
    self,
    item_set_dict: gb.ItemSetDict,
    graph: gb.sampling_graph.SamplingGraph,
    feat_store: gb.FeatureStore,
    device: DeviceInfo,
) -> float:
    # Return evaluation metric (AUC, F1, etc.)
```

## 9. Checkpointing & Serialization

### Required Methods:
```python
def checkpoint(self, ckpt_path: Path) -> None:
    torch.save(self.model.state_dict(), ckpt_path / 'model.pt')
    yaml_utils.save_pyd(self.solution_config, ckpt_path / 'solution_config.yaml')
    yaml_utils.save_pyd(self.data_config, ckpt_path / 'data_config.yaml')

def load_from_checkpoint(self, ckpt_path: Path) -> None:
    # Load model state and configs
```

## 10. Registration Pattern

### Required Decorator Usage:
```python
@gml_solution  # Registers with _GML_SOLUTION_REGISTRY
class HHGTSolution(BaseGMLSolution):
    config_class = HHGTSolutionConfig
    name = "hhgt"  # CLI name for dbinfer fit-gml ... hhgt
```

## Next Steps for Phase C:
1. Implement `DGLToHypergraphAdapter` class
2. Create `HHGTSolutionConfig` with ablation flags
3. Wrap existing hHGTN model with 4DBInfer interface
4. Test with synthetic data first, then real datasets
