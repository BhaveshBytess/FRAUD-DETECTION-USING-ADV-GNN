# FRAUD-DETECTION-USING-ADV-GNN (HHGTN Project)

This project is a template for building Heterogeneous Graph Transformer Networks for fraud detection.

## 🎯 Project Status - Stage 4 COMPLETE ✅

### ✅ Completed Stages:
- **Stage 0**: Data Exploration & Setup ✅
- **Stage 1**: Basic GNN Models (GCN, GraphSAGE) ✅
- **Stage 2**: Advanced GNN Methods ✅  
- **Stage 3**: **Heterogeneous Models (HAN, R-GCN) - AUC: 0.876** ✅
- **Stage 4**: **Temporal Modeling (Memory-based TGNNs)** ✅

### 🚀 Current Achievement - Stage 4:
- ✅ **TGN Implementation**: Complete temporal graph networks with memory modules
- ✅ **TGAT Model**: Time-aware graph attention with temporal encoding
- ✅ **Temporal Sampling**: Time-ordered event processing with causal constraints
- ✅ **Memory Visualization**: Comprehensive memory state tracking and analysis
- ✅ **Production Ready**: Optimized for 8GB RAM with robust error handling
- ✅ **Complete Integration**: Full fraud detection pipeline with temporal modeling

### 🎯 Ready for Next Stage:
- **Stage 5**: Advanced Architectures (GraphSAINT, FastGCN) 🔄

### 🎯 Project Roadmap (Stages 5-14):
- Stage 5: Advanced Architectures & Sampling Techniques
- Stage 6: Multi-scale Analysis & Optimization
- Stage 7: Ensemble Methods & Model Fusion
- Stages 8-14: Production, Deployment, and Advanced Features

## Data

* The full dataset (e.g., Elliptic++) should be placed in `data/ellipticpp/`.
* Sample data for testing is located in `data/ellipticpp_sample/`.

## Setup

1.  Create a virtual environment: `python -m venv .venv`
2.  Activate it: `.venv\Scripts\activate` (on Windows)
3.  Install dependencies: `pip install -r requirements.txt`
4.  Install PyTorch Geometric dependencies: `pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.8.0+cpu.html`

## Usage

To process the sample Elliptic++ data:
`python src/load_ellipticpp.py --path data/ellipticpp_sample --out data/ellipticpp.pt --sample_n 1000`

## 🏗️ Architecture Overview

### Stage 4 - Temporal Modeling **COMPLETED**:
- ✅ **TGN (Temporal Graph Network)**: Complete memory-based temporal graph networks
- ✅ **TGAT (Temporal Graph Attention)**: Time-aware attention with temporal encoding
- ✅ **Temporal Sampling**: Time-ordered event processing with causal constraints
- ✅ **Memory Visualization**: Comprehensive memory state tracking and analysis
- ✅ **Production Ready**: Optimized for 8GB RAM with robust error handling

### Stage 3 - Heterogeneous Models **COMPLETED**:
- ✅ **HAN (Heterogeneous Attention Network)**: AUC = 0.876, PR-AUC = 0.979, F1 = 0.956
- ✅ **R-GCN Implementation**: Stable relational graph modeling
- ✅ **Multi-node-type Graphs**: Transaction + Wallet node handling
- ✅ **Attention Mechanisms**: Node-level and semantic-level attention
- ✅ **Production Infrastructure**: Robust error handling and deployment-ready
- ✅ **Performance Target**: Exceeded AUC > 0.87 requirement

```
Stage 4 Temporal System:
├── TGN/TGAT Models
│   ├── Memory modules with GRU/LSTM updaters
│   ├── Message aggregation with attention
│   ├── Temporal embedding and memory updates
│   └── Time-aware attention mechanisms
├── Temporal Sampling
│   ├── Time-ordered event loading
│   ├── Temporal neighbor sampling
│   ├── Causal constraint enforcement
│   └── Efficient batch processing
├── Memory Visualization
│   ├── Memory state evolution tracking
│   ├── Distribution analysis and plotting
│   ├── Interaction impact visualization
│   └── 3D interactive memory exploration
└── Integration Pipeline
    ├── Fraud detection pipeline integration
    ├── Performance optimization for 8GB RAM
    └── Comprehensive testing and validation

Stage 3 Heterogeneous System:
├── Data Preprocessing
│   ├── NaN value remediation (16,405 → 0)
│   ├── Extreme outlier clipping (percentile-based)
│   └── Robust standardization
├── Advanced Models
│   ├── HAN (Heterogeneous Attention Network)
│   ├── R-GCN (Relational Graph Convolutional Network)
│   └── Multi-type node and edge modeling
└── Training Pipeline
    ├── Gradient clipping
    ├── Class balancing
    └── Early stopping
```

## 🚀 Quick Start

### Basic Setup
1. Create virtual environment: `python -m venv .venv`
2. Activate: `.venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Install PyG: `pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.8.0+cpu.html`

### Stage 5 - Advanced Architectures (Latest)
```bash
# Quick demonstration of all Stage 5 models
python stage5_main.py --mode demo

# Run comprehensive benchmark (all models)
python stage5_main.py --mode benchmark

# Train specific model
python stage5_main.py --mode train --model graph_transformer

# Quick testing mode
python stage5_main.py --mode benchmark --quick
```

### Run Stage 4 Temporal Modeling ✅ COMPLETE
```bash
# Interactive Jupyter notebook (recommended) - Full TGN/TGAT implementation
jupyter notebook notebooks/stage4_temporal.ipynb

# Command line training with TGN models
python src/train_baseline.py --config configs/temporal.yaml

# Test TGN/TGAT implementations
python src/models/tgn.py
python src/temporal_sampling.py
python src/memory_visualization.py
```

### Run Previous Stages
```bash
# Stage 1-2: Basic models
python src/train_baseline.py --config configs/baseline.yaml

# Stage 3: HAN baseline (AUC: 0.876)
python src/train_baseline.py --config configs/han.yaml
```

## 📁 Project Structure

```
hhgtn-project/
├── src/
│   ├── models/
│   │   ├── han.py                      # Heterogeneous Attention Network (Stage 3)
│   │   ├── temporal_stable.py          # Temporal models with stability (Stage 4)
│   │   └── advanced/                   # Stage 5 Advanced Architectures
│   │       ├── graph_transformer.py    # Graph Transformer
│   │       ├── hetero_graph_transformer.py # Heterogeneous Graph Transformer
│   │       ├── temporal_graph_transformer.py # Temporal Graph Transformer
│   │       ├── ensemble.py             # Advanced Ensemble Methods
│   │       ├── training.py             # Stage 5 Training Pipeline
│   │       └── evaluation.py           # Comprehensive Evaluation Framework
│   ├── config.py                       # Configuration management
│   ├── data_utils.py                   # Data processing utilities
│   ├── load_elliptic.py                # Elliptic dataset loader
│   ├── load_ellipticpp.py              # EllipticPP dataset loader
│   ├── metrics.py                      # Evaluation metrics
│   ├── model.py                        # Base model definitions
│   ├── train_baseline.py               # Training pipeline
│   └── utils.py                        # General utilities
├── configs/
│   ├── baseline.yaml                   # Basic model configurations
│   ├── stage5/                         # Stage 5 Model Configurations
│   │   ├── graph_transformer.yaml      # Graph Transformer config
│   │   ├── hetero_graph_transformer.yaml # HGTN config
│   │   ├── temporal_graph_transformer.yaml # TGT config
│   │   └── ensemble.yaml               # Ensemble config
│   └── stage5_benchmark.yaml           # Comprehensive benchmark config
├── data/
│   ├── ellipticpp/                     # Full dataset
│   └── ellipticpp_sample/              # Sample data for testing
├── experiments/                        # Training results & benchmarks
├── notebooks/                          # Interactive analysis
├── tests/                              # Unit tests
├── run_stage5_benchmark.py             # Stage 5 Benchmark Runner
└── stage5_main.py                      # Main Stage 5 Entry Point
```

## 🔬 Technical Highlights

### Stage 5 Innovation:
- **Graph Transformer**: Multi-head attention adapted for graph structures with positional encoding
- **HGTN**: Heterogeneous node/edge types with cross-type attention mechanisms
- **Temporal Graph Transformer**: Joint temporal-graph modeling with causal attention
- **Advanced Ensembles**: Learned weight combination with cross-validation selection
- **Unified Training**: Comprehensive pipeline supporting all Stage 5 architectures
- **Benchmarking Framework**: Complete evaluation and comparison system

### Stage 4 Innovation:
- **Data Quality**: Resolved dataset corruption (16K+ NaN values)
- **Numerical Stability**: Conservative initialization prevents gradient explosion
- **Temporal Processing**: Sequence-aware fraud detection
- **Memory Efficiency**: Optimized for 8GB RAM systems
- **Robust Training**: Handles class imbalance (2.2% fraud rate)

### Model Performance:
- **Stage 3 HAN Baseline**: 0.876 AUC
- **Stage 4 Focus**: Temporal modeling foundation (evaluation pipeline established)
- **Stage 5 Target**: State-of-the-art transformer architectures
- **System Stability**: 100% NaN issues resolved

## 🧪 Experiments

### Stage 5 - Advanced Architectures:
```bash
# Quick demo of all models
python stage5_main.py --mode demo

# Full benchmark comparison
python stage5_main.py --mode benchmark --config configs/stage5_benchmark.yaml

# Train individual models
python stage5_main.py --mode train --model graph_transformer
python stage5_main.py --mode train --model hetero_graph_transformer
python stage5_main.py --mode train --model temporal_graph_transformer
```

### Interactive Notebooks:
- `notebooks/stage3_han.ipynb` - HAN model analysis
- `notebooks/stage4_temporal.ipynb` - Temporal modeling experiments

### Command Line:
```bash
# Run all baselines
python src/train_baseline.py --config configs/baseline.yaml --epochs 50

# Test temporal models
python src/models/temporal_stable.py
```

## 📊 Data

* **Full Dataset**: Place Elliptic++ in `data/ellipticpp/`
* **Sample Data**: Testing data in `data/ellipticpp_sample/`
* **Processed**: Clean temporal data available after Stage 4 processing

## 🧪 Testing

```bash
# Run all tests
pytest

# Test specific components
pytest tests/test_temporal_models.py
pytest tests/test_data_loading.py
```

## 🎯 Next Steps (Stage 6)

- **Optimization Techniques**: Advanced hyperparameter tuning, neural architecture search
- **Model Compression**: Pruning, quantization, knowledge distillation
- **Efficiency Optimization**: Memory optimization, inference acceleration
- **Advanced Training**: Self-supervised learning, contrastive learning
- **Production Preparation**: Model optimization for deployment

