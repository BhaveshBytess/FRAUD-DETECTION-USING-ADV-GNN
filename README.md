# FRAUD-DETECTION-USING-ADV-GNN (HHGTN Project)

A comprehensive 14-stage system for building Heterogeneous Graph Transformer Networks for fraud detection, featuring advanced temporal modeling and numerical stability.

## 🎯 Project Status - Stage 5 COMPLETE ✅

### ✅ Completed Stages:
- **Stage 1**: Basic Models & Infrastructure ✅
- **Stage 2**: Graph Neural Networks ✅  
- **Stage 3**: Heterogeneous Graph Attention (HAN) - AUC: 0.876 ✅
- **Stage 4**: Temporal Modeling - **COMPLETE & STABLE** ✅
- **Stage 5**: Advanced Architectures - **COMPLETE** ✅

### 🚀 Stage 5 - Advanced Architectures **COMPLETED**:
- ✅ **Graph Transformer**: Multi-head attention with graph structure awareness
- ✅ **Heterogeneous Graph Transformer**: Multi-type node/edge modeling  
- ✅ **Temporal Graph Transformer**: Joint temporal-graph attention mechanisms
- ✅ **Advanced Ensemble System**: Learned weights, cross-validation, stacking
- ✅ **Comprehensive Training Pipeline**: Stage 5 unified training framework
- ✅ **Evaluation Framework**: Complete benchmarking and comparison system

### 🎯 Ready for Next Stage:
- **Stage 6**: Optimization Techniques - **READY TO BEGIN** 🚀

### 🎯 Upcoming Stages (6-14):
- Stage 6: Optimization Techniques
- Stage 7: Multi-scale Analysis
- Stages 8-14: Production, Deployment, Monitoring

## 📊 Key Achievements

### Stage 4 - Temporal Modeling **COMPLETED**:
- ✅ **Production-Ready Models**: SimpleLSTM, SimpleGRU, SimpleTemporalMLP all stable
- ✅ **Numerical Stability Achieved**: Zero NaN issues, gradient explosion resolved
- ✅ **Comprehensive Framework**: Complete temporal processing pipeline
- ✅ **Data Quality Assurance**: Robust 16,405 NaN remediation system
- ✅ **Training Infrastructure**: Memory-efficient, early stopping, class balancing
- ✅ **Ready for Evaluation**: All systems validated and production-ready

## 🏗️ Architecture Overview

```
Stage 5 Advanced Architectures:
├── Graph Transformer
│   ├── Multi-head attention with graph structure
│   ├── Positional encoding for nodes
│   └── 256 hidden dim, 6 layers, 8 heads
├── Heterogeneous Graph Transformer (HGTN)
│   ├── Multi-type node and edge modeling
│   ├── Cross-type attention mechanisms
│   └── Type-specific transformations
├── Temporal Graph Transformer
│   ├── Joint temporal-graph attention
│   ├── Causal temporal modeling
│   └── Spatio-temporal fusion
└── Advanced Ensemble System
    ├── Learned weight combination
    ├── Cross-validation selection
    └── Stacking meta-learners

Stage 4 Temporal System:
├── Data Preprocessing
│   ├── NaN value remediation (16,405 → 0)
│   ├── Extreme outlier clipping (percentile-based)
│   └── Robust standardization
├── Stable Models
│   ├── SimpleLSTM (64 hidden, conservative init)
│   ├── SimpleGRU (64 hidden, single layer)
│   └── SimpleTemporalMLP (128 hidden, avg pooling)
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

### Run Stage 4 Temporal Modeling
```bash
# Interactive Jupyter notebook (recommended)
jupyter notebook notebooks/stage4_temporal.ipynb

# Command line training
python src/train_baseline.py --config configs/temporal.yaml
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

