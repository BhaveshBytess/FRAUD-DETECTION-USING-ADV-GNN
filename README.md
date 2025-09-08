# FRAUD-DETECTION-USING-ADV-GNN (HHGTN Project)

A comprehensive 14-stage system for building Heterogeneous Graph Transformer Networks for fraud detection, featuring advanced temporal modeling and numerical stability.

## 🎯 Project Status - Stage 4 COMPLETED

### ✅ Completed Stages:
- **Stage 1**: Basic Models & Infrastructure ✅
- **Stage 2**: Graph Neural Networks ✅  
- **Stage 3**: Heterogeneous Graph Attention (HAN) - AUC: 0.876 ✅
- **Stage 4**: Temporal Modeling - Numerical Stability Resolved ✅

### 🔄 Current Stage:
- **Stage 5**: Advanced Architectures (Next)

### 🎯 Upcoming Stages (5-14):
- Stage 5: Advanced Architectures
- Stage 6: Optimization Techniques
- Stage 7: Multi-scale Analysis
- Stages 8-14: Production, Deployment, Monitoring

## 📊 Key Achievements

### Stage 4 - Temporal Modeling Breakthroughs:
- ✅ **Numerical Stability Resolved**: Fixed 16,405 NaN values in dataset
- ✅ **Robust Data Pipeline**: Implemented outlier clipping and normalization
- ✅ **Stable Model Architectures**: Created SimpleLSTM, SimpleGRU, SimpleTemporalMLP
- ✅ **Temporal Framework**: Established time-series processing infrastructure
- ✅ **Production Foundation**: Built stable training and evaluation systems

## 🏗️ Architecture Overview

```
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
├── data/
│   ├── ellipticpp/           # Full dataset
│   └── ellipticpp_sample/    # Sample data for testing
├── src/
│   ├── models/
│   │   ├── temporal.py       # Original temporal models
│   │   └── temporal_stable.py # Stage 4 stable models
│   ├── train_baseline.py     # Training scripts
│   └── load_ellipticpp.py    # Data loading
├── notebooks/
│   ├── stage3_han.ipynb      # Stage 3 HAN analysis
│   └── stage4_temporal.ipynb # Stage 4 temporal modeling
├── configs/
│   ├── baseline.yaml         # Basic model configs
│   └── temporal.yaml         # Temporal model configs
└── experiments/              # Training results
```

## 🔬 Technical Highlights

### Stage 4 Innovation:
- **Data Quality**: Resolved dataset corruption (16K+ NaN values)
- **Numerical Stability**: Conservative initialization prevents gradient explosion
- **Temporal Processing**: Sequence-aware fraud detection
- **Memory Efficiency**: Optimized for 8GB RAM systems
- **Robust Training**: Handles class imbalance (2.2% fraud rate)

### Model Performance:
- **Stage 3 HAN Baseline**: 0.876 AUC
- **Stage 4 Focus**: Temporal modeling foundation (evaluation pipeline established)
- **System Stability**: 100% NaN issues resolved

## 🧪 Experiments

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

## 🎯 Next Steps (Stage 5)

- Advanced attention mechanisms
- Multi-scale temporal analysis  
- Hybrid graph-temporal architectures
- Optimization and hyperparameter tuning
- Production deployment preparation

