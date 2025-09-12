# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [6.0.0] - 2025-09-13

### Added - Stage 6: TDGNN + G-SAMPLER Implementation

#### 🚀 Major Features
- **TDGNN Framework**: Complete Timestamped Directed Graph Neural Networks implementation
- **G-SAMPLER**: GPU-native temporal neighbor sampling with CUDA kernel architecture
- **Temporal Graph Processing**: CSR format with precise timestamp indexing and binary search
- **Time-relaxed Sampling**: Configurable delta_t windows with multi-hop temporal constraints
- **Hypergraph Integration**: Seamless combination with Stage 5 hypergraph models

#### 🏗️ Core Components
- `src/sampling/cpu_fallback.py` - Pure PyTorch temporal sampling algorithms
- `src/sampling/gsampler.py` - GPU wrapper with automatic device selection
- `src/sampling/temporal_data_loader.py` - Temporal graph data loading utilities
- `src/models/tdgnn_wrapper.py` - TDGNN integration wrapper
- `src/train_tdgnn.py` - Complete TDGNN training pipeline

#### 🔬 Experimental Validation
- `experiments/phase_d_demo.py` - Comprehensive experimental framework
- Delta_t sensitivity analysis (50-400ms time windows)
- Performance benchmarking with GPU/CPU hybrid execution
- Baseline model comparisons and ablation studies

#### 📚 Documentation
- `docs/STAGE6_IMPLEMENTATION_ANALYSIS.md` - Technical implementation analysis
- `docs/STAGE6_COMPLETION_SUMMARY.md` - Complete project summary
- Comprehensive performance benchmarks and architectural documentation

#### ⚡ Performance Improvements
- Sub-100ms inference time for moderate-sized graphs
- Automatic GPU/CPU fallback ensuring universal deployment
- Memory-efficient CSR temporal graph representation
- Configurable sampling strategies (Conservative: 8→2, Aggressive: 42→67 frontier sizes)

#### 🧪 Testing & Validation
- `tests/test_temporal_sampling.py` - Temporal algorithm validation
- `tests/test_gsampler.py` - G-SAMPLER integration tests  
- `tests/test_tdgnn_integration.py` - End-to-end integration tests
- Production readiness validation with comprehensive error handling

### Technical Specifications
- **Algorithm**: Exact binary search temporal filtering per research specifications
- **Architecture**: GPU/CPU hybrid with automatic device selection
- **Integration**: Compatible with all Stage 5 hypergraph models
- **Performance**: Linear scaling with frontier size, configurable trade-offs
- **Deployment**: Production-ready with comprehensive configuration management

## [5.0.0] - 2025-09-10

### Added - Stage 5: Advanced Architectures
- Graph Transformer with multi-head attention
- Heterogeneous Graph Transformer (HGTN)
- Temporal Graph Transformer with spatio-temporal fusion
- Advanced ensemble methods with learned weights
- Unified training pipeline for all transformer architectures

## [4.0.0] - 2025-09-07

### Added - Stage 4: Temporal Modeling
- TGN (Temporal Graph Network) implementation
- TGAT (Temporal Graph Attention) models
- Memory modules with GRU/LSTM updaters
- Temporal sampling with causal constraints
- Memory visualization and analysis tools

## [3.0.0] - 2025-09-05

### Added - Stage 3: Heterogeneous Models
- HAN (Heterogeneous Attention Network) - AUC: 0.876
- R-GCN (Relational Graph Convolutional Network)
- Multi-node-type graph support (Transaction + Wallet)
- Attention mechanisms (node-level and semantic-level)
- Production infrastructure with robust error handling

## [2.0.0] - 2025-09-02

### Added - Stage 2: Advanced GNN Methods
- Advanced GNN architectures
- Improved training pipelines
- Enhanced evaluation frameworks

## [1.0.0] - 2025-08-30

### Added - Stage 1: Basic GNN Models
- GCN (Graph Convolutional Network)
- GraphSAGE implementation
- Basic fraud detection pipeline
- Elliptic++ dataset support

## [0.1.0] - 2025-08-28

### Added - Stage 0: Initial Setup
- Project structure and configuration
- Data exploration notebooks
- Basic preprocessing utilities
- Development environment setup
