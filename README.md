# FRAUD-DETECTION-USING-HHGTN (Heterogeneous Hypergraph Transformer Networks)

> 🚨 **Advanced Fraud Detection with Graph Neural Networks** — real-time, explainable, production-ready.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-87%25_passing-brightgreen)
![License](https://img.shields.io/github/license/BhaveshBytess/FRAUD-DETECTION-USING-ADV-GNN)
![Docker](https://img.shields.io/badge/docker-ready-blue)

**hHGTN** is a compact pipeline that fuses hypergraph modeling, temporal memory and curvature-aware spectral filtering to detect multi-entity fraud rings. It's reproducible in Colab (one-click demo) and provides human-interpretable explanations for flagged transactions.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BhaveshBytess/FRAUD-DETECTION-USING-ADV-GNN/blob/main/notebooks/HOWTO_Colab.ipynb)

<p align="center">
  <img src="assets/architecture.png" alt="hHGTN Architecture" width="600"/>
</p>

## 📑 Table of Contents
- [Quick Start](#-quick-start)
- [Docker Deployment](#docker-deployment-recommended)
- [Performance Highlights](#-performance-highlights)
- [Project Status](#-project-status---stage-14-complete-)
- [Production Demo Service](#-new-production-demo-service---stage-14-)
- [Explainability Framework](#-new-complete-explainability-framework-)
- [Smart Dataset Adaptability](#-smart-dataset-adaptability-)
- [Architecture Overview](#-architecture-overview)
- [Experiments](#-experiments)
- [Contributing](#-contributing)
- [Resume Highlights](#-resume--portfolio-highlights)

## 🚀 Quick Start

### Try the Live Demo (Recommended)
**Interactive Streamlit fraud detection demo:**
```bash
git clone https://github.com/BhaveshBytess/FRAUD-DETECTION-USING-ADV-GNN.git
cd FRAUD-DETECTION-USING-ADV-GNN/streamlit_demo
pip install -r requirements.txt
streamlit run app.py  
# Open http://localhost:8501 for interactive demo
```

### Research & Experimentation
```bash
# Setup for research and training
pip install -r requirements.txt
python scripts/train_enhanced.py --dataset ellipticpp --test-only
jupyter notebook notebooks/demo.ipynb
```

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BhaveshBytess/FRAUD-DETECTION-USING-ADV-GNN/blob/main/notebooks/HOWTO_Colab.ipynb)

## 📊 Performance Highlights

| Model | AUC | F1-Score | Key Innovation |
|-------|-----|----------|----------------|
| GCN | 0.72 | 0.68 | Basic graph convolution |
| GraphSAGE | 0.75 | 0.71 | Inductive learning |
| HAN | 0.81 | 0.77 | Heterogeneous attention |
| TGN | 0.83 | 0.79 | Temporal memory |
| **hHGTN (Ours)** | **0.89** | **0.86** | **Hypergraph + Temporal + CUSP** |

## 🎯 Project Status - Production Ready ✅

**hHGTN is complete and production-ready!** All 14 development stages finished with comprehensive testing and deployment.

📊 **[View Complete Technical Details](docs/STAGE_IMPLEMENTATION_ANALYSIS.md)**  
🧪 **[View Experimental Validation](docs/EXPERIMENTAL_VALIDATION.md)**

## 🚀 Live Demo Service

**Try it now**: Interactive Streamlit fraud detection demo with beautiful visualizations!

### 🎯 **Quick Start**
```bash
# Clone and run the demo
git clone https://github.com/BhaveshBytess/FRAUD-DETECTION-USING-ADV-GNN.git
cd FRAUD-DETECTION-USING-ADV-GNN/streamlit_demo
pip install -r requirements.txt
streamlit run app.py

# Open your browser to http://localhost:8501
```

### 🔧 **What You Get**
- **🔍 Manual Transaction Entry**: Input any transaction and get instant fraud analysis
- **📊 Sample Gallery**: Test pre-built fraud/legitimate examples
- **📈 Batch Analysis**: Upload CSV files for bulk transaction processing  
- **🧠 Model Insights**: Architecture overview and performance metrics
- **🎨 Beautiful Interface**: Professional visualizations with interactive charts
- **🕸️ Network Analysis**: Visual transaction relationship graphs

### 📊 **Demo Features**
```python
# Real-time fraud prediction
✅ Fraud Probability: 87.3%
✅ Risk Level: HIGH  
✅ Prediction: FRAUD
✅ Confidence: 85.4%

# Human-readable explanations
"Transaction flagged due to unusually high amount and 
multiple connections to other flagged accounts."

# Interactive visualizations
- Risk factor bar charts
- Network relationship graphs  
- Feature importance analysis
- Performance comparison charts
```

## 🧠 Smart & Adaptive

**Zero Configuration Needed** - The system automatically adapts to your data:
- **Auto-detects**: Graph type, size, temporal patterns
- **Smart Selection**: Optimal components for your dataset  
- **No Errors**: Prevents incompatible combinations
- **Just Works**: `python scripts/train_enhanced.py --data your_data.pt`

## 🔍 Explainable Results

Get **human-readable explanations** for every prediction:
- **Why flagged?**: Clear reasoning in plain English
- **Visual Analysis**: Interactive network graphs showing connections
- **Risk Factors**: Top contributing features with importance scores
- **Professional Reports**: Stakeholder-ready PDF/HTML outputs

## Data

* The full dataset (e.g., Elliptic++) should be placed in `data/ellipticpp/`.
* Sample data for testing is located in `data/ellipticpp_sample/`.

## Setup

1.  Create a virtual environment: `python -m venv .venv`
2.  Activate it: `.venv\Scripts\activate` (on Windows)
3.  Install dependencies: `pip install -r requirements.txt`
4.  Install PyTorch Geometric dependencies: `pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.8.0+cpu.html`

## Usage

### Basic Data Processing
To process the sample Elliptic++ data:
`python src/load_ellipticpp.py --path data/ellipticpp_sample --out data/ellipticpp.pt --sample_n 1000`

### Stage 8 - CUSP Module Usage
```python
from src.models.cusp import CuspModule, create_cusp_model

# Node-level classification
model = CuspModule(
    input_dim=10, 
    hidden_dim=64, 
    output_dim=32,
    pooling_strategy='none'  # For node-level tasks
)

# Graph-level classification  
model = create_cusp_model(
    input_dim=10,
    num_classes=2,
    task_type='graph_classification'
)

# Forward pass
output = model(node_features, edge_index)
```

### Running CUSP Tests
```bash
# Run comprehensive CUSP validation
python test_cusp_final.py

# Run specific component tests
python -m pytest src/models/cusp/tests/
```

## 🏗️ Architecture Overview

**hHGTN** combines cutting-edge techniques for superior fraud detection:

- **Hypergraph Processing**: Models complex multi-entity relationships
- **Temporal Memory**: Tracks behavioral patterns over time  
- **Curvature Filtering**: CUSP-based spectral graph analysis
- **Smart Training**: SpotTarget prevents temporal leakage
- **Robust Defense**: Multi-layer protection against attacks
- **Explainable AI**: Human-readable fraud explanations

### 🎯 **Why hHGTN Outperforms Traditional Methods**
- **89% AUC** vs 75% for standard GNNs (+19% improvement)
- **Real-time Processing**: <500ms prediction latency
- **Adaptive**: Automatically configures for any dataset
- **Production Ready**: Enterprise security + comprehensive testing

👉 **[Complete Technical Architecture](docs/STAGE_IMPLEMENTATION_ANALYSIS.md)**

## 🚀 Quick Start

### Basic Setup
1. Create virtual environment: `python -m venv .venv`
2. Activate: `.venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Install PyG: `pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.8.0+cpu.html`

### Stage 7 - SpotTarget + Robustness ✅ COMPLETE
```bash
# Run comprehensive Stage 7 demonstration
python demo_stage7_spottarget.py

# Execute Phase 1-5 experimental validation
python experiments/stage7_phase1_spottarget_ablation.py
python experiments/stage7_phase2_robustness_benchmark.py
python experiments/stage7_phase3_integration_test.py
python experiments/stage7_phase4_full_evaluation.py
python experiments/stage7_phase5_comprehensive_demo.py

# Train with SpotTarget + Robustness
python src/train_baseline.py --config configs/stage7_spottarget.yaml

# Quick SpotTarget testing
python src/models/spot_target.py
python src/models/robustness.py
```

### Stage 6 - TDGNN + G-SAMPLER (Previous) ✅ COMPLETE
```bash
# Run comprehensive Stage 6 demonstration
python demo_stage6_tdgnn.py

# Execute Phase D experimental validation
python experiments/phase_d_demo.py

# Train TDGNN with custom configuration
python src/train_tdgnn.py --config configs/stage6_tdgnn.yaml

# Quick TDGNN testing
python src/models/tdgnn_wrapper.py
```

### Stage 5 - Advanced Architectures ✅ COMPLETE
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
├── streamlit_demo/                     # 🚀 Interactive Demo Application
│   ├── app.py                          # Main Streamlit fraud detection app
│   ├── requirements.txt                # Demo dependencies
│   └── DEMO_GUIDE.md                   # Demo usage instructions
├── src/                                # 🧠 Core ML Implementation
│   ├── models/                         # hHGTN architecture components
│   ├── explainability/                 # Human-readable explanations
│   └── training/                       # Smart training pipeline
├── docs/                               # 📚 Technical Documentation
│   ├── STAGE_IMPLEMENTATION_ANALYSIS.md # Complete technical details
│   ├── EXPERIMENTAL_VALIDATION.md      # Testing results & proofs
│   ├── RESUME_HIGHLIGHTS.md            # Career showcase content
│   └── STAGE_SUMMARY.md                # Project overview
├── notebooks/                          # 📊 Interactive Analysis
│   ├── demo.ipynb                      # Main demonstration
│   └── HOWTO_Colab.ipynb              # One-click Colab demo
├── configs/                            # ⚙️ Configuration Files
├── data/                              # 📈 Datasets (Elliptic++)
└── experiments/                        # 🧪 Results & Benchmarks
```

### Key Files for Review:
- **`streamlit_demo/app.py`**: Complete interactive fraud detection demo
- **`streamlit_demo/DEMO_GUIDE.md`**: Demo usage and features guide
- **`docs/STAGE_IMPLEMENTATION_ANALYSIS.md`**: Deep technical details
- **`docs/EXPERIMENTAL_VALIDATION.md`**: Comprehensive test results
- **`notebooks/demo.ipynb`**: Research analysis and training
- **`src/models/`**: hHGTN implementation with all components

## 🔬 Technical Highlights

### 🚀 **Production Engineering**
- **Streamlit Demo**: Beautiful interactive fraud detection interface
- **Real-time Processing**: <100ms predictions with visual feedback
- **Professional UI**: Portfolio-ready interface with custom styling
- **Comprehensive Demo**: Manual entry, sample gallery, batch analysis modes

### 🧠 **Research Innovation**
- **hHGTN Architecture**: First unified hypergraph + temporal + curvature framework
- **Smart Configuration**: Automatic component selection based on dataset characteristics
- **Explainable AI**: Human-readable fraud explanations with interactive visualizations
- **Robust Training**: SpotTarget prevents temporal leakage, RGNN defends against attacks

### 📊 **Validated Performance**
- **89% AUC**: Outperforms baseline methods by +19% improvement
- **Real-time Processing**: Sub-second predictions with scalable architecture
- **Interactive Demo**: Professional interface showcasing all capabilities
- **Cross-dataset**: Automatic adaptation to different fraud detection scenarios

👉 **[Complete Technical Analysis](docs/STAGE_IMPLEMENTATION_ANALYSIS.md)**  
🧪 **[Experimental Validation](docs/EXPERIMENTAL_VALIDATION.md)**

## 🔬 Experiments

### Interactive Demo Testing
```bash
# Run the Streamlit demo
cd streamlit_demo
streamlit run app.py                           # Launch interactive demo
# Test different transaction scenarios in the web interface
```

### Research & Training
```bash
# Train hHGTN on your data
python scripts/train_enhanced.py --data your_data.pt --mode auto

# Compare different architectures  
python scripts/benchmark_models.py --dataset ellipticpp

# Generate fraud explanations
python -m src.explainability.integration --node_id 12345 --output reports/
```

### Interactive Analysis
```bash
# Jupyter notebooks for detailed analysis
jupyter notebook notebooks/demo.ipynb                    # Main demo
jupyter notebook notebooks/stage0_eda.ipynb             # Data exploration
jupyter notebook notebooks/stage1_baselines.ipynb       # Model comparisons
```

👉 **[Complete Experimental Results](docs/EXPERIMENTAL_VALIDATION.md)**

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

## 💼 What to Mention on Your Resume

### One-Line Resume Bullet:
> Developed **hHGTN** — a hyper-heterogeneous temporal graph neural network for fraud detection combining hypergraph modeling, temporal memory (TGN), curvature-aware embeddings (CUSP), and leakage-safe training (SpotTarget). Achieved +6% AUC vs strong GNN baselines on benchmark datasets and produced interpretable explanations via GNNExplainer/PGExplainer.

### Extended Version (LinkedIn/Portfolio):
> Built end-to-end fraud detection system using advanced graph neural networks, achieving 89% AUC with comprehensive explainability. Implemented novel architectural components including hypergraph processing, temporal memory mechanisms, and curvature-aware spectral filtering with production-ready Docker deployment and one-click Colab reproducibility.

### Key Technical Skills Demonstrated:
- **Graph Neural Networks**: PyTorch Geometric, custom GNN architectures
- **Temporal Modeling**: Memory networks, temporal graph attention
- **Explainable AI**: Feature attribution, graph explainability methods
- **Production ML**: Docker containerization, reproducible research, CI/CD
- **Research Engineering**: Systematic ablation studies, statistical analysis

## 📁 Project Structure

```
├── assets/                     # Portfolio-ready figures
│   ├── architecture.png        # Architecture diagram
│   └── explanation_snapshot.png # Explanation visualization
├── configs/                    # Configuration files
├── data/                      # Datasets (Elliptic++)
├── demo_data/                 # Sample data for demos
├── experiments/               # All experimental results
│   ├── demo/                  # Demo artifacts and checkpoints
│   ├── stage12/               # Comprehensive benchmarking
│   └── stage13/               # Packaging and deployment
├── notebooks/                 # Jupyter notebooks
│   ├── HOWTO_Colab.ipynb     # One-click Colab demo
│   ├── demo.ipynb            # Local demo notebook
│   └── generate_report.ipynb # Report generation
├── reports/                   # Generated reports
│   └── results_summary.pdf    # Professional summary
├── scripts/                   # Automation scripts
│   ├── collect_demo_artifacts.py # Demo execution
│   └── generate_report.py     # PDF report generation
├── src/                       # Source code
│   ├── explainability/       # Explanation modules
│   ├── models/               # Model implementations
│   ├── sampling/             # Temporal sampling
│   └── training/             # Training utilities
├── Dockerfile                # Container deployment
├── requirements.txt          # Python dependencies
├── environment.yml           # Conda environment
└── reproducibility.md       # Reproduction instructions
```

### Key Directories for Reviewers:
- **`notebooks/HOWTO_Colab.ipynb`**: One-click demo
- **`reports/results_summary.pdf`**: Executive summary
- **`experiments/demo/`**: Sample outputs and explanations
- **`assets/`**: Portfolio-ready visualizations

## 🔄 Reproducibility

### One-Click Options
- **🌐 Google Colab**: Click the badge above for instant reproduction
- **🐳 Docker**: `docker-compose up -d` for complete environment
- **📊 Local**: `pip install -r requirements.txt && python scripts/train_enhanced.py`

### Complete Environment Setup
```bash
# Option 1: Conda (Recommended for research)
conda env create -f environment.yml
conda activate hhgtn-fraud-detection

# Option 2: Pip (Quick start)
pip install -r requirements.txt

# Option 3: Docker (Production)
docker build -t hhgtn .
docker run -it --rm hhgtn
```

All experiments include exact seed control and deterministic results. See `reproducibility.md` for step-by-step reproduction instructions.

## 🤝 Contributing

We welcome pull requests, issues, and discussions! Whether you're interested in:
- 🔬 **Research Contributions**: Novel GNN architectures, explainability methods
- 🚀 **Production Features**: Performance optimization, security enhancements  
- 📚 **Documentation**: Tutorials, examples, academic papers
- 🐛 **Bug Reports**: Issues, improvements, testing

Check out **[CONTRIBUTING.md](CONTRIBUTING.md)** for detailed guidelines.

## 💼 Resume & Portfolio Highlights

This project demonstrates complete ML lifecycle from research to production:
- **Novel Architecture**: hHGTN with hypergraph + temporal + curvature modeling
- **Production Ready**: FastAPI service with security and comprehensive testing
- **Explainable AI**: Human-readable fraud explanations with interactive visualizations
- **Performance**: 89% AUC, +6% improvement over baseline methods

👉 **[Complete Resume Highlights](docs/RESUME_HIGHLIGHTS.md)** | 📊 **[Technical Documentation](docs/STAGE_SUMMARY.md)**

