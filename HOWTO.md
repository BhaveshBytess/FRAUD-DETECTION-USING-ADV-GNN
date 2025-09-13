# hHGTN - How To Guide

This guide provides minimal commands for reproducing hHGTN fraud detection results in various environments.

## 🚀 Quick Start Options

### Option 1: Google Colab (Fastest)

**One-click demo** - No setup required:

1. Click the Colab badge in README.md
2. Run all cells in sequence
3. View explanations and results inline

**Time**: ~5 minutes | **Requirements**: Google account

---

### Option 2: Local Demo (Recommended for Development)

**Prerequisites**: Python 3.9-3.12, Git

```bash
# 1. Clone and setup
git clone https://github.com/BhaveshBytess/FRAUD-DETECTION-USING-ADV-GNN.git
cd FRAUD-DETECTION-USING-ADV-GNN
pip install -r requirements.txt

# 2. Run demo
python scripts/collect_demo_artifacts.py

# 3. View results
# Check: experiments/demo/run_<timestamp>/
# - preds.csv (predictions)
# - explanations/*.html (explanations)
# - metrics.json (performance)

# 4. Interactive demo
jupyter notebook notebooks/demo.ipynb
```

**Time**: ~10 minutes | **Output**: Full demo artifacts

---

### Option 3: Docker (Production Environment)

**Prerequisites**: Docker installed

```bash
# 1. Build container
git clone https://github.com/BhaveshBytess/FRAUD-DETECTION-USING-ADV-GNN.git
cd FRAUD-DETECTION-USING-ADV-GNN
docker build -t hhgtn-fraud-detection .

# 2. Run demo
docker run --rm -v $(pwd)/experiments:/app/experiments hhgtn-fraud-detection \
    python scripts/collect_demo_artifacts.py

# 3. Interactive session
docker run -it --rm -v $(pwd)/experiments:/app/experiments hhgtn-fraud-detection bash
```

**Time**: ~15 minutes | **Environment**: Fully isolated

---

## 🧪 Full Experimental Reproduction

### Lite Mode (Recommended)

**Quick validation of all components**:

```bash
# Setup environment
conda env create -f environment.yml
conda activate hhgtn-fraud-detection

# Run ablation studies (lite)
cd experiments/stage12/ablation
python matrix_runner.py --config ../configs/ablation_grid.yaml --lite

# Generate reports
python scripts/generate_report.py

# Expected output:
# - reports/results_summary.pdf
# - experiments/stage12/summary.md
```

**Time**: ~30 minutes | **Scope**: All architectural components

### Full Mode (Research Reproduction)

**Complete experimental pipeline**:

```bash
# 1. Setup with full dataset
# Download Elliptic++ dataset to data/ellipticpp/

# 2. Run complete pipeline
python experiments/stage12/ablation/matrix_runner.py
python experiments/stage12/scalability/scalability_runner.py  
python experiments/stage12/robustness/robustness_runner.py
python experiments/stage12/statistical_analysis.py

# 3. Generate comprehensive report
python scripts/generate_report.py --input experiments/stage12/ --out reports/full_results.pdf
```

**Time**: ~4-6 hours | **Scope**: Complete research reproduction

---

## 📊 Key Output Files

### Demo Results
```
experiments/demo/run_<timestamp>/
├── preds.csv                      # Fraud predictions
├── explanations/                  # HTML explanations
│   └── transaction_*.html
└── metrics.json                   # Performance metrics
```

### Full Experimental Results
```
experiments/stage12/
├── ablation/results.csv           # Ablation study results
├── scalability/benchmarks.json    # Performance benchmarks  
├── robustness/defense_analysis.csv # Robustness evaluation
└── summary.md                     # Comprehensive analysis
```

### Reports and Assets
```
reports/
└── results_summary.pdf            # Professional report

assets/
├── architecture.png               # Architecture diagram
└── explanation_snapshot.png       # Explanation example
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Installation Problems**
```bash
# Clear cache and reinstall
pip cache purge
pip install --no-cache-dir -r requirements.txt

# Alternative: Use conda
conda env create -f environment.yml
```

**2. Memory Issues**
```bash
# Use lite mode
python scripts/collect_demo_artifacts.py --lite

# Reduce batch size in configs
# Edit configs/*.yaml and set batch_size: 32
```

**3. CUDA/GPU Issues**
```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
python scripts/collect_demo_artifacts.py
```

**4. Jupyter Notebook Issues**
```bash
# Install kernel
python -m ipykernel install --user --name=hhgtn

# Start notebook
jupyter notebook --ip=0.0.0.0 --port=8888
```

### Validation Commands

**Test installation**:
```bash
python -c "import torch; import torch_geometric; print('✅ Installation OK')"
```

**Test demo**:
```bash
python scripts/collect_demo_artifacts.py --out test_output
ls test_output/  # Should contain preds.csv, explanations/, metrics.json
```

**Test report generation**:
```bash
python scripts/generate_report.py --out test_report.pdf
ls reports/  # Should contain test_report.pdf
```

---

## 📈 Performance Expectations

| Environment | Setup Time | Demo Time | Full Reproduction |
|-------------|------------|-----------|-------------------|
| **Colab** | 0 min | 5 min | 30 min |
| **Local** | 5 min | 10 min | 4-6 hours |
| **Docker** | 10 min | 15 min | 4-6 hours |

### Expected Results
- **AUC Score**: ~0.89 (±0.02)
- **F1 Score**: ~0.86 (±0.02)  
- **Demo Size**: 1-20 transactions
- **Explanation Files**: 1-5 HTML files
- **Report Size**: ~50KB PDF

---

## 🆘 Support

**Quick Help**:
1. Check `reproducibility.md` for exact reproduction commands
2. Review `CHANGELOG.md` for version-specific notes
3. Inspect logs in `experiments/stage13/logs/`

**Common Commands**:
```bash
# Check Python environment
python --version  # Should be 3.9-3.12

# Check key packages
pip show torch torch-geometric

# Test core functionality
python -c "from src.data_utils import *; print('✅ Core modules OK')"

# Generate fresh demo
rm -rf experiments/demo/test_*
python scripts/collect_demo_artifacts.py --out experiments/demo/test_fresh
```

**Expected File Sizes**:
- Demo artifacts: ~100KB total
- Full results: ~10MB total  
- Report PDF: ~50KB
- Checkpoint files: ~1-10MB each
