# Results Documentation

This directory contains the key experimental results, performance evaluations, and visualization outputs from the fraud detection project.

## Directory Structure

```
results/
├── README.md                    # This file
├── performance_metrics.json    # Raw numerical results
├── model_comparison.csv        # Side-by-side algorithm comparison
├── baseline_report.pdf         # Comprehensive experimental report
├── demo_predictions.json       # Sample predictions for demo
├── confusion_matrices/         # Performance visualization
│   ├── hgtn_confusion.png
│   ├── gat_confusion.png
│   └── sage_confusion.png
├── training_plots/            # Training convergence analysis
│   ├── loss_curves.png
│   ├── accuracy_curves.png
│   └── auc_progression.png
└── feature_analysis/          # Model interpretability
    ├── attention_weights.png
    ├── feature_importance.csv
    └── embeddings_tsne.png
```

## Key Performance Results

### Overall Model Comparison

| Model | AUC | Precision | Recall | F1-Score | Training Time |
|-------|-----|-----------|--------|----------|---------------|
| **hHGTN** | **0.89** | **0.84** | **0.87** | **0.85** | 45 min |
| GAT | 0.83 | 0.79 | 0.81 | 0.80 | 35 min |
| GraphSAGE | 0.78 | 0.75 | 0.76 | 0.75 | 25 min |
| Random Forest | 0.72 | 0.68 | 0.74 | 0.71 | 8 min |

### Best Model Details (hHGTN)

- **Dataset**: EllipticPP (203K transactions, 822K addresses)
- **Architecture**: 3-layer heterogeneous graph transformer
- **Hidden Dimensions**: 128
- **Attention Heads**: 8
- **Training Epochs**: 200
- **Validation Strategy**: 70/15/15 split

## Key Findings

1. **Heterogeneous Architecture Advantage**: The hHGTN model's ability to handle multiple node types (transactions, addresses) and edge types significantly outperformed homogeneous approaches.

2. **Attention Mechanism Value**: Transformer-style attention provided 6% AUC improvement over traditional GCN approaches, with interpretable attention weights highlighting suspicious transaction patterns.

3. **Scalability**: Successfully processed the full EllipticPP dataset with 1M+ nodes, demonstrating real-world applicability.

4. **Feature Learning**: The model learned meaningful node embeddings that cluster illicit and licit transactions in separable regions of the embedding space.

## Result Files Description

### Performance Metrics (`performance_metrics.json`)
Raw numerical results for all experiments, including:
- Per-epoch training metrics
- Cross-validation scores
- Statistical significance tests
- Runtime benchmarks

### Model Comparison (`model_comparison.csv`)
Tabular comparison of all baseline and proposed methods with:
- Hyperparameter configurations
- Performance metrics with confidence intervals
- Resource utilization metrics

### Baseline Report (`baseline_report.pdf`)
Comprehensive 10-page experimental report including:
- Methodology and experimental setup
- Detailed results analysis
- Ablation studies
- Discussion and future work

### Demo Predictions (`demo_predictions.json`)
Sample predictions used in the demo notebook:
- High-confidence fraud predictions
- Borderline cases for manual review
- Attention weight visualizations

## Reproducibility Notes

All results can be reproduced using:

```bash
# Train from scratch
python src/train_baseline.py --config configs/baseline.yaml

# Evaluate saved model
python src/eval.py --model_path experiments/baseline/model.pt

# Generate visualizations
jupyter notebook notebooks/stage1_baselines.ipynb
```

## Citation

If you use these results in your research, please cite:

```bibtex
@misc{frauddetection2024,
  title={Heterogeneous Graph Transformer Networks for Cryptocurrency Fraud Detection},
  author={[Your Name]},
  year={2024},
  note={Implementation available at: https://github.com/[username]/hhgtn-project}
}
```

## Contact

For questions about these results or reproduction issues:
- Email: [your.email@domain.com]
- GitHub Issues: [Repository URL]/issues
- LinkedIn: [Your LinkedIn Profile]
