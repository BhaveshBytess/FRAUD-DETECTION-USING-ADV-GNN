# GCN Baseline Experiments

This directory contains trained models and evaluation results for the GCN baseline implementation.

## Quick Start

### Lite Mode (Fast Testing)
```bash
# Train GCN baseline with sampling (2000 nodes)
python src/train_baseline.py --config configs/gcn.yaml

# Evaluate the trained model
python src/eval.py --ckpt experiments/baseline/lite_gcn/ckpt.pth --data_path data/ellipticpp/ellipticpp.pt --model gcn --sample 2000
```

### Full Mode (Complete Dataset)
```bash
# Train on full dataset (no sampling)
python src/train_baseline.py --model gcn --data_path data/ellipticpp/ellipticpp.pt --out_dir experiments/baseline/gcn_full --epochs 20 --sample null

# Evaluate full model
python src/eval.py --ckpt experiments/baseline/gcn_full/ckpt.pth --data_path data/ellipticpp/ellipticpp.pt --model gcn
```

## Directory Structure

```
experiments/baseline/
├── lite_gcn/           # Lite mode results (sampled data)
│   ├── ckpt.pth       # Model checkpoint
│   └── metrics.json   # Evaluation metrics
├── lite_graphsage/    # GraphSAGE lite results
├── lite_rgcn/         # RGCN lite results
└── gcn_full/          # Full dataset results (when available)
```

## Configuration Files

All model configurations are stored in `configs/` directory:
- `configs/gcn.yaml` - GCN baseline settings
- `configs/graphsage.yaml` - GraphSAGE settings  
- `configs/rgcn.yaml` - RGCN settings

## Expected Results (Lite Mode)

Typical performance metrics for GCN baseline on sampled data:
- **AUC**: 0.70-0.85
- **PR-AUC**: 0.60-0.80
- **F1**: 0.65-0.75
- **Recall**: 0.60-0.80

*Note: Results may vary based on random sampling and initialization.*

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce `sample` size or use `--device cpu`
2. **File not found**: Ensure `data/ellipticpp/ellipticpp.pt` exists (run Stage 0 first)
3. **Import errors**: Activate virtual environment and install requirements

### Performance Tips
- Use `--sample 1000` for very quick testing
- Increase `hidden_dim` for better performance (requires more memory)
- Adjust learning rate if training is unstable
