# Baseline Experiment

This directory contains the results of the baseline model training.

## How to Reproduce

To reproduce the results, run the following command from the root of the project:

### Lite Mode (using a sample of the data)
```bash
python src/train_baseline.py --config configs/baseline.yaml
```

### Full Mode
To run on the full dataset, you can either create a new config file or override the `sample` argument:
```bash
python src/train_baseline.py --config configs/baseline.yaml --sample null
```

## Evaluation
To evaluate a trained model, run:
```bash
python src/eval.py --ckpt experiments/baseline/lite/ckpt.pth --data_path data/ellipticpp.pt
```
