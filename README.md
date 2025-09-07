# FRAUD-DETECTION-USING-ADV-GNN (HHGTN Project)

This project is a template for building Heterogeneous Graph Transformer Networks for fraud detection.

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

## Testing

Run tests with `pytest`.

