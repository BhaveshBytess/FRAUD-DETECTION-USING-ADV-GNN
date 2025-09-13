"""
Explainability package for hHGTN.

This package provides explainability and interpretability tools for the hHGTN model,
including subgraph extraction, explainer wrappers, visualizations, and reporting.
"""

__version__ = "1.0.0"
__author__ = "hHGTN Team"

from .extract_subgraph import (
    extract_khop_subgraph,
    extract_hetero_subgraph,
    SubgraphExtractor
)

__all__ = [
    "extract_khop_subgraph",
    "extract_hetero_subgraph", 
    "SubgraphExtractor"
]
