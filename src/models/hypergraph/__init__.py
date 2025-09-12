"""
Stage 5: PhenomNN-Based Hypergraph Neural Networks for Fraud Detection

This module implements hypergraph neural networks based on the PhenomNN paper:
"From Hypergraph Energy Functions to Hypergraph Neural Networks"

Key components:
- HypergraphData: Data structure with incidence matrices
- PhenomNNLayer: Energy-based layer implementations (Equations 22, 25)  
- HypergraphNN: Multi-layer architecture
- Fraud-specific hyperedge construction utilities

Mathematical foundation follows the energy-based formulation with:
- Clique expansion: AC = B @ inv(DH) @ B.T
- Star expansion: AS_bar = B @ inv(DH) @ B.T  
- Preconditioner: D̃ = λ0*DC + λ1*DS_bar + I
"""

from .hypergraph_data import HypergraphData, construct_hypergraph_from_hetero
from .construction import construct_fraud_hyperedges, FraudHyperedgeConstructor
from .phenomnn import PhenomNNSimpleLayer, PhenomNNLayer
from .architecture import HypergraphNN, HypergraphClassifier, HypergraphConfig, create_hypergraph_model
from .utils import (
    compute_degree_matrices, 
    compute_expansion_matrices,
    validate_hypergraph_structure
)

__all__ = [
    'HypergraphData',
    'construct_hypergraph_from_hetero', 
    'construct_fraud_hyperedges',
    'FraudHyperedgeConstructor',
    'PhenomNNSimpleLayer',
    'PhenomNNLayer',
    'HypergraphNN',
    'HypergraphClassifier',
    'HypergraphConfig',
    'create_hypergraph_model',
    'compute_degree_matrices',
    'compute_expansion_matrices', 
    'validate_hypergraph_structure'
]
