# src/sampling/__init__.py
"""
Stage 6 TDGNN + G-SAMPLER Sampling Module
Implements time-relaxed neighbor sampling per STAGE6_TDGNN_GSAMPLER_REFERENCE.md
"""

from .cpu_fallback import sample_time_relaxed_neighbors, TemporalGraph
from .utils import build_temporal_adjacency, validate_temporal_constraints

__all__ = [
    'sample_time_relaxed_neighbors',
    'TemporalGraph', 
    'build_temporal_adjacency',
    'validate_temporal_constraints'
]
