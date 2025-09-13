"""
CUSP (Curvature-aware Filtering & Product-Manifold Pooling) Module
Implementation following STAGE8_CUSP_Reference.md

Main components:
- ORC computation and Cusp Laplacian construction
- GPR filter bank for spectral propagation
- Curvature positional encoding
- Product-manifold operations and Cusp Pooling
"""

# Import Phase 1-2 components
from .cusp_orc import compute_orc
from .cusp_laplacian import build_cusp_laplacian
from .cusp_gpr import GPRFilterBank, gpr_filter_bank, ManifoldGPRFilter

# Import Phase 3 components
from .cusp_encoding import (
    curvature_positional_encoding,
    advanced_curvature_encoding,
    CurvatureEncodingLayer
)

# Import Phase 4 components
from .cusp_manifold import (
    ManifoldUtils,
    CuspAttentionPooling,
    ProductManifoldEmbedding,
    validate_product_manifold_operations
)

# CuspModule will be imported after Phase 5 implementation

__all__ = [
    'compute_orc', 
    'build_cusp_laplacian',
    'GPRFilterBank',
    'gpr_filter_bank',
    'ManifoldGPRFilter',
    'curvature_positional_encoding',
    'advanced_curvature_encoding',
    'CurvatureEncodingLayer',
    'ManifoldUtils',
    'CuspAttentionPooling',
    'ProductManifoldEmbedding',
    'validate_product_manifold_operations'
]
