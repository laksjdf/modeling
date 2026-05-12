"""Fusion-node semantic field derivation.

Re-exports the public surface tests rely on (``annotate_fused_node``,
``TensorView``, ``compute_flops``, ``compute_memory``, ``derive_shape_axes``,
``resolve_io_views``).  Implementations live in :mod:`.annotator`.
"""
from __future__ import annotations

from .annotator import (
    TensorView,
    _FLATTEN_AXES,
    _io_view_to_dict,
    _safe_call,
    _safe_eval_formula,
    annotate_fused_node,
    compute_flops,
    compute_memory,
    derive_shape_axes,
    resolve_io_views,
)
from .safe_eval import FormulaError, safe_eval

__all__ = [
    "FormulaError",
    "TensorView",
    "annotate_fused_node",
    "compute_flops",
    "compute_memory",
    "derive_shape_axes",
    "resolve_io_views",
    "safe_eval",
]
