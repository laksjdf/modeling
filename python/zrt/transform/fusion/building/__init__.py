"""Fused-node construction: builder, IO resolver, annotation propagator."""
from __future__ import annotations

from .annotation_propagator import _propagated_annotations
from .io_resolver import (
    FusedIOPort,
    _child_ops_external_io,
    _external_io,
    resolve_io,
    resolve_io_tensors,
)

__all__ = [
    "FusedIOPort",
    "_child_ops_external_io",
    "_external_io",
    "_propagated_annotations",
    "resolve_io",
    "resolve_io_tensors",
]
