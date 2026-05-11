"""Fusion: operator fusion rules and passes.

Core fusion algorithm lives in ``core.py``.
- ``FusionPass`` : GraphPass for the TransformPipeline (OpGraph IR).
- ``FusionSpec`` : Dataclass for discovered fusion patterns (Excel export).
- ``fuse_records()`` : Dict records → fused records (replaces FusionEngine).
"""
from .pass_ import FusionPass
from ._dict_bridge import FusionSpec, fuse_records

__all__ = [
    "FusionPass",
    "FusionSpec",
    "fuse_records",
]
