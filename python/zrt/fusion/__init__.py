"""Fusion rule discovery tool (development utility).

Re-exports the public API of the :mod:`zrt.fusion.discover` subpackage so
legacy imports such as::

    from zrt.fusion.discover import discover_fusion_rules

continue to work unchanged after the package split.
"""
from __future__ import annotations

from .discover import (
    AstClassInfo,
    AstScanResult,
    TEMPLATES,
    discover_fusion_rules,
    get_template,
    join_rules,
    run_runtime_trace,
    scan_model_file,
)

__all__ = [
    "discover_fusion_rules",
    "scan_model_file",
    "run_runtime_trace",
    "join_rules",
    "TEMPLATES",
    "get_template",
    "AstScanResult",
    "AstClassInfo",
]
