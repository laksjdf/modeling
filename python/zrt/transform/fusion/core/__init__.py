"""Core value objects for fusion: pattern, IO role, rule.

Re-exports the public types so other sub-packages can do
``from .core import ModuleFusionRule, MatchPattern, IORole``.
"""
from __future__ import annotations

from .io_role import IORole, IOSpec, ShapeDerivation
from .pattern import DEFAULT_SKIP_OPS, MatchKind, MatchPattern
from .rule import ModuleFusionRule

__all__ = [
    "DEFAULT_SKIP_OPS",
    "IORole",
    "IOSpec",
    "MatchKind",
    "MatchPattern",
    "ModuleFusionRule",
    "ShapeDerivation",
]
