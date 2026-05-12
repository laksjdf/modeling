"""Operator fusion module.

See ``docs/fusion_v2_rich_rules_zh.md`` for the rule contract.

Public API:
- ``FusionPass`` — GraphPass for the TransformPipeline.
- ``ModuleFusionRule`` — fusion rule (rich format).
- ``IORole``, ``IOSpec`` — IO role declaration (``IOSpec`` is alias).
- ``MatchPattern``, ``MatchKind``, ``DEFAULT_SKIP_OPS`` — match patterns.
- ``ShapeDerivation`` — symbolic axis derivation.
- ``register_rule``, ``clear_rules``, ``all_rules`` — registry forwarders.
- ``load_yaml_rules``, ``add_yaml_search_dir`` — YAML loading.
- ``resolve_fusion_config``, ``load_fusion_config_file`` — FusionConfig.
"""
from __future__ import annotations

from .api import FusionPass
from .core.io_role import IORole, IOSpec, ShapeDerivation
from .core.pattern import DEFAULT_SKIP_OPS, MatchKind, MatchPattern
from .core.rule import ModuleFusionRule
from .loading.fusion_config import (
    load_fusion_config_file,
    resolve_fusion_config,
)
from .loading.yaml_rule_loader import add_yaml_search_dir, load_yaml_rules
from .registry import all_rules, clear_rules, register_rule

__all__ = [
    "FusionPass",
    "ModuleFusionRule",
    "IORole",
    "IOSpec",
    "MatchPattern",
    "MatchKind",
    "ShapeDerivation",
    "DEFAULT_SKIP_OPS",
    "register_rule",
    "clear_rules",
    "all_rules",
    "load_yaml_rules",
    "add_yaml_search_dir",
    "resolve_fusion_config",
    "load_fusion_config_file",
]
