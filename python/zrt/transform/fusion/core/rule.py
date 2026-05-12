"""ModuleFusionRule v2 (rich format) — single dataclass holding a fusion rule.

YAML-helper functions live in ``loading/yaml_rule_loader.py``;
``from_yaml_dict`` delegates to the loader so this module stays slim.

Backward compatibility: ``IOSpec`` is preserved as an alias of ``IORole``
in ``core.io_role``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from .io_role import IORole, ShapeDerivation
from .pattern import MatchPattern


@dataclass(frozen=True)
class ModuleFusionRule:
    """One fusion rule keyed on an nn.Module class (or class-name regex).

    See ``docs/fusion_v2_rich_rules_zh.md`` §2.4 for the full schema.
    """

    target_class: type | str | tuple[str, ...]

    op_type: Optional[str] = None

    # Matching
    pattern: Optional[MatchPattern] = None
    op_sequences: tuple[tuple[str, ...], ...] = ()

    # IO roles
    io_roles: tuple[IORole, ...] = ()
    inputs:  tuple[IORole, ...] = ()
    outputs: tuple[IORole, ...] = ()

    # Semantic derivation
    shape_derivation: Optional[ShapeDerivation] = None
    flops_formula:  Optional[str] = None
    memory_formula: Optional[str] = None
    flops_callable:  Optional[Callable[[dict], float]] = None
    memory_callable: Optional[Callable[[dict], float]] = None
    flops_kind: Literal["sum_children", "from_io", "formula", "custom"] = "sum_children"
    custom_resolver: Optional[Callable] = None

    # Metadata
    annotations: dict = field(default_factory=dict)
    priority: int = 10

    # Identity & user-facing config — see plan for FusionConfig.
    name: str = ""
    description: str = ""
    default_phases: tuple[str, ...] = ("inference", "training")

    def __post_init__(self) -> None:
        # Default name = op_type (when supplied) — class-only rules without
        # op_type fall back to a synthesized identifier.
        if not self.name:
            tc = self.target_class
            if isinstance(tc, type):
                tc_str = tc.__name__
            elif isinstance(tc, tuple):
                tc_str = "+".join(str(c) for c in tc)
            else:
                tc_str = str(tc)
            object.__setattr__(self, "name", self.op_type or tc_str)

        # Auto-merge legacy inputs/outputs into io_roles
        if self.inputs or self.outputs:
            merged = list(self.io_roles) + list(self.inputs) + list(self.outputs)
            object.__setattr__(self, "io_roles", tuple(merged))
            object.__setattr__(self, "inputs", ())
            object.__setattr__(self, "outputs", ())

        # Auto-build pattern when only op_sequences is provided.
        if self.pattern is None:
            if not self.op_sequences:
                # No matching info → match on class only.
                object.__setattr__(self, "pattern",
                                   MatchPattern(kind="class_only",
                                                op_regexes=(),
                                                op_multiset=()))
            elif len(self.op_sequences) == 1:
                seq = self.op_sequences[0]
                if not seq:
                    # Empty sequence → class_only.
                    object.__setattr__(self, "pattern",
                                       MatchPattern(kind="class_only",
                                                    op_regexes=(),
                                                    op_multiset=()))
                else:
                    object.__setattr__(self, "pattern",
                                       MatchPattern(
                                           kind="ordered_regex",
                                           op_regexes=tuple(re.escape(o) for o in seq),
                                           min_ops=len(seq),
                                       ))
            else:
                # Multiple alternatives must be expanded at YAML-load
                # time.  Reject construction here to surface bugs early.
                raise ValueError(
                    "multiple op_sequences must be expanded into multiple "
                    "rules by the loader; ModuleFusionRule allows at most one"
                )

        # Promote flops_kind to "formula" automatically when a formula or
        # callable is supplied but flops_kind was left at the default.
        if self.flops_kind == "sum_children" and (
            self.flops_formula or self.flops_callable
        ):
            object.__setattr__(self, "flops_kind", "formula")

    # ── Backward-compatible YAML constructor ────────────────────────────

    @classmethod
    def from_yaml_dict(cls, d: dict) -> "ModuleFusionRule":
        """Construct a rule from a parsed YAML dict (new or legacy schema).

        Implementation lives in ``loading/yaml_rule_loader.py`` so this
        module stays dataclass-only.
        """
        from python.zrt.transform.fusion.loading.yaml_rule_loader import (
            rule_from_yaml_dict,
        )
        return rule_from_yaml_dict(cls, d)
