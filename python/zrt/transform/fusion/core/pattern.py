"""MatchPattern + MatchKind + DEFAULT_SKIP_OPS — pattern definitions for fusion rules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


MatchKind = Literal["class_only", "ordered_regex", "dag_signature"]


# ATen ops that may interleave between matched ops without breaking
# an ``ordered_regex`` match.  Shape / dtype-cast / no-op rewires.
DEFAULT_SKIP_OPS: frozenset[str] = frozenset({
    "aten.view.default",
    "aten.reshape.default",
    "aten.permute.default",
    "aten.transpose.int",
    "aten.expand.default",
    "aten.squeeze.dim",
    "aten.unsqueeze.default",
    "aten.flatten.using_ints",
    "aten.unflatten.int",
    "aten.contiguous.memory_format",
    "aten._to_copy.default",
    "aten.to.dtype",
    "aten.to.dtype_layout",
    "aten.clone.default",
    "aten.detach.default",
    "aten.alias.default",
})


@dataclass(frozen=True)
class MatchPattern:
    """Pattern definition for matching a fusion group.

    See ``docs/fusion_v2_rich_rules_zh.md`` §2.1 for full semantics.
    """

    kind: MatchKind = "ordered_regex"
    op_regexes: tuple[str, ...] = ()
    op_multiset: tuple[tuple[str, int], ...] = ()
    skip_ops: frozenset[str] = field(default_factory=lambda: DEFAULT_SKIP_OPS)
    min_ops: int = 1
    max_ops: int = 1024

    def __post_init__(self) -> None:
        if self.kind == "ordered_regex" and not self.op_regexes:
            # ordered_regex with no regexes is invalid
            raise ValueError("ordered_regex requires non-empty op_regexes")
        if self.kind == "dag_signature" and not self.op_multiset:
            raise ValueError("dag_signature requires non-empty op_multiset")
        if self.min_ops < 1:
            raise ValueError("min_ops must be >= 1")
        if self.max_ops < self.min_ops:
            raise ValueError("max_ops must be >= min_ops")
