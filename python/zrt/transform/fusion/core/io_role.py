"""IORole / IOSpec / ShapeDerivation — IO descriptor types for fusion rules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class IORole:
    """How to extract one semantic IO tensor from a fusion group.

    See ``docs/fusion_v2_rich_rules_zh.md`` §2.2.
    """

    role: str
    source_kind: Literal["input", "output"] = "input"
    source_op_index: int = -1
    source_arg_index: int = -1
    shape_role: Optional[str] = None
    expected_dtype: Optional[str] = None


# Backward-compat alias for code paths that still import IOSpec.
IOSpec = IORole


@dataclass(frozen=True)
class ShapeDerivation:
    """Symbolic axis derivation expressions.

    Each value is a string expression evaluated by ``_safe_eval`` over
    ``{role: TensorView, ...}`` plus previously-derived axes (so later
    axes can reference earlier ones).
    """

    batch_size:       Optional[str] = None
    seq_len:          Optional[str] = None
    hidden_in:        Optional[str] = None
    hidden_out:       Optional[str] = None
    num_heads:        Optional[str] = None
    head_dim:         Optional[str] = None
    intermediate_dim: Optional[str] = None
    vocab_size:       Optional[str] = None
    num_experts:      Optional[str] = None
    topk:             Optional[str] = None
    extra: tuple[tuple[str, str], ...] = ()

    def items(self) -> list[tuple[str, str]]:
        """Return ``(name, expr)`` pairs in declaration order.

        Standard axes come first, then ``extra`` in declared order.
        Earlier axes are visible in later ones' namespace.
        """
        out: list[tuple[str, str]] = []
        for name in ("batch_size", "seq_len", "hidden_in", "hidden_out",
                     "num_heads", "head_dim", "intermediate_dim",
                     "vocab_size", "num_experts", "topk"):
            v = getattr(self, name)
            if v is not None:
                out.append((name, v))
        out.extend(self.extra)
        return out


# Re-exported only to silence unused-import warnings in callers that
# pass ``Any``-typed values through.
_ = Any
