"""Fusion-node semantic field derivation.

Given a matched fusion group + ``ModuleFusionRule``, derive the semantic
quantities (batch_size / seq_len / hidden_in / hidden_out / dtype /
flops / memory_bytes) and write them into ``fused_node.annotations``
for downstream simulator consumption.

The pipeline is:

    resolve_io_views  →  derive_shape_axes  →  compute_flops / compute_memory
                                              ↓
                                       annotate_fused_node

Failures at any stage are logged as warnings and never raise — fusion
must continue even if a rule's formula is malformed.

See ``docs/fusion_v2_rich_rules_zh.md`` §2.2 / §2.3 / §2.4 / §4.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any, Optional

from .safe_eval import FormulaError, safe_eval

if TYPE_CHECKING:  # pragma: no cover
    from python.zrt.ir.node import OpNode
    from python.zrt.ir.types import TensorMeta
    from python.zrt.transform.fusion.core.rule import ModuleFusionRule


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TensorView
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TensorView:
    """Lightweight tensor descriptor exposed to ``safe_eval`` formulas.

    Matches the contract in ``docs/fusion_v2_rich_rules_zh.md`` §4.
    """

    shape: tuple[int, ...]
    dtype: str          # e.g. "bf16" — DType.value
    bytes: int
    numel: int
    itemsize: float

    @classmethod
    def from_tensor_meta(cls, m: "TensorMeta") -> "TensorView":
        shape = tuple(m.shape)
        numel = prod(shape) if shape else 1
        return cls(
            shape=shape,
            dtype=m.dtype.value,
            bytes=int(m.mem_bytes),
            numel=int(numel),
            itemsize=float(m.dtype.itemsize),
        )


# ─────────────────────────────────────────────────────────────────────────────
# IO resolution
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_idx(idx: int, length: int) -> int:
    return idx if idx >= 0 else length + idx


def resolve_io_views(
    child_ops: list,
    rule: "ModuleFusionRule",
) -> dict[str, TensorView]:
    """Resolve ``rule.io_roles`` against ``child_ops`` into TensorViews.

    Behaviour:
      - Out-of-range op or arg index → warn, skip that role.
      - Same ``role`` declared twice → later overwrites earlier, warn.
      - Empty ``child_ops`` or empty ``io_roles`` → returns ``{}``.
    """
    views: dict[str, TensorView] = {}
    if not child_ops:
        return views

    for spec in rule.io_roles:
        op_idx = _normalize_idx(spec.source_op_index, len(child_ops))
        if op_idx < 0 or op_idx >= len(child_ops):
            logger.warning(
                "fusion.semantics: role %r source_op_index %d out of range "
                "(%d ops); skipping",
                spec.role, spec.source_op_index, len(child_ops),
            )
            continue
        op = child_ops[op_idx]
        tensors = op.inputs if spec.source_kind == "input" else op.outputs
        if not tensors:
            logger.warning(
                "fusion.semantics: role %r — op %s has no %s tensors; skipping",
                spec.role, op.id, spec.source_kind,
            )
            continue
        arg_idx = _normalize_idx(spec.source_arg_index, len(tensors))
        if arg_idx < 0 or arg_idx >= len(tensors):
            logger.warning(
                "fusion.semantics: role %r source_arg_index %d out of range "
                "(%d %s tensors on op %s); skipping",
                spec.role, spec.source_arg_index, len(tensors),
                spec.source_kind, op.id,
            )
            continue
        meta = tensors[arg_idx]
        if spec.role in views:
            logger.warning(
                "fusion.semantics: role %r declared multiple times; "
                "overwriting previous value",
                spec.role,
            )
        try:
            views[spec.role] = TensorView.from_tensor_meta(meta)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning(
                "fusion.semantics: failed to build TensorView for role %r: %s",
                spec.role, e,
            )
    return views


# ─────────────────────────────────────────────────────────────────────────────
# Shape derivation
# ─────────────────────────────────────────────────────────────────────────────

def derive_shape_axes(
    io_views: dict[str, TensorView],
    rule: "ModuleFusionRule",
) -> dict[str, Any]:
    """Evaluate each axis expression in ``rule.shape_derivation``.

    Earlier axes are visible to later expressions.  A failed expression
    is logged and skipped; remaining axes still get evaluated.
    """
    axes: dict[str, Any] = {}
    sd = rule.shape_derivation
    if sd is None:
        return axes

    for name, expr in sd.items():
        ns: dict[str, Any] = {}
        ns.update(io_views)
        ns.update(axes)
        try:
            value = safe_eval(expr, ns)
        except FormulaError as e:
            logger.warning(
                "fusion.semantics: shape axis %r expr %r failed: %s",
                name, expr, e,
            )
            continue
        except Exception as e:  # pragma: no cover — defensive
            logger.warning(
                "fusion.semantics: shape axis %r expr %r raised: %s",
                name, expr, e,
            )
            continue
        axes[name] = value
    return axes


# ─────────────────────────────────────────────────────────────────────────────
# FLOPs / memory
# ─────────────────────────────────────────────────────────────────────────────

def _safe_eval_formula(
    label: str, expr: str, namespace: dict,
) -> Optional[float]:
    try:
        v = safe_eval(expr, namespace)
    except FormulaError as e:
        logger.warning("fusion.semantics: %s formula %r failed: %s",
                       label, expr, e)
        return None
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("fusion.semantics: %s formula %r raised: %s",
                       label, expr, e)
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        logger.warning(
            "fusion.semantics: %s formula %r returned non-numeric %r",
            label, expr, v,
        )
        return None
    return float(v)


def _safe_call(
    label: str, fn, namespace: dict,
) -> Optional[float]:
    try:
        v = fn(namespace)
    except Exception as e:
        logger.warning("fusion.semantics: %s callable raised: %s", label, e)
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        logger.warning(
            "fusion.semantics: %s callable returned non-numeric %r",
            label, v,
        )
        return None
    return float(v)


def compute_flops(
    rule: "ModuleFusionRule",
    namespace: dict,
    child_ops: list,
) -> Optional[float]:
    """Compute fused-node FLOPs.

    Priority:
      1. ``rule.flops_callable`` (Python registration)
      2. ``rule.flops_formula`` (AST safe_eval)
      3. ``flops_kind == 'from_io'`` → ``2 * total_input_numel`` (rough)
      4. ``flops_kind == 'sum_children'`` → sum of ``child.annotations['flops']``
      5. otherwise → None
    """
    if rule.flops_callable is not None:
        return _safe_call("flops", rule.flops_callable, namespace)

    if rule.flops_formula:
        return _safe_eval_formula("flops", rule.flops_formula, namespace)

    kind = rule.flops_kind
    if kind == "from_io":
        if not child_ops:
            return None
        total = 0
        for op in child_ops:
            for t in op.inputs:
                shape = t.shape
                if shape:
                    n = 1
                    for d in shape:
                        n *= d
                    total += n
                else:
                    total += 1
        if total == 0:
            return None
        return float(2 * total)

    if kind == "sum_children":
        total = 0.0
        seen = False
        for op in child_ops:
            v = op.annotations.get("flops") if hasattr(op, "annotations") else None
            if v is None:
                continue
            try:
                total += float(v)
                seen = True
            except (TypeError, ValueError):
                continue
        return total if seen else None

    return None


def compute_memory(
    rule: "ModuleFusionRule",
    namespace: dict,
    child_ops: list,
) -> Optional[float]:
    """Compute fused-node memory bytes.

    Priority:
      1. ``rule.memory_callable``
      2. ``rule.memory_formula``
      3. ``flops_kind == 'from_io'`` (also drives memory) → sum input + output bytes
      4. otherwise → None

    Memory has no ``sum_children`` analogue; the per-op memory accounting
    is implicit in the children's own annotations and is simulator-side.
    """
    if rule.memory_callable is not None:
        return _safe_call("memory", rule.memory_callable, namespace)

    if rule.memory_formula:
        return _safe_eval_formula("memory", rule.memory_formula, namespace)

    if rule.flops_kind == "from_io":
        if not child_ops:
            return None
        total = 0
        for op in child_ops:
            for t in op.inputs:
                total += int(getattr(t, "mem_bytes", 0))
            for t in op.outputs:
                total += int(getattr(t, "mem_bytes", 0))
        return float(total) if total else None

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main entry: annotate_fused_node
# ─────────────────────────────────────────────────────────────────────────────

# Axes that get flattened onto annotations directly (in addition to sem_shape).
_FLATTEN_AXES: tuple[str, ...] = (
    "batch_size", "seq_len", "hidden_in", "hidden_out",
    "num_heads", "head_dim", "intermediate_dim",
    "vocab_size", "num_experts", "topk",
)


def _io_view_to_dict(v: TensorView) -> dict:
    return {"shape": list(v.shape), "dtype": v.dtype, "bytes": v.bytes}


def annotate_fused_node(
    fused_node: "OpNode",
    child_ops: list,
    rule: "ModuleFusionRule",
) -> None:
    """Write derived semantic fields into ``fused_node.annotations``.

    Annotations written (each prefixed with ``sem_`` to avoid collisions):
      - ``sem_io``           : ``{role: {shape, dtype, bytes}}``
      - ``sem_shape``        : ``{axis: value}``
      - ``sem_flops``        : ``float | None``
      - ``sem_memory_bytes`` : ``float | None``
      - ``sem_dtype``        : main activation dtype (``activation`` view if
                                present, else ``output``).

    Plus a flattened convenience layer (``batch_size``, ``seq_len``, ...)
    so simple downstream consumers don't need to dig through ``sem_shape``.

    Existing annotations (e.g. those already written by ``rule.annotations``
    or upstream passes) are NOT overwritten on the flat layer.

    All steps are wrapped in a top-level ``try`` — semantic derivation
    must never break the fusion pass.
    """
    try:
        ann = fused_node.annotations

        io_views = resolve_io_views(child_ops, rule)
        shape_axes = derive_shape_axes(io_views, rule)

        namespace: dict[str, Any] = {}
        namespace.update(io_views)
        namespace.update(shape_axes)

        flops = compute_flops(rule, namespace, child_ops)
        memory_bytes = compute_memory(rule, namespace, child_ops)

        ann["sem_io"] = {role: _io_view_to_dict(v) for role, v in io_views.items()}
        ann["sem_shape"] = dict(shape_axes)
        ann["sem_flops"] = flops
        ann["sem_memory_bytes"] = memory_bytes

        # Main activation dtype.
        main_view: Optional[TensorView] = (
            io_views.get("activation") or io_views.get("output")
        )
        if main_view is not None:
            ann["sem_dtype"] = main_view.dtype

        # Flatten common axes — do not overwrite anything already there
        # (e.g. values seeded by rule.annotations).
        for axis in _FLATTEN_AXES:
            if axis in shape_axes and axis not in ann:
                ann[axis] = shape_axes[axis]

    except Exception as e:  # pragma: no cover — last-resort guard
        logger.warning(
            "fusion.semantics: annotate_fused_node failed for op %s: %s",
            getattr(fused_node, "id", "?"), e,
        )
