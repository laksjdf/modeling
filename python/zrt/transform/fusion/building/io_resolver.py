"""FusedIOPort + IO resolution.

Two complementary IO paths exist:

* ``resolve_io`` / ``resolve_io_tensors`` — declarative, driven by
  ``rule.io_roles``.  Reads ``child_ops[op].inputs[arg]`` directly so
  placeholder tensors (model weights, ``input_ids``, RMSNorm gamma)
  are captured even though they have no producer ``OpNode`` / ``Edge``.
  This is the primary path used by ``build_fused_node``.
* ``_child_ops_external_io`` — fallback when a rule has no ``io_roles``.
  Walks each child op's ``.inputs`` / ``.outputs`` and computes external
  IO by ``tensor.id`` set-difference.  Also placeholder-safe.
* ``_external_io`` — legacy edge-based derivation.  Misses placeholders.
  Retained for ``_build_collapsed_node``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from python.zrt.transform.fusion.core.io_role import IOSpec
    from python.zrt.transform.fusion.core.rule import ModuleFusionRule


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FusedIOPort:
    """Provenance of one tensor in a fused node's inputs or outputs."""

    role: str                                    # "activation" / "weight" / "output" / ...
    origin_node_id: str                          # source child OpNode.id
    origin_op_type: str                          # e.g. "aten.mm.default"
    origin_arg_index: int                        # which tensor slot
    origin_kind: Literal["input", "output"]      # taken from input or output side
    origin_dtype_from: str = "explicit"          # how dtype was determined


def resolve_io(
    child_ops: list,
    spec: "IOSpec",
) -> Optional[FusedIOPort]:
    """Build a FusedIOPort from an IOSpec and the child ops of a fusion group.

    Returns ``None`` if indices are out of range.
    """
    if not child_ops:
        return None

    op_idx = spec.source_op_index
    if op_idx < 0:
        op_idx = len(child_ops) + op_idx
    if op_idx < 0 or op_idx >= len(child_ops):
        return None

    op = child_ops[op_idx]

    arg_idx = spec.source_arg_index
    tensors = op.inputs if spec.source_kind == "input" else op.outputs
    if arg_idx < 0:
        arg_idx = len(tensors) + arg_idx
    if arg_idx < 0 or arg_idx >= len(tensors):
        return None

    return FusedIOPort(
        role=spec.role,
        origin_node_id=op.id,
        origin_op_type=op.op_type,
        origin_arg_index=arg_idx,
        origin_kind=spec.source_kind,
    )


def _external_io(
    graph,
    group_ids: set,
) -> tuple[list, list]:
    """Return (ext_inputs, ext_outputs) for a node group via graph edges.

    Legacy edge-based derivation.  Misses placeholder tensors (weights,
    ``input_ids``, gamma) that have no producer ``OpNode``.  Retained
    for ``_build_collapsed_node``; new code should prefer
    :func:`_child_ops_external_io` or :func:`resolve_io_tensors`.
    """
    seen_in: set = set()
    seen_out: set = set()
    inputs: list = []
    outputs: list = []

    for e in graph.edges:
        if e.src not in group_ids and e.dst in group_ids:
            key = (e.tensor_id, e.dst_idx)
            if key not in seen_in and e.tensor is not None:
                seen_in.add(key)
                inputs.append(e.tensor)
        if e.src in group_ids and e.dst not in group_ids:
            key = (e.tensor_id, e.src_idx)
            if key not in seen_out and e.tensor is not None:
                seen_out.add(key)
                outputs.append(e.tensor)

    return inputs, outputs


def _normalize_idx(idx: int, length: int) -> int:
    return idx if idx >= 0 else length + idx


def resolve_io_tensors(
    child_ops: list,
    rule: "ModuleFusionRule",
) -> tuple[list, list]:
    """Return (inputs, outputs) TensorMeta lists derived from ``rule.io_roles``.

    Walks ``rule.io_roles`` in declaration order.  For each role:

    * ``source_kind == "input"``  → append ``child_ops[op].inputs[arg]``
      to the inputs list.
    * ``source_kind == "output"`` → append ``child_ops[op].outputs[arg]``
      to the outputs list.

    Negative indices are normalized.  Out-of-range indices or missing
    tensors are logged and skipped (never raised — fusion must continue
    even when a rule is malformed).  Duplicates are removed by
    ``tensor.id``, preserving first occurrence.

    Mirrors ``semantics.annotator.resolve_io_views`` but returns raw
    ``TensorMeta`` instead of ``TensorView`` so the output can populate
    ``OpNode.inputs`` / ``OpNode.outputs`` directly.
    """
    inputs: list = []
    outputs: list = []
    if not child_ops:
        return inputs, outputs

    seen_in: set = set()
    seen_out: set = set()

    for spec in rule.io_roles:
        op_idx = _normalize_idx(spec.source_op_index, len(child_ops))
        if op_idx < 0 or op_idx >= len(child_ops):
            logger.warning(
                "fusion.io: role %r source_op_index %d out of range "
                "(%d ops); skipping",
                spec.role, spec.source_op_index, len(child_ops),
            )
            continue
        op = child_ops[op_idx]
        tensors = op.inputs if spec.source_kind == "input" else op.outputs
        if not tensors:
            logger.warning(
                "fusion.io: role %r — op %s has no %s tensors; skipping",
                spec.role, op.id, spec.source_kind,
            )
            continue
        arg_idx = _normalize_idx(spec.source_arg_index, len(tensors))
        if arg_idx < 0 or arg_idx >= len(tensors):
            logger.warning(
                "fusion.io: role %r source_arg_index %d out of range "
                "(%d %s tensors on op %s); skipping",
                spec.role, spec.source_arg_index, len(tensors),
                spec.source_kind, op.id,
            )
            continue
        meta = tensors[arg_idx]
        if spec.source_kind == "input":
            if meta.id in seen_in:
                continue
            seen_in.add(meta.id)
            inputs.append(meta)
        else:
            if meta.id in seen_out:
                continue
            seen_out.add(meta.id)
            outputs.append(meta)

    return inputs, outputs


def _child_ops_external_io(child_ops: list) -> tuple[list, list]:
    """Fallback IO when ``rule.io_roles`` is empty or partial.

    Walks ``child_ops`` in order and classifies each tensor by id:

    * input  := tensor in ``child.inputs`` whose id is NOT produced by
      any child (i.e. crosses the group boundary or is a placeholder).
    * output := tensor in ``child.outputs`` whose id is NOT consumed by
      any child (i.e. consumed outside the group, or terminal).

    When the strict "no internal consumer" rule leaves outputs empty
    (e.g. ``rms_coef`` whose only meaningful product IS consumed inside
    a downstream group later), fall back to the last child's outputs.
    Order is preserved by traversal; duplicates are removed by
    ``tensor.id``.
    """
    inputs: list = []
    outputs: list = []
    if not child_ops:
        return inputs, outputs

    internal_produced: set = set()
    internal_consumed: set = set()
    for op in child_ops:
        for t in op.outputs:
            internal_produced.add(t.id)
        for t in op.inputs:
            internal_consumed.add(t.id)

    seen_in: set = set()
    seen_out: set = set()
    for op in child_ops:
        for t in op.inputs:
            if t.id in internal_produced or t.id in seen_in:
                continue
            seen_in.add(t.id)
            inputs.append(t)
        for t in op.outputs:
            if t.id in internal_consumed or t.id in seen_out:
                continue
            seen_out.add(t.id)
            outputs.append(t)

    if not outputs:
        # Terminal-product fallback: every output consumed inside the
        # group — surface the last child's outputs so the fused node is
        # still self-describing.
        for t in child_ops[-1].outputs:
            if t.id in seen_out:
                continue
            seen_out.add(t.id)
            outputs.append(t)

    return inputs, outputs
