"""Construct fused / collapsed :class:`OpNode` from a :class:`FusionGroup`."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .annotation_propagator import _propagated_annotations
from .io_resolver import (
    FusedIOPort,
    _child_ops_external_io,
    _external_io,
    resolve_io,
    resolve_io_tensors,
)

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.ir.node import OpNode
    from python.zrt.transform.fusion.bucketing.call_id_bucketer import FusionGroup
    from python.zrt.transform.fusion.core.rule import ModuleFusionRule


def build_fused_node(
    group: "FusionGroup",
    rule: "ModuleFusionRule",
    graph: "OpGraph",
    fuse_idx: int,
) -> "OpNode":
    """Construct a fused OpNode from a FusionGroup and its matched rule."""
    from python.zrt.ir.node import OpNode
    from python.zrt.transform.fusion.semantics import annotate_fused_node

    first = group.child_ops[0]
    tc = rule.target_class
    op_type = rule.op_type or (
        tc.__name__ if isinstance(tc, type) else str(tc)
    )

    # Collect source op_types (deduplicated, ordered)
    fused_from = list(dict.fromkeys(op.op_type for op in group.child_ops))

    # External IO — rule-declared takes precedence, child_ops fallback
    # fills any side the rule left empty.  Edge-based ``_external_io``
    # is intentionally not used here: it misses placeholder tensors
    # (model weights, ``input_ids``, RMSNorm gamma) that have no
    # producer ``OpNode``.
    if rule.io_roles:
        ext_inputs, ext_outputs = resolve_io_tensors(group.child_ops, rule)
        if not ext_inputs or not ext_outputs:
            fb_in, fb_out = _child_ops_external_io(group.child_ops)
            if not ext_inputs:
                ext_inputs = fb_in
            if not ext_outputs:
                ext_outputs = fb_out
    else:
        ext_inputs, ext_outputs = _child_ops_external_io(group.child_ops)

    # Build provenance from ``rule.io_roles``.  ``rule.inputs`` and
    # ``rule.outputs`` are always cleared by ``ModuleFusionRule.__post_init__``
    # (merged into ``io_roles``), so iterating them was dead code.
    provenance_parts: list[FusedIOPort] = []
    for spec in rule.io_roles:
        port = resolve_io(group.child_ops, spec)
        if port is not None:
            provenance_parts.append(port)

    propagated = _propagated_annotations(group)
    level = "parent" if len(group.child_ops) > 3 else "leaf"

    node = OpNode(
        id=f"fused_{fuse_idx}_{first.id}",
        op_type=op_type,
        inputs=ext_inputs,
        outputs=ext_outputs,
        scope=group.scope,
        category=first.category,
        module_class=group.module_class,
        layer=first.layer,
        component=first.component,
        fused_from=fused_from,
        num_sub_ops=len(group.child_ops),
        fusion_level=level,
        name=group.leaf_attr,
        provenance=tuple(provenance_parts),
    )
    node.annotations.update(propagated)
    node.annotations.update(rule.annotations)
    # Track which raw OpNodes were absorbed so the Excel export can show
    # the Raw-Operators sheet IDs that fused into this node.
    node.annotations["source_op_ids"] = [op.id for op in group.child_ops]
    node.annotations["fused_by_rule"] = rule.name
    # Derive semantic fields (batch_size / seq_len / dtype / flops / memory)
    # from rule.io_roles + shape_derivation + flops/memory formulas and
    # write them into node.annotations.  Failures are logged, not raised.
    annotate_fused_node(node, group.child_ops, rule)
    return node


def _build_collapsed_node(
    group: "FusionGroup",
    graph: "OpGraph",
    fuse_idx: int,
) -> "OpNode":
    """Plain structural collapse: ``op_type = module_class``, no rule fields.

    Used when a multi-op (scope, module_class) bucket has no matching
    fusion rule.  Records absorbed children in ``fused_from`` and
    propagates split-stage annotations that all children agree on.
    """
    from python.zrt.ir.node import OpNode

    first = group.child_ops[0]
    group_ids = {op.id for op in group.child_ops}
    ext_inputs, ext_outputs = _external_io(graph, group_ids)
    fused_from = list(dict.fromkeys(op.op_type for op in group.child_ops))
    propagated = _propagated_annotations(group)
    level = "parent" if len(group.child_ops) > 3 else "leaf"

    node = OpNode(
        id=f"collapsed_{fuse_idx}_{first.id}",
        op_type=group.module_class,
        inputs=ext_inputs,
        outputs=ext_outputs,
        scope=group.scope,
        category=first.category,
        module_class=group.module_class,
        layer=first.layer,
        component=first.component,
        fused_from=fused_from,
        num_sub_ops=len(group.child_ops),
        fusion_level=level,
        name=group.leaf_attr,
    )
    node.annotations.update(propagated)
    node.annotations["source_op_ids"] = [op.id for op in group.child_ops]
    node.annotations["fused_by_rule"] = "_collapsed"
    return node
