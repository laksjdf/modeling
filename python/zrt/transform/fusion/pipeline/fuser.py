"""Top-level fusion entry point: ``MultiPassFuser`` + ``fuse()``.

``fuse()`` is a fixed-point loop.  Each pass buckets the graph and,
for each bucket, tries:

1. Whole-bucket match against the rule set.
2. If no whole-bucket match: sliding-window partial scan
   (``SlidingWindowScanner``) to fuse contiguous subsequences.
3. If neither matches and ``allow_structural_collapse`` is enabled,
   collapse the bucket as a legacy escape hatch.

The loop terminates when graph node count stops shrinking, or after
``MAX_PASSES`` iterations.  Add+Norm composition runs exactly once
after the loop.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from python.zrt.transform.fusion.bucketing.call_id_bucketer import (
    FusionGroup,
    bucket_into_groups,
)
from python.zrt.transform.fusion.building.node_builder import (
    _build_collapsed_node,
    build_fused_node,
)
from python.zrt.transform.fusion.matching.sliding_window import (
    SlidingWindowScanner,
)
from python.zrt.transform.fusion.registry import (
    iter_active_rules,
    lookup_rule,
)

from .compositors import _ADD_NORM_RULE_NAME, _compose_add_norm

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.fusion.registry.rule_registry import RuleRegistry


MAX_PASSES = 5


# ── Public entry: bucket → (rule | sliding-window | collapse) → Add+Norm ────


def fuse(graph: "OpGraph", ctx=None) -> "OpGraph":
    """Bucket the graph and rewrite each bucket via whole-bucket match
    or sliding-window scan; iterate until fixed point (max ``MAX_PASSES``).

    Each pass repeats: bucket → match → rewrite.  This allows fused
    nodes from pass N (e.g. a 4-op ``rms_coef``) to participate in
    larger patterns at pass N+1 (e.g. ``hc_pre_attn`` = ``rms_coef +
    mm + ...``).

    ``class_only`` rules are only considered when the bucket spans the
    entire forward call; they cannot fire on fragments.

    No structural collapse fires unless
    ``ctx.fusion.allow_structural_collapse=True``.

    Add+Norm composition runs at the end (outside the loop), gated by
    the ``"add_norm"`` rule being in the active set.
    """
    from python.zrt.transform.context import FusionConfig

    fusion_cfg = getattr(ctx, "fusion", None) or FusionConfig()
    phase = (ctx.phase_for_fusion()
             if ctx is not None and hasattr(ctx, "phase_for_fusion")
             else "inference")
    active = iter_active_rules(fusion_cfg, phase)
    active_names = {r.name for r in active}
    scanner = SlidingWindowScanner(active)

    working_graph = graph.clone()
    fuse_idx = 0
    for _pass_n in range(MAX_PASSES):
        previous_count = len(working_graph.nodes)
        groups = bucket_into_groups(
            working_graph,
            merge_sibling_classes=set(fusion_cfg.merge_sibling_classes),
        )
        for group in groups:
            fuse_idx = _fuse_group(
                working_graph, group, active, fusion_cfg, scanner, fuse_idx,
            )
        if len(working_graph.nodes) >= previous_count:
            break  # fixed point — no progress this pass

    if _ADD_NORM_RULE_NAME in active_names:
        working_graph = _compose_add_norm(working_graph)
    return working_graph


def _fuse_group(graph, group, active, fusion_cfg, scanner, fuse_idx):
    """Apply the best matching strategy to a single bucket.

    Returns the updated ``fuse_idx`` so the caller can continue
    numbering fused nodes uniquely.
    """
    # ── Singleton bucket ────────────────────────────────────────────────
    if len(group.child_ops) <= 1:
        node = group.child_ops[0]
        if node.category == "communication" or not node.module_class:
            return fuse_idx
        rule = lookup_rule(
            (node.op_type,),
            module_class=node.module_class,
            active_rules=active,
        )
        if rule is None:
            return fuse_idx
        if rule.pattern is None or rule.pattern.kind != "class_only":
            return fuse_idx
        # class_only rules require the bucket to span the full forward
        # call.  A singleton bucket only does so when no child fired.
        if not group.is_full_forward:
            return fuse_idx
        fused = build_fused_node(group, rule, graph, fuse_idx)
        _propagate_call_id_and_provenance(fused, group)
        fuse_idx += 1
        graph.replace_subgraph({node.id}, fused)
        return fuse_idx

    # ── Multi-op bucket ─────────────────────────────────────────────────
    if not group.scope or not group.module_class:
        return fuse_idx

    operator_sequence = tuple(op.op_type for op in group.child_ops)
    rule = lookup_rule(
        operator_sequence,
        module_class=group.module_class,
        active_rules=active,
    )
    # ``class_only`` rules only verify group size; they must not fire on
    # fragments of a forward call (e.g. the inline RMSNorm sub-block of
    # Attention.forward) — the resulting node would be tagged with the
    # parent's semantic op_type and produce wrong shape/FLOPs.
    if (rule is not None
            and rule.pattern is not None
            and rule.pattern.kind == "class_only"
            and not group.is_full_forward):
        rule = None

    if rule is not None:
        group_ids = {op.id for op in group.child_ops}
        replacement = build_fused_node(group, rule, graph, fuse_idx)
        _propagate_call_id_and_provenance(replacement, group)
        fuse_idx += 1
        graph.replace_subgraph(group_ids, replacement)
        return fuse_idx

    # ── Sliding-window partial-match fallback ───────────────────────────
    return _apply_partial_matches(graph, group, scanner, fusion_cfg, fuse_idx)


def _apply_partial_matches(graph, group, scanner, fusion_cfg, fuse_idx):
    """Greedy partial-match within one bucket via ``SlidingWindowScanner``.

    Replaces each matched contiguous subsequence with a fused node.
    When no partial match is found, falls back to the legacy
    ``allow_structural_collapse`` escape hatch (if enabled).
    """
    operator_types = [op.op_type for op in group.child_ops]
    matches = scanner.scan(operator_types)
    if not matches:
        # No partial match either — leave bucket as raw aten unless the
        # legacy structural-collapse escape hatch is enabled.
        if fusion_cfg.allow_structural_collapse:
            group_ids = {op.id for op in group.child_ops}
            replacement = _build_collapsed_node(group, graph, fuse_idx)
            fuse_idx += 1
            graph.replace_subgraph(group_ids, replacement)
        return fuse_idx

    for rule, start, end in matches:
        sub_ops = group.child_ops[start:end]
        sub_group = FusionGroup(
            scope=sub_ops[0].scope,
            module_class=sub_ops[0].module_class,
            module_class_obj=None,
            child_ops=list(sub_ops),
            leaf_attr=sub_ops[0].name,
            call_id=group.call_id,
            is_full_forward=False,
        )
        replacement = build_fused_node(sub_group, rule, graph, fuse_idx)
        _propagate_call_id_and_provenance(replacement, sub_group)
        fuse_idx += 1
        graph.replace_subgraph({op.id for op in sub_ops}, replacement)
    return fuse_idx


def _propagate_call_id_and_provenance(replacement, group):
    """Carry call_id forward and flatten source_op_ids/fused_from across passes.

    Without this, a node fused during pass 2 from a group whose children
    include a pass-1 fused node would record the pass-1 node's id (e.g.
    ``fused_0_op_0``) in its ``source_op_ids`` — that id no longer exists
    in the raw graph, breaking the replace_subgraph integrity check.
    """
    if group.call_id:
        replacement.call_id = group.call_id

    flat_ids: list[str] = []
    flat_op_types: list[str] = []
    seen_ids: set[str] = set()
    seen_types: set[str] = set()
    for child in group.child_ops:
        child_src_ids = child.annotations.get("source_op_ids") or []
        if child_src_ids:
            for sid in child_src_ids:
                if sid not in seen_ids:
                    seen_ids.add(sid)
                    flat_ids.append(sid)
        else:
            if child.id not in seen_ids:
                seen_ids.add(child.id)
                flat_ids.append(child.id)
        # Flatten fused_from too: prefer the child's own fused_from
        # (which is already flattened raw op_types) when present.
        child_fused_from = list(child.fused_from) if child.fused_from else [child.op_type]
        for ot in child_fused_from:
            if ot not in seen_types:
                seen_types.add(ot)
                flat_op_types.append(ot)

    replacement.annotations["source_op_ids"] = flat_ids
    replacement.fused_from = flat_op_types
    # Recount sub-ops to reflect total absorbed raw ops (across passes).
    replacement.num_sub_ops = len(flat_ids)


# ─────────────────────────────────────────────────────────────────────────────
# OO entry-point — delegates to the module-level ``fuse``.
# ─────────────────────────────────────────────────────────────────────────────

class MultiPassFuser:
    """OO entry-point for fusion.

    Delegates to the module-level :func:`fuse` for the fixed-point loop
    with sliding-window partial matching.
    """

    def __init__(self, registry: "RuleRegistry | None" = None) -> None:
        # ``registry`` is accepted for forward compatibility but the
        # underlying ``fuse`` still uses the process-wide singleton via
        # ``iter_active_rules`` / ``lookup_rule``.
        self._registry = registry

    def fuse(self, graph: "OpGraph", ctx=None) -> "OpGraph":
        return fuse(graph, ctx)
