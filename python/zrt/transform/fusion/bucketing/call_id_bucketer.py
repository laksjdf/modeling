"""Bucket OpGraph nodes by ``call_id`` (forward-call instance)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.ir.node import OpNode


# ── Intermediate group representation ────────────────────────────────────────

@dataclass
class FusionGroup:
    """A contiguous run of nodes sharing the same leaf module scope."""
    scope: str
    module_class: str
    module_class_obj: Optional[type]
    child_ops: list["OpNode"]
    leaf_attr: str
    call_id: int = 0
    is_full_forward: bool = False  # True iff bucket contains every op of the call


# ── Pass 1: bucket by leaf module ────────────────────────────────────────────

def bucket_nodes_by_leaf_module(graph: "OpGraph") -> list[FusionGroup]:
    """Group nodes by ``call_id`` (forward-call instance).

    Each module forward invocation has a unique ``call_id`` recorded by
    :class:`ModuleTracker` and stored on every dispatched op.  Grouping
    by ``call_id`` keeps fragments of the same parent scope (interleaved
    with child-module calls) **separate**, which is the only correct
    granularity for ``class_only`` rules — those rules assume the bucket
    represents an entire forward call.

    Communication ops and scopeless ops always become singleton groups.
    Within a single ``call_id``, ops are kept in topological order.
    Legacy graphs without ``call_id`` (e.g. captured before this field
    existed) fall back to ``(scope, module_class, layer)`` keying so old
    fixtures and torch.compile-mode captures still produce sensible
    groups.
    """
    groups: list[FusionGroup] = []
    current: Optional[FusionGroup] = None
    last_key: Any = None
    seen_call_ids_in_topo: set[int] = set()

    def _flush() -> None:
        nonlocal current
        current = None

    for node in graph.topo_sort():
        # Comm / scopeless nodes break the current group
        if node.category == "communication" or not node.scope:
            _flush()
            groups.append(FusionGroup(
                scope=node.scope,
                module_class=node.module_class,
                module_class_obj=None,
                child_ops=[node],
                leaf_attr=node.name,
                call_id=getattr(node, "call_id", 0) or 0,
                is_full_forward=False,
            ))
            last_key = None
            continue

        node_call_id = getattr(node, "call_id", 0) or 0
        if node_call_id > 0:
            key = ("call", node_call_id)
        else:
            # Legacy fallback: bucket by scope/class/layer.
            key = ("legacy", node.scope, node.module_class, node.layer)

        if current is not None and key == last_key:
            current.child_ops.append(node)
            continue

        current = FusionGroup(
            scope=node.scope,
            module_class=node.module_class,
            module_class_obj=_extract_class_obj(node),
            child_ops=[node],
            leaf_attr=node.scope.rsplit(".", 1)[-1] if node.scope else "",
            call_id=node_call_id,
            is_full_forward=False,
        )
        groups.append(current)
        last_key = key

    # Mark each group as "full-forward" iff every op tagged with that call_id
    # is contained in this single group (i.e. no child call interleaved).
    if any(g.call_id > 0 for g in groups):
        call_id_total: dict[int, int] = {}
        for n in graph.nodes.values():
            cid = getattr(n, "call_id", 0) or 0
            if cid:
                call_id_total[cid] = call_id_total.get(cid, 0) + 1
        for g in groups:
            if g.call_id > 0 and len(g.child_ops) == call_id_total.get(g.call_id, -1):
                g.is_full_forward = True

    return groups


# ── Pass 2: parent merge ─────────────────────────────────────────────────────

def _parent(scope: str) -> str:
    return scope.rsplit(".", 1)[0] if "." in scope else ""


def bucket_into_groups(
    graph: "OpGraph",
    *,
    max_parent_ops: int = 60,
    merge_sibling_classes: set[str] | None = None,
) -> list[FusionGroup]:
    """Bucket the graph into FusionGroups (leaf bucket + optional parent merge).

    ``merge_sibling_classes``
        When non-empty, identical-class sibling leaf groups under a common
        parent scope are concatenated (legacy behaviour, useful for MoE
        ``Expert`` lists).  When empty / ``None`` (the default), each
        forward-call instance stays in its own bucket — this is required
        for class-only rules to be gated correctly by ``is_full_forward``.
    """
    leaf_groups = bucket_nodes_by_leaf_module(graph)
    if merge_sibling_classes:
        return _merge_parent_groups(
            leaf_groups, graph,
            max_parent_ops=max_parent_ops,
            allowed_classes=merge_sibling_classes,
        )
    return leaf_groups


def _merge_parent_groups(
    leaf_groups: list[FusionGroup],
    graph: "OpGraph",
    *,
    max_parent_ops: int = 60,
    allowed_classes: set[str] | None = None,
) -> list[FusionGroup]:
    """Merge consecutive leaf groups that share a common parent scope.

    ``allowed_classes`` restricts merging to specific module classes.  When
    ``None`` every class is eligible (legacy behaviour).
    """
    if not leaf_groups:
        return []

    merged: list[FusionGroup] = []
    i = 0
    while i < len(leaf_groups):
        group = leaf_groups[i]

        if len(group.child_ops) <= 1 or not group.scope:
            merged.append(group)
            i += 1
            continue

        if allowed_classes is not None and group.module_class not in allowed_classes:
            merged.append(group)
            i += 1
            continue

        parent_scope = _parent(group.scope)
        if not parent_scope:
            merged.append(group)
            i += 1
            continue

        total_ops = len(group.child_ops)
        j = i + 1
        while j < len(leaf_groups):
            next_group = leaf_groups[j]
            if not next_group.scope:
                break
            if _parent(next_group.scope) != parent_scope:
                break
            if next_group.module_class != group.module_class:
                break
            if total_ops + len(next_group.child_ops) > max_parent_ops:
                break
            total_ops += len(next_group.child_ops)
            group = _merge_two_groups(group, next_group)
            j += 1

        merged.append(group)
        i = j

    return merged


def _merge_two_groups(a: FusionGroup, b: FusionGroup) -> FusionGroup:
    return FusionGroup(
        scope=a.scope,
        module_class=a.module_class,
        module_class_obj=a.module_class_obj,
        child_ops=a.child_ops + b.child_ops,
        leaf_attr=a.leaf_attr,
    )


def _extract_class_obj(node: "OpNode") -> Optional[type]:
    obj = node.annotations.get("module_class_obj")
    if isinstance(obj, type):
        return obj
    return None
