"""FusionCore: platform-aware operator fusion algorithm.

This is the **single source of truth** for all fusion logic.  Both
FusionPass (OpGraph IR) and FusionEngine (Dict records for Excel export)
delegate to the functions defined here.

Algorithm (5 passes):
  Pass 1 (leaf)    – group consecutive same-scope+layer+phase compute nodes.
  Pass 2 (parent)  – merge consecutive leaf groups sharing a fusible parent.
  Pass 3a (label)  – assign semantic label via sub-pattern + module_class lookup.
  Pass 3b (add_norm) – cross-boundary Add+Norm → AddRMSNorm detection.
  Pass 4 (expand)  – expand unfused container groups back to individual items.

Combines advantages from both legacy implementations:
  - FusionPass:  comm-node breaking, phase isolation, empty-scope handling.
  - FusionEngine: Add+Norm cross-boundary fusion, max_leaf_ops cap,
    correct container expansion (generates independent child rows).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from python.zrt.transform.fusion.rules import (
    CONTAINER_SEMANTICS,
    PATTERN_SKIP,
    get_platform_settings,
    get_semantic_label,
    get_subpatterns,
    match_subsequence,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Minimal data structure used internally by all passes.
# Callers create these from their native representation (OpNode or Dict).
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FusionItem:
    """A single item in the fusion pipeline.

    Attributes mirror the union of what OpNode and Dict-record provide.
    """
    scope: str                    # module_path / OpNode.scope
    module_class: str
    op_type: str                  # aten op name or current fused label
    layer: str
    phase: str                    # prefill / decode / forward / backward
    category: str                 # "compute" | "communication" | "memory"
    num_sub_ops: int = 1
    children: list = field(default_factory=list)  # raw child records / nodes
    input_ids: list = field(default_factory=list)
    output_ids: list = field(default_factory=list)
    annotations: dict = field(default_factory=dict)
    _meta: dict = field(default_factory=dict)  # caller-specific extras


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parent(scope: str) -> str:
    return scope.rsplit(".", 1)[0] if "." in scope else ""


def _effective_ops(group: List[FusionItem]) -> List[str]:
    """Aten op names with PATTERN_SKIP ops removed."""
    return [item.op_type for item in group if item.op_type not in PATTERN_SKIP]


# ─────────────────────────────────────────────────────────────────────────────
# Pass 1: Leaf grouping
# ─────────────────────────────────────────────────────────────────────────────

def _pass1_leaf(
    items: List[FusionItem],
    *,
    max_leaf_ops: int = 0,
) -> List[List[FusionItem]]:
    """Group consecutive compute+memory nodes with identical scope+layer+phase.

    Advantages preserved:
    - **FusionPass**: communication nodes and empty-scope nodes always break
      groups (become standalone single-item groups).
    - **FusionPass**: phase is part of the group key, preventing forward/
      backward boundary crossing.
    - **FusionEngine**: ``max_leaf_ops`` cap prevents backward graphs (where
      module attribution is unreliable) from fusing unrelated ops.
    """
    if not items:
        return []

    effective_max = max_leaf_ops or 60  # generous default
    groups: List[List[FusionItem]] = []
    current: List[FusionItem] = []

    for item in items:
        # Comm nodes and scopeless nodes are always standalone group-breakers
        if item.category == "communication" or not item.scope:
            if current:
                groups.append(current)
                current = []
            groups.append([item])
            continue

        if current:
            first = current[0]
            same_scope = item.scope == first.scope
            same_layer = item.layer == first.layer
            same_phase = item.phase == first.phase
            within_cap = len(current) < effective_max

            if same_scope and same_layer and same_phase and within_cap:
                current.append(item)
                continue

            groups.append(current)
            current = []

        current = [item]

    if current:
        groups.append(current)

    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Pass 2: Parent merge
# ─────────────────────────────────────────────────────────────────────────────

def _pass2_parent(
    leaf_groups: List[List[FusionItem]],
    path_to_class: Dict[str, str],
    path_to_children: Dict[str, Set[str]],
    *,
    max_parent_ops: int,
    max_children: int,
) -> List[List[FusionItem]]:
    """Merge consecutive leaf groups that share a fusible common parent scope.

    Two rules (identical in both legacy implementations):
    - Rule 1: root-level modules (no parent) are structural wrappers, never fuse.
    - Rule 2: if any child scope has sub-children, the parent is structural.
    """
    if not leaf_groups:
        return []

    # Build per-parent stats
    parent_child_scopes: Dict[str, Set[str]] = {}
    parent_total_ops: Dict[str, int] = {}

    for g in leaf_groups:
        scope = g[0].scope
        if not scope:
            continue
        p = _parent(scope)
        if p:
            parent_child_scopes.setdefault(p, set()).add(scope)
            parent_total_ops[p] = parent_total_ops.get(p, 0) + len(g)

    def _is_fusible(p: str) -> bool:
        if p not in path_to_class:
            return False
        if p not in path_to_children:
            return False
        if len(parent_child_scopes.get(p, set())) > max_children:
            return False
        if parent_total_ops.get(p, 0) > max_parent_ops:
            return False
        # Rule 1: root-level
        if not _parent(p):
            return False
        # Rule 2: children with sub-children
        children = path_to_children.get(p, set())
        if any(child in path_to_children for child in children):
            return False
        return True

    def _phase(g: List[FusionItem]) -> str:
        return g[0].phase if g else ""

    result: List[List[FusionItem]] = []
    i = 0
    while i < len(leaf_groups):
        g = leaf_groups[i]
        scope = g[0].scope if g else ""
        p = _parent(scope) if scope else ""
        layer = g[0].layer if g else ""
        phase = _phase(g)

        # Comm/scopeless groups are never merged upward
        if not scope or g[0].category == "communication":
            result.append(g)
            i += 1
            continue

        if p and _is_fusible(p):
            j = i + 1
            total_ops = len(g)
            while j < len(leaf_groups):
                ng = leaf_groups[j]
                nscope = ng[0].scope if ng else ""
                np_ = _parent(nscope) if nscope else ""
                if ((np_ == p or nscope == p)
                        and ng[0].layer == layer
                        and _phase(ng) == phase):
                    total_ops += len(ng)
                    if total_ops > max_parent_ops:
                        break
                    j += 1
                else:
                    break

            if j > i + 1:
                merged: List[FusionItem] = []
                for g2 in leaf_groups[i:j]:
                    merged.extend(g2)
                result.append(merged)
                i = j
                continue

        result.append(g)
        i += 1

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pass 3a: Semantic label
# ─────────────────────────────────────────────────────────────────────────────

def _semantic_label(
    group: List[FusionItem],
    path_to_class: Dict[str, str],
    platform: str,
) -> str:
    """Determine the fused op_type label for a group.

    Priority: sub-patterns (most specific) > semantic label > module_class.
    """
    scope = group[0].scope
    mc = path_to_class.get(scope, group[0].module_class)
    op_types = [item.op_type for item in group]

    # 1. Sub-pattern: specific class + op-sequence match (highest priority)
    for sp in get_subpatterns(platform):
        if sp.matches_class(mc) and sp.matches_ops(op_types):
            return sp.name

    # 2. Semantic label from module class name
    label = get_semantic_label(mc) if mc else None
    if label:
        return label

    # 3. Fallback: module class or first op_type
    return mc if mc else group[0].op_type


# ─────────────────────────────────────────────────────────────────────────────
# Pass 3b: Add+Norm cross-boundary fusion (FusionEngine advantage)
# ─────────────────────────────────────────────────────────────────────────────

def _is_add_norm_pair(
    g_prev: List[FusionItem],
    g_norm: List[FusionItem],
    path_to_class: Dict[str, str],
) -> bool:
    """Return True when *g_prev* ends with a residual-add and *g_norm* is a norm."""
    # Determine semantic labels
    norm_mc = path_to_class.get(g_norm[0].scope, g_norm[0].module_class)
    norm_label = get_semantic_label(norm_mc) if norm_mc else ""
    if norm_label not in ("rms_norm", "layer_norm"):
        return False

    # Check g_prev contains a residual add
    prev_ops = _effective_ops(g_prev)
    if not any("add" in op for op in prev_ops):
        return False

    # g_prev must be the module-hierarchy parent of g_norm
    norm_parent = _parent(g_norm[0].scope) if g_norm[0].scope else ""
    if not norm_parent or norm_parent != g_prev[0].scope:
        return False
    return True


def _merge_add_norm(
    g_add: List[FusionItem],
    g_norm: List[FusionItem],
    path_to_class: Dict[str, str],
) -> FusionItem:
    """Merge a residual-add group and a norm group into an Add+Norm fused item."""
    all_children: list = []
    for item in g_add + g_norm:
        all_children.extend(item.children if item.children else [item])

    norm_mc = path_to_class.get(g_norm[0].scope, g_norm[0].module_class)
    norm_label = get_semantic_label(norm_mc) if norm_mc else "rms_norm"
    label = "add_rms_norm" if norm_label == "rms_norm" else "add_layer_norm"

    return FusionItem(
        scope=g_add[0].scope,
        module_class=path_to_class.get(g_add[0].scope, g_add[0].module_class),
        op_type=label,
        layer=g_add[0].layer,
        phase=g_add[0].phase,
        category=g_add[0].category,
        num_sub_ops=len(g_add) + len(g_norm),
        children=all_children,
        annotations={},
    )


def _pass3b_add_norm(
    groups: List[List[FusionItem]],
    enabled: bool,
    path_to_class: Dict[str, str],
) -> List[List[FusionItem]]:
    """Scan for adjacent (add-group, norm-group) pairs and merge them."""
    if not enabled:
        return groups

    result: List[List[FusionItem]] = []
    i = 0
    while i < len(groups):
        if (i + 1 < len(groups)
                and _is_add_norm_pair(groups[i], groups[i + 1], path_to_class)):
            result.append([_merge_add_norm(groups[i], groups[i + 1], path_to_class)])
            i += 2
        else:
            result.append(groups[i])
            i += 1
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pass 4: Expand unfused containers (FusionEngine correct way)
# ─────────────────────────────────────────────────────────────────────────────

def _pass4_expand_containers(
    groups: List[List[FusionItem]],
    platform: str,
    path_to_class: Dict[str, str],
) -> List[List[FusionItem]]:
    """Expand multi-op container groups not matched by any hardware subpattern.

    Uses FusionEngine's correct approach: generate independent child rows
    rather than clearing metadata (the fragile FusionPass approach).
    """
    patterns = get_subpatterns(platform)
    result: List[List[FusionItem]] = []

    for group in groups:
        if len(group) <= 1:
            result.append(group)
            continue

        scope = group[0].scope
        mc = path_to_class.get(scope, group[0].module_class)
        label = _semantic_label(group, path_to_class, platform)

        # If label is still in CONTAINER_SEMANTICS, no hardware pattern matched
        if label in CONTAINER_SEMANTICS:
            # Check if any subpattern matches the class (class-only match)
            matched = any(sp.matches_class(mc) for sp in patterns)
            if not matched:
                # Expand: each child becomes its own single-item group
                for child in (group[0].children if group[0].children else group):
                    if isinstance(child, FusionItem):
                        result.append([child])
                    else:
                        # Raw child record: wrap in minimal FusionItem
                        result.append([FusionItem(
                            scope=group[0].scope,
                            module_class=mc,
                            op_type=child.get("op_type", "unknown") if isinstance(child, dict) else str(child),
                            layer=group[0].layer,
                            phase=group[0].phase,
                            category=group[0].category,
                            num_sub_ops=1,
                        )])
                continue

        result.append(group)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_fusion(
    items: List[FusionItem],
    *,
    path_to_class: Dict[str, str],
    path_to_children: Dict[str, Set[str]],
    platform: str = "generic",
    max_leaf_ops: int = 0,
    add_norm_fusion: bool = False,
    debug: bool = False,
) -> List[List[FusionItem]]:
    """Run the full 5-pass fusion pipeline with error boundaries.

    Each pass is wrapped in try/except -- a failure in one pass leaves the
    groups from the previous pass intact and logs a warning.

    Parameters
    ----------
    items : List[FusionItem]
        Items in topological order.
    path_to_class / path_to_children :
        Module hierarchy maps (from ModuleTracker or rebuilt from OpNode.scope).
    platform :
        One of "cuda", "ascend_npu", "cpu", "generic".
    max_leaf_ops :
        Cap leaf group size (0 = use default 60). Set low (e.g. 15) for
        backward graphs where module attribution is unreliable.
    add_norm_fusion :
        Whether to detect cross-boundary Add+Norm -> AddRMSNorm.
    debug :
        If True, log detailed fusion decisions.

    Returns
    -------
    List[List[FusionItem]]
        Final fusion groups. Each group should be replaced by a single fused
        node/record by the caller.
    """
    cfg = get_platform_settings(platform)
    max_parent_ops = cfg["max_parent_ops"]
    max_children = cfg["max_children"]

    if debug:
        logger.info("[fusion-debug] Platform: %s | Input items: %d",
                    platform, len(items))

    # Pass 1: Leaf grouping
    try:
        leaf_groups = _pass1_leaf(items, max_leaf_ops=max_leaf_ops)
        if debug:
            logger.info("[fusion-debug] Pass 1 (leaf): %d → %d groups",
                        len(items), len(leaf_groups))
    except Exception as e:
        logger.warning("Fusion Pass 1 (leaf) failed: %s. Returning raw items.",
                       e, exc_info=True)
        return [[item] for item in items]

    # Pass 2: Parent merge
    try:
        n_before = len(leaf_groups)
        final_groups = _pass2_parent(
            leaf_groups, path_to_class, path_to_children,
            max_parent_ops=max_parent_ops, max_children=max_children,
        )
        if debug:
            logger.info("[fusion-debug] Pass 2 (parent): %d → %d groups",
                        n_before, len(final_groups))
    except Exception as e:
        logger.warning("Fusion Pass 2 (parent) failed: %s. Keeping leaf groups.",
                       e, exc_info=True)
        final_groups = leaf_groups

    # Pass 3a: Sub-pattern relabeling (in-place on group[0].op_type)
    try:
        for group in final_groups:
            label = _semantic_label(group, path_to_class, platform)
            group[0].op_type = label
        if debug:
            sub_matched = sum(1 for g in final_groups
                              if g[0].op_type not in CONTAINER_SEMANTICS
                              and len(g) > 1)
            logger.info("[fusion-debug] Pass 3a (sub-patterns): %d groups, "
                        "%d non-container fused", len(final_groups), sub_matched)
    except Exception as e:
        logger.warning("Fusion Pass 3a (sub-patterns) failed: %s.",
                       e, exc_info=True)

    # Pass 3b: Add+Norm cross-boundary fusion
    try:
        final_groups = _pass3b_add_norm(
            final_groups, add_norm_fusion, path_to_class,
        )
    except Exception as e:
        logger.warning("Fusion Pass 3b (add_norm) failed: %s. Continuing.",
                       e, exc_info=True)

    # Pass 4: Expand unfused containers
    try:
        n_before = len(final_groups)
        final_groups = _pass4_expand_containers(
            final_groups, platform, path_to_class,
        )
        if debug:
            logger.info("[fusion-debug] Pass 4 (expand): %d → %d groups",
                        n_before, len(final_groups))
    except Exception as e:
        logger.warning("Fusion Pass 4 (expand) failed: %s. Keeping groups.",
                       e, exc_info=True)

    if debug:
        logger.info("[fusion-debug] Final: %d fused groups", len(final_groups))

    return final_groups
