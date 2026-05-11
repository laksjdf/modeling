"""FusionPass: apply module-scope fusion rules directly to OpGraph IR.

Thin adapter around :mod:`python.zrt.transform.fusion.core`.  All fusion
algorithm logic lives in core.py; this module handles OpNode ↔ FusionItem
conversion and in-place graph replacement.

Also exposes ``FusionPass.fuse_records()`` as a classmethod for callers
that work with raw Dict records (Excel export, ONNX graph building) instead
of the OpGraph IR.
"""
from __future__ import annotations

import copy
import re as _re
from collections import defaultdict
from typing import TYPE_CHECKING

from python.zrt.transform.base import GraphPass
from python.zrt.transform.fusion.core import FusionItem, run_fusion
from python.zrt.transform.fusion.rules import (
    CONTAINER_SEMANTICS,
    PATTERN_SKIP,
    get_platform_settings,
    get_semantic_label,
    get_subpatterns,
)

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.ir.node import OpNode
    from python.zrt.transform.context import TransformContext


# ── npu_sas annotation helpers ───────────────────────────────────────────────

_LAYER_RE = _re.compile(r"layers\.(\d+)\.")


def _sas_attn_type(cr: int) -> str:
    if cr == 0:
        return "SWA"
    if cr == 4:
        return "CSA"
    return "HCA"


def _annotate_npu_sas(g: "OpGraph") -> None:
    """Annotate npu_sas nodes with attn_type/compress_ratio from V4 config."""
    compress_ratios = g.metadata.get("compress_ratios")
    if not compress_ratios:
        return
    for node in g.topo_sort():
        if node.op_type != "npu_sas":
            continue
        m = _LAYER_RE.search(node.scope or "")
        if m is None:
            continue
        idx = int(m.group(1))
        cr = compress_ratios[idx] if idx < len(compress_ratios) else 0
        node.annotations["attn_type"] = _sas_attn_type(cr)
        node.annotations["compress_ratio"] = cr


# ── scope helpers ─────────────────────────────────────────────────────────────

def _parent(scope: str) -> str:
    return scope.rsplit(".", 1)[0] if "." in scope else ""


def _build_scope_maps(
    graph: "OpGraph",
) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Rebuild path_to_class and path_to_children from graph nodes."""
    path_to_class: dict[str, str] = {}
    path_to_children: dict[str, set[str]] = defaultdict(set)
    for node in graph.nodes.values():
        if node.scope and node.module_class:
            path_to_class[node.scope] = node.module_class
        if node.scope and "." in node.scope:
            path_to_children[_parent(node.scope)].add(node.scope)
    return path_to_class, dict(path_to_children)


# ── OpNode → FusionItem ──────────────────────────────────────────────────────

def _node_to_item(node: "OpNode") -> FusionItem:
    return FusionItem(
        scope=node.scope,
        module_class=node.module_class,
        op_type=node.op_type,
        layer=node.layer,
        phase=node.annotations.get("phase", ""),
        category=node.category,
        num_sub_ops=node.num_sub_ops or 1,
        annotations=dict(node.annotations),
        _meta={"node_id": node.id, "node": node},
    )


# ── FusionItem group → fused OpNode ──────────────────────────────────────────

def _build_fused_node(
    group: list[FusionItem],
    label: str,
    path_to_class: dict[str, str],
    graph: "OpGraph",
    fuse_idx: int,
) -> "OpNode":
    """Construct a fused OpNode from a group of FusionItems."""
    from python.zrt.ir.node import OpNode

    first_item = group[0]
    first_node = first_item._meta.get("node")
    mc = path_to_class.get(first_item.scope, first_item.module_class)

    # Collect source op_types (deduplicated, ordered)
    fused_from = list(dict.fromkeys(item.op_type for item in group))

    # Gather external inputs/outputs from all nodes in the group
    group_ids = {item._meta["node_id"] for item in group}
    ext_inputs = []
    ext_outputs = []
    seen_in: set = set()
    seen_out: set = set()

    for e in graph.edges:
        if e.src not in group_ids and e.dst in group_ids:
            key = (e.tensor_id, e.dst_idx)
            if key not in seen_in and e.tensor is not None:
                seen_in.add(key)
                ext_inputs.append(e.tensor)
        if e.src in group_ids and e.dst not in group_ids:
            key = (e.tensor_id, e.src_idx)
            if key not in seen_out and e.tensor is not None:
                seen_out.add(key)
                ext_outputs.append(e.tensor)

    # Propagate invariant annotations
    propagated: dict[str, object] = {}
    import logging as _logging
    _logger = _logging.getLogger(__name__)
    for key in ("stage_id", "phase"):
        vals = {item.annotations.get(key) for item in group
                if key in item.annotations}
        if len(vals) == 1:
            propagated[key] = vals.pop()
        elif len(vals) > 1:
            _logger.error(
                "_build_fused_node: group has mixed %r values %r",
                key, vals,
            )

    level = "parent" if len(group) > 3 else "leaf"

    node = OpNode(
        id=f"fused_{fuse_idx}_{first_item._meta['node_id']}",
        op_type=label,
        inputs=ext_inputs,
        outputs=ext_outputs,
        scope=first_item.scope,
        category=first_item.category,
        module_class=mc,
        layer=first_item.layer,
        component=first_node.component if first_node else "",
        fused_from=fused_from,
        num_sub_ops=sum(item.num_sub_ops for item in group),
        fusion_level=level,
    )
    node.annotations.update(propagated)
    return node


# ── External I/O for group expansion ─────────────────────────────────────────

def _external_io(
    graph: "OpGraph",
    group_ids: set[str],
) -> tuple[list, list]:
    """Return (ext_inputs, ext_outputs) for a node group."""
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


# ── Platform inference ───────────────────────────────────────────────────────

def _infer_platform(ctx: "TransformContext") -> str:
    """Best-effort platform from hw_spec vendor/device_type."""
    if ctx.hw_spec is None:
        return "generic"
    vendor = getattr(ctx.hw_spec, "vendor", "").lower()
    device_type = getattr(ctx.hw_spec, "device_type", "").lower()
    if "nvidia" in vendor or "cuda" in vendor:
        return "cuda"
    if "ascend" in vendor or "huawei" in vendor or "npu" in vendor or device_type == "npu":
        return "ascend_npu"
    return "generic"


# ── Incremental mode detection ───────────────────────────────────────────────

def _is_prefused(graph: "OpGraph") -> bool:
    """Check if graph is already fused by Stage-1 capture."""
    topo = graph.topo_sort()
    compute_nodes = [n for n in topo if n.category != "communication"]
    if not compute_nodes:
        return True
    return all(
        getattr(n, "fusion_level", "") for n in compute_nodes
    )


# ── Pass 4: expand unfused containers ────────────────────────────────────────

def _expand_containers(
    graph: "OpGraph",
    platform: str,
    path_to_class: dict[str, str],
) -> list[tuple[set[str], "OpNode"]]:
    """Scan for multi-op container groups without hardware subpattern match
    and clear their fusion metadata so downstream treats them as individual ops.

    This is the safety net for pre-fused graphs.
    """
    patterns = get_subpatterns(platform)
    fusions: list[tuple[set[str], "OpNode"]] = []

    for node in graph.topo_sort():
        if node.category == "communication":
            continue
        if getattr(node, "num_sub_ops", 1) <= 1:
            continue

        label = node.op_type
        if label not in CONTAINER_SEMANTICS:
            continue

        mc = node.module_class or ""
        matched = any(sp.matches_class(mc) for sp in patterns)
        if matched:
            continue  # container IS fusible — keep it

        # Unfused container: clear fusion metadata
        node.fused_from = []
        node.num_sub_ops = 0
        node.fusion_level = ""

    return fusions


# ── FusionPass ────────────────────────────────────────────────────────────────

class FusionPass(GraphPass):
    """Apply module-scope fusion rules to the OpGraph.

    Converts groups of consecutive same-scope aten ops into single fused nodes
    with semantic labels (e.g. ``flash_attn``, ``gated_mlp``, ``rms_norm``).
    Communication nodes are never fused and always act as group-breakers.

    **Incremental mode**: when all non-comm nodes already carry ``fusion_level``
    (from Stage-1 graph capture), Pass 1/2 (grouping + parent merge) are skipped.
    Only Pass 3 (semantic relabel) and Pass 4 (expand unfused containers) run,
    making this O(n) instead of O(n²) on pre-fused graphs.
    """

    name = "fusion"

    def __init__(
        self,
        *,
        platform: str = "generic",
        max_leaf_ops: int = 0,
        add_norm_fusion: bool = False,
    ):
        self._platform = platform
        self._max_leaf_ops = max_leaf_ops
        self._add_norm_fusion = add_norm_fusion

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        platform = self._platform if self._platform != "generic" else _infer_platform(ctx)
        cfg = get_platform_settings(platform)
        max_parent_ops = cfg["max_parent_ops"]
        max_children = cfg["max_children"]

        g = graph.clone()
        path_to_class, path_to_children = _build_scope_maps(g)

        # ── No-op detection: skip full regrouping if graph already fused ──────
        if _is_prefused(g):
            # Graph already fused by Stage-1 capture → only Pass 4 needed
            self._expand_containers(g, platform, path_to_class)
            _annotate_npu_sas(g)
            return g

        # ── OpNode → FusionItem ───────────────────────────────────────────────
        items = [_node_to_item(n) for n in g.topo_sort()]

        # ── Core fusion algorithm ─────────────────────────────────────────────
        groups = run_fusion(
            items,
            path_to_class=path_to_class,
            path_to_children=path_to_children,
            platform=platform,
            max_leaf_ops=self._max_leaf_ops,
            add_norm_fusion=self._add_norm_fusion,
        )

        # ── FusionItem groups → OpNode replacements ───────────────────────────
        fuse_idx = 0
        for group in groups:
            if len(group) == 1:
                continue  # single node: no replacement needed

            group_ids = {item._meta["node_id"] for item in group}
            label = group[0].op_type  # already set by run_fusion() Pass 3a
            new_node = _build_fused_node(group, label, path_to_class, g, fuse_idx)
            fuse_idx += 1
            g.replace_subgraph(group_ids, new_node)

        _annotate_npu_sas(g)
        return g

    @staticmethod
    def _expand_containers(
        graph: "OpGraph",
        platform: str,
        path_to_class: dict[str, str],
    ) -> None:
        """In-place pass 4 for pre-fused graphs."""
        _expand_containers(graph, platform, path_to_class)

    # ── Dict record convenience ──────────────────────────────────────────────

    @classmethod
    def fuse_records(
        cls,
        records: list[dict],
        tracker,
        *,
        platform: str = "generic",
        max_leaf_ops: int = 0,
        keep_children: bool = True,
        debug: bool = False,
    ) -> list[dict]:
        """One-shot fusion of Dict records → fused Dict records.

        This replaces ``FusionEngine.fuse()`` and ``FusionEngine.fuse_keep_children()``.

        Parameters
        ----------
        records :
            Raw aten op records from dispatch tracing.
        tracker :
            ModuleTracker with path_to_class / path_to_children.
        platform :
            One of "cuda", "ascend_npu", "cpu", "generic".
        max_leaf_ops :
            Leaf group cap. Set ~15 for backward graphs.
        keep_children :
            Embed _children in output (True for ONNX/graph building,
            False for Excel display).
        debug :
            Log detailed fusion decisions.

        Returns
        -------
        List[Dict] in the exact format expected by downstream consumers
        (ExcelWriter, ONNX exporter, fused_records_to_opgraph, etc.).
        """
        from python.zrt.transform.fusion._dict_bridge import fuse_records as _fuse

        cfg = get_platform_settings(platform)
        return _fuse(
            records, tracker,
            platform=platform,
            max_leaf_ops=max_leaf_ops,
            add_norm_fusion=cfg["add_norm_fusion"],
            keep_children=keep_children,
            debug=debug,
        )
