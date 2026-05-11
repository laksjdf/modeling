"""Dict ↔ FusionItem conversion bridge.

Extracted from the legacy ``graph/fusion.py`` FusionEngine.  These functions
handle the Dict-record format required by Excel export, ONNX graph building,
and JSON output.  The fusion algorithm itself lives in ``core.py``.

Public helpers for FusionPass.fuse_records():
  - ``records_to_fusable_items(records, tracker)`` → List[FusionItem]
  - ``fused_items_to_records(groups, tracker, keep_children)`` → List[Dict]
  - ``extract_fusion_specs(fused_records)`` → List[FusionSpec]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from python.zrt.transform.fusion.core import FusionItem, run_fusion
from python.zrt.transform.fusion.rules import (
    CONTAINER_SEMANTICS,
    PATTERN_SKIP,
    get_platform_settings,
    get_semantic_label,
    get_subpatterns,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FusionSpec (unchanged from graph/fusion.py)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FusionSpec:
    """A fusion pattern auto-discovered from dispatch tracing."""
    module_class: str
    aten_op_sequence: List[str]
    num_sub_ops: int
    fusion_level: str
    example_module_path: str
    occurrences: int = 1
    fused_input_shapes: str = ""
    fused_input_dtypes: str = ""
    fused_input_sources: str = ""
    fused_output_shapes: str = ""
    fused_output_dtypes: str = ""
    fused_output_sources: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers (verbatim from graph/fusion.py)
# ─────────────────────────────────────────────────────────────────────────────

def _split_shape_list(s: str) -> List[str]:
    """Split '[1, 128], [64]' into ['[1, 128]', '[64]']."""
    if not s:
        return []
    result, depth, current = [], 0, []
    for ch in s:
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        result.append("".join(current).strip())
    return result


def _strip_layer_prefix(module_path: str) -> str:
    parts = module_path.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                int(parts[i + 1])
                return ".".join(parts[i + 2:]) or module_path
            except ValueError:
                pass
    return module_path


def _parent_path(module_path: str) -> str:
    if "." in module_path:
        return module_path.rsplit(".", 1)[0]
    return ""


def _compute_fused_io(ops: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute fused I/O from a list of raw op records.

    Determines external inputs (consumed but not produced within the group)
    and external outputs (produced but not consumed within the group).
    """
    produced_ids: set = set()
    for op in ops:
        for tid in op.get("_output_ids", []):
            produced_ids.add(tid)

    consumed_ids: set = set()
    for op in ops:
        for tid in op.get("_input_ids", []):
            consumed_ids.add(tid)

    external_input_ids = consumed_ids - produced_ids
    external_output_ids = produced_ids - consumed_ids

    seen_in: set = set()
    fused_input_ids = []
    for op in ops:
        for tid in op.get("_input_ids", []):
            if tid in external_input_ids and tid not in seen_in:
                seen_in.add(tid)
                fused_input_ids.append(tid)

    seen_out: set = set()
    fused_output_ids = []
    for op in reversed(ops):
        for tid in op.get("_output_ids", []):
            if tid in external_output_ids and tid not in seen_out:
                seen_out.add(tid)
                fused_output_ids.insert(0, tid)

    input_shapes, input_dtypes, input_sources = [], [], []
    for tid in fused_input_ids:
        for op in ops:
            if tid in op.get("_input_ids", []):
                idx = op["_input_ids"].index(tid)
                shapes = _split_shape_list(op["input_shapes"])
                dtypes = op["input_dtypes"].split(", ")
                input_shapes.append(shapes[idx] if idx < len(shapes) else "?")
                input_dtypes.append(dtypes[idx] if idx < len(dtypes) else "?")
                input_sources.append(f"{op['aten_op']} (in[{idx}])")
                break

    output_shapes, output_dtypes, output_sources = [], [], []
    for tid in fused_output_ids:
        for op in reversed(ops):
            if tid in op.get("_output_ids", []):
                idx = op["_output_ids"].index(tid)
                shapes = _split_shape_list(op["output_shapes"])
                dtypes = op["output_dtypes"].split(", ")
                output_shapes.append(shapes[idx] if idx < len(shapes) else "?")
                output_dtypes.append(dtypes[idx] if idx < len(dtypes) else "?")
                output_sources.append(f"{op['aten_op']} (out[{idx}])")
                break

    return {
        "fused_input_shapes": ", ".join(input_shapes),
        "fused_input_dtypes": ", ".join(input_dtypes),
        "fused_input_sources": " | ".join(input_sources),
        "fused_output_shapes": ", ".join(output_shapes),
        "fused_output_dtypes": ", ".join(output_dtypes),
        "fused_output_sources": " | ".join(output_sources),
        "_input_ids": fused_input_ids,
        "_output_ids": fused_output_ids,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dict construction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_fused_entry(
    ops: List[Dict[str, Any]],
    tracker: Any,
    fusion_level: str = "leaf",
) -> Dict[str, Any]:
    """Build one fused-op record from a list of raw op records.

    Format must match the legacy FusionEngine output exactly.
    """
    first, last = ops[0], ops[-1]
    path = first["module_path"]
    module_class = tracker.path_to_class.get(path, first.get("module_class", ""))

    aten_ops = list(dict.fromkeys(r["aten_op"] for r in ops))
    short_path = _strip_layer_prefix(path) if path else ""

    # Path-based override: lm_head is always "lm_head"
    _path_lower = path.lower()
    if _path_lower == "lm_head" or _path_lower.endswith(".lm_head"):
        semantic = "lm_head"
    else:
        semantic = get_semantic_label(module_class) if module_class else None

    effective_ops = [r["aten_op"] for r in ops if r["aten_op"] not in PATTERN_SKIP]
    ref_op = effective_ops[0] if effective_ops else first["aten_op"]
    fn_parts = ref_op.split(".")
    aten_label = fn_parts[1] if len(fn_parts) >= 2 else ref_op

    if semantic and semantic not in CONTAINER_SEMANTICS:
        label = semantic
    else:
        label = aten_label

    io = _compute_fused_io(ops)

    return {
        "fused_op": label,
        "_semantic": semantic,
        "module_path": path,
        "module_class": module_class,
        "fusion_level": fusion_level,
        "aten_ops": " \u2192 ".join(aten_ops),
        "num_sub_ops": len(ops),
        "layer": first["layer"],
        "input_shapes": first["input_shapes"],
        "input_dtypes": first["input_dtypes"],
        "output_shapes": last["output_shapes"],
        "output_dtypes": last["output_dtypes"],
        "component": first.get("component", ""),
        **io,
        "_children": ops,
    }


def _make_individual_entry(op: Dict[str, Any]) -> Dict[str, Any]:
    """Build a singleton fused-op record from one raw dispatch record."""
    fn_parts = op["aten_op"].split(".")
    label = fn_parts[1] if len(fn_parts) >= 2 else op["aten_op"]
    io = _compute_fused_io([op])
    return {
        "fused_op": label,
        "_semantic": None,
        "module_path": op["module_path"],
        "module_class": op.get("module_class", ""),
        "fusion_level": "unfused",
        "aten_ops": op["aten_op"],
        "num_sub_ops": 1,
        "layer": op["layer"],
        "input_shapes": op["input_shapes"],
        "input_dtypes": op["input_dtypes"],
        "output_shapes": op["output_shapes"],
        "output_dtypes": op["output_dtypes"],
        "component": op.get("component", ""),
        **io,
        "_children": [op],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API: records → fused records
# ─────────────────────────────────────────────────────────────────────────────

def records_to_fusable_items(
    records: List[Dict[str, Any]],
    tracker: Any,
    *,
    max_leaf_ops: int = 0,
) -> Tuple[List[FusionItem], Dict[str, str], Dict[str, set]]:
    """Convert raw Dict records → FusionItem list + scope maps.

    Parameters
    ----------
    records :
        Raw aten op records from dispatch tracing.
    tracker :
        ModuleTracker with path_to_class / path_to_children.
    max_leaf_ops :
        Leaf group size cap for backward graphs.

    Returns
    -------
    (items, path_to_class, path_to_children)
    """
    items: List[FusionItem] = []
    for rec in records:
        # Determine phase from record
        phase = rec.get("phase", "")

        # Category: communication ops are identified by prefix
        aten = rec.get("aten_op", "")
        if aten.startswith("dist.") or aten.startswith("nccl."):
            category = "communication"
        else:
            category = "compute"

        item = FusionItem(
            scope=rec["module_path"],
            module_class=rec.get("module_class", ""),
            op_type=aten,
            layer=str(rec.get("layer", "")),
            phase=phase,
            category=category,
            num_sub_ops=1,
            children=[rec],
            input_ids=rec.get("_input_ids", []),
            output_ids=rec.get("_output_ids", []),
            _meta={"record": rec},
        )
        items.append(item)

    path_to_class = tracker.path_to_class
    path_to_children = tracker.path_to_children

    return items, path_to_class, path_to_children


def fused_items_to_records(
    groups: List[List[FusionItem]],
    tracker: Any,
    *,
    keep_children: bool = True,
) -> List[Dict[str, Any]]:
    """Convert fused FusionItem groups → Dict records (Excel/export format).

    Parameters
    ----------
    groups :
        Output of run_fusion().
    tracker :
        ModuleTracker for class name lookups.
    keep_children :
        If True, embed ``_children`` list in each record (for ONNX graph
        building).  If False, strip ``_children`` (for Excel display).

    Returns
    -------
    List[Dict] in the exact format expected by ExcelWriter, ONNX exporter, etc.
    """
    result: List[Dict[str, Any]] = []

    for group in groups:
        if len(group) == 1:
            item = group[0]
            entry: Dict[str, Any]
            # Check if this was expanded from a container (no children)
            if item.children and len(item.children) == 1 and isinstance(item.children[0], dict):
                # Single child was expanded → make individual entry
                entry = _make_individual_entry(item.children[0])
            elif item._meta.get("record"):
                # Original single record → make individual entry
                entry = _make_individual_entry(item._meta["record"])
            else:
                # Fallback: build minimal entry from FusionItem
                fn_parts = item.op_type.split(".")
                label = fn_parts[1] if len(fn_parts) >= 2 else item.op_type
                entry = {
                    "fused_op": label,
                    "_semantic": None,
                    "module_path": item.scope,
                    "module_class": item.module_class,
                    "fusion_level": "unfused",
                    "aten_ops": item.op_type,
                    "num_sub_ops": item.num_sub_ops,
                    "layer": item.layer,
                    "input_shapes": "",
                    "input_dtypes": "",
                    "output_shapes": "",
                    "output_dtypes": "",
                    "component": "",
                    "fused_input_shapes": "",
                    "fused_input_dtypes": "",
                    "fused_input_sources": "",
                    "fused_output_shapes": "",
                    "fused_output_dtypes": "",
                    "fused_output_sources": "",
                    "_children": item.children if keep_children else [],
                }
            # node_id is added here (was missing in _make_*_entry)
            entry["node_id"] = len(result)
            if not keep_children:
                entry.pop("_children", None)
                entry.pop("_semantic", None)
            result.append(entry)
            continue

        # Multi-item group → fused entry
        # Collect raw child records
        raw_children: List[Dict[str, Any]] = []
        for item in group:
            if item.children:
                for c in item.children:
                    if isinstance(c, dict):
                        raw_children.append(c)
            elif item._meta.get("record"):
                raw_children.append(item._meta["record"])

        if raw_children:
            entry = _make_fused_entry(raw_children, tracker)
            entry["fusion_level"] = "parent" if len(group) > 3 else "leaf"
            # Update label from the group's computed op_type
            entry["fused_op"] = group[0].op_type
            if not keep_children:
                entry.pop("_children", None)
                entry.pop("_semantic", None)
            entry["node_id"] = len(result)
            result.append(entry)

    return result


def extract_fusion_specs(fused: List[Dict[str, Any]]) -> List[FusionSpec]:
    """Extract FusionSpec list from fused records.

    Used by ExcelWriter to generate the fusion rules JSON.
    """
    specs_by_key: Dict[Tuple[str, str], FusionSpec] = {}
    for g in fused:
        if g.get("num_sub_ops", 1) <= 1:
            continue
        key = (g.get("module_class", ""), g.get("fusion_level", ""))
        if key in specs_by_key:
            specs_by_key[key].occurrences += 1
        else:
            specs_by_key[key] = FusionSpec(
                module_class=g.get("module_class", ""),
                aten_op_sequence=g.get("aten_ops", "").split(" \u2192 "),
                num_sub_ops=g.get("num_sub_ops", 1),
                fusion_level=g.get("fusion_level", ""),
                example_module_path=g.get("module_path", ""),
                occurrences=1,
                fused_input_shapes=g.get("fused_input_shapes", ""),
                fused_input_dtypes=g.get("fused_input_dtypes", ""),
                fused_input_sources=g.get("fused_input_sources", ""),
                fused_output_shapes=g.get("fused_output_shapes", ""),
                fused_output_dtypes=g.get("fused_output_dtypes", ""),
                fused_output_sources=g.get("fused_output_sources", ""),
            )
    return sorted(specs_by_key.values(), key=lambda s: -s.occurrences)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level convenience: records → fused records (one-shot)
# ─────────────────────────────────────────────────────────────────────────────

def fuse_records(
    records: List[Dict[str, Any]],
    tracker: Any,
    *,
    platform: str = "generic",
    max_leaf_ops: int = 0,
    add_norm_fusion: bool = False,
    keep_children: bool = True,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """One-shot fusion: Dict records → fused Dict records.

    This replaces FusionEngine.fuse() / fuse_keep_children().

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
    add_norm_fusion :
        Enable cross-boundary Add+Norm fusion.
    keep_children :
        Embed _children in output (True for ONNX, False for Excel display).
    debug :
        Log detailed fusion decisions.

    Returns
    -------
    List[Dict] in the exact format expected by downstream consumers.
    """
    items, path_to_class, path_to_children = records_to_fusable_items(
        records, tracker, max_leaf_ops=max_leaf_ops,
    )

    groups = run_fusion(
        items,
        path_to_class=path_to_class,
        path_to_children=path_to_children,
        platform=platform,
        max_leaf_ops=max_leaf_ops,
        add_norm_fusion=add_norm_fusion,
        debug=debug,
    )

    return fused_items_to_records(groups, tracker, keep_children=keep_children)
