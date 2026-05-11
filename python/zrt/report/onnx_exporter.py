"""Export computation graphs to JSON and ONNX format.

ONNX export leverages Netron's scope-nesting: each ONNX node's ``name``
uses ``/`` separators encoding the module hierarchy so that Netron renders
collapsible groups matching the nn.Module tree.

The ONNX ``op_type`` is set to a clean short name (e.g. ``mul``, ``mm``,
``softmax``), and rich metadata (module_class, component, layer, shapes)
is stored as ONNX node attributes visible in Netron's sidebar.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

try:
    import onnx
    from onnx import AttributeProto, TensorProto, helper
    _HAS_ONNX = True
except ImportError:
    onnx = None  # type: ignore[assignment]
    _HAS_ONNX = False

logger = logging.getLogger(__name__)

# ── Torch dtype → ONNX elem_type ─────────────────────────────────────────────

_DTYPE_MAP = {
    "torch.float16":  TensorProto.FLOAT16,
    "torch.float32":  TensorProto.FLOAT,
    "torch.float64":  TensorProto.DOUBLE,
    "torch.bfloat16": TensorProto.BFLOAT16,
    "torch.int8":     TensorProto.INT8,
    "torch.int16":    TensorProto.INT16,
    "torch.int32":    TensorProto.INT32,
    "torch.int64":    TensorProto.INT64,
    "torch.uint8":    TensorProto.UINT8,
    "torch.bool":     TensorProto.BOOL,
}


def _to_onnx_elem_type(dtype_str: str) -> int:
    return _DTYPE_MAP.get(dtype_str.strip(), TensorProto.FLOAT)


def _parse_shape(shape_str: str) -> List[int]:
    """Parse '[1, 128, 7168]' into [1, 128, 7168]."""
    s = shape_str.strip().strip("[]")
    if not s:
        return []
    try:
        return [int(x.strip()) for x in s.split(",")]
    except ValueError:
        return []


def _split_shape_list(s: str) -> List[str]:
    """Split '[1, 128], [64]' into ['[1, 128]', '[64]']."""
    if not s:
        return []
    result: List[str] = []
    depth = 0
    current: List[str] = []
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


# ── aten op → clean short name ───────────────────────────────────────────────

def _aten_op_short_name(aten_op: str) -> str:
    """Extract the core operation name from an aten op string.

    Examples::

        aten.mul.Tensor       → mul
        aten.mm.default       → mm
        aten._to_copy.default → to_copy
        aten.pow.Tensor_Scalar→ pow
        aten.embedding.default→ embedding
        softmax               → softmax   (passthrough)
    """
    parts = aten_op.split(".")
    if len(parts) >= 2 and parts[0] == "aten":
        name = parts[1]
    elif len(parts) >= 2:
        name = parts[1]
    else:
        name = aten_op
    # Strip leading underscore from internal ops like _to_copy
    return name.lstrip("_") if name.startswith("_") else name


# ── Module path → ONNX scope ────────────────────────────────────────────────

def _module_path_to_scope(module_path: str) -> str:
    """Convert ``model.layers.0.self_attn.q_a_proj`` to
    ``model/layers.0/self_attn/q_a_proj`` for Netron grouping.

    Keeps container+index pairs together (``layers.0``, ``experts.3``)
    as a single scope level for readability.
    """
    if not module_path:
        return ""
    parts = module_path.split(".")
    scope_parts: List[str] = []
    _CONTAINERS = {"layers", "blocks", "h", "layer", "experts"}
    i = 0
    while i < len(parts):
        p = parts[i]
        if p in _CONTAINERS and i + 1 < len(parts):
            try:
                int(parts[i + 1])
                scope_parts.append(f"{p}.{parts[i + 1]}")
                i += 2
                continue
            except ValueError:
                pass
        scope_parts.append(p)
        i += 1
    return "/".join(scope_parts)


# ── JSON export ──────────────────────────────────────────────────────────────

def export_graph_json(G: nx.DiGraph, output_path: Path) -> Path:
    """Export graph as node-link JSON."""
    data = nx.node_link_data(G)

    for node in data.get("nodes", []):
        for k, v in list(node.items()):
            if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                node[k] = str(v)
    for key in ("links", "edges"):
        for edge in data.get(key, []):
            for k, v in list(edge.items()):
                if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    edge[k] = str(v)

    if "links" in data and "edges" not in data:
        data["edges"] = data.pop("links")
    data["metadata"] = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    logger.info("Exported graph JSON to %s", output_path)
    return output_path


# ── ONNX export ──────────────────────────────────────────────────────────────

def _build_onnx_from_records(
    records: List[Dict[str, Any]],
    model_name: str,
    is_fused: bool = False,
) -> onnx.ModelProto:
    """Build an ONNX ModelProto from traced operator records.

    Design goals (Netron-friendly):
    - Each ONNX node  = one operator (raw aten op or fused op)
    - Edges           = implicit data flow via shared tensor names
    - Shape / dtype   = node attributes (readable in Netron sidebar)
    - No value_info   = no edge-label clutter
    - No weight boxes = weight / initialiser tensors are NOT declared as
                        graph.input, so they don't appear as big input nodes

    ``node.name`` uses ``/``-separated scopes so Netron collapses sub-graphs:
    - Raw graph:   ``model/layers.0/self_attn/mm_5``
    - Fused graph: ``layer0/rms_norm_0`` (flat layer prefix)
    """
    onnx_nodes: List[onnx.NodeProto] = []
    tid_to_name: Dict[int, str] = {}

    # ── Pass 1: assign stable tensor names ───────────────────────────────
    # Producers first so names come from output slots; consumers fill gaps.
    for rec in records:
        for tid in rec.get("_output_ids", []):
            if tid not in tid_to_name:
                tid_to_name[tid] = f"t{tid}"
    for rec in records:
        for tid in rec.get("_input_ids", []):
            if tid not in tid_to_name:
                tid_to_name[tid] = f"t{tid}"

    # ── Pass 2: build ONNX nodes ──────────────────────────────────────────
    for rec in records:
        idx          = rec["node_id"]
        module_path  = rec.get("module_path",  "")
        module_class = rec.get("module_class", "")
        component    = rec.get("component",    "")
        layer        = rec.get("layer",        "")

        if is_fused:
            fused_op     = rec.get("fused_op", "op")
            op_type      = _clean_op_type(fused_op)   # safe identifier
            aten_ops_str = rec.get("aten_ops", "")
            in_shapes    = rec.get("fused_input_shapes",  rec.get("input_shapes",  ""))
            in_dtypes    = rec.get("fused_input_dtypes",  rec.get("input_dtypes",  ""))
            out_shapes   = rec.get("fused_output_shapes", rec.get("output_shapes", ""))
            out_dtypes   = rec.get("fused_output_dtypes", rec.get("output_dtypes", ""))
            # Node name: flat layer scope for a clean fused graph
            layer_scope  = f"layer{layer}" if layer else "global"
            node_name    = f"{layer_scope}/{op_type}_{idx}"
        else:
            raw_op       = rec.get("aten_op", "op")
            op_type      = _aten_op_short_name(raw_op)
            aten_ops_str = ""
            in_shapes    = rec.get("input_shapes",  "")
            in_dtypes    = rec.get("input_dtypes",  "")
            out_shapes   = rec.get("output_shapes", "")
            out_dtypes   = rec.get("output_dtypes", "")
            # Node name: module-hierarchy scope for raw graph
            scope        = _module_path_to_scope(module_path)
            node_name    = f"{scope}/{op_type}_{idx}" if scope else f"{op_type}_{idx}"

        in_ids  = rec.get("_input_ids",  [])
        out_ids = rec.get("_output_ids", [])
        input_names  = [tid_to_name[tid] for tid in in_ids  if tid in tid_to_name]
        output_names = [tid_to_name[tid] for tid in out_ids if tid in tid_to_name]

        node = helper.make_node(
            op_type=op_type,
            inputs=input_names,
            outputs=output_names,
            name=node_name,
        )

        # ── Attributes (visible in Netron properties panel) ───────────────
        def _attr(name: str, val: Any) -> None:
            if val is None:
                return
            if isinstance(val, int):
                node.attribute.append(helper.make_attribute(name, val))
            else:
                s = str(val)
                if s:
                    node.attribute.append(helper.make_attribute(name, s))

        _attr("layer",        layer)
        _attr("module_class", module_class)
        _attr("module_path",  module_path)
        _attr("component",    component)
        _attr("input_shapes",  in_shapes)
        _attr("input_dtypes",  in_dtypes)
        _attr("output_shapes", out_shapes)
        _attr("output_dtypes", out_dtypes)

        if is_fused:
            _attr("fused_op",     fused_op)
            _attr("aten_ops",     aten_ops_str)
            _attr("num_sub_ops",  rec.get("num_sub_ops", 0))
            _attr("fusion_level", rec.get("fusion_level", ""))
        else:
            _attr("aten_op", raw_op)

        onnx_nodes.append(node)

    # ── Graph wrapper ─────────────────────────────────────────────────────
    # Only a minimal dummy input/output is declared.  All real tensor
    # connections are implicit via shared names — no value_info needed,
    # no weight tensors appear as large input boxes in Netron.
    _dummy = helper.make_tensor_type_proto(TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        onnx_nodes,
        name=model_name,
        inputs=[helper.make_value_info("graph_input",  _dummy)],
        outputs=[helper.make_value_info("graph_output", _dummy)],
        value_info=[],
    )

    model = helper.make_model(graph)
    model.ir_version = 8
    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.domain = "ai.aten"
    opset.version = 1
    model.doc_string = f"Computation graph for {model_name}"
    return model


def _fused_op_display_name(fused_op: str, module_class: str) -> str:
    """Create a clean display name for a fused op.

    Examples::

        "self_attn.q_a_proj (Linear)"           → "q_a_proj"
        "input_layernorm (DeepseekV3RMSNorm)"   → "RMSNorm"
        "self_attn (DeepseekV3Attention)"        → "Attention"
        "mlp (DeepseekV3MLP)"                    → "MLP"
        "model.layers.0"                         → "residual"
        "self_attn"                              → "self_attn"
    """
    # If module_class is present, extract a readable short form
    if module_class:
        # DeepseekV3RMSNorm → RMSNorm, DeepseekV3Attention → Attention
        short_class = module_class
        for prefix in ("DeepseekV3", "DeepseekV2", "Deepseek",
                        "Qwen2", "Qwen", "Llama", "Mistral", "Mixtral"):
            if short_class.startswith(prefix):
                short_class = short_class[len(prefix):]
                break
        if short_class:
            return short_class

    # Extract the last meaningful segment
    # "self_attn.q_a_proj (Linear)" → take part before " ("
    base = fused_op.split(" (")[0] if " (" in fused_op else fused_op
    # "self_attn.q_a_proj" → "q_a_proj"
    if "." in base:
        return base.rsplit(".", 1)[-1]
    return base


def _clean_op_type(fused_op: str) -> str:
    """Return a valid ONNX op_type identifier from a fused-op label.

    Handles labels produced by the fusion engine:
    - Semantic labels (already clean): ``rms_norm``, ``flash_attn`` → unchanged
    - Legacy format: ``self_attn.q_a_proj (Linear)``    → ``q_a_proj``
    - Legacy format: ``input_layernorm (LlamaRMSNorm)`` → ``rms_norm``
      (class-based short name via the same stripping as _fused_op_display_name)
    """
    # If it contains " (" it's the old "path (ClassName)" format
    if " (" in fused_op:
        # Try to extract class-based short name
        class_part = fused_op.split(" (", 1)[1].rstrip(")")
        short = _fused_op_display_name(fused_op, class_part)
        return _sanitise(short)
    return _sanitise(fused_op)


def _sanitise(name: str) -> str:
    """Replace characters that are invalid in ONNX op_type with underscores."""
    import re as _re
    return _re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_") or "op"


def export_onnx_from_records(
    records: List[Dict[str, Any]],
    output_path: Path,
    model_name: str,
    is_fused: bool = False,
) -> Path:
    """Export traced records directly to ONNX."""
    onnx_model = _build_onnx_from_records(records, model_name, is_fused)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(onnx_model, str(output_path))
    logger.info("Exported ONNX to %s (%d nodes)",
                output_path, len(onnx_model.graph.node))
    return output_path


# ── Export all ────────────────────────────────────────────────────────────────

def export_all(
    raw_graph: nx.DiGraph,
    fused_graph: Optional[nx.DiGraph],
    raw_records: List[Dict[str, Any]],
    fused_records: List[Dict[str, Any]],
    output_dir: Path,
    model_name: str,
    phase: str = "forward",
) -> Dict[str, Path]:
    """Export all graph artifacts to the output directory."""
    if not _HAS_ONNX:
        raise ImportError("onnx is required for graph export. Install it with: pip install onnx")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Path] = {}

    # Raw op graph
    raw_json_path = output_dir / f"{model_name}_{phase}_raw_graph.json"
    results["raw_graph_json"] = export_graph_json(raw_graph, raw_json_path)

    raw_onnx_path = output_dir / f"{model_name}_{phase}_raw_graph.onnx"
    results["raw_graph_onnx"] = export_onnx_from_records(
        raw_records, raw_onnx_path, f"{model_name}_raw_{phase}")

    # Fused op graph
    if fused_graph is not None and fused_records:
        fused_json_path = output_dir / f"{model_name}_{phase}_fused_graph.json"
        results["fused_graph_json"] = export_graph_json(fused_graph, fused_json_path)

        fused_onnx_path = output_dir / f"{model_name}_{phase}_fused_graph.onnx"
        results["fused_graph_onnx"] = export_onnx_from_records(
            fused_records, fused_onnx_path, f"{model_name}_fused_{phase}",
            is_fused=True)

    return results
