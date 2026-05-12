"""Build a NetworkX DAG from traced operator records.

Each node represents an aten operator captured during the forward pass.
Edges represent tensor data flow: an edge from node A to node B means
that at least one output tensor of A is consumed as an input tensor of B.

Node attributes include operator type, module path, input/output shapes
and dtypes, layer index, and component classification.

Edge attributes include the tensor ID(s) that flow along that edge,
along with shape and dtype information.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


def build_op_graph(records: List[Dict[str, Any]]) -> nx.DiGraph:
    """Build a directed acyclic graph from traced operator records.

    Parameters
    ----------
    records:
        List of op-record dicts produced by ``RecordingDispatch``.
        Each record must contain ``_input_ids`` and ``_output_ids``
        (lists of integer tensor IDs).

    Returns
    -------
    nx.DiGraph
        A DAG where each node ID is the op index (``rec["node_id"]``) and
        edges encode tensor data-flow dependencies.
    """
    G = nx.DiGraph()

    # Phase 1: add all operator nodes
    for rec in records:
        node_id = f"op_{rec['node_id']}"
        G.add_node(node_id, **_node_attrs(rec))

    # Phase 2: build tensor-id -> producer-node mapping
    # A tensor can be produced by exactly one op (the first one that outputs it).
    tensor_producer: Dict[int, Tuple[str, int]] = {}  # tid -> (node_id, output_slot)
    for rec in records:
        node_id = f"op_{rec['node_id']}"
        for slot, tid in enumerate(rec.get("_output_ids", [])):
            if tid not in tensor_producer:
                tensor_producer[tid] = (node_id, slot)

    # Phase 3: add edges based on tensor consumption
    for rec in records:
        consumer_id = f"op_{rec['node_id']}"
        for in_slot, tid in enumerate(rec.get("_input_ids", [])):
            if tid in tensor_producer:
                producer_id, out_slot = tensor_producer[tid]
                if producer_id == consumer_id:
                    continue  # skip self-loops
                # There may be multiple tensors flowing between the same pair;
                # accumulate tensor IDs on the edge.
                if G.has_edge(producer_id, consumer_id):
                    edge = G.edges[producer_id, consumer_id]
                    edge["tensor_ids"].append(tid)
                    edge["label"] = f"{len(edge['tensor_ids'])} tensors"
                else:
                    # Extract shape/dtype info for this tensor from the producer
                    shape_info = _get_tensor_info(
                        records[int(producer_id.split("_")[1])], tid, "output")
                    G.add_edge(producer_id, consumer_id,
                               tensor_ids=[tid],
                               label=shape_info.get("shape", ""),
                               dtype=shape_info.get("dtype", ""),
                               shape=shape_info.get("shape", ""))

    logger.info("Built graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def build_fused_op_graph(fused_records: List[Dict[str, Any]],
                         raw_records: List[Dict[str, Any]]) -> nx.DiGraph:
    """Build a DAG from fused operator records.

    Similar to ``build_op_graph`` but operates on the fused op list.
    Data-flow edges are determined by examining which fused ops consume
    tensors produced by other fused ops (using the external I/O computed
    during fusion).
    """
    G = nx.DiGraph()

    # Collect all raw-level tensor IDs produced/consumed per fused op
    fused_io: List[Dict[str, Any]] = []
    for frec in fused_records:
        # Reconstruct external inputs/outputs from constituent raw ops
        sub_ops = _find_sub_ops(frec, raw_records)
        produced = set()
        consumed = set()
        for op in sub_ops:
            for tid in op.get("_output_ids", []):
                produced.add(tid)
            for tid in op.get("_input_ids", []):
                consumed.add(tid)
        external_in = consumed - produced
        external_out = produced - consumed
        fused_io.append({
            "external_in": external_in,
            "external_out": external_out,
        })

    # Add fused op nodes
    for frec in fused_records:
        node_id = f"fused_{frec['node_id']}"
        G.add_node(node_id, **_fused_node_attrs(frec))

    # Build tensor -> fused-producer map
    tensor_fused_producer: Dict[int, str] = {}
    for i, frec in enumerate(fused_records):
        node_id = f"fused_{frec['node_id']}"
        for tid in fused_io[i]["external_out"]:
            if tid not in tensor_fused_producer:
                tensor_fused_producer[tid] = node_id

    # Add edges
    for i, frec in enumerate(fused_records):
        consumer_id = f"fused_{frec['node_id']}"
        for tid in fused_io[i]["external_in"]:
            if tid in tensor_fused_producer:
                producer_id = tensor_fused_producer[tid]
                if producer_id == consumer_id:
                    continue
                if G.has_edge(producer_id, consumer_id):
                    edge = G.edges[producer_id, consumer_id]
                    edge["tensor_ids"].append(tid)
                    edge["label"] = f"{len(edge['tensor_ids'])} tensors"
                else:
                    G.add_edge(producer_id, consumer_id,
                               tensor_ids=[tid],
                               label="",
                               dtype="",
                               shape="")

    logger.info("Built fused graph: %d nodes, %d edges",
                G.number_of_nodes(), G.number_of_edges())
    return G


def build_fused_nx_from_opgraph(fused_opgraph) -> nx.DiGraph:
    """Build a NetworkX DiGraph from a fused OpGraph IR.

    Replaces ``build_fused_op_graph`` for the new pipeline that works
    directly with OpGraph instead of fused record dicts.
    """
    G = nx.DiGraph()

    for node in fused_opgraph.topo_sort():
        G.add_node(node.id, **{
            "op_type": node.op_type,
            "aten_ops": " → ".join(node.fused_from) if node.fused_from else node.op_type,
            "module_path": node.scope,
            "module_class": node.module_class,
            "layer": node.layer,
            "fusion_level": node.fusion_level,
            "num_sub_ops": node.num_sub_ops,
            "input_shapes": ", ".join(str(list(t.shape)) for t in node.inputs),
            "input_dtypes": ", ".join(str(t.dtype) for t in node.inputs),
            "output_shapes": ", ".join(str(list(t.shape)) for t in node.outputs),
            "output_dtypes": ", ".join(str(t.dtype) for t in node.outputs),
            "label": node.op_type,
        })

    for edge in fused_opgraph.edges:
        if edge.src in G.nodes and edge.dst in G.nodes:
            if G.has_edge(edge.src, edge.dst):
                existing = G.edges[edge.src, edge.dst]
                existing["label"] = f"{int(existing.get('label', '0').split()[0]) + 1} tensors" if existing.get("label") else "2 tensors"
            else:
                G.add_edge(edge.src, edge.dst,
                           tensor_ids=[edge.tensor_id] if edge.tensor_id is not None else [],
                           label="",
                           dtype=str(edge.tensor.dtype) if edge.tensor else "",
                           shape=str(list(edge.tensor.shape)) if edge.tensor else "")

    logger.info("Built fused NX graph from OpGraph: %d nodes, %d edges",
                G.number_of_nodes(), G.number_of_edges())
    return G


# ── Helpers ───────────────────────────────────────────────────────────────────

def _node_attrs(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Extract node attributes from a raw op record."""
    return {
        "op_type": rec["aten_op"],
        "module_path": rec.get("module_path", ""),
        "module_class": rec.get("module_class", ""),
        "layer": rec.get("layer", ""),
        "component": rec.get("component", ""),
        "input_shapes": rec.get("input_shapes", ""),
        "input_dtypes": rec.get("input_dtypes", ""),
        "output_shapes": rec.get("output_shapes", ""),
        "output_dtypes": rec.get("output_dtypes", ""),
        "num_inputs": rec.get("num_inputs", 0),
        "num_outputs": rec.get("num_outputs", 0),
        "label": f"{rec['aten_op']}",
    }


def _fused_node_attrs(frec: Dict[str, Any]) -> Dict[str, Any]:
    """Extract node attributes from a fused op record."""
    op_type = frec.get("op_type", frec.get("fused_op", ""))
    return {
        "op_type": op_type,
        "op_name": frec.get("op_name", ""),
        "aten_ops": frec.get("aten_ops", ""),
        "module_path": frec.get("module_path", ""),
        "module_class": frec.get("module_class", ""),
        "layer": frec.get("layer", ""),
        "fusion_level": frec.get("fusion_level", ""),
        "num_sub_ops": frec.get("num_sub_ops", 0),
        "input_shapes": frec.get("fused_input_shapes", frec.get("input_shapes", "")),
        "input_dtypes": frec.get("fused_input_dtypes", frec.get("input_dtypes", "")),
        "output_shapes": frec.get("fused_output_shapes", frec.get("output_shapes", "")),
        "output_dtypes": frec.get("fused_output_dtypes", frec.get("output_dtypes", "")),
        "label": op_type,
    }


def _find_sub_ops(frec: Dict[str, Any],
                  raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find the raw ops that belong to a fused record.

    Uses the aten_ops string and module_path to match. Falls back to
    returning a minimal record with just the fused I/O info.
    """
    # If _children was preserved (before idx reassignment), use it
    if "_children" in frec:
        return frec["_children"]

    # Otherwise match by module_path and layer.
    # Use path-boundary-aware prefix matching to avoid false positives
    # on module names with shared prefixes (e.g. "model.layers.1" should
    # NOT match "model.layers.10.self_attn" within the same layer).
    module_path = frec.get("module_path", "")
    layer = frec.get("layer", "")

    def _path_match(rec_path: str) -> bool:
        if not rec_path.startswith(module_path):
            return False
        if len(rec_path) == len(module_path):
            return True
        # Ensure dot boundary: "model.layers.1" matches "model.layers.1.self_attn"
        # but NOT "model.layers.10.self_attn"
        return rec_path[len(module_path)] == "."

    matched = [r for r in raw_records
               if _path_match(r.get("module_path", ""))
               and r.get("layer", "") == layer]
    return matched if matched else raw_records[:0]


def _get_tensor_info(rec: Dict[str, Any], tid: int,
                     direction: str) -> Dict[str, str]:
    """Get shape/dtype info for a specific tensor ID from a record."""
    ids_key = f"_{direction}_ids"
    shapes_key = f"{direction}_shapes"
    dtypes_key = f"{direction}_dtypes"

    ids = rec.get(ids_key, [])
    if tid not in ids:
        return {}

    idx = ids.index(tid)
    shapes = _split_shape_list(rec.get(shapes_key, ""))
    dtypes = rec.get(dtypes_key, "").split(", ")

    return {
        "shape": shapes[idx] if idx < len(shapes) else "",
        "dtype": dtypes[idx] if idx < len(dtypes) else "",
    }


def _split_shape_list(s: str) -> List[str]:
    """Split '[1, 128], [64]' into ['[1, 128]', '[64]']."""
    if not s:
        return []
    result = []
    depth = 0
    current = []
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
