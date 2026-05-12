"""Adapter: convert between the new OpGraph IR and existing representations.

Three conversion directions are provided:

1. ``records_to_opgraph``
   Raw op-record dicts (from RecordingDispatch) → OpGraph.
   This is the **preferred** path for new code: builds OpGraph directly
   from dispatch output without going through NetworkX.

2. ``fused_records_to_opgraph``
   Fused op-record dicts (from FusionEngine) → OpGraph.
   Uses the ``_children`` list inside each fused record to reconstruct
   TensorMeta and connectivity.

3. ``nx_to_opgraph`` / ``opgraph_to_nx``
   Bidirectional bridge for code that still uses NetworkX DiGraph.
   Useful during the transition period when only some modules have been
   migrated to OpGraph.

All converters preserve the full set of fields stored in the existing records
(module_path, component, layer, src_*, extra_args, ...) in OpNode's
provenance fields or ``attrs`` dict so nothing is silently lost.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from .edge import Edge
from .graph import OpGraph
from .node import OpNode, infer_category
from .types import (
    DType,
    TensorMeta,
    dtype_from_torch,
    parse_shape,
    split_shape_list,
    memory_bytes,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tid_str(tid: int) -> str:
    """Integer tensor ID → stable string key."""
    return f"t{tid}"


def _fake_tid_str(node_id: str, slot: int, direction: str) -> str:
    """Synthesise a tensor ID when no real dispatch ID is available."""
    return f"{node_id}_{direction}{slot}"


def _parse_tensor_list(
    ids:    list[int],
    shapes: str,
    dtypes: str,
    use_ids: bool = True,
) -> list[TensorMeta]:
    """Build a list of TensorMeta from parallel lists of IDs / shape string / dtype string.

    ``shapes`` is a comma-separated bracketed string, e.g. "[1, 128], [7168]".
    ``dtypes`` is a comma-separated string, e.g. "torch.bfloat16, torch.bfloat16".
    """
    shape_parts = split_shape_list(shapes) if shapes else []
    dtype_parts = [s.strip() for s in dtypes.split(",")] if dtypes else []

    result: list[TensorMeta] = []
    for i, tid in enumerate(ids):
        shape_s = shape_parts[i] if i < len(shape_parts) else ""
        dtype_s = dtype_parts[i] if i < len(dtype_parts) else ""
        tensor_id = _tid_str(tid) if use_ids else f"slot{i}"
        result.append(TensorMeta.from_strings(tensor_id, shape_s, dtype_s))
    return result


def _parse_attrs(extra_args: str) -> dict[str, Any]:
    """Parse the JSON extra_args string stored in dispatch records."""
    if not extra_args:
        return {}
    try:
        return json.loads(extra_args)
    except (json.JSONDecodeError, TypeError):
        return {"raw": extra_args}


# ─────────────────────────────────────────────────────────────────────────────
# 1. records_to_opgraph  (raw dispatch records → OpGraph)
# ─────────────────────────────────────────────────────────────────────────────

def records_to_opgraph(
    records:   list[dict[str, Any]],
    name:      str,
    phase:     str,
    metadata:  dict[str, Any] | None = None,
) -> OpGraph:
    """Convert raw RecordingDispatch records directly to an OpGraph.

    This is the preferred factory for new capture code.  It builds TensorMeta
    from the per-slot shape/dtype info embedded in each record, creating a
    richer IR than the NetworkX path.

    Parameters
    ----------
    records  : list of op-record dicts produced by RecordingDispatch
    name     : graph name (e.g. "DeepSeek-V3_prefill")
    phase    : "prefill" | "decode" | "forward"
    metadata : arbitrary extra metadata (batch_size, seq_len, model_id, ...)

    Returns
    -------
    OpGraph with nodes and data-flow edges.
    """
    graph = OpGraph(name=name, phase=phase, metadata=metadata or {})

    # ── Pass 1: create OpNodes ────────────────────────────────────────────────
    for rec in records:
        node_id  = f"op_{rec['node_id']}"
        op_type  = rec["aten_op"]
        component = rec.get("component", "")

        inputs  = _parse_tensor_list(
            rec.get("_input_ids",  []),
            rec.get("input_shapes",  ""),
            rec.get("input_dtypes",  ""),
        )
        outputs = _parse_tensor_list(
            rec.get("_output_ids", []),
            rec.get("output_shapes", ""),
            rec.get("output_dtypes", ""),
        )

        node = OpNode(
            id=node_id,
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            attrs=_parse_attrs(rec.get("extra_args", "")),
            scope=rec.get("module_path", ""),
            category=infer_category(op_type, component),
            op_short=rec.get("op_short", ""),
            module_class=rec.get("module_class", ""),
            layer=str(rec.get("layer", "")),
            component=component,
            src_file=rec.get("src_file", ""),
            src_line=rec.get("src_line", 0),
            src_code=rec.get("src_code", ""),
            name=rec.get("leaf_attr", ""),
            call_id=int(rec.get("call_id", 0) or 0),
        )
        graph.add_node(node)

    # ── Pass 2: build tensor_id → (producer_node_id, output_slot) map ────────
    tensor_producer: dict[int, tuple[str, int]] = {}
    for rec in records:
        node_id = f"op_{rec['node_id']}"
        for slot, tid in enumerate(rec.get("_output_ids", [])):
            if tid not in tensor_producer:
                tensor_producer[tid] = (node_id, slot)

    # ── Pass 3: create data-flow edges ────────────────────────────────────────
    seen_pairs: dict[tuple[str, str], int] = {}   # (src, dst) → edge count

    for rec in records:
        consumer_id = f"op_{rec['node_id']}"
        consumer_idx = int(consumer_id.split("_")[1])
        for dst_idx, tid in enumerate(rec.get("_input_ids", [])):
            if tid not in tensor_producer:
                continue
            producer_id, src_idx = tensor_producer[tid]
            if producer_id == consumer_id:
                continue  # skip self-loops

            # Skip backward edges (where producer comes after consumer in execution order)
            # This handles KV cache aliasing in decode phase where tensors produced late
            # should not feed back to earlier layers
            producer_idx = int(producer_id.split("_")[1])
            if producer_idx > consumer_idx:
                logger.debug(
                    "Skipping backward edge: %s (op %d) -> %s (op %d) for tensor %d",
                    producer_id, producer_idx, consumer_id, consumer_idx, tid
                )
                continue

            # Look up the TensorMeta we built in pass 1
            producer_node = graph.nodes.get(producer_id)
            tensor: TensorMeta | None = None
            if producer_node and src_idx < len(producer_node.outputs):
                tensor = producer_node.outputs[src_idx]

            edge = Edge(
                src=producer_id,
                src_idx=src_idx,
                dst=consumer_id,
                dst_idx=dst_idx,
                tensor=tensor,
                tensor_id=tid,
            )
            graph.add_edge(edge)

    logger.debug(
        "records_to_opgraph: %d nodes, %d edges (phase=%s)",
        len(graph.nodes), len(graph.edges), phase,
    )
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# 2. fused_records_to_opgraph  (fused records → OpGraph)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fused_external_io(
    fused_rec: dict[str, Any],
) -> tuple[set[int], set[int]]:
    """Return (external_input_tids, external_output_tids) for a fused record.

    Uses the ``_children`` list embedded by FusionEngine.
    """
    children = fused_rec.get("_children", [])
    produced: set[int] = set()
    consumed: set[int] = set()
    for child in children:
        for tid in child.get("_output_ids", []):
            produced.add(tid)
        for tid in child.get("_input_ids", []):
            consumed.add(tid)
    return consumed - produced, produced - consumed


def fused_records_to_opgraph(
    fused_records: list[dict[str, Any]],
    name:          str,
    phase:         str,
    metadata:      dict[str, Any] | None = None,
) -> OpGraph:
    """Convert FusionEngine output records to an OpGraph.

    Each fused record corresponds to one (potentially multi-op) node.
    External I/O is derived from the ``_children`` list.

    Parameters
    ----------
    fused_records : list of fused op records from FusionEngine.fuse()
    name          : graph name
    phase         : "prefill" | "decode" | "forward"
    metadata      : arbitrary extra metadata
    """
    graph = OpGraph(name=name, phase=phase, metadata=metadata or {})

    # Precompute external IO per fused record
    io_list: list[tuple[set[int], set[int]]] = [
        _compute_fused_external_io(fr) for fr in fused_records
    ]

    # ── Pass 1: create OpNodes ────────────────────────────────────────────────
    for frec, (ext_in, ext_out) in zip(fused_records, io_list):
        node_id  = f"fused_{frec['node_id']}"
        op_type  = frec.get("fused_op", frec.get("op_type", "fused.unknown"))
        children = frec.get("_children", [])

        # Build inputs from first child's input shapes (external only)
        first_child = children[0] if children else frec
        last_child  = children[-1] if children else frec

        # Use fused_input_shapes if available, else fall back to first child
        in_shapes_str  = frec.get("fused_input_shapes",  first_child.get("input_shapes",  ""))
        in_dtypes_str  = frec.get("fused_input_dtypes",  first_child.get("input_dtypes",  ""))
        out_shapes_str = frec.get("fused_output_shapes", last_child.get("output_shapes", ""))
        out_dtypes_str = frec.get("fused_output_dtypes", last_child.get("output_dtypes", ""))

        # IDs for external tensors (use sorted order for stability)
        in_ids  = sorted(ext_in)
        out_ids = sorted(ext_out)

        inputs  = _parse_tensor_list(in_ids,  in_shapes_str,  in_dtypes_str)
        outputs = _parse_tensor_list(out_ids, out_shapes_str, out_dtypes_str)

        # Collect aten ops from children for provenance
        fused_from = list(dict.fromkeys(
            c.get("aten_op", "") for c in children if c.get("aten_op")
        ))

        component = frec.get("component", children[0].get("component", "") if children else "")

        node = OpNode(
            id=node_id,
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            attrs={},
            scope=frec.get("module_path", ""),
            category=infer_category(op_type, component),
            module_class=frec.get("module_class", ""),
            layer=str(frec.get("layer", "")),
            component=component,
            fused_from=fused_from,
            num_sub_ops=frec.get("num_sub_ops", len(children)),
            fusion_level=frec.get("fusion_level", ""),
        )
        graph.add_node(node)

    # ── Pass 2: build tensor_id → producer fused-node map ────────────────────
    tensor_producer: dict[int, str] = {}
    for frec, (ext_in, ext_out) in zip(fused_records, io_list):
        node_id = f"fused_{frec['node_id']}"
        for tid in ext_out:
            if tid not in tensor_producer:
                tensor_producer[tid] = node_id

    # ── Pass 3: create data-flow edges ────────────────────────────────────────
    for frec, (ext_in, ext_out) in zip(fused_records, io_list):
        consumer_id = f"fused_{frec['node_id']}"
        consumer_idx = int(consumer_id.split("_")[1])
        for dst_idx, tid in enumerate(sorted(ext_in)):
            if tid not in tensor_producer:
                continue
            producer_id = tensor_producer[tid]
            if producer_id == consumer_id:
                continue

            # Skip backward edges in fused graphs too
            producer_idx = int(producer_id.split("_")[1])
            if producer_idx > consumer_idx:
                logger.debug(
                    "Skipping backward edge in fused graph: %s (op %d) -> %s (op %d) for tensor %d",
                    producer_id, producer_idx, consumer_id, consumer_idx, tid
                )
                continue

            producer_node = graph.nodes.get(producer_id)
            tensor: TensorMeta | None = None
            if producer_node:
                try:
                    src_idx = sorted(io_list[
                        next(i for i, fr in enumerate(fused_records)
                             if f"fused_{fr['node_id']}" == producer_id)
                    ][1]).index(tid)
                    if src_idx < len(producer_node.outputs):
                        tensor = producer_node.outputs[src_idx]
                except (StopIteration, ValueError):
                    src_idx = 0

            edge = Edge(
                src=producer_id,
                src_idx=0,           # simplified: slot tracking for fused
                dst=consumer_id,
                dst_idx=dst_idx,
                tensor=tensor,
                tensor_id=tid,
            )
            graph.add_edge(edge)

    logger.debug(
        "fused_records_to_opgraph: %d nodes, %d edges (phase=%s)",
        len(graph.nodes), len(graph.edges), phase,
    )
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# 3a. nx_to_opgraph  (NetworkX DiGraph → OpGraph)
# ─────────────────────────────────────────────────────────────────────────────

def nx_to_opgraph(
    G:        "nx.DiGraph",  # noqa: F821
    name:     str,
    phase:    str,
    metadata: dict[str, Any] | None = None,
    is_fused: bool = False,
) -> OpGraph:
    """Convert a NetworkX DiGraph (built by graph_builder.py) to an OpGraph.

    Works for both raw graphs (node IDs "op_N") and fused graphs ("fused_N").

    Limitations
    -----------
    - TensorMeta IDs are synthesised (no original dispatch IDs available in NX)
    - Per-slot shapes/dtypes for edges are approximate (taken from edge attrs)
    - Call-site provenance (src_file/line/code) is not available in NX nodes
    """
    graph = OpGraph(name=name, phase=phase, metadata=metadata or {})

    # ── nodes ─────────────────────────────────────────────────────────────────
    for nid, attrs in G.nodes(data=True):
        op_type   = attrs.get("op_type", "unknown")
        scope     = attrs.get("module_path", "")
        component = attrs.get("component", "")
        layer     = str(attrs.get("layer", ""))
        mc        = attrs.get("module_class", "")

        in_shapes_str  = attrs.get("input_shapes",  "")
        in_dtypes_str  = attrs.get("input_dtypes",  "")
        out_shapes_str = attrs.get("output_shapes", "")
        out_dtypes_str = attrs.get("output_dtypes", "")

        # Build TensorMeta from shape/dtype strings using synthesised IDs
        in_shape_parts  = split_shape_list(in_shapes_str)
        in_dtype_parts  = [s.strip() for s in in_dtypes_str.split(",")] if in_dtypes_str else []
        out_shape_parts = split_shape_list(out_shapes_str)
        out_dtype_parts = [s.strip() for s in out_dtypes_str.split(",")] if out_dtypes_str else []

        def _build_metas(nid, direction, shape_parts, dtype_parts, count):
            metas = []
            n = count or max(len(shape_parts), len(dtype_parts), 0)
            for i in range(n):
                tid   = _fake_tid_str(nid, i, direction)
                ss    = shape_parts[i] if i < len(shape_parts) else ""
                ds    = dtype_parts[i]  if i < len(dtype_parts)  else ""
                metas.append(TensorMeta.from_strings(tid, ss, ds))
            return metas

        num_in  = attrs.get("num_inputs",  0)
        num_out = attrs.get("num_outputs", 0)
        inputs  = _build_metas(nid, "in",  in_shape_parts,  in_dtype_parts,  num_in)
        outputs = _build_metas(nid, "out", out_shape_parts, out_dtype_parts, num_out)

        fused_from: list[str] = []
        num_sub = 0
        fusion_level = ""
        if is_fused:
            aten_ops_str = attrs.get("aten_ops", "")
            if aten_ops_str:
                fused_from = [s.strip() for s in aten_ops_str.replace("→", ",").split(",")
                              if s.strip()]
            num_sub      = attrs.get("num_sub_ops", 0)
            fusion_level = attrs.get("fusion_level", "")

        node = OpNode(
            id=nid,
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            scope=scope,
            category=infer_category(op_type, component),
            module_class=mc,
            layer=layer,
            component=component,
            fused_from=fused_from,
            num_sub_ops=num_sub,
            fusion_level=fusion_level,
        )
        graph.add_node(node)

    # ── edges ─────────────────────────────────────────────────────────────────
    for src, dst, eattrs in G.edges(data=True):
        # NX may have multiple tensor IDs per edge; we create one Edge per tensor.
        tensor_ids: list[int] = eattrs.get("tensor_ids", [])
        shape_s = str(eattrs.get("shape", ""))
        dtype_s = str(eattrs.get("dtype", ""))
        tensor: TensorMeta | None = None
        if shape_s or dtype_s:
            tid_str = _fake_tid_str(src, 0, "edge")
            tensor = TensorMeta.from_strings(tid_str, shape_s, dtype_s)

        edge = Edge(
            src=src,
            src_idx=0,
            dst=dst,
            dst_idx=0,
            tensor=tensor,
            tensor_id=tensor_ids[0] if tensor_ids else None,
        )
        graph.add_edge(edge)

    logger.debug(
        "nx_to_opgraph: %d nodes, %d edges (phase=%s, fused=%s)",
        len(graph.nodes), len(graph.edges), phase, is_fused,
    )
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# 3b. opgraph_to_nx  (OpGraph → NetworkX DiGraph)
# ─────────────────────────────────────────────────────────────────────────────

def opgraph_to_nx(graph: OpGraph) -> "nx.DiGraph":  # noqa: F821
    """Convert an OpGraph back to a NetworkX DiGraph.

    The resulting NX graph uses the same node/edge attribute schema as
    graph_builder.py so it is compatible with existing exporters
    (graph_exporter.py, excel_writer.py, etc.).
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("networkx is required for opgraph_to_nx") from exc

    G: nx.DiGraph = nx.DiGraph()

    # ── nodes ─────────────────────────────────────────────────────────────────
    for nid, node in graph.nodes.items():
        # Re-format shapes/dtypes as comma-separated strings
        in_shapes  = ", ".join(str(list(t.shape)) for t in node.inputs)
        in_dtypes  = ", ".join(f"torch.{t.dtype.value}" if not t.dtype.value.startswith("torch") else t.dtype.value
                               for t in node.inputs)
        out_shapes = ", ".join(str(list(t.shape)) for t in node.outputs)
        out_dtypes = ", ".join(f"torch.{t.dtype.value}" if not t.dtype.value.startswith("torch") else t.dtype.value
                               for t in node.outputs)

        attrs: dict = {
            "op_type":     node.op_type,
            "module_path": node.scope,
            "module_class": node.module_class,
            "layer":       node.layer,
            "component":   node.component,
            "category":    node.category,
            "input_shapes":  in_shapes,
            "input_dtypes":  in_dtypes,
            "output_shapes": out_shapes,
            "output_dtypes": out_dtypes,
            "num_inputs":  len(node.inputs),
            "num_outputs": len(node.outputs),
            "label":       node.op_type,
            "annotations": node.annotations,
        }
        if node.is_fused:
            attrs.update({
                "fused_op":    node.op_type,
                "aten_ops":    " → ".join(node.fused_from),
                "num_sub_ops": node.num_sub_ops,
                "fusion_level": node.fusion_level,
                "fused_input_shapes":  in_shapes,
                "fused_input_dtypes":  in_dtypes,
                "fused_output_shapes": out_shapes,
                "fused_output_dtypes": out_dtypes,
            })
        G.add_node(nid, **attrs)

    # ── edges ─────────────────────────────────────────────────────────────────
    # Accumulate multiple Edge objects for the same (src, dst) pair
    edge_groups: dict[tuple[str, str], list[Edge]] = {}
    for e in graph.edges:
        key = (e.src, e.dst)
        edge_groups.setdefault(key, []).append(e)

    for (src, dst), edges in edge_groups.items():
        tensor_ids = [e.tensor_id for e in edges if e.tensor_id is not None]
        # shape / dtype from first data edge
        first_data = next((e for e in edges if e.tensor), None)
        shape_s = str(list(first_data.tensor.shape)) if first_data and first_data.tensor else ""
        dtype_s = (f"torch.{first_data.tensor.dtype.value}"
                   if first_data and first_data.tensor else "")
        G.add_edge(src, dst,
                   tensor_ids=tensor_ids,
                   shape=shape_s,
                   dtype=dtype_s,
                   label=shape_s)

    logger.debug(
        "opgraph_to_nx: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges()
    )
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: both phases at once
# ─────────────────────────────────────────────────────────────────────────────

def records_pair_to_opgraphs(
    raw_records:   list[dict[str, Any]],
    fused_records: list[dict[str, Any]],
    name:          str,
    phase:         str,
    metadata:      dict[str, Any] | None = None,
) -> tuple[OpGraph, OpGraph]:
    """Build (raw_graph, fused_graph) from the two record lists in one call."""
    raw   = records_to_opgraph(raw_records,   f"{name}_raw",   phase, metadata)
    fused = fused_records_to_opgraph(fused_records, f"{name}_fused", phase, metadata)
    return raw, fused


# ─────────────────────────────────────────────────────────────────────────────
# 4. stitch_fwd_bwd  (forward + backward OpGraph → unified training OpGraph)
# ─────────────────────────────────────────────────────────────────────────────

# Scope suffixes that carry trainable weight parameters.
_PARAM_SCOPE_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "embed_tokens", "lm_head",
    "q_a_proj", "kv_a_proj", "q_b_proj", "kv_b_proj",
    "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj",
)

# Op types that represent weight/parameter reads (as opposed to data-path ops).
_PARAM_READ_OPS = frozenset({
    "aten.mm.default", "aten.addmm.default", "aten.linear.default",
    "aten.bmm.default", "aten.matmul.default",
    "aten.embedding.default",
    "aten._convolution.default",
})


def _is_param_node(node: OpNode) -> bool:
    """Heuristic: does this node read a weight parameter?

    Detects by either:
    1. Scope-suffix match (Llama/DeepSeek-style naming) + param-read op type.
    2. Op type alone for embedding/conv ops (always param reads).
    """
    if node.op_type not in _PARAM_READ_OPS:
        return False
    # Embedding and conv ops are always parameter reads regardless of scope
    if node.op_type in ("aten.embedding.default", "aten._convolution.default"):
        return True
    scope = node.scope
    return any(scope.rstrip(".").endswith(s) for s in _PARAM_SCOPE_SUFFIXES)


def stitch_fwd_bwd(
    fwd_graph: OpGraph,
    bwd_graph: OpGraph,
    name: str | None = None,
) -> OpGraph:
    """Merge a forward and backward *OpGraph* into a single unified graph.

    The result represents one complete training step (single rank, single PP
    stage).  Backward node IDs are prefixed with ``"bwd_"`` to avoid
    collisions with forward IDs.  Every node receives an ``annotations["phase"]``
    of ``"fwd"`` or ``"bwd"``, and weight-read nodes are annotated with
    ``annotations["is_param"] = True``.

    Cross-graph dependency edges are added when a backward node consumes a
    tensor that was produced by a forward node.  Matching uses **tensor ID**
    (when the same *TensorTracker* was shared across phases) as the primary
    strategy, with (shape, dtype) as a fallback for tensors without IDs.

    Parameters
    ----------
    fwd_graph : OpGraph
        Computation graph from a ``train_forward`` trace.
    bwd_graph : OpGraph
        Computation graph from a ``train_backward`` trace.
    name : str or None
        Name for the unified graph.  Defaults to ``fwd_graph.name``.

    Returns
    -------
    OpGraph
        Unified graph with ``phase="train"``.
    """
    combined = OpGraph(
        name=name or fwd_graph.name,
        phase="train",
        metadata={
            **bwd_graph.metadata,     # bwd first so fwd wins on conflicts
            **fwd_graph.metadata,
            "fwd_graph_name": fwd_graph.name,
            "bwd_graph_name": bwd_graph.name,
            "fwd_metadata": dict(fwd_graph.metadata),
            "bwd_metadata": dict(bwd_graph.metadata),
            "fwd_bwd_stitched": True,
        },
    )

    # ── Phase 1: add forward nodes ──────────────────────────────────────────
    for node in fwd_graph:
        n = node.clone()
        n.annotations["phase"] = "fwd"
        if _is_param_node(n):
            n.annotations["is_param"] = True
        combined.add_node(n)

    # ── Phase 2: add backward nodes (with bwd_ prefix) ─────────────────────
    for node in bwd_graph:
        n = node.clone()
        n.id = f"bwd_{n.id}"
        n.annotations["phase"] = "bwd"
        combined.add_node(n)

    # ── Phase 3: copy forward→forward edges ─────────────────────────────────
    for edge in fwd_graph.edges:
        combined.add_edge(Edge(
            src=edge.src,
            src_idx=edge.src_idx,
            dst=edge.dst,
            dst_idx=edge.dst_idx,
            tensor=edge.tensor,
            tensor_id=edge.tensor_id,
        ))

    # ── Phase 4: copy backward→backward edges (prefixed IDs) ────────────────
    for edge in bwd_graph.edges:
        combined.add_edge(Edge(
            src=f"bwd_{edge.src}",
            src_idx=edge.src_idx,
            dst=f"bwd_{edge.dst}",
            dst_idx=edge.dst_idx,
            tensor=edge.tensor,
            tensor_id=edge.tensor_id,
        ))

    # ── Phase 5: cross-graph dependency edges ───────────────────────────────
    # Primary strategy: match by tensor_id (exact, when shared TensorTracker)
    # Fallback: match by (shape, dtype) with same-layer preference
    fwd_id_index: dict[str, tuple[str, int]] = {}  # tensor_id_str → (node_id, slot)
    fwd_tensor_index: dict[tuple[tuple[int, ...], Any], list[tuple[str, int]]] = {}
    for node in fwd_graph:
        for slot, tmeta in enumerate(node.outputs):
            fwd_id_index[tmeta.id] = (node.id, slot)
            key = (tmeta.shape, tmeta.dtype)
            fwd_tensor_index.setdefault(key, []).append((node.id, slot))

    cross_edges = 0
    for node in bwd_graph:
        bwd_id = f"bwd_{node.id}"
        for dst_idx, tmeta in enumerate(node.inputs):
            fwd_node_id: str | None = None
            src_idx = 0
            matched_tensor: TensorMeta | None = None

            # Strategy 1: exact tensor ID match
            if tmeta.id in fwd_id_index:
                fwd_node_id, src_idx = fwd_id_index[tmeta.id]
                src_node = fwd_graph.nodes.get(fwd_node_id)
                if src_node and src_idx < len(src_node.outputs):
                    matched_tensor = src_node.outputs[src_idx]

            # Strategy 2: shape+dtype heuristic with same-layer preference
            if fwd_node_id is None:
                key = (tmeta.shape, tmeta.dtype)
                candidates = fwd_tensor_index.get(key)
                if candidates:
                    fwd_node_id, src_idx = _best_cross_match(
                        candidates, node.layer, fwd_graph, node.scope)
                    src_node = fwd_graph.nodes.get(fwd_node_id)
                    if src_node and src_idx < len(src_node.outputs):
                        matched_tensor = src_node.outputs[src_idx]

            if fwd_node_id is None:
                continue

            combined.add_edge(Edge(
                src=fwd_node_id,
                src_idx=src_idx,
                dst=bwd_id,
                dst_idx=dst_idx,
                tensor=matched_tensor or tmeta,
            ))
            cross_edges += 1

    logger.debug(
        "stitch_fwd_bwd: %d fwd + %d bwd nodes, %d cross-graph edges",
        len(fwd_graph), len(bwd_graph), cross_edges,
    )
    return combined


def _best_cross_match(
    candidates: list[tuple[str, int]],
    bwd_layer: str,
    fwd_graph: OpGraph,
    bwd_scope: str = "",
) -> tuple[str, int]:
    """Pick the best forward-output candidate for a backward input.

    Priority order:
    1. Same layer AND scope prefix match (most specific).
    2. Same layer only.
    3. Last candidate (LIFO — most recently produced tensor is most
       likely the immediate input to the gradient op).
    """
    if len(candidates) == 1:
        return candidates[0]

    same_layer = []
    for nid, slot in candidates:
        fwd_node = fwd_graph.nodes.get(nid)
        if fwd_node and fwd_node.layer == bwd_layer:
            same_layer.append((nid, slot, fwd_node.scope))

    if same_layer:
        # Among same-layer candidates, prefer scope prefix match
        if bwd_scope:
            for nid, slot, scope in same_layer:
                if bwd_scope.startswith(scope) or scope.startswith(bwd_scope):
                    return (nid, slot)
        # LIFO: return last same-layer candidate (most recently produced)
        return (same_layer[-1][0], same_layer[-1][1])

    # No same-layer match; LIFO fallback
    return candidates[-1]
