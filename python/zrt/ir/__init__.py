"""zrt.ir — OpGraph IR: the central data structure of the simulation system.

Public API
----------
Types::

    from zrt.ir import DType, TensorMeta
    from zrt.ir import OpNode
    from zrt.ir import Edge
    from zrt.ir import OpGraph
    from zrt.ir import HierNode, GraphHierarchy

Serialization::

    from zrt.ir import save_json, load_json
    from zrt.ir import opgraph_to_dict, opgraph_from_dict

Adapters (bridge to existing code)::

    from zrt.ir import records_to_opgraph        # RecordingDispatch records → OpGraph
    from zrt.ir import fused_records_to_opgraph  # FusionEngine records → OpGraph
    from zrt.ir import nx_to_opgraph             # NetworkX DiGraph → OpGraph
    from zrt.ir import opgraph_to_nx             # OpGraph → NetworkX DiGraph
    from zrt.ir import records_pair_to_opgraphs  # (raw, fused) records → (OpGraph, OpGraph)
    from zrt.ir import stitch_fwd_bwd           # (fwd_graph, bwd_graph) → unified training OpGraph

Utilities::

    from zrt.ir import dtype_from_torch, dtype_from_str
    from zrt.ir import parse_shape, split_shape_list
    from zrt.ir import infer_category
"""

# ── core types ────────────────────────────────────────────────────────────────
from .types import (
    DType,
    TensorMeta,
    dtype_from_torch,
    dtype_from_str,
    parse_shape,
    split_shape_list,
    memory_bytes,
)

from .node import OpNode, infer_category
from .edge import Edge
from .graph import OpGraph
from .hierarchy import HierNode, GraphHierarchy
from .param_count import count_params, op_short

# ── serialization ─────────────────────────────────────────────────────────────
from .serde import (
    tensor_meta_to_dict, tensor_meta_from_dict,
    op_node_to_dict,     op_node_from_dict,
    edge_to_dict,        edge_from_dict,
    opgraph_to_dict,     opgraph_from_dict,
    save_json,           load_json,
)

# ── adapters ──────────────────────────────────────────────────────────────────
from .adapter import (
    records_to_opgraph,
    fused_records_to_opgraph,
    nx_to_opgraph,
    opgraph_to_nx,
    records_pair_to_opgraphs,
    stitch_fwd_bwd,
)

__all__ = [
    # types
    "DType", "TensorMeta",
    "OpNode",
    "Edge",
    "OpGraph",
    "HierNode", "GraphHierarchy",
    # type utilities
    "dtype_from_torch", "dtype_from_str",
    "parse_shape", "split_shape_list", "memory_bytes",
    "infer_category",
    "count_params", "op_short",
    # serde
    "tensor_meta_to_dict", "tensor_meta_from_dict",
    "op_node_to_dict",     "op_node_from_dict",
    "edge_to_dict",        "edge_from_dict",
    "opgraph_to_dict",     "opgraph_from_dict",
    "save_json",           "load_json",
    # adapters
    "records_to_opgraph",
    "fused_records_to_opgraph",
    "nx_to_opgraph",
    "opgraph_to_nx",
    "records_pair_to_opgraphs",
    "stitch_fwd_bwd",
]
