"""OpNode: single operator node in the computation graph IR."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .types import TensorMeta


# ── Category inference ────────────────────────────────────────────────────────

# op_type prefixes / substrings that indicate communication ops
_COMM_OPS: frozenset[str] = frozenset({
    "comm.all_reduce", "comm.all_gather", "comm.reduce_scatter",
    "comm.all_to_all", "comm.send_recv", "comm.broadcast",
})

# aten ops that are primarily memory-movement, not compute
_MEMORY_OPS: frozenset[str] = frozenset({
    "aten.copy_.default", "aten.clone.default",
    "aten.view.default", "aten.reshape.default",
    "aten.permute.default", "aten.transpose.int",
    "aten.slice.Tensor", "aten.select.int",
    "aten.expand.default", "aten.as_strided.default",
    "aten.contiguous.memory_format",
    "aten.cat.default", "aten.stack.default",
    "aten.split.Tensor", "aten.chunk.default",
    "aten.squeeze.dim", "aten.unsqueeze.default",
    "aten.flatten.using_ints",
})


def infer_category(op_type: str, component: str = "") -> str:
    """Derive 'compute' | 'communication' | 'memory' from op_type/component."""
    if op_type.startswith("comm.") or op_type in _COMM_OPS:
        return "communication"
    if op_type in _MEMORY_OPS:
        return "memory"
    # component hint from classifier
    if component in ("comm", "communication"):
        return "communication"
    return "compute"


# ─────────────────────────────────────────────────────────────────────────────
# OpNode
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpNode:
    """A single operator node in the computation graph.

    Core fields
    -----------
    id          : unique node identifier (e.g. "op_0", "fused_3")
    op_type     : canonical operator name, e.g. "aten.mm.default" or
                  "comm.all_reduce" or "fused.SelfAttention"
    inputs      : ordered list of input TensorMeta (may be empty for constants)
    outputs     : ordered list of output TensorMeta
    attrs       : op-level attributes (kernel params, group size, eps, ...)
    scope       : nn.Module path, e.g. "model.layers.0.self_attn.q_proj"
    category    : "compute" | "communication" | "memory"
    annotations : mutable side-channel filled by Transform passes

    Extra fields preserved from dispatch records
    -------------------------------------------
    op_short        : short op name, e.g. "mm"
    module_class    : nn.Module class name at scope
    layer           : layer index string, e.g. "0"
    component       : semantic component label from classifier
    fused_from      : list of raw op_types absorbed into this node (fused only)
    num_sub_ops     : number of absorbed raw ops (fused only)
    fusion_level    : "leaf" | "parent" (fused only)
    src_file        : source file of the call site
    src_line        : line number
    src_code        : source line text
    """

    # ── required ──
    id: str
    op_type: str

    # ── tensor metadata ──
    inputs:  list[TensorMeta] = field(default_factory=list)
    outputs: list[TensorMeta] = field(default_factory=list)

    # ── op semantics ──
    attrs:    dict[str, Any] = field(default_factory=dict)
    scope:    str = ""
    category: str = "compute"   # "compute" | "communication" | "memory"

    # ── mutable side-channel for Transform passes ──
    annotations: dict[str, Any] = field(default_factory=dict)

    # ── provenance from dispatch records ──
    op_short:     str = ""
    module_class: str = ""
    layer:        str = ""
    component:    str = ""

    # ── fusion metadata ──
    fused_from:   list[str] = field(default_factory=list)  # constituent op_types
    num_sub_ops:  int = 0
    fusion_level: str = ""  # "leaf" | "parent"

    # ── variable name + IO provenance (filled by fusion v2) ──
    name:         str = ""              # e.g. "wo_a" — leaf_attr from dispatch
    provenance:   tuple = ()            # tuple[FusedIOPort, ...] — IO trace

    # ── call-site provenance ──
    src_file: str = ""
    src_line: int = 0
    src_code: str = ""

    # ── module forward-call instance (used by fusion bucketing) ──
    call_id: int = 0

    # ── convenience ──────────────────────────────────────────────────────────

    @property
    def is_fused(self) -> bool:
        return bool(self.fused_from) or self.num_sub_ops > 1

    @property
    def is_comm(self) -> bool:
        return self.category == "communication"

    def input_shapes(self) -> list[tuple[int, ...]]:
        return [t.shape for t in self.inputs]

    def output_shapes(self) -> list[tuple[int, ...]]:
        return [t.shape for t in self.outputs]

    def total_input_bytes(self) -> int:
        return sum(t.mem_bytes for t in self.inputs)

    def total_output_bytes(self) -> int:
        return sum(t.mem_bytes for t in self.outputs)

    def clone(self) -> "OpNode":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        ins  = ", ".join(repr(t) for t in self.inputs)
        outs = ", ".join(repr(t) for t in self.outputs)
        return f"OpNode({self.id}, {self.op_type}, [{ins}] → [{outs}])"
