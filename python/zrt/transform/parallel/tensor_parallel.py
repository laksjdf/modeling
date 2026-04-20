"""Tensor Parallel pass: annotate and split linear ops along the tensor dimension."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext


# Ops we know how to split
_MATMUL_OPS = frozenset({
    "aten.mm.default", "aten.mm",
    "aten.addmm.default", "aten.addmm",
    "aten.linear.default", "aten.linear",
    "aten.bmm.default", "aten.bmm",
})

# Scope keywords → column parallel (split output last dim)
_COL_PARALLEL = ("q_proj", "k_proj", "v_proj",
                 "gate_proj", "up_proj", "w1", "w3")

# Scope keywords → row parallel (split input last dim, need all_reduce after)
_ROW_PARALLEL = ("o_proj", "down_proj", "w2")


@dataclass
class TPRule:
    split_dim:    int         # dim being split on outputs (col) or inputs (row)
    comm_after:   str | None  # "all_reduce" if this rank needs reduction
    input_split:  bool        # whether the input dim is also halved


def _classify(scope: str) -> TPRule | None:
    s = scope.lower()
    if any(k in s for k in _COL_PARALLEL):
        return TPRule(split_dim=-1, comm_after=None, input_split=False)
    if any(k in s for k in _ROW_PARALLEL):
        return TPRule(split_dim=-1, comm_after="all_reduce", input_split=True)
    return None


class TensorParallelPass(GraphPass):
    """Annotate matmul/linear nodes with TP split rules and adjust shapes.

    Column parallel: output last dim → dim / tp
    Row    parallel: first input last dim → dim / tp; comm_after = all_reduce
    Shape propagation is local only (direct node outputs / first input).
    """

    name = "tensor_parallel"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        tp = ctx.parallel.tp
        if tp <= 1:
            return graph

        g = graph.clone()
        for node in g.topo_sort(debug=True):
            if node.op_type not in _MATMUL_OPS:
                continue
            rule = _classify(node.scope)
            if rule is None:
                continue

            node.annotations["tp_split"] = {
                "split_dim":   rule.split_dim,
                "comm_after":  rule.comm_after,
                "input_split": rule.input_split,
                "tp":          tp,
            }

            if rule.input_split:
                # Row parallel: halve the last dim of the first input tensor
                if node.inputs:
                    old = node.inputs[0]
                    if old.shape and old.shape[-1] % tp == 0:
                        new_shape = old.shape[:-1] + (old.shape[-1] // tp,)
                        node.inputs[0] = old.with_shape(new_shape)
            else:
                # Column parallel: halve the last dim of each output tensor
                for i, out in enumerate(node.outputs):
                    if out.shape and out.shape[-1] % tp == 0:
                        new_shape = out.shape[:-1] + (out.shape[-1] // tp,)
                        node.outputs[i] = out.with_shape(new_shape)
                        # Also update the corresponding out-edges
                        for e in g.out_edges(node.id):
                            if e.src_idx == i and e.tensor is not None:
                                e.tensor = e.tensor.with_shape(new_shape)

        return g
