"""DataParallelPass: global gradient reduction communication insertion."""
from __future__ import annotations

import logging

from python.zrt.ir.edge import Edge
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.transform.base import GraphPass
from python.zrt.transform.context import TransformContext

logger = logging.getLogger(__name__)

_BWD_PHASES = {"bwd", "backward", "train_backward"}


class DataParallelPass(GraphPass):
    """Insert global gradient reduction comm node for data parallelism.

    ZeRO-0/1: Single communication at the **end of the entire backward pass**.
      - ZeRO-0/1 → all_reduce (full gradient sync)
      - ZeRO-2   → reduce_scatter (gradient sharding)
    ZeRO-3: Skipped here; handled by ZeroFSDPPass (per-layer FSDP communication).

    Annotations written:
      - node.annotations["dp_comm"] = True
      - node.annotations["dp_overlap_in_bubble"] = bool (if training.dp_overlap_in_bubble)
      - attrs["bucket_bytes"] = total gradient tensor bytes
    """
    name = "data_parallel"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        if not ctx.is_training or ctx.parallel.dp <= 1:
            return graph

        g = graph.clone()
        dp = ctx.parallel.dp
        zero_stage = ctx.training.zero_stage if ctx.training else 0

        # ZeRO-3 handled by ZeroFSDPPass (per-layer FSDP communication)
        if zero_stage >= 3:
            return g

        # Check if we should process this graph (backward-phase only)
        graph_phase = g.metadata.get("phase", "")
        if graph_phase and graph_phase not in ("train_backward", "backward"):
            return g

        # ZeRO mapping:
        #   ZeRO-0/1 → all_reduce (full gradient sync)
        #   ZeRO-2   → reduce_scatter (gradient sharding)
        collective = "all_reduce" if zero_stage <= 1 else "reduce_scatter"

        # 1. Find all backward nodes in topological order
        bwd_nodes = [n for n in g.topo_sort() if self._is_backward_node(n)]
        if not bwd_nodes:
            return g

        # 2. Find the last backward node in topological order
        last_bwd_node = bwd_nodes[-1]

        # 3. Calculate total gradient bytes from all backward node outputs
        total_grad_bytes = sum(
            o.mem_bytes for n in bwd_nodes for o in n.outputs
            if hasattr(o, 'mem_bytes')
        )

        comm_node = OpNode(
            id="comm_dp_grad_reduce",
            op_type=f"comm.{collective}",
            inputs=[],
            outputs=[],
            attrs={
                "group_size": dp,
                "collective": collective,
                "role": "dp_grad_reduce",
                "bucket_bytes": total_grad_bytes,
                "dp_grad_group_idx": 0,
            },
            scope="data_parallel.grad_reduce.global",
            category="communication",
        )

        comm_node.annotations["inserted_by"] = "data_parallel_pass"
        comm_node.annotations["dp_comm"] = True
        comm_node.annotations["phase"] = "bwd"

        dp_overlap = getattr(ctx.training, "dp_overlap_in_bubble", True) if ctx.training else True
        if dp_overlap:
            comm_node.annotations["overlap_in_bubble"] = True

        self._insert_after(g, last_bwd_node, comm_node)

        return g

    def _is_backward_node(self, node: OpNode) -> bool:
        """Check if a node is a backward-phase node."""
        # Check node-level phase annotation
        phase = node.annotations.get("phase", "")
        if phase in _BWD_PHASES:
            return True

        # Check op_type for backward indicators
        op_lower = node.op_type.lower()
        if "grad" in op_lower or "backward" in op_lower:
            return True

        # Check scope for grad context
        scope = node.scope.lower()
        return "grad" in scope

    def _insert_after(self, graph: OpGraph, src_node: OpNode, comm_node: OpNode) -> None:
        """Insert comm_node between src_node and all its current successors."""
        src_id = src_node.id

        # Collect out-edges of src that we need to reroute
        old_out = [e for e in graph.edges if e.src == src_id]
        if not old_out:
            # src is a leaf node — just add the comm node
            graph.nodes[comm_node.id] = comm_node
            graph._succ[comm_node.id] = []
            graph._pred[comm_node.id] = []
            # Create a single edge src → comm
            if src_node.outputs:
                graph.edges.append(Edge(
                    src=src_id, src_idx=0,
                    dst=comm_node.id, dst_idx=0,
                    tensor=src_node.outputs[0],
                ))
            graph._rebuild_adjacency()
            return

        # Remove out-edges from src
        graph.edges = [e for e in graph.edges if e.src != src_id]

        # Add comm node
        graph.nodes[comm_node.id] = comm_node
        graph._succ[comm_node.id] = []
        graph._pred[comm_node.id] = []

        # src → comm
        for i, out_tensor in enumerate(src_node.outputs):
            graph.edges.append(Edge(
                src=src_id, src_idx=i,
                dst=comm_node.id, dst_idx=i,
                tensor=out_tensor,
            ))

        # comm → old successors
        for e in old_out:
            graph.edges.append(Edge(
                src=comm_node.id, src_idx=e.src_idx,
                dst=e.dst, dst_idx=e.dst_idx,
                tensor=e.tensor,
            ))

        graph._rebuild_adjacency()
