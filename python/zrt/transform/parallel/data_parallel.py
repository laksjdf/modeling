"""DataParallelPass: per-gradient-group communication insertion."""
from __future__ import annotations

from collections import defaultdict

from python.zrt.ir.edge import Edge
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.transform.base import GraphPass
from python.zrt.transform.context import TransformContext


class DataParallelPass(GraphPass):
    """Insert per-group gradient reduction comm nodes for data parallelism.

    Instead of one global tail node, groups backward/gradient-producing nodes
    by layer and inserts a comm node after each group boundary.

    ZeRO mapping:
      - ZeRO-0 → all_reduce
      - ZeRO-2/3 → reduce_scatter
    """
    name = "data_parallel"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        if not ctx.is_training or ctx.parallel.dp <= 1 or graph.phase != "train_backward":
            return graph

        g = graph.clone()
        dp = ctx.parallel.dp
        zero_stage = ctx.training.zero_stage if ctx.training else 0

        collective = "all_reduce" if zero_stage == 0 else "reduce_scatter"

        # 1. Find gradient-producing nodes and group by layer
        grad_groups: dict[str, list[OpNode]] = defaultdict(list)
        for node in g.topo_sort():
            if self._is_grad_producer(node):
                layer = node.layer if node.layer else "0"
                grad_groups[layer].append(node)

        if not grad_groups:
            return g

        # 2. For each layer group, create a comm node after the last grad node
        for group_idx, (layer_key, nodes) in enumerate(
            sorted(grad_groups.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)
        ):
            last_node = nodes[-1]
            grad_bytes = sum(
                o.mem_bytes for n in nodes for o in n.outputs
            )

            comm_node = OpNode(
                id=f"comm_grad_reduce_layer_{layer_key}",
                op_type=f"comm.{collective}",
                inputs=[],
                outputs=[],
                attrs={
                    "group_size": dp,
                    "collective": collective,
                    "role": "dp_grad_reduce",
                    "bucket_bytes": grad_bytes,
                    "dp_grad_group_idx": group_idx,
                },
                scope=f"data_parallel.grad_reduce.layer_{layer_key}",
                category="communication",
            )

            comm_node.annotations["inserted_by"] = "data_parallel_pass"
            comm_node.annotations["dp_comm"] = True

            dp_overlap = getattr(ctx.training, "dp_overlap_in_bubble", True) if ctx.training else False
            if dp_overlap:
                comm_node.annotations["overlap_in_bubble"] = True

            self._insert_after(g, last_node, comm_node)

        return g

    def _is_grad_producer(self, node: OpNode) -> bool:
        """Check if a node produces gradient tensors."""
        op_lower = node.op_type.lower()
        return "grad" in op_lower or "backward" in op_lower

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
