from __future__ import annotations

from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.edge import Edge
from zrt.transform.base import GraphPass
from zrt.transform.context import TransformContext


class OffloadPass(GraphPass):
    """OffloadPass for host-device memory offloading."""
    name = "offload"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        """Run OffloadPass on the graph.

        Args:
            graph: Input OpGraph
            ctx: TransformContext with training config

        Returns:
            New OpGraph with offload communication nodes
        """
        g = graph.clone()
        if not ctx.training:
            return g

        offload = ctx.training.offload
        if offload is None or offload.pct <= 0:
            return g
        
        # Insert offload nodes based on offload policy
        if offload.opt_state:
            self._insert_opt_state_offload(g, ctx)
        
        if offload.grads:
            self._insert_grads_offload(g, ctx)
        
        if offload.params:
            self._insert_params_offload(g, ctx)
        
        return g

    def _insert_opt_state_offload(self, graph: OpGraph, ctx: TransformContext):
        """Insert optimizer state offload nodes.
        
        Args:
            graph: OpGraph to modify
            ctx: TransformContext with training config
        """
        # Find optimizer step node
        optimizer_nodes = [
            node for node in graph.nodes.values()
            if node.op_type.startswith("optimizer.")
        ]
        
        for node in optimizer_nodes:
            # Insert D2H before optimizer step (save to host)
            d2h_id = f"comm_d2h_opt_state_{node.id}"
            d2h_node = OpNode(
                id=d2h_id,
                op_type="comm.d2h",
                inputs=node.inputs.copy(),
                outputs=node.inputs.copy(),
                attrs={
                    "role": "opt_state_offload",
                    "pct": ctx.training.offload.pct
                },
                scope=node.scope,
                category="memory"
            )
            d2h_node.annotations["inserted_by"] = "offload_pass"
            self._prepend_comm(graph, node.id, d2h_node)
            
            # Insert H2D after optimizer step (load from host)
            h2d_id = f"comm_h2d_opt_state_{node.id}"
            h2d_node = OpNode(
                id=h2d_id,
                op_type="comm.h2d",
                inputs=node.outputs.copy(),
                outputs=node.outputs.copy(),
                attrs={
                    "role": "opt_state_load",
                    "pct": ctx.training.offload.pct
                },
                scope=node.scope,
                category="memory"
            )
            h2d_node.annotations["inserted_by"] = "offload_pass"
            self._rewire(graph, node.id, h2d_node)

    def _insert_grads_offload(self, graph: OpGraph, ctx: TransformContext):
        """Insert gradients offload nodes.
        
        Args:
            graph: OpGraph to modify
            ctx: TransformContext with training config
        """
        # Find gradient reduction nodes
        grad_nodes = [
            node for node in graph.nodes.values()
            if node.annotations.get("inserted_by") == "data_parallel_pass"
        ]
        
        for node in grad_nodes:
            # Insert D2H after gradient reduction (save to host)
            d2h_id = f"comm_d2h_grads_{node.id}"
            d2h_node = OpNode(
                id=d2h_id,
                op_type="comm.d2h",
                inputs=node.outputs.copy(),
                outputs=node.outputs.copy(),
                attrs={
                    "role": "grads_offload",
                    "pct": ctx.training.offload.pct
                },
                scope=node.scope,
                category="memory"
            )
            d2h_node.annotations["inserted_by"] = "offload_pass"
            self._rewire(graph, node.id, d2h_node)

    def _insert_params_offload(self, graph: OpGraph, ctx: TransformContext):
        """Insert parameters offload nodes.
        
        Args:
            graph: OpGraph to modify
            ctx: TransformContext with training config
        """
        # Find parameter-related nodes (e.g., linear layers)
        param_nodes = [
            node for node in graph.nodes.values()
            if node.op_type in ("aten.mm", "aten.linear", "aten.addmm")
        ]
        
        for node in param_nodes:
            # Insert H2D before parameter usage (load from host)
            h2d_id = f"comm_h2d_params_{node.id}"
            h2d_node = OpNode(
                id=h2d_id,
                op_type="comm.h2d",
                inputs=node.inputs.copy(),
                outputs=node.inputs.copy(),
                attrs={
                    "role": "params_load",
                    "pct": ctx.training.offload.pct
                },
                scope=node.scope,
                category="memory"
            )
            h2d_node.annotations["inserted_by"] = "offload_pass"
            self._prepend_comm(graph, node.id, h2d_node)
            
            # Insert D2H after parameter usage (save to host)
            d2h_id = f"comm_d2h_params_{node.id}"
            d2h_node = OpNode(
                id=d2h_id,
                op_type="comm.d2h",
                inputs=node.outputs.copy(),
                outputs=node.outputs.copy(),
                attrs={
                    "role": "params_offload",
                    "pct": ctx.training.offload.pct
                },
                scope=node.scope,
                category="memory"
            )
            d2h_node.annotations["inserted_by"] = "offload_pass"
            self._rewire(graph, node.id, d2h_node)

    def _prepend_comm(self, g: OpGraph, dst_id: str, comm_node: OpNode):
        """Insert comm_node between all predecessors of dst_id and dst_id itself.
        
        Args:
            g: OpGraph to modify
            dst_id: ID of the destination node
            comm_node: Communication node to insert
        """
        in_edges = [e for e in g.edges if e.dst == dst_id]
        g.edges = [e for e in g.edges if e.dst != dst_id]

        # Add the comm node
        g.nodes[comm_node.id] = comm_node

        # predecessors → comm
        for e in in_edges:
            g.edges.append(Edge(
                src=e.src, src_idx=e.src_idx,
                dst=comm_node.id, dst_idx=e.dst_idx,
                tensor=e.tensor,
            ))

        # comm → dst
        for i, out_tensor in enumerate(comm_node.outputs):
            g.edges.append(Edge(
                src=comm_node.id, src_idx=i,
                dst=dst_id, dst_idx=i,
                tensor=out_tensor,
            ))

        g._rebuild_adjacency()

    def _rewire(self, g: OpGraph, src_id: str, comm_node: OpNode):
        """Insert comm_node between src_id and all its current successors.
        
        Args:
            g: OpGraph to modify
            src_id: ID of the source node
            comm_node: Communication node to insert
        """
        # Collect out-edges of src that we need to reroute
        old_out = [e for e in g.edges if e.src == src_id]

        # Remove those edges from the graph
        g.edges = [e for e in g.edges if e.src != src_id]

        # Add the comm node
        g.nodes[comm_node.id] = comm_node

        # src → comm (one edge per output slot of src)
        src_node = g.nodes[src_id]
        for i, out_tensor in enumerate(src_node.outputs):
            g.edges.append(Edge(
                src=src_id, src_idx=i,
                dst=comm_node.id, dst_idx=i,
                tensor=out_tensor,
            ))

        # comm → old successors
        for e in old_out:
            g.edges.append(Edge(
                src=comm_node.id, src_idx=e.src_idx,
                dst=e.dst, dst_idx=e.dst_idx,
                tensor=e.tensor,
            ))

        g._rebuild_adjacency()
