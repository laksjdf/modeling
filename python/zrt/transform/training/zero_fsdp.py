from __future__ import annotations

from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.ir.edge import Edge
from python.zrt.transform.base import GraphPass
from python.zrt.transform.context import TransformContext


class ZeroFSDPPass(GraphPass):
    """ZeroFSDPPass for ZeRO/FSDP sharding annotations."""
    name = "zero_fsdp"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        """Run ZeroFSDPPass on the graph.
        
        Args:
            graph: Input OpGraph
            ctx: TransformContext with training config
            
        Returns:
            New OpGraph with Zero/FSDP annotations
        """
        g = graph.clone()
        if not ctx.training:
            return g
        
        z = ctx.training.zero_stage
        dp = ctx.parallel.dp
        
        # shard factors per memory bucket
        g.metadata["zero"] = {
            "stage": z,
            "weight_shard": dp if z >= 3 else 1,
            "grad_shard": dp if z >= 2 else 1,
            "optstate_shard": dp if z >= 1 else 1,
        }
        
        # ZeRO-3 / FSDP: insert per-layer communication
        # Forward:  all_gather before each layer (gather sharded weights)
        # Backward: all_gather before each layer + reduce_scatter after
        if z >= 3:
            self._insert_fsdp_ag(g, ctx)
        
        return g

    def _insert_fsdp_ag(self, graph: OpGraph, ctx: TransformContext):
        """Insert FSDP all-gather / reduce-scatter collectives for ZeRO-3.

        Forward:  all_gather before first fwd node (gather sharded weights).
        Backward: all_gather before first bwd node (re-gather weights) +
                  reduce_scatter after last bwd node (scatter gradients).

        Args:
            graph: OpGraph to modify
            ctx: TransformContext with training config
        """
        dp = ctx.parallel.dp

        layers = self._get_layers(graph)

        for layer in layers:
            fwd_nodes = [
                n for n in graph.topo_sort()
                if self._node_in_layer(n, layer)
                and n.annotations.get("phase") == "fwd"
            ]
            bwd_nodes = [
                n for n in graph.topo_sort()
                if self._node_in_layer(n, layer)
                and n.annotations.get("phase") == "bwd"
            ]

            if fwd_nodes:
                first_fwd = fwd_nodes[0]
                ag_fwd_id = f"comm_fsdp_ag_{first_fwd.id}"
                ag_fwd = OpNode(
                    id=ag_fwd_id,
                    op_type="comm.all_gather",
                    inputs=first_fwd.inputs.copy(),
                    outputs=first_fwd.inputs.copy(),
                    attrs={
                        "group_size": dp,
                        "collective": "all_gather",
                        "role": "fsdp_ag",
                    },
                    scope=first_fwd.scope,
                    category="communication",
                )
                ag_fwd.annotations["inserted_by"] = "zero_fsdp_pass"
                ag_fwd.annotations["phase"] = "fwd"
                self._prepend_comm(graph, first_fwd.id, ag_fwd)

            if bwd_nodes:
                first_bwd = bwd_nodes[0]
                last_bwd = bwd_nodes[-1]

                ag_bwd_id = f"comm_fsdp_ag_{first_bwd.id}"
                ag_bwd = OpNode(
                    id=ag_bwd_id,
                    op_type="comm.all_gather",
                    inputs=first_bwd.inputs.copy(),
                    outputs=first_bwd.inputs.copy(),
                    attrs={
                        "group_size": dp,
                        "collective": "all_gather",
                        "role": "fsdp_ag",
                    },
                    scope=first_bwd.scope,
                    category="communication",
                )
                ag_bwd.annotations["inserted_by"] = "zero_fsdp_pass"
                ag_bwd.annotations["phase"] = "bwd"
                self._prepend_comm(graph, first_bwd.id, ag_bwd)

                rs_id = f"comm_fsdp_rs_{last_bwd.id}"
                rs = OpNode(
                    id=rs_id,
                    op_type="comm.reduce_scatter",
                    inputs=last_bwd.outputs.copy(),
                    outputs=last_bwd.outputs.copy(),
                    attrs={
                        "group_size": dp,
                        "collective": "reduce_scatter",
                        "role": "fsdp_rs",
                    },
                    scope=last_bwd.scope,
                    category="communication",
                )
                rs.annotations["inserted_by"] = "zero_fsdp_pass"
                rs.annotations["phase"] = "bwd"
                self._rewire(graph, last_bwd.id, rs)

    def _get_layers(self, graph: OpGraph) -> list[str]:
        """Get layer scopes from the graph.
        
        Args:
            graph: OpGraph to analyze
            
        Returns:
            List of layer scopes
        """
        layers = set()
        for node in graph.nodes.values():
            scope = node.scope
            # Extract layer scope (assuming format like "model.layers.0.self_attn")
            parts = scope.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    layer_scope = ".".join(parts[:i+2])
                    layers.add(layer_scope)
                    break
        return list(layers)

    def _node_in_layer(self, node: OpNode, layer_scope: str) -> bool:
        """Check if a node is in a layer.
        
        Args:
            node: OpNode to check
            layer_scope: Layer scope to check against
            
        Returns:
            True if node is in the layer, False otherwise
        """
        return node.scope.startswith(layer_scope)

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
