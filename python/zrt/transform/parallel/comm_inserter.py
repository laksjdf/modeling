"""CommInserterPass: insert communication nodes at parallel split boundaries."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from python.zrt.ir.edge import Edge
from python.zrt.ir.node import OpNode
from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext


def _make_comm_node(node_id: str, collective: str,
                    src_node: OpNode, group_size: int) -> OpNode:
    """Create a comm.* OpNode that wraps src_node's outputs."""
    return OpNode(
        id=node_id,
        op_type=f"comm.{collective}",
        inputs=copy.deepcopy(src_node.outputs),
        outputs=copy.deepcopy(src_node.outputs),  # all_reduce: shape unchanged
        attrs={"group_size": group_size, "collective": collective},
        scope=src_node.scope,
        layer=src_node.layer,
        category="communication",
    )


def _rewire(g: "OpGraph", src_id: str, comm_node: OpNode) -> None:
    """Insert comm_node between src_id and all its current successors.

    Before: src → [s1, s2, ...]
    After:  src → comm → [s1, s2, ...]
    """
    # Collect out-edges of src that we need to reroute
    old_out = [e for e in g.edges if e.src == src_id]

    # Remove those edges from the graph
    g.edges = [e for e in g.edges if e.src != src_id]

    # Add the comm node
    g.nodes[comm_node.id] = comm_node
    g._succ[comm_node.id] = []
    g._pred[comm_node.id] = []

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


class CommInserterPass(GraphPass):
    """Insert comm nodes at positions annotated by TensorParallelPass / ExpertParallelPass.

    TP all_reduce:
        Inserted after every row-parallel linear node
        (those with annotations["tp_split"]["comm_after"] = "all_reduce").

    EP all-to-all:
        For the first expert op in each MoE block, insert a dispatch A2A before it.
        For the last expert op, insert a combine A2A after it.
        (Simplified: annotates boundary nodes; full MoE block detection is heuristic.)
    """

    name = "comm_inserter"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        g = graph.clone()
        self._insert_tp_comm(g, ctx)
        self._insert_ep_comm(g, ctx)
        self._insert_cp_comm(g, ctx)
        return g

    # ── TP: all_reduce after row-parallel linears ─────────────────────────────

    def _insert_tp_comm(self, g: "OpGraph", ctx: "TransformContext") -> None:
        tp = ctx.parallel.tp
        if tp <= 1:
            return

        # Iterate over a snapshot; we'll mutate g.nodes during the loop
        tp_nodes = [
            n for n in list(g.topo_sort())
            if n.annotations.get("tp_split", {}).get("comm_after") == "all_reduce"
        ]
        for i, node in enumerate(tp_nodes):
            comm_id = f"comm_allreduce_{node.id}"
            if comm_id in g.nodes:
                continue
            comm_node = _make_comm_node(comm_id, "all_reduce", node, tp)
            comm_node.annotations["inserted_by"] = "tp_pass"
            _rewire(g, node.id, comm_node)

    # ── EP: all-to-all dispatch/combine around expert blocks ──────────────────

    def _insert_ep_comm(self, g: "OpGraph", ctx: "TransformContext") -> None:
        ep = ctx.parallel.ep
        if ep <= 1:
            return

        # Find nodes that need A2A
        ep_nodes = [
            n for n in g.topo_sort()
            if n.annotations.get("ep_needs_a2a")
               and not n.annotations.get("ep_a2a_inserted")
        ]
        if not ep_nodes:
            return

        # Message-size parameters from graph metadata and training config
        seq_len = g.metadata.get("seq_len", 2048)
        hidden = g.metadata.get("hidden", 4096)
        dtype_bytes = 2  # BF16
        micro_batch = ctx.training.micro_batch if ctx.training else 1
        topk = ctx.profile.moe_active if ctx.profile else 8

        # EP dispatch/combine: A2A distributes routed tokens across EP ranks
        # Each rank sends/receives 1/EP of the total routed data
        # Message size = batch * seq_len * hidden * topk / EP * dtype_bytes
        ep_msg_bytes = micro_batch * seq_len * hidden * topk * dtype_bytes // ep

        # Import TensorMeta for creating correct tensor shapes
        from python.zrt.ir.types import TensorMeta, DType

        # Create tensor metadata with correct message size
        # Shape: [batch, seq_len, hidden] for dispatch/combine
        dispatch_tensor = TensorMeta.from_shape_dtype(
            "ep_dispatch_hidden",
            shape=(micro_batch, seq_len, hidden),
            dtype=DType.BF16,
        )
        combine_tensor = TensorMeta.from_shape_dtype(
            "ep_combine_hidden",
            shape=(micro_batch, seq_len, hidden),
            dtype=DType.BF16,
        )

        # Group by scope prefix (everything up to "experts")
        # Insert one dispatch A2A before the first expert node in the block
        # and one combine A2A after the last expert node in the block.
        # Simple heuristic: treat all ep_nodes as one block per scope root.
        processed_scopes: set[str] = set()
        for node in ep_nodes:
            scope_root = _moe_scope_root(node.scope)
            if scope_root in processed_scopes:
                continue
            processed_scopes.add(scope_root)

            # Nodes in this scope
            block = [n for n in ep_nodes if _moe_scope_root(n.scope) == scope_root]
            first, last = block[0], block[-1]

            # dispatch A2A: insert before first expert node (as a predecessor)
            dispatch_id = f"comm_a2a_dispatch_{first.id}"
            combine_id  = f"comm_a2a_combine_{last.id}"

            if dispatch_id not in g.nodes:
                dispatch = OpNode(
                    id=dispatch_id,
                    op_type="comm.all_to_all",
                    inputs=[dispatch_tensor],
                    outputs=[dispatch_tensor],
                    attrs={"group_size": ep, "collective": "all_to_all",
                           "role": "dispatch", "msg_bytes": ep_msg_bytes},
                    scope=first.scope,
                    layer=first.layer,
                    category="communication",
                )
                dispatch.annotations["inserted_by"] = "ep_pass"
                g.add_node(dispatch)
                # Rewire: in-edges of first → dispatch → first
                _prepend_comm(g, first.id, dispatch)

            if combine_id not in g.nodes:
                combine = OpNode(
                    id=combine_id,
                    op_type="comm.all_to_all",
                    inputs=[combine_tensor],
                    outputs=[combine_tensor],
                    attrs={"group_size": ep, "collective": "all_to_all",
                           "role": "combine", "msg_bytes": ep_msg_bytes},
                    scope=last.scope,
                    layer=last.layer,
                    category="communication",
                )
                combine.annotations["inserted_by"] = "ep_pass"
                g.add_node(combine)
                _rewire(g, last.id, combine)

            for n in block:
                n.annotations["ep_a2a_inserted"] = True

    # ── CP: all-to-all for Ulysses, P2P for Ring ─────────────────────────────

    def _insert_cp_comm(self, g: "OpGraph", ctx: "TransformContext") -> None:
        cp = ctx.parallel.cp
        if cp <= 1:
            return

        # Message-size parameters from graph metadata and training config
        seq_len = g.metadata.get("seq_len", 2048)
        hidden = g.metadata.get("hidden", 4096)
        dtype_bytes = 2  # BF16
        micro_batch = ctx.training.micro_batch if ctx.training else 1

        # Ulysses A2A: each rank gets seq_len/cp tokens, full hidden dim
        ulysses_msg_bytes = micro_batch * (seq_len // cp) * hidden * dtype_bytes
        # Ring P2P: each round sends a KV chunk of seq_len/cp tokens
        ring_msg_bytes = micro_batch * (seq_len // cp) * hidden * dtype_bytes

        # Iterate over nodes with CP split annotations
        cp_nodes = [
            n for n in list(g.topo_sort())
            if n.annotations.get("cp_split")
        ]

        for node in cp_nodes:
            cp_split = node.annotations.get("cp_split", {})
            cp_kind = cp_split.get("kind", "none")

            if cp_kind in ("ulysses", "hybrid"):
                # Ulysses: A2A scatter-seq/gather-heads BEFORE attn; inverse A2A AFTER
                pre_comm_id = f"comm_a2a_cp_pre_{node.id}"
                post_comm_id = f"comm_a2a_cp_post_{node.id}"
                role_prefix = "cp_ulysses" if cp_kind == "ulysses" else "cp_hybrid_ulysses"

                if pre_comm_id not in g.nodes:
                    pre_comm = OpNode(
                        id=pre_comm_id,
                        op_type="comm.all_to_all",
                        inputs=copy.deepcopy(node.inputs),
                        outputs=copy.deepcopy(node.inputs),
                        attrs={"group_size": cp, "collective": "all_to_all",
                               "role": f"{role_prefix}_pre",
                               "message_size_bytes": ulysses_msg_bytes,
                               "msg_bytes": ulysses_msg_bytes},
                        scope=node.scope,
                        layer=node.layer,
                        category="communication",
                    )
                    pre_comm.annotations["inserted_by"] = "cp_pass"
                    g.nodes[pre_comm.id] = pre_comm
                    _prepend_comm(g, node.id, pre_comm)

                if post_comm_id not in g.nodes:
                    post_comm = OpNode(
                        id=post_comm_id,
                        op_type="comm.all_to_all",
                        inputs=copy.deepcopy(node.outputs),
                        outputs=copy.deepcopy(node.outputs),
                        attrs={"group_size": cp, "collective": "all_to_all",
                               "role": f"{role_prefix}_post",
                               "message_size_bytes": ulysses_msg_bytes,
                               "msg_bytes": ulysses_msg_bytes},
                        scope=node.scope,
                        layer=node.layer,
                        category="communication",
                    )
                    post_comm.annotations["inserted_by"] = "cp_pass"
                    _rewire(g, node.id, post_comm)

            if cp_kind in ("ring", "hybrid"):
                p2p_rounds = cp_split.get("p2p_rounds", cp)
                for i in range(p2p_rounds):
                    p2p_prefix = "ring" if cp_kind == "ring" else "hybrid_ring"
                    p2p_id = f"comm_p2p_cp_{p2p_prefix}_{node.id}_round_{i}"
                    if p2p_id not in g.nodes:
                        p2p_comm = OpNode(
                            id=p2p_id,
                            op_type="comm.send_recv",
                            inputs=copy.deepcopy(node.inputs),
                            outputs=copy.deepcopy(node.inputs),
                            attrs={"group_size": cp, "collective": "send_recv",
                                   "role": f"cp_{p2p_prefix}", "round": i,
                                   "message_size_bytes": ring_msg_bytes,
                                   "msg_bytes": ring_msg_bytes,
                                   "cp_rounds": p2p_rounds,
                                   "scope": node.scope, "layer": node.layer},
                            scope=node.scope,
                            layer=node.layer,
                            category="communication",
                        )
                        p2p_comm.annotations["inserted_by"] = "cp_pass"
                        p2p_comm.annotations["overlap_target"] = f"fa_tile:{node.id}"
                        g.nodes[p2p_comm.id] = p2p_comm
                        _prepend_comm(g, node.id, p2p_comm)

            if cp_kind == "compressed":
                stage1_id = f"comm_p2p_cp_compressed_stage1_{node.id}"
                stage2_id = f"comm_ag_cp_compressed_stage2_{node.id}"
                if stage1_id not in g.nodes:
                    stage1 = OpNode(
                        id=stage1_id,
                        op_type="comm.send_recv",
                        inputs=copy.deepcopy(node.inputs),
                        outputs=copy.deepcopy(node.inputs),
                        attrs={"group_size": cp, "collective": "send_recv",
                               "role": "cp_compressed_stage1",
                               "message_size_bytes": ring_msg_bytes,
                               "msg_bytes": ring_msg_bytes,
                               "scope": node.scope, "layer": node.layer},
                        scope=node.scope,
                        layer=node.layer,
                        category="communication",
                    )
                    stage1.annotations["inserted_by"] = "cp_pass"
                    stage1.annotations["overlap_target"] = f"fa_tile:{node.id}"
                    g.nodes[stage1.id] = stage1
                    _prepend_comm(g, node.id, stage1)

                if stage2_id not in g.nodes:
                    stage2 = OpNode(
                        id=stage2_id,
                        op_type="comm.all_gather",
                        inputs=copy.deepcopy(node.inputs),
                        outputs=copy.deepcopy(node.inputs),
                        attrs={"group_size": cp, "collective": "all_gather",
                               "role": "cp_compressed_stage2",
                               "message_size_bytes": ulysses_msg_bytes,
                               "msg_bytes": ulysses_msg_bytes},
                        scope=node.scope,
                        layer=node.layer,
                        category="communication",
                    )
                    stage2.annotations["inserted_by"] = "cp_pass"
                    g.nodes[stage2.id] = stage2
                    _prepend_comm(g, node.id, stage2)


def _moe_scope_root(scope: str) -> str:
    """Return the scope prefix up to (not including) the expert index."""
    for kw in ("experts.", "expert_"):
        idx = scope.lower().find(kw)
        if idx >= 0:
            return scope[:idx]
    return scope


def _prepend_comm(g: "OpGraph", dst_id: str, comm_node: OpNode) -> None:
    """Insert comm_node between all predecessors of dst_id and dst_id itself."""
    in_edges = [e for e in g.edges if e.dst == dst_id]
    g.edges = [e for e in g.edges if e.dst != dst_id]

    # predecessors → comm
    for e in in_edges:
        g.edges.append(Edge(
            src=e.src, src_idx=e.src_idx,
            dst=comm_node.id, dst_idx=e.dst_idx,
            tensor=e.tensor,
        ))

    # comm → dst: preserve original dst_idx so input slots are not remapped
    for e in in_edges:
        g.edges.append(Edge(
            src=comm_node.id, src_idx=e.dst_idx,
            dst=dst_id, dst_idx=e.dst_idx,
            tensor=e.tensor,
        ))

    g._rebuild_adjacency()
