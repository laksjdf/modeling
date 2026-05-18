"""CommInserterPass: insert communication nodes at parallel split boundaries.

Context Parallel (CP) communication insertion uses **block boundary** strategy:
- Group CP nodes by layer (forward and backward separately)
- Each group (layer block) only inserts ONE set of communication nodes
- Communication nodes positioned at block entry/exit (not per-op)

Reference: docs/superpowers/specs/2025-05-12-cp-shape-split-design.md (Lines 362-697)
"""
from __future__ import annotations

import copy
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List

from python.zrt.ir.edge import Edge
from python.zrt.ir.node import OpNode
from python.zrt.ir.types import TensorMeta, DType
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
    old_out = [e for e in g.edges if e.src == src_id]
    g.edges = [e for e in g.edges if e.src != src_id]

    g.nodes[comm_node.id] = comm_node
    g._succ[comm_node.id] = []
    g._pred[comm_node.id] = []

    src_node = g.nodes[src_id]
    for i, out_tensor in enumerate(src_node.outputs):
        g.edges.append(Edge(
            src=src_id, src_idx=i,
            dst=comm_node.id, dst_idx=i,
            tensor=out_tensor,
        ))

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

    CP block boundary:
        Group by layer, insert communication at block boundaries (not per-op).
        Supports Ulysses, Ring, Hybrid, Compressed strategies.
    """

    name = "comm_inserter"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        g = graph.clone()
        self._insert_tp_comm(g, ctx)
        self._insert_ep_comm(g, ctx)
        self._insert_cp_comm(g, ctx)
        return g

    def _insert_tp_comm(self, g: "OpGraph", ctx: "TransformContext") -> None:
        tp = ctx.parallel.tp
        if tp <= 1:
            return

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

    def _insert_ep_comm(self, g: "OpGraph", ctx: "TransformContext") -> None:
        ep = ctx.parallel.ep
        if ep <= 1:
            return

        ep_nodes = [
            n for n in g.topo_sort()
            if n.annotations.get("ep_needs_a2a")
               and not n.annotations.get("ep_a2a_inserted")
        ]
        if not ep_nodes:
            return

        seq_len = g.metadata.get("seq_len", 2048)
        hidden = g.metadata.get("hidden", 4096)
        dtype_bytes = 2
        micro_batch = ctx.training.micro_batch if ctx.training else 1
        topk = ctx.profile.moe_active if ctx.profile else 8

        ep_msg_bytes = micro_batch * seq_len * hidden * topk * dtype_bytes // ep

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

        processed_scopes: set[str] = set()
        for node in ep_nodes:
            scope_root = _moe_scope_root(node.scope)
            if scope_root in processed_scopes:
                continue
            processed_scopes.add(scope_root)

            block = [n for n in ep_nodes if _moe_scope_root(n.scope) == scope_root]
            first, last = block[0], block[-1]

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

    def _insert_cp_comm(self, g: "OpGraph", ctx: "TransformContext") -> None:
        """插入CP通信算子，按layer分组，每组只插入一组通信（block boundary策略）。

        Design principles:
        1. Collect all cp_split nodes
        2. Group by layer (区分forward和backward)
        3. Each group only inserts communication at first/last node
        4. Topological sort within each group to identify entry/exit

        """
        cp = ctx.parallel.cp
        if cp <= 1:
            return

        seq_len = ctx.training.seq_len if ctx.training else 2048
        hidden = ctx.training.hidden if ctx.training else 7168
        micro_batch = ctx.training.micro_batch if ctx.training else 1
        
        # Resolve cp_kind based on model type
        if ctx.training:
            cp_kind_resolved = ctx.training.resolve_cp_kind(ctx.model_id, cp)
        else:
            cp_kind_resolved = "ulysses"

        cp_nodes = [n for n in g.topo_sort() if n.annotations.get("cp_split")]

        if not cp_nodes:
            return

        topo_order = {node.id: i for i, node in enumerate(g.topo_sort())}

        layer_fwd_groups: Dict[str, List[OpNode]] = defaultdict(list)
        layer_bwd_groups: Dict[str, List[OpNode]] = defaultdict(list)

        for node in cp_nodes:
            layer = node.layer or "root"
            phase = node.annotations.get("phase", "fwd")
            if phase == "bwd":
                layer_bwd_groups[f"bwd_{layer}"].append(node)
            else:
                layer_fwd_groups[layer].append(node)

        for layer, nodes in layer_fwd_groups.items():
            if not nodes:
                continue

            sorted_nodes = sorted(nodes, key=lambda n: topo_order[n.id])
            first_node = sorted_nodes[0]
            last_node = sorted_nodes[-1]

            cp_kind = first_node.annotations.get("cp_split", {}).get("kind", "ulysses")

            if cp_kind == "ulysses":
                pre_comm, post_comm = self._create_ulysses_comm_nodes(
                    first_node, last_node, seq_len, cp, hidden, micro_batch, phase="fwd"
                )
                _prepend_comm(g, first_node.id, pre_comm)
                self._insert_after(g, last_node, post_comm)

            elif cp_kind == "ring":
                ring_comm = self._create_ring_comm_node(
                    first_node, seq_len, cp, hidden, micro_batch, phase="fwd"
                )
                _prepend_comm(g, first_node.id, ring_comm)

            elif cp_kind == "hybrid":
                a2a_pre, a2a_post, p2p = self._create_hybrid_comm_nodes(
                    first_node, seq_len, cp, hidden, micro_batch, phase="fwd"
                )
                _prepend_comm(g, first_node.id, a2a_pre)
                self._insert_after(g, last_node, p2p)
                self._insert_after(g, p2p, a2a_post)

            elif cp_kind == "compressed":
                self._insert_compressed_cp_comm_block(
                    g, first_node, last_node, seq_len, cp, hidden, micro_batch, phase="fwd"
                )

        for layer, nodes in layer_bwd_groups.items():
            if not nodes:
                continue

            sorted_nodes = sorted(nodes, key=lambda n: topo_order[n.id])
            first_node = sorted_nodes[0]
            last_node = sorted_nodes[-1]

            cp_kind = first_node.annotations.get("cp_split", {}).get("kind", "ulysses")

            if cp_kind == "ulysses":
                pre_comm, post_comm = self._create_ulysses_comm_nodes(
                    first_node, last_node, seq_len, cp, hidden, micro_batch, phase="bwd"
                )
                _prepend_comm(g, first_node.id, pre_comm)
                self._insert_after(g, last_node, post_comm)

            elif cp_kind == "ring":
                ring_comm = self._create_ring_comm_node(
                    first_node, seq_len, cp, hidden, micro_batch, phase="bwd"
                )
                _prepend_comm(g, first_node.id, ring_comm)

            elif cp_kind == "hybrid":
                a2a_pre, a2a_post, p2p = self._create_hybrid_comm_nodes(
                    first_node, seq_len, cp, hidden, micro_batch, phase="bwd"
                )
                _prepend_comm(g, first_node.id, a2a_pre)
                self._insert_after(g, last_node, p2p)
                self._insert_after(g, p2p, a2a_post)

            elif cp_kind == "compressed":
                self._insert_compressed_cp_comm_block(
                    g, first_node, last_node, seq_len, cp, hidden, micro_batch, phase="bwd"
                )

    def _create_ulysses_comm_nodes(
        self, first_node: OpNode, last_node: OpNode,
        seq_len: int, cp: int, hidden: int, micro_batch: int, phase: str
    ) -> tuple[OpNode, OpNode]:
        """创建Ulysses pre/post A2A通信节点。"""
        seq_local = seq_len // cp
        layer = first_node.layer or "root"
        phase_suffix = phase

        if first_node.inputs:
            pre_in_shape = first_node.inputs[0].shape
        else:
            pre_in_shape = (micro_batch, seq_local, hidden)

        pre_out_shape = pre_in_shape
        a2a_bytes = micro_batch * seq_local * hidden * 2

        pre_comm = OpNode(
            id=f"comm_a2a_cp_pre_layer_{layer}_{phase_suffix}",
            op_type="comm.all_to_all",
            inputs=[TensorMeta.from_shape_dtype("pre_in", pre_in_shape, DType.BF16)],
            outputs=[TensorMeta.from_shape_dtype("pre_out", pre_out_shape, DType.BF16)],
            attrs={
                "group_size": cp,
                "collective": "all_to_all",
                "role": "cp_ulysses_pre",
                "bytes": a2a_bytes,
                "layer": layer,
            },
            annotations={
                "phase": phase,
                "inserted_by": "cp_pass",
                "mask": False,  # A2A is blocking, cannot overlap
                "mask_type": "a2a_blocking",
            },
            scope=first_node.scope,
            layer=layer,
            category="communication",
        )

        post_comm = OpNode(
            id=f"comm_a2a_cp_post_layer_{layer}_{phase_suffix}",
            op_type="comm.all_to_all",
            inputs=[TensorMeta.from_shape_dtype("post_in", pre_in_shape, DType.BF16)],
            outputs=[TensorMeta.from_shape_dtype("post_out", pre_in_shape, DType.BF16)],
            attrs={
                "group_size": cp,
                "collective": "all_to_all",
                "role": "cp_ulysses_post",
                "bytes": a2a_bytes,
                "layer": layer,
            },
            annotations={
                "phase": phase,
                "inserted_by": "cp_pass",
                "mask": False,  # A2A is blocking, cannot overlap
                "mask_type": "a2a_blocking",
            },
            scope=last_node.scope,
            layer=layer,
            category="communication",
        )

        return pre_comm, post_comm

    def _create_ring_comm_node(
        self, first_node: OpNode, seq_len: int, cp: int, hidden: int, micro_batch: int, phase: str
    ) -> OpNode:
        """创建Ring CP的P2P通信节点。"""
        seq_local = seq_len // cp
        p2p_bytes = micro_batch * seq_local * hidden * 2
        layer = first_node.layer or "root"
        phase_suffix = phase

        if first_node.inputs:
            ring_shape = first_node.inputs[0].shape
        else:
            ring_shape = (micro_batch, seq_local, hidden)

        ring_comm = OpNode(
            id=f"comm_p2p_ring_layer_{layer}_{phase_suffix}",
            op_type="comm.send_recv",
            inputs=[TensorMeta.from_shape_dtype("ring_in", ring_shape, DType.BF16)],
            outputs=[TensorMeta.from_shape_dtype("ring_out", ring_shape, DType.BF16)],
            attrs={
                "group_size": cp,
                "collective": "send_recv",
                "role": "cp_ring",
                "bytes": p2p_bytes,
                "rounds": cp,
                "overlap": True,
                "layer": layer,
            },
            annotations={
                "overlap_target": "attention_block",
                "phase": phase,
                "inserted_by": "cp_pass",
                "mask": True,
                "mask_type": "p2p_overlap",
            },
            scope=first_node.scope,
            layer=layer,
            category="communication",
        )

        return ring_comm

    def _create_hybrid_comm_nodes(
        self, first_node: OpNode, seq_len: int, cp: int, hidden: int, micro_batch: int, phase: str
    ) -> tuple[OpNode, OpNode, OpNode]:
        """创建Hybrid CP通信节点组（A2A pre + P2P + A2A post）。"""
        seq_local = seq_len // cp
        a2a_bytes = micro_batch * seq_local * hidden * 2
        p2p_bytes = micro_batch * seq_local * hidden * 2
        layer = first_node.layer or "root"
        phase_suffix = phase

        if first_node.inputs:
            shape = first_node.inputs[0].shape
        else:
            shape = (micro_batch, seq_local, hidden)

        a2a_pre = OpNode(
            id=f"comm_a2a_hybrid_pre_layer_{layer}_{phase_suffix}",
            op_type="comm.all_to_all",
            inputs=[TensorMeta.from_shape_dtype("a2a_in", shape, DType.BF16)],
            outputs=[TensorMeta.from_shape_dtype("a2a_out", shape, DType.BF16)],
            attrs={
                "group_size": cp,
                "collective": "all_to_all",
                "role": "cp_hybrid_a2a_pre",
                "bytes": a2a_bytes,
                "layer": layer,
            },
            annotations={
                "phase": phase,
                "inserted_by": "cp_pass",
                "mask": False,  # A2A is blocking, cannot overlap
                "mask_type": "a2a_blocking",
            },
            scope=first_node.scope,
            layer=layer,
            category="communication",
        )

        p2p = OpNode(
            id=f"comm_p2p_hybrid_layer_{layer}_{phase_suffix}",
            op_type="comm.send_recv",
            inputs=[TensorMeta.from_shape_dtype("p2p_in", shape, DType.BF16)],
            outputs=[TensorMeta.from_shape_dtype("p2p_out", shape, DType.BF16)],
            attrs={
                "group_size": cp,
                "collective": "send_recv",
                "role": "cp_hybrid_p2p",
                "bytes": p2p_bytes,
                "rounds": cp,
                "overlap": True,
                "layer": layer,
            },
            annotations={
                "overlap_target": "attention_block",
                "phase": phase,
                "inserted_by": "cp_pass",
                "mask": True,
                "mask_type": "p2p_overlap",
            },
            scope=first_node.scope,
            layer=layer,
            category="communication",
        )

        a2a_post = OpNode(
            id=f"comm_a2a_hybrid_post_layer_{layer}_{phase_suffix}",
            op_type="comm.all_to_all",
            inputs=[TensorMeta.from_shape_dtype("a2a_post_in", shape, DType.BF16)],
            outputs=[TensorMeta.from_shape_dtype("a2a_post_out", shape, DType.BF16)],
            attrs={
                "group_size": cp,
                "collective": "all_to_all",
                "role": "cp_hybrid_a2a_post",
                "bytes": a2a_bytes,
                "layer": layer,
            },
            annotations={
                "phase": phase,
                "inserted_by": "cp_pass",
                "mask": False,  # A2A is blocking, cannot overlap
                "mask_type": "a2a_blocking",
            },
            scope=first_node.scope,
            layer=layer,
            category="communication",
        )

        return a2a_pre, a2a_post, p2p

    def _insert_compressed_cp_comm_block(
        self, g: "OpGraph", first_node: OpNode, last_node: OpNode,
        seq_len: int, cp: int, hidden: int, micro_batch: int, phase: str
    ) -> None:
        """DeepSeek-V4两段式Compressed CP，只在block边界插入。"""
        layer = first_node.layer or "root"

        cp_split_info = first_node.annotations.get("cp_split", {})
        if cp_split_info.get("skip_comm", False):
            return

        if "swa" in layer.lower() or "sliding" in first_node.scope.lower():
            return

        seq_local = seq_len // cp
        phase_suffix = phase

        if first_node.inputs:
            shape = first_node.inputs[0].shape
        else:
            shape = (micro_batch, seq_local, hidden)

        compression_ratio = 4

        p2p_bytes = micro_batch * seq_local * hidden * 2
        p2p_comm = OpNode(
            id=f"comm_p2p_compressed_stage1_layer_{layer}_{phase_suffix}",
            op_type="comm.send_recv",
            inputs=[TensorMeta.from_shape_dtype("p2p_in", shape, DType.BF16)],
            outputs=[TensorMeta.from_shape_dtype("p2p_out", shape, DType.BF16)],
            attrs={
                "group_size": cp,
                "collective": "send_recv",
                "role": "cp_compressed_stage1",
                "stage": "p2p_boundary_exchange",
                "bytes": p2p_bytes,
                "layer": layer,
            },
            annotations={
                "phase": phase,
                "inserted_by": "cp_pass",
                "mask": True,
                "mask_type": "p2p_overlap",
            },
            scope=first_node.scope,
            layer=layer,
            category="communication",
        )
        _prepend_comm(g, first_node.id, p2p_comm)

        # Stage 2: AllGather compressed KV (after block)
        ag_bytes = p2p_bytes // compression_ratio
        ag_comm = OpNode(
            id=f"comm_ag_compressed_stage2_layer_{layer}_{phase_suffix}",
            op_type="comm.all_gather",
            inputs=[TensorMeta.from_shape_dtype("ag_in", shape, DType.BF16)],
            outputs=[TensorMeta.from_shape_dtype("ag_out", shape, DType.BF16)],
            attrs={
                "group_size": cp,
                "collective": "all_gather",
                "role": "cp_compressed_stage2",
                "stage": "allgather_compressed_kv",
                "bytes": ag_bytes,
                "compression_ratio": compression_ratio,
                "layer": layer,
            },
            annotations={
                "phase": phase,
                "inserted_by": "cp_pass",
                "mask": False,  # AllGather is blocking, cannot overlap
                "mask_type": "ag_blocking",
            },
            scope=last_node.scope,
            layer=layer,
            category="communication",
        )
        self._insert_after(g, last_node, ag_comm)

    def _insert_after(self, graph: "OpGraph", src_node: OpNode, comm_node: OpNode) -> None:
        """Insert comm_node between src_node and all its current successors."""
        src_id = src_node.id

        old_out = [e for e in graph.edges if e.src == src_id]
        if not old_out:
            graph.nodes[comm_node.id] = comm_node
            graph._succ[comm_node.id] = []
            graph._pred[comm_node.id] = []
            if src_node.outputs:
                graph.edges.append(Edge(
                    src=src_id, src_idx=0,
                    dst=comm_node.id, dst_idx=0,
                    tensor=src_node.outputs[0],
                ))
            graph._rebuild_adjacency()
            return

        graph.edges = [e for e in graph.edges if e.src != src_id]

        graph.nodes[comm_node.id] = comm_node
        graph._succ[comm_node.id] = []
        graph._pred[comm_node.id] = []

        for i, out_tensor in enumerate(src_node.outputs):
            graph.edges.append(Edge(
                src=src_id, src_idx=i,
                dst=comm_node.id, dst_idx=i,
                tensor=out_tensor,
            ))

        for e in old_out:
            graph.edges.append(Edge(
                src=comm_node.id, src_idx=e.src_idx,
                dst=e.dst, dst_idx=e.dst_idx,
                tensor=e.tensor,
            ))

        graph._rebuild_adjacency()


def _moe_scope_root(scope: str) -> str:
    """Return the scope prefix up to (not including) the expert index."""
    for kw in ("experts.", "expert_"):
        idx = scope.lower().find(kw)
        if idx >= 0:
            return scope[:idx]
    return scope


def _prepend_comm(g: "OpGraph", dst_id: str, comm_node: OpNode) -> None:
    """Insert comm_node between all predecessors of dst_id and dst_id itself.
    
    Handles two cases:
    1. dst has predecessors: reroute preds → comm → dst
    2. dst has no predecessors: add comm as new graph entry node
    """
    # Add comm node to graph first
    g.nodes[comm_node.id] = comm_node
    g._succ[comm_node.id] = []
    g._pred[comm_node.id] = []
    
    in_edges = [e for e in g.edges if e.dst == dst_id]
    g.edges = [e for e in g.edges if e.dst != dst_id]

    if not in_edges:
        # No predecessors: add comm as entry node, create edge comm → dst
        dst_node = g.nodes[dst_id]
        if dst_node.inputs:
            # Use first input tensor for the edge
            g.edges.append(Edge(
                src=comm_node.id, src_idx=0,
                dst=dst_id, dst_idx=0,
                tensor=dst_node.inputs[0],
            ))
    else:
        # Has predecessors: reroute
        for e in in_edges:
            g.edges.append(Edge(
                src=e.src, src_idx=e.src_idx,
                dst=comm_node.id, dst_idx=e.dst_idx,
                tensor=e.tensor,
            ))

        for e in in_edges:
            g.edges.append(Edge(
                src=comm_node.id, src_idx=e.dst_idx,
                dst=dst_id, dst_idx=e.dst_idx,
                tensor=e.tensor,
            ))

    g._rebuild_adjacency()