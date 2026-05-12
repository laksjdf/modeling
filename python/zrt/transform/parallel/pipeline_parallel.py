"""Pipeline Parallel pass: stage assignment + P2P boundary insertion.

For each node annotates ``node.annotations["stage_id"] = int`` (0-indexed).
Inserts ``comm.send_recv`` nodes at stage boundaries to model activation
transfer latency between adjacent pipeline stages.

Layer partitioning strategies:
  - VPP/Interleaved (vpp_chunks > 1, pp_schedule="interleaved"): interleaved assignment
    Each device holds vpp_chunks virtual stages, layers distributed round-robin.
    Example: pp=2, vpp_chunks=2, 8 layers → Device0=[L0,L1,L4,L5], Device1=[L2,L3,L6,L7]
  - DualPipeV (vpp_chunks > 1, pp_schedule="dualpipev"): interleaved assignment (same as VPP)
    Combines VPP layer distribution with DualPipe F/B parallel scheduling.
  - Standard 1F1B (vpp_chunks=1): greedy bin-packing by compute load.
  - DualPipe (pp_schedule="dualpipe"): greedy bin-packing (same as 1F1B)
    Layer assignment identical to 1F1B; F/B parallel scheduling reduces bubble.
  - Explicit: user-provided pp_layer_assignment overrides automatic assignment.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, TYPE_CHECKING

from python.zrt.ir.edge import Edge
from python.zrt.ir.node import OpNode
from python.zrt.ir.types import TensorMeta, DType
from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext

logger = logging.getLogger(__name__)


@dataclass
class LayerGroup:
    """Layers assigned to a single pipeline stage (physical device)."""
    stage_id: int
    layer_ids: List[int] = field(default_factory=list)
    node_ids: Set[str] = field(default_factory=set)
    total_compute_us: float = 0.0
    virtual_stage_ids: List[int] = field(default_factory=list)


class PipelineParallelPass(GraphPass):
    """Annotate pipeline stage IDs and insert P2P comm nodes at boundaries.

    Pass order: runs in the ``split`` stage, after TP/EP but before Fusion.
    Requires ``ctx.parallel.pp > 1`` to do anything.

    Annotations written
    -------------------
    ``node.annotations["stage_id"]`` : int  — 0-indexed pipeline stage (physical device).
    ``node.annotations["virtual_stage_id"]`` : int — virtual stage within device (VPP only).
    """

    name = "pipeline_parallel"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        from python.zrt.ir.graph import OpGraph  # local to avoid circular

        pp = ctx.parallel.pp if ctx.parallel else 1

        if pp <= 1:
            # Annotate everything as stage 0 for consistency with downstream passes
            g = graph.clone()
            for node in g.nodes.values():
                node.annotations.setdefault("stage_id", 0)
            return g

        g = graph.clone()

        pp_layer_assignment: Optional[List[int]] = (
            getattr(ctx.training, "pp_layer_assignment", None)
            if ctx.training else None
        )

        vpp_chunks = max(1, getattr(ctx.training, "vpp_chunks", 1) if ctx.training else 1)
        pp_schedule = getattr(ctx.training, "pp_schedule", "1f1b") if ctx.training else "1f1b"
        is_vpp = vpp_chunks > 1 and pp_schedule in ("interleaved", "i1f1b", "dualpipev")

        # 1. Build layer_id → {node_ids} and per-layer compute load
        layer_nodes: Dict[int, Set[str]] = {}
        layer_load:  Dict[int, float]    = {}

        for node in g.nodes.values():
            try:
                layer_idx = int(node.layer) if node.layer else -1
            except (ValueError, TypeError):
                layer_idx = -1

            layer_nodes.setdefault(layer_idx, set()).add(node.id)

            load = (
                node.annotations.get("compute_us")
                or node.annotations.get("latency_us")
                or node.annotations.get("flops", 0) / 1e12
                or 1.0
            )
            layer_load[layer_idx] = layer_load.get(layer_idx, 0.0) + load

        sorted_layers = sorted(k for k in layer_nodes if k >= 0)

        # 2. Partition layers → pp stages
        stages = self._partition(sorted_layers, layer_nodes, layer_load,
                                 pp, pp_layer_assignment, vpp_chunks, is_vpp)

        # 3. Build lookup: layer_id → stage_id, layer_id → virtual_stage_id (VPP)
        layer_to_stage: Dict[int, int] = {
            lid: s.stage_id
            for s in stages
            for lid in s.layer_ids
        }
        layer_to_virtual_stage: Dict[int, int] = {}
        if is_vpp:
            total_chunks = pp * vpp_chunks
            layers_per_chunk = max(1, len(sorted_layers) // total_chunks)
            for idx, lid in enumerate(sorted_layers):
                layer_to_virtual_stage[lid] = min(idx // layers_per_chunk, total_chunks - 1)

        # 4. Annotate stage_id and virtual_stage_id on every node
        for node in g.nodes.values():
            try:
                layer_idx = int(node.layer) if node.layer else -1
            except (ValueError, TypeError):
                layer_idx = -1
            node.annotations["stage_id"] = layer_to_stage.get(layer_idx, 0)
            if is_vpp and layer_idx >= 0:
                node.annotations["virtual_stage_id"] = layer_to_virtual_stage.get(layer_idx, 0)

        # 5. Insert P2P send_recv at each stage boundary
        self._insert_p2p_nodes(g, stages, is_vpp)

        # 6. Warn if imbalanced
        self._check_balance(stages)

        return g

    # ── partitioning ──────────────────────────────────────────────────────────

    def _partition(
        self,
        sorted_layers: List[int],
        layer_nodes: Dict[int, Set[str]],
        layer_load: Dict[int, float],
        pp: int,
        explicit: Optional[List[int]],
        vpp_chunks: int = 1,
        is_vpp: bool = False,
    ) -> List[LayerGroup]:
        stages = [LayerGroup(stage_id=i) for i in range(pp)]

        if not sorted_layers:
            return stages

        n_layers = len(sorted_layers)

        if explicit and len(explicit) == n_layers:
            for idx, layer_id in enumerate(sorted_layers):
                s_idx = max(0, min(explicit[idx], pp - 1))
                stages[s_idx].layer_ids.append(layer_id)
                stages[s_idx].node_ids.update(layer_nodes[layer_id])
                stages[s_idx].total_compute_us += layer_load.get(layer_id, 0.0)
        elif is_vpp:
            total_chunks = pp * vpp_chunks
            layers_per_chunk = max(1, n_layers // total_chunks)
            for idx, layer_id in enumerate(sorted_layers):
                chunk_id = min(idx // layers_per_chunk, total_chunks - 1)
                s_idx = chunk_id % pp
                stages[s_idx].layer_ids.append(layer_id)
                stages[s_idx].node_ids.update(layer_nodes[layer_id])
                load = layer_load.get(layer_id, 0.0)
                stages[s_idx].total_compute_us += load
            if vpp_chunks > 1:
                logger.info(
                    "PipelineParallelPass: VPP interleaved assignment "
                    "(pp=%d, vpp_chunks=%d, layers_per_chunk=%d, total_chunks=%d)",
                    pp, vpp_chunks, layers_per_chunk, total_chunks,
                )
        else:
            stage_load = [0.0] * pp
            for layer_id in sorted_layers:
                min_s = int(min(range(pp), key=lambda i: stage_load[i]))
                stages[min_s].layer_ids.append(layer_id)
                stages[min_s].node_ids.update(layer_nodes[layer_id])
                load = layer_load.get(layer_id, 0.0)
                stages[min_s].total_compute_us += load
                stage_load[min_s] += load

        return stages

    # ── P2P insertion ─────────────────────────────────────────────────────────

    def _insert_p2p_nodes(self, graph: "OpGraph",
                          stages: List[LayerGroup],
                          is_vpp: bool = False) -> None:
        """Insert comm.send_recv at stage boundaries and rewire receiver edges.

        Detects edges that cross stage boundaries, inserts one comm node per
        crossing edge, and rewires receiver-side edges so the comm node is on
        the real dependency path.

        For VPP, also checks virtual_stage_id boundaries to insert P2P between
        virtual stages on different physical devices.
        """
        node_stage = {nid: node.annotations.get("stage_id", 0)
                      for nid, node in graph.nodes.items()}
        node_virtual_stage = {nid: node.annotations.get("virtual_stage_id", -1)
                              for nid, node in graph.nodes.items()}

        crossing_edges: List[Edge] = []
        for edge in list(graph.edges):
            ss = node_stage.get(edge.src, -1)
            ds = node_stage.get(edge.dst, -1)
            if ss >= 0 and ds >= 0 and ss != ds:
                crossing_edges.append(edge)
            elif is_vpp:
                sv = node_virtual_stage.get(edge.src, -1)
                dv = node_virtual_stage.get(edge.dst, -1)
                if sv >= 0 and dv >= 0 and sv != dv and ss != ds:
                    crossing_edges.append(edge)

        # Dedupe: emit one send_recv per (src_node, src_stage, dst_stage)
        # tuple — a producer with N cross-stage consumers needs only ONE
        # P2P node, with all N consumers rewired through it.  Without this
        # dedupe a single boundary tensor with many consumers (e.g. shared
        # embed/head residual) generates a flood of duplicate send_recvs.
        bucketed: Dict[tuple, List[Edge]] = {}
        for edge in crossing_edges:
            ss = node_stage.get(edge.src, -1)
            ds = node_stage.get(edge.dst, -1)
            sv = node_virtual_stage.get(edge.src, -1)
            dv = node_virtual_stage.get(edge.dst, -1)
            key = (edge.src, edge.src_idx, ss, ds, sv, dv)
            bucketed.setdefault(key, []).append(edge)

        p2p_idx = 0
        for (src_id, src_idx, ss, ds, sv, dv), edges in bucketed.items():
            # Use the first edge as the prototype; all share src_node and
            # src_idx so the carried tensor is identical.
            edge = edges[0]
            ss = node_stage[edge.src]
            ds = node_stage[edge.dst]
            sv = node_virtual_stage.get(edge.src, -1)
            dv = node_virtual_stage.get(edge.dst, -1)
            src_phase = graph.nodes[edge.src].annotations.get("phase", "")
            dst_phase = graph.nodes[edge.dst].annotations.get("phase", "")
            direction = "bwd" if "bwd" in (src_phase, dst_phase) else "fwd"

            tensor = edge.tensor
            act_bytes = tensor.mem_bytes if tensor else 4

            if tensor:
                recv_tensor = TensorMeta(
                    id=f"p2p_{direction}_{ss}_{ds}_{p2p_idx}",
                    shape=tensor.shape, dtype=tensor.dtype,
                    mem_bytes=tensor.mem_bytes,
                )
            else:
                recv_tensor = TensorMeta.from_shape_dtype(
                    f"p2p_{direction}_{ss}_{ds}_{p2p_idx}", (1,), DType.BF16
                )

            p2p_id = f"comm_p2p_{direction}_{ss}_{ds}_{p2p_idx}"
            p2p_idx += 1

            p2p_node = OpNode(
                id=p2p_id,
                op_type="comm.send_recv",
                inputs=[tensor] if tensor else [],
                outputs=[recv_tensor],
                attrs={
                    "src_stage": ss,
                    "dst_stage": ds,
                    "message_size_bytes": act_bytes,
                    "src_virtual_stage": sv if is_vpp and sv >= 0 else None,
                    "dst_virtual_stage": dv if is_vpp and dv >= 0 else None,
                },
                scope=f"pipeline.p2p.{direction}.stage{ss}_to_{ds}",
                category="communication",
            )
            p2p_node.annotations["stage_id"] = ds
            p2p_node.annotations["phase"] = direction
            if is_vpp and dv >= 0:
                p2p_node.annotations["virtual_stage_id"] = dv

            # Insert one comm node after the producer; route every cross-stage
            # consumer of this producer through it.
            graph.insert_after(edge.src, p2p_node, [Edge(
                src=edge.src, src_idx=src_idx,
                dst=p2p_id, dst_idx=0,
                tensor=tensor,
            )])

            for ce in edges:
                if ce in graph.edges:
                    graph.edges.remove(ce)
                graph.edges.append(Edge(
                    src=p2p_id, src_idx=0,
                    dst=ce.dst, dst_idx=ce.dst_idx,
                    tensor=recv_tensor,
                ))
            graph._rebuild_adjacency()

    # ── balance check ─────────────────────────────────────────────────────────

    def _check_balance(self, stages: List[LayerGroup]) -> None:
        loads = [s.total_compute_us for s in stages if s.total_compute_us > 0]
        if len(loads) < 2:
            return
        ratio = max(loads) / min(loads)
        if ratio > 1.5:
            logger.warning(
                "PipelineParallelPass: stage imbalance %.2fx (max/min). "
                "Consider setting --pp-layer-assignment.",
                ratio,
            )
