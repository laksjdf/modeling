from __future__ import annotations

import logging
import math
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.ir.edge import Edge
from python.zrt.ir.types import TensorMeta, DType
from python.zrt.ir.param_count import count_params
from python.zrt.transform.base import GraphPass
from python.zrt.transform.context import TransformContext
from python.zrt.training.models.optimizer import (
    adam_step_flops,
    muon_optimizer_step_flops,
    muon_flops_from_geometry,
)
from zrt.training.models.memory import routed_expert_params

logger = logging.getLogger(__name__)


class OptimizerPass(GraphPass):
    """OptimizerPass for optimizer step annotation.

    Adds an ``optimizer_step`` node to the backward graph to represent
    the optimizer update cost. This node is annotated with:
      - state_bytes: Optimizer state memory per rank
      - step_flops: Optimizer step FLOPs

    Runs only on backward-phase graphs (when metadata["phase"] indicates
    backward or when individual nodes have backward phase annotations).
    """
    name = "optimizer"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        """Run OptimizerPass on the graph.

        Args:
            graph: Input OpGraph
            ctx: TransformContext with training config

        Returns:
            New OpGraph with optimizer step node
        """
        g = graph.clone()
        if not ctx.training or not ctx.is_training:
            return g

        # Check graph-level phase (for unified stitched graphs)
        graph_phase = g.metadata.get("phase", "")

        # For stitched graphs (phase="train"), optimizer runs on the unified graph
        # For separate backward graphs (phase="train_backward" or "backward"), optimizer runs on backward
        if graph_phase and graph_phase not in ("train", "train_backward", "backward"):
            return g

        # For non-stitched graphs, check if any node is a backward node
        has_backward = any(
            n.annotations.get("phase") in ("bwd", "backward", "train_backward")
            for n in g.nodes.values()
        )
        if not has_backward and not graph_phase:
            return g

        # Calculate total parameters on rank using graph parameter count.
        # If total_params is not in metadata (no explicit override), apply layer_scale
        # from num_layers/num_layers_traced so state_bytes reflects the full model.
        total_params = count_params(g)
        if g.metadata.get("total_params", 0) == 0:
            _nl = g.metadata.get("num_layers", 0)
            _nlt = g.metadata.get("num_layers_traced", _nl)
            if _nlt > 0 and _nl != _nlt:
                total_params = int(total_params * (_nl / _nlt))
        tp = ctx.parallel.tp if ctx.parallel else 1
        dp = ctx.parallel.dp if ctx.parallel else 1
        pp = ctx.parallel.pp if ctx.parallel else 1
        cp = getattr(ctx.parallel, "cp", 1) if ctx.parallel else 1

        # Apply PP sharding: optimizer runs in the last stage, so only 1/pp of parameters
        if pp > 1:
            params = int(total_params / pp)
        else:
            params = total_params

        # Apply TP/DP sharding
        zero_stage = ctx.training.zero_stage if ctx.training else 0
        if zero_stage >= 3:
            params //= (tp * dp)
        else:
            params //= tp

        opt = ctx.training.optimizer if ctx.training else "adam"
        muon_fraction = ctx.training.muon_param_fraction if ctx.training else None
        muon_rotation = getattr(ctx.training, "muon_rotation", True) if ctx.training else True
        dp = ctx.parallel.dp if ctx.parallel else 1

        # Optimizer runs in the last stage (after all backward ops complete)
        optimizer_stage_id = max(0, pp - 1)

        # Estimate hidden dimension from graph metadata
        hidden = g.metadata.get("hidden", None)
        model_type = g.metadata.get("model_type", None)

        # Muon-specific attributes
        params_muon = 0
        params_adam = params
        ns_steps_resolved = 0
        muon_ag_bytes = 0
        f_muon = 0.85

        if opt == "muon":
            f_muon = muon_fraction if muon_fraction is not None else 0.85
            params_muon = int(params * f_muon)
            params_adam = params - params_muon
            # Resolve NS steps using priority chain (per §5.1.1 of muon_optimizer_design.md)
            if ctx.training:
                ns_steps_resolved = ctx.training.effective_ns_steps(model_type)
            else:
                ns_steps_resolved = 5
            # Muon AG bytes: pre-DP payload (collective formula scales by dp internally)
            if dp > 1:
                muon_params_pre_dp = params_muon * dp if zero_stage >= 3 else params_muon
                muon_ag_bytes = int(muon_params_pre_dp * 4)

        # Compute routed/non-routed split for calibrated Muon comm model.
        # Comm payloads use pre-DP params: the collective alpha-beta formula
        # already scales by group_size internally (matching Stack A's
        # _params_on_rank_for_dp which explicitly does NOT divide by dp).
        params_for_comm = params * dp if (zero_stage >= 3 and dp > 1) else params
        P_routed = 0
        muon_ag_bytes_non_routed = 0
        muon_ag_bytes_routed = 0
        expert_dp = dp
        if opt == "muon" and dp > 1:
            ep = ctx.parallel.ep if ctx.parallel else 1
            if ep > 1:
                meta = g.metadata
                n_moe = (meta.get("layer_type_counts") or {}).get("moe", 0)
                h_meta = meta.get("hidden", ctx.training.hidden if ctx.training else hidden)
                moe_ffn_meta = meta.get("moe_ffn_hidden") or 0
                num_exp = meta.get("moe_total_experts") or 0
                n_layers_meta = n_moe + (meta.get("layer_type_counts") or {}).get("dense", 0) + (meta.get("layer_type_counts") or {}).get("mtp", 0)
                P_routed = routed_expert_params(
                    n_moe=n_moe, hidden=h_meta or 0, moe_ffn=moe_ffn_meta,
                    num_experts=num_exp, tp=tp, ep=ep, pp=pp, n_layers=max(n_layers_meta, 1),
                )
                expert_dp = max(1, dp // ep)
        P_non_routed = max(0, params_for_comm - P_routed)
        if opt == "muon" and dp > 1:
            muon_ag_bytes_non_routed = int(P_non_routed * f_muon * 4)
            muon_ag_bytes_routed = int(P_routed * f_muon * 4)

        # Compute step FLOPs: arch-driven path when geometry metadata available
        step_flops_val = self._opt_step_flops(opt, params, ns_steps_resolved, muon_fraction, hidden)
        if opt == "muon":
            meta = g.metadata
            layer_counts = meta.get("layer_type_counts") or {}
            ffn_val = meta.get("ffn_hidden") or 0
            moe_ffn_val = meta.get("moe_ffn_hidden") or 0
            h_val = hidden or meta.get("hidden") or 0
            if h_val > 0 and layer_counts and (ffn_val > 0 or moe_ffn_val > 0):
                ep = ctx.parallel.ep if ctx.parallel else 1
                step_flops_val = muon_flops_from_geometry(
                    hidden=h_val, ffn=ffn_val, moe_ffn=moe_ffn_val,
                    n_dense=layer_counts.get("dense", 0),
                    n_moe=layer_counts.get("moe", 0),
                    n_mtp=layer_counts.get("mtp", 0),
                    num_experts=meta.get("moe_total_experts") or 0,
                    n_shared_experts=meta.get("n_shared_experts") or 1,
                    tp=tp, ep=ep, pp=pp, dp=dp,
                    zero_stage=zero_stage,
                    K=ns_steps_resolved, muon_fraction=f_muon,
                )

        # Create optimizer step node
        step_node = OpNode(
            id="optimizer_step",
            op_type=f"optimizer.{opt}",
            inputs=[],
            outputs=[],
            attrs={
                "optimizer": opt,
                "params_total": params,
                "params_muon": params_muon,
                "params_adam": params_adam,
                "state_bytes": self._opt_state_bytes(opt, params, muon_fraction=muon_fraction),
                "step_flops": step_flops_val,
                "ns_steps": ns_steps_resolved,
                "ns_rotation": muon_rotation if opt == "muon" else False,
                "muon_ag_bytes": muon_ag_bytes,
                "muon_ag_bytes_non_routed": muon_ag_bytes_non_routed,
                "muon_ag_bytes_routed": muon_ag_bytes_routed,
                "expert_dp": expert_dp,
            },
            scope="optimizer.step",
            category="compute",
        )
        step_node.annotations["phase"] = "bwd"
        step_node.annotations["stage_id"] = optimizer_stage_id
        step_node.annotations["optimizer_step"] = True

        # Build optimizer chain: [AG] -> optimizer_step -> [RS]
        # AG: AllGather momentum before optimizer step (Muon + ZeRO-1 + DP>1)
        # RS: ReduceScatter gradient after optimizer step (rotation=True)
        optimizer_chain: list[OpNode] = []

        if opt == "muon" and dp > 1 and muon_ag_bytes > 0:
            ag_node = OpNode(
                id="muon_ag",
                op_type="comm.all_gather",
                inputs=[],
                outputs=[],
                attrs={
                    "bytes": muon_ag_bytes,
                    "group_size": dp,
                    "collective": "all_gather",
                    "optimizer": "muon",
                },
                scope="optimizer.comm",
                category="communication",
            )
            ag_node.annotations["phase"] = "bwd"
            ag_node.annotations["stage_id"] = optimizer_stage_id
            ag_node.annotations["muon_comm"] = "ag"
            optimizer_chain.append(ag_node)

        optimizer_chain.append(step_node)

        if opt == "muon" and dp > 1 and muon_ag_bytes > 0 and muon_rotation:
            rs_node = OpNode(
                id="muon_rs",
                op_type="comm.reduce_scatter",
                inputs=[],
                outputs=[],
                attrs={
                    "bytes": muon_ag_bytes,
                    "group_size": dp,
                    "collective": "reduce_scatter",
                    "optimizer": "muon",
                },
                scope="optimizer.comm",
                category="communication",
            )
            rs_node.annotations["phase"] = "bwd"
            rs_node.annotations["stage_id"] = optimizer_stage_id
            rs_node.annotations["muon_comm"] = "rs"
            optimizer_chain.append(rs_node)

        # Append the optimizer chain at the end of the graph
        self._append_chain_at_end(g, optimizer_chain)

        return g

    def _opt_state_bytes(self, optimizer: str, params: int, master_bytes: int = 4, muon_fraction: float | None = None) -> int:
        """Calculate optimizer state bytes.

        Args:
            optimizer: Optimizer name
            params: Number of parameters
            master_bytes: Bytes per parameter in master dtype (default FP32=4)
            muon_fraction: Fraction of params using Muon (default 0.85)

        Returns:
            Optimizer state bytes.
            - Adam: master copy + momentum (m) + variance (v) = 12B/param
            - Muon: P_muon × 8B + P_adam × 12B = P × (12 - f_muon × 4)
        """
        if optimizer in ("adam", "adamw"):
            return params * master_bytes * 3
        elif optimizer == "muon":
            f_muon = muon_fraction if muon_fraction is not None else 0.85
            return int(params * (12 - f_muon * 4))
        else:
            return params * master_bytes * 3

    def _opt_step_flops(self, optimizer: str, params: int, muon_ns_steps: int | None = None, muon_fraction: float | None = None, hidden: int | None = None) -> int:
        """Calculate optimizer step FLOPs.

        Args:
            optimizer: Optimizer name
            params: Number of parameters
            muon_ns_steps: Newton-Schulz iterations for Muon
            muon_fraction: Fraction of params using Muon
            hidden: Model hidden dimension (estimated from params if None)

        Returns:
            Optimizer step FLOPs
            - Adam: 16 FLOPs per parameter
            - Muon: K × 4 × hidden × hidden² + other ops (mixed with Adam)
        """
        if optimizer in ("adam", "adamw"):
            return adam_step_flops(params)
        elif optimizer == "muon":
            K = muon_ns_steps if muon_ns_steps is not None else 5
            f_muon = muon_fraction if muon_fraction is not None else 0.85
            if hidden is None:
                hidden = int(math.sqrt(params / 100)) if params > 0 else 128
            return muon_optimizer_step_flops(params, K, hidden, f_muon)
        else:
            return params * 16

    def _append_at_end(self, graph: OpGraph, new_node: OpNode) -> None:
        """Append a node at the end of the graph.

        Finds all sink nodes (no outgoing edges) and connects them
        to the new node, then adds the new node to the graph.

        Args:
            graph: OpGraph to modify
            new_node: OpNode to append
        """
        self._append_chain_at_end(graph, [new_node])

    def _append_chain_at_end(self, graph: OpGraph, chain: list[OpNode]) -> None:
        """Append a chain of nodes at the end of the graph.

        Finds all sink nodes (no outgoing edges) and connects them
        to the first node in the chain, then connects chain nodes
        sequentially.

        Args:
            graph: OpGraph to modify
            chain: List of OpNodes to append in order
        """
        if not chain:
            return

        # Build adjacency to find sink nodes
        has_out_edge = set()
        for edge in graph.edges:
            has_out_edge.add(edge.src)

        sink_nodes = [
            graph.nodes[nid] for nid in graph.nodes
            if nid not in has_out_edge
        ]

        # Add all chain nodes to the graph
        for node in chain:
            graph.nodes[node.id] = node
            if node.id not in graph._pred:
                graph._pred[node.id] = []
            if node.id not in graph._succ:
                graph._succ[node.id] = []

        # Connect all sink nodes to the first chain node
        first_node = chain[0]
        for sink_node in sink_nodes:
            if sink_node.outputs:
                graph.edges.append(Edge(
                    src=sink_node.id,
                    src_idx=0,
                    dst=first_node.id,
                    dst_idx=0,
                    tensor=sink_node.outputs[0],
                ))
                graph._succ[sink_node.id].append(first_node.id)
                graph._pred[first_node.id].append(sink_node.id)

        # Connect chain nodes sequentially
        for i in range(len(chain) - 1):
            src_node = chain[i]
            dst_node = chain[i + 1]
            graph.edges.append(Edge(
                src=src_node.id,
                src_idx=0,
                dst=dst_node.id,
                dst_idx=0,
                tensor=TensorMeta.from_shape_dtype(
                    f"opt_chain_{i}",
                    shape=(1,),
                    dtype=DType.BF16,
                ),
            ))
            graph._succ[src_node.id].append(dst_node.id)
            graph._pred[dst_node.id].append(src_node.id)
