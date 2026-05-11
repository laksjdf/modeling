from __future__ import annotations

import logging
import math
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.ir.edge import Edge
from python.zrt.ir.param_count import count_params
from python.zrt.transform.base import GraphPass
from python.zrt.transform.context import TransformContext
from python.zrt.training.models.optimizer import (
    adam_step_flops,
    muon_optimizer_step_flops,
)

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

        # For stitched graphs, only add optimizer to backward
        if graph_phase and graph_phase not in ("train_backward", "backward"):
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

        if opt == "muon":
            f_muon = muon_fraction if muon_fraction is not None else 0.85
            params_muon = int(params * f_muon)
            params_adam = params - params_muon
            # Resolve NS steps using priority chain (per §5.1.1 of muon_optimizer_design.md)
            if ctx.training:
                ns_steps_resolved = ctx.training.effective_ns_steps(model_type)
            else:
                ns_steps_resolved = 5
            # Muon AG bytes: total bytes to gather = P_muon × 4B
            # Ring factor applied in timing calculation, not pre-scaled here
            if dp > 1:
                muon_ag_bytes = int(params_muon * 4)

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
                "step_flops": self._opt_step_flops(opt, params, ns_steps_resolved, muon_fraction, hidden),
                "ns_steps": ns_steps_resolved,
                "ns_rotation": muon_rotation if opt == "muon" else False,
                "muon_ag_bytes": muon_ag_bytes,
            },
            scope="optimizer.step",
            category="compute",
        )
        step_node.annotations["phase"] = "bwd"
        step_node.annotations["stage_id"] = optimizer_stage_id
        step_node.annotations["optimizer_step"] = True

        # Append the optimizer step node at the end of the graph
        self._append_at_end(g, step_node)

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
        # Build adjacency to find sink nodes
        has_out_edge = set()
        for edge in graph.edges:
            has_out_edge.add(edge.src)

        sink_nodes = [
            graph.nodes[nid] for nid in graph.nodes
            if nid not in has_out_edge
        ]

        # Add the new node first
        graph.nodes[new_node.id] = new_node
        if new_node.id not in graph._pred:
            graph._pred[new_node.id] = []
        if new_node.id not in graph._succ:
            graph._succ[new_node.id] = []

        # Connect all sink nodes to the new node
        for sink_node in sink_nodes:
            if sink_node.outputs:
                graph.edges.append(Edge(
                    src=sink_node.id,
                    src_idx=0,
                    dst=new_node.id,
                    dst_idx=0,
                    tensor=sink_node.outputs[0],
                ))
                # Update adjacency structures
                graph._succ[sink_node.id].append(new_node.id)
                graph._pred[new_node.id].append(sink_node.id)
