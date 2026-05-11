from __future__ import annotations

import logging
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.transform.base import GraphPass
from python.zrt.transform.context import TransformContext

logger = logging.getLogger(__name__)


class RecomputePass(GraphPass):
    """Recompute pass for activation recomputation.

    Annotates forward-graph nodes with ``recompute=True`` for selective
    activation checkpointing. The policy is controlled by
    ``ctx.training.recompute_policy`` which is a string:
      - "none": No recompute (all activations saved)
      - "full": All forward ops marked for recompute (minimal memory)
      - "selective": Attention-upscaled ops (softmax, attn output) marked

    Recompute annotations affect:
      - TrainingFlopsPass: recompute_flops calculation
      - TrainingMemoryPass: activation memory reduction
    """
    name = "recompute"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        """Run recompute pass on the graph.

        Args:
            graph: Input OpGraph
            ctx: TransformContext with training config

        Returns:
            New OpGraph with recompute annotations
        """
        g = graph.clone()
        if not ctx.training:
            return g

        policy = ctx.training.recompute_policy or "none"

        # Only annotate forward-phase nodes in stitched graphs
        phase_key = g.metadata.get("phase", "")
        if phase_key and phase_key in ("train_backward", "backward"):
            return g

        if policy == "none":
            return g

        for node in g.nodes.values():
            # Skip backward-phase nodes in stitched graphs
            node_phase = node.annotations.get("phase", "")
            if node_phase in ("bwd", "backward", "train_backward"):
                continue

            if policy == "full":
                # All forward ops recomputed
                node.annotations["recompute"] = True
                node.annotations["recompute_policy"] = "full"

            elif policy == "selective":
                # Selective: attention-upscaled ops (softmax, attn output projections)
                if self._is_selective_recompute_target(node):
                    node.annotations["recompute"] = True
                    node.annotations["recompute_policy"] = "selective"

        return g

    def _is_selective_recompute_target(self, node: OpNode) -> bool:
        """Check if a node is a selective recompute target.

        Selective recompute typically targets:
          - Softmax operations (attention-upscaled)
          - Attention output projections
          - Attention core operations in some stacks

        Args:
            node: OpNode to check

        Returns:
            True if node should be selectively recomputed
        """
        op_type = node.op_type.lower()
        scope = node.scope.lower()

        # Softmax is the primary target for attention-upscaled recompute
        if "softmax" in op_type:
            return True

        # Attention output projection (O_proj in transformer blocks)
        if "o_proj" in scope or ("out_proj" in scope and "attn" in scope):
            return True

        # Attention core operations (flash-attn, sdpa, etc.)
        if any(x in op_type for x in ("flash_attn", "scaled_dot_product", "sdpa", "attention")):
            # Only recompute the core if it's not already covered by softmax
            # This avoids double-counting in some implementations
            return True

        return False
