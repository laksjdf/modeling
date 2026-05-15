"""Expert Parallel pass: annotate MoE expert ops for EP splitting."""
from __future__ import annotations

from typing import TYPE_CHECKING

from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext


# Scope substrings that identify individual expert computation
_EXPERT_KEYWORDS = ("experts.", "expert_", ".experts[", "moe_ffn")


def _is_expert_scope(scope: str) -> bool:
    s = scope.lower()
    if "shared_expert" in s:
        return False
    return any(k in s for k in _EXPERT_KEYWORDS)


class ExpertParallelPass(GraphPass):
    """Annotate MoE expert nodes with EP metadata.

    Sets:
      annotations["ep_experts_local"] = num_experts_per_rank
      annotations["ep_needs_a2a"]     = True  (needs all-to-all dispatch/combine)
    """

    name = "expert_parallel"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        ep = ctx.parallel.ep
        if ep <= 1:
            return graph

        profile = ctx.profile
        num_experts = getattr(profile, "num_experts", None) if profile else None
        if not num_experts or num_experts <= 1:
            return graph

        experts_per_rank = max(1, num_experts // ep)
        g = graph.clone()

        for node in g.nodes.values():
            if _is_expert_scope(node.scope):
                node.annotations["ep_experts_local"] = experts_per_rank
                node.annotations["ep_needs_a2a"] = True

        return g
