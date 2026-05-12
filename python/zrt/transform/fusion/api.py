"""FusionPass: graph transform that applies MRO-based operator fusion."""
from __future__ import annotations

from typing import TYPE_CHECKING

from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext


class FusionPass(GraphPass):
    """Apply MRO-based fusion to an OpGraph.

    Rules are loaded from ``registry/platforms/`` based on the model
    classes found in the OpGraph's node metadata.
    """

    name = "fusion"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        from python.zrt.transform.fusion.loading import initialize_rules
        from python.zrt.transform.fusion.loading.fusion_config import (
            resolve_fusion_config,
        )
        from python.zrt.transform.fusion.pipeline.fuser import MultiPassFuser
        from python.zrt.transform.fusion.registry import default_registry

        initialize_rules(getattr(ctx, "model_id", "") or "")

        # Resolve fusion config from disk only when the caller hasn't
        # explicitly set one — CLI / tests that pre-populate ctx.fusion
        # win over auto-discovery.
        existing = getattr(ctx, "fusion", None)
        is_default = (
            existing is None
            or (
                existing.enabled_rules is None
                and not existing.disabled_rules
                and not existing.allow_structural_collapse
                and not existing.merge_sibling_classes
            )
        )
        if is_default:
            phase = ctx.phase_for_fusion() if ctx is not None else "inference"
            ctx.fusion = resolve_fusion_config(
                getattr(ctx, "model_id", "") or "", phase, explicit_path=None,
            )

        fuser = MultiPassFuser(registry=default_registry())
        return fuser.fuse(graph, ctx)
