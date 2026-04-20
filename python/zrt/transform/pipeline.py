"""TransformPipeline: 4-stage graph transformation orchestrator.

Stage order (fixed): split → fuse → optim → analyze

Each stage runs its registered passes in order.
A pass can declare a condition; if the condition returns False it is skipped.
"""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext

_STAGES = ("split", "fuse", "optim", "analyze")
_PassEntry = tuple[GraphPass, Callable[["TransformContext"], bool] | None]


class TransformPipeline:
    def __init__(self) -> None:
        self._stages: dict[str, list[_PassEntry]] = {s: [] for s in _STAGES}

    def add(self, stage: str, pass_: GraphPass,
            condition: Callable[["TransformContext"], bool] | None = None) -> None:
        if stage not in self._stages:
            raise ValueError(f"Unknown stage {stage!r}; valid: {_STAGES}")
        self._stages[stage].append((pass_, condition))

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        current = graph
        for stage in _STAGES:
            for pass_, cond in self._stages[stage]:
                if cond is not None and not cond(ctx):
                    continue
                current = pass_.run(current, ctx)
        return current

    def __repr__(self) -> str:
        lines = ["TransformPipeline:"]
        for stage in _STAGES:
            entries = self._stages[stage]
            if entries:
                lines.append(f"  [{stage}]")
                for p, cond in entries:
                    cond_str = f" (if {cond})" if cond else ""
                    lines.append(f"    {p.name}{cond_str}")
        return "\n".join(lines)


def build_default_pipeline() -> TransformPipeline:
    """Build the standard 4-stage pipeline with all default passes registered."""
    from python.zrt.transform.parallel import (
        TensorParallelPass, ExpertParallelPass, CommInserterPass,
    )
    from python.zrt.transform.fusion import FusionPass
    from python.zrt.transform.optim import (
        QuantizationPass, EPLBPass, SharedExpertPass, MTPPass,
    )
    from python.zrt.transform.analysis import FlopsPass, RooflinePass, CommLatencyPass, StreamAssignPass

    pipe = TransformPipeline()

    # ── Stage 1: Split ────────────────────────────────────────────────────────
    pipe.add("split", TensorParallelPass(),
             condition=lambda c: c.parallel.tp > 1)
    pipe.add("split", ExpertParallelPass(),
             condition=lambda c: c.parallel.ep > 1)
    pipe.add("split", CommInserterPass(),
             condition=lambda c: c.parallel.tp > 1 or c.parallel.ep > 1)

    # ── Stage 2: Fuse ─────────────────────────────────────────────────────────
    pipe.add("fuse", FusionPass())

    # ── Stage 3: Optim ────────────────────────────────────────────────────────
    pipe.add("optim", QuantizationPass(),
             condition=lambda c: c.quant is not None)
    pipe.add("optim", EPLBPass(),
             condition=lambda c: "eplb" in c.optim_flags)
    pipe.add("optim", SharedExpertPass(),
             condition=lambda c: "shared_expert_external" in c.optim_flags)
    pipe.add("optim", MTPPass(),
             condition=lambda c: "mtp" in c.optim_flags)

    # ── Stage 4: Analyze ──────────────────────────────────────────────────────
    pipe.add("analyze", FlopsPass())
    pipe.add("analyze", RooflinePass())
    pipe.add("analyze", CommLatencyPass())  # Override comm latency with interconnect BW
    pipe.add("analyze", StreamAssignPass())

    return pipe
