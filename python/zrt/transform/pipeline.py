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


def build_pipeline() -> TransformPipeline:
    """Build the unified transform pipeline for both inference and training.

    Pass selection is fully condition-driven:
    - Training passes (Recompute, Optimizer, ZeRO, TrainFlops, TrainingMemory, TrainingPipeline)
      only run when ctx.training is not None.
    - DP / CP passes only run when the corresponding degree > 1 (and training for DP).
    - CommInsert fires when TP, EP, or CP > 1.
    """
    from python.zrt.transform.parallel import (
        TensorParallelPass, ExpertParallelPass, CommInserterPass,
        PipelineParallelPass,
    )
    from python.zrt.transform.parallel.context_parallel import ContextParallelPass
    from python.zrt.transform.parallel.data_parallel import DataParallelPass
    from python.zrt.transform.fusion import FusionPass
    from python.zrt.transform.optim import (
        QuantizationPass, EPLBPass, SharedExpertPass, MTPPass,
    )
    from python.zrt.transform.analysis import (
        FlopsPass, RooflinePass, CommLatencyPass, StreamAssignPass,
        TrainingFlopsPass, TrainingMemoryPass, TrainingPipelinePass,
    )
    from python.zrt.transform.training.zero_fsdp import ZeroFSDPPass
    from python.zrt.transform.training.recompute import RecomputePass
    from python.zrt.transform.training.optimizer import OptimizerPass
    from python.zrt.transform.training.offload import OffloadPass

    is_train = lambda c: c.is_training

    pipe = TransformPipeline()

    # ── Stage 1: Split ────────────────────────────────────────────────────────
    pipe.add("split", DataParallelPass(),
             condition=lambda c: c.parallel.dp > 1 and c.is_training)
    pipe.add("split", TensorParallelPass(),
             condition=lambda c: c.parallel.tp > 1)
    pipe.add("split", ExpertParallelPass(),
             condition=lambda c: c.parallel.ep > 1)
    pipe.add("split", ContextParallelPass(),
             condition=lambda c: c.parallel.cp > 1)
    pipe.add("split", CommInserterPass(),
             condition=lambda c: c.parallel.tp > 1 or c.parallel.ep > 1 or c.parallel.cp > 1)
    pipe.add("split", PipelineParallelPass(),
             condition=lambda c: c.parallel.pp > 1)

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
    # RecomputePass must run before ZeroFSDPPass (ZeRO needs recompute annotations)
    pipe.add("optim", RecomputePass(),        condition=is_train)
    pipe.add("optim", ZeroFSDPPass(),         condition=is_train)
    # OptimizerPass adds optimizer step node after all backward ops
    pipe.add("optim", OptimizerPass(),        condition=is_train)
    # OffloadPass inserts CPU-GPU transfer nodes for memory offloading
    pipe.add("optim", OffloadPass(),
             condition=lambda c: c.is_training and c.training.offload is not None and c.training.offload.pct > 0)

    # ── Stage 4: Analyze ──────────────────────────────────────────────────────
    pipe.add("analyze", FlopsPass())
    pipe.add("analyze", RooflinePass())
    pipe.add("analyze", CommLatencyPass())
    pipe.add("analyze", StreamAssignPass())
    pipe.add("analyze", TrainingFlopsPass(),    condition=is_train)
    pipe.add("analyze", TrainingMemoryPass(),   condition=is_train)
    pipe.add("analyze", TrainingPipelinePass(), condition=is_train)

    return pipe


build_default_pipeline = build_pipeline
