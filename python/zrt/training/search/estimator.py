"""Single-point estimator — the main entry point.

Flow: validate → build_graph → op_cost → stage_time → pipeline_step_time → Report
"""

from __future__ import annotations

from dataclasses import dataclass, field

from zrt.training.compose.pipeline import StepResult, pipeline_step_time
from zrt.training.ir.builders import build_graph
from zrt.training.ir.validate import validate as ir_validate
from zrt.training.models.flops import total_training_flops
from zrt.training.models.memory import MemBreakdown
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import SystemSpec


@dataclass
class Report:
    step_time_ms: float = 0.0
    mfu: float = 0.0
    memory: MemBreakdown | None = None
    per_stage: list = field(default_factory=list)
    total_flops: float = 0.0
    warnings: list[str] = field(default_factory=list)
    config_summary: dict = field(default_factory=dict)


def estimate(
    model: ModelSpec, system: SystemSpec, strategy: Strategy,
) -> Report:
    """Single-point evaluation of a training config.

    Returns a Report with step time, MFU, memory, and per-stage breakdown.
    """
    # Validate
    strategy.validate(model, system)
    warnings = ir_validate(model, system, strategy)

    # Build IR
    graph = build_graph(model, strategy)

    # Total training FLOPs
    total_flops = total_training_flops(graph, model, strategy)

    # Pipeline step time (includes per-stage timing + memory + MFU)
    step_result: StepResult = pipeline_step_time(graph, model, system, strategy)

    # Config summary
    config_summary = {
        "model": f"hidden={model.hidden}, layers={len(model.layers)}, heads={model.num_heads}",
        "system": f"{system.gpu.name} x {system.world_size}",
        "strategy": f"TP={strategy.tp} CP={strategy.cp} PP={strategy.pp} EP={strategy.ep} DP={strategy.dp}",
        "parallelism": f"TP*CP*PP*EP*DP = {strategy.tp * strategy.cp * strategy.pp * strategy.ep * strategy.dp}",
        "micro_batch": strategy.micro_batch,
        "global_batch": strategy.global_batch,
        "num_microbatches": strategy.num_microbatches(),
        "zero_stage": strategy.zero_stage,
    }

    return Report(
        step_time_ms=step_result.step_time * 1000,  # convert to ms
        mfu=step_result.mfu,
        memory=step_result.memory,
        per_stage=step_result.per_stage,
        total_flops=total_flops,
        warnings=warnings,
        config_summary=config_summary,
    )
