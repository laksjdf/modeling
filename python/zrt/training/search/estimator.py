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
    hfu: float = 0.0
    memory: MemBreakdown | None = None
    per_stage: list = field(default_factory=list)
    total_flops: float = 0.0
    warnings: list[str] = field(default_factory=list)
    config_summary: dict = field(default_factory=dict)
    bubble_fraction: float = 0.0
    schedule_name: str = "1f1b"


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
        hfu=step_result.hfu,
        memory=step_result.memory,
        per_stage=step_result.per_stage,
        total_flops=total_flops,
        warnings=warnings,
        config_summary=config_summary,
        bubble_fraction=step_result.bubble_fraction,
        schedule_name=step_result.schedule_name,
    )


def grid_search(
    model: ModelSpec, system: SystemSpec, space: "SearchSpace",
) -> list[Report]:
    """Grid search over all valid parallel configurations.

    Returns list of Reports sorted by step_time_ms (ascending).
    Invalid configurations (validation errors) are skipped.
    """
    from zrt.training.search.space import SearchSpace

    strategies = space.strategies(system.world_size)
    reports = []

    for strategy in strategies:
        try:
            strategy.validate(model, system)
        except ValueError:
            continue

        try:
            report = estimate(model, system, strategy)
            if report.memory is not None:
                total_gb = report.memory.total / 1e9
                if total_gb > space.max_memory_gb:
                    continue
            reports.append(report)
        except Exception:
            continue

    reports.sort(key=lambda r: r.step_time_ms)
    return reports


def pareto_frontier(reports: list[Report]) -> list[Report]:
    """Extract Pareto frontier (step_time_ms, peak_hbm) with deterministic ordering.

    A config is on the Pareto frontier if no other config has both:
      - lower step_time_ms AND lower peak_hbm

    Deterministic ordering: sort by (step_time_ms, peak_hbm) to ensure
    reproducible frontier construction. When two configs have identical
    step_time and memory, the first one in sorted order is preferred.

    TODO Phase 3: Pruning rules below depend on CP/EP implementation status:
      - no cross-node TP (requires NVLink topology awareness)
      - CP only when seq_len >= 32768 (depends on CP comm cost model)
      - EP only when num_experts > 1 (requires EP dispatch/all-to-all)
      - ZeRO stage requires dp > 1 (already enforced)

    These should become feature flags in SearchSpace once phase 3 provides
    the missing communication and memory semantics.
    """
    if not reports:
        return []

    # Deterministic sort: by step_time, then by memory (asc)
    sorted_reports = sorted(
        reports,
        key=lambda r: (r.step_time_ms, r.memory.total / 1e9 if r.memory else float("inf"))
    )

    frontier = []
    min_memory = float("inf")

    for report in sorted_reports:
        mem_gb = report.memory.total / 1e9 if report.memory else None
        if not frontier:
            frontier.append(report)
            min_memory = mem_gb if mem_gb is not None else float("inf")
        elif mem_gb is not None and mem_gb < min_memory:
            frontier.append(report)
            min_memory = mem_gb

    return frontier
