"""Single-point estimator — the main entry point.

Flow: validate → build_graph → op_cost → stage_time → pipeline_step_time → TrainingReport
"""

from __future__ import annotations

from dataclasses import dataclass, field

from zrt.training.compose.schedules import StepResult, pipeline_step_time
from zrt.training.ir.builders import build_graph
from zrt.training.ir.training_graph import Graph as _GraphType
from zrt.training.ir.validate import validate as ir_validate
from zrt.training.models.flops import total_training_flops, forward_backward_flops
from zrt.training.models.memory import MemBreakdown
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.report import TrainingReport
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import SystemSpec


# Legacy alias for backward compatibility
Report = TrainingReport


def estimate(
    model: ModelSpec, system: SystemSpec, strategy: Strategy,
    graph: "_GraphType | None" = None,
) -> TrainingReport:
    """Single-point evaluation of a training config.

    Returns a TrainingReport with step time, MFU, memory, and per-stage breakdown.

    Parameters
    ----------
    graph : optional
        Pre-built IR graph. When provided, skips internal build_graph() call.
        This avoids duplicate work when the caller already built the graph
        (e.g., for op_cost computation in the CLI).
    """
    # Validate
    strategy.validate(model, system)
    warnings = ir_validate(model, system, strategy)

    # Build or reuse IR
    if graph is None:
        graph = build_graph(model, strategy)

    # Total training FLOPs (graph-based, split by phase)
    total_flops = total_training_flops(graph, model, strategy, system)
    fwd_flops, bwd_flops = forward_backward_flops(graph, model, strategy, system)

    # Pipeline step time (includes per-stage timing + memory + MFU via 6P rule)
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

    s = step_result
    # Derived metrics
    tokens = strategy.global_batch * model.seq_len if strategy.global_batch > 0 else strategy.micro_batch * strategy.dp * model.seq_len
    # Use pipeline_time (step_time minus optimizer overhead) for throughput.
    # Note: s.step_time at this point already includes optimizer addition from
    # pipeline_step_time(), so we use s.pipeline_time instead.
    pipeline_time = s.pipeline_time
    tokens_per_sec = tokens / pipeline_time if pipeline_time > 0 else 0.0
    flops_per_token = total_flops / tokens if tokens > 0 else 0.0

    return TrainingReport(
        step_time_ms=s.step_time * 1000,
        pipeline_time_ms=s.pipeline_time * 1000,
        mfu=s.mfu,
        hfu=s.hfu,
        memory=s.memory,
        per_stage=s.per_stage,
        total_flops=total_flops,
        forward_flops=fwd_flops,
        backward_flops=bwd_flops,
        training_flops=total_flops,
        total_params=model.total_params(),
        warnings=warnings,
        config_summary=config_summary,
        bubble_fraction=s.bubble_fraction,
        schedule_name=s.schedule_name,
        warmup_steps=s.warmup_steps,
        cooldown_steps=s.cooldown_steps,
        steady_steps=max(0, strategy.num_microbatches() - s.warmup_steps - s.cooldown_steps),
        warmup_ms=s.warmup * 1000,
        steady_ms=s.steady * 1000,
        cooldown_ms=s.cooldown * 1000,
        dp_exposed_ms=s.dp_exposed * 1000,
        optimizer_time_ms=s.optimizer_time * 1000,
        optimizer_comm_ms=s.optimizer_comm * 1000,
        warmup_fwd_ms=s.warmup_fwd * 1000,
        warmup_bwd_ms=s.warmup_bwd * 1000,
        steady_fwd_ms=s.steady_fwd * 1000,
        steady_bwd_ms=s.steady_bwd * 1000,
        cooldown_fwd_ms=s.cooldown_fwd * 1000,
        cooldown_bwd_ms=s.cooldown_bwd * 1000,
        steady_fwd_per_mb_ms=s.steady_fwd_per_mb * 1000,
        steady_bwd_per_mb_ms=s.steady_bwd_per_mb * 1000,
        steady_per_mb_ms=s.steady_per_mb * 1000,
        compute_time_ms=s.compute_time * 1000,
        fwd_compute_ms=s.fwd_compute * 1000,
        bwd_compute_ms=s.bwd_compute * 1000,
        exposed_comm_ms=s.exposed_comm * 1000,
        tp_exposed_ms=s.tp_exposed * 1000,
        cp_exposed_ms=s.cp_exposed * 1000,
        ep_exposed_ms=s.ep_exposed * 1000,
        pp_exposed_ms=s.pp_exposed * 1000,
        hidden_comm_ms=s.hidden_comm * 1000,
        dp_hidden_ms=s.dp_hidden * 1000,
        tp_hidden_ms=s.tp_hidden * 1000,
        ep_hidden_ms=s.ep_hidden * 1000,
        total_comm_volume_ms=s.total_comm_volume * 1000,
        # Derived metrics
        tokens_per_sec=tokens_per_sec,
        effective_params=model.effective_params_for_flops(),
        flops_per_token=flops_per_token,
    )


def grid_search(
    model: ModelSpec, system: SystemSpec, space: "SearchSpace",
) -> list[TrainingReport]:
    """Grid search over all valid parallel configurations.

    Returns list of TrainingReports sorted by step_time_ms (ascending).
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


def pareto_frontier(reports: list[TrainingReport]) -> list[TrainingReport]:
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
