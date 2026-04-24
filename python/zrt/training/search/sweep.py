"""Configuration sweep and Pareto frontier analysis.

Grid search over (tp, cp, pp, ep, dp, zero_stage) with pruning rules:
- No cross-node TP (tp <= gpus_per_node)
- CP only when seq >= 32k
- EP only when num_experts > 1

Output Pareto frontier sorted by (step_time, peak_hbm).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING

from zrt.training.search.estimator import Report, estimate
from zrt.training.spec.strategy import CPKind, PPSched, RecomputePolicy, Strategy

if TYPE_CHECKING:
    from zrt.training.spec.model import ModelSpec
    from zrt.training.spec.system import SystemSpec


@dataclass
class SweepConfig:
    tp_range: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    cp_range: list[int] = field(default_factory=lambda: [1, 2, 4])
    pp_range: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    ep_range: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    dp_range: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    zero_range: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    cp_kind: CPKind = CPKind.ULYSSES
    seq_threshold_for_cp: int = 32768
    gpus_per_node: int = 8


@dataclass
class SweepResult:
    config: Strategy
    report: Report
    is_pareto: bool = False


def sweep(
    model: "ModelSpec",
    system: "SystemSpec",
    cfg: SweepConfig | None = None,
) -> list[SweepResult]:
    """Grid search over parallelism configs.

    Returns all valid configs with reports, sorted by step_time.
    """
    if cfg is None:
        cfg = SweepConfig()

    results: list[SweepResult] = []

    for tp, cp, pp, ep, dp, zero in product(
        cfg.tp_range, cfg.cp_range, cfg.pp_range,
        cfg.ep_range, cfg.dp_range, cfg.zero_range,
    ):
        if not _is_valid_combo(tp, cp, pp, ep, dp, zero, model, system, cfg):
            continue

        strategy = Strategy(
            tp=tp, cp=cp, pp=pp, ep=ep, dp=dp,
            zero_stage=zero,
            cp_kind=cfg.cp_kind if cp > 1 else CPKind.NONE,
        )

        try:
            strategy.validate(model, system)
            report = estimate(model, system, strategy)
            results.append(SweepResult(config=strategy, report=report))
        except ValueError:
            continue

    results.sort(key=lambda r: r.report.step_time_ms)

    pareto_results = _compute_pareto_frontier(results)
    for r in pareto_results:
        r.is_pareto = True

    return results


def _is_valid_combo(
    tp: int, cp: int, pp: int, ep: int, dp: int, zero: int,
    model: "ModelSpec", system: "SystemSpec", cfg: SweepConfig,
) -> bool:
    """Prune invalid or infeasible configurations."""
    total = tp * cp * pp * ep * dp

    if total != system.world_size:
        return False

    if tp > cfg.gpus_per_node:
        return False

    if cp > 1 and model.seq_len < cfg.seq_threshold_for_cp:
        return False

    if ep > 1 and model.num_experts <= 0:
        return False

    if zero >= 1 and dp <= 1:
        return False

    if model.num_heads % tp != 0:
        return False

    if model.num_kv_heads % tp != 0:
        return False

    if model.ffn % tp != 0:
        return False

    if ep > 1 and model.num_experts % ep != 0:
        return False

    if pp > len(model.layers):
        return False

    return True


def _compute_pareto_frontier(results: list[SweepResult]) -> list[SweepResult]:
    """Compute Pareto frontier by (step_time, peak_hbm).

    A config is Pareto-optimal if no other config has both lower step_time
    AND lower peak_hbm.
    """
    if not results:
        return []

    pareto: list[SweepResult] = []

    for r in results:
        step = r.report.step_time_ms
        mem = _peak_hbm_gb(r.report)

        dominated = False
        for p in pareto:
            p_step = p.report.step_time_ms
            p_mem = _peak_hbm_gb(p.report)

            if p_step <= step and p_mem <= mem:
                dominated = True
                break

        if not dominated:
            pareto.append(r)
            pareto = [p for p in pareto
                      if not (step < p.report.step_time_ms and mem < _peak_hbm_gb(p.report))]

    return pareto


def _peak_hbm_gb(report: Report) -> float:
    """Extract peak HBM usage in GB from report."""
    if report.memory is None:
        return 0.0
    gb = report.memory.to_gb()
    return gb.get("total_gb", 0.0)


def pareto_frontier(results: list[SweepResult]) -> list[SweepResult]:
    """Return only Pareto-optimal configs."""
    return [r for r in results if r.is_pareto]


def sweep_summary(results: list[SweepResult]) -> str:
    """Human-readable summary of sweep results."""
    lines = []
    lines.append("=" * 70)
    lines.append("Configuration Sweep Results")
    lines.append("=" * 70)

    pareto_count = sum(1 for r in results if r.is_pareto)
    lines.append(f"Total configs: {len(results)}, Pareto-optimal: {pareto_count}")
    lines.append("")

    if not results:
        lines.append("No valid configurations found.")
        return "\n".join(lines)

    lines.append("Pareto Frontier (sorted by step_time):")
    lines.append("-" * 70)

    for r in results:
        if not r.is_pareto:
            continue

        s = r.config
        rep = r.report
        mem_gb = _peak_hbm_gb(rep)

        lines.append(
            f"  TP={s.tp} CP={s.cp} PP={s.pp} EP={s.ep} DP={s.dp} Z={s.zero_stage}"
            f" | step={rep.step_time_ms:.1f}ms MFU={rep.mfu:.1%}"
            f" | mem={mem_gb:.2f}GB"
        )

    lines.append("")
    lines.append("Top 10 fastest configs:")
    lines.append("-" * 70)

    sorted_results = sorted(results, key=lambda r: r.report.step_time_ms)[:10]
    for r in sorted_results:
        s = r.config
        rep = r.report
        mem_gb = _peak_hbm_gb(rep)
        pareto_mark = "*" if r.is_pareto else ""

        lines.append(
            f"  {pareto_mark}TP={s.tp} CP={s.cp} PP={s.pp} EP={s.ep} DP={s.dp} Z={s.zero_stage}"
            f" | step={rep.step_time_ms:.1f}ms MFU={rep.mfu:.1%}"
            f" | mem={mem_gb:.2f}GB"
        )

    lines.append("=" * 70)
    return "\n".join(lines)