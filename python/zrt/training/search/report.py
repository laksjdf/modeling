"""Report formatters — JSON, dict, summary string."""

from __future__ import annotations

import json
from pathlib import Path

from zrt.training.search.estimator import Report


def report_to_dict(report: Report) -> dict:
    """Convert Report to a JSON-serializable dict."""
    d = {
        "step_time_ms": round(report.step_time_ms, 3),
        "pipeline_time_ms": round(report.pipeline_time_ms, 3),
        "mfu": round(report.mfu, 4),
        "hfu": round(report.hfu, 4),
        "total_flops": report.total_flops,
        "warnings": report.warnings,
        "config_summary": report.config_summary,
        "schedule_name": report.schedule_name,
        "bubble_fraction": round(report.bubble_fraction, 4),
        "warmup_ms": round(report.warmup_ms, 3),
        "steady_ms": round(report.steady_ms, 3),
        "cooldown_ms": round(report.cooldown_ms, 3),
        "dp_exposed_ms": round(report.dp_exposed_ms, 3),
        "optimizer_time_ms": round(report.optimizer_time_ms, 3),
        "optimizer_comm_ms": round(report.optimizer_comm_ms, 3),
        "warmup_fwd_ms": round(report.warmup_fwd_ms, 3),
        "warmup_bwd_ms": round(report.warmup_bwd_ms, 3),
        "steady_fwd_ms": round(report.steady_fwd_ms, 3),
        "steady_bwd_ms": round(report.steady_bwd_ms, 3),
        "cooldown_fwd_ms": round(report.cooldown_fwd_ms, 3),
        "cooldown_bwd_ms": round(report.cooldown_bwd_ms, 3),
        "steady_fwd_per_mb_ms": round(report.steady_fwd_per_mb_ms, 3),
        "steady_bwd_per_mb_ms": round(report.steady_bwd_per_mb_ms, 3),
        "steady_per_mb_ms": round(report.steady_per_mb_ms, 3),
        "compute_time_ms": round(report.compute_time_ms, 3),
        "fwd_compute_ms": round(report.fwd_compute_ms, 3),
        "bwd_compute_ms": round(report.bwd_compute_ms, 3),
        "exposed_comm_ms": round(report.exposed_comm_ms, 3),
        "tp_exposed_ms": round(report.tp_exposed_ms, 3),
        "cp_exposed_ms": round(report.cp_exposed_ms, 3),
        "ep_exposed_ms": round(report.ep_exposed_ms, 3),
        "pp_exposed_ms": round(report.pp_exposed_ms, 3),
        "hidden_comm_ms": round(report.hidden_comm_ms, 3),
        "dp_hidden_ms": round(report.dp_hidden_ms, 3),
        "tp_hidden_ms": round(report.tp_hidden_ms, 3),
        "ep_hidden_ms": round(report.ep_hidden_ms, 3),
        "total_comm_volume_ms": round(report.total_comm_volume_ms, 3),
        "tokens_per_sec": round(report.tokens_per_sec, 1),
        "effective_params": report.effective_params,
        "flops_per_token": report.flops_per_token,
    }

    if report.memory is not None:
        d["memory"] = report.memory.to_gb()

    if report.per_stage:
        stages = []
        for i, st in enumerate(report.per_stage):
            stages.append({
                "stage": i,
                "fwd_ms": round(st.fwd * 1000, 3),
                "bwd_ms": round(st.bwd * 1000, 3),
                "total_ms": round((st.fwd + st.bwd) * 1000, 3),
            })
        d["per_stage"] = stages

    return d


def report_to_json(report: Report, path: str | Path) -> None:
    """Write Report to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(report_to_dict(report), f, indent=2, default=str)


def report_summary(report: Report) -> str:
    """Human-readable single-config summary string."""
    lines = []
    lines.append("=" * 60)
    lines.append("Training Estimation Report")
    lines.append("=" * 60)

    cs = report.config_summary
    lines.append(f"  Model:    {cs.get('model', 'N/A')}")
    lines.append(f"  System:   {cs.get('system', 'N/A')}")
    lines.append(f"  Strategy: {cs.get('strategy', 'N/A')}")
    lines.append(f"  Microbatches: {cs.get('num_microbatches', 'N/A')}")
    lines.append("")

    lines.append(f"  Step time:  {report.step_time_ms:.1f} ms")
    lines.append(f"  Schedule:   {report.schedule_name}")
    lines.append(f"  Bubble:     {report.bubble_fraction:.1%}")
    lines.append(f"  MFU:        {report.mfu:.1%}")
    lines.append(f"  HFU:        {report.hfu:.1%}")

    if report.flops_per_token > 0:
        lines.append(f"  FLOPs/token: {report.flops_per_token:.2e}")
    if report.tokens_per_sec > 0:
        lines.append(f"  Tokens/s:    {report.tokens_per_sec:.0f}")

    # ── Step Time Breakdown Table ────────────────────────────────────
    st = report.step_time_ms
    if st > 0:
        lines.append("")
        lines.append("  Step Time Breakdown:")
        lines.append(f"  {'Component':<38s} {'Time (ms)':>10s} {'%':>8s}")
        lines.append(f"  {'─' * 38} {'─' * 10} {'─' * 8}")

        # Core compute components
        lines.append(f"  {'Forward Compute':<38s} {report.fwd_compute_ms:>10.2f} {report.fwd_compute_ms/st*100:>7.1f}%")
        lines.append(f"  {'Backward Compute':<38s} {report.bwd_compute_ms:>10.2f} {report.bwd_compute_ms/st*100:>7.1f}%")

        # Exposed communication
        lines.append(f"  {'Communication (exposed)':<38s} {report.exposed_comm_ms:>10.2f} {report.exposed_comm_ms/st*100:>7.1f}%")
        if report.tp_exposed_ms > 0:
            lines.append(f"    {'TP (RS/AG)':<36s} {report.tp_exposed_ms:>10.2f} {report.tp_exposed_ms/st*100:>7.1f}%")
        if report.cp_exposed_ms > 0:
            lines.append(f"    {'CP (A2A)':<36s} {report.cp_exposed_ms:>10.2f} {report.cp_exposed_ms/st*100:>7.1f}%")
        if report.ep_exposed_ms > 0:
            lines.append(f"    {'EP (A2A)':<36s} {report.ep_exposed_ms:>10.2f} {report.ep_exposed_ms/st*100:>7.1f}%")
        if report.pp_exposed_ms > 0:
            lines.append(f"    {'PP (P2P)':<36s} {report.pp_exposed_ms:>10.2f} {report.pp_exposed_ms/st*100:>7.1f}%")
        if report.dp_exposed_ms > 0:
            lines.append(f"    {'DP (AR/RS)':<36s} {report.dp_exposed_ms:>10.2f} {report.dp_exposed_ms/st*100:>7.1f}%")

        # Optimizer
        if report.optimizer_time_ms > 0:
            lines.append(f"  {'Optimizer (compute)':<38s} {report.optimizer_time_ms:>10.2f} {report.optimizer_time_ms/st*100:>7.1f}%")
        if report.optimizer_comm_ms > 0:
            lines.append(f"  {'Optimizer (comm)':<38s} {report.optimizer_comm_ms:>10.2f} {report.optimizer_comm_ms/st*100:>7.1f}%")

        lines.append(f"  {'─' * 38} {'─' * 10} {'─' * 8}")
        lines.append(f"  {'TOTAL STEP TIME':<38s} {st:>10.2f} {100.0:>7.1f}%")

        # Hidden communication (not on critical path)
        if report.hidden_comm_ms > 0:
            lines.append("")
            lines.append(f"  Communication Hidden (overlapped with compute):")
            lines.append(f"    {'DP hidden':<36s} {report.dp_hidden_ms:>10.2f} ms")
            lines.append(f"    {'TP hidden':<36s} {report.tp_hidden_ms:>10.2f} ms")
            lines.append(f"    {'EP hidden':<36s} {report.ep_hidden_ms:>10.2f} ms")
            lines.append(f"    {'Total hidden':<36s} {report.hidden_comm_ms:>10.2f} ms")

    # ── Phase Breakdown ─────────────────────────────────────────────
    if report.warmup_ms > 0 or report.steady_ms > 0:
        lines.append("")
        lines.append("  Phase Breakdown:")
        lines.append(f"  {'Phase':<14s} {'FWD (ms)':>10s} {'BWD (ms)':>10s} {'Total (ms)':>10s}")
        lines.append(f"  {'─' * 14} {'─' * 10} {'─' * 10} {'─' * 10}")
        lines.append(f"  {'Warmup':<14s} {report.warmup_fwd_ms:>10.2f} {report.warmup_bwd_ms:>10.2f} {report.warmup_ms:>10.2f}")
        lines.append(f"  {'Steady':<14s} {report.steady_fwd_ms:>10.2f} {report.steady_bwd_ms:>10.2f} {report.steady_ms:>10.2f}")
        lines.append(f"  {'Cooldown':<14s} {report.cooldown_fwd_ms:>10.2f} {report.cooldown_bwd_ms:>10.2f} {report.cooldown_ms:>10.2f}")

        total_fwd = report.warmup_fwd_ms + report.steady_fwd_ms + report.cooldown_fwd_ms
        total_bwd = report.warmup_bwd_ms + report.steady_bwd_ms + report.cooldown_bwd_ms
        total_phase = report.warmup_ms + report.steady_ms + report.cooldown_ms
        lines.append(f"  {'─' * 14} {'─' * 10} {'─' * 10} {'─' * 10}")
        lines.append(f"  {'TOTAL':<14s} {total_fwd:>10.2f} {total_bwd:>10.2f} {total_phase:>10.2f}")

        if report.steady_per_mb_ms > 0:
            mb_count = int(report.steady_ms / report.steady_per_mb_ms) if report.steady_per_mb_ms > 0 else 0
            lines.append(f"  Per-microbatch (steady): FWD={report.steady_fwd_per_mb_ms:.2f}ms  BWD={report.steady_bwd_per_mb_ms:.2f}ms  Total={report.steady_per_mb_ms:.2f}ms  ({mb_count} microbatches)")

    # ── Memory ──────────────────────────────────────────────────────
    if report.memory is not None:
        gb = report.memory.to_gb()
        lines.append("")
        lines.append("  Memory (per GPU):")
        lines.append(f"    weights:     {gb['weights_gb']:.2f} GB")
        lines.append(f"    grads:       {gb['grads_gb']:.2f} GB")
        lines.append(f"    opt_state:   {gb['opt_state_gb']:.2f} GB")
        lines.append(f"    activations: {gb['activations_gb']:.2f} GB")
        lines.append(f"    comm_buf:    {gb['comm_buffers_gb']:.2f} GB")
        lines.append(f"    TOTAL:       {gb['total_gb']:.2f} GB")

    # ── Per-stage times ─────────────────────────────────────────────
    if report.per_stage:
        lines.append("")
        lines.append("  Per-stage times:")
        for i, stg in enumerate(report.per_stage):
            lines.append(
                f"    Stage {i}: fwd={stg.fwd*1000:.2f}ms  bwd={stg.bwd*1000:.2f}ms  "
                f"total={((stg.fwd+stg.bwd)*1000):.2f}ms"
            )

    # ── Warnings ────────────────────────────────────────────────────
    if report.warnings:
        lines.append("")
        lines.append("  Warnings:")
        for w in report.warnings:
            lines.append(f"    - {w}")

    lines.append("=" * 60)
    return "\n".join(lines)