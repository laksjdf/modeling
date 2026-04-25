"""Report formatters — JSON, dict, summary string."""

from __future__ import annotations

import json
from pathlib import Path

from zrt.training.search.estimator import Report


def report_to_dict(report: Report) -> dict:
    """Convert Report to a JSON-serializable dict."""
    d = {
        "step_time_ms": round(report.step_time_ms, 3),
        "mfu": round(report.mfu, 4),
        "hfu": round(report.hfu, 4),
        "total_flops": report.total_flops,
        "warnings": report.warnings,
        "config_summary": report.config_summary,
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
    lines.append(f"  MFU:        {report.mfu:.1%}")
    lines.append(f"  HFU:        {report.hfu:.1%}")

    if report.memory is not None:
        gb = report.memory.to_gb()
        lines.append(f"  Memory:")
        lines.append(f"    weights:     {gb['weights_gb']:.2f} GB")
        lines.append(f"    grads:       {gb['grads_gb']:.2f} GB")
        lines.append(f"    opt_state:   {gb['opt_state_gb']:.2f} GB")
        lines.append(f"    activations: {gb['activations_gb']:.2f} GB")
        lines.append(f"    comm_buf:    {gb['comm_buffers_gb']:.2f} GB")
        lines.append(f"    TOTAL:       {gb['total_gb']:.2f} GB")

    if report.per_stage:
        lines.append(f"  Per-stage times:")
        for i, st in enumerate(report.per_stage):
            lines.append(
                f"    Stage {i}: fwd={st.fwd*1000:.2f}ms  bwd={st.bwd*1000:.2f}ms  "
                f"total={((st.fwd+st.bwd)*1000):.2f}ms"
            )

    if report.warnings:
        lines.append(f"  Warnings:")
        for w in report.warnings:
            lines.append(f"    - {w}")

    lines.append("=" * 60)
    return "\n".join(lines)
