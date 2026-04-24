"""Chrome Trace export for pipeline visualization.

Generates trace.json compatible with chrome://tracing.

Each PP stage is a track; fwd/bwd ops are events within each track.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from zrt.training.compose.pipeline import StepResult

if TYPE_CHECKING:
    from zrt.training.compose.stage import StageTime


def build_chrome_trace(
    per_stage: list["StageTime"],
    step_result: StepResult,
    schedule_name: str = "1f1b",
    pp: int = 1,
    M: int = 1,
) -> dict:
    """Build Chrome trace JSON structure.

    Each PP stage becomes a track (process).
    Each fwd/bwd microbatch becomes an event (thread slice).
    """
    events: list[dict] = []
    track_names: dict[int, str] = {}

    for s in range(len(per_stage)):
        track_names[s * 2] = f"Stage {s} / Forward"
        track_names[s * 2 + 1] = f"Stage {s} / Backward"

    t_fwd = [st.fwd * 1e6 for st in per_stage] if per_stage else [0] * pp
    t_bwd = [st.bwd * 1e6 for st in per_stage] if per_stage else [0] * pp

    base_time = 0.0

    if schedule_name == "1f1b":
        events = _build_1f1b_events(t_fwd, t_bwd, pp, M, base_time)
    elif schedule_name == "i1f1b":
        events = _build_interleaved_events(t_fwd, t_bwd, pp, M, base_time, step_result.vpp_chunks if hasattr(step_result, 'vpp_chunks') else 1)
    elif schedule_name in ("dualpipe", "dualpipev"):
        events = _build_dualpipe_events(t_fwd, t_bwd, pp, M, base_time)
    else:
        events = _build_1f1b_events(t_fwd, t_bwd, pp, M, base_time)

    for e in events:
        pid = e["pid"]
        if pid in track_names:
            e["name"] = f"{track_names[pid]}: {e.get('mb', '?')}"

    trace = {
        "traceEvents": events,
        "metadata": {
            "step_time_ms": step_result.step_time * 1000,
            "bubble_fraction": step_result.bubble_fraction,
            "mfu": step_result.mfu,
            "schedule": schedule_name,
            "pp": pp,
            "microbatches": M,
        },
    }

    return trace


def _build_1f1b_events(
    t_fwd: list[float], t_bwd: list[float],
    pp: int, M: int, base_time: float,
) -> list[dict]:
    """Build events for standard 1F1B schedule."""
    events: list[dict] = []
    time = base_time

    warmup_fwd = pp - 1
    for mb in range(warmup_fwd):
        for s in range(pp):
            start = time + s * t_fwd[0]
            events.append({
                "name": f"Fwd mb={mb}",
                "cat": "forward",
                "ph": "X",
                "ts": start,
                "dur": t_fwd[s] if s < len(t_fwd) else t_fwd[-1],
                "pid": s * 2,
                "tid": mb,
                "mb": mb,
            })
        time += t_fwd[0]

    steady = M - warmup_fwd
    for mb in range(warmup_fwd, M):
        for s in range(pp):
            fwd_start = time + s * t_fwd[0]
            events.append({
                "name": f"Fwd mb={mb}",
                "cat": "forward",
                "ph": "X",
                "ts": fwd_start,
                "dur": t_fwd[s] if s < len(t_fwd) else t_fwd[-1],
                "pid": s * 2,
                "tid": mb,
                "mb": mb,
            })

            bwd_start = fwd_start + t_fwd[s] + (pp - 1 - s) * t_fwd[0]
            events.append({
                "name": f"Bwd mb={mb - pp + 1}",
                "cat": "backward",
                "ph": "X",
                "ts": bwd_start,
                "dur": t_bwd[s] if s < len(t_bwd) else t_bwd[-1],
                "pid": s * 2 + 1,
                "tid": mb - pp + 1,
                "mb": mb - pp + 1,
            })
        time += t_fwd[0]

    cooldown_start = time
    for mb in range(M - pp + 1, M):
        for s in range(pp):
            bwd_start = cooldown_start + s * t_bwd[-1]
            events.append({
                "name": f"Bwd mb={mb}",
                "cat": "backward",
                "ph": "X",
                "ts": bwd_start,
                "dur": t_bwd[s] if s < len(t_bwd) else t_bwd[-1],
                "pid": s * 2 + 1,
                "tid": mb,
                "mb": mb,
            })
        cooldown_start += t_bwd[-1]

    return events


def _build_interleaved_events(
    t_fwd: list[float], t_bwd: list[float],
    pp: int, M: int, base_time: float, vpp: int,
) -> list[dict]:
    """Build events for VPP/Interleaved 1F1B schedule."""
    events: list[dict] = []

    chunk_time_fwd = [t / vpp for t in t_fwd] if t_fwd else [0] * pp
    chunk_time_bwd = [t / vpp for t in t_bwd] if t_bwd else [0] * pp

    time = base_time
    warmup_chunks = (pp - 1) * vpp

    for chunk in range(warmup_chunks):
        mb = chunk // vpp
        v = chunk % vpp
        for s in range(pp):
            start = time + s * chunk_time_fwd[0]
            events.append({
                "name": f"Fwd mb={mb} v={v}",
                "cat": "forward",
                "ph": "X",
                "ts": start,
                "dur": chunk_time_fwd[s] if s < len(chunk_time_fwd) else chunk_time_fwd[-1],
                "pid": s * 2,
                "tid": chunk,
                "mb": mb,
            })
        time += chunk_time_fwd[0]

    steady_chunks = M * vpp - warmup_chunks
    for chunk in range(warmup_chunks, M * vpp):
        mb = chunk // vpp
        v = chunk % vpp
        for s in range(pp):
            fwd_start = time + s * chunk_time_fwd[0]
            events.append({
                "name": f"Fwd mb={mb} v={v}",
                "cat": "forward",
                "ph": "X",
                "ts": fwd_start,
                "dur": chunk_time_fwd[s] if s < len(chunk_time_fwd) else chunk_time_fwd[-1],
                "pid": s * 2,
                "tid": chunk,
                "mb": mb,
            })

            bwd_start = fwd_start + chunk_time_fwd[s] + (pp - 1 - s) * chunk_time_fwd[0]
            events.append({
                "name": f"Bwd mb={mb - pp + 1}",
                "cat": "backward",
                "ph": "X",
                "ts": bwd_start,
                "dur": chunk_time_bwd[s] if s < len(chunk_time_bwd) else chunk_time_bwd[-1],
                "pid": s * 2 + 1,
                "tid": chunk - (pp - 1) * vpp,
                "mb": mb - pp + 1,
            })
        time += chunk_time_fwd[0]

    return events


def _build_dualpipe_events(
    t_fwd: list[float], t_bwd: list[float],
    pp: int, M: int, base_time: float,
) -> list[dict]:
    """Build events for DualPipe schedule (concurrent fwd+bwd)."""
    events: list[dict] = []
    time = base_time

    warmup_fwd = pp - 1
    for mb in range(warmup_fwd):
        for s in range(pp):
            fwd_start = time + s * max(t_fwd[0], t_bwd[0])
            events.append({
                "name": f"Fwd mb={mb}",
                "cat": "forward",
                "ph": "X",
                "ts": fwd_start,
                "dur": t_fwd[s] if s < len(t_fwd) else t_fwd[-1],
                "pid": s * 2,
                "tid": mb,
                "mb": mb,
            })
        time += max(t_fwd[0], t_bwd[0])

    steady = M - warmup_fwd
    for mb in range(warmup_fwd, M):
        for s in range(pp):
            stage_time = max(t_fwd[s] if s < len(t_fwd) else t_fwd[-1],
                            t_bwd[s] if s < len(t_bwd) else t_bwd[-1])
            fwd_start = time + s * stage_time
            events.append({
                "name": f"Fwd mb={mb}",
                "cat": "forward",
                "ph": "X",
                "ts": fwd_start,
                "dur": t_fwd[s] if s < len(t_fwd) else t_fwd[-1],
                "pid": s * 2,
                "tid": mb,
                "mb": mb,
            })

            bwd_start = fwd_start
            events.append({
                "name": f"Bwd mb={mb - pp + 1}",
                "cat": "backward",
                "ph": "X",
                "ts": bwd_start,
                "dur": t_bwd[s] if s < len(t_bwd) else t_bwd[-1],
                "pid": s * 2 + 1,
                "tid": mb - pp + 1,
                "mb": mb - pp + 1,
            })
        time += max(t_fwd[0], t_bwd[0])

    cooldown_start = time
    for mb in range(M - pp + 1, M):
        for s in range(pp):
            bwd_start = cooldown_start + s * t_bwd[-1]
            events.append({
                "name": f"Bwd mb={mb}",
                "cat": "backward",
                "ph": "X",
                "ts": bwd_start,
                "dur": t_bwd[s] if s < len(t_bwd) else t_bwd[-1],
                "pid": s * 2 + 1,
                "tid": mb,
                "mb": mb,
            })
        cooldown_start += t_bwd[-1]

    return events


def write_chrome_trace(
    trace: dict, path: str | Path,
) -> None:
    """Write trace.json to file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(trace, f, indent=2)


def trace_summary(trace: dict) -> str:
    """Human-readable summary of trace metadata."""
    meta = trace.get("metadata", {})
    lines = []
    lines.append(f"Schedule: {meta.get('schedule', 'unknown')}")
    lines.append(f"PP stages: {meta.get('pp', 0)}")
    lines.append(f"Microbatches: {meta.get('microbatches', 0)}")
    lines.append(f"Step time: {meta.get('step_time_ms', 0):.1f} ms")
    lines.append(f"Bubble fraction: {meta.get('bubble_fraction', 0):.1%}")
    lines.append(f"MFU: {meta.get('mfu', 0):.1%}")
    return "\n".join(lines)