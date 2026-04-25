"""Chrome Trace builder — returns dict for chrome://tracing or writes to file."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.executor.scheduler import Timeline

logger = logging.getLogger(__name__)

# Category color map for chrome://tracing
CATEGORY_COLORS = {
    "compute": "#4CAF50",
    "communication": "#F44336",
    "memory": "#FF9800",
}

# Op type friendly name mapping
_OP_SHORT_NAMES: dict[str, str] = {
    "aten.mm.default": "mm",
    "aten.addmm.default": "addmm",
    "aten.bmm.default": "bmm",
    "aten.linear.default": "linear",
    "aten._scaled_dot_product_flash_attention.default": "flash_attn",
    "aten.scaled_dot_product_attention.default": "sdpa",
    "aten.layer_norm.default": "layer_norm",
    "aten.native_layer_norm.default": "layer_norm",
    "aten.rms_norm.default": "rms_norm",
    "aten.native_group_norm.default": "group_norm",
    "aten.softmax.int": "softmax",
    "aten._softmax.default": "softmax",
    "aten.gelu.default": "gelu",
    "aten.silu.default": "silu",
    "aten.relu.default": "relu",
    "aten.mul.Tensor": "mul",
    "aten.add.Tensor": "add",
    "aten.div.Tensor": "div",
    "aten.exp.default": "exp",
    "aten.log.default": "log",
    "aten.sum.dim_IntList": "sum",
    "aten.mean.dim": "mean",
    "aten.matmul.default": "matmul",
    "aten.embedding.default": "embedding",
    "aten.all_gather.default": "all_gather",
    "aten.all_reduce.default": "all_reduce",
    "aten.reduce_scatter.default": "reduce_scatter",
    "aten.broadcast.default": "broadcast",
}


def _short_name(op_type: str) -> str:
    """Convert aten op type to friendly short name."""
    if op_type in _OP_SHORT_NAMES:
        return _OP_SHORT_NAMES[op_type]
    # Generic fallback: strip "aten." and ".default"
    parts = op_type.split(".")
    if len(parts) >= 2:
        return parts[1]
    return op_type


def build_chrome_trace(
    timeline: "Timeline",
    name: str = "",
    metadata: dict | None = None,
) -> dict:
    """Convert a Timeline to Chrome Trace Event Format dict.

    Parameters
    ----------
    timeline : Timeline
        Scheduling result from DAGScheduler.
    name : str
        Display name for the trace viewer.
    metadata : dict | None
        Additional metadata (model, hardware, phase, parallel, etc.).

    Returns
    -------
    dict
        Chrome Trace Event Format compatible dict.
    """
    events = []
    for op in timeline.scheduled_ops:
        cat = op.category if op.category in CATEGORY_COLORS else "compute"
        stream_label = f"{op.stream_type}_{op.stream_id}"
        events.append({
            "name": _short_name(op.op_type),
            "cat": cat,
            "ph": "X",
            "pid": 0,
            "tid": stream_label,
            "ts": op.start_us,
            "dur": op.latency_us,
            "args": {
                "op_type": op.op_type,
                "stream_type": op.stream_type,
                "node_id": op.node_id,
            },
        })

    trace_meta = {"name": name or timeline.graph_name}
    if metadata:
        trace_meta.update(metadata)

    return {
        "traceEvents": events,
        "displayTimeUnit": "us",
        "metadata": trace_meta,
    }


def build_chrome_trace_multi(
    phases: dict[str, "Timeline"],
    name: str = "",
    metadata: dict | None = None,
) -> dict:
    """Merge multiple phase timelines into a single Chrome Trace.

    Each phase gets its own pid so they appear as separate processes
    in chrome://tracing.

    Parameters
    ----------
    phases : dict[str, Timeline]
        Mapping from phase name to Timeline (e.g. {"prefill": ..., "decode": ...}).
    name : str
        Display name for the trace viewer.
    metadata : dict | None
        Additional metadata.

    Returns
    -------
    dict
        Merged Chrome Trace Event Format dict.
    """
    all_events = []
    for pid, (phase_name, tl) in enumerate(phases.items()):
        for op in tl.scheduled_ops:
            cat = op.category if op.category in CATEGORY_COLORS else "compute"
            stream_label = f"{op.stream_type}_{op.stream_id}"
            all_events.append({
                "name": _short_name(op.op_type),
                "cat": cat,
                "ph": "X",
                "pid": pid,
                "pid_name": phase_name,
                "tid": stream_label,
                "ts": op.start_us,
                "dur": op.latency_us,
                "args": {
                    "op_type": op.op_type,
                    "stream_type": op.stream_type,
                    "node_id": op.node_id,
                    "phase": phase_name,
                },
            })

    trace_meta = {"name": name or "multi_phase"}
    if metadata:
        trace_meta.update(metadata)

    return {
        "traceEvents": all_events,
        "displayTimeUnit": "us",
        "metadata": trace_meta,
    }


def export_chrome_trace(
    timeline: "Timeline",
    output_path: Path,
    name: str = "",
    metadata: dict | None = None,
) -> Path:
    """Build and write a Chrome Trace file.

    Parameters
    ----------
    timeline : Timeline
        Scheduling result from DAGScheduler.
    output_path : Path
        Output .json file path.
    name : str
        Display name for the trace viewer.
    metadata : dict | None
        Additional metadata (model, hardware, phase, parallel, etc.).

    Returns
    -------
    Path
        The output file path.
    """
    trace = build_chrome_trace(timeline, name=name, metadata=metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trace, indent=2))
    logger.info("Exported Chrome Trace to %s (%d events)", output_path,
                len(trace["traceEvents"]))
    return output_path


def export_chrome_trace_multi(
    phases: dict[str, "Timeline"],
    output_path: Path,
    name: str = "",
    metadata: dict | None = None,
) -> Path:
    """Build and write a multi-phase Chrome Trace file.

    Parameters
    ----------
    phases : dict[str, Timeline]
        Mapping from phase name to Timeline.
    output_path : Path
        Output .json file path.
    name : str
        Display name for the trace viewer.
    metadata : dict | None
        Additional metadata.

    Returns
    -------
    Path
        The output file path.
    """
    trace = build_chrome_trace_multi(phases, name=name, metadata=metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trace, indent=2))
    logger.info("Exported multi-phase Chrome Trace to %s (%d events)",
                output_path, len(trace["traceEvents"]))
    return output_path
