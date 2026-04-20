"""E2ESummary: end-to-end performance report for one inference phase.

Usage::

    from python.zrt.report import build_summary

    summary = build_summary(
        model="DeepSeek-V3",
        hardware="nvidia_h100_sxm",
        phase="prefill",          # or "decode"
        batch_size=1,
        seq_len=4096,
        graph=transformed_graph,
        sim_results=sim_results,  # dict[node_id, SimResult]
        timeline=timeline,
        hw_spec=hw,
        parallel_desc="TP8-EP8",
    )
    print(summary)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.simulator.result import SimResult
    from python.zrt.executor.scheduler import Timeline
    from python.zrt.hardware.spec import HardwareSpec
    from python.zrt.memory import MemoryBudget


@dataclass
class E2ESummary:
    """End-to-end performance summary for one model + hardware + phase combination."""

    # ── metadata ──────────────────────────────────────────────────────────────
    model:         str
    hardware:      str
    phase:         str          # "prefill" | "decode"
    parallel_desc: str          # "TP8-EP8-PP1" | "single"
    batch_size:    int
    seq_len:       int          # prompt tokens (prefill) or 1 per decode step

    # ── core LLM metrics ─────────────────────────────────────────────────────
    latency_ms:     float
    tokens_per_sec: float
    ttft_ms:        float | None    # prefill only
    tpot_ms:        float | None    # decode only

    # ── compute / comm decomposition ──────────────────────────────────────────
    compute_ms:      float
    comm_ms:         float
    exposed_comm_ms: float          # comm not hidden by compute overlap
    overlap_ratio:   float          # fraction of comm hidden [0, 1]

    # ── hw efficiency ─────────────────────────────────────────────────────────
    mfu:                float       # model FLOPs utilization [0, 1]
    hbm_bandwidth_util: float       # HBM bandwidth utilization [0, 1]
    total_flops:        int
    total_bytes:        int

    # ── hierarchical decomposition ────────────────────────────────────────────
    by_component: dict[str, float]              # component → % of total serial latency
    by_layer:     list[float]                   # per-layer latency (ms), ordered by index
    top_bottleneck_ops: list[tuple[str, float]] # [(op_desc, latency_us), ...]

    # ── memory budget (optional) ──────────────────────────────────────────────
    memory_budget: "MemoryBudget | None" = None    # memory breakdown estimate

    # ── string representation ─────────────────────────────────────────────────

    def __str__(self) -> str:
        lines = [
            f"=== E2E Summary: {self.model} | {self.hardware} | {self.phase.upper()} ===",
            f"  Parallel:      {self.parallel_desc}",
            f"  Batch/SeqLen:  bs={self.batch_size}, seq={self.seq_len}",
            "",
            f"  Latency:       {self.latency_ms:.3f} ms",
        ]
        if self.ttft_ms is not None:
            lines.append(f"  TTFT:          {self.ttft_ms:.3f} ms")
        if self.tpot_ms is not None:
            lines.append(f"  TPOT:          {self.tpot_ms:.3f} ms/token")
        lines += [
            f"  Throughput:    {self.tokens_per_sec:.1f} tokens/s",
            "",
            f"  Compute:       {self.compute_ms:.3f} ms",
            f"  Comm:          {self.comm_ms:.3f} ms",
            f"  Exposed comm:  {self.exposed_comm_ms:.3f} ms",
            f"  Overlap ratio: {self.overlap_ratio:.1%}",
            "",
            f"  MFU:           {self.mfu:.2%}",
            f"  HBM BW util:   {self.hbm_bandwidth_util:.2%}",
            f"  Total FLOPs:   {self.total_flops / 1e12:.3f} TFLOPs",
            f"  Total bytes:   {self.total_bytes / 1e9:.3f} GB",
        ]
        if self.by_component:
            lines.append("")
            lines.append("  By component:")
            for comp, pct in sorted(self.by_component.items(), key=lambda x: -x[1]):
                lines.append(f"    {comp:<24s}: {pct:.1f}%")
        if self.by_layer:
            n = len(self.by_layer)
            avg = sum(self.by_layer) / n
            lines += ["", f"  By layer ({n} layers, avg {avg:.3f} ms):"]
            show = self.by_layer if n <= 6 else (self.by_layer[:3] + [...] + self.by_layer[-3:])  # type: ignore[list-item]
            idx = 0
            for item in show:
                if item is ...:
                    lines.append("    ...")
                else:
                    lines.append(f"    Layer {idx:3d}: {item:.3f} ms")
                    idx += 1
        if self.top_bottleneck_ops:
            lines += ["", "  Top bottleneck ops:"]
            for op_desc, lat_us in self.top_bottleneck_ops:
                lines.append(f"    {op_desc:<44s}: {lat_us:.1f} µs")
        return "\n".join(lines)


# ── builder ───────────────────────────────────────────────────────────────────

def build_summary(
    model:         str,
    hardware:      str,
    phase:         str,
    batch_size:    int,
    seq_len:       int,
    graph:         "OpGraph",
    sim_results:   "dict[str, SimResult]",
    timeline:      "Timeline",
    hw_spec:       "HardwareSpec",
    parallel_desc: str = "single",
    top_n:         int = 10,
    memory_budget: "MemoryBudget | None" = None,
) -> E2ESummary:
    """Build an E2ESummary from simulation outputs.

    Parameters
    ----------
    model / hardware / phase / batch_size / seq_len
        Descriptive metadata.  ``seq_len`` is the prompt length for prefill
        and 1 for a single decode step.
    graph
        The transformed OpGraph (used for hierarchical breakdown).
    sim_results
        ``dict[node_id → SimResult]`` from ``SimulatorHub.simulate_graph()``.
    timeline
        ``Timeline`` from ``DAGScheduler.schedule()``.
    hw_spec
        Used to compute MFU and HBM bandwidth utilisation.
    parallel_desc
        Human-readable parallel config string, e.g. ``"TP8-EP8"``.
    top_n
        How many bottleneck ops to include in the report.
    """
    from python.zrt.ir.types import DType
    from python.zrt.ir.hierarchy import GraphHierarchy

    latency_us = timeline.total_latency_us
    latency_ms = latency_us / 1_000.0
    latency_s  = latency_us * 1e-6

    # ── LLM metrics ───────────────────────────────────────────────────────────
    if phase == "prefill":
        ttft_ms        = latency_ms
        tpot_ms        = None
        tokens_per_sec = (batch_size * seq_len / latency_s) if latency_s > 0 else 0.0
    else:
        ttft_ms        = None
        tpot_ms        = latency_ms
        tokens_per_sec = (batch_size / latency_s) if latency_s > 0 else 0.0

    # ── comm decomposition ────────────────────────────────────────────────────
    compute_ms      = timeline.compute_time_us / 1_000.0
    comm_ms         = timeline.comm_time_us    / 1_000.0
    overlap_ms      = timeline.overlap_us      / 1_000.0
    exposed_comm_ms = max(0.0, comm_ms - overlap_ms)
    overlap_ratio   = (overlap_ms / comm_ms) if comm_ms > 0 else 1.0

    # ── hw efficiency ─────────────────────────────────────────────────────────
    total_flops = sum(r.flops      for r in sim_results.values())
    total_bytes = sum(r.total_bytes for r in sim_results.values())

    peak_flops = hw_spec.peak_flops(DType.BF16)
    hbm_bw     = hw_spec.hbm_bandwidth()

    mfu = (
        min(1.0, total_flops / (latency_s * peak_flops))
        if (peak_flops > 0 and latency_s > 0) else 0.0
    )
    hbm_bandwidth_util = (
        min(1.0, total_bytes / (latency_s * hbm_bw))
        if (hbm_bw > 0 and latency_s > 0) else 0.0
    )

    # ── hierarchical decomposition ────────────────────────────────────────────
    latency_map  = {r.op_node_id: r.latency_us for r in sim_results.values()}
    total_sim_us = sum(latency_map.values()) or 1.0
    hier         = GraphHierarchy(graph)

    # by_component: aggregate depth-4 scopes, group by last segment name
    comp_totals: dict[str, float] = {}
    for scope, val in hier.module_breakdown(latency_map, depth=4).items():
        comp = scope.rsplit(".", 1)[-1] if "." in scope else scope
        comp_totals[comp] = comp_totals.get(comp, 0.0) + val
    by_component = {
        k: v / total_sim_us * 100.0
        for k, v in comp_totals.items() if v > 0
    }

    # by_layer: depth-3 numeric nodes under a "layers" parent
    layer_latencies: dict[int, float] = {}
    for hn in hier.at_depth(3):
        if hn.name.isdigit() and "layers" in hn.scope:
            layer_latencies[int(hn.name)] = hier.aggregate(hn, latency_map) / 1_000.0
    by_layer = [layer_latencies[i] for i in sorted(layer_latencies)]

    # top_bottleneck_ops
    sorted_ops = sorted(sim_results.values(), key=lambda r: r.latency_us, reverse=True)
    top_bottleneck_ops: list[tuple[str, float]] = []
    for r in sorted_ops[:top_n]:
        node    = graph.nodes.get(r.op_node_id)
        op_type = node.op_type if node else r.op_node_id
        suffix  = f" [{node.scope.rsplit('.', 1)[-1]}]" if (node and node.scope) else ""
        top_bottleneck_ops.append((f"{op_type}{suffix}", r.latency_us))

    return E2ESummary(
        model          = model,
        hardware       = hardware,
        phase          = phase,
        parallel_desc  = parallel_desc,
        batch_size     = batch_size,
        seq_len        = seq_len,
        latency_ms     = latency_ms,
        tokens_per_sec = tokens_per_sec,
        ttft_ms        = ttft_ms,
        tpot_ms        = tpot_ms,
        compute_ms     = compute_ms,
        comm_ms        = comm_ms,
        exposed_comm_ms= exposed_comm_ms,
        overlap_ratio  = overlap_ratio,
        mfu            = mfu,
        hbm_bandwidth_util = hbm_bandwidth_util,
        total_flops    = total_flops,
        total_bytes    = total_bytes,
        by_component   = by_component,
        by_layer       = by_layer,
        top_bottleneck_ops = top_bottleneck_ops,
        memory_budget  = memory_budget,
    )
