"""E2ESummary / TrainingSummary: per-phase and full-step performance reports.

Inference usage::

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

Training usage::

    from python.zrt.report import build_training_summary

    summary = build_training_summary(
        model="DeepSeek-V3",
        hardware="nvidia_h100_sxm",
        batch_size=1,
        seq_len=4096,
        fwd_graph=fwd_transformed,
        bwd_graph=bwd_transformed,
        fwd_results=fwd_sim_results,
        bwd_results=bwd_sim_results,
        fwd_timeline=fwd_timeline,
        bwd_timeline=bwd_timeline,
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
    from python.zrt.transform.analysis.training import TrainingMemoryBreakdown


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
    total_bytes:        int         # read_bytes + write_bytes
    read_bytes:         int
    write_bytes:        int
    arithmetic_intensity: float     # total_flops / total_bytes [ops/byte]

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
            f"  Arith intensity: {self.arithmetic_intensity:.2f} ops/byte",
            f"  Total FLOPs:   {self.total_flops / 1e12:.3f} TFLOPs",
            f"  Total bytes:   {self.total_bytes / 1e9:.3f} GB"
            f"  (R={self.read_bytes/1e9:.2f} + W={self.write_bytes/1e9:.2f} GB)",
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
    total_flops = sum(r.flops       for r in sim_results.values())
    read_bytes  = sum(r.read_bytes  for r in sim_results.values())
    write_bytes = sum(r.write_bytes for r in sim_results.values())
    total_bytes = read_bytes + write_bytes

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
    arithmetic_intensity = total_flops / total_bytes if total_bytes > 0 else 0.0

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
        hbm_bandwidth_util   = hbm_bandwidth_util,
        total_flops    = total_flops,
        total_bytes    = total_bytes,
        read_bytes     = read_bytes,
        write_bytes    = write_bytes,
        arithmetic_intensity = arithmetic_intensity,
        by_component   = by_component,
        by_layer       = by_layer,
        top_bottleneck_ops = top_bottleneck_ops,
        memory_budget  = memory_budget,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TrainingSummary
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingSummary:
    """End-to-end performance summary for one training step (forward + backward).

    A training step is forward_ms + backward_ms; the optimizer step is not
    modeled here (it is typically < 1% of step time for standard Adam).
    """

    # ── metadata ──────────────────────────────────────────────────────────────
    model:         str
    hardware:      str
    parallel_desc: str
    batch_size:    int
    seq_len:       int

    # ── step timing ───────────────────────────────────────────────────────────
    step_ms:     float      # forward_ms + backward_ms
    forward_ms:  float
    backward_ms: float

    # ── throughput ────────────────────────────────────────────────────────────
    samples_per_sec: float  # batch_size / step_s
    tokens_per_sec:  float  # batch_size * seq_len / step_s

    # ── compute / comm decomposition per phase ────────────────────────────────
    fwd_compute_ms:      float
    fwd_comm_ms:         float
    fwd_exposed_comm_ms: float
    fwd_overlap_ratio:   float

    bwd_compute_ms:      float
    bwd_comm_ms:         float
    bwd_exposed_comm_ms: float
    bwd_overlap_ratio:   float

    # ── hw efficiency ─────────────────────────────────────────────────────────
    mfu:                  float   # (fwd_flops + bwd_flops) / (step_s * peak_flops)
    hbm_bw_util:          float
    arithmetic_intensity: float   # total_flops / total_bytes [ops/byte]

    # ── FLOPs / bytes ─────────────────────────────────────────────────────────
    fwd_flops:       int
    bwd_flops:       int
    total_flops:     int
    fwd_read_bytes:  int
    fwd_write_bytes: int
    fwd_bytes:       int   # fwd_read_bytes + fwd_write_bytes
    bwd_read_bytes:  int
    bwd_write_bytes: int
    bwd_bytes:       int   # bwd_read_bytes + bwd_write_bytes

    # ── recompute (activation checkpointing) ─────────────────────────────────
    recompute_op_count: int     # number of recomputed ops detected
    recompute_ratio:    float   # recomputed_latency / bwd_compute_ms

    # ── hierarchical breakdown ────────────────────────────────────────────────
    by_component:       dict[str, float]              # component → % of step latency
    by_layer:           list[float]                   # per-layer step latency (ms)
    top_bottleneck_ops: list[tuple[str, float]]       # [(desc, latency_us), ...]

    # ── memory (optional) ──────────────────────────────────────────────────────
    memory_breakdown: "TrainingMemoryBreakdown | None" = None

    # ── chrome trace (optional) ───────────────────────────────────────────────
    chrome_trace: dict | None = None

    # ── optimizer (optional, per §6.3 of muon_optimizer_design.md) ────────────
    optimizer_type: str = "adam"              # "adam" or "muon"
    muon_param_fraction: float = 0.0          # 0.0 for Adam, 0.85 for Muon default
    opt_state_gb: float = 0.0                 # optimizer state memory in GB
    opt_state_savings_gb: float = 0.0         # savings vs pure Adam
    optimizer_step_ms: float = 0.0            # optimizer step time in ms
    muon_ag_rs_ms: float = 0.0                # Muon AG+RS communication time
    muon_ns_tflops: float = 0.0               # NS iteration FLOPs in TFLOPs
    optimizer_time_fraction: float = 0.0     # optimizer_step_ms / total_step_ms

    # ── string representation ─────────────────────────────────────────────────

    def __str__(self) -> str:
        fwd_pct = self.forward_ms / self.step_ms * 100 if self.step_ms > 0 else 0.0
        bwd_pct = self.backward_ms / self.step_ms * 100 if self.step_ms > 0 else 0.0
        lines = [
            f"=== Training Summary: {self.model} | {self.hardware} ===",
            f"  Parallel:      {self.parallel_desc}",
            f"  Batch/SeqLen:  bs={self.batch_size}, seq={self.seq_len}",
            "",
            f"  Step latency:  {self.step_ms:.3f} ms",
            f"  Forward:       {self.forward_ms:.3f} ms  ({fwd_pct:.1f}%)",
            f"  Backward:      {self.backward_ms:.3f} ms  ({bwd_pct:.1f}%)",
            "",
            f"  Samples/sec:   {self.samples_per_sec:.2f}",
            f"  Tokens/sec:    {self.tokens_per_sec:.1f}",
            "",
            f"  MFU:           {self.mfu:.2%}",
            f"  HBM BW util:   {self.hbm_bw_util:.2%}",
            f"  Arith intensity: {self.arithmetic_intensity:.2f} ops/byte",
            f"  Total FLOPs:   {self.total_flops / 1e12:.3f} TFLOPs"
            f"  (fwd {self.fwd_flops/1e12:.3f} + bwd {self.bwd_flops/1e12:.3f})",
            f"  Total R/W:     fwd R={self.fwd_read_bytes/1e9:.2f}G W={self.fwd_write_bytes/1e9:.2f}G"
            f"  bwd R={self.bwd_read_bytes/1e9:.2f}G W={self.bwd_write_bytes/1e9:.2f}G",
            "",
            "  Forward compute/comm:",
            f"    compute    {self.fwd_compute_ms:.3f} ms",
            f"    comm       {self.fwd_comm_ms:.3f} ms"
            f"  (exposed {self.fwd_exposed_comm_ms:.3f} ms,"
            f" overlap {self.fwd_overlap_ratio:.1%})",
            "  Backward compute/comm:",
            f"    compute    {self.bwd_compute_ms:.3f} ms",
            f"    comm       {self.bwd_comm_ms:.3f} ms"
            f"  (exposed {self.bwd_exposed_comm_ms:.3f} ms,"
            f" overlap {self.bwd_overlap_ratio:.1%})",
        ]
        if self.recompute_op_count > 0:
            lines += [
                "",
                f"  Recompute ops: {self.recompute_op_count}"
                f"  (overhead {self.recompute_ratio:.1%} of bwd compute)",
            ]
        if self.memory_breakdown is not None:
            mb = self.memory_breakdown
            lines += [
                "",
                "  Memory (per GPU):",
                f"    weights    {mb.weights/1e9:.2f} GB",
                f"    grads      {mb.grads/1e9:.2f} GB",
                f"    opt_state  {mb.opt_state/1e9:.2f} GB",
                f"    activations {mb.activations/1e9:.2f} GB",
                f"    comm_buf   {mb.comm_buffers/1e9:.2f} GB",
                f"    total      {mb.total/1e9:.2f} GB",
            ]
        if self.by_component:
            lines += ["", "  By component (% of step):"]
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
                lines.append(f"    {op_desc:<44s}: {lat_us:.1f} us")
        return "\n".join(lines)


# ── builder ───────────────────────────────────────────────────────────────────

def build_training_summary(
    model:         str,
    hardware:      str,
    batch_size:    int,
    seq_len:       int,
    fwd_graph:     "OpGraph",
    bwd_graph:     "OpGraph",
    fwd_results:   "dict[str, SimResult]",
    bwd_results:   "dict[str, SimResult]",
    fwd_timeline:  "Timeline",
    bwd_timeline:  "Timeline",
    hw_spec:       "HardwareSpec",
    parallel_desc: str = "single",
    top_n:         int = 10,
    memory_breakdown: "TrainingMemoryBreakdown | None" = None,
) -> TrainingSummary:
    """Build a TrainingSummary from forward + backward simulation outputs.

    Parameters
    ----------
    model / hardware / batch_size / seq_len
        Descriptive metadata.
    fwd_graph / bwd_graph
        Transformed OpGraphs for train_forward / train_backward phases.
    fwd_results / bwd_results
        ``dict[node_id → SimResult]`` from ``SimulatorHub.simulate_graph()``.
    fwd_timeline / bwd_timeline
        ``Timeline`` from ``DAGScheduler.schedule()``.
    hw_spec
        Used for MFU and HBM bandwidth utilisation.
    parallel_desc
        Human-readable string, e.g. ``"TP8-EP8"``.
    top_n
        How many bottleneck ops to surface per phase.
    memory_breakdown
        Optional ``TrainingMemoryBreakdown`` from ``TrainingMemoryPass``.
    """
    from python.zrt.ir.types import DType
    from python.zrt.ir.hierarchy import GraphHierarchy

    # ── phase timings ─────────────────────────────────────────────────────────
    fwd_us = fwd_timeline.total_latency_us
    bwd_us = bwd_timeline.total_latency_us
    step_us = fwd_us + bwd_us
    step_ms = step_us / 1_000.0
    fwd_ms  = fwd_us  / 1_000.0
    bwd_ms  = bwd_us  / 1_000.0
    step_s  = step_us * 1e-6

    # ── throughput ────────────────────────────────────────────────────────────
    samples_per_sec = batch_size / step_s if step_s > 0 else 0.0
    tokens_per_sec  = batch_size * seq_len / step_s if step_s > 0 else 0.0

    # ── comm decomposition per phase ──────────────────────────────────────────
    def _decompose(tl: "Timeline") -> tuple[float, float, float, float]:
        compute_ms      = tl.compute_time_us / 1_000.0
        comm_ms         = tl.comm_time_us    / 1_000.0
        overlap_ms      = tl.overlap_us      / 1_000.0
        exposed_comm_ms = max(0.0, comm_ms - overlap_ms)
        overlap_ratio   = (overlap_ms / comm_ms) if comm_ms > 0 else 1.0
        return compute_ms, comm_ms, exposed_comm_ms, overlap_ratio

    fwd_compute_ms, fwd_comm_ms, fwd_exposed_ms, fwd_overlap = _decompose(fwd_timeline)
    bwd_compute_ms, bwd_comm_ms, bwd_exposed_ms, bwd_overlap = _decompose(bwd_timeline)

    # ── FLOPs / bytes ─────────────────────────────────────────────────────────
    fwd_flops       = sum(r.flops       for r in fwd_results.values())
    fwd_read_bytes  = sum(r.read_bytes  for r in fwd_results.values())
    fwd_write_bytes = sum(r.write_bytes for r in fwd_results.values())
    fwd_bytes       = fwd_read_bytes + fwd_write_bytes
    bwd_flops       = sum(r.flops       for r in bwd_results.values())
    bwd_read_bytes  = sum(r.read_bytes  for r in bwd_results.values())
    bwd_write_bytes = sum(r.write_bytes for r in bwd_results.values())
    bwd_bytes       = bwd_read_bytes + bwd_write_bytes
    total_flops = fwd_flops + bwd_flops
    total_bytes = fwd_bytes + bwd_bytes

    # ── hw efficiency ─────────────────────────────────────────────────────────
    peak_flops = hw_spec.peak_flops(DType.BF16)
    hbm_bw     = hw_spec.hbm_bandwidth()

    mfu = (
        min(1.0, total_flops / (step_s * peak_flops))
        if (peak_flops > 0 and step_s > 0) else 0.0
    )
    hbm_bw_util = (
        min(1.0, total_bytes / (step_s * hbm_bw))
        if (hbm_bw > 0 and step_s > 0) else 0.0
    )
    arithmetic_intensity = total_flops / total_bytes if total_bytes > 0 else 0.0

    # ── recompute detection ───────────────────────────────────────────────────
    # Recomputed ops are flagged during graph capture (node.annotations["recompute"])
    # or stored as node attribute from the dispatch recorder.
    recompute_lat_us = 0.0
    recompute_op_count = 0
    for node in bwd_graph.nodes.values():
        is_recompute = (
            node.annotations.get("recompute", False)
            or node.attrs.get("recompute", False)
        )
        if is_recompute:
            recompute_op_count += 1
            r = bwd_results.get(node.id)
            if r:
                recompute_lat_us += r.latency_us

    recompute_ratio = (
        recompute_lat_us / (bwd_compute_ms * 1_000.0)
        if bwd_compute_ms > 0 else 0.0
    )

    # ── hierarchical breakdown (step-level) ───────────────────────────────────
    # Merge both phase latency maps; backward nodes are scaled by backward weight.
    all_latency: dict[str, float] = {}
    for r in fwd_results.values():
        all_latency[r.op_node_id] = r.latency_us
    for r in bwd_results.values():
        all_latency[r.op_node_id] = r.latency_us

    total_sim_us = (sum(r.latency_us for r in fwd_results.values()) +
                    sum(r.latency_us for r in bwd_results.values())) or 1.0

    # by_component: use forward graph (canonical layer structure)
    hier_fwd = GraphHierarchy(fwd_graph)
    comp_totals: dict[str, float] = {}
    for scope, val in hier_fwd.module_breakdown(all_latency, depth=4).items():
        comp = scope.rsplit(".", 1)[-1] if "." in scope else scope
        comp_totals[comp] = comp_totals.get(comp, 0.0) + val
    # Also aggregate backward-only nodes that don't appear in fwd_graph
    by_component = {
        k: v / total_sim_us * 100.0
        for k, v in comp_totals.items() if v > 0
    }

    # by_layer: per-layer total (fwd + bwd combined)
    layer_us: dict[int, float] = {}
    for graph, results in ((fwd_graph, fwd_results), (bwd_graph, bwd_results)):
        for node in graph.nodes.values():
            if node.layer and node.layer.isdigit():
                idx = int(node.layer)
                r = results.get(node.id)
                layer_us[idx] = layer_us.get(idx, 0.0) + (r.latency_us if r else 0.0)
    by_layer = [layer_us[i] / 1_000.0 for i in sorted(layer_us)]

    # top_bottleneck_ops: from both phases, sorted by latency
    all_results = list(fwd_results.values()) + list(bwd_results.values())
    sorted_all  = sorted(all_results, key=lambda r: r.latency_us, reverse=True)
    top_bottleneck_ops: list[tuple[str, float]] = []
    for r in sorted_all[:top_n]:
        node = fwd_graph.nodes.get(r.op_node_id) or bwd_graph.nodes.get(r.op_node_id)
        op_type = node.op_type if node else r.op_node_id
        phase   = "fwd" if r.op_node_id in fwd_results else "bwd"
        suffix  = f" [{node.scope.rsplit('.', 1)[-1]}|{phase}]" if (node and node.scope) else f" [{phase}]"
        top_bottleneck_ops.append((f"{op_type}{suffix}", r.latency_us))

    return TrainingSummary(
        model              = model,
        hardware           = hardware,
        parallel_desc      = parallel_desc,
        batch_size         = batch_size,
        seq_len            = seq_len,
        step_ms            = step_ms,
        forward_ms         = fwd_ms,
        backward_ms        = bwd_ms,
        samples_per_sec    = samples_per_sec,
        tokens_per_sec     = tokens_per_sec,
        fwd_compute_ms     = fwd_compute_ms,
        fwd_comm_ms        = fwd_comm_ms,
        fwd_exposed_comm_ms= fwd_exposed_ms,
        fwd_overlap_ratio  = fwd_overlap,
        bwd_compute_ms     = bwd_compute_ms,
        bwd_comm_ms        = bwd_comm_ms,
        bwd_exposed_comm_ms= bwd_exposed_ms,
        bwd_overlap_ratio  = bwd_overlap,
        mfu                    = mfu,
        hbm_bw_util            = hbm_bw_util,
        arithmetic_intensity   = arithmetic_intensity,
        fwd_flops          = fwd_flops,
        bwd_flops          = bwd_flops,
        total_flops        = total_flops,
        fwd_read_bytes     = fwd_read_bytes,
        fwd_write_bytes    = fwd_write_bytes,
        fwd_bytes          = fwd_bytes,
        bwd_read_bytes     = bwd_read_bytes,
        bwd_write_bytes    = bwd_write_bytes,
        bwd_bytes          = bwd_bytes,
        recompute_op_count = recompute_op_count,
        recompute_ratio    = recompute_ratio,
        by_component       = by_component,
        by_layer           = by_layer,
        top_bottleneck_ops = top_bottleneck_ops,
        memory_breakdown   = memory_breakdown,
    )
