"""Shared training report types.

This module provides unified report types used by both spec-based (Stack A)
and graph-native (Stack B) training estimation paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from zrt.training.models.memory import MemBreakdown


@dataclass
class TrainingReport:
    """Unified training performance estimation report.

    Used by both spec-based estimation (Stack A) and graph-native estimation
    (Stack B) to return consistent, comparable results.

    Core timing metrics
    ------------------
    step_time_ms: Total step time in milliseconds
    per_stage: Per-stage timing breakdown (list of StageTime from compose/stage.py) [Stack A]
    per_stage_ms: Average per-stage time in milliseconds [Stack B]

    Efficiency metrics
    -----------------
    mfu: Model FLOPs Utilization (excludes recompute overhead)
    hfu: Hardware FLOPs Utilization (includes recompute overhead)

    FLOPs breakdown
    --------------
    total_flops: Total training FLOPs (forward + backward + recompute) [Stack A]
    forward_flops: Forward pass FLOPs [Stack B]
    backward_flops: Backward pass FLOPs [Stack B]
    training_flops: Alias for total_flops [Stack B]

    Memory metrics
    -------------
    memory: MemBreakdown with weights, grads, opt_state, activations, comm_buffers [Stack A]
    memory_breakdown: Dict representation of memory breakdown [Stack B]

    Pipeline metrics
    ---------------
    bubble_fraction: Fraction of step time spent in pipeline bubble
    schedule_name: Pipeline schedule name ("1f1b", "i1f1b", "dualpipe", etc.)
    warmup_steps: Number of warmup microbatches [Stack B]
    cooldown_steps: Number of cooldown microbatches [Stack B]
    steady_steps: Number of steady-state microbatches [Stack B]

    Config info
    ----------
    config_summary: Human-readable config summary (str) [Stack B] or detailed dict [Stack A]
    warnings: Validation warnings [Stack A]

    Model info
    ---------
    total_params: Total model parameters [Stack B]
    """

    # Core timing metrics
    step_time_ms: float = 0.0
    per_stage: list = field(default_factory=list)  # List[StageTime] [Stack A]
    per_stage_ms: float = 0.0  # [Stack B]

    # Efficiency metrics
    mfu: float = 0.0
    hfu: float = 0.0

    # FLOPs breakdown
    total_flops: float = 0.0  # [Stack A]
    forward_flops: float = 0.0  # [Stack B]
    backward_flops: float = 0.0  # [Stack B]
    training_flops: float = 0.0  # [Stack B]

    # Memory metrics
    memory: MemBreakdown | None = None  # [Stack A]
    memory_breakdown: dict = field(default_factory=dict)  # [Stack B]

    # Pipeline metrics
    bubble_fraction: float = 0.0
    schedule_name: str = "1f1b"
    warmup_steps: int = 0  # [Stack B]
    cooldown_steps: int = 0  # [Stack B]
    steady_steps: int = 0  # [Stack B]
    
    # Step time breakdown (milliseconds)
    # Invariants:
    #   step_time_ms     = pipeline_time_ms + optimizer_time_ms + optimizer_comm_ms
    #   pipeline_time_ms = compute_time_ms + exposed_comm_ms
    #   exposed_comm_ms  = Σ *_exposed_ms fields
    #   hidden_comm_ms   = dp_hidden_ms + tp_hidden_ms + ep_hidden_ms
    #   total_comm_volume_ms = exposed_comm_ms + hidden_comm_ms
    pipeline_time_ms: float = 0.0
    warmup_ms: float = 0.0
    steady_ms: float = 0.0
    cooldown_ms: float = 0.0
    dp_exposed_ms: float = 0.0
    optimizer_time_ms: float = 0.0
    optimizer_comm_ms: float = 0.0

    # Fwd/Bwd breakdown per phase (milliseconds)
    warmup_fwd_ms: float = 0.0
    warmup_bwd_ms: float = 0.0
    steady_fwd_ms: float = 0.0
    steady_bwd_ms: float = 0.0
    cooldown_fwd_ms: float = 0.0
    cooldown_bwd_ms: float = 0.0

    # Per-microbatch time in steady phase (milliseconds)
    steady_fwd_per_mb_ms: float = 0.0
    steady_bwd_per_mb_ms: float = 0.0
    steady_per_mb_ms: float = 0.0

    # Compute / comm breakdown (milliseconds)
    compute_time_ms: float = 0.0        # Pure compute on critical path
    fwd_compute_ms: float = 0.0         # Forward compute only (excludes all comm)
    bwd_compute_ms: float = 0.0         # Backward compute only (excludes all comm)
    exposed_comm_ms: float = 0.0        # Comm on critical path = Σ *_exposed_ms

    # Per-group exposed comm (Σ = exposed_comm_ms)
    tp_exposed_ms: float = 0.0          # TP RS/AG (after CoC/MC2 reduction)
    cp_exposed_ms: float = 0.0          # CP A2A
    ep_exposed_ms: float = 0.0          # EP A2A (after wave-overlap reduction)
    pp_exposed_ms: float = 0.0          # PP P2P
    # dp_exposed_ms declared above

    # Hidden comm — NOT on critical path
    hidden_comm_ms: float = 0.0         # Total hidden = Σ *_hidden_ms
    dp_hidden_ms: float = 0.0           # DP AR absorbed in pipeline bubble
    tp_hidden_ms: float = 0.0           # TP hidden by CoC/MC2
    ep_hidden_ms: float = 0.0           # EP hidden by wave-overlap

    # Total comm volume = exposed + hidden
    total_comm_volume_ms: float = 0.0

    # Config info
    config_summary: str | dict = ""  # [Stack B] uses str, [Stack A] uses dict
    warnings: list[str] = field(default_factory=list)  # [Stack A]

    # Model info
    total_params: int = 0  # [Stack B]

    # Derived metrics (graph-based FLOPs accounting)
    tokens_per_sec: float = 0.0      # Training throughput
    effective_params: int = 0        # P_eff used for MoE-aware accounting
    flops_per_token: float = 0.0     # Actual FLOPs consumed per token

    # Fused-operator summary (graph-native runs only).
    # Maps fused op_type → {count, sample_names, total_flops_pct, dtype, module_class}.
    fused_ops_summary: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Sync total_flops with training_flops for unified contract.

        When training_flops is set but total_flops is not (defaults to 0.0),
        copy training_flops to total_flops. This ensures both Stack A and Stack B
        callers see consistent FLOPs data regardless of which field they access.
        """
        # Sync total_flops ↔ training_flops (whichever is set, copy to the other)
        if self.total_flops == 0.0 and self.training_flops > 0:
            self.total_flops = self.training_flops
        elif self.training_flops == 0.0 and self.total_flops > 0:
            self.training_flops = self.total_flops

    def to_dict(self) -> dict:
        """Convert report to JSON-serializable dict."""
        result = {
            "step_time_ms": self.step_time_ms,
            "pipeline_time_ms": self.pipeline_time_ms,
            "mfu": self.mfu,
            "hfu": self.hfu,
            "total_flops": self.total_flops,
            "bubble_fraction": self.bubble_fraction,
            "schedule_name": self.schedule_name,
            "warmup_ms": self.warmup_ms,
            "steady_ms": self.steady_ms,
            "cooldown_ms": self.cooldown_ms,
            "dp_exposed_ms": self.dp_exposed_ms,
            "optimizer_time_ms": self.optimizer_time_ms,
            "optimizer_comm_ms": self.optimizer_comm_ms,
            "warmup_fwd_ms": self.warmup_fwd_ms,
            "warmup_bwd_ms": self.warmup_bwd_ms,
            "steady_fwd_ms": self.steady_fwd_ms,
            "steady_bwd_ms": self.steady_bwd_ms,
            "cooldown_fwd_ms": self.cooldown_fwd_ms,
            "cooldown_bwd_ms": self.cooldown_bwd_ms,
            "compute_time_ms": self.compute_time_ms,
            "fwd_compute_ms": self.fwd_compute_ms,
            "bwd_compute_ms": self.bwd_compute_ms,
            "exposed_comm_ms": self.exposed_comm_ms,
            "tp_exposed_ms": self.tp_exposed_ms,
            "cp_exposed_ms": self.cp_exposed_ms,
            "ep_exposed_ms": self.ep_exposed_ms,
            "pp_exposed_ms": self.pp_exposed_ms,
            "hidden_comm_ms": self.hidden_comm_ms,
            "dp_hidden_ms": self.dp_hidden_ms,
            "tp_hidden_ms": self.tp_hidden_ms,
            "ep_hidden_ms": self.ep_hidden_ms,
            "total_comm_volume_ms": self.total_comm_volume_ms,
        }

        # Add optional fields if present
        if self.per_stage_ms > 0:
            result["per_stage_ms"] = self.per_stage_ms
        if self.forward_flops > 0:
            result["forward_flops"] = self.forward_flops
        if self.backward_flops > 0:
            result["backward_flops"] = self.backward_flops
        if self.training_flops > 0:
            result["training_flops"] = self.training_flops
        if self.memory:
            result["memory_gb"] = self.memory.to_gb()
        if self.memory_breakdown:
            result["memory_breakdown_gb"] = {
                k: v / 1e9 for k, v in self.memory_breakdown.items()
            }
        if self.warmup_steps > 0:
            result["warmup_steps"] = self.warmup_steps
        if self.cooldown_steps > 0:
            result["cooldown_steps"] = self.cooldown_steps
        if self.steady_steps > 0:
            result["steady_steps"] = self.steady_steps
        if self.total_params > 0:
            result["total_params"] = self.total_params
        if self.effective_params > 0:
            result["effective_params"] = self.effective_params
        if self.flops_per_token > 0:
            result["flops_per_token"] = self.flops_per_token
        if self.tokens_per_sec > 0:
            result["tokens_per_sec"] = self.tokens_per_sec
        if self.fwd_compute_ms > 0:
            result["fwd_compute_ms"] = self.fwd_compute_ms
        if self.bwd_compute_ms > 0:
            result["bwd_compute_ms"] = self.bwd_compute_ms
        if self.warnings:
            result["warnings"] = self.warnings
        if self.fused_ops_summary:
            result["fused_ops_summary"] = self.fused_ops_summary

        # Config summary (handle both str and dict)
        if isinstance(self.config_summary, str):
            result["config_summary"] = self.config_summary
        elif isinstance(self.config_summary, dict):
            result["config_summary"] = self.config_summary

        return result

    def summary(self) -> str:
        """Return a compact, human-readable training report.

        Layout (one block, ~10 metric lines + a fused-operator table):

            Training Report — <config>
            ==========================
            Step:     1097.8 ms  (per-stage 32.9 ms)
            Util:     MFU 15.83%   HFU 15.83%
            FLOPs:    Total 5.31T  =  Fwd 2.36T + Bwd 2.96T
            Memory:   8.08 GB/GPU  (W 0.83 + G 0.83 + Opt 4.98 + Act 1.45)
            Pipeline: 1+32+1 microbatches, bubble 3.0%
            Params:   6.63 B

            Fused operators (top 12 by count):
              op_type                  count    module_class           dtype     ΣFLOPs    sample names
              rms_norm                    30    RMSNorm                bf16      ...       attn_norm, q_norm, kv_norm
              ...
        """
        config_str = self.config_summary
        if isinstance(self.config_summary, dict):
            config_str = ", ".join(f"{k}={v}" for k, v in self.config_summary.items())

        def _util(x: float) -> str:
            if x >= 1e-3:
                return f"{x:.2%}"
            if x > 0:
                return f"{x:.4%} ({x*1e6:.1f}ppm)"
            return "0.00%"

        def _fmt_t(flops: float) -> str:
            """Format FLOPs as G / T scale, no decimal noise."""
            if flops >= 1e12:
                return f"{flops/1e12:.2f}T"
            if flops >= 1e9:
                return f"{flops/1e9:.2f}G"
            if flops >= 1e6:
                return f"{flops/1e6:.2f}M"
            if flops > 0:
                return f"{flops/1e3:.2f}K"
            return "—"

        title = f"Training Report — {config_str}"
        lines: list[str] = [title, "=" * len(title)]

        # ── Step ──
        if self.per_stage_ms > 0:
            lines.append(f"Step:     {self.step_time_ms:.1f} ms  (per-stage {self.per_stage_ms:.1f} ms)")
        else:
            lines.append(f"Step:     {self.step_time_ms:.1f} ms")

        # ── Util ──
        lines.append(f"Util:     MFU {_util(self.mfu)}   HFU {_util(self.hfu)}")

        # ── FLOPs ──
        if self.training_flops > 0 or self.total_flops > 0:
            tot = self.training_flops or self.total_flops
            parts = []
            if self.forward_flops > 0:
                parts.append(f"Fwd {_fmt_t(self.forward_flops)}")
            if self.backward_flops > 0:
                parts.append(f"Bwd {_fmt_t(self.backward_flops)}")
            decomposition = "  =  " + " + ".join(parts) if parts else ""
            lines.append(f"FLOPs:    Total {_fmt_t(tot)}{decomposition}")

        # ── Memory ──
        if self.memory:
            mem_dict = self.memory.to_gb()
        elif self.memory_breakdown:
            mem_dict = {k: v / 1e9 for k, v in self.memory_breakdown.items()}
        else:
            mem_dict = {}
        if mem_dict:
            total = mem_dict.get("total")
            if total is None:
                total = sum(v for k, v in mem_dict.items() if k != "total")
            parts = []
            for key, label in [("weights", "W"), ("grads", "G"), ("opt_state", "Opt"),
                               ("activations", "Act"), ("comm_buffers", "Comm")]:
                v = mem_dict.get(key)
                if v is not None and v > 0:
                    parts.append(f"{label} {v:.2f}")
            tail = f"  ({' + '.join(parts)})" if parts else ""
            lines.append(f"Memory:   {total:.2f} GB/GPU{tail}")

        # ── Pipeline ──
        if self.warmup_steps > 0 or self.steady_steps > 0 or self.cooldown_steps > 0:
            lines.append(
                f"Pipeline: {self.warmup_steps}+{self.steady_steps}+{self.cooldown_steps} "
                f"microbatches, bubble {self.bubble_fraction:.1%}"
            )
        elif self.bubble_fraction > 0:
            lines.append(f"Pipeline: bubble {self.bubble_fraction:.1%}")

        # ── Params ──
        if self.total_params > 0:
            if self.total_params >= 1e9:
                lines.append(f"Params:   {self.total_params/1e9:.2f} B")
            else:
                lines.append(f"Params:   {self.total_params/1e6:.2f} M")

        # ── Warnings (Stack A) ──
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")

        # ── Fused operators (graph-native runs only) ──
        if self.fused_ops_summary:
            lines.append("")
            lines.append(f"Fused operators (top {min(12, len(self.fused_ops_summary))} by count):")
            header = f"  {'op_type':<24} {'count':>5}  {'module_class':<22} {'dtype':<6} {'ΣFLOPs':>9}  sample names"
            lines.append(header)
            sorted_ops = sorted(
                self.fused_ops_summary.items(),
                key=lambda kv: kv[1].get("count", 0),
                reverse=True,
            )
            for op_type, info in sorted_ops[:12]:
                count = info.get("count", 0)
                mc = info.get("module_class", "") or "—"
                dtype = info.get("dtype", "") or "—"
                total_flops = info.get("total_flops", 0) or 0
                samples = info.get("sample_names", []) or []
                samples_str = ", ".join(samples[:3])
                if len(samples) > 3:
                    samples_str += f", … (+{len(samples)-3})"
                lines.append(
                    f"  {op_type:<24} {count:>5}  {mc:<22} {dtype:<6} "
                    f"{_fmt_t(total_flops):>9}  {samples_str}"
                )

        return "\n".join(lines)
