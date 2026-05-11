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

        # Config summary (handle both str and dict)
        if isinstance(self.config_summary, str):
            result["config_summary"] = self.config_summary
        elif isinstance(self.config_summary, dict):
            result["config_summary"] = self.config_summary

        return result

    def summary(self) -> str:
        """Return a human-readable summary (Stack B format)."""
        # For Stack A reports with dict config_summary, convert to string
        config_str = self.config_summary
        if isinstance(self.config_summary, dict):
            config_str = ", ".join(f"{k}={v}" for k, v in self.config_summary.items())

        def _fmt_util(x: float) -> str:
            # Avoid round-down-to-zero on uncalibrated / very-low utilisation:
            # show full precision below 0.1%, percentage otherwise.
            if x >= 1e-3:
                return f"{x:.2%}"
            if x > 0:
                return f"{x:.4%}  ({x*1e6:.1f} ppm)"
            return "0.00%"

        lines = [
            "Training Estimation Report",
            "=" * 40,
            f"Config: {config_str}",
            "",
            "Timing:",
            f"  Step time: {self.step_time_ms:.2f} ms",
            f"  Per-stage: {self.per_stage_ms:.2f} ms" if self.per_stage_ms > 0 else "",
            "",
            "Efficiency:",
            f"  MFU: {_fmt_util(self.mfu)}",
            f"  HFU: {_fmt_util(self.hfu)}",
            "",
            "FLOPs:",
        ]

        if self.training_flops > 0:
            lines.append(f"  Training: {self.training_flops/1e12:.2f} TFLOPs")
        if self.forward_flops > 0:
            lines.append(f"  Forward: {self.forward_flops/1e12:.2f} TFLOPs")
        if self.backward_flops > 0:
            lines.append(f"  Backward: {self.backward_flops/1e12:.2f} TFLOPs")

        lines.extend([
            "",
            "Memory (per GPU):",
        ])

        if self.memory:
            for k, v in self.memory.to_gb().items():
                lines.append(f"  {k}: {v:.2f} GB")
        elif self.memory_breakdown:
            for k, v in self.memory_breakdown.items():
                lines.append(f"  {k}: {v/1e9:.2f} GB")

        lines.extend([
            "",
            "Pipeline:",
            f"  Warmup steps: {self.warmup_steps}" if self.warmup_steps > 0 else "",
            f"  Steady steps: {self.steady_steps}" if self.steady_steps > 0 else "",
            f"  Cooldown steps: {self.cooldown_steps}" if self.cooldown_steps > 0 else "",
            f"  Bubble fraction: {self.bubble_fraction:.1%}",
            "",
            f"Total params: {self.total_params/1e9:.2f}B" if self.total_params > 0 else "",
        ])

        # Filter out empty lines
        return "\n".join(line for line in lines if line)
