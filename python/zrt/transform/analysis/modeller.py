"""Training modeller: main entry point for training performance estimation.

This module provides the estimate_training() function which runs training-specific
analysis passes on an OpGraph and returns training performance metrics.

Usage:
    from python.zrt.transform.analysis import estimate_training
    from python.zrt.transform.context import TransformContext, TrainingConfig

    ctx = TransformContext(
        hw_spec=my_hardware_spec,
        training=TrainingConfig(
            optimizer="adam",
            zero_stage=1,
            micro_batch=1,
            global_batch=32,
        ),
    )
    result = estimate_training(graph, ctx)
    print(f"Step time: {result.step_time_ms:.2f} ms")
    print(f"MFU: {result.mfu:.1%}")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext


@dataclass
class TrainingReport:
    """Training performance estimation report."""

    # Config summary
    config_summary: str = ""

    # Timing metrics
    step_time_ms: float = 0.0
    per_stage_ms: float = 0.0

    # Efficiency metrics
    mfu: float = 0.0  # Model FLOPs Utilization

    # FLOPs breakdown
    training_flops: float = 0.0
    forward_flops: float = 0.0
    backward_flops: float = 0.0

    # Memory breakdown (per GPU)
    memory_breakdown: dict[str, float] = field(default_factory=dict)

    # Pipeline metrics
    warmup_steps: int = 0
    cooldown_steps: int = 0
    steady_steps: int = 0
    bubble_fraction: float = 0.0

    # Model info
    total_params: int = 0

    def to_dict(self) -> dict:
        """Convert report to JSON-serializable dict."""
        return {
            "config_summary": self.config_summary,
            "step_time_ms": self.step_time_ms,
            "per_stage_ms": self.per_stage_ms,
            "mfu": self.mfu,
            "training_flops": self.training_flops,
            "forward_flops": self.forward_flops,
            "backward_flops": self.backward_flops,
            "memory_breakdown_gb": {
                k: v / 1e9 for k, v in self.memory_breakdown.items()
            },
            "warmup_steps": self.warmup_steps,
            "cooldown_steps": self.cooldown_steps,
            "steady_steps": self.steady_steps,
            "bubble_fraction": self.bubble_fraction,
            "total_params": self.total_params,
        }

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "Training Estimation Report",
            "=" * 40,
            f"Config: {self.config_summary}",
            "",
            "Timing:",
            f"  Step time: {self.step_time_ms:.2f} ms",
            f"  Per-stage: {self.per_stage_ms:.2f} ms",
            "",
            "Efficiency:",
            f"  MFU: {self.mfu:.1%}",
            "",
            "FLOPs:",
            f"  Training: {self.training_flops/1e12:.2f} TFLOPs",
            f"  Forward: {self.forward_flops/1e12:.2f} TFLOPs",
            f"  Backward: {self.backward_flops/1e12:.2f} TFLOPs",
            "",
            "Memory (per GPU):",
        ]
        for k, v in self.memory_breakdown.items():
            lines.append(f"  {k}: {v/1e9:.2f} GB")
        lines.extend([
            "",
            "Pipeline:",
            f"  Warmup steps: {self.warmup_steps}",
            f"  Steady steps: {self.steady_steps}",
            f"  Cooldown steps: {self.cooldown_steps}",
            f"  Bubble fraction: {self.bubble_fraction:.1%}",
            "",
            f"Total params: {self.total_params/1e9:.2f}B",
        ])
        return "\n".join(lines)


def estimate_training(
    graph: "OpGraph",
    ctx: "TransformContext",
) -> TrainingReport:
    """Estimate training performance metrics.

    This function runs training-specific analysis passes on the graph
    and returns a comprehensive training performance report.

    Parameters
    ----------
    graph : OpGraph
        The computation graph (typically a forward pass graph)
    ctx : TransformContext
        Transform context with training configuration (ctx.training must be set)

    Returns
    -------
    TrainingReport
        Training performance estimation report

    Examples
    --------
    >>> from python.zrt.transform.context import TransformContext, TrainingConfig
    >>> ctx = TransformContext(
    ...     hw_spec=my_hw,
    ...     training=TrainingConfig(optimizer="adam", zero_stage=1, ...),
    ... )
    >>> report = estimate_training(graph, ctx)
    >>> print(report.summary())
    """
    from .training import TrainingFlopsPass, TrainingMemoryPass, TrainingPipelinePass  # noqa: F401

    # Run training analysis passes
    flops_pass = TrainingFlopsPass()
    memory_pass = TrainingMemoryPass()
    pipeline_pass = TrainingPipelinePass()

    g = flops_pass.run(graph, ctx)
    g = memory_pass.run(g, ctx)
    g = pipeline_pass.run(g, ctx)

    # Extract metrics from graph metadata
    pipeline_metrics = g.metadata.get("pipeline_metrics")
    memory_breakdown = g.metadata.get("memory_breakdown")

    # Build config summary
    parallel = ctx.parallel
    training = ctx.training
    config_parts = []
    if parallel.tp > 1:
        config_parts.append(f"TP{parallel.tp}")
    if parallel.pp > 1:
        config_parts.append(f"PP{parallel.pp}")
    if parallel.ep > 1:
        config_parts.append(f"EP{parallel.ep}")
    if parallel.dp > 1:
        config_parts.append(f"DP{parallel.dp}")
    if training:
        config_parts.append(f"ZeRO-{training.zero_stage}")
        config_parts.append(f"{training.optimizer}")
        config_parts.append(f"micro{training.micro_batch}")

    config_summary = "-".join(config_parts) if config_parts else "default"

    # Build report
    report = TrainingReport(
        config_summary=config_summary,
        step_time_ms=pipeline_metrics.step_time_ms if pipeline_metrics else 0.0,
        per_stage_ms=pipeline_metrics.per_stage_ms if pipeline_metrics else 0.0,
        mfu=pipeline_metrics.mfu if pipeline_metrics else 0.0,
        training_flops=g.metadata.get("training_flops", 0.0),
        forward_flops=g.metadata.get("forward_flops", 0.0),
        backward_flops=g.metadata.get("backward_flops", 0.0),
        memory_breakdown=memory_breakdown.to_dict() if memory_breakdown else {},
        warmup_steps=pipeline_metrics.warmup_steps if pipeline_metrics else 0,
        cooldown_steps=pipeline_metrics.cooldown_steps if pipeline_metrics else 0,
        steady_steps=pipeline_metrics.steady_steps if pipeline_metrics else 0,
        bubble_fraction=pipeline_metrics.bubble_fraction if pipeline_metrics else 0.0,
        total_params=g.metadata.get("total_params", 0),
    )

    return report
