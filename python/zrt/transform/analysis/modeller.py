"""Training modeller: main entry point for training performance estimation.

Two usage modes:

1. **From an existing OpGraph** (lower-level)::

       from python.zrt.transform.analysis import estimate_training
       report = estimate_training(graph, ctx)

2. **From a model directory** (end-to-end, captures graph then estimates)::

       from python.zrt.transform.analysis import model_training
       report = model_training(
           model_id="hf_models/deepseek_v3_2",
           num_layers=4,
           seq_len=128,
           hw_spec=my_hardware_spec,
           total_params=671e9,
           tp=8, pp=4, dp=2,
           micro_batch=1, global_batch=8192,
       )
       print(report.summary())
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
    hfu: float = 0.0  # Hardware FLOPs Utilization (includes recompute)

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
            "hfu": self.hfu,
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
            f"  HFU: {self.hfu:.1%}",
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
    """
    from python.zrt.transform.pipeline import build_training_pipeline

    pipe = build_training_pipeline()
    g = pipe.run(graph, ctx)

    pipeline_metrics = g.metadata.get("pipeline_metrics")
    memory_breakdown = g.metadata.get("memory_breakdown")

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

    return TrainingReport(
        config_summary=config_summary,
        step_time_ms=pipeline_metrics.step_time_ms if pipeline_metrics else 0.0,
        per_stage_ms=pipeline_metrics.per_stage_ms if pipeline_metrics else 0.0,
        mfu=pipeline_metrics.mfu if pipeline_metrics else 0.0,
        hfu=pipeline_metrics.hfu if pipeline_metrics else 0.0,
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


def estimate_training_from_graphs(
    *,
    forward_graph: "OpGraph",
    backward_graph: "OpGraph | None" = None,
    hw_spec: "HardwareSpec | None" = None,
    total_params: int | None = None,
    hidden: int = 7168,
    num_layers: int = 4,
    num_layers_full: int | None = None,
    seq_len: int = 128,
    batch_size: int = 1,
    tp: int = 1, pp: int = 1, ep: int = 1, dp: int = 1, cp: int = 1,
    zero_stage: int = 1,
    optimizer: str = "adam",
    micro_batch: int = 1,
    global_batch: int = 32,
    pp_schedule: str = "1f1b",
    vpp_chunks: int = 1,
) -> TrainingReport:
    """Estimate training performance from pre-built OpGraph instances.

    Takes already-captured forward and backward computation graphs and
    runs the training analysis pipeline. Use this when the graphs have
    already been captured by ``run_trace_phases``.
    """
    from python.zrt.transform.context import ParallelConfig, TrainingConfig, TransformContext
    from python.zrt.transform.pipeline import build_training_pipeline

    metadata: dict = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_layers": num_layers_full or num_layers,
        "num_layers_traced": num_layers,
        "hidden": hidden,
    }
    if total_params is not None:
        metadata["total_params"] = int(total_params)

    for key, val in metadata.items():
        if key not in forward_graph.metadata:
            forward_graph.metadata[key] = val
    if backward_graph is not None:
        for key, val in metadata.items():
            if key not in backward_graph.metadata:
                backward_graph.metadata[key] = val

    ctx = TransformContext(
        hw_spec=hw_spec,
        parallel=ParallelConfig(tp=tp, pp=pp, ep=ep, dp=dp, cp=cp),
        training=TrainingConfig(
            optimizer=optimizer,
            zero_stage=zero_stage,
            micro_batch=micro_batch,
            global_batch=global_batch,
            pp_schedule=pp_schedule,
            vpp_chunks=vpp_chunks,
        ),
    )

    pipe = build_training_pipeline()
    results: dict[str, "OpGraph"] = {}

    if backward_graph is not None:
        from python.zrt.ir.adapter import stitch_fwd_bwd
        unified = stitch_fwd_bwd(forward_graph, backward_graph)
        for key, val in metadata.items():
            if key not in unified.metadata:
                unified.metadata[key] = val
        results["unified"] = pipe.run(unified, ctx)
    else:
        results["train_forward"] = pipe.run(forward_graph, ctx)

    if "unified" in results:
        g = results["unified"]
        pipeline_metrics = g.metadata.get("pipeline_metrics")
        memory_breakdown = g.metadata.get("memory_breakdown")
        training_flops = g.metadata.get("training_flops", 0.0)
        forward_flops = g.metadata.get("forward_flops", 0.0)
        backward_flops = g.metadata.get("backward_flops", 0.0)
        total_params = g.metadata.get("total_params", 0)
    else:
        fwd = results["train_forward"]
        pipeline_metrics = fwd.metadata.get("pipeline_metrics")

        memory_breakdown = fwd.metadata.get("memory_breakdown")
        training_flops = fwd.metadata.get("training_flops", 0.0)
        forward_flops = fwd.metadata.get("forward_flops", 0.0)
        backward_flops = fwd.metadata.get("backward_flops", 0.0)
        total_params = fwd.metadata.get("total_params", 0)

    step_time_ms = pipeline_metrics.step_time_ms if pipeline_metrics else 0.0
    per_stage_ms = pipeline_metrics.per_stage_ms if pipeline_metrics else 0.0
    mfu = pipeline_metrics.mfu if pipeline_metrics else 0.0
    hfu = pipeline_metrics.hfu if pipeline_metrics else 0.0
    warmup_steps = pipeline_metrics.warmup_steps if pipeline_metrics else 0
    cooldown_steps = pipeline_metrics.cooldown_steps if pipeline_metrics else 0
    steady_steps = pipeline_metrics.steady_steps if pipeline_metrics else 0
    bubble_fraction = pipeline_metrics.bubble_fraction if pipeline_metrics else 0.0

    parallel = ctx.parallel
    training = ctx.training
    config_parts: list[str] = []
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

    return TrainingReport(
        config_summary=config_summary,
        step_time_ms=step_time_ms,
        per_stage_ms=per_stage_ms,
        mfu=mfu,
        hfu=hfu,
        training_flops=training_flops,
        forward_flops=forward_flops,
        backward_flops=backward_flops,
        memory_breakdown=memory_breakdown.to_dict() if memory_breakdown else {},
        warmup_steps=warmup_steps,
        cooldown_steps=cooldown_steps,
        steady_steps=steady_steps,
        bubble_fraction=bubble_fraction,
        total_params=total_params,
    )


def model_training(
    model_id: str,
    num_layers: int = 4,
    batch_size: int = 1,
    seq_len: int = 128,
    total_params: int | None = None,
    hidden: int = 7168,
    num_layers_full: int | None = None,
    hw_spec: "HardwareSpec | None" = None,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    dp: int = 1,
    cp: int = 1,
    zero_stage: int = 1,
    optimizer: str = "adam",
    micro_batch: int = 1,
    global_batch: int = 32,
    pp_schedule: str = "1f1b",
    vpp_chunks: int = 1,
    output_dir: str | None = None,
) -> TrainingReport:
    """Capture a model graph and estimate training performance end to end."""
    from python.zrt.graph import run_trace_phases
    from python.zrt.ir.adapter import records_to_opgraph

    _, phase_records = run_trace_phases(
        model_id=model_id,
        num_layers=num_layers,
        batch_size=batch_size,
        seq_len=seq_len,
        phases=("train_forward", "train_backward"),
        output_dir=output_dir,
    )

    metadata: dict = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_layers": num_layers_full or num_layers,
        "num_layers_traced": num_layers,
        "hidden": hidden,
    }
    if total_params is not None:
        metadata["total_params"] = int(total_params)

    fwd_graph: "OpGraph | None" = None
    bwd_graph: "OpGraph | None" = None
    for phase in ("train_forward", "train_backward"):
        records = phase_records.get(phase, [])
        if records:
            g = records_to_opgraph(
                records=records,
                name=f"{model_id.replace('/', '_')}_{phase}",
                phase=phase,
                metadata={**metadata},
            )
            if phase == "train_forward":
                fwd_graph = g
            else:
                bwd_graph = g

    if fwd_graph is None:
        raise ValueError("train_forward phase produced no records")

    return estimate_training_from_graphs(
        forward_graph=fwd_graph,
        backward_graph=bwd_graph,
        hw_spec=hw_spec,
        total_params=total_params,
        hidden=hidden,
        num_layers=num_layers,
        num_layers_full=num_layers_full,
        seq_len=seq_len,
        batch_size=batch_size,
        tp=tp, pp=pp, ep=ep, dp=dp, cp=cp,
        zero_stage=zero_stage,
        optimizer=optimizer,
        micro_batch=micro_batch,
        global_batch=global_batch,
        pp_schedule=pp_schedule,
        vpp_chunks=vpp_chunks,
    )
