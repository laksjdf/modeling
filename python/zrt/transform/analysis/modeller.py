"""Training modeller: estimate training performance from captured computation graphs.

Usage::

    from python.zrt.transform.analysis import estimate_training_from_graphs
    report = estimate_training_from_graphs(
        forward_graph=fwd, backward_graph=bwd,
        hw_spec=hw, tp=8, pp=4, dp=2, ...
    )
    print(report.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext

# Import shared TrainingReport type (canonical import path)
from zrt.training.spec.report import TrainingReport


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
    return_transformed: bool = False,
    quant: str | None = None,
) -> "TrainingReport | tuple[TrainingReport, TransformContext, dict[str, OpGraph]]":
    """Estimate training performance from pre-built OpGraph instances.

    Takes already-captured forward and backward computation graphs and
    runs the training analysis pipeline. Use this when the graphs have
    already been captured by ``run_trace_phases``.

    Parameters
    ----------
    return_transformed : bool, default False
        If True, return (TrainingReport, TransformContext, transformed_graphs)
        where transformed_graphs contains the pipeline-processed graphs.
        This enables downstream Excel export via ``export_training_graphs``.
    """
    from python.zrt.transform.context import ParallelConfig, QuantConfig, TrainingConfig, TransformContext
    from python.zrt.transform.pipeline import build_default_pipeline

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

    quant_cfg = QuantConfig(weight=quant, activation=quant) if quant else None
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
        quant=quant_cfg,
    )

    pipe = build_default_pipeline()
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

    report = TrainingReport(
        config_summary=config_summary,
        step_time_ms=step_time_ms,
        per_stage_ms=per_stage_ms,
        mfu=mfu,
        hfu=hfu,
        total_flops=training_flops,  # Alias for Stack A compatibility
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

    if return_transformed:
        return report, ctx, results
    return report
