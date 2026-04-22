"""Training analysis passes: FLOPs (6P rule), Memory (ZeRO), Pipeline (1F1B)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext


# ── TrainingFlopsPass ───────────────────────────────────────────────────────────

class TrainingFlopsPass(GraphPass):
    """Annotate graph with training FLOPs using the 6P rule.

    Training FLOPs = 6 * total_params * tokens (for dense transformers)
    Forward: 2 * params * tokens, Backward: 4 * params * tokens

    Adds to graph.metadata:
      "training_flops": float
      "forward_flops": float
      "backward_flops": float
      "total_params": int
    """

    name = "training_flops"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        g = graph.clone()

        # Estimate total params from weight tensors in the graph
        total_params = 0
        for node in g.nodes.values():
            if node.category == "compute":
                # Count parameters from weight inputs (model parameters)
                # Heuristic: inputs with "weight" in the id or with large 2D shapes
                for inp in node.inputs:
                    if ("weight" in inp.id or "param" in inp.id) and inp.shape:
                        total_params += math.prod(inp.shape)

        # Get sequence length and batch size from metadata
        seq_len = g.metadata.get("seq_len", 2048)
        batch_size = ctx.training.micro_batch if ctx.training else 1
        tokens = seq_len * batch_size

        # 6P rule: 2 params * tokens (forward) + 4 params * tokens (backward)
        forward_flops = 2 * total_params * tokens
        backward_flops = 4 * total_params * tokens
        training_flops = forward_flops + backward_flops

        g.metadata["training_flops"] = training_flops
        g.metadata["forward_flops"] = forward_flops
        g.metadata["backward_flops"] = backward_flops
        g.metadata["total_params"] = total_params

        return g


# ── TrainingMemoryPass ──────────────────────────────────────────────────────────

@dataclass
class TrainingMemoryBreakdown:
    """Training memory breakdown per GPU."""
    weights: float = 0.0      # Model weights (bytes)
    grads: float = 0.0        # Gradients (bytes)
    opt_state: float = 0.0    # Optimizer state (bytes)
    activations: float = 0.0  # Activations (bytes)
    comm_buffers: float = 0.0 # Communication buffers (bytes)

    @property
    def total(self) -> float:
        return self.weights + self.grads + self.opt_state + self.activations + self.comm_buffers

    def to_dict(self) -> dict[str, float]:
        return {
            "weights": self.weights,
            "grads": self.grads,
            "opt_state": self.opt_state,
            "activations": self.activations,
            "comm_buffers": self.comm_buffers,
            "total": self.total,
        }


class TrainingMemoryPass(GraphPass):
    """Annotate graph with training memory breakdown using ZeRO sharding.

    ZeRO stages:
      - 0: No sharding (all replicas store full weights, grads, opt_state)
      - 1: Optimizer state sharded across DP
      - 2: Gradients + optimizer state sharded across DP
      - 3: Weights + gradients + optimizer state sharded across DP

    Adds to graph.metadata:
      "memory_breakdown": TrainingMemoryBreakdown
    """

    name = "training_memory"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        g = graph.clone()

        # Get dtype bytes
        param_dtype = 2 if ctx.quant and ctx.quant.weight == "bf16" else 4  # BF16 or FP32

        # Count parameters
        total_params = 0
        for node in g.nodes.values():
            if node.category == "compute":
                for inp in node.inputs:
                    # Heuristic: inputs with "weight" in the id or with large 2D shapes
                    if ("weight" in inp.id or "param" in inp.id) and inp.shape:
                        total_params += math.prod(inp.shape)

        # Get parallel config
        dp = ctx.parallel.dp if ctx.parallel else 1
        tp = ctx.parallel.tp if ctx.parallel else 1
        zero_stage = ctx.training.zero_stage if ctx.training else 0

        # Weights: sharded by TP, optionally by ZeRO-3
        weights_sharding = tp
        if zero_stage >= 3:
            weights_sharding *= dp
        weights_bytes = (total_params * param_dtype) / weights_sharding

        # Gradients: same as weights, optionally sharded by ZeRO-2/3
        grads_sharding = weights_sharding
        if zero_stage >= 2 and zero_stage < 3:
            grads_sharding = dp  # ZeRO-2 shards grads but not weights
        grads_bytes = (total_params * param_dtype) / grads_sharding

        # Optimizer state: Adam uses 2 states per param (momentum + variance)
        # Sharded by ZeRO-1/2/3
        opt_sharding = 1
        if zero_stage >= 1:
            opt_sharding = dp
        opt_bytes = (total_params * param_dtype * 2) / opt_sharding

        # Activations: estimate using Korthikanti formula
        seq_len = g.metadata.get("seq_len", 2048)
        hidden = g.metadata.get("hidden", 4096)
        num_layers = g.metadata.get("num_layers", 32)
        batch_size = ctx.training.micro_batch if ctx.training else 1

        # Korthikanti formula: 10 * hidden * seq_len * num_layers * batch_size / tp
        # Factor of 10 accounts for attention + FFN activations
        activations_bytes = (10 * hidden * seq_len * num_layers * batch_size) / tp

        # Communication buffers: AG/RS buffers
        # Approx: 2 * hidden * seq_len / tp per layer
        comm_bytes = (2 * hidden * seq_len * num_layers) / tp

        breakdown = TrainingMemoryBreakdown(
            weights=weights_bytes,
            grads=grads_bytes,
            opt_state=opt_bytes,
            activations=activations_bytes,
            comm_buffers=comm_bytes,
        )

        g.metadata["memory_breakdown"] = breakdown

        return g


# ── TrainingPipelinePass ────────────────────────────────────────────────────────

@dataclass
class PipelineStepMetrics:
    """Pipeline step metrics for 1F1B schedule."""
    step_time_ms: float = 0.0
    per_stage_ms: float = 0.0
    warmup_steps: int = 0
    cooldown_steps: int = 0
    steady_steps: int = 0
    bubble_fraction: float = 0.0
    mfu: float = 0.0  # Model FLOPs Utilization

    def to_dict(self) -> dict[str, float]:
        return {
            "step_time_ms": self.step_time_ms,
            "per_stage_ms": self.per_stage_ms,
            "warmup_steps": self.warmup_steps,
            "cooldown_steps": self.cooldown_steps,
            "steady_steps": self.steady_steps,
            "bubble_fraction": self.bubble_fraction,
            "mfu": self.mfu,
        }


class TrainingPipelinePass(GraphPass):
    """Annotate graph with pipeline schedule metrics (1F1B).

    1F1B schedule:
      - Warmup: PP-1 steps (ramp up microbatches)
      - Steady: M - PP + 1 steps (full pipeline utilization)
      - Cooldown: PP-1 steps (drain microbatches)

    Bubble fraction = (warmup + cooldown) / total_steps

    Adds to graph.metadata:
      "pipeline_metrics": PipelineStepMetrics
    """

    name = "training_pipeline"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        g = graph.clone()

        # Get config
        pp = ctx.parallel.pp if ctx.parallel else 1
        num_microbatches = ctx.training.num_microbatches if ctx.training else 1
        hw = ctx.hw_spec

        # Estimate per-stage time from node latencies
        stage_time_us = 0.0
        num_stages = pp

        # Sum up latencies for a single forward pass
        for node in g.topo_sort():
            if node.category == "compute":
                stage_time_us += node.annotations.get("latency_us", 0.0)
            elif node.category == "communication":
                stage_time_us += node.annotations.get("latency_us", 0.0)

        # Divide by PP for per-stage time (rough approximation)
        per_stage_us = stage_time_us / num_stages if num_stages > 0 else stage_time_us

        # 1F1B schedule
        warmup_steps = max(0, pp - 1)
        cooldown_steps = max(0, pp - 1)
        steady_steps = max(0, num_microbatches - pp + 1)

        # Total step time (for one global batch)
        step_time_us = per_stage_us * (warmup_steps + num_microbatches + cooldown_steps)
        step_time_ms = step_time_us / 1000.0
        per_stage_ms = per_stage_us / 1000.0

        # Bubble fraction
        total_steps = warmup_steps + num_microbatches + cooldown_steps
        bubble_fraction = (warmup_steps + cooldown_steps) / total_steps if total_steps > 0 else 0.0

        # Compute MFU
        training_flops = g.metadata.get("training_flops", 0.0)
        world_size = ctx.parallel.total_devices if ctx.parallel else 1

        # Peak FLOPs per GPU
        from python.zrt.ir.types import DType
        dtype = DType.BF16
        peak_flops_per_gpu = hw.peak_flops(dtype)

        # MFU = (training_flops / step_time) / (world_size * peak_flops_per_gpu)
        step_time_sec = step_time_us / 1e6
        achieved_flops = training_flops / step_time_sec if step_time_sec > 0 else 0.0
        peak_flops_total = world_size * peak_flops_per_gpu
        mfu = achieved_flops / peak_flops_total if peak_flops_total > 0 else 0.0

        metrics = PipelineStepMetrics(
            step_time_ms=step_time_ms,
            per_stage_ms=per_stage_ms,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            steady_steps=steady_steps,
            bubble_fraction=bubble_fraction,
            mfu=mfu,
        )

        g.metadata["pipeline_metrics"] = metrics

        return g
