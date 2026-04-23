"""Training analysis passes: FLOPs (6P rule), Memory (ZeRO), Pipeline (1F1B)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from python.zrt.ir.param_count import count_params
from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext


# ── TrainingFlopsPass ───────────────────────────────────────────────────────────

class TrainingFlopsPass(GraphPass):
    """Annotate graph with training FLOPs.

    Strategy (priority order):
    1. Per-node annotations: sum flops_fwd / flops_dx / flops_dw from
       ``TrainFlopsPass`` when available (more accurate for MoE, comm, etc.)
    2. 6P rule fallback: 6 * total_params * tokens (dense transformers)

    Adds to graph.metadata:
      "training_flops": float
      "forward_flops": float
      "backward_flops": float
      "total_params": int
    """

    name = "training_flops"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        g = graph.clone()

        has_param_override = g.metadata.get("total_params", 0) > 0

        total_params = count_params(g)

        num_layers = g.metadata.get("num_layers", 0)
        num_layers_traced = g.metadata.get("num_layers_traced", num_layers)
        layer_scale = num_layers / num_layers_traced if num_layers_traced > 0 and num_layers != num_layers_traced else 1.0

        if not has_param_override and layer_scale != 1.0:
            total_params = int(total_params * layer_scale)

        # ── Try per-node annotation path ────────────────────────────────────
        forward_flops = sum(
            n.annotations.get("flops_fwd", 0) for n in g.nodes.values()
        )
        backward_flops = sum(
            n.annotations.get("flops_dx", 0) + n.annotations.get("flops_dw", 0)
            for n in g.nodes.values()
        )

        if forward_flops > 0 or backward_flops > 0:
            # Per-node annotations available — scale to full model
            if layer_scale != 1.0:
                forward_flops = int(forward_flops * layer_scale)
                backward_flops = int(backward_flops * layer_scale)
            training_flops = forward_flops + backward_flops
        else:
            # Fallback: 6P rule
            seq_len = g.metadata.get("seq_len", 2048)
            batch_size = ctx.training.micro_batch if ctx.training else 1
            tokens = seq_len * batch_size
            forward_flops = 2 * total_params * tokens
            backward_flops = 4 * total_params * tokens
            training_flops = forward_flops + backward_flops

        g.metadata["training_flops"] = training_flops
        g.metadata["forward_flops"] = forward_flops
        g.metadata["backward_flops"] = backward_flops
        g.metadata["total_params"] = total_params
        g.metadata["layer_scale"] = layer_scale

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

    Activation memory uses the Korthikanti formula (34 * h * s * L * bs)
    with recompute-policy multiplier, CP sharding, and PP inflight depth.

    When ``g.metadata["zero"]`` is present (written by ZeroFSDPPass),
    weight/grad/opt-state sharding factors are read from it; otherwise
    self-derived from ZeRO stage + DP/TP.

    Adds to graph.metadata:
      "memory_breakdown": TrainingMemoryBreakdown
    """

    name = "training_memory"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        g = graph.clone()

        param_dtype = 2  # BF16

        total_params = count_params(g)

        dp = ctx.parallel.dp if ctx.parallel else 1
        tp = ctx.parallel.tp if ctx.parallel else 1
        cp = getattr(ctx.parallel, "cp", 1) if ctx.parallel else 1
        pp = ctx.parallel.pp if ctx.parallel else 1
        zero_stage = ctx.training.zero_stage if ctx.training else 0

        # ── Weight / grad / opt-state sharding ──────────────────────────────
        zero_meta = g.metadata.get("zero")
        if zero_meta:
            weight_shard = zero_meta["weight_shard"] * tp
            grad_shard = zero_meta["grad_shard"] * tp
            opt_shard = zero_meta["optstate_shard"]
        else:
            weight_shard = tp
            if zero_stage >= 3:
                weight_shard *= dp
            grad_shard = weight_shard
            if zero_stage >= 2 and zero_stage < 3:
                grad_shard = dp
            opt_shard = dp if zero_stage >= 1 else 1

        weights_bytes = (total_params * param_dtype) / weight_shard
        grads_bytes = (total_params * param_dtype) / grad_shard

        optimizer = ctx.training.optimizer if ctx.training else "adam"
        opt_bytes_per_param = 12 if optimizer in ("adam", "adamw") else 8
        opt_bytes = (total_params * opt_bytes_per_param) / opt_shard

        # ── Activation memory (Korthikanti) ─────────────────────────────────
        seq_len = g.metadata.get("seq_len", 2048)
        hidden = g.metadata.get("hidden", 4096)
        num_layers = g.metadata.get("num_layers", 32)
        batch_size = ctx.training.micro_batch if ctx.training else 1

        # Korthikanti upper bound: 34 * hidden * seq_len * num_layers * batch_size
        base = 34 * hidden * seq_len * num_layers * batch_size

        # Parallel sharding: TP cuts hidden/attn; CP cuts seq (if cp > 1)
        shard = tp * max(cp, 1)

        # Recompute policy multiplier
        rc = getattr(ctx.training, "recompute_policy", "selective") if ctx.training else "selective"
        rc_mult = {"none": 1.0, "selective": 0.5, "full": 0.1}.get(rc, 1.0)

        # PP inflight depth: 1F1B steady state, stage 0 holds pp microbatches
        # Phase 2 will replace with (pp - stage_id); Phase 1 uses conservative pp
        inflight = pp

        activations_bytes = (base / shard) * rc_mult * inflight

        # Communication buffers: AG/RS buffers
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

    Correct 1F1B schedule (homogeneous stages):
      step_time = (M + pp - 1) * t_stage
      bubble_fraction = (pp - 1) / (M + pp - 1)

    where M = num_microbatches, t_stage = per-stage latency.

    Adds to graph.metadata:
      "pipeline_metrics": PipelineStepMetrics
    """

    name = "training_pipeline"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        g = graph.clone()

        pp = ctx.parallel.pp if ctx.parallel else 1
        num_microbatches = ctx.training.num_microbatches if ctx.training else 1
        hw = ctx.hw_spec
        layer_scale = g.metadata.get("layer_scale", 1.0)

        from python.zrt.executor.scheduler import DAGScheduler
        sched = DAGScheduler(hw)

        if pp > 1 and any("stage_id" in n.annotations for n in g.nodes.values()):
            # Per-stage scheduling: schedule each stage's subgraph independently
            stage_node_sets: dict[int, set[str]] = {}
            for node in g.nodes.values():
                sid = node.annotations.get("stage_id", 0)
                stage_node_sets.setdefault(sid, set()).add(node.id)

            stage_latencies: dict[int, float] = {}
            for s_id in range(pp):
                node_ids = stage_node_sets.get(s_id, set())
                if not node_ids:
                    stage_latencies[s_id] = 0.0
                    continue
                sub = g.subgraph(node_ids)
                tl = sched.schedule(sub)
                lat = tl.total_latency_us
                if layer_scale != 1.0:
                    lat *= layer_scale
                stage_latencies[s_id] = lat

            g.metadata["stage_timelines_fwd"] = dict(stage_latencies)

            # Bottleneck stage (max per-stage fwd latency)
            per_stage_us = max(stage_latencies.values(), default=0.0)
        else:
            # pp=1 (or stage_id not yet annotated): whole graph is scheduled as one
            # unit. total_latency_us covers all pp stages linearly, so divide by pp.
            tl = sched.schedule(g)
            per_stage_us = tl.total_latency_us / pp
            if layer_scale != 1.0:
                per_stage_us *= layer_scale

        # 1F1B schedule formulas (standard):
        #   step_time = (M + pp - 1) * t_stage
        #   bubble     = (pp - 1) / (M + pp - 1)
        #   steady_steps = M  (the M back-to-back 1F1B microbatches)
        warmup_steps = max(0, pp - 1)
        cooldown_steps = max(0, pp - 1)
        steady_steps = max(0, num_microbatches - pp + 1)
        effective_steps = num_microbatches + pp - 1

        step_time_us = per_stage_us * effective_steps
        step_time_ms = step_time_us / 1000.0
        per_stage_ms = per_stage_us / 1000.0

        bubble_fraction = (pp - 1) / effective_steps if effective_steps > 0 else 0.0

        training_flops = g.metadata.get("training_flops", 0.0)
        world_size = ctx.parallel.total_devices if ctx.parallel else 1

        from python.zrt.ir.types import DType
        peak_flops_per_gpu = hw.peak_flops(DType.BF16)

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
