"""Training analysis passes: FLOPs, Memory, and pipeline scheduling."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from python.zrt.ir.param_count import count_params
from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext

logger = logging.getLogger(__name__)


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

        # Recompute overhead: for nodes with recompute annotation, flops_fwd
        # already includes 2x multiplier (flops_train.py:37), so base fwd = flops_fwd / 2
        recompute_flops = sum(
            n.annotations.get("flops_fwd", 0) // 2
            for n in g.nodes.values()
            if n.annotations.get("recompute")
        )
        if layer_scale != 1.0:
            recompute_flops = int(recompute_flops * layer_scale)
        g.metadata["recompute_flops"] = recompute_flops

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

        # ── Activation memory ─────────────────────────────────────────────────
        seq_len = g.metadata.get("seq_len", 2048)
        hidden = g.metadata.get("hidden", 4096)
        num_layers = g.metadata.get("num_layers", 32)
        batch_size = ctx.training.micro_batch if ctx.training else 1

        if g.metadata.get("fwd_bwd_stitched"):
            # Graph-native path: sum saved activations from fwd→bwd tensor liveness
            activations_bytes = self._graph_native_activations(g, tp, cp)
        else:
            # Korthikanti formula fallback
            activations_bytes = self._korthikanti_activations(
                g, ctx, seq_len, hidden, num_layers, batch_size, tp, cp, pp,
            )

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

    # ── Activation memory strategies ─────────────────────────────────────────

    def _korthikanti_activations(self, g, ctx, seq_len, hidden, num_layers,
                                  batch_size, tp, cp, pp):
        """Korthikanti formula: 34 * h * s * L * bs with sharding and inflight."""
        base = 34 * hidden * seq_len * num_layers * batch_size
        shard = tp * max(cp, 1)

        rc = getattr(ctx.training, "recompute_policy", "selective") if ctx.training else "selective"
        rc_mult = {"none": 1.0, "selective": 0.5, "full": 0.1}.get(rc, 1.0)

        # Stage-aware inflight (peak over local stages):
        # approximate activation volume per stage and scale by each stage's
        # inflight depth (pp - stage_id), then take the peak.
        stage_ids = {n.annotations.get("stage_id") for n in g.nodes.values()
                     if "stage_id" in n.annotations}
        if stage_ids and pp > 1:
            stage_ids_int = sorted(
                sid for sid in stage_ids if isinstance(sid, int)
            )
            if stage_ids_int:
                num_local_stages = len(stage_ids_int)
                per_stage_base = (base / shard) * rc_mult / max(num_local_stages, 1)
                peak_stage = max(max(pp - s, 1) for s in stage_ids_int)
                return per_stage_base * peak_stage
            max_inflight = pp
        else:
            max_inflight = pp

        return (base / shard) * rc_mult * max_inflight

    def _graph_native_activations(self, g, tp, cp):
        """Graph-native: sum saved activations from fwd→bwd edge liveness.

        Requires fwd_bwd_stitched=True in metadata. Accepts phase aliases:
        "fwd"/"forward"/"train_forward" and
        "bwd"/"backward"/"train_backward".
        Nodes with annotations["recompute"] == True have their outputs excluded.
        """
        shard = tp * max(cp, 1)

        # Classify nodes by phase annotation
        fwd_nodes = set()
        bwd_nodes = set()
        for nid, node in g.nodes.items():
            phase = node.annotations.get("phase", "")
            if phase in {"bwd", "backward", "train_backward"}:
                bwd_nodes.add(nid)
            else:
                fwd_nodes.add(nid)

        # Sum bytes on edges from forward to backward nodes = saved activations
        saved_bytes = 0
        recomputed_nodes = {
            nid for nid, n in g.nodes.items() if n.annotations.get("recompute")
        }

        for edge in g.edges:
            if edge.src in fwd_nodes and edge.dst in bwd_nodes:
                # Skip activations from recomputed nodes (they are freed)
                if edge.src in recomputed_nodes:
                    continue
                saved_bytes += edge.tensor.mem_bytes if hasattr(edge.tensor, 'mem_bytes') else 0

        return saved_bytes / shard


# ── TrainingPipelinePass ────────────────────────────────────────────────────────

@dataclass
class PipelineStepMetrics:
    """Pipeline step metrics for the selected PP schedule."""
    step_time_ms: float = 0.0
    per_stage_ms: float = 0.0
    warmup_steps: int = 0
    cooldown_steps: int = 0
    steady_steps: int = 0
    bubble_fraction: float = 0.0
    mfu: float = 0.0  # Model FLOPs Utilization
    hfu: float = 0.0  # Hardware FLOPs Utilization (includes recompute overhead)

    def to_dict(self) -> dict[str, float]:
        return {
            "step_time_ms": self.step_time_ms,
            "per_stage_ms": self.per_stage_ms,
            "warmup_steps": self.warmup_steps,
            "cooldown_steps": self.cooldown_steps,
            "steady_steps": self.steady_steps,
            "bubble_fraction": self.bubble_fraction,
            "mfu": self.mfu,
            "hfu": self.hfu,
        }


class TrainingPipelinePass(GraphPass):
    """Annotate graph with pipeline schedule metrics.

    Supported schedules include 1F1B, interleaved 1F1B, DualPipe,
    DualPipeV, and ZeroBubble. Correct 1F1B schedule (homogeneous stages):
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

            stage_fwd: dict[int, float] = {}
            stage_bwd: dict[int, float] = {}
            for s_id in range(pp):
                node_ids = stage_node_sets.get(s_id, set())
                if not node_ids:
                    stage_fwd[s_id] = 0.0
                    stage_bwd[s_id] = 0.0
                    continue
                sub = g.subgraph(node_ids)
                tl = sched.schedule(sub)
                fwd = tl.phase_latency("fwd")
                bwd = tl.phase_latency("bwd")
                # If no phase annotations, fall back to total latency as fwd
                if fwd == 0.0 and bwd == 0.0:
                    fwd = tl.total_latency_us
                if layer_scale != 1.0:
                    fwd *= layer_scale
                    bwd *= layer_scale
                stage_fwd[s_id] = fwd
                stage_bwd[s_id] = bwd

            g.metadata["stage_timelines_fwd"] = dict(stage_fwd)
            g.metadata["stage_timelines_bwd"] = dict(stage_bwd)
            stage_bwd_dw = {
                s_id: self._estimate_stage_dw_us(
                    g, node_ids, stage_bwd.get(s_id, 0.0),
                    warn_on_missing=False,
                )
                for s_id, node_ids in stage_node_sets.items()
            }
            for s_id in range(pp):
                stage_bwd_dw.setdefault(s_id, 0.0)
            g.metadata["stage_timelines_bwd_dw"] = dict(stage_bwd_dw)

            # Heterogeneous 1F1B when both fwd and bwd are populated
            if pp > 1 and stage_fwd and stage_bwd and any(v > 0 for v in stage_bwd.values()):
                t_fwd_0 = stage_fwd.get(0, 0.0)
                t_bwd_last = stage_bwd.get(pp - 1, 0.0)
                bottleneck_stage = max(
                    range(pp),
                    key=lambda s: stage_fwd.get(s, 0.0) + stage_bwd.get(s, 0.0),
                )
                t_stage = stage_fwd.get(bottleneck_stage, 0.0) + stage_bwd.get(bottleneck_stage, 0.0)
                t_w = stage_bwd_dw.get(bottleneck_stage, 0.0)

                # Apply VPP/DualPipe schedule-type adjustments to per-stage path
                pp_schedule = (ctx.training.pp_schedule if ctx.training else "1f1b")
                V = max(1, ctx.training.vpp_chunks if ctx.training else 1)

                if pp_schedule == "interleaved" and V > 1:
                    # VPP: warmup/cooldown reduced by V virtual stages
                    warmup = (pp - 1) * t_fwd_0 / V
                    cooldown = (pp - 1) * t_bwd_last / V
                elif pp_schedule == "dualpipev" and V > 1:
                    # DualPipeV: warmup=cooldown, both reduced by 2V
                    warmup = cooldown = (pp - 1) * t_stage / (2.0 * V)
                elif pp_schedule == "dualpipe":
                    # DualPipe: warmup=cooldown, reduced by 2
                    warmup = cooldown = (pp - 1) * t_stage / 2.0
                elif pp_schedule in {"zb", "zero_bubble"}:
                    if t_w <= 0.0:
                        logger.debug(
                            "ZeroBubble selected but no flops_dw annotations were "
                            "available for the bottleneck stage; using no dW bubble fill."
                        )
                    # ZeroBubble: dW work fills the remaining pipeline bubble.
                    bubble = (pp - 1) * max(t_stage - t_w, 0.0)
                    warmup = cooldown = bubble / 2.0
                else:
                    # Standard 1F1B
                    warmup = (pp - 1) * t_fwd_0
                    cooldown = (pp - 1) * t_bwd_last

                step_time_us = warmup + num_microbatches * t_stage + cooldown
                bubble_us = warmup + cooldown
                bubble_fraction = bubble_us / step_time_us if step_time_us > 0 else 0.0
                per_stage_us = t_stage
            else:
                # Homogeneous fallback
                per_stage_us = max(stage_fwd.values(), default=0.0)
                pp_schedule = (ctx.training.pp_schedule if ctx.training else "1f1b")
                V = max(1, ctx.training.vpp_chunks if ctx.training else 1)

                if pp_schedule == "interleaved" and V > 1:
                    bubble_us = (pp - 1) * per_stage_us / V
                elif pp_schedule == "dualpipev" and V > 1:
                    bubble_us = (pp - 1) * per_stage_us / (2.0 * V)
                elif pp_schedule == "dualpipe":
                    bubble_us = (pp - 1) * per_stage_us / 2.0
                elif pp_schedule in {"zb", "zero_bubble"}:
                    bottleneck_stage = max(
                        range(pp),
                        key=lambda s: stage_fwd.get(s, 0.0) + stage_bwd.get(s, 0.0),
                    )
                    t_w = stage_bwd_dw.get(bottleneck_stage, 0.0)
                    if t_w <= 0.0:
                        logger.debug(
                            "ZeroBubble selected but no flops_dw annotations were "
                            "available for the bottleneck stage; using no dW bubble fill."
                        )
                    bubble_us = (pp - 1) * max(per_stage_us - t_w, 0.0)
                else:
                    bubble_us = (pp - 1) * per_stage_us

                step_time_us = num_microbatches * per_stage_us + bubble_us
                bubble_fraction = bubble_us / step_time_us if step_time_us > 0 else 0.0
        else:
            # pp=1 (or stage_id not yet annotated): whole graph is scheduled as one
            # unit. total_latency_us covers all pp stages linearly, so divide by pp.
            tl = sched.schedule(g)
            per_stage_us = tl.total_latency_us / pp
            if layer_scale != 1.0:
                per_stage_us *= layer_scale

            # Schedule-type bubble for the non-phase-aware path.
            # Phase-aware path (pp>1, stage_id present) already computed step_time_us
            # and bubble_fraction via the heterogeneous/homogeneous formula above.
            # TODO Phase 3: apply VPP/DualPipe adjustments to the per-stage path too.
            pp_schedule = (ctx.training.pp_schedule if ctx.training else "1f1b")
            V = max(1, ctx.training.vpp_chunks if ctx.training else 1)

            if pp_schedule == "interleaved" and V > 1 and pp > 1:
                bubble_us = (pp - 1) * per_stage_us / V
            elif pp_schedule == "dualpipev" and pp > 1:
                bubble_us = (pp - 1) * per_stage_us / (2.0 * max(1, V))
            elif pp_schedule == "dualpipe" and pp > 1:
                bubble_us = (pp - 1) * per_stage_us / 2.0
            elif pp_schedule in {"zb", "zero_bubble"} and pp > 1:
                logger.debug(
                    "ZeroBubble selected without stage/phase or flops_dw annotations; "
                    "using no dW bubble fill."
                )
                bubble_us = (pp - 1) * per_stage_us
            else:  # "1f1b" or pp == 1
                bubble_us = (pp - 1) * per_stage_us

            step_time_us = num_microbatches * per_stage_us + bubble_us
            bubble_fraction = bubble_us / step_time_us if step_time_us > 0 else 0.0

        step_time_ms = step_time_us / 1000.0
        per_stage_ms = per_stage_us / 1000.0

        # DP-in-bubble: if DP AR fits inside the bubble window, it is free;
        # otherwise the exposed portion adds to step time.
        dp = ctx.parallel.dp if ctx.parallel else 1
        if dp > 1 and ctx.training and ctx.training.dp_overlap_in_bubble:
            dp_comm_nodes = [
                n for n in g.nodes.values()
                if n.annotations.get("dp_comm") and n.attrs.get("bucket_bytes", 0) > 0
            ]
            if dp_comm_nodes:
                # Prefer latency annotations on comm nodes when available
                latency_sum = sum(
                    n.annotations.get("latency_us", 0.0) for n in dp_comm_nodes
                )
                if latency_sum > 0.0:
                    t_dp_ar_us = latency_sum
                else:
                    # Fallback: alpha-beta bandwidth formula
                    bucket_bytes = sum(n.attrs["bucket_bytes"] for n in dp_comm_nodes)
                    dp_bw_bytes_per_us = hw.interconnect.inter_node.bandwidth_gbps * 1e9 / 8 / 1e6
                    ring_factor = 2.0 * (dp - 1) / dp
                    t_dp_ar_us = ring_factor * bucket_bytes / dp_bw_bytes_per_us if dp_bw_bytes_per_us > 0 else 0.0
                t_exposed_dp_us = max(0.0, t_dp_ar_us - bubble_us)
                step_time_us += t_exposed_dp_us
                step_time_ms = step_time_us / 1000.0

        # Overlap-aware comm time: reduce step_time by hidden comm
        overlap_nodes = [
            n for n in g.nodes.values()
            if n.category == "communication"
            and n.annotations.get("overlap_type", "none") != "none"
            and n.annotations.get("latency_us", 0) > 0
        ]
        if overlap_nodes:
            total_comm_us = 0.0
            total_exposed_us = 0.0
            for cn in overlap_nodes:
                comm_lat = cn.annotations["latency_us"]
                otype = cn.annotations["overlap_type"]
                # Resolve target compute node latency for overlap
                target_lat = 0.0
                target_key = cn.annotations.get("overlap_target", "")
                if target_key:
                    # For ring_cp: target_key is "fa_tile:<node_id>"
                    target_id = target_key.split(":", 1)[1] if ":" in target_key else target_key
                    target_node = g.nodes.get(target_id)
                    if target_node:
                        target_lat = target_node.annotations.get("latency_us", 0.0)
                # CoC fallback: if no explicit overlap target, use predecessor
                # compute latency as the overlap window reference.
                if otype == "coc" and target_lat <= 0.0:
                    pred_ids = g.predecessors(cn.id)
                    pred_lats = [
                        g.nodes[p].annotations.get("latency_us", 0.0)
                        for p in pred_ids
                        if g.nodes[p].category != "communication"
                    ]
                    if pred_lats:
                        target_lat = max(pred_lats)
                exposed = compute_exposed_comm_time(
                    comm_lat, otype, target_lat,
                    coc_tile_k=cn.attrs.get("coc_tile_k", 4),
                )
                total_comm_us += comm_lat
                total_exposed_us += exposed
            hidden_us = total_comm_us - total_exposed_us
            step_time_us -= hidden_us
            step_time_ms = step_time_us / 1000.0

        warmup_steps = max(0, pp - 1)
        cooldown_steps = max(0, pp - 1)
        steady_steps = num_microbatches
        training_flops = g.metadata.get("training_flops", 0.0)
        world_size = ctx.parallel.total_devices if ctx.parallel else 1

        from python.zrt.ir.types import DType
        peak_flops_per_gpu = hw.peak_flops(DType.BF16)

        step_time_sec = step_time_us / 1e6
        peak_flops_total = world_size * peak_flops_per_gpu

        # MFU: model FLOPs only (excludes recompute overhead)
        recompute_flops = float(g.metadata.get("recompute_flops", 0))
        model_flops = training_flops - recompute_flops
        mfu = (model_flops / step_time_sec / peak_flops_total) if (step_time_sec > 0 and peak_flops_total > 0) else 0.0

        # HFU: all executed FLOPs including recompute
        hfu = (training_flops / step_time_sec / peak_flops_total) if (step_time_sec > 0 and peak_flops_total > 0) else 0.0

        metrics = PipelineStepMetrics(
            step_time_ms=step_time_ms,
            per_stage_ms=per_stage_ms,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            steady_steps=steady_steps,
            bubble_fraction=bubble_fraction,
            mfu=min(mfu, 1.0),
            hfu=min(hfu, 1.0),
        )

        g.metadata["pipeline_metrics"] = metrics
        return g

    @staticmethod
    def _estimate_stage_dw_us(
        g: "OpGraph",
        node_ids: set[str],
        bwd_us: float,
        warn_on_missing: bool = True,
    ) -> float:
        """Estimate stage dW time from backward FLOPs annotations.

        Graph scheduling currently exposes aggregate backward latency by stage.
        Training FLOPs annotations carry the dX/dW split, so we apportion the
        scheduled backward time by the dW FLOPs ratio for ZeroBubble.
        """
        if bwd_us <= 0.0:
            return 0.0

        dx_flops = 0.0
        dw_flops = 0.0
        for node_id in node_ids:
            node = g.nodes[node_id]
            dx_flops += float(node.annotations.get("flops_dx", 0.0))
            dw_flops += float(node.annotations.get("flops_dw", 0.0))

        total_bwd_flops = dx_flops + dw_flops
        if total_bwd_flops <= 0.0 or dw_flops <= 0.0:
            if warn_on_missing:
                logger.debug(
                    "Unable to estimate ZeroBubble dW time: missing flops_dw "
                    "annotations for %d stage nodes.",
                    len(node_ids),
                )
            return 0.0
        return bwd_us * dw_flops / total_bwd_flops


# ── Exposed comm-time helper ────────────────────────────────────────────────────

def compute_exposed_comm_time(
    comm_latency_us: float,
    overlap_type: str,
    target_latency_us: float = 0.0,
    coc_tile_k: int = 4,
) -> float:
    """Compute exposed (non-hidden) communication time under overlap.

    Args:
        comm_latency_us: latency of the comm node in microseconds.
        overlap_type: "coc", "mc2", "ring_cp", or "none".
        target_latency_us: latency of the compute node being overlapped.
        coc_tile_k: number of tiles for CoC overlap (default 4).

    Returns:
        Exposed comm time in microseconds (>= 0).
    """
    if overlap_type == "mc2":
        # MC2: fully fused AG+matmul, zero exposed comm
        return 0.0
    elif overlap_type == "coc":
        # CoC: comm overlaps with matmul*(k-1)/k window
        overlap_window = target_latency_us * (coc_tile_k - 1) / coc_tile_k
        return max(0.0, comm_latency_us - overlap_window)
    elif overlap_type == "ring_cp":
        # Ring-CP: P2P overlaps with FA tile
        return max(0.0, comm_latency_us - target_latency_us)
    else:
        # No overlap: full comm time is exposed
        return comm_latency_us
