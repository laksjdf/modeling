"""Training analysis passes: FLOPs, Memory, and pipeline scheduling."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from python.zrt.ir.param_count import count_params
from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.hardware.spec import HardwareSpec
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext

logger = logging.getLogger(__name__)

# Phase aliases for backward-pass nodes in stitched graphs
_BWD_PHASES = {"bwd", "backward", "train_backward"}


# ── TrainingFlopsPass ───────────────────────────────────────────────────────────

class TrainingFlopsPass(GraphPass):
    """Annotate graph with training FLOPs.

    Strategy (priority order):
    1. Per-node annotations: sum flops_fwd / flops_dx / flops_dw from
       ``FlopsPass`` when available (more accurate for MoE, comm, etc.)
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
        is_stitched = g.metadata.get("fwd_bwd_stitched", False)

        if is_stitched:
            # Graph-native path: each node's flops_fwd = cost of that op (fwd OR bwd)
            # Filter by phase to correctly split forward/backward FLOPs
            forward_flops = sum(
                n.annotations.get("flops_fwd", 0) for n in g.nodes.values()
                if n.annotations.get("phase", "fwd") not in _BWD_PHASES
            )
            backward_flops = sum(
                n.annotations.get("flops_fwd", 0) for n in g.nodes.values()
                if n.annotations.get("phase", "") in _BWD_PHASES
            )
        else:
            # Non-stitched (fwd graph only): estimate backward via dx/dw ratios
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
        # already includes 2x multiplier (passes.py FlopsPass), so base fwd = flops_fwd / 2
        # Only fwd-phase nodes can be recomputed (bwd-phase nodes are not recomputed)
        recompute_flops = sum(
            n.annotations.get("flops_fwd", 0) // 2
            for n in g.nodes.values()
            if n.annotations.get("recompute")
            and n.annotations.get("phase", "fwd") not in _BWD_PHASES
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

    Activation memory uses graph-native tensor liveness on stitched graphs
    or the Korthikanti formula (34 * h * s * L * bs) as fallback, with
    recompute-policy multiplier, CP sharding, and PP inflight depth.

    Sharding factors are read from ``g.metadata["zero"]`` (written by
    ZeroFSDPPass which always runs first).  Every bucket is also divided
    by TP because TP splits weight matrices across ranks.

    Adds to graph.metadata:
      "memory_breakdown": TrainingMemoryBreakdown
    """

    name = "training_memory"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        g = graph.clone()

        # ── Determine param dtype ─────────────────────────────────────────────
        # Use actual model dtype from metadata (set by FlopsPass/graph capture),
        # fallback to BF16=2 bytes when unavailable.
        param_dtype = g.metadata.get("param_dtype_bytes", 2)

        # ── Layer scaling for partial-trace graphs ─────────────────────────────
        # When only a subset of layers is traced, count_params(g) returns the
        # parameter count for those traced layers.  Scale to full model using
        # the same layer_scale factor computed by TrainingFlopsPass.
        num_layers = g.metadata.get("num_layers", 0)
        num_layers_traced = g.metadata.get("num_layers_traced", num_layers)
        layer_scale = (
            num_layers / num_layers_traced
            if num_layers_traced > 0 and num_layers != num_layers_traced
            else 1.0
        )

        total_params = count_params(g)
        if layer_scale != 1.0:
            total_params = int(total_params * layer_scale)

        dp = ctx.parallel.dp if ctx.parallel else 1
        tp = ctx.parallel.tp if ctx.parallel else 1
        cp = getattr(ctx.parallel, "cp", 1) if ctx.parallel else 1
        pp = ctx.parallel.pp if ctx.parallel else 1

        # ── Weight / grad / opt-state sharding ──────────────────────────────
        # Weight and grad buckets shrink by pp (each stage holds 1/pp layers),
        # tp (tensor parallel splits matrices), and DP for ZeRO-2/3.
        # opt_shard is kept pp-free because the fallback formula handles pp
        # separately via total_params_for_opt, and opt_state_from_graph path
        # uses dp_for_opt directly (not opt_shard).
        zero_meta = g.metadata.get("zero")
        if zero_meta is not None:
            weight_shard = zero_meta["weight_shard"] * tp * pp
            grad_shard = zero_meta["grad_shard"] * tp * pp
            opt_shard = zero_meta["optstate_shard"] * tp  # pp handled by total_params_for_opt
        else:
            zero_stage = ctx.training.zero_stage if ctx.training else 0
            dp_factor = dp if zero_stage >= 1 else 1
            weight_shard = pp * tp * (dp if zero_stage >= 3 else 1)
            grad_shard = pp * tp * (dp if zero_stage >= 2 else 1)
            opt_shard = tp * dp_factor  # pp handled by total_params_for_opt

        weights_bytes = (total_params * param_dtype) / weight_shard
        grads_bytes = (total_params * param_dtype) / grad_shard

        # ── Optimizer state memory ───────────────────────────────────────────────
        # Try to use OptimizerPass annotation first (graph-based modeling)
        opt_state_from_graph = None
        for node in g.nodes.values():
            if node.annotations.get("optimizer_step"):
                opt_state_from_graph = node.attrs.get("state_bytes")
                break

        if opt_state_from_graph is not None:
            # state_bytes from OptimizerPass already accounts for TP, PP, and
            # (for ZeRO-3) DP sharding.  Only apply the ZeRO-1/2 DP sharding.
            zero_stage = ctx.training.zero_stage if ctx.training else 0
            dp_for_opt = dp if (1 <= zero_stage < 3) else 1
            opt_bytes = opt_state_from_graph / dp_for_opt
        else:
            # Fallback: independent calculation (backward compatibility)
            optimizer = ctx.training.optimizer if ctx.training else "adam"
            master_bytes = 4  # FP32 master dtype

            if optimizer in ("adam", "adamw"):
                # Adam: master copy + m + v = 3 × P × master_dtype
                opt_bytes_per_param = master_bytes * 3  # 12 B/P
            elif optimizer == "muon":
                # Muon: master copy + momentum ≈ 2.1 × P × master_dtype
                opt_bytes_per_param = int(master_bytes * 2.1)
            else:
                opt_bytes_per_param = master_bytes * 3

            # Apply PP sharding for fallback calculation
            total_params_for_opt = total_params
            if pp > 1:
                total_params_for_opt = int(total_params / pp)

            opt_bytes = (total_params_for_opt * opt_bytes_per_param) / opt_shard

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
        """Korthikanti formula: 34 * h * s * L * bs with sharding and inflight.

        Improved to use actual recompute annotation fraction when available,
        falling back to policy-based multipliers for "none"/"selective"/"full".

        TP with SP (Sequence Parallel): shards seq by TP.
        CP: further shards seq by CP.
        Combined: shard = max(tp, 1) * max(cp, 1).
        """
        base = 34 * hidden * seq_len * num_layers * batch_size

        tp_sp = tp if tp > 1 else 1
        cp_factor = cp if cp > 1 else 1
        shard = tp_sp * cp_factor

        rc_mult = self._derive_recompute_multiplier(g, ctx)

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

    def _derive_recompute_multiplier(self, g, ctx):
        """Derive recompute memory multiplier from node annotations.

        Priority order:
        1. Count forward nodes with recompute=True annotation (graph-native)
        2. Use training.recompute_policy string with fixed multipliers (fallback)

        Returns:
            Memory multiplier: 1.0 (no recompute) to ~0.1 (full recompute)
        """
        # Try graph-native: count actual recompute annotations
        fwd_nodes = [
            n for n in g.nodes.values()
            if n.annotations.get("phase", "fwd") not in _BWD_PHASES
        ]
        if fwd_nodes:
            recompute_count = sum(
                1 for n in fwd_nodes if n.annotations.get("recompute")
            )
            if recompute_count > 0:
                # Linear interpolation: more recompute = less memory
                # Full selective recompute typically saves ~50%
                fraction = recompute_count / len(fwd_nodes)
                # Map fraction to memory saved: 0→1.0, 0.5→0.5, 1.0→0.1
                if fraction >= 0.9:
                    return 0.1  # Full recompute
                elif fraction >= 0.3:
                    return 1.0 - (fraction * 0.8)  # Selective scales to 0.1
                else:
                    return 1.0 - (fraction * 0.5)  # Light recompute

        # Fallback: use policy string
        rc = getattr(ctx.training, "recompute_policy", "none") if ctx.training else "none"
        return {"none": 1.0, "selective": 0.5, "full": 0.1}.get(rc, 1.0)

    def _graph_native_activations(self, g, tp, cp):
        """Graph-native: sum saved activations from fwd→bwd edge liveness.

        Requires fwd_bwd_stitched=True in metadata. Accepts phase aliases:
        "fwd"/"forward"/"train_forward" and
        "bwd"/"backward"/"train_backward".
        Nodes with annotations["recompute"] == True have their outputs excluded.

        TP with SP and CP: shard = max(tp, 1) * max(cp, 1).
        """
        tp_sp = tp if tp > 1 else 1
        cp_factor = cp if cp > 1 else 1
        shard = tp_sp * cp_factor

        fwd_nodes = set()
        bwd_nodes = set()
        for nid, node in g.nodes.items():
            phase = node.annotations.get("phase", "")
            if phase in _BWD_PHASES:
                bwd_nodes.add(nid)
            else:
                fwd_nodes.add(nid)

        saved_bytes = 0
        recomputed_nodes = {
            nid for nid, n in g.nodes.items() if n.annotations.get("recompute")
        }

        for edge in g.edges:
            if edge.src in fwd_nodes and edge.dst in bwd_nodes:
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

        # Per-stage scheduling state (populated by both branches below)
        stage_fwd: dict[int, float] = {}
        stage_bwd: dict[int, float] = {}
        stage_bwd_dw: dict[int, float] = {}

        if pp > 1 and any("stage_id" in n.annotations for n in g.nodes.values()):
            # Per-stage scheduling: schedule each stage's subgraph independently
            stage_node_sets: dict[int, set[str]] = {}
            for node in g.nodes.values():
                sid = node.annotations.get("stage_id", 0)
                stage_node_sets.setdefault(sid, set()).add(node.id)

            # Detect EP load imbalance before scheduling (affects all stages)
            ep_imbalance = self._detect_ep_imbalance(g, ctx)

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

                # Apply EP load imbalance factor to stage latency
                if ep_imbalance > 1.0:
                    fwd *= ep_imbalance
                    bwd *= ep_imbalance

                stage_fwd[s_id] = fwd
                stage_bwd[s_id] = bwd

            g.metadata["stage_timelines_fwd"] = dict(stage_fwd)
            g.metadata["stage_timelines_bwd"] = dict(stage_bwd)
            stage_bwd_dw.update({
                s_id: self._estimate_stage_dw_us(
                    g, node_ids, stage_bwd.get(s_id, 0.0),
                    warn_on_missing=False,
                )
                for s_id, node_ids in stage_node_sets.items()
            })
            for s_id in range(pp):
                stage_bwd_dw.setdefault(s_id, 0.0)
            g.metadata["stage_timelines_bwd_dw"] = dict(stage_bwd_dw)

        else:
            # pp=1 or no stage_id: schedule whole graph as single unit.
            # When pp > 1 without stage annotations, divide total by pp for
            # per-stage estimate (homogeneous fallback).
            if pp > 1:
                logger.warning(
                    "PipelineParallelPass has not assigned stage_id annotations; "
                    "dividing whole-graph latency by pp=%d as homogeneous fallback. "
                    "Results will ignore real stage heterogeneity and warmup/cooldown "
                    "structure. Run PipelineParallelPass before this pass for accurate "
                    "per-stage scheduling.",
                    pp,
                )
            tl = sched.schedule(g)
            _fwd = tl.phase_latency("fwd")
            _bwd = tl.phase_latency("bwd")
            fwd = _fwd if isinstance(_fwd, (int, float)) else 0.0
            bwd = _bwd if isinstance(_bwd, (int, float)) else 0.0
            if fwd == 0.0 and bwd == 0.0:
                fwd = tl.total_latency_us
            if layer_scale != 1.0:
                fwd *= layer_scale
                bwd *= layer_scale
            per_stage_fwd = fwd / pp
            per_stage_bwd = bwd / pp
            for s in range(pp):
                stage_fwd[s] = per_stage_fwd
                stage_bwd[s] = per_stage_bwd
                stage_bwd_dw[s] = 0.0

        # ── Delegate to PipelineComposer ──────────────────────────────────
        from python.zrt.training.compose.stage import StageTime
        from python.zrt.training.compose.schedules import (
            PP_SCHED_BY_NAME, COMPOSER_BY_SCHED,
        )
        from python.zrt.training.spec.strategy import (
            Strategy, OptKind,
        )
        OPT_MAP = {"adam": OptKind.ADAM, "adamw": OptKind.ADAM, "muon": OptKind.MUON}

        pp_schedule = ctx.training.pp_schedule if ctx.training else "1f1b"
        opt_str = ctx.training.optimizer if ctx.training else "adam"

        stage_times_list = [
            StageTime(
                fwd=stage_fwd.get(s, 0.0) / 1e6,
                bwd=stage_bwd.get(s, 0.0) / 1e6,
                bwd_dw=stage_bwd_dw.get(s, 0.0) / 1e6,
            )
            for s in range(pp)
        ]

        dp_ar_time_s = self._compute_dp_ar_time(g, hw, ctx) / 1e6

        strategy_proxy = Strategy(
            tp=ctx.parallel.tp if ctx.parallel else 1,
            pp=pp,
            ep=ctx.parallel.ep if ctx.parallel else 1,
            dp=ctx.parallel.dp if ctx.parallel else 1,
            cp=getattr(ctx.parallel, "cp", 1) if ctx.parallel else 1,
            micro_batch=ctx.training.micro_batch if ctx.training else 1,
            global_batch=ctx.training.global_batch if ctx.training else 32,
            pp_schedule=PP_SCHED_BY_NAME.get(pp_schedule, PP_SCHED_BY_NAME["1f1b"]),
            vpp_chunks=max(1, ctx.training.vpp_chunks if ctx.training else 1),
            zero_stage=ctx.training.zero_stage if ctx.training else 0,
            optimizer=OPT_MAP.get(opt_str, OptKind.ADAM),
            dp_overlap_in_bubble=ctx.training.dp_overlap_in_bubble if ctx.training else True,
        )

        composer_cls = COMPOSER_BY_SCHED.get(strategy_proxy.pp_schedule)
        if composer_cls is None:
            composer_cls = COMPOSER_BY_SCHED[PP_SCHED_BY_NAME["1f1b"]]
        step_result = composer_cls().compose(
            stage_times_list, num_microbatches, pp, dp_ar_time_s, strategy_proxy
        )

        step_time_us = step_result.step_time * 1e6
        per_stage_us = max(
            (st.fwd + st.bwd) * 1e6 for st in stage_times_list
        ) if stage_times_list else 0.0

        step_time_ms = step_time_us / 1000.0
        per_stage_ms = per_stage_us / 1000.0

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
                cp_rounds = cn.attrs.get("cp_rounds", 1)
                target_lat = 0.0
                target_key = cn.annotations.get("overlap_target", "")
                if target_key:
                    target_id = target_key.split(":", 1)[1] if ":" in target_key else target_key
                    target_node = g.nodes.get(target_id)
                    if target_node:
                        target_lat = target_node.annotations.get("latency_us", 0.0)
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
                    cp_rounds=cp_rounds,
                )
                total_comm_us += comm_lat
                total_exposed_us += exposed
            hidden_us = total_comm_us - total_exposed_us
            step_time_us -= hidden_us
            step_time_ms = step_time_us / 1000.0

        warmup_steps = step_result.warmup_steps
        cooldown_steps = step_result.cooldown_steps
        steady_steps = num_microbatches
        training_flops = g.metadata.get("training_flops", 0.0)
        world_size = ctx.parallel.total_devices if ctx.parallel else 1

        from python.zrt.ir.types import DType
        peak_flops_per_gpu = hw.peak_flops(DType.BF16)

        step_time_sec = step_time_us / 1e6
        peak_flops_total = world_size * peak_flops_per_gpu

        # MFU/HFU: per-GPU utilization = (FLOPs per GPU per step) / (peak_per_gpu × step_time)
        # training_flops is per micro-batch; multiply by num_microbatches for full step.
        # Use per-GPU peak (not cluster total) so result is per-device utilization.
        # Divide by PP because training_flops includes ALL layer FLOPs (via layer_scale),
        # but each GPU only handles 1/pp of the layers.
        from python.zrt.training.compose.schedules import util_from_flops
        recompute_flops = float(g.metadata.get("recompute_flops", 0))
        pp = ctx.parallel.pp if ctx.parallel else 1
        model_flops = (training_flops - recompute_flops) / pp
        total_flops_for_hfu = training_flops / pp
        mfu = util_from_flops(model_flops * num_microbatches, peak_flops_per_gpu, step_time_sec)
        hfu = util_from_flops(total_flops_for_hfu * num_microbatches, peak_flops_per_gpu, step_time_sec)

        metrics = PipelineStepMetrics(
            step_time_ms=step_time_ms,
            per_stage_ms=per_stage_ms,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            steady_steps=steady_steps,
            bubble_fraction=step_result.bubble_fraction,
            mfu=min(mfu, 1.0),
            hfu=min(hfu, 1.0),
        )

        g.metadata["pipeline_metrics"] = metrics

        # Add optimizer step time (per §5.5.2 of design doc)
        opt_step_time_us = self._compute_optimizer_step_time(g, hw, ctx)
        if opt_step_time_us > 0:
            g.metadata["optimizer_step_time_us"] = opt_step_time_us
            step_time_us += opt_step_time_us
            metrics.step_time_ms = step_time_us / 1000.0

        return g

    @staticmethod
    def _compute_optimizer_step_time(
        g: "OpGraph", hw: "HardwareSpec", ctx: "TransformContext",
    ) -> float:
        """Compute optimizer step time in microseconds.

        Includes:
        1. Optimizer compute FLOPs (Adam: memory-bound; Muon: compute-bound)
        2. Muon AllGather/ReduceScatter communication (Muon + ZeRO + DP>1)
        """
        opt_node = g.nodes.get("optimizer_step")
        if opt_node is None:
            return 0.0

        optimizer = opt_node.attrs.get("optimizer", "adam")
        step_flops = float(opt_node.attrs.get("step_flops", 0))

        from python.zrt.ir.types import DType
        if optimizer == "muon":
            # Muon NS is compute-bound (large GEMMs)
            peak_flops = hw.peak_flops(DType.BF16)
            compute_time_us = (step_flops / peak_flops) * 1e6 if peak_flops > 0 else 0.0
        else:
            # Adam is memory-bound (element-wise ops)
            opt_state_bytes = float(opt_node.attrs.get("state_bytes", 0))
            hbm_bw = hw.memory.hbm_bandwidth_gbps * 1e9 / 8
            compute_time_us = (opt_state_bytes / hbm_bw) * 1e6 if hbm_bw > 0 else 0.0

        # Muon additional communication (AllGather momentum)
        comm_time_us = 0.0
        if optimizer == "muon":
            ag_bytes = float(opt_node.attrs.get("muon_ag_bytes", 0))
            ns_rotation = opt_node.attrs.get("ns_rotation", True)
            if ag_bytes > 0:
                dp = ctx.parallel.dp if ctx.parallel else 1
                gpus_per_node = hw.interconnect.intra_node.num_devices
                link = hw.interconnect.inter_node if dp > gpus_per_node else hw.interconnect.intra_node
                dp_bw = link.bandwidth_gbps * 1e9 / 8
                # Ring factor: AG always, RS only when rotation=True
                # AG: (dp-1)/dp factor; RS: same factor when rotation=True
                if ns_rotation:
                    ring_factor = 2.0 * (dp - 1) / dp  # AG + RS
                else:
                    ring_factor = 1.0 * (dp - 1) / dp  # AG only (Moonshot optimization)
                comm_time_us = (ring_factor * ag_bytes / dp_bw) * 1e6 if dp_bw > 0 else 0.0

        return compute_time_us + comm_time_us

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

    @staticmethod
    def _compute_dp_ar_time(
        g: "OpGraph", hw: "HardwareSpec", ctx: "TransformContext",
    ) -> float:
        """Compute DP allreduce time in microseconds from graph annotations."""
        dp = ctx.parallel.dp if ctx.parallel else 1
        if dp <= 1:
            return 0.0

        dp_comm_nodes = [
            n for n in g.nodes.values()
            if n.annotations.get("dp_comm") and n.attrs.get("bucket_bytes", 0) > 0
        ]
        if not dp_comm_nodes:
            return 0.0

        latency_sum = sum(
            n.annotations.get("latency_us", 0.0) for n in dp_comm_nodes
        )
        if latency_sum > 0.0:
            return latency_sum

        bucket_bytes = sum(n.attrs["bucket_bytes"] for n in dp_comm_nodes)
        gpus_per_node = hw.interconnect.intra_node.num_devices
        link = hw.interconnect.inter_node if dp > gpus_per_node else hw.interconnect.intra_node
        dp_bw_bytes_per_us = (
            link.bandwidth_gbps * 1e9 / 8 / 1e6
        )
        ring_factor = 2.0 * (dp - 1) / dp
        return (
            ring_factor * bucket_bytes / dp_bw_bytes_per_us
            if dp_bw_bytes_per_us > 0 else 0.0
        )

    @staticmethod
    def _detect_ep_imbalance(g: "OpGraph", ctx: "TransformContext") -> float:
        """Detect Expert Parallel load imbalance factor.

        EP imbalance occurs when experts receive uneven token loads, causing
        some ranks to wait longer than others. The bottleneck stage time is
        multiplied by this factor.

        Returns:
            Imbalance factor (1.0 = balanced, >1.0 = bottleneck). Defaults to 1.0
            when EP is not used or imbalance cannot be detected.
        """
        ep = ctx.parallel.ep if ctx.parallel else 1
        if ep <= 1:
            return 1.0

        # Count expert FFN nodes (nodes in MoE expert blocks)
        expert_nodes = [
            n for n in g.nodes.values()
            if n.category == "compute" and ("expert" in n.scope.lower() or "moe" in n.scope.lower())
        ]

        if not expert_nodes:
            return 1.0

        # Simple heuristic: assume 10% imbalance for EP > 1
        # In production, this would be derived from actual token routing statistics
        # For now, use a conservative estimate based on EP degree
        imbalance_factors = {2: 1.05, 4: 1.10, 8: 1.15, 16: 1.20}
        return imbalance_factors.get(ep, 1.15)


# ── Exposed comm-time helper ────────────────────────────────────────────────────

def compute_exposed_comm_time(
    comm_latency_us: float,
    overlap_type: str,
    target_latency_us: float = 0.0,
    coc_tile_k: int = 4,
    cp_rounds: int = 1,
) -> float:
    """Compute exposed (non-hidden) communication time under overlap.

    Args:
        comm_latency_us: latency of the comm node in microseconds.
        overlap_type: "coc", "mc2", "ring_cp", or "none".
        target_latency_us: latency of the compute node being overlapped.
        coc_tile_k: number of tiles for CoC overlap (default 4).
        cp_rounds: number of CP P2P rounds (for ring_cp).

    Returns:
        Exposed comm time in microseconds (>= 0).
    """
    if overlap_type == "mc2":
        return 0.0
    elif overlap_type == "coc":
        overlap_window = target_latency_us * (coc_tile_k - 1) / coc_tile_k
        return max(0.0, comm_latency_us - overlap_window)
    elif overlap_type == "ring_cp":
        fa_tile_latency = target_latency_us / cp_rounds if cp_rounds > 1 else target_latency_us
        p2p_round_latency = comm_latency_us / cp_rounds if cp_rounds > 1 else comm_latency_us
        exposed_per_round = max(0.0, p2p_round_latency - fa_tile_latency)
        return exposed_per_round * cp_rounds
    else:
        return comm_latency_us
