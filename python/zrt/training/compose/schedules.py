"""Pipeline composers for training step-time schedules.

Reference: Megatron-LM (Narayanan et al. 2021) §3.2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from zrt.training.compose.stage import StageTime, stage_time
from zrt.training.ir.training_graph import Graph
from zrt.training.models.comm import total_comm_time, optimizer_comm_time
from zrt.training.topology import CommDomain
from zrt.training.models.flops import recompute_overhead_flops
from zrt.training.models.memory import MemBreakdown, memory_breakdown
from zrt.training.models.optimizer import (
    muon_step_flops_from_arch,
)
from zrt.training.io.perf_tables import effective_flops, effective_hbm_bw_bps
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.strategy import PPSched, Strategy, resolve_muon_ns_steps
from zrt.training.spec.system import SystemSpec

PP_SCHED_BY_NAME: dict[str, PPSched] = {
    "1f1b": PPSched.ONE_F_ONE_B,
    "interleaved": PPSched.INTERLEAVED,
    "i1f1b": PPSched.INTERLEAVED,
    "zb": PPSched.ZERO_BUBBLE,
    "zero_bubble": PPSched.ZERO_BUBBLE,
    "dualpipe": PPSched.DUALPIPE,
    "dualpipev": PPSched.DUALPIPE_V,
}


@dataclass
class StepResult:
    """Training step time breakdown.

    Strict invariants (all in seconds):
      step_time        = pipeline_time + optimizer_time + optimizer_comm
      pipeline_time    = compute_time + exposed_comm + bubble
      compute_time     = fwd_compute + bwd_compute + recompute_time
      bubble           = warmup + cooldown   (absolute pipeline idle, seconds)
      exposed_comm     = tp_exposed + cp_exposed + ep_exposed + pp_exposed + dp_exposed
      hidden_comm      = dp_hidden + tp_hidden + ep_hidden + pp_hidden
      total_comm_volume = exposed_comm + hidden_comm
    """

    # ── Core step timing ──────────────────────────────────────────────────
    step_time: float = 0.0  # Total step time (seconds)
    pipeline_time: float = 0.0  # Pipeline time = compute_time + exposed_comm + bubble
    optimizer_time: float = 0.0  # Optimizer step compute
    optimizer_comm: float = 0.0  # Optimizer step comm exposed on critical path (post-hide)
    optimizer_comm_hidden: float = 0.0  # Optimizer AG hidden under NS compute (Moonshot rotation)

    # ── Pipeline structure (set by composers) ─────────────────────────────
    bubble_fraction: float = 0.0
    bubble: float = 0.0  # Absolute pipeline-idle time = warmup + cooldown (s)
    warmup: float = 0.0
    steady: float = 0.0
    cooldown: float = 0.0
    dp_exposed: float = 0.0  # DP AR/RS on critical path (set by composer; also DP group share)
    schedule_name: str = "1f1b"
    warmup_steps: int = 0
    cooldown_steps: int = 0
    per_stage: list[StageTime] = field(default_factory=list)
    memory: MemBreakdown | None = None
    mfu: float = 0.0
    hfu: float = 0.0
    mfu_native: float = 0.0  # MFU vs op-mix-weighted effective peak

    # ── Fwd/Bwd phase breakdown (seconds) ────────────────────────────────
    warmup_fwd: float = 0.0
    warmup_bwd: float = 0.0
    steady_fwd: float = 0.0
    steady_bwd: float = 0.0
    cooldown_fwd: float = 0.0
    cooldown_bwd: float = 0.0
    steady_fwd_per_mb: float = 0.0
    steady_bwd_per_mb: float = 0.0
    steady_per_mb: float = 0.0

    # ── Compute / comm breakdown (set by pipeline_step_time) ─────────────
    compute_time: float = 0.0  # Useful compute on critical path, excluding bubble
    fwd_compute: float = 0.0  # Forward compute only (excludes all comm)
    bwd_compute: float = 0.0  # Backward compute only (excludes comm AND recompute)
    recompute_time: float = 0.0  # Activation-recompute fwd-redo on critical path.
    # 0 with no recompute policy; >0 for full/selective.
    # Part of compute_time, attributed out of bwd_compute.
    recompute_time_raw: float = 0.0  # Raw recompute magnitude = M × max-over-stages
    # per-mb recompute. NOT in step_time / compute_time:
    # when the recomputed stage is not the pipeline
    # bottleneck this work is hidden and recompute_time
    # (critical-path) is 0 while this stays > 0.
    exposed_comm: float = 0.0  # Comm on critical path = Σ *_exposed fields

    # Per-group exposed comm (Σ = exposed_comm)
    tp_exposed: float = 0.0  # TP RS/AG (after CoC/MC2 reduction)
    cp_exposed: float = 0.0  # CP A2A
    ep_exposed: float = 0.0  # EP A2A (after wave-overlap reduction)
    pp_exposed: float = 0.0  # PP P2P
    # dp_exposed declared in pipeline section; equals DP group contribution to exposed_comm

    # Hidden comm — runs in parallel with compute, NOT on critical path
    hidden_comm: float = 0.0  # Total hidden = Σ *_hidden fields
    dp_hidden: float = 0.0  # DP AR absorbed in pipeline bubble
    tp_hidden: float = 0.0  # TP hidden by CoC/MC2
    ep_hidden: float = 0.0  # EP hidden by wave-overlap
    pp_hidden: float = 0.0  # PP P2P hidden by DualPipe/DualPipeV bwd_dw stream

    # Total comm volume = exposed + hidden
    total_comm_volume: float = 0.0  # All comm in step

    # ── v2 mixed-quant HBM traffic diagnostics ───────────────────────────
    # Per-step HBM bytes attributed to each operand role. ``weight_hbm_gb``
    # drops when routed_expert_weight_dtype switches BF16 → FP4; ``cast_hbm_gb``
    # is 0 when QuantPolicy.assume_all_casts_fused is True and grows when
    # the user opts into unfused diagnostics. All values cover one full
    # training step (M microbatches × per-stage ops, scaled to bottleneck).
    weight_hbm_gb: float = 0.0
    act_hbm_gb: float = 0.0
    grad_hbm_gb: float = 0.0
    cast_hbm_gb: float = 0.0

    def __post_init__(self) -> None:
        # Absolute pipeline-idle time. Derived once here so every composer
        # (and the pp=1 path) gets it without duplicating the expression.
        # The dual-batch branch in pipeline_step_time mutates warmup/cooldown
        # afterwards and re-derives bubble there explicitly.
        if self.bubble == 0.0:
            self.bubble = self.warmup + self.cooldown


def _dp_hide_window(
        cooldown: float,
        steady_bwd_total: float,
        strategy: Strategy,
) -> float:
    """Time window in which DP grad-reduce can be hidden.

    cooldown      — explicit pipeline drain phase (composer-specific).
    steady_bwd_total — total steady-state backward compute on the critical path
                       (for pp=1: bwd · M; for pp>1: bottleneck stage's bwd · M).

    Behavior:
      - If dp_overlap_in_bubble is False → return 0 (DP fully exposed).
      - Else return cooldown + ratio · steady_bwd_total, with ratio drawn from
        strategy.dp_steady_overlap_ratio (default 0.5).

    Callers go through _dp_hidden(), which clamps this window against
    dp_ar_time minus a non-overlappable last-bucket residual.
    """
    if not strategy.dp_overlap_in_bubble:
        return 0.0
    ratio = max(0.0, min(1.0, strategy.dp_steady_overlap_ratio))
    return cooldown + ratio * steady_bwd_total


def _dp_hidden(
        dp_ar_time: float,
        cooldown: float,
        steady_bwd_total: float,
        strategy: Strategy,
) -> float:
    """Hidden portion of DP grad-reduce time (seconds).

    DDP/ZeRO overlaps gradient reduction with backward via bucketing, but the
    last bucket's collective cannot start until the final gradient is produced.
    A residual of ~dp_ar_time / dp_grad_buckets is therefore always exposed,
    even when the hide window (_dp_hide_window) is effectively unbounded —
    which previously forced dp_exposed to exactly 0 for any realistic
    compute/comm ratio.

    Returns the amount of dp_ar_time that is hidden; the caller computes
    dp_exposed = dp_ar_time - hidden.
    """
    if dp_ar_time <= 0:
        return 0.0
    window = _dp_hide_window(cooldown, steady_bwd_total, strategy)
    n = max(1, strategy.dp_grad_buckets)
    max_hidable = dp_ar_time * (1.0 - 1.0 / n)
    return min(window, max_hidable)


def _dualbatch_bubble_floor(pp: int, V: int, t_stage_max: float) -> float:
    """Irreducible fill/drain for antiparallel micro-batch streams.

    Two streams running in opposite directions cannot eliminate the
    fill/drain region at the pipeline boundaries; the minimum achievable
    bubble is (pp-1)/(2V) * t_stage_max.
    """
    return (pp - 1) / (2.0 * V) * t_stage_max


class PipelineComposer(ABC):

    @abstractmethod
    def compose(
            self,
            stage_times: list[StageTime],
            M: int,
            pp: int,
            dp_ar_time: float,
            strategy: Strategy,
    ) -> StepResult:
        ...


class OneF1BComposer(PipelineComposer):

    def compose(
            self,
            stage_times: list[StageTime],
            M: int,
            pp: int,
            dp_ar_time: float,
            strategy: Strategy,
    ) -> StepResult:
        """Standard 1F1B pipeline schedule.

        Uses bottleneck stage times to avoid underestimating bubble:
        warmup   = (pp - 1) * max(t_fwd[s])
        steady   = M * max(t_fwd[s] + t_bwd[s])
        cooldown = (pp - 1) * max(t_bwd[s])
        step     = warmup + steady + cooldown + dp_ar_exposed
        """
        if pp == 1:
            # No pipeline: just fwd + bwd for single stage
            st = stage_times[0] if stage_times else StageTime()
            step = st.fwd + st.bwd
            steady_bwd_total = st.bwd * M
            hidden = _dp_hidden(dp_ar_time, 0.0, steady_bwd_total, strategy)
            dp_exposed = dp_ar_time - hidden

            ideal_step = M * (st.fwd + st.bwd)
            bubble_frac = 0.0

            return StepResult(
                step_time=step * M + dp_exposed,
                pipeline_time=step * M + dp_exposed,
                bubble_fraction=bubble_frac,
                warmup=0.0,
                steady=step * M,
                cooldown=0.0,
                dp_exposed=dp_exposed,
                dp_hidden=hidden,
                schedule_name="1f1b",
                warmup_steps=0,
                cooldown_steps=0,
                warmup_fwd=0.0,
                warmup_bwd=0.0,
                steady_fwd=st.fwd * M,
                steady_bwd=st.bwd * M,
                cooldown_fwd=0.0,
                cooldown_bwd=0.0,
                steady_fwd_per_mb=st.fwd,
                steady_bwd_per_mb=st.bwd,
                steady_per_mb=step,
            )

        # With pipeline parallelism
        t_fwd_max = max(st.fwd for st in stage_times) if stage_times else 0
        t_bwd_max = max(st.bwd for st in stage_times) if stage_times else 0
        t_stage_max = max(st.fwd + st.bwd for st in stage_times) if stage_times else 0

        warmup = (pp - 1) * t_fwd_max
        steady = M * t_stage_max
        cooldown = (pp - 1) * t_bwd_max

        if strategy.dualbatch and pp > 1:
            floor = _dualbatch_bubble_floor(pp, 1, t_stage_max)
            bubble = min(warmup + cooldown, floor)
            warmup = bubble / 2.0
            cooldown = bubble / 2.0
        else:
            bubble = warmup + cooldown

        steady_bwd_total = M * t_bwd_max
        hidden = _dp_hidden(dp_ar_time, cooldown, steady_bwd_total, strategy)
        dp_exposed = dp_ar_time - hidden

        step = warmup + steady + cooldown + dp_exposed
        bubble_frac = bubble / step if step > 0 else 0.0

        return StepResult(
            step_time=step,
            pipeline_time=step,
            bubble_fraction=bubble_frac,
            warmup=warmup,
            steady=steady,
            cooldown=cooldown,
            dp_exposed=dp_exposed,
            dp_hidden=hidden,
            schedule_name="1f1b",
            warmup_steps=pp - 1,
            cooldown_steps=pp - 1,
            warmup_fwd=warmup,
            warmup_bwd=0.0,
            steady_fwd=M * t_fwd_max,
            steady_bwd=M * t_bwd_max,
            cooldown_fwd=0.0,
            cooldown_bwd=cooldown,
            steady_fwd_per_mb=t_fwd_max,
            steady_bwd_per_mb=t_bwd_max,
            steady_per_mb=t_stage_max,
        )


class Interleaved1F1BComposer(PipelineComposer):
    """VPP / Interleaved 1F1B pipeline schedule.

    Each device holds `vpp_chunks` virtual stages, reducing pipeline
    bubble by interleaving forward and backward passes.

    With V virtual stages per device, each virtual stage has 1/V of the
    layers, so per-virtual-stage times are t_fwd/V and t_bwd/V.
    The warmup/cooldown fill (pp-1) pipeline slots but each slot is
    1/V the time, giving a bubble that is V times smaller than standard 1F1B.
    """

    def compose(
            self,
            stage_times: list[StageTime],
            M: int,
            pp: int,
            dp_ar_time: float,
            strategy: Strategy,
    ) -> StepResult:
        V = strategy.vpp_chunks
        if V <= 1 or pp <= 1:
            return OneF1BComposer().compose(stage_times, M, pp, dp_ar_time, strategy)

        t_fwd_max = max(st.fwd for st in stage_times) if stage_times else 0
        t_bwd_max = max(st.bwd for st in stage_times) if stage_times else 0
        t_stage_max = max(st.fwd + st.bwd for st in stage_times) if stage_times else 0

        warmup = (pp - 1) * t_fwd_max / V
        steady = M * t_stage_max
        cooldown = (pp - 1) * t_bwd_max / V

        if strategy.dualbatch and pp > 1:
            floor = _dualbatch_bubble_floor(pp, V, t_stage_max)
            bubble = min(warmup + cooldown, floor)
            warmup = bubble / 2.0
            cooldown = bubble / 2.0
        else:
            bubble = warmup + cooldown

        steady_bwd_total = M * t_bwd_max
        hidden = _dp_hidden(dp_ar_time, cooldown, steady_bwd_total, strategy)
        dp_exposed = dp_ar_time - hidden

        step = warmup + steady + cooldown + dp_exposed
        bubble_frac = bubble / step if step > 0 else 0.0

        return StepResult(
            step_time=step,
            pipeline_time=step,
            bubble_fraction=bubble_frac,
            warmup=warmup,
            steady=steady,
            cooldown=cooldown,
            dp_exposed=dp_exposed,
            dp_hidden=hidden,
            schedule_name="i1f1b",
            warmup_steps=max(1, -(-(pp - 1) // V)),
            cooldown_steps=max(1, -(-(pp - 1) // V)),
            warmup_fwd=warmup,
            warmup_bwd=0.0,
            steady_fwd=M * t_fwd_max,
            steady_bwd=M * t_bwd_max,
            cooldown_fwd=0.0,
            cooldown_bwd=cooldown,
            steady_fwd_per_mb=t_fwd_max,
            steady_bwd_per_mb=t_bwd_max,
            steady_per_mb=t_stage_max,
        )


class DualPipeComposer(PipelineComposer):
    """DualPipe schedule — two anti-parallel micro-batch streams.

    Reference: DualPipe README bubble formula (PP/2−1)(F&B+B−3W)
      F&B = max(F,B): execution time of two mutually overlapped F+B chunks
      B:   one additional full backward exposed per warmup/cooldown slot
      3W:  three bwd_dw chunks deferred out of each bubble slot
    pp=2: factor = 0 → zero bubble (both streams start simultaneously).
    """

    def compose(
            self,
            stage_times: list[StageTime],
            M: int,
            pp: int,
            dp_ar_time: float,
            strategy: Strategy,
    ) -> StepResult:
        if pp <= 1:
            return OneF1BComposer().compose(stage_times, M, pp, dp_ar_time, strategy)

        t_fwd_max = max((st.fwd for st in stage_times), default=0.0)
        t_bwd_max = max((st.bwd for st in stage_times), default=0.0)
        # 一个阶段分成F、B，先F后B 时间依然是加法
        t_stage_max = t_fwd_max + t_bwd_max
        t_fb = max(t_fwd_max, t_bwd_max)  # F&B = max(F,B)
        t_w = max((st.bwd_dw for st in stage_times), default=0.0)  # W = bwd_dw

        # Per design doc §7.2.1 and §7.2.4:
        #   warmup   = (PP/2-1) * max(F&B - 2*W, ZB_BUBBLE_FLOOR)
        #   cooldown = (PP/2-1) * max(B - W, ZB_BUBBLE_FLOOR)
        #   bubble   = warmup + cooldown = (PP/2-1)(F&B+B-3W) with floor per-slot
        #
        # DualPipe cooldown uses full B (not B_dx) because the backward drain
        # phase in DualPipe's antiparallel design does NOT split B into dX/dW —
        # W fills part of the backward slot, but unlike ZeroBubble the split
        # is implicit in the overlapping schedule.
        ZB_BUBBLE_FLOOR_PER_TRANSITION = 2e-6  # 2 µs P2P latency per slot
        slots = max(pp / 2 - 1, 0)
        warmup_bubble_per_slot = max(t_fb - 2 * t_w, ZB_BUBBLE_FLOOR_PER_TRANSITION)
        cooldown_bubble_per_slot = max(t_bwd_max - t_w, ZB_BUBBLE_FLOOR_PER_TRANSITION)
        warmup = slots * warmup_bubble_per_slot
        cooldown = slots * cooldown_bubble_per_slot
        bubble = warmup + cooldown
        steady = M * t_stage_max
        steady_bwd_total = M * t_bwd_max
        hidden = _dp_hidden(dp_ar_time, cooldown, steady_bwd_total, strategy)
        dp_exposed = dp_ar_time - hidden

        # =========================================================================
        # 视角 A：真实物理演进视角 (Physical Timeline Perspective)
        # =========================================================================
        #    StepTime = [Steady计算+纯空泡时间] + [Warmup 计算+纯空泡时间] + [cooldown 计算+纯空泡时间] + [暴露通信时间]
        #             =      steadyTime      +     warmupTime      +     cooldownTime      +   dp_exposed

        # =========================================================================
        # 视角 B：代码数学重构视角 (Mathematical Reconstruct Perspective)
        # 相当对计算、空泡时间做了汇总
        #    StepTime = [全流程纯计算时间] + [Warmup空泡时间] + [cooldown空泡时间] + [暴露通信时间]
        #             =      M * t_stage_max      +     warmup      +     cooldown      +   dp_exposed
        # =========================================================================


        step = warmup + steady + cooldown + dp_exposed
        bubble_frac = bubble / step if step > 0 else 0.0

        return StepResult(
            step_time=step,
            pipeline_time=step,
            bubble_fraction=bubble_frac,
            warmup=warmup,
            steady=steady,
            cooldown=cooldown,
            dp_exposed=dp_exposed,
            dp_hidden=hidden,
            schedule_name="dualpipe",
            warmup_steps=max(0, pp // 2 - 1),
            cooldown_steps=max(0, pp // 2 - 1),
            warmup_fwd=warmup,
            warmup_bwd=0.0,
            steady_fwd=M * t_fwd_max,
            steady_bwd=M * t_bwd_max,
            cooldown_fwd=0.0,
            cooldown_bwd=cooldown,
            steady_fwd_per_mb=t_fwd_max,
            steady_bwd_per_mb=t_bwd_max,
            steady_per_mb=t_stage_max,
        )


class DualPipeVComposer(PipelineComposer):
    """DualPipeV — DualPipe with virtual stage splitting.

    Combines DualPipe's F/B parallelism with VPP's virtual stages.
    Reference: DualPipe README bubble formula (PP/2−1)(F&B+B−3W)/V
    Same per-slot cost as DualPipe, reduced by V virtual-stage chunks.
    """

    def compose(
            self,
            stage_times: list[StageTime],
            M: int,
            pp: int,
            dp_ar_time: float,
            strategy: Strategy,
    ) -> StepResult:
        V = strategy.vpp_chunks
        if V <= 1:
            return DualPipeComposer().compose(stage_times, M, pp, dp_ar_time, strategy)
        if pp <= 1:
            return OneF1BComposer().compose(stage_times, M, pp, dp_ar_time, strategy)

        t_fwd_max = max((st.fwd for st in stage_times), default=0.0)
        t_bwd_max = max((st.bwd for st in stage_times), default=0.0)
        t_stage_max = t_fwd_max + t_bwd_max
        t_fb = max(t_fwd_max, t_bwd_max)  # F&B = max(F,B)
        t_w = max((st.bwd_dw for st in stage_times), default=0.0)  # W = bwd_dw

        # Per design doc §7.2.1 and §7.2.4 (V-chunk variant):
        #   warmup   = (PP/2-1)/V * max(F&B - 2*W, ZB_BUBBLE_FLOOR)
        #   cooldown = (PP/2-1)/V * max(B - W, ZB_BUBBLE_FLOOR)
        #   bubble   = warmup + cooldown = (PP/2-1)(F&B+B-3W)/V with floor per-slot
        ZB_BUBBLE_FLOOR_PER_TRANSITION = 2e-6  # 2 µs P2P latency per slot
        slots = max(pp / 2 - 1, 0)
        warmup_bubble_per_slot = max(t_fb - 2 * t_w, ZB_BUBBLE_FLOOR_PER_TRANSITION)
        cooldown_bubble_per_slot = max(t_bwd_max - t_w, ZB_BUBBLE_FLOOR_PER_TRANSITION)
        warmup = slots / V * warmup_bubble_per_slot
        cooldown = slots / V * cooldown_bubble_per_slot
        bubble = warmup + cooldown
        steady = M * t_stage_max
        steady_bwd_total = M * t_bwd_max
        hidden = _dp_hidden(dp_ar_time, cooldown, steady_bwd_total, strategy)
        dp_exposed = dp_ar_time - hidden

        step = warmup + steady + cooldown + dp_exposed
        bubble_frac = bubble / step if step > 0 else 0.0

        return StepResult(
            step_time=step,
            pipeline_time=step,
            bubble_fraction=bubble_frac,
            warmup=warmup,
            steady=steady,
            cooldown=cooldown,
            dp_exposed=dp_exposed,
            dp_hidden=hidden,
            schedule_name="dualpipev",
            warmup_steps=max(0, (pp // 2 - 1 + V - 1) // V),
            cooldown_steps=max(0, (pp // 2 - 1 + V - 1) // V),
            warmup_fwd=warmup,
            warmup_bwd=0.0,
            steady_fwd=M * t_fwd_max,
            steady_bwd=M * t_bwd_max,
            cooldown_fwd=0.0,
            cooldown_bwd=cooldown,
            steady_fwd_per_mb=t_fwd_max,
            steady_bwd_per_mb=t_bwd_max,
            steady_per_mb=t_stage_max,
        )


class ZeroBubbleComposer(PipelineComposer):
    """ZeroBubble schedule with backward split into dX and dW phases.

    The critical path is the per-stage F+B time.  Weight-gradient work can
    be delayed to fill pipeline bubbles, so the exposed bubble is reduced by
    the bottleneck stage's dW time:

        step = M * t_stage + (pp - 1) * max(t_stage - t_w, ZB_BUBBLE_FLOOR)

    where ZB_BUBBLE_FLOOR represents the residual per-transition P2P latency
    that ZB-1P/ZB-V cannot eliminate.
    """

    def compose(
            self,
            stage_times: list[StageTime],
            M: int,
            pp: int,
            dp_ar_time: float,
            strategy: Strategy,
    ) -> StepResult:
        if pp <= 1:
            return OneF1BComposer().compose(stage_times, M, pp, dp_ar_time, strategy)

        bottleneck = max(stage_times, key=lambda st: st.fwd + st.bwd) if stage_times else StageTime()
        t_stage = bottleneck.fwd + bottleneck.bwd
        t_fwd = bottleneck.fwd
        t_bwd = bottleneck.bwd
        t_w = bottleneck.bwd_dw

        # ZB-1P/ZB-V keep a residual per-transition bubble even when t_w ≈ t_stage.
        # The empirical floor is roughly 2 microseconds (P2P latency) per pp-1
        # transition. This avoids the "0 bubble" artifact in search.
        #
        # Per design doc §7.2.1 and §7.2.4:
        #   warmup   = (PP-1) * max(F - W, ZB_BUBBLE_FLOOR)
        #   cooldown = (PP-1) * max(B_dx - W, ZB_BUBBLE_FLOOR)
        #   bubble   = warmup + cooldown = (PP-1)(F+B_dx-2W) with floor per-transition
        #
        # Note: cooldown uses B_dx (activation gradient backward) not B (full backward),
        # because W (weight gradient) can fill only the activation-gradient portion.
        ZB_BUBBLE_FLOOR_PER_TRANSITION = 2e-6  # 2 µs P2P latency per pp transition
        warmup_bubble_per_stage = max(t_fwd - t_w, ZB_BUBBLE_FLOOR_PER_TRANSITION)
        cooldown_bubble_per_stage = max(bottleneck.bwd_dx - t_w, ZB_BUBBLE_FLOOR_PER_TRANSITION)
        warmup = (pp - 1) * warmup_bubble_per_stage
        cooldown = (pp - 1) * cooldown_bubble_per_stage
        bubble = warmup + cooldown

        if strategy.dualbatch and pp > 1:
            V = max(1, strategy.vpp_chunks)
            dual_batch_bubble = _dualbatch_bubble_floor(pp, V, t_stage)
            if dual_batch_bubble < bubble:
                bubble = dual_batch_bubble
                warmup = bubble / 2.0
                cooldown = bubble / 2.0

        steady = M * t_stage

        steady_bwd_total = M * t_bwd
        hidden = _dp_hidden(dp_ar_time, cooldown, steady_bwd_total, strategy)
        dp_exposed = dp_ar_time - hidden

        step = warmup + steady + cooldown + dp_exposed
        bubble_frac = bubble / step if step > 0 else 0.0

        return StepResult(
            step_time=step,
            pipeline_time=step,
            bubble_fraction=bubble_frac,
            warmup=warmup,
            steady=steady,
            cooldown=cooldown,
            dp_exposed=dp_exposed,
            dp_hidden=hidden,
            schedule_name="zb",
            warmup_steps=pp - 1,
            cooldown_steps=pp - 1,
            warmup_fwd=warmup,
            warmup_bwd=0.0,
            steady_fwd=M * t_fwd,
            steady_bwd=M * t_bwd,
            cooldown_fwd=0.0,
            cooldown_bwd=cooldown,
            steady_fwd_per_mb=t_fwd,
            steady_bwd_per_mb=t_bwd,
            steady_per_mb=t_stage,
        )


COMPOSER_BY_SCHED: dict[PPSched, type[PipelineComposer]] = {
    PPSched.ONE_F_ONE_B: OneF1BComposer,
    PPSched.INTERLEAVED: Interleaved1F1BComposer,
    PPSched.ZERO_BUBBLE: ZeroBubbleComposer,
    PPSched.DUALPIPE: DualPipeComposer,
    PPSched.DUALPIPE_V: DualPipeVComposer,
}


def pipeline_step_time(
        graph: Graph,
        model: ModelSpec,
        system: SystemSpec,
        strategy: Strategy,
) -> StepResult:
    """Compute full training step time from IR + strategy.

    Graph-native counterpart: TrainingPipelinePass in transform/analysis/training.py
    — uses per-stage timelines from DAGScheduler instead of formula-based stage_time().

    Current path (IR-based reference implementation):
      - Use IR-level stage_time() from graph.ops_for_stage()
      - Apply 1F1B/VPP/DualPipe formulas on aggregated times
      - DP overlap uses simple bubble-window heuristic

    Both paths use the same PipelineComposer classes (OneF1BComposer, Interleaved1F1BComposer,
    ZeroBubbleComposer, DualPipeComposer, DualPipeVComposer) and converge to the same
    StepResult interface for compatibility.
    """
    pp = strategy.pp
    M = strategy.num_microbatches()

    # One resolver per estimate() call. ParallelGroups is enumerated
    # lazily on first .time(c) / .ranks() / .link() lookup, then cached
    # so per-stage and per-collective queries all share it.
    domain = CommDomain(system=system, strategy=strategy)

    # Compute per-stage times
    stage_ids = _assign_stages(model, strategy)
    stage_times: list[StageTime] = []

    for s in range(pp):
        layer_ids = stage_ids[s]
        stage_ops = graph.ops_for_stage(layer_ids)

        stage_colls = [
            c for c in graph.collectives
            if any(
                (c.inserted_after and c.inserted_after.startswith(f"L{lid}.")) or
                (c.inserted_before and c.inserted_before.startswith(f"L{lid}."))
                for lid in layer_ids
            )
        ]

        st = stage_time(stage_ops, stage_colls, model, system, strategy, domain=domain)
        stage_times.append(st)

    # Compute DP allreduce time and PP P2P overhead
    comm_times = total_comm_time(graph, model, system, strategy, domain=domain)
    dp_ar_time = comm_times.get("dp_grad_reduce", 0.0)

    # ── PP P2P + recompute placement (schedule-specific) ─────────────────
    # Both pp_p2p (activation send per stage boundary) and recompute
    # (forward redo for backward) are backward-side overhead. The model
    # decides where they sit on the timeline based on schedule semantics:
    #
    #   • Serial (1F1B / VPP / ZB)
    #       PP P2P is added once to fwd and once to bwd (boundary send each
    #       direction). Recompute is added to bwd_dx (lives inside backward
    #       critical path, as it physically does). Behavior is bit-exact
    #       with the legacy ``t_bwd_dx += recompute_t`` (stage.py) +
    #       ``+= pp_p2p`` (here) — see test_anchor_step_time_strict.
    #
    #   • Dual-stream (DualPipe / DualPipeV)
    #       The W (bwd_dw) phase runs on a second stream parallel with
    #       activation P2P + recompute. Only the residual beyond ``bwd_dw``
    #       extent stays on the critical path; the hidden share is split
    #       proportionally between pp_p2p and recompute so the report
    #       attributes each correctly.
    #
    # We track the exposed amounts per-stage so:
    #   (a) per-stage recompute distribution is preserved for the
    #       non-bottleneck-hidden-by-bottleneck case
    #       (tests/training/test_bubble_recompute_breakdown.py::test_dense_recompute_pipeline_hidden_raw_visible)
    #   (b) post-compose pp_exposed extraction uses the bottleneck stage's
    #       exposed sum, not a hardcoded ``2*pp_p2p``.
    pp_p2p = comm_times.get("pp_p2p", 0.0)
    is_dual = strategy.pp_schedule in (PPSched.DUALPIPE, PPSched.DUALPIPE_V)
    # Capture raw recompute max BEFORE augmentation — recompute_time_raw
    # uses this (the work that would be done if nothing hid it).
    recompute_raw_per_mb = max((st.recompute for st in stage_times), default=0.0)

    pp_p2p_fwd_exposed: list[float] = []
    pp_p2p_bwd_exposed: list[float] = []
    pp_p2p_hidden: list[float] = []
    recompute_exposed: list[float] = []
    for st in stage_times:
        if pp == 1 or not is_dual:
            pp_p2p_fwd_exposed.append(pp_p2p)
            pp_p2p_bwd_exposed.append(pp_p2p)
            pp_p2p_hidden.append(0.0)
            recompute_exposed.append(st.recompute)
            continue
        # Dual-stream hide. 2x pp_p2p (one fwd-boundary + one bwd-boundary
        # per mb) and st.recompute share the bwd_dw budget on this stage.
        # Attribute the bwd_dw hide window to PP first: PP is communication
        # scheduled on the opposite stream, while any remaining residual from
        # a large recompute belongs to recompute itself, not PP.
        pp_p2p_total = 2.0 * pp_p2p
        total_to_hide = pp_p2p_total + st.recompute
        if total_to_hide <= 0:
            pp_p2p_fwd_exposed.append(0.0)
            pp_p2p_bwd_exposed.append(0.0)
            pp_p2p_hidden.append(0.0)
            recompute_exposed.append(0.0)
            continue
        pp_p2p_share = max(0.0, pp_p2p_total - st.bwd_dw)
        pp_p2p_hide = pp_p2p_total - pp_p2p_share
        recompute_hide_budget = max(0.0, st.bwd_dw - pp_p2p_total)
        recompute_share = max(0.0, st.recompute - recompute_hide_budget)
        # Split the pp_p2p residual evenly between fwd and bwd boundaries.
        pp_p2p_fwd_exposed.append(pp_p2p_share / 2.0)
        pp_p2p_bwd_exposed.append(pp_p2p_share / 2.0)
        pp_p2p_hidden.append(pp_p2p_hide)
        recompute_exposed.append(recompute_share)

    # Build the augmented stage_times the composer needs to see (recompute +
    # pp_p2p on the bwd critical path) WITHOUT mutating the original list.
    # ``step.per_stage`` will expose the PURE original copy so downstream
    # consumers (Excel / HTML / JSON / search report) see per-stage bwd that
    # is just compute + native exposed comm — no pp_p2p, no recompute.
    if pp_p2p > 0 or recompute_raw_per_mb > 0:
        stage_times_aug = [
            StageTime(
                fwd=st.fwd + pp_p2p_fwd_exposed[i],
                # bwd = bwd_dx + bwd_dw, so adding to bwd_dx propagates to bwd.
                bwd=st.bwd + pp_p2p_bwd_exposed[i] + recompute_exposed[i],
                bwd_dx=st.bwd_dx + pp_p2p_bwd_exposed[i] + recompute_exposed[i],
                bwd_dw=st.bwd_dw,
                comm_fwd=st.comm_fwd + pp_p2p_fwd_exposed[i],
                comm_bwd=st.comm_bwd + pp_p2p_bwd_exposed[i],
                ep_hidden=st.ep_hidden,
                tp_hidden=st.tp_hidden,
                tp_exposed=st.tp_exposed,
                ep_exposed=st.ep_exposed,
                cp_exposed=st.cp_exposed,
                recompute=recompute_exposed[i],
            )
            for i, st in enumerate(stage_times)
        ]
    else:
        stage_times_aug = stage_times

    # Compose using the augmented view (composer needs recompute on the
    # critical path for its bubble formula).
    composer_cls = COMPOSER_BY_SCHED.get(strategy.pp_schedule, OneF1BComposer)
    composer = composer_cls()
    step = composer.compose(stage_times_aug, M, pp, dp_ar_time, strategy)

    # per_stage exposes ORIGINAL (pure) stage_times. ``st.bwd`` here is pure
    # backward compute + native exposed comm. ``st.recompute`` is the FULL
    # per-mb recompute work (pre-hide) — even when dual-stream hiding makes
    # the critical-path residual smaller than this.
    step.per_stage = stage_times

    # === Communication and compute breakdown ===
    # Placed after compose() so step.pipeline_time / step.dp_exposed are final.
    #
    # Invariants enforced here:
    #   pipeline_time    = compute_time + exposed_comm + bubble   (exact)
    #   exposed_comm     = tp + cp + ep + pp + dp _exposed       (exact)
    #   hidden_comm      = dp_hidden + tp_hidden + ep_hidden + pp_hidden (exact)
    #   total_comm_volume = exposed_comm + hidden_comm            (exact)
    #
    # comm_fwd/comm_bwd in StageTime are already-reduced exposed portions
    # (TP CoC/MC2 and EP wave-overlap applied inside stage_time()).
    # The bottleneck stage ratio gives a schedule-agnostic critical-path split.

    # Bottleneck selection drives the timeline-based attribution (scale,
    # comm scaling, recompute_time). Must use the AUGMENTED view so it
    # matches what the composer saw — picking from pure stage_times would
    # under-count stages whose timeline weight comes from recompute.
    bot_idx, s_bot = max(
        enumerate(stage_times_aug), key=lambda kv: kv[1].fwd + kv[1].bwd
    )
    bot_total = s_bot.fwd + s_bot.bwd

    # Non-bubble pipeline work excludes the warmup/cooldown bubble and the
    # exposed DP AR tail (which sits after cooldown). This is the useful
    # stage work used to attribute compute vs exposed comm.
    _useful_pipeline_time = max(
        0.0, step.pipeline_time - step.dp_exposed - step.bubble,
    )

    # ── Exposed comm: bottom-up from per-type fields ─────────────────────
    # Scale each comm type's exposed time directly from bottleneck stage.
    # No remain, no ratio splits, no conservation caps.
    # Invariant: tp + ep + cp + pp + dp = exposed_comm (by construction)
    if bot_total > 0:
        scale = _useful_pipeline_time / bot_total
        step.tp_exposed = scale * s_bot.tp_exposed
        step.ep_exposed = scale * s_bot.ep_exposed
        step.cp_exposed = scale * s_bot.cp_exposed
        # PP exposed: bottleneck stage's exposed fwd+bwd boundary cost
        # (already schedule-adjusted in the augmentation block above).
        # For 1F1B/VPP/ZB this is ``2*pp_p2p`` (bit-exact with legacy).
        # For DualPipe(V) it is the residual share after bwd_dw hides.
        step.pp_exposed = scale * (
            pp_p2p_fwd_exposed[bot_idx] + pp_p2p_bwd_exposed[bot_idx]
        )
        step.tp_hidden = scale * s_bot.tp_hidden
        step.ep_hidden = scale * s_bot.ep_hidden
        step.pp_hidden = scale * pp_p2p_hidden[bot_idx]
    else:
        step.tp_exposed = step.ep_exposed = step.cp_exposed = step.pp_exposed = 0.0
        step.tp_hidden = step.ep_hidden = step.pp_hidden = 0.0

    exposed_comm_excl_dp = (step.tp_exposed + step.ep_exposed
                            + step.cp_exposed + step.pp_exposed)
    step.exposed_comm = exposed_comm_excl_dp + step.dp_exposed
    # compute_time exact: compute + exposed comm + bubble == pipeline_time.
    step.compute_time = max(0.0, step.pipeline_time - step.exposed_comm - step.bubble)

    # Split compute_time into fwd vs bwd using bottleneck stage ratios.
    # Each stage's fwd/bwd already includes embedded exposed comm, so we
    # subtract comm_fwd/comm_bwd to get pure compute, then scale to the
    # full pipeline's compute_time.
    fwd_compute_per_mb = max(0.0, s_bot.fwd - s_bot.comm_fwd)
    bwd_compute_per_mb = max(0.0, s_bot.bwd - s_bot.comm_bwd)
    compute_per_mb = fwd_compute_per_mb + bwd_compute_per_mb
    if compute_per_mb > 0:
        fwd_ratio = fwd_compute_per_mb / compute_per_mb
        step.fwd_compute = step.compute_time * fwd_ratio
        step.bwd_compute = step.compute_time * (1.0 - fwd_ratio)
    else:
        step.fwd_compute = step.compute_time
        step.bwd_compute = 0.0

    # ── Recompute as its own term, attributed out of bwd_compute ──────────
    # Recompute (activation-recompute fwd redo) is backward-side work. For
    # serial schedules (1F1B / VPP / ZB) it sits on the critical path and
    # the augmentation block above adds it to bwd_dx so the composer
    # timeline carries it; we then split it back OUT here so the report
    # shows it as its own term:
    #   compute_time = fwd_compute + bwd_compute + recompute_time   (exact)
    # For DualPipe(V) the augmentation already trimmed st.recompute to the
    # post-bwd_dw-hide residual, so ``M * s_bot.recompute`` is the exposed
    # critical-path portion (the hidden part is absorbed by bwd_dw).
    #
    # recompute_time_raw is the work that WOULD be done if nothing hid it
    # (M × max across stages, pre-augmentation). recompute_time_raw stays a
    # reporting field — it does not affect step_time.
    step.recompute_time_raw = M * recompute_raw_per_mb

    if bwd_compute_per_mb > 0 and s_bot.recompute > 0:
        bottleneck_recompute = M * s_bot.recompute
        step.recompute_time = min(
            step.bwd_compute, bottleneck_recompute, step.recompute_time_raw
        )
        step.bwd_compute -= step.recompute_time
    else:
        step.recompute_time = 0.0

    # ── Hidden comm ───────────────────────────────────────────────────────
    # DP AR hidden in pipeline bubble — set by composer (and updated by
    # dual-batch above). Verify the invariant: dp_hidden = dp_ar_time - dp_exposed.
    step.dp_hidden = max(0.0, dp_ar_time - step.dp_exposed)

    step.hidden_comm = (
        step.dp_hidden + step.tp_hidden + step.ep_hidden + step.pp_hidden
    )
    step.total_comm_volume = step.exposed_comm + step.hidden_comm

    # Optimizer time and communication
    opt_time = _compute_optimizer_time(model, system, strategy)
    opt_comm_parts = optimizer_comm_time(model, system, strategy, domain=domain)
    ag_time = opt_comm_parts.get("muon_ag", 0.0)
    rs_time = opt_comm_parts.get("muon_rs", 0.0)

    # Moonshot rotation hides two parts:
    #   1. AG (param gather, before NS) hides under two windows:
    #        (a) opt_time  — chunk-pipelined with NS compute (max(AG, NS)).
    #        (b) the remainder of the NEXT iteration's fwd window, mirroring
    #            Megatron's --overlap-param-gather (per-layer chunked AG
    #            absorbed into next-step F1B / warmup forward).
    #      AG and RS compete for the same fwd window — RS books its share
    #      first (it lands in time right after NS), AG uses what's left.
    #   2. RS (updated-param scatter, after NS) overlaps with the NEXT
    #      iteration's first microbatch(es) of forward — steady-state
    #      assumption, matches Megatron-Core distributed-optimizer behavior.
    #      Hide window: warmup_fwd for pp>1; one microbatch fwd for pp=1.
    from zrt.training.models.optimizer import moonshot_optimizer_hiding
    rotation_active = (
            strategy.optimizer.value == "muon"
            and strategy.muon_config is not None
            and strategy.muon_config.rotation
    )
    fwd_window = max(step.warmup_fwd, step.steady_fwd_per_mb) if rotation_active else 0.0
    opt_comm_exposed_s, opt_comm_hidden_s = moonshot_optimizer_hiding(
        compute_us=opt_time * 1e6,
        ag_us=ag_time * 1e6,
        rs_us=rs_time * 1e6,
        fwd_window_us=fwd_window * 1e6,
        rotation=rotation_active,
    )
    opt_comm_exposed = opt_comm_exposed_s / 1e6
    opt_comm_hidden = opt_comm_hidden_s / 1e6

    step.optimizer_time = opt_time
    step.optimizer_comm = opt_comm_exposed
    step.optimizer_comm_hidden = opt_comm_hidden

    # Memory breakdown
    step.memory = memory_breakdown(graph, model, system, strategy)

    # v2 HBM traffic diagnostics. Aggregate per-step bytes by operand
    # role for the entire graph (× M microbatches per step). This is a
    # report-only signal — the values are NOT on the critical path of
    # any composer / mfu calculation. See §4.8 of the v2 doc.
    _populate_hbm_traffic(step, graph, model, system, strategy)

    # MFU uses pipeline_time (excludes optimizer, per design doc §5.5.2)
    step.mfu = compute_mfu(model, strategy, system, step.pipeline_time, graph)

    # HFU
    step.hfu = compute_hfu(model, strategy, system, step.pipeline_time, graph)
    step.mfu_native = compute_mfu_native(model, strategy, system, step.pipeline_time, graph)

    # Assemble step_time additively: pipeline + optimizer.
    # Invariant holds by construction: step_time = pipeline_time + optimizer_time + optimizer_comm
    step.step_time = step.pipeline_time + step.optimizer_time + step.optimizer_comm

    return step


def _populate_hbm_traffic(
    step: "StepResult", graph: Graph, model: ModelSpec,
    system: SystemSpec, strategy: Strategy,
) -> None:
    """Sum per-op HBM bytes by operand role and write into ``step``.

    Splits each matmul's ``fwd_bytes`` into the (A, W, C) terms using the
    ``OpDtypeBundle`` to avoid double-counting. Attention / elementwise
    ops contribute to ``act_hbm``. Backward bytes (dx + dw) go to
    ``grad_hbm``. cast ops feed ``cast_hbm`` independently.

    Multiplied by ``M = num_microbatches()`` and divided by 1 GiB (== 2**30
    bytes) at the end. Result is **per-step per-rank** — the graph is
    already TP/EP-sharded by build_graph.
    """
    from zrt.training.models.flops import op_cost as _op_cost
    from zrt.training.models.quant import resolve_op_dtypes as _bundle

    GB = float(1 << 30)
    M = max(1, strategy.num_microbatches())

    weight_bytes = 0.0
    act_bytes = 0.0
    grad_bytes = 0.0
    cast_bytes = 0.0

    for op in graph.ops:
        cost = _op_cost(op, model, system)
        if op.kind == "cast":
            cast_bytes += cost.fwd_bytes + cost.dx_bytes
            continue

        # Forward: split bytes into weight vs activation if matmul.
        if op.kind == "matmul":
            d = _bundle(op, model)
            # _matmul_cost rebuilds these shapes; recompute to split.
            meta_k = op.meta.get("k", 0)
            use_meta = (
                op.meta.get("fused_weight_dims", False)
                or not op.inputs or not op.outputs
                or (meta_k > 0 and op.inputs[0].shape_logical[-1] != meta_k)
            )
            if use_meta:
                m = op.meta.get("m", 0)
                n = op.meta.get("n_local", op.meta.get("n", 0))
                k = op.meta.get("k_local", op.meta.get("k", 0))
            else:
                m = op.inputs[0].shape_local[0]
                k = op.inputs[0].shape_local[-1]
                n = op.outputs[0].shape_local[-1]
            mult = op.meta.get("fwd_multiplier", 1.0)
            # Apply fwd_multiplier the same way _matmul_cost folds it into
            # FLOPs; for bytes the routed-expert fused op visits each of
            # the top_k expert tiles, so weight/act/grad bytes scale too.
            scale = float(mult)
            weight_bytes += scale * (k * n * d.weight.stored_bytes
                                     + k * n * d.weight.stored_bytes  # dx reads W
                                     + k * n * d.grad_weight.stored_bytes)  # dw writes dW
            act_bytes += scale * (m * k * d.in_act.bytes + m * n * d.out_act.bytes)
            grad_bytes += scale * (m * n * d.grad_in.bytes + m * k * d.grad_act.bytes
                                    + m * n * d.grad_in.bytes + m * k * d.in_act.bytes)
        else:
            # Non-matmul: lump all fwd bytes into act, bwd into grad.
            act_bytes += cost.fwd_bytes
            grad_bytes += cost.dx_bytes + cost.dw_bytes

    step.weight_hbm_gb = weight_bytes * M / GB
    step.act_hbm_gb = act_bytes * M / GB
    step.grad_hbm_gb = grad_bytes * M / GB
    step.cast_hbm_gb = cast_bytes * M / GB


def _assign_stages(model: ModelSpec, strategy: Strategy) -> list[list[int]]:
    """Assign layer IDs to PP stages.

    Returns list of pp stage → list of layer IDs.
    """
    n_layers = len(model.layers)
    pp = strategy.pp

    if pp == 1:
        return [list(range(n_layers))]

    if strategy.pp_layer_assignment is not None:
        # Explicit assignment: pp_layer_assignment[i] = stage for layer i
        stages: list[list[int]] = [[] for _ in range(pp)]
        for i, s in enumerate(strategy.pp_layer_assignment):
            if s < pp:
                stages[s].append(i)
        return stages

    # Auto-balance: greedy bin-pack on number of layers
    stages = [[] for _ in range(pp)]
    for i in range(n_layers):
        stages[i % pp].append(i)
    return stages


_ADAM_UPDATE_BYTES_PER_PARAM = 28


def _adam_optimizer_step_time(params: int, system: SystemSpec) -> float:
    """Memory-bound Adam update time for FP32 master/grad/m/v state."""
    bytes_ = max(0, int(params)) * _ADAM_UPDATE_BYTES_PER_PARAM
    bw = effective_hbm_bw_bps(system.gpu, bytes_)
    return bytes_ / bw if bw > 0 else 0.0


def _compute_optimizer_time(model: ModelSpec, system: SystemSpec, strategy: Strategy) -> float:
    """Compute optimizer step time in seconds.

    Uses roofline model with achieved efficiency from perf_tables.
    """
    from zrt.training.spec.model import LayerKind

    P = model.total_params()
    if strategy.ep > 1:
        if strategy.dp < strategy.ep:
            raise ValueError(
                f"dp must be >= ep for expert-DP sharding "
                f"(dp={strategy.dp}, ep={strategy.ep})"
            )
        if strategy.dp % strategy.ep != 0:
            raise ValueError(
                f"dp must be divisible by ep for expert-DP sharding "
                f"(dp={strategy.dp}, ep={strategy.ep})"
            )
    if strategy.tp > 1:
        P //= strategy.tp
    if strategy.pp > 1:
        n_layers = len(model.layers)
        embed = model.vocab * model.hidden * 2
        non_embed = P - embed
        non_embed = int(non_embed * (n_layers / strategy.pp) / n_layers)
        P = non_embed + embed // strategy.pp
    # EP shards expert params across ep ranks: each GPU holds num_experts/ep experts.
    if strategy.ep > 1:
        n_moe = sum(1 for lk in model.layers if lk == LayerKind.MOE)
        if n_moe > 0 and model.moe_ffn > 0:
            expert_p_all = n_moe * 3 * model.hidden * model.moe_ffn * model.num_experts
            # Match the TP+PP sharding already applied to P, otherwise the
            # subtraction below clamps non_expert_p to 0 and we lose all
            # non-routed params from the Muon NS budget.
            expert_p_stage = expert_p_all
            if strategy.tp > 1:
                expert_p_stage //= strategy.tp
            if strategy.pp > 1:
                expert_p_stage //= strategy.pp
            non_expert_p = max(0, P - expert_p_stage)
            P = non_expert_p + expert_p_stage // strategy.ep
    # ZeRO-1/2/3 all shard optimizer states across DP: each GPU updates P/dp params.
    if strategy.zero_stage >= 1:
        P //= strategy.dp

    gpu = system.gpu
    peak_flops = gpu.flops_bf16 * 1e12
    if peak_flops <= 0:
        return 0.0

    if strategy.optimizer.value == "muon":
        K = resolve_muon_ns_steps(strategy.muon_config, model)
        f_muon = (
            strategy.muon_config.muon_param_fraction
            if strategy.muon_config and strategy.muon_config.muon_param_fraction is not None
            else 0.85
        )
        # Architecture-driven NS FLOPs: walk the actual weight-matrix
        # inventory. The legacy P/hidden² path clamps num_matrices to 1
        # under ZeRO-3 + EP and yields a constant ~8 ms across the grid.
        muon_flops = muon_step_flops_from_arch(model, strategy, K, f_muon)
        eff_flops = effective_flops(gpu, Dtype.BF16, muon_flops)
        muon_time = muon_flops / eff_flops if eff_flops > 0 else 0.0
        adam_time = _adam_optimizer_step_time(int(P * (1 - f_muon)), system)
        return muon_time + adam_time
    else:
        return _adam_optimizer_step_time(P, system)


def _compute_optimizer_comm_time(
    model: ModelSpec, system: SystemSpec, strategy: Strategy,
    domain: CommDomain | None = None,
) -> float:
    """Compute optimizer communication time (Muon ZeRO-1 AllGather + ReduceScatter).

    Optional ``domain`` lets the caller reuse a pre-built resolver from
    :func:`pipeline_step_time`. Without it, a local fall-back domain is
    instantiated — equivalent to the previous behavior for back-compat.
    """
    comm_times = optimizer_comm_time(model, system, strategy, domain=domain)
    return comm_times.get("muon_ag", 0.0) + comm_times.get("muon_rs", 0.0)


def util_from_flops(flops: float, peak_flops_total: float, step_time_s: float) -> float:
    """Compute utilization ratio from FLOPs.

    Args:
        flops: Total FLOPs executed (e.g., model FLOPs or model+recompute FLOPs)
        peak_flops_total: Peak FLOP/s of entire system (world_size * per_gpu_peak)
        step_time_s: Step time in seconds

    Returns:
        Utilization ratio capped at 1.0, or 0.0 if inputs are invalid.
    """
    if step_time_s <= 0 or peak_flops_total <= 0:
        return 0.0
    return min(flops / (peak_flops_total * step_time_s), 1.0)


def compute_mfu(
        model: ModelSpec, strategy: Strategy,
        system: SystemSpec, step_time: float,
        graph: Graph,
) -> float:
    """Model FLOPs Utilization.

    MFU = (total_training_flops / PP) / (per_gpu_peak * step_time)

    total_training_flops is per-GPU (the graph models TP/EP-sharded computation),
    so world_size cancels between numerator and denominator. We divide by PP
    because the graph covers all layers but each GPU handles only 1/PP of them.

    Uses actual graph-level FLOP accounting (Σ op_fwd+op_dx+op_dw × M)
    instead of the 6P rule-of-thumb, which overestimates for MoE + low-rank
    architectures (e.g. DeepSeek-V4: 6P gives 30× more FLOPs than actual).
    """
    from zrt.training.models.flops import total_training_flops

    tokens = strategy.global_batch * model.seq_len if strategy.global_batch > 0 else strategy.micro_batch * strategy.dp * model.seq_len
    actual_flops = total_training_flops(graph, model, strategy, system)
    # models sharded computation; cluster-wide FLOPs = per_gpu × world_size,
    # and cluster peak = per_gpu_peak × world_size — the world_size cancels).
    # NOTE: raw theoretical peak by definition — MFU is achieved-vs-peak.
    # Do NOT route through effective_flops (would double-count the
    # utilization already reflected in step_time).
    peak = system.gpu.flops_bf16 * 1e12

    # Divide by PP because total_training_flops includes ALL layer FLOPs,
    # but each GPU only handles 1/pp of the layers. Per-GPU FLOPs = total / pp.
    pp_flops = actual_flops / strategy.pp

    return util_from_flops(pp_flops, peak, step_time)


def compute_hfu(
        model: ModelSpec, strategy: Strategy,
        system: SystemSpec, step_time: float,
        graph: Graph,
) -> float:
    """Hardware FLOPs Utilization — accounts for recomputed activations.

    HFU = (actual_training_flops + recompute_overhead) / (peak * step_time)
    """
    from zrt.training.models.flops import total_training_flops, recompute_overhead_flops

    actual_flops = total_training_flops(graph, model, strategy, system)

    # Peak FLOP/s of single GPU (total_flops is per-GPU because the graph
    rc_overhead = recompute_overhead_flops(graph, model, strategy, system)
    # NOTE: raw theoretical peak by definition (see compute_mfu) — excluded
    # from the unified effective_flops entry.
    peak = system.gpu.flops_bf16 * 1e12

    # Divide by PP (same rationale as compute_mfu)
    pp_flops = (actual_flops + rc_overhead) / strategy.pp

    return util_from_flops(pp_flops, peak, step_time)


def compute_mfu_native(
        model: ModelSpec, strategy: Strategy,
        system: SystemSpec, step_time: float,
        graph: Graph,
) -> float:
    """MFU with denominator = effective hardware peak under mixed precision.

    The effective peak is the harmonic-mean of per-dtype peaks weighted
    by per-dtype FLOPs share, derived from each op's component tag:

      effective_peak = total_flops / Σ (flops_by_dtype[d] / peak_for[d])

    Reduces to ``compute_mfu`` (BF16 peak) when all ops are BF16-typed.
    Returns 0 when step_time <= 0 or total flops <= 0.
    """
    from zrt.training.io.perf_tables import peak_tflops_for
    from zrt.training.models.flops import op_cost, total_training_flops
    from zrt.training.compose.stage import _resolve_compute_dtype

    if step_time <= 0:
        return 0.0

    actual_flops = total_training_flops(graph, model, strategy, system)
    if actual_flops <= 0:
        return 0.0

    # Aggregate per-dtype FLOPs by walking the graph
    flops_by_dtype: dict[Dtype, float] = {}
    for op in graph.ops:
        cost = op_cost(op, model, system)
        op_flops = (cost.fwd_cube_flops + cost.fwd_vector_flops
                    + cost.dx_cube_flops + cost.dx_vector_flops
                    + cost.dw_cube_flops + cost.dw_vector_flops)
        if op_flops <= 0:
            continue
        d = _resolve_compute_dtype(op, model)
        flops_by_dtype[d] = flops_by_dtype.get(d, 0.0) + op_flops

    gpu = system.gpu
    weighted_time = 0.0
    total = sum(flops_by_dtype.values())
    if total <= 0:
        return 0.0
    for d, f in flops_by_dtype.items():
        peak = peak_tflops_for(gpu, d)
        if peak <= 0:
            continue
        weighted_time += f / peak
    if weighted_time <= 0:
        return 0.0
    effective_peak = total / weighted_time
    pp_flops = actual_flops / strategy.pp
    return util_from_flops(pp_flops, effective_peak, step_time)
