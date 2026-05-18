"""Pipeline composers for training step-time schedules.

Reference: Megatron-LM (Narayanan et al. 2021) §3.2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from zrt.training.compose.stage import StageTime, stage_time
from zrt.training.ir.training_graph import Graph
from zrt.training.models.comm import total_comm_time, optimizer_comm_time
from zrt.training.models.flops import recompute_overhead_flops
from zrt.training.models.memory import MemBreakdown, memory_breakdown
from zrt.training.models.optimizer import (
    muon_optimizer_step_flops,
    muon_step_flops_from_arch,
    adam_step_flops,
)
from zrt.training.io.perf_tables import achieved_flops_efficiency
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
      pipeline_time    = compute_time + exposed_comm
      exposed_comm     = tp_exposed + cp_exposed + ep_exposed + pp_exposed + dp_exposed
      hidden_comm      = dp_hidden + tp_hidden + ep_hidden
      total_comm_volume = exposed_comm + hidden_comm
    """

    # ── Core step timing ──────────────────────────────────────────────────
    step_time: float = 0.0          # Total step time (seconds)
    pipeline_time: float = 0.0      # Pipeline time = compute_time + exposed_comm
    optimizer_time: float = 0.0     # Optimizer step compute
    optimizer_comm: float = 0.0     # Optimizer step communication (e.g. Muon AG+RS)

    # ── Pipeline structure (set by composers) ─────────────────────────────
    bubble_fraction: float = 0.0
    warmup: float = 0.0
    steady: float = 0.0
    cooldown: float = 0.0
    dp_exposed: float = 0.0         # DP AR/RS on critical path (set by composer; also DP group share)
    schedule_name: str = "1f1b"
    warmup_steps: int = 0
    cooldown_steps: int = 0
    per_stage: list[StageTime] = field(default_factory=list)
    memory: MemBreakdown | None = None
    mfu: float = 0.0
    hfu: float = 0.0
    mfu_native: float = 0.0   # MFU vs op-mix-weighted effective peak

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
    compute_time: float = 0.0       # Pure compute on critical path
    fwd_compute: float = 0.0        # Forward compute only (excludes all comm)
    bwd_compute: float = 0.0        # Backward compute only (excludes all comm)
    exposed_comm: float = 0.0       # Comm on critical path = Σ *_exposed fields

    # Per-group exposed comm (Σ = exposed_comm)
    tp_exposed: float = 0.0         # TP RS/AG (after CoC/MC2 reduction)
    cp_exposed: float = 0.0         # CP A2A
    ep_exposed: float = 0.0         # EP A2A (after wave-overlap reduction)
    pp_exposed: float = 0.0         # PP P2P
    # dp_exposed declared in pipeline section; equals DP group contribution to exposed_comm

    # Hidden comm — runs in parallel with compute, NOT on critical path
    hidden_comm: float = 0.0        # Total hidden = Σ *_hidden fields
    dp_hidden: float = 0.0          # DP AR absorbed in pipeline bubble
    tp_hidden: float = 0.0          # TP hidden by CoC/MC2
    ep_hidden: float = 0.0          # EP hidden by wave-overlap

    # Total comm volume = exposed + hidden
    total_comm_volume: float = 0.0  # All comm in step


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
                bubble_fraction=bubble_frac,
                warmup=0.0,
                steady=step * M,
                cooldown=0.0,
                dp_exposed=dp_exposed,
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

        # DP AR: hide in cooldown (backward drain phase) if enabled
        bubble = warmup + cooldown
        steady_bwd_total = M * t_bwd_max
        hidden = _dp_hidden(dp_ar_time, cooldown, steady_bwd_total, strategy)
        dp_exposed = dp_ar_time - hidden

        step = warmup + steady + cooldown + dp_exposed
        ideal_step = M * t_stage_max
        bubble_frac = (warmup + cooldown) / step if step > 0 else 0.0

        return StepResult(
            step_time=step,
            bubble_fraction=bubble_frac,
            warmup=warmup,
            steady=steady,
            cooldown=cooldown,
            dp_exposed=dp_exposed,
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

        bubble = warmup + cooldown
        steady_bwd_total = M * t_bwd_max
        hidden = _dp_hidden(dp_ar_time, cooldown, steady_bwd_total, strategy)
        dp_exposed = dp_ar_time - hidden

        step = warmup + steady + cooldown + dp_exposed
        bubble_frac = (warmup + cooldown) / step if step > 0 else 0.0

        return StepResult(
            step_time=step,
            bubble_fraction=bubble_frac,
            warmup=warmup,
            steady=steady,
            cooldown=cooldown,
            dp_exposed=dp_exposed,
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
    """DualPipe schedule — forward and backward on different stages in parallel.

    Key insight: in standard 1F1B, each stage alternates F then B.
    DualPipe splits the pipeline so that while stage S does forward,
    stage S+1 does backward, reducing bubble.

    Bubble fraction ≈ (pp - 1) / (2 * M + pp - 1)
    Step time ≈ (M + (pp-1)/2) * t_stage + dp_exposed
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

        t_stage_max = max(st.fwd + st.bwd for st in stage_times) if stage_times else 0

        bubble = (pp - 1) / 2.0 * t_stage_max
        warmup = bubble / 2.0
        cooldown = bubble / 2.0
        steady = M * t_stage_max
        steady_bwd_total = M * t_stage_max / 2
        hidden = _dp_hidden(dp_ar_time, cooldown, steady_bwd_total, strategy)
        dp_exposed = dp_ar_time - hidden

        step = warmup + steady + cooldown + dp_exposed
        bubble_frac = bubble / step if step > 0 else 0.0

        return StepResult(
            step_time=step,
            bubble_fraction=bubble_frac,
            warmup=warmup,
            steady=steady,
            cooldown=cooldown,
            dp_exposed=dp_exposed,
            schedule_name="dualpipe",
            warmup_steps=max(1, -(-(pp - 1) // 2)),
            cooldown_steps=max(1, -(-(pp - 1) // 2)),
            warmup_fwd=warmup,
            warmup_bwd=0.0,
            steady_fwd=M * t_stage_max / 2,
            steady_bwd=M * t_stage_max / 2,
            cooldown_fwd=0.0,
            cooldown_bwd=cooldown,
            steady_fwd_per_mb=t_stage_max / 2,
            steady_bwd_per_mb=t_stage_max / 2,
            steady_per_mb=t_stage_max,
        )


class DualPipeVComposer(PipelineComposer):
    """DualPipeV — DualPipe with virtual stage splitting.

    Combines DualPipe's F/B parallelism with VPP's virtual stages.
    Bubble ≈ (pp - 1) / (2 * V * M + pp - 1)
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

        t_stage_max = max(st.fwd + st.bwd for st in stage_times) if stage_times else 0

        bubble = (pp - 1) / (2.0 * V) * t_stage_max
        warmup = bubble / 2.0
        cooldown = bubble / 2.0
        steady = M * t_stage_max
        steady_bwd_total = M * t_stage_max / 2
        hidden = _dp_hidden(dp_ar_time, cooldown, steady_bwd_total, strategy)
        dp_exposed = dp_ar_time - hidden

        step = warmup + steady + cooldown + dp_exposed
        bubble_frac = bubble / step if step > 0 else 0.0

        return StepResult(
            step_time=step,
            bubble_fraction=bubble_frac,
            warmup=warmup,
            steady=steady,
            cooldown=cooldown,
            dp_exposed=dp_exposed,
            schedule_name="dualpipev",
            warmup_steps=max(1, -(-(pp - 1) // (2 * V))),
            cooldown_steps=max(1, -(-(pp - 1) // (2 * V))),
            warmup_fwd=warmup,
            warmup_bwd=0.0,
            steady_fwd=M * t_stage_max / 2,
            steady_bwd=M * t_stage_max / 2,
            cooldown_fwd=0.0,
            cooldown_bwd=cooldown,
            steady_fwd_per_mb=t_stage_max / 2,
            steady_bwd_per_mb=t_stage_max / 2,
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
        t_w = bottleneck.bwd_dw

        # ZB-1P/ZB-V keep a residual per-transition bubble even when t_w ≈ t_stage.
        # The empirical floor is roughly 2 microseconds (P2P latency) per pp-1
        # transition. We use 1e-6 s here as the minimum unit; callers that want
        # a hardware-derived floor can pass it in via stage_times comm_bwd already
        # baked into t_stage. This avoids the "0 bubble" artifact in search.
        ZB_BUBBLE_FLOOR_PER_TRANSITION = 2e-6  # 2 µs P2P latency per pp transition
        bubble = max((pp - 1) * max(t_stage - t_w, 0.0),
                     (pp - 1) * ZB_BUBBLE_FLOOR_PER_TRANSITION)
        warmup = bubble / 2.0
        steady = M * t_stage
        cooldown = bubble / 2.0

        steady_bwd_total = M * (t_stage / 2)
        hidden = _dp_hidden(dp_ar_time, cooldown, steady_bwd_total, strategy)
        dp_exposed = dp_ar_time - hidden

        step = warmup + steady + cooldown + dp_exposed
        bubble_frac = bubble / step if step > 0 else 0.0

        return StepResult(
            step_time=step,
            bubble_fraction=bubble_frac,
            warmup=warmup,
            steady=steady,
            cooldown=cooldown,
            dp_exposed=dp_exposed,
            schedule_name="zb",
            warmup_steps=pp - 1,
            cooldown_steps=pp - 1,
            warmup_fwd=warmup,
            warmup_bwd=0.0,
            steady_fwd=M * t_stage / 2,
            steady_bwd=M * t_stage / 2,
            cooldown_fwd=0.0,
            cooldown_bwd=cooldown,
            steady_fwd_per_mb=t_stage / 2,
            steady_bwd_per_mb=t_stage / 2,
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

        st = stage_time(stage_ops, stage_colls, model, system, strategy)
        stage_times.append(st)

    # Compute DP allreduce time and PP P2P overhead
    comm_times = total_comm_time(graph, model, system, strategy)
    dp_ar_time = comm_times.get("dp_grad_reduce", 0.0)

    # Add PP P2P per-microbatch cost to each stage's fwd and bwd
    pp_p2p = comm_times.get("pp_p2p", 0.0)
    if pp_p2p > 0:
        stage_times = [
            StageTime(
                fwd=st.fwd + pp_p2p,
                bwd=st.bwd + pp_p2p,
                bwd_dx=st.bwd_dx + pp_p2p,
                bwd_dw=st.bwd_dw,
                comm_fwd=st.comm_fwd + pp_p2p,
                comm_bwd=st.comm_bwd + pp_p2p,
                ep_hidden=st.ep_hidden,
                tp_hidden=st.tp_hidden,
            )
            for st in stage_times
        ]

    # Compose according to schedule
    composer_cls = COMPOSER_BY_SCHED.get(strategy.pp_schedule, OneF1BComposer)
    composer = composer_cls()
    step = composer.compose(stage_times, M, pp, dp_ar_time, strategy)

    step.per_stage = stage_times

    # Dual-batch overlap on pipeline bubble.
    #
    # Two micro-batch streams running in opposite pipeline directions can
    # overlap each other's fill/drain, but the bubble can NEVER reach zero:
    # at t=0 only stage 0 of stream-A and stage pp-1 of stream-B are busy;
    # stages between them remain idle until the two fronts meet in the
    # middle at t = (pp-1)/2 · t_stage. The symmetric drain at the end
    # adds the same residual. No third stream exists to fill that
    # fill/drain region, so the irreducible bubble is exactly the
    # DualPipe(V) formula:  (pp-1)/(2·V) · t_stage_max.
    #
    
    # The earlier formulation `residual = max(warmup + cooldown - steady, 0)`
    # subtracted steady wall-time (busy slots) from bubble wall-time (idle
    # slots) — dimensionally meaningless — and produced bubble_fraction=0
    # whenever M >> pp-1, which is non-physical.
    #
    # Schedules that already model antiparallel streams (DualPipe,
    # DualPipeV) bake this (pp-1)/(2·V) reduction into the composer; the
    # `dualbatch` flag on those schedules only enables stage-level comm
    # hiding (compose/stage.py). To avoid double-counting, we skip the
    # pipeline-bubble adjustment for those schedules.
    DUALBATCH_BAKED_IN = {PPSched.DUALPIPE, PPSched.DUALPIPE_V}
    if (strategy.dualbatch and pp > 1
            and strategy.pp_schedule not in DUALBATCH_BAKED_IN):
        original_bubble = step.warmup + step.cooldown
        V = max(1, strategy.vpp_chunks)
        t_stage_max = max(
            (st.fwd + st.bwd for st in stage_times), default=0.0
        )
        # Antiparallel-stream floor — irreducible fill/drain.
        target_bubble = (pp - 1) / (2.0 * V) * t_stage_max
        # If the schedule's natural bubble is already at or below this
        # floor (e.g., ZeroBubble), dualbatch yields no additional benefit.
        residual_bubble = min(original_bubble, target_bubble)
        bubble_saved = original_bubble - residual_bubble
        new_cooldown = residual_bubble / 2.0
        # Recompute DP exposure through the same helper used by composers,
        # so steady-BWD overlap continues even when dualbatch shrinks
        # cooldown.
        steady_bwd_total_bot = max(0.0, step.steady_bwd_per_mb * M)
        if dp_ar_time > 0:
            new_dp_exposed = dp_ar_time - _dp_hidden(
                dp_ar_time, new_cooldown, steady_bwd_total_bot, strategy)
        else:
            new_dp_exposed = step.dp_exposed
        dp_delta = new_dp_exposed - step.dp_exposed
        step.step_time = step.step_time - bubble_saved + dp_delta
        step.warmup = residual_bubble / 2.0
        step.cooldown = new_cooldown
        step.dp_exposed = new_dp_exposed
        step.bubble_fraction = residual_bubble / step.step_time if step.step_time > 0 else 0.0

    # === Communication and compute breakdown ===
    # Placed after dual-batch so step.step_time / step.dp_exposed are final.
    #
    # Invariants enforced here:
    #   pipeline_time    = compute_time + exposed_comm            (exact)
    #   exposed_comm     = tp + cp + ep + pp + dp _exposed       (exact)
    #   hidden_comm      = dp_hidden + tp_hidden + ep_hidden      (exact)
    #   total_comm_volume = exposed_comm + hidden_comm            (exact)
    #
    # comm_fwd/comm_bwd in StageTime are already-reduced exposed portions
    # (TP CoC/MC2 and EP wave-overlap applied inside stage_time()).
    # The bottleneck stage ratio gives a schedule-agnostic critical-path split.

    s_bot = max(stage_times, key=lambda st: st.fwd + st.bwd)
    bot_total = s_bot.fwd + s_bot.bwd
    bot_comm = s_bot.comm_fwd + s_bot.comm_bwd  # exposed comm per stage per microbatch

    # pipeline_time excludes the exposed DP AR tail (which sits after cooldown)
    _pipeline_time = step.step_time - step.dp_exposed
    comm_frac = (bot_comm / bot_total) if bot_total > 0 else 0.0

    # ── Exposed comm ──────────────────────────────────────────────────────
    exposed_comm_excl_dp = _pipeline_time * comm_frac
    step.exposed_comm = exposed_comm_excl_dp + step.dp_exposed
    # compute_time exact: compute_time + exposed_comm == _pipeline_time + dp_exposed == step_time(pre-opt)
    step.compute_time = step.step_time - step.exposed_comm

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

    # PP P2P: pp_p2p already baked into bot_comm; scale to critical path via same ratio
    pp_p2p_exposed = (_pipeline_time * (2.0 * pp_p2p) / bot_total) if bot_total > 0 else 0.0

    # TP exposed / hidden: use StageTime.tp_hidden (GEMM-bound, set by
    # compose/stage.py). Scale to critical path the same way as ep_hidden.
    raw_tp = sum(comm_times.get(c.name, 0.0) for c in graph.collectives if c.group == "TP")
    raw_cp = sum(comm_times.get(c.name, 0.0) for c in graph.collectives if c.group == "CP")
    raw_ep = sum(comm_times.get(c.name, 0.0) for c in graph.collectives if c.group == "EP")

    # tp_hidden scaled to critical path (matches the ep_hidden treatment below)
    if bot_total > 0 and s_bot.tp_hidden > 0:
        step.tp_hidden = _pipeline_time * s_bot.tp_hidden / bot_total
    else:
        step.tp_hidden = 0.0

    # tp_exposed = raw TP volume (per stage, scaled to critical path) − tp_hidden
    raw_tp_critical = (_pipeline_time * (raw_tp / max(pp, 1)) / bot_total) \
                      if bot_total > 0 else 0.0
    # Conservation fix: cap tp_exposed_volume so the per-group exposed
    # times sum to exposed_comm_excl_dp exactly. Without this cap,
    # raw_tp_critical (computed from un-overlapped raw_tp / pp) can exceed
    # the exposed-comm budget that was already reduced by stage-level
    # overlap, breaking the documented invariant
    #     exposed_comm = tp + cp + ep + pp + dp_exposed
    tp_exposed_volume = max(0.0, raw_tp_critical - step.tp_hidden)
    tp_exposed_volume = min(tp_exposed_volume,
                            max(0.0, exposed_comm_excl_dp - pp_p2p_exposed))

    # CP / EP exposed: proportional split of the remaining exposed-comm budget
    # after PP P2P and TP exposed have been subtracted.
    eff_total = raw_cp + raw_ep  # TP no longer participates in proportional split
    remain = max(0.0, exposed_comm_excl_dp - pp_p2p_exposed - tp_exposed_volume)
    if eff_total > 0 and remain > 0:
        step.tp_exposed = tp_exposed_volume
        step.cp_exposed = remain * raw_cp / eff_total
        step.ep_exposed = remain * raw_ep / eff_total
    else:
        step.tp_exposed = tp_exposed_volume
        step.cp_exposed = 0.0
        step.ep_exposed = 0.0
    step.pp_exposed = pp_p2p_exposed
    # step.dp_exposed already set by composer / dual-batch

    # ── Hidden comm ───────────────────────────────────────────────────────
    # DP AR hidden in pipeline bubble — independent, exact.
    step.dp_hidden = max(0.0, dp_ar_time - step.dp_exposed)

    # EP hidden by wave-overlap — from StageTime.ep_hidden, scaled to critical path.
    if bot_total > 0 and s_bot.ep_hidden > 0:
        step.ep_hidden = _pipeline_time * s_bot.ep_hidden / bot_total
    else:
        step.ep_hidden = 0.0

    step.hidden_comm = step.dp_hidden + step.tp_hidden + step.ep_hidden
    step.total_comm_volume = step.exposed_comm + step.hidden_comm

    # Optimizer time and communication
    opt_time = _compute_optimizer_time(model, system, strategy)
    opt_comm = _compute_optimizer_comm_time(model, system, strategy)
    step.optimizer_time = opt_time
    step.optimizer_comm = opt_comm

    # Memory breakdown
    step.memory = memory_breakdown(graph, model, system, strategy)

    # MFU (before adding optimizer time, per design doc §5.5.2)
    step.mfu = compute_mfu(model, strategy, system, step.step_time, graph)

    # HFU
    step.hfu = compute_hfu(model, strategy, system, step.step_time, graph)
    step.mfu_native = compute_mfu_native(model, strategy, system, step.step_time, graph)

    # Add optimizer time to step_time (per §5.5.2 of muon_optimizer_design.md)
    # This must happen after MFU/HFU calculation so MFU excludes optimizer overhead.
    # pipeline_time is set after this addition to satisfy:
    #   step_time = pipeline_time + optimizer_time + optimizer_comm
    step.step_time += opt_time + opt_comm
    step.pipeline_time = step.step_time - step.optimizer_time - step.optimizer_comm

    return step


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


def _compute_optimizer_time(model: ModelSpec, system: SystemSpec, strategy: Strategy) -> float:
    """Compute optimizer step time in seconds.

    Uses roofline model with achieved efficiency from perf_tables.
    """
    from zrt.training.spec.model import LayerKind

    P = model.total_params()
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
        adam_flops = adam_step_flops(int(P * (1 - f_muon)))
        flops = muon_flops + adam_flops
        eff = achieved_flops_efficiency(gpu.name, Dtype.BF16, flops)
        return flops / (peak_flops * eff) if eff > 0 else 0.0
    else:
        flops = adam_step_flops(P)
        eff = achieved_flops_efficiency(gpu.name, Dtype.BF16, flops)
        return flops / (peak_flops * eff) if eff > 0 else 0.0


def _compute_optimizer_comm_time(model: ModelSpec, system: SystemSpec, strategy: Strategy) -> float:
    """Compute optimizer communication time (Muon ZeRO-1 AllGather + ReduceScatter)."""
    comm_times = optimizer_comm_time(model, system, strategy)
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
