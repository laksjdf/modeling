"""Pipeline composers for training step-time schedules.

Reference: Megatron-LM (Narayanan et al. 2021) §3.2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from zrt.training.compose.stage import StageTime, stage_time
from zrt.training.ir.graph import Graph
from zrt.training.models.comm import total_comm_time
from zrt.training.models.flops import recompute_overhead_flops
from zrt.training.models.memory import MemBreakdown, memory_breakdown
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.strategy import PPSched, Strategy
from zrt.training.spec.system import SystemSpec


@dataclass
class StepResult:
    step_time: float = 0.0         # seconds
    per_stage: list[StageTime] = field(default_factory=list)
    bubble_fraction: float = 0.0
    warmup: float = 0.0
    steady: float = 0.0
    cooldown: float = 0.0
    dp_ar_exposed: float = 0.0
    memory: MemBreakdown | None = None
    mfu: float = 0.0
    hfu: float = 0.0
    schedule_name: str = "1f1b"    # Pipeline schedule identifier


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

        warmup   = (pp - 1) * t_fwd[0]
        steady   = M * max(t_fwd[s] + t_bwd[s])
        cooldown = (pp - 1) * t_bwd[-1]
        step     = warmup + steady + cooldown + dp_ar_exposed
        """
        if pp == 1:
            # No pipeline: just fwd + bwd for single stage
            st = stage_times[0] if stage_times else StageTime()
            step = st.fwd + st.bwd
            dp_exposed = dp_ar_time
            if strategy.dp_overlap_in_bubble and dp_ar_time > 0:
                hidden = min(st.bwd * M, dp_ar_time)
                dp_exposed = dp_ar_time - hidden

            ideal_step = M * (st.fwd + st.bwd)
            bubble_frac = 0.0

            return StepResult(
                step_time=step * M + dp_exposed,
                bubble_fraction=bubble_frac,
                warmup=0.0,
                steady=step * M,
                cooldown=0.0,
                dp_ar_exposed=dp_exposed,
                schedule_name="1f1b",
            )

        # With pipeline parallelism
        t_fwd_max = max(st.fwd for st in stage_times) if stage_times else 0
        t_bwd_max = max(st.bwd for st in stage_times) if stage_times else 0
        t_stage_max = max(st.fwd + st.bwd for st in stage_times) if stage_times else 0

        warmup = (pp - 1) * t_fwd_max
        steady = M * t_stage_max
        cooldown = (pp - 1) * t_bwd_max

        # DP AR: hide in bubble if enabled
        bubble = warmup + cooldown
        dp_exposed = dp_ar_time
        if strategy.dp_overlap_in_bubble and dp_ar_time > 0:
            hidden = min(bubble, dp_ar_time)
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
            dp_ar_exposed=dp_exposed,
            schedule_name="1f1b",
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
        dp_exposed = dp_ar_time
        if strategy.dp_overlap_in_bubble and dp_ar_time > 0:
            hidden = min(bubble, dp_ar_time)
            dp_exposed = dp_ar_time - hidden

        step = warmup + steady + cooldown + dp_exposed
        bubble_frac = (warmup + cooldown) / step if step > 0 else 0.0

        return StepResult(
            step_time=step,
            bubble_fraction=bubble_frac,
            warmup=warmup,
            steady=steady,
            cooldown=cooldown,
            dp_ar_exposed=dp_exposed,
            schedule_name="i1f1b",
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

        warmup = (pp - 1) / 2.0 * t_stage_max
        steady = M * t_stage_max
        cooldown = (pp - 1) / 2.0 * t_stage_max

        bubble = warmup + cooldown
        dp_exposed = dp_ar_time
        if strategy.dp_overlap_in_bubble and dp_ar_time > 0:
            hidden = min(bubble, dp_ar_time)
            dp_exposed = dp_ar_time - hidden

        step = warmup + steady + cooldown + dp_exposed
        bubble_frac = bubble / step if step > 0 else 0.0

        return StepResult(
            step_time=step,
            bubble_fraction=bubble_frac,
            warmup=warmup,
            steady=steady,
            cooldown=cooldown,
            dp_ar_exposed=dp_exposed,
            schedule_name="dualpipe",
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

        warmup = (pp - 1) / (2.0 * V) * t_stage_max
        steady = M * t_stage_max
        cooldown = (pp - 1) / (2.0 * V) * t_stage_max

        bubble = warmup + cooldown
        dp_exposed = dp_ar_time
        if strategy.dp_overlap_in_bubble and dp_ar_time > 0:
            hidden = min(bubble, dp_ar_time)
            dp_exposed = dp_ar_time - hidden

        step = warmup + steady + cooldown + dp_exposed
        bubble_frac = bubble / step if step > 0 else 0.0

        return StepResult(
            step_time=step,
            bubble_fraction=bubble_frac,
            warmup=warmup,
            steady=steady,
            cooldown=cooldown,
            dp_ar_exposed=dp_exposed,
            schedule_name="dualpipev",
        )


class ZeroBubbleComposer(PipelineComposer):
    """ZeroBubble schedule with backward split into dX and dW phases.

    The critical path is the per-stage F+B time.  Weight-gradient work can
    be delayed to fill pipeline bubbles, so the exposed bubble is reduced by
    the bottleneck stage's dW time:

        step = M * t_stage + (pp - 1) * max(t_stage - t_w, 0)
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

        bubble = (pp - 1) * max(t_stage - t_w, 0.0)
        warmup = bubble / 2.0
        steady = M * t_stage
        cooldown = bubble / 2.0

        dp_exposed = dp_ar_time
        if strategy.dp_overlap_in_bubble and dp_ar_time > 0:
            hidden = min(bubble, dp_ar_time)
            dp_exposed = dp_ar_time - hidden

        step = warmup + steady + cooldown + dp_exposed
        bubble_frac = bubble / step if step > 0 else 0.0

        return StepResult(
            step_time=step,
            bubble_fraction=bubble_frac,
            warmup=warmup,
            steady=steady,
            cooldown=cooldown,
            dp_ar_exposed=dp_exposed,
            schedule_name="zb",
        )


_COMPOSERS: dict[PPSched, PipelineComposer] = {
    PPSched.ONE_F_ONE_B: OneF1BComposer(),
    PPSched.INTERLEAVED: Interleaved1F1BComposer(),
    PPSched.ZERO_BUBBLE: ZeroBubbleComposer(),
    PPSched.DUALPIPE: DualPipeComposer(),
    PPSched.DUALPIPE_V: DualPipeVComposer(),
}


def pipeline_step_time(
    graph: Graph,
    model: ModelSpec,
    system: SystemSpec,
    strategy: Strategy,
) -> StepResult:
    """Compute full training step time from IR + strategy.

    TODO Phase 3: Integrate with graph-native per-stage timelines from executor/.

    Graph-native path (phase 3):
      1. Run DAGScheduler per PP stage on stitched fwd+bwd OpGraph
      2. Extract (t_fwd[s], t_bwd[s]) from timeline.phase_latency("fwd")/("bwd")
      3. Pass per-stage timelines to VPP/DualPipe/DualPipeV composers
      4. Compute DP-in-bubble from actual overlap windows
      5. Replace formula-based stage_time() with timeline-derived values

    Current path (reference implementation):
      - Use IR-level stage_time() from graph.ops_for_stage()
      - Apply 1F1B/VPP/DualPipe formulas on aggregated times
      - DP overlap uses simple bubble-window heuristic

    Both paths should converge to the same StepResult interface for compatibility.
    """
    pp = strategy.pp
    M = strategy.num_microbatches()

    # Compute per-stage times
    stage_ids = _assign_stages(model, strategy)
    stage_times: list[StageTime] = []

    for s in range(pp):
        layer_ids = stage_ids[s]
        stage_ops = graph.ops_for_stage(layer_ids)

        # Collectives belonging to this stage
        stage_colls = [
            c for c in graph.collectives
            if any(c.inserted_after.startswith(f"L{lid}") for lid in layer_ids)
        ]

        st = stage_time(stage_ops, stage_colls, model, system, strategy)
        stage_times.append(st)

    # Compute DP allreduce time
    comm_times = total_comm_time(graph, model, system, strategy)
    dp_ar_time = comm_times.get("dp_grad_reduce", 0.0)

    # Compose according to schedule
    composer = _COMPOSERS.get(strategy.pp_schedule, OneF1BComposer())
    step = composer.compose(stage_times, M, pp, dp_ar_time, strategy)

    step.per_stage = stage_times

    # Memory breakdown
    step.memory = memory_breakdown(graph, model, system, strategy)

    # MFU
    step.mfu = compute_mfu(model, strategy, system, step.step_time)

    # HFU
    step.hfu = compute_hfu(model, strategy, system, step.step_time, graph)

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


def compute_mfu(
    model: ModelSpec, strategy: Strategy,
    system: SystemSpec, step_time: float,
) -> float:
    """Model FLOPs Utilization.

    MFU = model_flops_per_token * tokens_per_step / (world_size * peak_flops * step_time)

    For MoE models, uses effective_params_for_flops() which accounts for top_k
    expert sparsity instead of counting all expert parameters.
    """
    if step_time <= 0:
        return 0.0

    # Model FLOPs per token for forward pass: ~6 * P (P = effective params)
    # Full training step (fwd+bwd): ~6P per token
    # For MoE: effective_params_for_flops() only counts active expert params (top_k/num_experts)
    P = model.effective_params_for_flops()
    tokens = strategy.global_batch * model.seq_len if strategy.global_batch > 0 else strategy.micro_batch * strategy.dp * model.seq_len
    model_flops = 6.0 * P * tokens  # 6P rule

    # Peak FLOP/s of entire cluster
    peak = system.gpu.flops_bf16 * 1e12 * system.world_size  # TFLOP/s -> FLOP/s, times world

    mfu = model_flops / (peak * step_time)
    return min(mfu, 1.0)  # cap at 100%


def compute_hfu(
    model: ModelSpec, strategy: Strategy,
    system: SystemSpec, step_time: float,
    graph: Graph,
) -> float:
    """Hardware FLOPs Utilization — accounts for recomputed activations.

    HFU = (model_flops + recompute_overhead) / (peak * step_time)
    """
    if step_time <= 0:
        return 0.0

    P = model.effective_params_for_flops()
    tokens = strategy.global_batch * model.seq_len if strategy.global_batch > 0 else strategy.micro_batch * strategy.dp * model.seq_len
    model_flops = 6.0 * P * tokens
    rc_overhead = recompute_overhead_flops(graph, model, strategy)
    peak = system.gpu.flops_bf16 * 1e12 * system.world_size

    hfu = (model_flops + rc_overhead) / (peak * step_time)
    return min(hfu, 1.0)
