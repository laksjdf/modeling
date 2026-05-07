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
from zrt.training.models.optimizer import muon_optimizer_step_flops, adam_step_flops
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
    warmup_steps: int = 0          # Number of warmup microbatches
    cooldown_steps: int = 0        # Number of cooldown microbatches
    optimizer_time: float = 0.0    # Optimizer step time (seconds)
    optimizer_comm: float = 0.0    # Optimizer communication time (seconds)


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
                warmup_steps=0,
                cooldown_steps=0,
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
            warmup_steps=pp - 1,
            cooldown_steps=pp - 1,
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
            warmup_steps=max(1, -(-(pp - 1) // V)),  # ceil((pp-1)/V)
            cooldown_steps=max(1, -(-(pp - 1) // V)),
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
            warmup_steps=max(1, -(-(pp - 1) // 2)),  # ceil((pp-1)/2)
            cooldown_steps=max(1, -(-(pp - 1) // 2)),
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
            warmup_steps=max(1, -(-(pp - 1) // (2 * V))),  # ceil((pp-1)/(2*V))
            cooldown_steps=max(1, -(-(pp - 1) // (2 * V))),
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
            warmup_steps=pp - 1,
            cooldown_steps=pp - 1,
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
                (c.inserted_after and c.inserted_after.startswith(f"L{lid}")) or
                (c.inserted_before and c.inserted_before.startswith(f"L{lid}"))
                for lid in layer_ids
            )
        ]

        st = stage_time(stage_ops, stage_colls, model, system, strategy)
        stage_times.append(st)

    # Compute DP allreduce time
    comm_times = total_comm_time(graph, model, system, strategy)
    dp_ar_time = comm_times.get("dp_grad_reduce", 0.0)

    # Compose according to schedule
    composer_cls = COMPOSER_BY_SCHED.get(strategy.pp_schedule, OneF1BComposer)
    composer = composer_cls()
    step = composer.compose(stage_times, M, pp, dp_ar_time, strategy)

    step.per_stage = stage_times

    # Optimizer time and communication
    opt_time = _compute_optimizer_time(model, system, strategy)
    opt_comm = _compute_optimizer_comm_time(model, system, strategy)
    step.optimizer_time = opt_time
    step.optimizer_comm = opt_comm

    # Memory breakdown
    step.memory = memory_breakdown(graph, model, system, strategy)

    # MFU (before adding optimizer time, per design doc §5.5.2)
    step.mfu = compute_mfu(model, strategy, system, step.step_time)

    # HFU
    step.hfu = compute_hfu(model, strategy, system, step.step_time, graph)

    # Add optimizer time to step_time (per §5.5.2 of muon_optimizer_design.md)
    # This must happen after MFU/HFU calculation so MFU excludes optimizer overhead
    step.step_time += opt_time + opt_comm

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
    P = model.total_params()
    if strategy.tp > 1:
        P //= strategy.tp
    if strategy.pp > 1:
        n_layers = len(model.layers)
        embed = model.vocab * model.hidden * 2
        non_embed = P - embed
        non_embed = int(non_embed * (n_layers / strategy.pp) / n_layers)
        P = non_embed + embed // strategy.pp
    if strategy.zero_stage >= 3:
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
        flops = muon_optimizer_step_flops(P, K, model.hidden, f_muon)
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
) -> float:
    """Model FLOPs Utilization.

    MFU = model_flops_per_token * tokens_per_step / (world_size * peak_flops * step_time)

    For MoE models, uses effective_params_for_flops() which accounts for top_k
    expert sparsity instead of counting all expert parameters.
    """
    # Model FLOPs per token for forward pass: ~6 * P (P = effective params)
    # Full training step (fwd+bwd): ~6P per token
    # For MoE: effective_params_for_flops() only counts active expert params (top_k/num_experts)
    P = model.effective_params_for_flops()
    tokens = strategy.global_batch * model.seq_len if strategy.global_batch > 0 else strategy.micro_batch * strategy.dp * model.seq_len
    model_flops = 6.0 * P * tokens  # 6P rule

    # Peak FLOP/s of entire cluster
    peak = system.gpu.flops_bf16 * 1e12 * system.world_size  # TFLOP/s -> FLOP/s, times world

    return util_from_flops(model_flops, peak, step_time)


def compute_hfu(
    model: ModelSpec, strategy: Strategy,
    system: SystemSpec, step_time: float,
    graph: Graph,
) -> float:
    """Hardware FLOPs Utilization — accounts for recomputed activations.

    HFU = (model_flops + recompute_overhead) / (peak * step_time)
    """
    P = model.effective_params_for_flops()
    tokens = strategy.global_batch * model.seq_len if strategy.global_batch > 0 else strategy.micro_batch * strategy.dp * model.seq_len
    model_flops = 6.0 * P * tokens
    rc_overhead = recompute_overhead_flops(graph, model, strategy)
    peak = system.gpu.flops_bf16 * 1e12 * system.world_size

    return util_from_flops(model_flops + rc_overhead, peak, step_time)
