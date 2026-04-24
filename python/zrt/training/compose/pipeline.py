"""Pipeline composer — multiple PP schedules → step time.

References:
- 1F1B: Megatron-LM (Narayanan et al. 2021) §3.2
- VPP/Interleaved: Megatron-LM 2.0 (Narayanan et al. 2021) §4
- DualPipe: DeepSeek-V3 Technical Report §5.4
"""

from __future__ import annotations

from dataclasses import dataclass, field

from zrt.training.compose.stage import StageTime, stage_time
from zrt.training.ir.graph import Graph
from zrt.training.models.comm import total_comm_time
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
    schedule_name: str = "1f1b"


def pipeline_step_time(
    graph: Graph,
    model: ModelSpec,
    system: SystemSpec,
    strategy: Strategy,
) -> StepResult:
    """Compute full training step time from IR + strategy.

    Phase 1: 1F1B schedule only.
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
    if strategy.pp_schedule == PPSched.ONE_F_ONE_B:
        step = _one_f_one_b(stage_times, M, pp, dp_ar_time, strategy)
    elif strategy.pp_schedule == PPSched.INTERLEAVED:
        step = _interleaved_1f1b(stage_times, M, pp, dp_ar_time, strategy)
    elif strategy.pp_schedule == PPSched.DUALPIPE:
        step = _dualpipe(stage_times, M, pp, dp_ar_time, strategy)
    elif strategy.pp_schedule == PPSched.DUALPIPE_V:
        step = _dualpipe_v(stage_times, M, pp, dp_ar_time, strategy)
    else:
        step = _one_f_one_b(stage_times, M, pp, dp_ar_time, strategy)

    step.per_stage = stage_times

    # Memory breakdown
    step.memory = memory_breakdown(graph, model, system, strategy)

    # MFU
    step.mfu = compute_mfu(model, strategy, system, step.step_time)

    return step


def _one_f_one_b(
    stage_times: list[StageTime], M: int, pp: int,
    dp_ar_time: float, strategy: Strategy,
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
        # DP overlap not applicable with PP=1
        dp_exposed = dp_ar_time

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
    """
    if step_time <= 0:
        return 0.0

    # Model FLOPs per token for forward pass: ~6 * P (P = total params)
    # Full training step (fwd+bwd): ~6P per token
    P = model.total_params()
    tokens = strategy.global_batch * model.seq_len if strategy.global_batch > 0 else strategy.micro_batch * strategy.dp * model.seq_len
    model_flops = 6.0 * P * tokens  # 6P rule

    # Peak FLOP/s of entire cluster
    peak = system.gpu.flops_bf16 * 1e12 * system.world_size  # TFLOP/s -> FLOP/s, times world

    mfu = model_flops / (peak * step_time)
    return min(mfu, 1.0)  # cap at 100%


def _interleaved_1f1b(
    stage_times: list[StageTime], M: int, pp: int,
    dp_ar_time: float, strategy: Strategy,
) -> StepResult:
    """VPP / Interleaved 1F1B pipeline schedule.

    Reference: Megatron-LM 2.0 (Narayanan et al. 2021) §4

    Bubble formula: (pp - 1) / (vpp_chunks * M)

    Each stage interleaves vpp_chunks microbatches, reducing bubble fraction
    by factor of vpp_chunks compared to standard 1F1B.
    """
    vpp = strategy.vpp_chunks

    if pp == 1:
        st = stage_times[0] if stage_times else StageTime()
        step = st.fwd + st.bwd
        dp_exposed = dp_ar_time
        ideal_step = M * (st.fwd + st.bwd)
        return StepResult(
            step_time=step * M + dp_exposed,
            bubble_fraction=0.0,
            warmup=0.0,
            steady=step * M,
            cooldown=0.0,
            dp_ar_exposed=dp_exposed,
            schedule_name="i1f1b",
        )

    t_fwd_max = max(st.fwd for st in stage_times) if stage_times else 0
    t_bwd_max = max(st.bwd for st in stage_times) if stage_times else 0
    t_stage_max = max(st.fwd + st.bwd for st in stage_times) if stage_times else 0

    warmup = (pp - 1) * t_fwd_max / vpp
    steady = M * t_stage_max
    cooldown = (pp - 1) * t_bwd_max / vpp

    bubble = warmup + cooldown
    dp_exposed = dp_ar_time
    if strategy.dp_overlap_in_bubble and dp_ar_time > 0:
        hidden = min(bubble, dp_ar_time)
        dp_exposed = dp_ar_time - hidden

    step = warmup + steady + cooldown + dp_exposed

    bubble_frac = (pp - 1) / (vpp * M) if M > 0 else 0.0

    return StepResult(
        step_time=step,
        bubble_fraction=bubble_frac,
        warmup=warmup,
        steady=steady,
        cooldown=cooldown,
        dp_ar_exposed=dp_exposed,
        schedule_name="i1f1b",
    )


def _dualpipe(
    stage_times: list[StageTime], M: int, pp: int,
    dp_ar_time: float, strategy: Strategy,
) -> StepResult:
    """DualPipe schedule from DeepSeek-V3.

    Reference: DeepSeek-V3 Technical Report §5.4

    Key insight: Each stage concurrently runs fwd of μbatch_i and bwd of μbatch_{i-1},
    reducing bubble to approximately half of interleaved 1F1B.

    When dualbatch=True, EP A2A is hidden by paired μbatch computation.
    """
    if pp == 1:
        st = stage_times[0] if stage_times else StageTime()
        step = max(st.fwd, st.bwd)
        dp_exposed = dp_ar_time
        ideal_step = M * (st.fwd + st.bwd)
        return StepResult(
            step_time=step * M + dp_exposed,
            bubble_fraction=0.0,
            warmup=0.0,
            steady=step * M,
            cooldown=0.0,
            dp_ar_exposed=dp_exposed,
            schedule_name="dualpipe",
        )

    t_fwd_max = max(st.fwd for st in stage_times) if stage_times else 0
    t_bwd_max = max(st.bwd for st in stage_times) if stage_times else 0
    t_stage_max = max(max(st.fwd, st.bwd) for st in stage_times) if stage_times else 0

    warmup = (pp - 1) * t_fwd_max
    steady = M * t_stage_max
    cooldown = (pp - 1) * t_bwd_max

    bubble = warmup + cooldown
    dp_exposed = dp_ar_time
    if strategy.dp_overlap_in_bubble and dp_ar_time > 0:
        hidden = min(bubble, dp_ar_time)
        dp_exposed = dp_ar_time - hidden

    step = warmup + steady + cooldown + dp_exposed

    bubble_frac = (pp - 1) / (2 * M) if M > 0 else 0.0

    if strategy.dualbatch:
        bubble_frac = bubble_frac * 0.5

    return StepResult(
        step_time=step,
        bubble_fraction=bubble_frac,
        warmup=warmup,
        steady=steady,
        cooldown=cooldown,
        dp_ar_exposed=dp_exposed,
        schedule_name="dualpipe",
    )


def _dualpipe_v(
    stage_times: list[StageTime], M: int, pp: int,
    dp_ar_time: float, strategy: Strategy,
) -> StepResult:
    """DualPipe-V (virtual pipeline parallel with dual batching).

    Combines VPP interleaving with DualPipe concurrent fwd+bwd execution.

    Bubble formula: (pp - 1) / (2 * vpp_chunks * M)
    """
    vpp = strategy.vpp_chunks

    if pp == 1:
        st = stage_times[0] if stage_times else StageTime()
        step = max(st.fwd, st.bwd)
        dp_exposed = dp_ar_time
        return StepResult(
            step_time=step * M + dp_exposed,
            bubble_fraction=0.0,
            warmup=0.0,
            steady=step * M,
            cooldown=0.0,
            dp_ar_exposed=dp_exposed,
            schedule_name="dualpipev",
        )

    t_fwd_max = max(st.fwd for st in stage_times) if stage_times else 0
    t_bwd_max = max(st.bwd for st in stage_times) if stage_times else 0
    t_stage_max = max(max(st.fwd, st.bwd) for st in stage_times) if stage_times else 0

    warmup = (pp - 1) * t_fwd_max / vpp
    steady = M * t_stage_max
    cooldown = (pp - 1) * t_bwd_max / vpp

    bubble = warmup + cooldown
    dp_exposed = dp_ar_time
    if strategy.dp_overlap_in_bubble and dp_ar_time > 0:
        hidden = min(bubble, dp_ar_time)
        dp_exposed = dp_ar_time - hidden

    step = warmup + steady + cooldown + dp_exposed

    bubble_frac = (pp - 1) / (2 * vpp * M) if M > 0 else 0.0

    if strategy.dualbatch:
        bubble_frac = bubble_frac * 0.5

    return StepResult(
        step_time=step,
        bubble_fraction=bubble_frac,
        warmup=warmup,
        steady=steady,
        cooldown=cooldown,
        dp_ar_exposed=dp_exposed,
        schedule_name="dualpipev",
    )
