"""Stage composer — per-stage time for one microbatch.

Converts per-op FLOPs and byte counts to wall-clock time using
roofline model with achieved efficiency curves.
"""

from __future__ import annotations

from dataclasses import dataclass

from zrt.training.ir.graph import Collective, Graph, Op
from zrt.training.io.perf_tables import achieved_bandwidth_efficiency, achieved_flops_efficiency
from zrt.training.models.comm import collective_time, tier_for_group, total_comm_time
from zrt.training.models.flops import OpCost, op_cost
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import SystemSpec


@dataclass
class StageTime:
    fwd: float = 0.0   # seconds
    bwd: float = 0.0   # seconds (includes recompute + dx + dw)
    comm_fwd: float = 0.0  # exposed comm time during forward
    comm_bwd: float = 0.0  # exposed comm time during backward
    expert_imbalance_factor: float = 0.0  # applied to MoE ops


def op_to_time(
    flops: float, bytes_: float, bound: str, system: SystemSpec,
    gpu_name: str = "", dtype: Dtype = Dtype.BF16,
) -> float:
    """Convert per-op cost to wall-clock time in seconds.

    compute-bound: flops / (peak * efficiency)
    memory-bound: bytes / (hbm_bw * efficiency)
    """
    gpu = system.gpu

    if bound == "compute" and flops > 0:
        peak = gpu.flops_bf16 * 1e12  # TFLOP/s -> FLOP/s
        eff = achieved_flops_efficiency(gpu_name or gpu.name, dtype, flops)
        return flops / (peak * eff) if peak > 0 else 0.0

    elif bound == "memory" and bytes_ > 0:
        bw = gpu.hbm_bw_gbps * 1e9 / 8  # GB/s -> bytes/s
        eff = achieved_bandwidth_efficiency(gpu_name or gpu.name, bytes_)
        return bytes_ / (bw * eff) if bw > 0 else 0.0

    return 0.0


def stage_time(
    stage_ops: list[Op],
    stage_collectives: list[Collective],
    model: ModelSpec,
    system: SystemSpec,
    strategy: Strategy,
) -> StageTime:
    """Compute forward + backward time for one PP stage and one microbatch."""
    gpu_name = system.gpu.name

    t_fwd = 0.0
    t_bwd = 0.0

    imbalance = model.expert_imbalance if model.num_experts > 0 else 0.0

    for op in stage_ops:
        cost = op_cost(op, model)

        is_expert_op = (
            op.layer_kind == LayerKind.MOE
            and op.kind in ("matmul", "swiglu")
        )
        imbalance_factor = 1.0 + imbalance if is_expert_op else 1.0

        if cost.bound == "compute":
            fwd_t = op_to_time(cost.fwd_flops, 0, "compute", system, gpu_name)
            dx_t = op_to_time(cost.dx_flops, 0, "compute", system, gpu_name)
            dw_t = op_to_time(cost.dw_flops, 0, "compute", system, gpu_name)
            t_fwd += fwd_t * imbalance_factor
            t_bwd += (dx_t + dw_t) * imbalance_factor
        else:
            fwd_t = op_to_time(0, cost.fwd_bytes, "memory", system, gpu_name)
            dx_t = op_to_time(0, cost.dx_bytes, "memory", system, gpu_name)
            dw_t = op_to_time(0, cost.dw_bytes, "memory", system, gpu_name)
            t_fwd += fwd_t * imbalance_factor
            t_bwd += (dx_t + dw_t) * imbalance_factor

    # Recompute: re-do forward for selected ops before backward
    recompute_t = _recompute_time(stage_ops, model, system, strategy, gpu_name)
    t_bwd += recompute_t

    # Communication time for this stage's collectives
    t_comm_fwd = 0.0
    t_comm_bwd = 0.0
    for c in stage_collectives:
        group_size = _group_size(c.group, strategy)
        tier = tier_for_group(c.group, group_size, system)
        ct = collective_time(c, group_size, tier)
        # AG/RS at matmul boundaries: split evenly between fwd and bwd
        t_comm_fwd += ct * 0.5
        t_comm_bwd += ct * 0.5

    t_fwd += t_comm_fwd
    t_bwd += t_comm_bwd

    return StageTime(
        fwd=t_fwd,
        bwd=t_bwd,
        comm_fwd=t_comm_fwd,
        comm_bwd=t_comm_bwd,
        expert_imbalance_factor=imbalance,
    )


def _recompute_time(
    ops: list[Op], model: ModelSpec, system: SystemSpec,
    strategy: Strategy, gpu_name: str,
) -> float:
    """Extra forward time for recomputed ops (Korthikanti-style)."""
    policy = strategy.recompute.per_layer
    if not policy:
        return 0.0

    t = 0.0
    for op in ops:
        if op.layer_id < 0:
            continue
        lk = model.layers[op.layer_id].value if op.layer_id < len(model.layers) else ""
        tiers = policy.get(lk, set())
        if not tiers:
            continue
        if "full" in tiers:
            cost = op_cost(op, model)
            t += op_to_time(cost.fwd_flops, cost.fwd_bytes, cost.bound, system, gpu_name)

    return t


def _group_size(group: str, strategy: Strategy) -> int:
    if group == "TP":
        return strategy.tp
    if group == "CP":
        return strategy.cp
    if group == "EP":
        return strategy.ep
    if group == "DP":
        return strategy.dp
    return 1
