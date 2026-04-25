"""Stage composer — per-stage time for one microbatch.

Converts per-op FLOPs and byte counts to wall-clock time using
roofline model with achieved efficiency curves.
"""

from __future__ import annotations

import math
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
    fwd: float = 0.0
    bwd: float = 0.0
    bwd_dx: float = 0.0
    bwd_dw: float = 0.0
    comm_fwd: float = 0.0
    comm_bwd: float = 0.0


def ep_imbalance_factor(num_experts: int, ep: int, topk: int = 1) -> float:
    """EP load imbalance factor >= 1.0.

    With EP parallelism, each GPU handles num_experts/ep experts.
    Due to token routing randomness, the actual load is imbalanced.

    Model: factor = 1 + (topk / (num_experts / ep)) * sqrt(log(num_experts / ep))

    This is a simplified model based on the balls-into-bins analysis.
    When ep == 1 or num_experts <= 0, returns 1.0 (no imbalance).
    """
    if ep <= 1 or num_experts <= 0:
        return 1.0
    experts_per_gpu = num_experts / ep
    if experts_per_gpu <= 0:
        return 1.0
    factor = 1.0 + (topk / experts_per_gpu) * math.sqrt(math.log(max(experts_per_gpu, 2)))
    return max(factor, 1.0)


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
    t_bwd_dx = 0.0
    t_bwd_dw = 0.0

    for op in stage_ops:
        cost = op_cost(op, model)

        if cost.bound == "compute":
            fwd_t = op_to_time(cost.fwd_flops, 0, "compute", system, gpu_name)
            dx_t = op_to_time(cost.dx_flops, 0, "compute", system, gpu_name)
            dw_t = op_to_time(cost.dw_flops, 0, "compute", system, gpu_name)
            t_fwd += fwd_t
            t_bwd_dx += dx_t
            t_bwd_dw += dw_t
        else:
            fwd_t = op_to_time(0, cost.fwd_bytes, "memory", system, gpu_name)
            dx_t = op_to_time(0, cost.dx_bytes, "memory", system, gpu_name)
            dw_t = op_to_time(0, cost.dw_bytes, "memory", system, gpu_name)
            t_fwd += fwd_t
            t_bwd_dx += dx_t
            t_bwd_dw += dw_t

    # Recompute: re-do forward for selected ops before backward
    recompute_t = _recompute_time(stage_ops, model, system, strategy, gpu_name)
    t_bwd_dx += recompute_t

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
    t_bwd_dx += t_comm_bwd

    if strategy.ep > 1 and model.num_experts > 0:
        has_moe = any(op.layer_kind == LayerKind.MOE for op in stage_ops)
        if has_moe:
            imb = ep_imbalance_factor(model.num_experts, strategy.ep,
                                       getattr(model, 'top_k', 1))
            t_fwd *= imb
            t_bwd_dx *= imb
            t_bwd_dw *= imb

    t_bwd = t_bwd_dx + t_bwd_dw
    return StageTime(
        fwd=t_fwd,
        bwd=t_bwd,
        bwd_dx=t_bwd_dx,
        bwd_dw=t_bwd_dw,
        comm_fwd=t_comm_fwd,
        comm_bwd=t_comm_bwd,
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
        cats = policy.get(lk, set())
        if not cats:
            continue

        op_cats = _op_recompute_categories(op)
        if "full" in cats or (op_cats & cats):
            cost = op_cost(op, model)
            t += op_to_time(cost.fwd_flops, cost.fwd_bytes, cost.bound, system, gpu_name)

    return t


def _op_recompute_categories(op: Op) -> set[str]:
    """Map an op to its recompute category set."""
    if op.kind == "attn_core":
        return {"attn"}
    if op.kind == "matmul":
        name = op.name.lower()
        if "qkv" in name or "o_proj" in name:
            return {"attn"}
        if "up_proj" in name or "gate_proj" in name or "down_proj" in name:
            return {"ffn_swiglu"}
        return set()
    if op.kind == "swiglu":
        return {"ffn_swiglu"}
    if op.kind == "ln":
        return {"ln"}
    return set()


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
