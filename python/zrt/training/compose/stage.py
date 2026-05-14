"""Stage composer — per-stage time for one microbatch.

Converts per-op FLOPs and byte counts to wall-clock time using
roofline model with achieved efficiency curves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from zrt.training.ir.training_graph import Collective, Graph, Op
from zrt.training.io.perf_tables import achieved_bandwidth_efficiency, achieved_flops_efficiency
from zrt.training.models.comm import collective_time, tier_for_group, total_comm_time
from zrt.training.models.flops import OpCost, op_cost
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import Strategy, TPOverlap
from zrt.training.spec.system import SystemSpec


@dataclass
class StageTime:
    fwd: float = 0.0
    bwd: float = 0.0
    bwd_dx: float = 0.0
    bwd_dw: float = 0.0
    comm_fwd: float = 0.0   # exposed comm in fwd (after TP/EP overlap reductions)
    comm_bwd: float = 0.0   # exposed comm in bwd (after TP/EP overlap reductions)
    ep_hidden: float = 0.0  # EP comm hidden by wave-overlap (fwd + bwd combined)


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
    flops: float, bytes_: float, system: SystemSpec,
    gpu_name: str = "", dtype: Dtype = Dtype.BF16,
) -> float:
    """Roofline: op time = max(compute_time, memory_time)."""
    gpu = system.gpu
    compute_t = 0.0
    if flops > 0:
        peak = gpu.flops_bf16 * 1e12
        eff = achieved_flops_efficiency(gpu_name or gpu.name, dtype, flops)
        compute_t = flops / (peak * eff) if peak > 0 else 0.0
    memory_t = 0.0
    if bytes_ > 0:
        bw = gpu.hbm_bw_gbps * 1e9
        eff = achieved_bandwidth_efficiency(gpu_name or gpu.name, bytes_)
        memory_t = bytes_ / (bw * eff) if bw > 0 else 0.0
    return max(compute_t, memory_t)


def has_heterogeneous_compute(system: SystemSpec) -> bool:
    """Return True when both heterogeneous compute peaks are configured."""
    gpu = system.gpu
    return gpu.cube_tflops is not None and gpu.vector_tflops is not None


def op_to_time_hetero(
    cube_flops: float, vector_flops: float, bytes_: float,
    system: SystemSpec, gpu_name: str = "", dtype: Dtype = Dtype.BF16,
    overlap_ratio: float = 0.0,
) -> float:
    """Heterogeneous roofline for matrix/Tensor/Cube + Vector hardware.

    compute_time = max(cube_t, vector_t) + (1 - overlap_ratio) * min(cube_t, vector_t)
    op_time = max(compute_time, memory_time)

    If either heterogeneous peak is missing, preserve the legacy unified-peak
    roofline instead of treating one side of the work as free.
    """
    gpu = system.gpu
    total_flops = cube_flops + vector_flops
    if not has_heterogeneous_compute(system):
        return op_to_time(total_flops, bytes_, system, gpu_name, dtype)

    compute_t = 0.0
    if total_flops > 0:
        eff = achieved_flops_efficiency(gpu_name or gpu.name, dtype, total_flops)
        cube_t = 0.0
        vector_t = 0.0
        if cube_flops > 0:
            peak_cube = gpu.cube_tflops * 1e12
            cube_t = cube_flops / (peak_cube * eff) if peak_cube > 0 else 0.0
        if vector_flops > 0:
            peak_vector = gpu.vector_tflops * 1e12
            vector_t = vector_flops / (peak_vector * eff) if peak_vector > 0 else 0.0
        if cube_t > 0 or vector_t > 0:
            compute_t = max(cube_t, vector_t) + (1.0 - overlap_ratio) * min(cube_t, vector_t)

    memory_t = 0.0
    if bytes_ > 0:
        bw = gpu.hbm_bw_gbps * 1e9
        eff_bw = achieved_bandwidth_efficiency(gpu_name or gpu.name, bytes_)
        memory_t = bytes_ / (bw * eff_bw) if bw > 0 else 0.0
    return max(compute_t, memory_t)


def _cost_phase_time(
    cost: OpCost, phase: str, system: SystemSpec,
    gpu_name: str, overlap: float = 0.0,
) -> float:
    """Compute time for one phase (fwd/dx/dw) using heterogeneous roofline."""
    cube = getattr(cost, f"{phase}_cube_flops")
    vector = getattr(cost, f"{phase}_vector_flops")
    bytes_ = getattr(cost, f"{phase}_bytes")
    return op_to_time_hetero(cube, vector, bytes_, system, gpu_name,
                             overlap_ratio=overlap)


def stage_time(
    stage_ops: list[Op],
    stage_collectives: list[Collective],
    model: ModelSpec,
    system: SystemSpec,
    strategy: Strategy,
) -> StageTime:
    """Compute forward + backward time for one PP stage and one microbatch."""
    gpu_name = system.gpu.name
    gpu = system.gpu

    t_fwd = 0.0
    t_bwd_dx = 0.0
    t_bwd_dw = 0.0

    for op in stage_ops:
        cost = op_cost(op, model, system)
        overlap = gpu.overlap_ratio.get(op.kind, 0.0)
        fwd_t = _cost_phase_time(cost, "fwd", system, gpu_name, overlap)
        dx_t  = _cost_phase_time(cost, "dx",  system, gpu_name, overlap)
        dw_t  = _cost_phase_time(cost, "dw",  system, gpu_name, overlap)
        t_fwd    += fwd_t
        t_bwd_dx += dx_t
        t_bwd_dw += dw_t

    # Recompute: re-do forward for selected ops before backward
    recompute_t = _recompute_time(stage_ops, model, system, strategy, gpu_name)
    t_bwd_dx += recompute_t

    t_comm_fwd = 0.0
    t_comm_bwd = 0.0
    t_tp_comm_fwd = 0.0
    t_tp_comm_bwd = 0.0
    for c in stage_collectives:
        group_size = _group_size(c.group, strategy)
        tier = tier_for_group(c.group, group_size, system)
        ct = collective_time(c, group_size, tier)

        if c.group == "CP":
            if c.phase == "fwd":
                t_comm_fwd += ct
            elif c.phase == "bwd":
                t_comm_bwd += ct
            elif c.phase == "both":
                t_comm_fwd += ct * 0.5
                t_comm_bwd += ct * 0.5
        elif c.group == "TP":
            t_tp_comm_fwd += ct * 0.5
            t_tp_comm_bwd += ct * 0.5
        else:
            t_comm_fwd += ct * 0.5
            t_comm_bwd += ct * 0.5

    # Apply TP overlap: CoC exposes ~10%, MC2 exposes 0%, NONE exposes 100%
    tp_expose = 1.0
    if strategy.tp_overlap == TPOverlap.MC2:
        tp_expose = 0.0
    elif strategy.tp_overlap == TPOverlap.COC:
        tp_expose = 0.1
    t_comm_fwd += t_tp_comm_fwd * tp_expose
    t_comm_bwd += t_tp_comm_bwd * tp_expose

    t_fwd += t_comm_fwd
    t_bwd_dx += t_comm_bwd

    if strategy.ep > 1 and model.num_experts > 0:
        has_moe = any(op.layer_kind == LayerKind.MOE for op in stage_ops)
        if has_moe:
            imb = ep_imbalance_factor(model.num_experts, strategy.ep,
                                       getattr(model, 'top_k', 1))
            # Apply imbalance only to EP-parallel fraction (routed expert FFN ops)
            # Non-EP ops (attention, shared expert, embed) are replicated and not imbalanced
            ep_frac = _ep_parallel_fraction(stage_ops, model, system, strategy, gpu_name)
            t_fwd = t_fwd * (1 - ep_frac) + t_fwd * ep_frac * imb
            t_bwd_dx = t_bwd_dx * (1 - ep_frac) + t_bwd_dx * ep_frac * imb
            t_bwd_dw = t_bwd_dw * (1 - ep_frac) + t_bwd_dw * ep_frac * imb
            t_comm_fwd = t_comm_fwd * (1 - ep_frac) + t_comm_fwd * ep_frac * imb
            t_comm_bwd = t_comm_bwd * (1 - ep_frac) + t_comm_bwd * ep_frac * imb

    # EP wave-overlap: split EP A2A into K waves, overlap with expert GEMM.
    # Fix (2026-05-10): compute fwd and bwd overlap independently, since
    # bwd GEMM (dx + dw) is ~2.5× fwd GEMM and can hide much more comm.
    # Also separate EP comm into fwd/bwd portions instead of treating them
    # as a single pool.
    t_ep_hidden = 0.0
    if strategy.ep_overlap and strategy.ep > 1 and model.num_experts > 0:
        # Separate EP comm into fwd and bwd portions
        t_comm_ep_fwd = 0.0
        t_comm_ep_bwd = 0.0
        for c in stage_collectives:
            if c.group != "EP":
                continue
            group_size = _group_size(c.group, strategy)
            tier = tier_for_group(c.group, group_size, system)
            ct = collective_time(c, group_size, tier)
            if c.phase == "fwd":
                t_comm_ep_fwd += ct
            elif c.phase == "bwd":
                t_comm_ep_bwd += ct
            elif c.phase == "both":
                t_comm_ep_fwd += ct * 0.5
                t_comm_ep_bwd += ct * 0.5

        K = 4  # number of overlap waves

        # Fwd overlap: use fwd GEMM time
        t_ep_gemm_fwd = _ep_gemm_time(stage_ops, model, system, strategy, gpu_name)
        saved_fwd = _wave_overlap_saved(t_comm_ep_fwd, t_ep_gemm_fwd, K)
        saved_fwd = min(saved_fwd, t_comm_fwd)  # can't save more than comm exists
        t_fwd -= saved_fwd
        t_comm_fwd -= saved_fwd
        t_ep_hidden += saved_fwd

        # Bwd overlap: use bwd GEMM time (dx + dw), not just fwd
        t_ep_gemm_bwd = 0.0
        for op in stage_ops:
            if op.kind == "matmul" and "routed_expert" in op.name:
                cost = op_cost(op, model, system)
                dx_t = _cost_phase_time(cost, "dx", system, gpu_name, overlap)
                dw_t = _cost_phase_time(cost, "dw", system, gpu_name, overlap)
                t_ep_gemm_bwd += dx_t + dw_t
        saved_bwd = _wave_overlap_saved(t_comm_ep_bwd, t_ep_gemm_bwd, K)
        saved_bwd = min(saved_bwd, t_comm_bwd)
        t_bwd_dx -= saved_bwd
        t_comm_bwd -= saved_bwd
        t_ep_hidden += saved_bwd

    t_bwd = t_bwd_dx + t_bwd_dw
    return StageTime(
        fwd=t_fwd,
        bwd=t_bwd,
        bwd_dx=t_bwd_dx,
        bwd_dw=t_bwd_dw,
        comm_fwd=t_comm_fwd,
        comm_bwd=t_comm_bwd,
        ep_hidden=t_ep_hidden,
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
            cost = op_cost(op, model, system)
            overlap = system.gpu.overlap_ratio.get(op.kind, 0.0)
            t += _cost_phase_time(cost, "fwd", system, gpu_name, overlap)

    return t


def _ep_parallel_fraction(
    ops: list[Op], model: ModelSpec, system: SystemSpec,
    strategy: Strategy, gpu_name: str,
) -> float:
    """Estimate the fraction of compute time from EP-parallel ops.

    Only routed expert FFN ops are EP-parallel; attention, shared expert,
    embedding, and other ops are replicated across EP ranks.
    Returns a value in [0, 1].
    """
    t_total = 0.0
    t_ep = 0.0
    for op in ops:
        cost = op_cost(op, model, system)
        if cost.fwd_cube_flops > 0 or cost.fwd_vector_flops > 0 or cost.fwd_bytes > 0:
            t = _cost_phase_time(cost, "fwd", system, gpu_name,
                                 system.gpu.overlap_ratio.get(op.kind, 0.0))
        else:
            continue
        t_total += t
        if op.kind == "matmul" and "routed_expert" in op.name:
            t_ep += t
    if t_total <= 0:
        return 0.0
    return t_ep / t_total


def _op_recompute_categories(op: Op) -> set[str]:
    """Map an op to its recompute category set."""
    if op.kind in ("attn_core", "sparse_attn", "hca_attn", "swa_attn"):
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


def _wave_overlap_saved(comm_time: float, gemm_time: float, K: int = 4) -> float:
    """Compute how much EP comm time is saved by wave overlap with GEMM.

    Model: split comm and GEMM into K waves. First wave's comm is fully
    exposed (no prior compute to overlap with). Subsequent waves can
    overlap comm with the previous wave's GEMM.

        exposed_total = comm_per_wave + (K-1) * max(comm_per_wave - gemm_per_wave, 0)
        saved = total_comm - exposed_total
    """
    if comm_time <= 0 or gemm_time <= 0 or K <= 0:
        return 0.0
    comm_per_wave = comm_time / K
    gemm_per_wave = gemm_time / K
    exposed_per_wave = max(comm_per_wave - gemm_per_wave, 0.0)
    exposed_total = comm_per_wave + (K - 1) * exposed_per_wave
    return max(0.0, comm_time - exposed_total)


def _ep_comm_time(
    collectives: list[Collective], strategy: Strategy, system: SystemSpec,
) -> float:
    """Total EP A2A communication time (seconds)."""
    from zrt.training.models.comm import collective_time, tier_for_group
    total = 0.0
    for c in collectives:
        if c.group == "EP":
            group_size = _group_size(c.group, strategy)
            tier = tier_for_group(c.group, group_size, system)
            total += collective_time(c, group_size, tier)
    return total


def _ep_gemm_time(
    ops: list[Op], model: ModelSpec, system: SystemSpec,
    strategy: Strategy, gpu_name: str,
) -> float:
    """Routed expert GEMM time (seconds) — the compute that overlaps with EP A2A."""
    total = 0.0
    for op in ops:
        if op.kind == "matmul" and "routed_expert" in op.name:
            cost = op_cost(op, model, system)
            total += _cost_phase_time(cost, "fwd", system, gpu_name,
                                      system.gpu.overlap_ratio.get(op.kind, 0.0))
    return total


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
