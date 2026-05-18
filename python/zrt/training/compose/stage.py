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
    tp_hidden: float = 0.0  # TP comm hidden by CoC/MC2 (fwd + bwd combined)
    tp_exposed: float = 0.0  # TP comm exposed after CoC/MC2 (fwd + bwd combined)
    ep_exposed: float = 0.0  # EP comm exposed after wave-overlap (fwd + bwd combined)
    cp_exposed: float = 0.0  # CP comm exposed — no overlap mechanism (fwd + bwd combined)


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
    from zrt.training.io.perf_tables import peak_tflops_for
    gpu = system.gpu
    compute_t = 0.0
    if flops > 0:
        peak = peak_tflops_for(gpu, dtype)
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
    from zrt.training.io.perf_tables import peak_tflops_for
    gpu = system.gpu
    total_flops = cube_flops + vector_flops
    if not has_heterogeneous_compute(system):
        return op_to_time(total_flops, bytes_, system, gpu_name, dtype)

    # Scale cube/vector peaks by dtype/bf16 ratio (cube_tflops is BF16 peak).
    bf16_peak = gpu.flops_bf16 * 1e12
    dtype_peak = peak_tflops_for(gpu, dtype)
    scale = dtype_peak / bf16_peak if bf16_peak > 0 else 1.0

    compute_t = 0.0
    if total_flops > 0:
        eff = achieved_flops_efficiency(gpu_name or gpu.name, dtype, total_flops)
        cube_t = 0.0
        vector_t = 0.0
        if cube_flops > 0:
            peak_cube = gpu.cube_tflops * 1e12 * scale
            cube_t = cube_flops / (peak_cube * eff) if peak_cube > 0 else 0.0
        if vector_flops > 0:
            peak_vector = gpu.vector_tflops * 1e12 * scale
            vector_t = vector_flops / (peak_vector * eff) if peak_vector > 0 else 0.0
        if cube_t > 0 or vector_t > 0:
            compute_t = max(cube_t, vector_t) + (1.0 - overlap_ratio) * min(cube_t, vector_t)

    memory_t = 0.0
    if bytes_ > 0:
        bw = gpu.hbm_bw_gbps * 1e9
        eff_bw = achieved_bandwidth_efficiency(gpu_name or gpu.name, bytes_)
        memory_t = bytes_ / (bw * eff_bw) if bw > 0 else 0.0
    return max(compute_t, memory_t)


def _resolve_compute_dtype(op: Op, model: ModelSpec) -> Dtype:
    """Map ``op.component`` to its compute dtype, falling back to act_dtype.

    Used by stage.py to per-op-dispatch compute dtypes in mixed-quant
    configurations (FP8 attention with BF16 norms, FP4 routed experts with
    BF16 shared experts, etc.). Embedding and norm components are always
    BF16 — they have negligible compute cost and quantizing them is
    numerically unstable.
    """
    comp = getattr(op, "component", None)
    if comp == "attention":
        return model.attn_compute_dtype
    if comp == "routed_expert":
        return model.routed_expert_compute_dtype
    if comp == "shared_expert":
        return model.shared_expert_compute_dtype
    if comp in ("embedding", "norm"):
        return Dtype.BF16
    return model.act_dtype


def _cost_phase_time(
    cost: OpCost, phase: str, system: SystemSpec,
    gpu_name: str, overlap: float = 0.0, dtype: Dtype = Dtype.BF16,
) -> float:
    """Compute time for one phase (fwd/dx/dw) using heterogeneous roofline."""
    cube = getattr(cost, f"{phase}_cube_flops")
    vector = getattr(cost, f"{phase}_vector_flops")
    bytes_ = getattr(cost, f"{phase}_bytes")
    return op_to_time_hetero(cube, vector, bytes_, system, gpu_name, dtype,
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
        op_dtype = _resolve_compute_dtype(op, model)
        fwd_t = _cost_phase_time(cost, "fwd", system, gpu_name, overlap, op_dtype)
        dx_t  = _cost_phase_time(cost, "dx",  system, gpu_name, overlap, op_dtype)
        dw_t  = _cost_phase_time(cost, "dw",  system, gpu_name, overlap, op_dtype)
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
    t_cp_comm_fwd = 0.0
    t_cp_comm_bwd = 0.0
    t_ep_raw_comm_fwd = 0.0
    t_ep_raw_comm_bwd = 0.0
    t_other_comm_fwd = 0.0  # DP and any other groups
    t_other_comm_bwd = 0.0
    for c in stage_collectives:
        group_size = _group_size(c.group, strategy)
        tier = tier_for_group(c.group, group_size, system)
        ct = collective_time(c, group_size, tier)

        if c.group == "CP":
            if c.phase == "fwd":
                t_cp_comm_fwd += ct
            elif c.phase == "bwd":
                t_cp_comm_bwd += ct
            else:  # both
                t_cp_comm_fwd += ct * 0.5
                t_cp_comm_bwd += ct * 0.5
        elif c.group == "TP":
            # TP AG/RS are inserted with phase='both' by ir/shard.py and we
            # split 50/50 across fwd/bwd because Megatron-SP performs the
            # inverse op in backward (AG→RS, RS→AG) with the same payload.
            # Pinned by tests/training/test_tp_overlap.py::TestTPCollectivePhaseSplit.
            t_tp_comm_fwd += ct * 0.5
            t_tp_comm_bwd += ct * 0.5
        elif c.group == "EP":
            if c.phase == "fwd":
                t_ep_raw_comm_fwd += ct
            elif c.phase == "bwd":
                t_ep_raw_comm_bwd += ct
            else:  # both
                t_ep_raw_comm_fwd += ct * 0.5
                t_ep_raw_comm_bwd += ct * 0.5
        else:
            t_other_comm_fwd += ct * 0.5
            t_other_comm_bwd += ct * 0.5

    # Apply TP overlap with GEMM bound (parallels EP wave-overlap):
    #   CoC: K=4 wave-overlap; exposed = max(comm - gemm*(K-1)/K, 0) per phase
    #   MC2: bounded by max(comm - gemm, 0) — fully covered iff GEMM ≥ comm
    #   NONE: all TP comm exposed
    t_tp_hidden = 0.0
    t_tp_exposed_fwd = 0.0
    t_tp_exposed_bwd = 0.0
    if strategy.tp_overlap == TPOverlap.NONE:
        t_tp_exposed_fwd = t_tp_comm_fwd
        t_tp_exposed_bwd = t_tp_comm_bwd
    else:
        t_tp_gemm_fwd = _tp_gemm_time(stage_ops, model, system, gpu_name, "fwd")
        t_tp_gemm_bwd = _tp_gemm_time(stage_ops, model, system, gpu_name, "bwd")
        if strategy.tp_overlap == TPOverlap.MC2:
            # MC2 = "as much as GEMM lets us hide". Equivalent to K → ∞ in
            # _wave_overlap_saved, which collapses to min(comm, gemm).
            saved_fwd = min(t_tp_comm_fwd, t_tp_gemm_fwd)
            saved_bwd = min(t_tp_comm_bwd, t_tp_gemm_bwd)
        else:  # COC
            saved_fwd = _wave_overlap_saved(t_tp_comm_fwd, t_tp_gemm_fwd, K=4)
            saved_bwd = _wave_overlap_saved(t_tp_comm_bwd, t_tp_gemm_bwd, K=4)
        t_tp_exposed_fwd = max(0.0, t_tp_comm_fwd - saved_fwd)
        t_tp_exposed_bwd = max(0.0, t_tp_comm_bwd - saved_bwd)
        t_tp_hidden = saved_fwd + saved_bwd

    # Build combined t_comm_fwd/bwd from all components
    t_comm_fwd = t_other_comm_fwd + t_tp_exposed_fwd + t_cp_comm_fwd + t_ep_raw_comm_fwd
    t_comm_bwd = t_other_comm_bwd + t_tp_exposed_bwd + t_cp_comm_bwd + t_ep_raw_comm_bwd

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
            # Apply imbalance to EP comm only (CP and TP are not EP-parallel)
            # Track the EP comm before imbalance to compute the delta
            ep_comm_fwd_before = t_ep_raw_comm_fwd
            ep_comm_bwd_before = t_ep_raw_comm_bwd
            t_ep_raw_comm_fwd *= imb
            t_ep_raw_comm_bwd *= imb
            # Rebuild combined t_comm_fwd/bwd with imbalanced EP
            t_comm_fwd = t_other_comm_fwd + t_tp_exposed_fwd + t_cp_comm_fwd + t_ep_raw_comm_fwd
            t_comm_bwd = t_other_comm_bwd + t_tp_exposed_bwd + t_cp_comm_bwd + t_ep_raw_comm_bwd
            # Add the EP comm delta to stage totals so the full imbalance is reflected
            t_fwd += (t_ep_raw_comm_fwd - ep_comm_fwd_before)
            t_bwd_dx += (t_ep_raw_comm_bwd - ep_comm_bwd_before)

    # EP wave-overlap: split EP A2A into K waves, overlap with expert GEMM.
    # Fix (2026-05-10): compute fwd and bwd overlap independently, since
    # bwd GEMM (dx + dw) is ~2.5× fwd GEMM and can hide much more comm.
    # Also separate EP comm into fwd/bwd portions instead of treating them
    # as a single pool.
    # Fix (2026-05-18): use pre-computed t_ep_raw_comm_fwd/bwd (post-imbalance);
    # no inner re-loop over collectives needed.
    t_ep_hidden = 0.0
    t_ep_exposed_fwd = t_ep_raw_comm_fwd  # default: no overlap
    t_ep_exposed_bwd = t_ep_raw_comm_bwd
    if strategy.ep_overlap and strategy.ep > 1 and model.num_experts > 0:
        # EP GEMM times needed by both K-wave formula and inter-batch non_ep_compute.
        t_ep_gemm_fwd = _ep_gemm_time(stage_ops, model, system, strategy, gpu_name)
        t_ep_gemm_bwd = 0.0
        for op in stage_ops:
            if op.kind == "matmul" and "routed_expert" in op.name:
                cost = op_cost(op, model, system)
                op_overlap = gpu.overlap_ratio.get(op.kind, 0.0)
                op_dtype = _resolve_compute_dtype(op, model)
                dx_t = _cost_phase_time(cost, "dx", system, gpu_name, op_overlap, op_dtype)
                dw_t = _cost_phase_time(cost, "dw", system, gpu_name, op_overlap, op_dtype)
                t_ep_gemm_bwd += dx_t + dw_t

        saved_fwd = 0.0
        saved_bwd = 0.0

        # Intra-batch K-wave overlap: splits EP A2A into K sub-waves, each pipelined
        # with expert GEMM within the same microbatch.
        # Requires hardware-level fine-grained A2A/GEMM streaming (NVIDIA CUDA: K=4;
        # Ascend HCCS: ep_overlap_waves=0, not supported).
        if gpu.ep_overlap_waves > 0:
            K = gpu.ep_overlap_waves
            saved_fwd = _wave_overlap_saved(t_ep_raw_comm_fwd, t_ep_gemm_fwd, K)
            saved_fwd = min(saved_fwd, t_ep_raw_comm_fwd)
            t_ep_hidden += saved_fwd

            saved_bwd = _wave_overlap_saved(t_ep_raw_comm_bwd, t_ep_gemm_bwd, K)
            saved_bwd = min(saved_bwd, t_ep_raw_comm_bwd)
            t_ep_hidden += saved_bwd

        # Inter-batch EP overlap: when dual_batch=True and pp>1, two microbatches
        # run simultaneously. Batch A's residual EP A2A (after K-wave overlap)
        # can be hidden by batch B's non-EP compute.
        #
        # This is a DualPipe scheduling property — independent of K-wave hardware
        # support. Ascend A6 with ep_overlap_waves=0 still benefits here because
        # HCCS EP A2A from one microbatch can overlap with DaVinci compute from
        # the concurrent microbatch.
        #
        # Non-EP compute = total pure compute minus expert GEMM.
        # We subtract ep_gemm to avoid double-counting: batch B's expert GEMM is
        # already consumed by its own intra-batch wave-overlap and cannot
        # simultaneously hide batch A's EP A2A.
        #
        # Head/tail correction: warmup and cooldown pipeline slots have no paired
        # microbatch, so inter-batch hiding does not apply there. DualPipe(V) has
        # bubble = (pp-1)/(2V) * t_stage. Scaling inter_saved by the steady-state
        # fraction M/(M + bubble_slots) makes the aggregate step saving exactly
        # M * inter_saved (steady only), not (M + bubble_slots) * inter_saved.
        if strategy.dualbatch and strategy.pp > 1:
            residual_ep_fwd = max(0.0, t_ep_raw_comm_fwd - saved_fwd)
            residual_ep_bwd = max(0.0, t_ep_raw_comm_bwd - saved_bwd)

            non_ep_compute_fwd = max(0.0, t_fwd - t_comm_fwd - t_ep_gemm_fwd)
            non_ep_compute_bwd = max(0.0, (t_bwd_dx + t_bwd_dw) - t_comm_bwd - t_ep_gemm_bwd)

            inter_saved_fwd = min(residual_ep_fwd, non_ep_compute_fwd)
            inter_saved_bwd = min(residual_ep_bwd, non_ep_compute_bwd)

            V = max(1, strategy.vpp_chunks)
            M_mb = strategy.num_microbatches()
            bubble_slots = (strategy.pp - 1) / (2.0 * V)
            denom = M_mb + bubble_slots
            steady_fraction = M_mb / denom if denom > 0 else 1.0
            inter_saved_fwd *= steady_fraction
            inter_saved_bwd *= steady_fraction

            t_ep_hidden += inter_saved_fwd + inter_saved_bwd
            saved_fwd += inter_saved_fwd
            saved_bwd += inter_saved_bwd

        # Remove hidden EP from combined comm totals
        t_comm_fwd -= saved_fwd
        t_comm_bwd -= saved_bwd
        # Update fwd/bwd totals (EP portion reduced)
        t_fwd -= saved_fwd
        t_bwd_dx -= saved_bwd

        # EP exposed per direction
        t_ep_exposed_fwd = max(0.0, t_ep_raw_comm_fwd - saved_fwd)
        t_ep_exposed_bwd = max(0.0, t_ep_raw_comm_bwd - saved_bwd)

    t_bwd = t_bwd_dx + t_bwd_dw
    return StageTime(
        fwd=t_fwd,
        bwd=t_bwd,
        bwd_dx=t_bwd_dx,
        bwd_dw=t_bwd_dw,
        comm_fwd=t_comm_fwd,
        comm_bwd=t_comm_bwd,
        ep_hidden=t_ep_hidden,
        tp_hidden=t_tp_hidden,
        tp_exposed=t_tp_exposed_fwd + t_tp_exposed_bwd,
        ep_exposed=t_ep_exposed_fwd + t_ep_exposed_bwd,
        cp_exposed=t_cp_comm_fwd + t_cp_comm_bwd,
    )


def _recompute_time(
    ops: list[Op], model: ModelSpec, system: SystemSpec,
    strategy: Strategy, gpu_name: str,
) -> float:
    """Extra forward time for recomputed ops (Korthikanti-style).

    Mirrors ``recompute_overhead_flops`` (FLOPs side) — same category
    mapping, same FA-kernel dedup. The category mapping is imported from
    flops.py so the two sides stay in sync.
    """
    from zrt.training.models.flops import _op_recompute_categories as _flops_op_cats

    policy = strategy.recompute.per_layer
    if not policy:
        return 0.0

    FA_KERNEL_KINDS = {"attn_core", "sparse_attn", "hca_attn", "swa_attn"}

    t = 0.0
    for op in ops:
        if op.layer_id < 0:
            continue
        lk = model.layers[op.layer_id].value if op.layer_id < len(model.layers) else ""
        cats = policy.get(lk, set())
        if not cats:
            continue

        op_cats = _flops_op_cats(op)
        if "full" in cats or (op_cats & cats):
            if op.kind in FA_KERNEL_KINDS:
                # FA backward already pays the recompute time via the
                # 2.5×fwd convention — see recompute_overhead_flops.
                continue
            cost = op_cost(op, model, system)
            overlap = system.gpu.overlap_ratio.get(op.kind, 0.0)
            op_dtype = _resolve_compute_dtype(op, model)
            t += _cost_phase_time(cost, "fwd", system, gpu_name, overlap, op_dtype)

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
            op_dtype = _resolve_compute_dtype(op, model)
            t = _cost_phase_time(cost, "fwd", system, gpu_name,
                                 system.gpu.overlap_ratio.get(op.kind, 0.0), op_dtype)
        else:
            continue
        t_total += t
        if op.kind == "matmul" and "routed_expert" in op.name:
            t_ep += t
    if t_total <= 0:
        return 0.0
    return t_ep / t_total


# NOTE: the canonical _op_recompute_categories lives in zrt.training.models.flops.
# stage.py and flops.py used to keep duplicate copies that drifted (this one
# was missing V3 MLA / V4 matmul names and the compressor/indexer pool entries),
# silently undercounting backward-stage recompute time for those ops. The
# single source of truth is now imported above in _recompute_time().


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
            op_dtype = _resolve_compute_dtype(op, model)
            total += _cost_phase_time(cost, "fwd", system, gpu_name,
                                      system.gpu.overlap_ratio.get(op.kind, 0.0), op_dtype)
    return total


def _tp_gemm_time(
    ops: list[Op], model: ModelSpec, system: SystemSpec,
    gpu_name: str, phase: str = "fwd",
) -> float:
    """Total matmul time available to overlap TP AG/RS (Megatron-SP).

    In Megatron-SP, each AG/RS is bracketed by a dense matmul that runs while
    the collective is in flight. The overlappable matmuls are exactly those
    that ir/shard.py:_insert_tp_collectives attaches an AG/RS to:
        - qkv (AG before)        - o_proj (RS after)
        - up_proj (AG before)    - down_proj (RS after)
        - gate_proj (AG before, SwiGLU path)

    phase="fwd" sums fwd time; phase="bwd" sums dx + dw time per matmul.
    """
    OVERLAPPABLE_SUBSTRINGS = ("qkv", "o_proj", "up_proj", "down_proj", "gate_proj")
    total = 0.0
    for op in ops:
        if op.kind != "matmul":
            continue
        if not any(s in op.name for s in OVERLAPPABLE_SUBSTRINGS):
            continue
        cost = op_cost(op, model, system)
        overlap = system.gpu.overlap_ratio.get(op.kind, 0.0)
        op_dtype = _resolve_compute_dtype(op, model)
        if phase == "fwd":
            total += _cost_phase_time(cost, "fwd", system, gpu_name, overlap, op_dtype)
        else:  # "bwd"
            total += _cost_phase_time(cost, "dx", system, gpu_name, overlap, op_dtype)
            total += _cost_phase_time(cost, "dw", system, gpu_name, overlap, op_dtype)
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
