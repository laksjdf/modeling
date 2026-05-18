"""Memory model — weights, grads, optimizer state, activations.

Reference: ZeRO (Rajbhandari et al. 2020), Korthikanti et al. 2022,
DeepSeek Memory Analysis (Yang et al. 2025).
"""

from __future__ import annotations

from dataclasses import dataclass

from zrt.training.ir.training_graph import Graph, Op
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import SystemSpec


@dataclass
class MemBreakdown:
    weights: float = 0.0       # bytes
    grads: float = 0.0         # bytes
    opt_state: float = 0.0     # bytes
    activations: float = 0.0   # bytes (includes hc_overhead_bytes)
    comm_buffers: float = 0.0  # bytes
    hc_overhead_bytes: float = 0.0  # bytes from HC residual replication
    muon_ns_buffer: float = 0.0     # bytes — transient AG + A=XᵀX during NS step

    # Phase-specific peaks (bytes). Each is what the GPU holds simultaneously
    # during that phase of one training step. peak_overall is the max — that's
    # the OOM-relevant number. ``total`` (sum of components) is the conservative
    # upper bound; in reality activations and opt_state never coexist.
    peak_forward: float = 0.0     # weights + activations + comm_buffers
    peak_backward: float = 0.0    # weights + activations + grads + comm_buffers
    peak_optimizer: float = 0.0   # weights + grads + opt_state
    peak_overall: float = 0.0     # max of the three above

    @property
    def total(self) -> float:
        """Conservative upper bound — algebraic sum of all components.

        Real GPU memory at any moment is bounded by ``peak_overall``; ``total``
        is always ≥ ``peak_overall``. Includes ``muon_ns_buffer`` (the transient
        AllGather + scratch that lives only during the NS step) so the invariant
        holds even when that spike exceeds activation memory.
        """
        return (self.weights + self.grads + self.opt_state + self.activations
                + self.comm_buffers + self.muon_ns_buffer)

    def to_gb(self) -> dict[str, float]:
        GB = 1024 ** 3
        return {
            "weights_gb": self.weights / GB,
            "grads_gb": self.grads / GB,
            "opt_state_gb": self.opt_state / GB,
            "activations_gb": self.activations / GB,
            "comm_buffers_gb": self.comm_buffers / GB,
            "hc_overhead_gb": self.hc_overhead_bytes / GB,
            "muon_ns_buffer_gb": self.muon_ns_buffer / GB,
            "total_gb": self.total / GB,
            "peak_gb": self.peak_overall / GB,
            "peak_forward_gb": self.peak_forward / GB,
            "peak_backward_gb": self.peak_backward / GB,
            "peak_optimizer_gb": self.peak_optimizer / GB,
        }


def _muon_ns_peak_buffer(model: ModelSpec, strategy: Strategy) -> int:
    """Transient memory spike during Muon Newton-Schulz orthogonalization.

    Two sources:
    1. AllGather momentum: ZeRO shards momentum by DP, but NS needs the full
       un-sharded matrix. Spike = P_muon × 4B × (DP-1)/DP.
    2. Intermediate A = XₖᵀXₖ ∈ R^{n×n}: computed per weight matrix
       sequentially. Peak = max_short_dim² × 4B (largest weight matrix on rank).
    """
    if strategy.optimizer.value != "muon":
        return 0

    muon_config = strategy.muon_config
    f_muon = (
        muon_config.muon_param_fraction
        if muon_config and muon_config.muon_param_fraction is not None
        else 0.85
    )

    P = _params_on_rank(model, strategy)
    P_muon = int(P * f_muon)

    # Buffer 1: AllGather momentum spike (ZeRO-1/2/3 only)
    if strategy.dp > 1 and strategy.zero_stage >= 1:
        # Rank normally holds P_muon × 4B / DP; AllGather brings it to P_muon × 4B
        ag_extra = int(P_muon * 4 * (strategy.dp - 1) / strategy.dp)
    else:
        ag_extra = 0

    # Buffer 2: Intermediate A = XₖᵀXₖ (largest TP-sharded weight matrix, FP32)
    # A = XᵀX where X ∈ R^{m×n} → A ∈ R^{short×short}, short = min(m, n).
    # After TP column-sharding the short dim is min(hidden, col_dim/tp).
    tp = max(1, strategy.tp)
    attn_short = model.hidden // tp                              # q/k/v/o: (hidden, hidden/tp)
    ffn_short  = min(model.hidden, model.ffn    // tp)          # up/gate/down: min(h, ffn/tp)
    moe_short  = min(model.hidden, model.moe_ffn // tp) if model.num_experts > 0 else 0
    max_short  = max(attn_short, ffn_short, moe_short)
    a_matrix = max_short * max_short * 4     # FP32, one matrix at a time

    return ag_extra + a_matrix


def memory_breakdown(
    graph: Graph,
    model: ModelSpec,
    system: SystemSpec,
    strategy: Strategy,
    stage_layer_ids: list[int] | None = None,
) -> MemBreakdown:
    """Compute per-rank memory breakdown in bytes.

    Accounts for TP/PP sharding, ZeRO stages, and activation memory.
    """
    # ── Parameters on this rank ──────────────────────────────────────────
    P = _params_on_rank(model, strategy)

    # Per-component weight bytes: routed-expert weights use
    # routed_expert_weight_dtype (FP4 stored-size includes per-block BF16 scale);
    # everything else uses param_dtype.
    P_expert = _routed_expert_params_on_rank(model, strategy)
    P_other = P - P_expert
    weights = int(
        P_expert * model.routed_expert_weight_dtype.stored_bytes
        + P_other * model.param_dtype.stored_bytes
    )

    grads = int(
        P_expert * model.routed_expert_grad_dtype.bytes
        + P_other * model.grad_dtype.bytes
    )
    opt_state = _optimizer_state_bytes(P, model, strategy)

    # ZeRO sharding
    dp = strategy.dp
    if strategy.zero_stage >= 3:
        weights //= dp
        grads //= dp
        opt_state //= dp
    elif strategy.zero_stage >= 2:
        grads //= dp
        opt_state //= dp
    elif strategy.zero_stage >= 1:
        opt_state //= dp

    # ── Activations ──────────────────────────────────────────────────────
    if stage_layer_ids is not None:
        layer_ids = stage_layer_ids
    else:
        n_layers = len(model.layers)
        if strategy.pp > 1:
            # When the caller doesn't pin a stage, assume the average per-stage
            # load. _activation_memory multiplies by _pp_in_flight, which
            # represents the number of microbatches co-resident on one rank;
            # the sum-over-all-layers path would double-count by `pp`.
            layers_per_stage = max(1, n_layers // strategy.pp)
            layer_ids = list(range(layers_per_stage))
        else:
            layer_ids = list(range(n_layers))

    activations, hc_overhead = _activation_memory(model, strategy, layer_ids)

    # ── Communication buffers ────────────────────────────────────────────
    comm_buffers = _comm_buffer_memory(model, strategy)

    # ── Offload ──────────────────────────────────────────────────────────
    off = strategy.offload
    if off.pct > 0:
        if off.opt_state:
            opt_state = int(opt_state * (1 - off.pct))
        if off.grads:
            grads = int(grads * (1 - off.pct))
        if off.params:
            weights = int(weights * (1 - off.pct))

    # ── Muon NS transient buffer ─────────────────────────────────────────────
    muon_ns_buf = _muon_ns_peak_buffer(model, strategy)

    mb = MemBreakdown(
        weights=weights,
        grads=grads,
        opt_state=opt_state,
        activations=activations,
        comm_buffers=comm_buffers,
        hc_overhead_bytes=hc_overhead,
        muon_ns_buffer=muon_ns_buf,
    )
    # Phase peaks: real GPU residency in each phase of one training step.
    # peak_overall is the OOM-relevant number; .total is a conservative
    # upper bound that adds opt_state on top of activations even though
    # they never coexist.
    mb.peak_forward = weights + activations + comm_buffers
    mb.peak_backward = weights + activations + grads + comm_buffers
    mb.peak_optimizer = weights + grads + opt_state + muon_ns_buf
    mb.peak_overall = max(mb.peak_forward, mb.peak_backward, mb.peak_optimizer)
    return mb


def _params_on_rank(model: ModelSpec, strategy: Strategy) -> int:
    """Total parameters held on one rank after TP + PP + EP sharding."""
    # Compute dense and MoE params separately for accurate EP sharding
    dense_params = _dense_params(model)
    moe_params = _moe_params(model)

    total = dense_params + moe_params

    # TP: shard all params by TP
    if strategy.tp > 1:
        total //= strategy.tp

    # EP: shard routed expert params by EP (shared experts NOT sharded by EP)
    if strategy.ep > 1 and model.num_experts > 0:
        shared_params = _shared_expert_params(model)
        routed_params = moe_params - shared_params
        if strategy.tp > 1:
            shared_params //= strategy.tp
            routed_params //= strategy.tp
        # EP sharding applies to routed experts only
        routed_after_ep = routed_params // strategy.ep
        moe_after_tp_ep = shared_params + routed_after_ep
        # Replace the TP-sharded MoE portion with TP*EP-sharded version
        moe_after_tp = moe_params // strategy.tp if strategy.tp > 1 else moe_params
        total = (total - moe_after_tp) + moe_after_tp_ep

    # PP: only hold params for layers on this stage
    if strategy.pp > 1:
        n_layers = len(model.layers)
        layers_per_stage = n_layers / strategy.pp
        # Proportional: non-embedding params scale with layers
        embed_params = model.vocab * model.hidden * 2  # embed + lm_head
        non_embed = total - embed_params
        non_embed = int(non_embed * layers_per_stage / n_layers)
        total = non_embed + embed_params // strategy.pp

    return total


def _dense_params(model: ModelSpec) -> int:
    """Parameters from dense layers only (no MoE experts)."""
    n_dense = sum(1 for lk in model.layers if lk.value == "dense")
    n_mtp = sum(1 for lk in model.layers if lk.value == "mtp")
    # Per dense layer: attn (4 * hidden^2) + FFN (2 * hidden * ffn) + norms/bias
    attn = 4 * model.hidden * model.hidden
    ffn = 2 * model.hidden * model.ffn
    per_dense = attn + ffn + 4 * model.hidden  # norms + biases
    # Embedding + lm_head (counted once, not per layer)
    embed = model.vocab * model.hidden * 2
    return n_dense * per_dense + embed + n_mtp * per_dense


def _moe_params(model: ModelSpec) -> int:
    """Parameters from MoE experts (routed + shared)."""
    if model.num_experts <= 0:
        return 0
    n_moe = sum(1 for lk in model.layers if lk.value == "moe")
    # Per expert: FFN (2 * hidden * moe_ffn)
    per_expert = 2 * model.hidden * model.moe_ffn
    # Shared expert: same as one routed expert
    n_shared = getattr(model, "n_shared_experts", 1) or 1
    # Total experts per MoE layer = num_experts + shared experts
    total_experts_per_layer = model.num_experts + n_shared
    return n_moe * total_experts_per_layer * per_expert


def _shared_expert_params(model: ModelSpec) -> int:
    """Shared expert parameters (not sharded by EP)."""
    if model.num_experts <= 0:
        return 0
    n_moe = sum(1 for lk in model.layers if lk.value == "moe")
    per_expert = 2 * model.hidden * model.moe_ffn
    n_shared = getattr(model, "n_shared_experts", 1) or 1
    return n_moe * n_shared * per_expert


def _routed_expert_params_on_rank(model: ModelSpec, strategy: Strategy) -> int:
    """Routed expert parameters on one rank after TP + EP + PP sharding."""
    if model.num_experts <= 0:
        return 0
    n_moe = sum(1 for lk in model.layers if lk.value == "moe")
    per_expert = 2 * model.hidden * model.moe_ffn
    routed_total = n_moe * model.num_experts * per_expert

    if strategy.tp > 1:
        routed_total //= strategy.tp

    if strategy.ep > 1:
        routed_total //= strategy.ep

    if strategy.pp > 1:
        n_layers = len(model.layers)
        routed_total = int(routed_total * (n_layers / strategy.pp) / n_layers)

    return routed_total


# Also need to store n_shared_experts on ModelSpec for the config loader
# (already supported via `getattr` default of 1)


def _optimizer_state_bytes(P: int, model: ModelSpec, strategy: Strategy) -> int:
    """Optimizer state memory in bytes for P parameters.

    Adam: P × 12B (master + m + v, each 4B)
    Muon: P × (12 - f_muon × 4)B = P_muon × 8B + P_adam × 12B
    """
    if strategy.optimizer.value == "adam":
        return P * 12
    elif strategy.optimizer.value == "muon":
        muon_config = strategy.muon_config
        f_muon = (
            muon_config.muon_param_fraction
            if muon_config and muon_config.muon_param_fraction is not None
            else 0.85
        )
        return int(P * (12 - f_muon * 4))
    return P * 12


def _activation_memory(
    model: ModelSpec, strategy: Strategy, layer_ids: list[int],
) -> tuple[int, int]:
    """Activation memory per rank using Korthikanti-style estimation.

    Per layer: seq * hidden * dtype_bytes * coefficient(layer_kind)
    Coefficient accounts for number of activation tensors held simultaneously.

    TP with SP (Sequence Parallel): shards activations by TP along sequence dim.
    CP: further shards activations by CP along sequence dim.
    Combined: total_shard = max(tp_sp, 1) * max(cp, 1)

    Note: TP SP only activates when TP>1 (Megatron-style SP).
    CP activates independently and stacks on top of SP.

    Activation checkpointing (recompute) reduces saved activations:
    - "attn": don't save attention intermediate activations (Q, K, V, scores)
    - "ffn_swiglu": don't save FFN intermediate activations (up, gate outputs)
    - "full": only save layer input/output, recompute everything inside
    """
    s = model.seq_len
    h = model.hidden
    act_bytes = model.act_dtype.bytes
    hc_mult = max(1, getattr(model, "hc_mult", 1))

    # Base coefficients: number of activation tensors saved per layer
    # These assume NO activation checkpointing (save all intermediates)
    # Dense: attn(5) + ffn_swiglu(3) + ln(2) ≈ 10
    # MoE: attn(5) + shared_ffn(3) + routed_ffn(4) + ln(2) ≈ 14
    # MTP: similar to dense with extra projection ≈ 12
    COEFF_DENSE_BASE = 10
    COEFF_MOE_BASE = 14
    COEFF_MTP_BASE = 12
    COEFF_HC_RESIDUAL = 2

    # Reduction when checkpointing specific components
    # attn: saves Q, K, V (3 tensors), attention scores (1), softmax output (1) ≈ 5
    # But Flash Attention already doesn't save scores, so effective saving is ~4
    # We use conservative estimate: -4
    REDUCE_ATTN = 4
    # ffn_swiglu: saves up output, gate output (before down_proj) ≈ 2
    REDUCE_FFN_SWIGLU = 2
    # ln: saves ln output (1) ≈ 1
    REDUCE_LN = 1
    # full checkpoint: only save layer input (1) + output (1) ≈ 2
    COEFF_FULL_CHECKPOINT = 2

    tp_sp = strategy.tp if strategy.tp > 1 else 1
    cp = strategy.cp if strategy.cp > 1 else 1
    total_seq_shard = tp_sp * cp

    # Get recompute policy
    recompute_policy = strategy.recompute.per_layer

    total_act = 0
    total_hc = 0
    for lid in layer_ids:
        if lid >= len(model.layers):
            continue
        lk = model.layers[lid]

        # Base coefficient by layer kind
        if lk.value == "dense":
            base_coeff = COEFF_DENSE_BASE
        elif lk.value == "moe":
            base_coeff = COEFF_MOE_BASE
        elif lk.value == "mtp":
            base_coeff = COEFF_MTP_BASE
        else:
            base_coeff = COEFF_DENSE_BASE

        # Apply recompute policy reduction
        cats_to_recompute = recompute_policy.get(lk.value, set())
        attn_cats = {"attn", "attn_core", "attn_block"}
        attn_recomputed = bool(cats_to_recompute & (attn_cats | {"full"}))
        # mhc recompute: drop the HC residual stream activations. Without this
        # branch, "hc" added recompute time in _recompute_time but freed no
        # memory, so mhc was always strictly worse than none.
        hc_recomputed = bool(cats_to_recompute & {"hc", "full"})

        if "full" in cats_to_recompute:
            # Full checkpoint: only save input + output
            coeff = COEFF_FULL_CHECKPOINT
        else:
            coeff = base_coeff
            if cats_to_recompute & attn_cats:
                coeff -= REDUCE_ATTN
            if "ffn_swiglu" in cats_to_recompute:
                coeff -= REDUCE_FFN_SWIGLU
            if "ln" in cats_to_recompute:
                coeff -= REDUCE_LN

        layer_act = s * h * act_bytes * coeff
        if hc_recomputed:
            hc_layer = 0
        else:
            hc_layer = (hc_mult - 1) * s * h * act_bytes * COEFF_HC_RESIDUAL
        layer_act += hc_layer

        # Attention-scores term: 5·heads·s·s_kv·bytes per layer, where
        # s_kv is the effective KV length seen by softmax.
        # DeepSeek-V4 layer-wise rules:
        #   compress_ratios[lid] == 0    → SWA-only, s_kv = swa_window
        #   compress_ratios[lid] > 1     → KV pool compressed to s/ratio
        #     CSA layers (ratio ≤ 4, indexer enabled) cap s_kv by index_topk
        #     (the indexer selects topk tokens per query, intermediate full
        #     scores are streamed via FlashAttention and not materialized).
        #   otherwise (dense / no compress_ratios) → full s × s.
        # Eliminated when attention is recomputed (selective attn recompute).
        if not attn_recomputed:
            num_heads = max(1, getattr(model, "num_heads", 1))
            attn_bytes = model.effective_attn_act_dtype().bytes
            compress_ratios = getattr(model, "compress_ratios", None) or []
            ratio = compress_ratios[lid] if lid < len(compress_ratios) else 1
            if ratio == 0:
                window = getattr(model, "swa_window", 0) or s
                s_kv = min(int(window), s)
            elif ratio > 1:
                s_kv = max(1, s // int(ratio))
                if ratio <= 4:
                    topk = getattr(model, "index_topk", 0) or 0
                    if topk > 0:
                        s_kv = min(s_kv, int(topk))
            else:
                s_kv = s
            layer_act += 5 * num_heads * s * s_kv * attn_bytes

        layer_act = layer_act // total_seq_shard
        hc_layer = hc_layer // total_seq_shard

        total_act += layer_act
        total_hc += hc_layer

    total_act *= strategy.micro_batch
    total_hc *= strategy.micro_batch

    if strategy.pp > 1:
        in_flight = _pp_in_flight(strategy)
        total_act *= in_flight
        total_hc *= in_flight

    return total_act, total_hc


def _pp_in_flight(strategy: Strategy) -> int:
    """Worst-rank in-flight microbatch count for the configured PP schedule.

    Activations of in-flight microbatches all coexist on the worst-loaded
    rank; sizing for that rank is what determines OOM. The previous code
    used a fixed ``pp // 2`` (1F1B mid-rank approximation), which under-
    counts most schedules.

    - 1F1B / ZeroBubble: worst rank (rank 0) holds ~``pp`` microbatches at
      the end of warmup.
    - Interleaved (VPP, ``vpp_chunks = v``): ~``pp · (v + 1) / 2``.
    - DualPipe / DualPipeV: two-direction chunks per rank, peak ≈ ``pp``.
    """
    from zrt.training.spec.strategy import PPSched

    pp = max(1, strategy.pp)
    if pp == 1:
        return 1
    sched = getattr(strategy, "pp_schedule", PPSched.ONE_F_ONE_B)

    if sched == PPSched.ONE_F_ONE_B:
        return pp
    if sched == PPSched.INTERLEAVED:
        vpp = max(1, getattr(strategy, "vpp_chunks", 1))
        return max(1, (pp * (vpp + 1)) // 2)
    if sched == PPSched.ZERO_BUBBLE:
        return pp
    if sched in (PPSched.DUALPIPE, PPSched.DUALPIPE_V):
        return pp
    # Unknown schedule: fall back to the (lossy) historical default.
    return max(1, pp // 2)


def _comm_buffer_memory(model: ModelSpec, strategy: Strategy) -> int:
    """Communication buffer memory (AG/RS + CP A2A + EP A2A buffers)."""
    s = model.seq_len
    h = model.hidden
    act_bytes = model.act_dtype.bytes
    n_layers = len(model.layers)
    
    if strategy.pp > 1:
        layers_per_stage = n_layers // strategy.pp
        n_layers = layers_per_stage

    total = 0

    # TP AG/RS buffers (only when TP>1)
    if strategy.tp > 1:
        h_tp = h // strategy.tp
        per_layer_tp = 4 * s * h_tp * act_bytes
        total += per_layer_tp * n_layers * strategy.micro_batch

    # CP A2A buffers (only when CP>1)
    # Note: Ulysses CP has 4 A2A per layer (fwd_before, fwd_after, bwd_before, bwd_after)
    # Buffer memory assumes no reuse between forward and backward (保守估算)
    if strategy.cp > 1:
        seq_cp = s // strategy.cp
        h_tp = h // strategy.tp if strategy.tp > 1 else h
        # CP A2A buffers shuttle attention activations across the sequence dim.
        cp_bytes = model.effective_attn_act_dtype().bytes
        per_layer_cp = 4 * seq_cp * h_tp * cp_bytes
        total += per_layer_cp * n_layers * strategy.micro_batch

    # EP A2A buffers (only when EP>1, MoE layers only)
    # Note: EP has 4 A2A per MoE layer (fwd_before, fwd_after, bwd_before, bwd_after)
    # Buffer memory assumes no reuse between forward and backward (保守估算)
    if strategy.ep > 1 and model.num_experts > 0:
        seq_cp = s // strategy.cp if strategy.cp > 1 else s
        h_tp = h // strategy.tp if strategy.tp > 1 else h
        # EP dispatch/combine carries routed-expert activations.
        ep_bytes = model.routed_expert_compute_dtype.bytes
        per_layer_ep = 4 * seq_cp * h_tp * ep_bytes
        n_moe = sum(1 for lk in model.layers if lk.value == "moe")
        if strategy.pp > 1:
            n_moe = max(1, n_moe // strategy.pp)
        total += per_layer_ep * n_moe * strategy.micro_batch

    return total
