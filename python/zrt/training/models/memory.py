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

    @property
    def total(self) -> float:
        return self.weights + self.grads + self.opt_state + self.activations + self.comm_buffers

    def to_gb(self) -> dict[str, float]:
        GB = 1024 ** 3
        return {
            "weights_gb": self.weights / GB,
            "grads_gb": self.grads / GB,
            "opt_state_gb": self.opt_state / GB,
            "activations_gb": self.activations / GB,
            "comm_buffers_gb": self.comm_buffers / GB,
            "hc_overhead_gb": self.hc_overhead_bytes / GB,
            "total_gb": self.total / GB,
        }


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

    # FP4 routed expert weights: 0.5 B/elem + per-block BF16 scale
    use_fp4 = getattr(model, "routed_expert_dtype", "bf16") == "fp4"
    if use_fp4:
        P_expert = _routed_expert_params_on_rank(model, strategy)
        P_other = P - P_expert
        FP4_BYTES_PER_ELEM = 0.5
        FP4_BLOCK_SIZE = 32
        expert_weight_bytes = int(
            P_expert * FP4_BYTES_PER_ELEM
            + (P_expert / FP4_BLOCK_SIZE) * 2  # BF16 scale per block
        )
        weights = expert_weight_bytes + P_other * model.param_dtype.bytes
    else:
        weights = P * model.param_dtype.bytes

    grads = P * model.grad_dtype.bytes
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
        layer_ids = list(range(len(model.layers)))

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

    return MemBreakdown(
        weights=weights,
        grads=grads,
        opt_state=opt_state,
        activations=activations,
        comm_buffers=comm_buffers,
        hc_overhead_bytes=hc_overhead,
    )


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
        if "full" in cats_to_recompute:
            # Full checkpoint: only save input + output
            coeff = COEFF_FULL_CHECKPOINT
        else:
            coeff = base_coeff
            if "attn" in cats_to_recompute:
                coeff -= REDUCE_ATTN
            if "ffn_swiglu" in cats_to_recompute:
                coeff -= REDUCE_FFN_SWIGLU
            if "ln" in cats_to_recompute:
                coeff -= REDUCE_LN

        layer_act = s * h * act_bytes * coeff
        hc_layer = (hc_mult - 1) * s * h * act_bytes * COEFF_HC_RESIDUAL
        layer_act += hc_layer

        layer_act = layer_act // total_seq_shard
        hc_layer = hc_layer // total_seq_shard

        total_act += layer_act
        total_hc += hc_layer

    total_act *= strategy.micro_batch
    total_hc *= strategy.micro_batch

    if strategy.pp > 1:
        in_flight = max(1, strategy.pp // 2)
        total_act *= in_flight
        total_hc *= in_flight

    return total_act, total_hc


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
        per_layer_cp = 4 * seq_cp * h_tp * act_bytes
        total += per_layer_cp * n_layers * strategy.micro_batch

    # EP A2A buffers (only when EP>1, MoE layers only)
    # Note: EP has 4 A2A per MoE layer (fwd_before, fwd_after, bwd_before, bwd_after)
    # Buffer memory assumes no reuse between forward and backward (保守估算)
    if strategy.ep > 1 and model.num_experts > 0:
        seq_cp = s // strategy.cp if strategy.cp > 1 else s
        h_tp = h // strategy.tp if strategy.tp > 1 else h
        per_layer_ep = 4 * seq_cp * h_tp * act_bytes
        n_moe = sum(1 for lk in model.layers if lk.value == "moe")
        if strategy.pp > 1:
            n_moe = max(1, n_moe // strategy.pp)
        total += per_layer_ep * n_moe * strategy.micro_batch

    return total
