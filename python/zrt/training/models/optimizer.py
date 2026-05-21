"""Optimizer FLOPs and communication modeling.

Reference: Muon optimizer (Kumar et al. 2025), ZeRO paper.
"""
from __future__ import annotations


def ns_flops(m: int, n: int, K: int) -> int:
    """FLOPs for K-step Newton-Schulz orthogonalization on (m, n) matrix.

    Each NS iteration computes a degree-4 polynomial:
      X' = X @ (a0*I + a1*X.T@X + a2*(X.T@X)^2)
    which decomposes into 3 GEMMs per iteration.

    FLOPs per iteration: 6 × max(m,n) × min(m,n)²

    For the dual-stage Muon optimizer (DeepSeek-V4 §2.4):
      Stage 1: 8 iterations with degree-4 polynomial → 6×m×n² per step
      Stage 2: 2 iterations with degree-2 polynomial → 4×m×n² per step

    This function uses 6× as the default per-step coefficient.
    For stage 2, use ns_flops_stage2() which uses 4×.

    Args:
        m: First dimension of matrix (rows)
        n: Second dimension of matrix (columns)
        K: Number of Newton-Schulz iterations

    Returns:
        Total FLOPs for K iterations
    """
    max_dim = max(m, n)
    min_dim = min(m, n)
    return K * 6 * max_dim * min_dim * min_dim


def ns_flops_stage2(m: int, n: int, K: int) -> int:
    """FLOPs for stage-2 NS iterations (degree-2 polynomial, 2 GEMMs per step).

    Stage 2 uses: X' = X @ (a0*I + a1*X.T@X) → 4×m×n² per step.
    """
    max_dim = max(m, n)
    min_dim = min(m, n)
    return K * 4 * max_dim * min_dim * min_dim


def adam_step_flops(P: int) -> int:
    """FLOPs for Adam optimizer step on P parameters.

    Adam update per parameter:
      1. m = β₁ × m + (1 - β₁) × g           → 2 FLOPs
      2. v = β₂ × v + (1 - β₂) × g²          → 3 FLOPs
      3. m̂ = m / (1 - β₁ᵗ)                   → 1 FLOP
      4. v̂ = v / (1 - β₂ᵗ)                   → 1 FLOP
      5. update = lr × m̂ / (√v̂ + ε)         → 3 FLOPs
      6. w = w - update                       → 1 FLOP
      Total: ~12 FLOPs per parameter

    Args:
        P: Number of parameters

    Returns:
        Total FLOPs for Adam optimizer step
    """
    return P * 12


def muon_step_flops(P: int, K: int, hidden: int) -> int:
    """FLOPs for Muon optimizer step on P parameters.

    Dual-stage Newton-Schulz (DeepSeek-V4 §2.4):
      Stage 1: 8 iterations with degree-4 polynomial (6×m×n² per step)
      Stage 2: 2 iterations with degree-2 polynomial (4×m×n² per step)

    For P params distributed across roughly square matrices of size hidden×hidden:
      - Number of matrices ≈ P / (hidden × hidden)
      - NS FLOPs per matrix = stage1_flops + stage2_flops

    Args:
        P: Number of parameters
        K: Total Newton-Schulz iterations (split as 80% stage-1, 20% stage-2)
        hidden: Model hidden dimension (for NS matrix sizing)

    Returns:
        Total FLOPs for Muon optimizer step
    """
    if hidden <= 0:
        return P * 4

    hidden_sq = hidden * hidden
    num_matrices = max(1, P // hidden_sq) if P >= hidden_sq else 1

    # Split K into stage 1 (80%) and stage 2 (20%)
    k1 = max(1, int(K * 0.8))
    k2 = max(1, K - k1)

    ns_per_matrix = ns_flops(hidden, hidden, k1) + ns_flops_stage2(hidden, hidden, k2)
    ns_total = ns_per_matrix * num_matrices
    other_flops = P * 4
    return ns_total + other_flops


def muon_optimizer_step_flops(
    P: int,
    K: int,
    hidden: int,
    muon_fraction: float = 0.85,
) -> int:
    """FLOPs for mixed Muon+Adam optimizer step.

    Muon is applied to a fraction of parameters (e.g., attention, FFN weights),
    while Adam is used for the remainder (embeddings, biases, router).

    Args:
        P: Total number of parameters
        K: Newton-Schulz iterations for Muon
        hidden: Model hidden dimension
        muon_fraction: Fraction of params using Muon (default 0.85)

    Returns:
        Total FLOPs for optimizer step
    """
    P_muon = int(P * muon_fraction)
    P_adam = P - P_muon

    muon_flops = muon_step_flops(P_muon, K, hidden)
    adam_flops = adam_step_flops(P_adam)

    return muon_flops + adam_flops


def _ns_pair_flops(m: int, n: int, K: int) -> int:
    """Per-matrix NS budget: 80% stage-1 (6·max·min²) + 20% stage-2 (4·max·min²)."""
    k1 = max(1, int(K * 0.8))
    k2 = max(1, K - k1)
    return ns_flops(m, n, k1) + ns_flops_stage2(m, n, k2)


def muon_step_flops_from_arch(
    model,
    strategy,
    K: int,
    muon_fraction: float = 0.85,
) -> int:
    """Architecture-driven per-rank Muon NS FLOPs.

    Walks the explicit weight-matrix inventory instead of inferring count
    from per-rank P. The legacy ``num_matrices = P // hidden²`` path clamps
    to 1 once ZeRO + EP shrink P below hidden², which makes optimizer time
    a constant across the entire search grid.

    Sharding model (matches the rest of the simulator):
      * Routed experts are placed across the EP dimension; the remaining
        DP/EP replica factor (``dp // ep``) further splits per-expert NS
        work after requiring a regular expert-DP layout.
      * Attention + shared experts + dense FFN replicate across EP and are
        ZeRO-sharded across the **full** DP group.
      * All weight matrices are TP-column-sharded (NS still operates on the
        TP-Gathered row dim = ``hidden``, col dim = sharded).
      * PP shards layers across the ``pp`` stages.
    """
    h = model.hidden
    tp = max(1, strategy.tp)
    ep = max(1, strategy.ep)
    pp = max(1, strategy.pp)
    dp = max(1, strategy.dp)
    if ep > 1:
        if dp < ep:
            raise ValueError(
                f"dp must be >= ep for expert-DP sharding (dp={dp}, ep={ep})"
            )
        if dp % ep != 0:
            raise ValueError(
                f"dp must be divisible by ep for expert-DP sharding "
                f"(dp={dp}, ep={ep})"
            )

    n_moe = sum(1 for l in model.layers if l.value == "moe")
    n_dense = sum(1 for l in model.layers if l.value == "dense")
    n_mtp = sum(1 for l in model.layers if l.value == "mtp")

    moe_ffn_col = max(1, model.moe_ffn // tp) if model.moe_ffn else 0
    ffn_col = max(1, model.ffn // tp) if model.ffn else 0
    attn_col = max(1, h // tp)

    ns_attn = _ns_pair_flops(h, attn_col, K)
    ns_moe_ffn = _ns_pair_flops(h, moe_ffn_col, K) if moe_ffn_col else 0
    ns_dense_ffn = _ns_pair_flops(h, ffn_col, K) if ffn_col else 0

    # ── Cluster-wide per-layer NS FLOPs ───────────────────────────────
    n_shared = max(1, getattr(model, "n_shared_experts", 1) or 1)

    # Attention: ~5 NS-eligible weight matrices per layer (V3 MLA / V4
    # grouped-MQA both fit this envelope: q_a / q_b / k_a-or-wkv / wo_a / wo_b).
    attn_layer_full = 5 * ns_attn

    routed_layer_full = 3 * model.num_experts * ns_moe_ffn if ns_moe_ffn else 0
    shared_layer_full = 3 * n_shared * ns_moe_ffn if ns_moe_ffn else 0
    dense_ffn_layer_full = 3 * ns_dense_ffn

    # ── Per-rank split ────────────────────────────────────────────────
    # Routed expert work lands on ``ep`` distinct ranks; within an EP rank,
    # remaining DP replicas (``dp // ep``) further parallelize NS.
    ep_dp_replica = dp // ep if ep > 1 else dp
    routed_per_rank_per_layer = routed_layer_full // (ep * ep_dp_replica)

    # Non-routed weights replicate across EP and split across full DP.
    non_routed_dp_div = dp if strategy.zero_stage >= 1 and dp > 1 else 1
    attn_per_rank_per_layer = attn_layer_full // non_routed_dp_div
    shared_per_rank_per_layer = shared_layer_full // non_routed_dp_div
    dense_ffn_per_rank_per_layer = dense_ffn_layer_full // non_routed_dp_div

    per_moe_layer = (
        attn_per_rank_per_layer
        + routed_per_rank_per_layer
        + shared_per_rank_per_layer
    )
    per_dense_layer = attn_per_rank_per_layer + dense_ffn_per_rank_per_layer
    per_mtp_layer = per_moe_layer if n_moe > 0 else per_dense_layer

    total = (
        n_moe * per_moe_layer
        + n_dense * per_dense_layer
        + n_mtp * per_mtp_layer
    )

    # PP: each rank owns 1/pp layers.
    total //= pp

    return int(total * muon_fraction)
