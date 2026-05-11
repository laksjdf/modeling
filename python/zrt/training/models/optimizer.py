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