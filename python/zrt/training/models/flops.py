"""Per-op analytical FLOPs model.

Returns raw cost per op. Recompute multiplier applied by the stage composer.
Reference: Calculon (Isaev et al. SC'23), Korthikanti et al. 2022.
"""

from __future__ import annotations

from dataclasses import dataclass

from zrt.training.ir.training_graph import Graph, Op
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.strategy import Strategy


@dataclass
class OpCost:
    fwd_flops: float = 0.0
    dx_flops: float = 0.0
    dw_flops: float = 0.0
    fwd_bytes: float = 0.0   # memory-bound ops: byte traffic
    dx_bytes: float = 0.0
    dw_bytes: float = 0.0
    bound: str = "compute"   # "compute" | "memory"


def op_cost(op: Op, model: ModelSpec) -> OpCost:
    """Compute raw cost per op. Bound determines the cost model used."""
    if op.kind == "matmul":
        return _matmul_cost(op)
    if op.kind == "attn_core":
        return _attn_cost(op, model)
    if op.kind == "mhc_pre":
        return _mhc_pre_cost(op)
    if op.kind == "mhc_post":
        return _mhc_post_cost(op)
    if op.kind == "mhc_head":
        return _mhc_head_cost(op)
    if op.kind == "hc_expand":
        return _memory_bound_cost(op)
    if op.kind in ("ln", "softmax", "rope", "swiglu", "add"):
        return _memory_bound_cost(op)
    if op.kind in ("embed", "lm_head"):
        return _matmul_cost(op)
    # Unknown ops: zero cost
    return OpCost()


def _matmul_cost(op: Op) -> OpCost:
    m = op.meta.get("m", 0)
    n = op.meta.get("n_local", op.meta.get("n", 0))
    k = op.meta.get("k_local", op.meta.get("k", 0))
    fwd = 2.0 * m * n * k

    # Apply fwd_multiplier if present (e.g., for MoE routed expert FFNs)
    fwd_multiplier = op.meta.get("fwd_multiplier", 1.0)
    fwd = fwd * fwd_multiplier

    return OpCost(
        fwd_flops=fwd,
        dx_flops=fwd,     # dX: 2*m*n*k
        dw_flops=fwd,     # dW: 2*m*n*k
    )


def _attn_cost(op: Op, model: ModelSpec) -> OpCost:
    b = op.meta.get("b", 1)
    s = op.meta.get("s", 0)
    h = op.meta.get("heads", 0)
    d = op.meta.get("head_dim", 0)
    causal = op.meta.get("causal", True)

    # Note: shard.py has already modified s and heads metadata to reflect
    # TP/CP sharding. We directly use the modified values here:
    # - Ulysses CP: s -> s/cp, heads -> heads_tp * cp
    #   FLOPs = 2 × b × (s/cp) × (s/cp) × (heads_tp × cp) × d
    #         = 2 × b × s × s × heads_tp × d / cp
    # - Ring CP: s -> s/cp, heads unchanged, cp_tiles = cp
    #   FLOPs = cp × [2 × b × (s/cp) × (s/cp) × heads_tp × d]
    #         = 2 × b × s × s × heads_tp × d / cp
    #   Note: Ring has cp rounds of attention, each round uses (s/cp) query length,
    #   but accesses full s KV length (handled by P2P通信).
    #   The meta["s"] is切分后的值, so we need to multiply by cp_tiles to get total FLOPs.
    # - TP: heads -> heads/tp
    # The FLOPs formula automatically accounts for sharding through these modified values.

    compression_ratio = _attn_compression_ratio(
        op.meta.get("attn_compression_ratio", model.attn_compression_ratio)
    )

    mult = 2.0 if causal else 4.0
    fwd = mult * b * s * s * h * d * compression_ratio

    # Ring-CP: multiply by cp_tiles to account for multiple rounds
    # Ulysses-CP: heads已经乘cp，所以不需要额外因子
    if op.meta.get("cp_tiles", 0) > 1:
        fwd *= op.meta.get("cp_tiles", 1)

    dx = 2.5 * fwd

    return OpCost(
        fwd_flops=fwd,
        dx_flops=dx,
        dw_flops=0.0,
    )


def _attn_compression_ratio(value: float) -> float:
    ratio = float(value)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"attn_compression_ratio must be in (0, 1], got {value}")
    return ratio


def _mhc_pre_cost(op: Op) -> OpCost:
    """Hyper-Connections pre-mix: mixes-Linear + sinkhorn iters + weighted sum.

    Math:
      mixes  = x[b,s,hc*h] @ hc_fn[hc*h, mix_hc]      → 2·b·s·(hc·h)·mix_hc  (compute-bound)
      sink   = sinkhorn_iters · O(b·s·mix_hc·hc)     elementwise              (memory-ish)
      sum    = pre[b,s,hc] · x[b,s,hc,h]              → 2·b·s·hc·h            (weighted sum)
    """
    b = op.meta.get("b", 1)
    s = op.meta.get("s", 0)
    h = op.meta.get("h", 0)
    hc = op.meta.get("hc", 1)
    mix = op.meta.get("mix_hc", (2 + hc) * hc)
    it = op.meta.get("sinkhorn_iters", 20)

    fwd_lin = 2.0 * b * s * (hc * h) * mix
    fwd_sink = float(it * b * s * mix * hc) * 4.0
    fwd_sum = float(b * s * hc * h) * 2.0
    fwd = fwd_lin + fwd_sink + fwd_sum

    return OpCost(
        fwd_flops=fwd,
        dx_flops=2.5 * fwd,
        dw_flops=fwd_lin,  # only the mixes Linear has trainable params
    )


def _mhc_post_cost(op: Op) -> OpCost:
    """Hyper-Connections post-mix: post·x + Σ comb·residual.

    Math:
      post · x:           b·s·hc·h            (broadcast multiply)
      comb · residual:    b·s·hc·hc·h         (full hc×hc combination)
      sum over hc:        b·s·hc·h
    No trainable parameters in this op (post / comb come from mhc_pre).
    """
    b = op.meta.get("b", 1)
    s = op.meta.get("s", 0)
    h = op.meta.get("h", 0)
    hc = op.meta.get("hc", 1)

    fwd = float(b * s * hc * h) * 2.0 + float(b * s * hc * hc * h) * 2.0

    return OpCost(
        fwd_flops=fwd,
        dx_flops=2.5 * fwd,
        dw_flops=0.0,
    )


def _mhc_head_cost(op: Op) -> OpCost:
    """Final HC mix-down before final_ln (no sinkhorn, no comb).

    Math:
      mixes  = x[b,s,hc*h] @ hc_head_fn[hc*h, hc]   → 2·b·s·hc·h·hc
      sum    = pre[b,s,hc] · x[b,s,hc,h]            → 2·b·s·hc·h
    """
    b = op.meta.get("b", 1)
    s = op.meta.get("s", 0)
    h = op.meta.get("h", 0)
    hc = op.meta.get("hc", 1)
    mix = op.meta.get("mix_hc", hc)

    fwd_lin = 2.0 * b * s * (hc * h) * mix
    fwd_sum = float(b * s * hc * h) * 2.0
    fwd = fwd_lin + fwd_sum

    return OpCost(
        fwd_flops=fwd,
        dx_flops=2.5 * fwd,
        dw_flops=fwd_lin,
    )


def _memory_bound_cost(op: Op) -> OpCost:
    bytes_fwd = op.meta.get("bytes_fwd", 0.0)
    # Bwd byte traffic ≈ fwd (read activations + write gradients)
    bytes_bwd = bytes_fwd * 1.5  # conservative: read input + write grad

    return OpCost(
        fwd_bytes=bytes_fwd,
        dx_bytes=bytes_bwd,
        dw_bytes=0.0,
        bound="memory",
    )


def total_training_flops(
    graph: Graph, model: ModelSpec, strategy: Strategy,
) -> float:
    """Total FLOPs per training step (forward + backward).

    Standard transformer: 6 * total_params * tokens (6P rule).
    With recompute: adds extra forward for recomputed ops.
    """
    total = 0.0
    for op in graph.ops:
        cost = op_cost(op, model)
        if cost.bound == "compute":
            # Forward + dx + dw = 3× fwd_flops (2mnk × 3 = 6mnk for matmul)
            total += cost.fwd_flops + cost.dx_flops + cost.dw_flops
        # Memory-bound ops contribute negligible FLOPs

    # Scale by microbatch count
    M = strategy.num_microbatches()
    total *= M

    return total


def recompute_overhead_flops(
    graph: Graph, model: ModelSpec, strategy: Strategy,
) -> float:
    """Extra FLOPs from recomputing forward activations during backward pass.

    Selective recompute re-runs the forward for specific ops (typically attention)
    during backward. Full recompute re-runs the entire forward pass.

    Respects per-layer policies: only ops belonging to a layer whose kind
    appears in ``RecomputePolicy.per_layer`` are counted.

    Returns the additional FLOPs (not the total).
    """
    rc = strategy.recompute
    if not rc.per_layer:
        return 0.0

    extra = 0.0
    for op in graph.ops:
        # Look up the layer kind for this op
        if op.layer_id < 0 or op.layer_id >= len(model.layers):
            continue
        lk = model.layers[op.layer_id].value
        cats = rc.per_layer.get(lk)
        if not cats:
            continue

        op_cats = _op_recompute_categories(op)
        if "full" in cats or (op_cats & cats):
            cost = op_cost(op, model)
            if cost.bound == "compute":
                extra += cost.fwd_flops

    M = strategy.num_microbatches()
    return extra * M


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
    if op.kind in ("mhc_pre", "mhc_post", "mhc_head"):
        return {"hc"}
    return set()
