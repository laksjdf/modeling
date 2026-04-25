"""Per-op analytical FLOPs model.

Returns raw cost per op. Recompute multiplier applied by the stage composer.
Reference: Calculon (Isaev et al. SC'23), Korthikanti et al. 2022.
"""

from __future__ import annotations

from dataclasses import dataclass

from zrt.training.ir.graph import Graph, Op
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

    # Sharded heads
    tp_factor = 1
    if model.num_heads > h > 0:
        tp_factor = model.num_heads // h

    compression_ratio = _attn_compression_ratio(
        op.meta.get("attn_compression_ratio", model.attn_compression_ratio)
    )

    # Fwd: flash-attn ≈ 2*b*s^2*h*d for causal (halved due to causal mask)
    # Non-causal: 4*b*s^2*h*d
    mult = 2.0 if causal else 4.0
    fwd = mult * b * s * s * h * d * compression_ratio

    # Bwd derives from compressed fwd, so dx inherits the same CSA/HCA ratio.
    dx = 2.5 * fwd

    return OpCost(
        fwd_flops=fwd,
        dx_flops=dx,
        dw_flops=0.0,  # Attention has no learnable parameters
    )


def _attn_compression_ratio(value: float) -> float:
    ratio = float(value)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"attn_compression_ratio must be in (0, 1], got {value}")
    return ratio


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
    return set()
