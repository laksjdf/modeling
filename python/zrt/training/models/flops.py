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
    fwd_bytes: float = 0.0
    dx_bytes: float = 0.0
    dw_bytes: float = 0.0
    # Legacy: used ONLY for FLOP accounting guards in total_training_flops/recompute_overhead_flops.
    # Timing model ALWAYS uses max(compute_time, memory_time).
    bound: str = "compute"   # "compute" | "memory"

    # Heterogeneous core FLOPs (Cube/Vector). For attn_core, cube+vector > fwd_flops
    # (vector portion is extra work not in the standard 6P accounting model).
    fwd_cube_flops: float = 0.0
    fwd_vector_flops: float = 0.0
    dx_cube_flops: float = 0.0
    dx_vector_flops: float = 0.0
    dw_cube_flops: float = 0.0
    dw_vector_flops: float = 0.0


def _bpe(op: Op) -> int:
    """Bytes per element from op's first tensor, defaulting to BF16=2."""
    if op.inputs:
        return op.inputs[0].dtype.bytes
    if op.outputs:
        return op.outputs[0].dtype.bytes
    return 2


def _bytes_from_tensors(op: Op) -> float:
    """Total bytes from all input + output tensors.

    Correct for memory-bound ops where Op.inputs/outputs capture all data movement.
    NOT suitable for matmul/attn (weight matrices are absent from the training IR).
    """
    return float(sum(t.nbytes() for t in op.inputs) + sum(t.nbytes() for t in op.outputs))


def op_cost(op: Op, model: ModelSpec) -> OpCost:
    """Compute raw cost per op (FLOPs + bytes for roofline timing)."""
    if op.kind == "matmul":
        return _matmul_cost(op)
    if op.kind == "sparse_attn":
        return _sparse_attn_cost(op)
    if op.kind == "hca_attn":
        return _hca_attn_cost(op)
    if op.kind == "swa_attn":
        return _swa_attn_cost(op)
    if op.kind == "attn_core":
        return _attn_cost(op, model)
    if op.kind == "mhc_pre":
        return _mhc_pre_cost(op)
    if op.kind == "mhc_post":
        return _mhc_post_cost(op)
    if op.kind == "mhc_head":
        return _mhc_head_cost(op)
    if op.kind in ("ln", "rmsnorm", "softmax", "rope", "swiglu", "add", "hc_expand"):
        return _elementwise_cost(op)
    if op.kind == "embed":
        return _embed_cost(op)
    if op.kind == "lm_head":
        return _matmul_cost(op)
    if op.kind == "compressor_pool":
        return _compressor_pool_cost(op)
    if op.kind == "indexer_topk":
        return _indexer_topk_cost(op)
    if op.kind == "hash_route":
        return OpCost()  # table lookup, negligible FLOPs
    # Unknown ops: zero cost
    return OpCost()


def _embed_cost(op: Op) -> OpCost:
    """Embedding lookup: gather operation, memory-bound with zero FLOPs.

    Embedding lookup reads rows from an embedding table based on indices.
    This is a gather operation, not a matmul, so FLOPs = 0.

    Forward: read embedding table rows -> write embeddings
    Backward: read embeddings -> update embedding table rows (gradient scatter)
    """
    # Get dimensions from op meta
    seq = op.meta.get("m", 0)  # sequence length
    hidden = op.meta.get("n", 0)  # embedding dimension

    # Forward bytes: read seq * hidden elements from embedding table
    # Backward bytes: same volume (gradient scatter)
    bpe = _bpe(op)
    fwd_bytes = seq * hidden * bpe

    return OpCost(
        fwd_flops=0.0,  # gather has no FLOPs
        dx_flops=0.0,   # scatter gradient has no FLOPs
        dw_flops=0.0,   # embedding table update is scatter, no FLOPs
        fwd_bytes=fwd_bytes,
        dx_bytes=fwd_bytes,
        dw_bytes=fwd_bytes,
        bound="memory",
    )


def _matmul_cost(op: Op) -> OpCost:
    m = op.meta.get("m", 0)
    n = op.meta.get("n_local", op.meta.get("n", 0))
    k = op.meta.get("k_local", op.meta.get("k", 0))
    bpe = _bpe(op)
    fwd = 2.0 * m * n * k * op.meta.get("fwd_multiplier", 1.0)
    # read A(m×k) + B(k×n), write C(m×n) — symmetric for fwd/dx/dw
    total_bytes = (m * k + k * n + m * n) * bpe
    return OpCost(
        fwd_flops=fwd,
        dx_flops=fwd,
        dw_flops=fwd,
        fwd_bytes=total_bytes,
        dx_bytes=total_bytes,
        dw_bytes=total_bytes,
        fwd_cube_flops=fwd,
        dx_cube_flops=fwd,
        dw_cube_flops=fwd,
    )


def _attn_cost(op: Op, model: ModelSpec) -> OpCost:
    b = op.meta.get("b", 1)
    s = op.meta.get("s", 0)
    h = op.meta.get("heads", 0)
    d = op.meta.get("head_dim", 0)
    causal = op.meta.get("causal", True)

    # V4 attention variants — identified by metadata:
    sparse_topk = op.meta.get("sparse_topk", 0)
    compress_ratio = op.meta.get("compress_ratio", 0)
    swa_window = op.meta.get("swa_window", 0)

    if sparse_topk > 0:
        # CSA: sparse attention over topk compressed KV + sliding window
        effective_len = sparse_topk + swa_window
        fwd = 2.0 * b * s * effective_len * h * d
        cube_fwd = fwd
        vector_fwd = 5.0 * b * h * s * effective_len
    elif compress_ratio > 0:
        # HCA: dense attention on compressed KV (seq/ratio) + sliding window
        compressed_len = max(1, s // compress_ratio)
        effective_len = compressed_len + swa_window
        fwd = 2.0 * b * s * effective_len * h * d
        cube_fwd = fwd
        vector_fwd = 5.0 * b * h * s * effective_len
    elif swa_window > 0:
        # SWA-only: pure sliding window attention
        fwd = 2.0 * b * s * swa_window * h * d
        cube_fwd = fwd
        vector_fwd = 5.0 * b * h * s * swa_window
    else:
        # Standard / MLA: full causal attention (possibly with compression ratio)
        compression_ratio = _attn_compression_ratio(
            op.meta.get("attn_compression_ratio", model.attn_compression_ratio)
        )
        mult = 2.0 if causal else 4.0
        fwd = mult * b * s * s * h * d * compression_ratio
        # Cube core always does full dense QK+AV matmuls (causal mask is Vector-side)
        cube_fwd = 4.0 * b * s * s * h * d * compression_ratio
        vector_fwd = 5.0 * b * h * s * s

    # Ring-CP: multiply by cp_tiles to account for multiple rounds
    if op.meta.get("cp_tiles", 0) > 1:
        cp_tiles = op.meta.get("cp_tiles", 1)
        fwd *= cp_tiles
        cube_fwd *= cp_tiles
        vector_fwd *= cp_tiles

    dx = 2.5 * fwd

    # KV length for byte estimation: sparse/compressed variants have shorter K, V
    if sparse_topk > 0:
        kv_len = sparse_topk + swa_window
    elif compress_ratio > 0:
        kv_len = max(1, s // compress_ratio) + swa_window
    elif swa_window > 0:
        kv_len = swa_window
    else:
        kv_len = s  # standard attention: K, V same length as Q

    bpe = _bpe(op)
    # Forward: read Q(s) + K(kv_len) + V(kv_len), write O(s)
    fwd_bytes = (2.0 * b * h * s * d + 2.0 * b * h * kv_len * d) * bpe
    # Backward: read Q(s) + K(kv_len) + V(kv_len) + dO(s),
    #           write dQ(s) + dK(kv_len) + dV(kv_len)
    # Flash attention recomputes O from QK^T, so O is not read from HBM
    dx_bytes = (3.0 * b * h * s * d + 4.0 * b * h * kv_len * d) * bpe

    return OpCost(
        fwd_flops=fwd,
        dx_flops=dx,
        dw_flops=0.0,
        fwd_bytes=fwd_bytes,
        dx_bytes=dx_bytes,
        dw_bytes=0.0,
        # Cube = dense QK+AV matmuls, Vector = softmax/scaling/mask
        fwd_cube_flops=cube_fwd,
        fwd_vector_flops=vector_fwd,
        dx_cube_flops=2.5 * cube_fwd,
        dx_vector_flops=2.5 * vector_fwd,
    )


def _attn_compression_ratio(value: float) -> float:
    ratio = float(value)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"attn_compression_ratio must be in (0, 1], got {value}")
    return ratio


def _sparse_attn_cost(op: Op) -> OpCost:
    """CSA / DSA: sparse attention over indexer top-k KV + sliding window.

    Q: (seq, heads, head_dim)
    K, V: (topk + swa_window, head_dim) per head  — MQA: KV shared across heads
    O: (seq, heads, head_dim)

    effective_len = topk + swa_window (the number of KV positions each Q attends to)
    """
    b = op.meta.get("b", 1)
    s = op.meta.get("s", 0)
    h = op.meta.get("heads", 0)
    d = op.meta.get("head_dim", 0)
    topk = op.meta.get("sparse_topk", 0)
    swa = op.meta.get("swa_window", 0)

    if topk <= 0:
        # Degenerate: no sparse_topk set — fall back to full causal
        return _attn_cost(op, None)  # model not needed for fallback

    effective_len = topk + swa
    fwd = 2.0 * b * s * effective_len * h * d
    dx = 2.5 * fwd
    # Cube = QK^T + AV matmuls over effective_len, Vector = softmax/scaling/mask
    cube_fwd = fwd
    vector_fwd = 5.0 * b * h * s * effective_len

    bpe = _bpe(op)
    fwd_bytes = (2.0 * b * h * s * d + 2.0 * b * h * effective_len * d) * bpe
    dx_bytes = (3.0 * b * h * s * d + 4.0 * b * h * effective_len * d) * bpe

    return OpCost(
        fwd_flops=fwd,
        dx_flops=dx,
        dw_flops=0.0,
        fwd_bytes=fwd_bytes,
        dx_bytes=dx_bytes,
        dw_bytes=0.0,
        fwd_cube_flops=cube_fwd,
        fwd_vector_flops=vector_fwd,
        dx_cube_flops=2.5 * cube_fwd,
        dx_vector_flops=2.5 * vector_fwd,
    )


def _hca_attn_cost(op: Op) -> OpCost:
    """HCA: dense attention on compressed KV (seq/ratio) + sliding window.

    Q: (seq, heads, head_dim)
    K, V: (seq // ratio + swa_window, head_dim) — compressed KV pool
    O: (seq, heads, head_dim)
    """
    b = op.meta.get("b", 1)
    s = op.meta.get("s", 0)
    h = op.meta.get("heads", 0)
    d = op.meta.get("head_dim", 0)
    ratio = op.meta.get("compress_ratio", 0)
    swa = op.meta.get("swa_window", 0)

    if ratio <= 0:
        # Degenerate: no compress_ratio — fall back to full causal
        return _attn_cost(op, None)

    compressed_len = max(1, s // ratio)
    effective_len = compressed_len + swa
    fwd = 2.0 * b * s * effective_len * h * d
    dx = 2.5 * fwd
    # Cube = QK^T + AV matmuls over effective_len, Vector = softmax/scaling/mask
    cube_fwd = fwd
    vector_fwd = 5.0 * b * h * s * effective_len

    bpe = _bpe(op)
    fwd_bytes = (2.0 * b * h * s * d + 2.0 * b * h * effective_len * d) * bpe
    dx_bytes = (3.0 * b * h * s * d + 4.0 * b * h * effective_len * d) * bpe

    return OpCost(
        fwd_flops=fwd,
        dx_flops=dx,
        dw_flops=0.0,
        fwd_bytes=fwd_bytes,
        dx_bytes=dx_bytes,
        dw_bytes=0.0,
        fwd_cube_flops=cube_fwd,
        fwd_vector_flops=vector_fwd,
        dx_cube_flops=2.5 * cube_fwd,
        dx_vector_flops=2.5 * vector_fwd,
    )


def _swa_attn_cost(op: Op) -> OpCost:
    """SWA-only: pure sliding window attention.

    Q: (seq, heads, head_dim)
    K, V: (swa_window, head_dim) — only local window
    O: (seq, heads, head_dim)
    """
    b = op.meta.get("b", 1)
    s = op.meta.get("s", 0)
    h = op.meta.get("heads", 0)
    d = op.meta.get("head_dim", 0)
    swa = op.meta.get("swa_window", 0)

    if swa <= 0:
        # Degenerate: no swa_window — fall back to full causal
        return _attn_cost(op, None)

    fwd = 2.0 * b * s * swa * h * d
    dx = 2.5 * fwd
    # Cube = QK^T + AV matmuls over swa_window, Vector = softmax/scaling/mask
    cube_fwd = fwd
    vector_fwd = 5.0 * b * h * s * swa

    bpe = _bpe(op)
    fwd_bytes = (2.0 * b * h * s * d + 2.0 * b * h * swa * d) * bpe
    dx_bytes = (3.0 * b * h * s * d + 4.0 * b * h * swa * d) * bpe

    return OpCost(
        fwd_flops=fwd,
        dx_flops=dx,
        dw_flops=0.0,
        fwd_bytes=fwd_bytes,
        dx_bytes=dx_bytes,
        dw_bytes=0.0,
        fwd_cube_flops=cube_fwd,
        fwd_vector_flops=vector_fwd,
        dx_cube_flops=2.5 * cube_fwd,
        dx_vector_flops=2.5 * vector_fwd,
    )


def _mhc_pre_cost(op: Op) -> OpCost:
    """Hyper-Connections pre-mix: mixes-Linear + sinkhorn iters + weighted sum."""
    b = op.meta.get("b", 1)
    s = op.meta.get("s", 0)
    h = op.meta.get("h", 0)
    hc = op.meta.get("hc", 1)
    mix = op.meta.get("mix_hc", (2 + hc) * hc)
    it = op.meta.get("sinkhorn_iters", 20)
    bpe = _bpe(op)

    fwd_lin = 2.0 * b * s * (hc * h) * mix
    fwd_sink = float(it * b * s * mix * hc) * 4.0
    fwd_sum = float(b * s * hc * h) * 2.0
    fwd = fwd_lin + fwd_sink + fwd_sum

    # Bytes: input(s, hc*h) + weights(hc*h, mix, read once) + output(s, mix)
    # plus sinkhorn intermediates (s, mix) and residual (s, hc*h)
    lin_in_bytes = b * s * (hc * h) * bpe
    lin_wt_bytes = (hc * h) * mix * bpe  # weights read once, NOT per-token
    lin_out_bytes = b * s * mix * bpe
    # Sinkhorn operates on (s, mix) — read/write per iteration
    sink_bytes = it * b * s * mix * bpe * 2
    # Residual sum: read (s, hc*h), write (s, hc*h)
    sum_bytes = b * s * hc * h * bpe * 2
    fwd_bytes = lin_in_bytes + lin_wt_bytes + lin_out_bytes + sink_bytes + sum_bytes

    return OpCost(
        fwd_flops=fwd,
        dx_flops=2.5 * fwd,
        dw_flops=fwd_lin,
        fwd_bytes=fwd_bytes,
        dx_bytes=fwd_bytes * 1.5,
        fwd_cube_flops=fwd,
        dx_cube_flops=2.5 * fwd,
        dw_cube_flops=fwd_lin,
    )


def _mhc_post_cost(op: Op) -> OpCost:
    """Hyper-Connections post-mix: post·x + Σ comb·residual."""
    b = op.meta.get("b", 1)
    s = op.meta.get("s", 0)
    h = op.meta.get("h", 0)
    hc = op.meta.get("hc", 1)
    bpe = _bpe(op)

    fwd = float(b * s * hc * h) * 2.0 + float(b * s * hc * hc * h) * 2.0
    fwd_bytes = float(b * s * hc * h * bpe * 2 + b * s * hc * hc * h * bpe * 2)

    return OpCost(
        fwd_flops=fwd,
        dx_flops=2.5 * fwd,
        dw_flops=0.0,
        fwd_bytes=fwd_bytes,
        dx_bytes=fwd_bytes * 1.5,
        fwd_cube_flops=fwd,
        dx_cube_flops=2.5 * fwd,
    )


def _mhc_head_cost(op: Op) -> OpCost:
    """Final HC mix-down before final_ln (no sinkhorn, no comb)."""
    b = op.meta.get("b", 1)
    s = op.meta.get("s", 0)
    h = op.meta.get("h", 0)
    hc = op.meta.get("hc", 1)
    mix = op.meta.get("mix_hc", hc)
    bpe = _bpe(op)

    fwd_lin = 2.0 * b * s * (hc * h) * mix
    fwd_sum = float(b * s * hc * h) * 2.0
    fwd = fwd_lin + fwd_sum
    fwd_bytes = float(2 * b * s * (hc * h) * mix * bpe + b * s * hc * h * bpe)

    return OpCost(
        fwd_flops=fwd,
        dx_flops=2.5 * fwd,
        dw_flops=fwd_lin,
        fwd_bytes=fwd_bytes,
        dx_bytes=fwd_bytes * 1.5,
        fwd_cube_flops=fwd,
        dx_cube_flops=2.5 * fwd,
        dw_cube_flops=fwd_lin,
    )


_ELEMENTWISE_FLOPS = {
    "ln":      7,   # mean + var + norm + scale + shift + 2 intermediate (LayerNorm)
    "rmsnorm": 5,   # rms + scale + shift + 2 intermediate (RMSNorm, no mean/var)
    "softmax": 4,   # max + sub + exp + sum + div
    "rope":    2,   # sin/cos multiply per pair
    "swiglu":  5,   # sigmoid(4) + multiply(1) — elementwise portion between matmuls
    "add":     1,   # elementwise residual add
}


def _elementwise_cost(op: Op) -> OpCost:
    """Memory-bound ops: trivial FLOPs + tensor-based byte count."""
    n = sum(t.num_elements() for t in op.outputs) if op.outputs else 0
    flops_per_elem = _ELEMENTWISE_FLOPS.get(op.kind, 1)
    fwd_flops = flops_per_elem * n
    fwd_bytes = _bytes_from_tensors(op) or op.meta.get("bytes_fwd", 0.0)
    return OpCost(
        fwd_flops=fwd_flops,
        dx_flops=fwd_flops * 2.5,
        dw_flops=0.0,
        fwd_bytes=fwd_bytes,
        dx_bytes=fwd_bytes * 1.5,
        dw_bytes=0.0,
        bound="memory",
        fwd_cube_flops=0.0,
        fwd_vector_flops=fwd_flops,
        dx_cube_flops=0.0,
        dx_vector_flops=fwd_flops * 2.5,
    )


def _compressor_pool_cost(op: Op) -> OpCost:
    """KV compressor gated pooling: softmax + weighted sum over compression windows."""
    s = op.meta.get("s", 0)
    m = op.meta.get("m", 4)
    coff = op.meta.get("coff", 1)
    d = op.meta.get("d_local", op.meta.get("d", 0))
    # softmax over coff*m elements × (s/m) groups, then weighted sum
    fwd_flops = 4.0 * (s // m) * coff * m * d
    bytes_fwd = op.meta.get("bytes_fwd", s * d * 4)  # read kv + write compressed
    return OpCost(fwd_flops=fwd_flops, dx_flops=fwd_flops,
                  fwd_bytes=bytes_fwd, dx_bytes=bytes_fwd * 1.5, bound="compute",
                  fwd_cube_flops=fwd_flops, dx_cube_flops=fwd_flops)


def _indexer_topk_cost(op: Op) -> OpCost:
    """Indexer scoring: einsum(q, kv) + ReLU + weighted sum + top-k."""
    s = op.meta.get("s", 0)
    ih = op.meta.get("ih_local", op.meta.get("ih", 0))
    id_ = op.meta.get("id", 0)
    topk = op.meta.get("topk", 0)
    # kv_len: full seq for V3.2, compressed seq//m for V4-CSA
    kv_len = op.meta.get("kv_len", s)
    einsum_flops = 2.0 * s * kv_len * ih * id_
    # ReLU + weighted sum + topk: memory-bound
    bytes_fwd = op.meta.get("bytes_fwd", s * ih * id_ * 4)
    return OpCost(
        fwd_flops=einsum_flops,
        dx_flops=2.0 * einsum_flops,
        dw_flops=0.0,
        fwd_bytes=bytes_fwd,
        bound="compute",
        fwd_cube_flops=einsum_flops,
        dx_cube_flops=2.0 * einsum_flops,
    )


def total_training_flops(
    graph: Graph, model: ModelSpec, strategy: Strategy,
) -> float:
    """Total FLOPs per training step (forward + backward).

    Only counts compute-bound ops (matmul, attention) to match the
    standard 6P convention used in MFU reporting. Memory-bound ops
    (rmsnorm, swiglu, rope, add, embed) are excluded because their
    execution time is dominated by HBM bandwidth, not compute throughput.

    Including memory-bound ops in the FLOP numerator while their time
    cost sits in the denominator (step_time) would artificially inflate MFU.
    """
    total = 0.0
    for op in graph.ops:
        cost = op_cost(op, model)
        if cost.bound == "compute":
            total += cost.fwd_flops + cost.dx_flops + cost.dw_flops

    # Scale by microbatch count
    M = strategy.num_microbatches()
    total *= M

    return total


def forward_backward_flops(
    graph: Graph, model: ModelSpec, strategy: Strategy,
) -> tuple[float, float]:
    """Return (forward_flops, backward_flops) separately.

    Same loop as total_training_flops but splits fwd from dx+dw.
    """
    fwd = 0.0
    bwd = 0.0
    for op in graph.ops:
        cost = op_cost(op, model)
        if cost.bound == "compute":
            fwd += cost.fwd_flops
            bwd += cost.dx_flops + cost.dw_flops
    M = strategy.num_microbatches()
    return fwd * M, bwd * M


def recompute_overhead_flops(
    graph: Graph, model: ModelSpec, strategy: Strategy,
) -> float:
    """Extra FLOPs from recomputing forward activations during backward pass.

    Selective recompute re-runs the forward for specific ops (typically attention)
    during backward. Full recompute re-runs the entire forward pass.

    Respects per-layer policies: only ops belonging to a layer whose kind
    appears in ``RecomputePolicy.per_layer`` are counted.

    Only counts compute-bound ops, consistent with total_training_flops.
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
    if op.kind in ("attn_core", "sparse_attn", "hca_attn", "swa_attn"):
        return {"attn"}
    if op.kind == "matmul":
        name = op.name.lower()
        if any(k in name for k in ("qkv", "q_a_proj", "q_b_proj", "kv_a_proj",
                                    "kv_b_proj", "o_proj", "wq_a", "wq_b",
                                    "wkv", "wo_a", "wo_b")):
            return {"attn"}
        if "up_proj" in name or "gate_proj" in name or "down_proj" in name:
            return {"ffn_swiglu"}
        return set()
    if op.kind in ("compressor_pool", "indexer_topk"):
        return {"attn"}
    if op.kind == "swiglu":
        return {"ffn_swiglu"}
    if op.kind == "ln":
        return {"ln"}
    if op.kind in ("mhc_pre", "mhc_post", "mhc_head"):
        return {"hc"}
    return set()
