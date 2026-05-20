"""Per-op analytical FLOPs model.

Returns raw cost per op. Recompute multiplier applied by the stage composer.
Reference: Calculon (Isaev et al. SC'23), Korthikanti et al. 2022.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from zrt.training.ir.training_graph import Graph, Op
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import SystemSpec


@dataclass
class OpCost:
    fwd_bytes: float = 0.0
    dx_bytes: float = 0.0
    dw_bytes: float = 0.0
    bound: str = "compute"   # diagnostic only — bound is determined dynamically by _is_compute_bound

    # Heterogeneous core FLOPs (Cube/Vector).
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


def op_cost(op: Op, model: ModelSpec, system: SystemSpec | None = None) -> OpCost:
    """Compute raw cost per op (FLOPs + bytes for roofline timing)."""
    if op.kind == "matmul":
        return _matmul_cost(op, model)
    if op.kind == "sparse_attn":
        return _sparse_attn_cost(op, system, model)
    if op.kind == "hca_attn":
        return _hca_attn_cost(op, system, model)
    if op.kind == "swa_attn":
        return _swa_attn_cost(op, system, model)
    if op.kind == "attn_core":
        return _attn_cost(op, model, system)
    if op.kind == "mhc_pre":
        return _mhc_pre_cost(op)
    if op.kind == "mhc_post":
        return _mhc_post_cost(op)
    if op.kind == "mhc_head":
        return _mhc_head_cost(op)
    if op.kind in ("ln", "rmsnorm", "softmax"):
        return _promote_aware_elementwise_cost(op, model, system)
    if op.kind in ("rope", "swiglu", "add", "hc_expand"):
        return _elementwise_cost(op)
    if op.kind == "embed":
        return _embed_cost(op)
    if op.kind == "lm_head":
        return _matmul_cost(op, model)
    if op.kind == "compressor_pool":
        return _compressor_pool_cost(op)
    if op.kind == "indexer_topk":
        return _indexer_topk_cost(op)
    if op.kind == "hash_route":
        return OpCost()  # table lookup, negligible FLOPs
    if op.kind == "cast":
        return _cast_cost(op)
    # Unknown ops: zero cost
    return OpCost()


def _cast_cost(op: Op) -> OpCost:
    """Dtype-boundary cast / quantize / dequantize op.

    fwd_bytes = n × (src.bytes + dst.stored_bytes)   if not fused
              + n × src.bytes                         if needs_amax (BF16→FP8/FP4)

    When ``op.meta["fused"]`` is True (e.g., LN epilog absorbs the cast,
    GEMM epilog fused), returns 0 — the kernel does not consume extra
    HBM bandwidth. ``QuantPolicy.assume_all_casts_fused`` (default True)
    sets this flag everywhere → cast ops are present in the IR for the
    reports but cost nothing, exactly matching v1 behaviour.

    dx_bytes mirrors fwd (the inverse cast happens during backward).
    dw_bytes = 0 (cast never produces a weight gradient).
    """
    if op.meta.get("fused", False):
        return OpCost(bound="memory")

    n = float(op.meta.get("num_elements", 0))
    src = op.meta.get("src_dtype")
    dst = op.meta.get("dst_dtype")
    if n <= 0 or src is None or dst is None:
        return OpCost(bound="memory")

    src_b = src.bytes
    dst_b = dst.stored_bytes
    needs_amax = bool(op.meta.get("needs_amax", False))

    main_bytes = n * (src_b + dst_b)
    amax_bytes = (n * src_b) if needs_amax else 0.0

    fwd_bytes = main_bytes + amax_bytes
    return OpCost(
        fwd_bytes=fwd_bytes,
        dx_bytes=fwd_bytes,        # reverse cast during backward
        dw_bytes=0.0,
        bound="memory",
    )


def _embed_cost(op: Op) -> OpCost:
    """Embedding lookup: gather operation, memory-bound with zero FLOPs.

    Embedding lookup reads rows from an embedding table based on indices.
    This is a gather operation, not a matmul, so FLOPs = 0.

    Forward: read embedding table rows -> write embeddings
    Backward: read embeddings -> update embedding table rows (gradient scatter)
    """
    seq = op.inputs[0].shape_local[0] if op.inputs else op.meta.get("m", 0)
    hidden = op.outputs[0].shape_local[-1] if op.outputs else op.meta.get("n", 0)

    bpe = _bpe(op)
    fwd_bytes = seq * hidden * bpe

    return OpCost(
        fwd_bytes=fwd_bytes,
        dx_bytes=fwd_bytes,
        dw_bytes=fwd_bytes,
        bound="memory",
    )


def _matmul_cost(op: Op, model: "ModelSpec | None" = None) -> OpCost:
    """Per-operand byte accounting for matmul (v2 mixed-quant).

    Forward GEMM: C[m,n] = A[m,k] @ W[k,n]
      fwd_bytes = m*k * in_act.bytes
                + k*n * weight.stored_bytes        ← FP4 block-scale aware
                + m*n * out_act.bytes
      dx_bytes  = m*n * grad_in.bytes
                + k*n * weight.stored_bytes
                + m*k * grad_act.bytes
      dw_bytes  = m*n * grad_in.bytes
                + m*k * in_act.bytes
                + k*n * grad_weight.stored_bytes

    Default bundle ``grad_in == grad_act == grad_weight == in_act`` so
    BF16 baseline reduces to v1's ``(mk+kn+mn) * act.bytes``. Per-component
    overrides (FP4 weight, FP8 act) reshape the formula naturally.
    """
    # Fused ops (routed_expert) and grouped matmuls (wo_a) have meta
    # dimensions that don't correspond to tensor shapes — must read from meta.
    meta_k = op.meta.get("k", 0)
    use_meta = (
        op.meta.get("fused_weight_dims", False)
        or not op.inputs
        or not op.outputs
        or (meta_k > 0 and op.inputs[0].shape_logical[-1] != meta_k)
    )
    if use_meta:
        m = op.meta.get("m", 0)
        n = op.meta.get("n_local", op.meta.get("n", 0))
        k = op.meta.get("k_local", op.meta.get("k", 0))
    else:
        m = op.inputs[0].shape_local[0]
        k = op.inputs[0].shape_local[-1]
        n = op.outputs[0].shape_local[-1]

    mult = op.meta.get("fwd_multiplier", 1.0)
    fwd = 2.0 * m * n * k * mult
    dx = 2.0 * m * n * k * mult
    dw = 2.0 * m * n * k * mult

    # Resolve per-operand dtypes via the centralized bundle. When ``model``
    # is None (legacy test harness), fall back to ``_bpe(op)`` for v1
    # behaviour.
    if model is None:
        bpe = _bpe(op)
        total_bytes = (m * k + k * n + m * n) * bpe
        return OpCost(
            fwd_bytes=total_bytes,
            dx_bytes=total_bytes,
            dw_bytes=total_bytes,
            fwd_cube_flops=fwd,
            dx_cube_flops=dx,
            dw_cube_flops=dw,
        )

    from zrt.training.models.quant import resolve_op_dtypes
    d = resolve_op_dtypes(op, model)
    A_b = d.in_act.bytes
    W_b = d.weight.stored_bytes
    C_b = d.out_act.bytes
    dC_b = d.grad_in.bytes
    dA_b = d.grad_act.bytes
    dW_b = d.grad_weight.stored_bytes

    fwd_bytes = m * k * A_b + k * n * W_b + m * n * C_b
    dx_bytes = m * n * dC_b + k * n * W_b + m * k * dA_b
    dw_bytes = m * n * dC_b + m * k * A_b + k * n * dW_b

    return OpCost(
        fwd_bytes=fwd_bytes,
        dx_bytes=dx_bytes,
        dw_bytes=dw_bytes,
        fwd_cube_flops=fwd,
        dx_cube_flops=dx,
        dw_cube_flops=dw,
    )


def _fa_tile_shape(head_dim: int, act_bpe: int, sram_bytes: int) -> tuple[int, int]:
    """Return (Br, Bc) tile dimensions for FlashAttention-2.

    Fits Q_tile + K_tile + V_tile + S_tile in SRAM:
      SRAM ≈ Br·d·bpe + 2·Bc·d·bpe + Br·Bc·4 (fp32 stats).
    Returns FA-2 defaults (128, 64) when sram_bytes == 0.
    """
    if sram_bytes <= 0:
        return 128, 64
    Bc = max(16, min(128, sram_bytes // (4 * head_dim * act_bpe)))
    Br = max(16, min(128, sram_bytes // (3 * head_dim * act_bpe)))
    return Br, Bc


def _attn_region_bytes(op: Op, model: "ModelSpec | None") -> tuple[float, float, float]:
    """Return per-element bytes for (Q/K/V reads, O write, grad activations).

    Attention has no weight matrix (QKV/O proj are separate matmul ops),
    so byte accounting only tracks activation tensors. The bundle's
    ``in_act`` covers Q/K/V reads, ``out_act`` covers the O write, and
    ``grad_in/grad_act`` cover bwd tensors. Default bundle keeps these
    all at the region's activation dtype so BF16 baseline matches v1.
    """
    if model is None:
        bpe = _bpe(op)
        return bpe, bpe, bpe
    from zrt.training.models.quant import resolve_op_dtypes
    d = resolve_op_dtypes(op, model)
    return d.in_act.bytes, d.out_act.bytes, d.grad_in.bytes


def _attn_cost(op: Op, model: ModelSpec | None, system: SystemSpec | None = None) -> OpCost:
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
        # FA-2-class kernels skip masked tiles in the cube engine
        cube_fwd = fwd
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

    qkv_b, o_b, grad_b = _attn_region_bytes(op, model)
    bpe_tile = _bpe(op)  # SRAM tile sizing uses fwd activation precision

    # Tile-aware byte formula: K,V are re-read per Q-block in FlashAttention.
    sram_bytes = int(system.gpu.sram_kb_per_sm * 1024) if system and system.gpu.sram_kb_per_sm > 0 else 0
    Br, Bc = _fa_tile_shape(d, bpe_tile, sram_bytes)
    Tr = math.ceil(s / Br)
    Tc = math.ceil(kv_len / Bc)
    # Causal: average K-blocks per Q-block is (Tc+1)/2 for dense contiguous mask,
    # but top-k positions are scattered — no causal halving for sparse.
    if sparse_topk > 0:
        tc_eff = Tc
    else:
        tc_eff = (Tc + 1) / 2 if causal else Tc

    q_elems  = b * h * s * d
    o_elems  = b * h * s * d
    kv_elems = b * h * Tr * tc_eff * Bc * d
    # fwd: read Q + 2×KV at fwd-act dtype; write O at out_act dtype.
    fwd_bytes = (q_elems + 2.0 * kv_elems) * qkv_b + o_elems * o_b
    # Backward: ≈2× fwd minus small saved-state (M, L) overhead. dO and
    # dQ/dK/dV all live at the grad activation dtype (= in_act by default,
    # preserving v1 baseline). Halves the bwd traffic when bwd runs in FP8.
    dx_bytes = (2.0 * q_elems + 4.0 * kv_elems) * grad_b

    return OpCost(
        fwd_bytes=fwd_bytes,
        dx_bytes=dx_bytes,
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


def _sparse_attn_cost(op: Op, system: SystemSpec | None = None,
                       model: "ModelSpec | None" = None) -> OpCost:
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
        return _attn_cost(op, model, system)

    effective_len = topk + swa
    fwd = 2.0 * b * s * effective_len * h * d
    cube_fwd = fwd
    vector_fwd = 5.0 * b * h * s * effective_len

    qkv_b, o_b, grad_b = _attn_region_bytes(op, model)
    bpe_tile = _bpe(op)
    sram_bytes = int(system.gpu.sram_kb_per_sm * 1024) if system and system.gpu.sram_kb_per_sm > 0 else 0
    Br, Bc = _fa_tile_shape(d, bpe_tile, sram_bytes)
    Tr = math.ceil(s / Br)
    Tc = math.ceil(effective_len / Bc)
    tc_eff = Tc  # top-k positions are scattered, no causal halving
    q_elems = b * h * s * d
    o_elems = b * h * s * d
    kv_elems = b * h * Tr * tc_eff * Bc * d
    fwd_bytes = (q_elems + 2.0 * kv_elems) * qkv_b + o_elems * o_b
    dx_bytes = (2.0 * q_elems + 4.0 * kv_elems) * grad_b

    return OpCost(
        fwd_bytes=fwd_bytes,
        dx_bytes=dx_bytes,
        fwd_cube_flops=cube_fwd,
        fwd_vector_flops=vector_fwd,
        dx_cube_flops=2.5 * cube_fwd,
        dx_vector_flops=2.5 * vector_fwd,
    )


def _hca_attn_cost(op: Op, system: SystemSpec | None = None,
                    model: "ModelSpec | None" = None) -> OpCost:
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
        return _attn_cost(op, model, system)

    compressed_len = max(1, s // ratio)
    effective_len = compressed_len + swa
    fwd = 2.0 * b * s * effective_len * h * d
    cube_fwd = fwd
    vector_fwd = 5.0 * b * h * s * effective_len

    qkv_b, o_b, grad_b = _attn_region_bytes(op, model)
    bpe_tile = _bpe(op)
    sram_bytes = int(system.gpu.sram_kb_per_sm * 1024) if system and system.gpu.sram_kb_per_sm > 0 else 0
    Br, Bc = _fa_tile_shape(d, bpe_tile, sram_bytes)
    Tr = math.ceil(s / Br)
    Tc = math.ceil(effective_len / Bc)
    tc_eff = (Tc + 1) / 2  # always causal
    q_elems = b * h * s * d
    o_elems = b * h * s * d
    kv_elems = b * h * Tr * tc_eff * Bc * d
    fwd_bytes = (q_elems + 2.0 * kv_elems) * qkv_b + o_elems * o_b
    dx_bytes = (2.0 * q_elems + 4.0 * kv_elems) * grad_b

    return OpCost(
        fwd_bytes=fwd_bytes,
        dx_bytes=dx_bytes,
        fwd_cube_flops=cube_fwd,
        fwd_vector_flops=vector_fwd,
        dx_cube_flops=2.5 * cube_fwd,
        dx_vector_flops=2.5 * vector_fwd,
    )


def _swa_attn_cost(op: Op, system: SystemSpec | None = None,
                    model: "ModelSpec | None" = None) -> OpCost:
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
        return _attn_cost(op, model, system)

    fwd = 2.0 * b * s * swa * h * d
    cube_fwd = fwd
    vector_fwd = 5.0 * b * h * s * swa

    qkv_b, o_b, grad_b = _attn_region_bytes(op, model)
    bpe_tile = _bpe(op)
    sram_bytes = int(system.gpu.sram_kb_per_sm * 1024) if system and system.gpu.sram_kb_per_sm > 0 else 0
    Br, Bc = _fa_tile_shape(d, bpe_tile, sram_bytes)
    Tr = math.ceil(s / Br)
    Tc = math.ceil(swa / Bc)
    tc_eff = (Tc + 1) / 2  # always causal
    q_elems = b * h * s * d
    o_elems = b * h * s * d
    kv_elems = b * h * Tr * tc_eff * Bc * d
    fwd_bytes = (q_elems + 2.0 * kv_elems) * qkv_b + o_elems * o_b
    dx_bytes = (2.0 * q_elems + 4.0 * kv_elems) * grad_b

    return OpCost(
        fwd_bytes=fwd_bytes,
        dx_bytes=dx_bytes,
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
        fwd_bytes=fwd_bytes,
        dx_bytes=fwd_bytes * 1.5,
        dw_bytes=0.0,
        bound="memory",
        fwd_cube_flops=0.0,
        fwd_vector_flops=fwd_flops,
        dx_cube_flops=0.0,
        dx_vector_flops=fwd_flops * 2.5,
    )


def _promote_aware_elementwise_cost(
    op: Op, model: "ModelSpec | None", system: SystemSpec | None = None,
) -> OpCost:
    """Softmax / LN / RMSNorm with optional FP32 promote on quantized input.

    When ``Strategy.quant.ln_softmax_promote_fp32`` is True (the default —
    matches production fused LN/softmax kernels) and the op's first input
    arrives in a quantized dtype (FP8 / FP4), we add an extra read of the
    input tensor's elements to model the per-tensor max-abs reduction
    plus FP32 promotion. The cost is:

      LN/RMSNorm: +1× input elements (one reduce pass for mean/var)
      Softmax:    +2× input elements (max + sum-of-exp passes)

    When the input is BF16/FP32 (no promote needed), behavior is
    identical to ``_elementwise_cost`` — preserving the v1 baseline.
    """
    base = _elementwise_cost(op)

    if model is None or not op.inputs:
        return base

    # QuantPolicy lives on Strategy but isn't reachable from op_cost; for
    # now, gate on whether input dtype is quantized. The strategy-level
    # toggle will be threaded through in Stage E when reports surface
    # cast/promote bytes.
    in_dtype = op.inputs[0].dtype
    QUANT_DTYPES = {Dtype.FP4, Dtype.FP8_E4M3, Dtype.FP8_E5M2}
    if in_dtype not in QUANT_DTYPES:
        return base

    n_in = op.inputs[0].num_elements()
    extra_reads = 2 if op.kind == "softmax" else 1
    extra_bytes = extra_reads * n_in * in_dtype.bytes
    return OpCost(
        fwd_bytes=base.fwd_bytes + extra_bytes,
        dx_bytes=base.dx_bytes + extra_bytes,
        dw_bytes=base.dw_bytes,
        bound="memory",
        fwd_cube_flops=base.fwd_cube_flops,
        fwd_vector_flops=base.fwd_vector_flops,
        dx_cube_flops=base.dx_cube_flops,
        dx_vector_flops=base.dx_vector_flops,
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
    return OpCost(fwd_bytes=bytes_fwd, dx_bytes=bytes_fwd * 1.5,
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
        fwd_bytes=bytes_fwd,
        fwd_cube_flops=einsum_flops,
        dx_cube_flops=2.0 * einsum_flops,
    )


def _is_compute_bound(cost: OpCost, phase: str, system: SystemSpec) -> bool:
    """Determine if an op phase is compute-bound at the roofline knee.

    Uses peak FLOPS and peak BW (no efficiency curves) so that bound is a
    structural property of the op shape, not dependent on the efficiency model.
    Intentionally excluded from the unified effective_flops/effective_hbm_bw
    entries for this reason.
    """
    flops = getattr(cost, f"{phase}_cube_flops") + getattr(cost, f"{phase}_vector_flops")
    bytes_ = getattr(cost, f"{phase}_bytes")
    if flops <= 0:
        return False  # memory-only op (e.g. embed)
    if bytes_ <= 0:
        return True
    compute_t = flops / (system.gpu.flops_bf16 * 1e12)
    memory_t = bytes_ / (system.gpu.hbm_bw_gbps * 1e9)
    return compute_t >= memory_t


def _accounted_flops(cost: OpCost, phase: str) -> float:
    """Return cube + vector FLOPs for a given phase."""
    return getattr(cost, f"{phase}_cube_flops") + getattr(cost, f"{phase}_vector_flops")


def total_training_flops(
    graph: Graph, model: ModelSpec, strategy: Strategy,
    system: SystemSpec,
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
        cost = op_cost(op, model, system)
        for phase in ("fwd", "dx", "dw"):
            if _is_compute_bound(cost, phase, system):
                total += _accounted_flops(cost, phase)

    # Scale by microbatch count
    M = strategy.num_microbatches()
    total *= M

    return total


def forward_backward_flops(
    graph: Graph, model: ModelSpec, strategy: Strategy,
    system: SystemSpec,
) -> tuple[float, float]:
    """Return (forward_flops, backward_flops) separately.

    Same loop as total_training_flops but splits fwd from dx+dw.
    """
    fwd = 0.0
    bwd = 0.0
    for op in graph.ops:
        cost = op_cost(op, model, system)
        if _is_compute_bound(cost, "fwd", system):
            fwd += _accounted_flops(cost, "fwd")
        if _is_compute_bound(cost, "dx", system):
            bwd += _accounted_flops(cost, "dx")
        if _is_compute_bound(cost, "dw", system):
            bwd += _accounted_flops(cost, "dw")
    M = strategy.num_microbatches()
    return fwd * M, bwd * M


def recompute_overhead_flops(
    graph: Graph, model: ModelSpec, strategy: Strategy,
    system: SystemSpec,
) -> float:
    """Extra FLOPs from recomputing forward activations during backward pass.

    Selective recompute re-runs the forward for specific ops (typically attention)
    during backward. Full recompute re-runs the entire forward pass.

    Respects per-layer policies: only ops belonging to a layer whose kind
    appears in ``RecomputePolicy.per_layer`` are counted.

    Only counts compute-bound ops, consistent with total_training_flops.

    FlashAttention-family kernels (``attn_core``/``sparse_attn``/``hca_attn``/
    ``swa_attn``) have a backward FLOPs convention of ``2.5×fwd`` — the extra
    ``0.5×fwd`` already represents FA's mandatory internal recompute of
    scores. Adding their forward FLOPs here again would triple-count
    attention. So we skip those op kinds from the overhead; non-FA targets
    (QKV/O linear projections, indexer/compressor, swiGLU, ln) keep their
    existing accounting.
    """
    rc = strategy.recompute
    if not rc.per_layer:
        return 0.0

    FA_KERNEL_KINDS = {"attn_core", "sparse_attn", "hca_attn", "swa_attn"}

    extra = 0.0
    for op in graph.ops:
        # Look up the layer kind for this op
        if op.layer_id < 0 or op.layer_id >= len(model.layers):
            continue
        if op.kind == "cast":
            # See §12.7 of mixed_quant_v2_op_bytes_zh.md — v2 first cut
            # skips cast in recompute accounting (small 2nd-order effect).
            continue
        lk = model.layers[op.layer_id].value
        cats = rc.per_layer.get(lk)
        if not cats:
            continue

        op_cats = _op_recompute_categories(op)
        if "full" in cats or (op_cats & cats):
            if op.kind in FA_KERNEL_KINDS:
                continue
            cost = op_cost(op, model, system)
            # Mirror _recompute_time (compose/stage.py) which counts the
            # actual recompute time including memory-bound ops. The previous
            # _is_compute_bound gate dropped low-AI ops like mhc_pre /
            # mhc_post (AI<10 vs B300 ridge=625), so HFU stayed equal to MFU
            # while step time grew — accounting drift between the two sides.
            extra += _accounted_flops(cost, "fwd")

    M = strategy.num_microbatches()
    return extra * M


def _op_recompute_categories(op: Op) -> set[str]:
    """Map an op to its recompute category set.

    Recompute categories:

    - ``"attn_core"`` — FA-kernel attention ops + indexer / compressor pool
      (selective recompute, Megatron-LM ``selective`` flavor).
    - ``"attn_block"`` — everything in ``"attn_core"`` *plus* the QKV / O
      linear projections (block recompute, the heavier flavor).
    - ``"attn"`` — deprecated alias of ``"attn_block"``. Returned alongside
      ``"attn_block"`` on every op that matches the heavier scope so that
      legacy YAML configurations keep working unchanged.
    - ``"ffn_swiglu"`` — FFN up / gate / down projections + SwiGLU.
    - ``"ln"`` — LayerNorm / RMSNorm.
    - ``"hc"`` — DeepSeek-V4 mhc_pre / mhc_post / mhc_head.
    """
    if op.kind in ("attn_core", "sparse_attn", "hca_attn", "swa_attn"):
        # FA kernel: in scope for both selective ("attn_core") and block
        # recompute ("attn_block"). Note: FA kernel forwards are still
        # skipped from overhead by recompute_overhead_flops (FA dedup).
        return {"attn_core", "attn_block", "attn"}
    if op.kind == "matmul":
        name = op.name.lower()
        if any(k in name for k in ("qkv", "q_a_proj", "q_b_proj", "kv_a_proj",
                                    "kv_b_proj", "o_proj", "wq_a", "wq_b",
                                    "wkv", "wo_a", "wo_b")):
            # QKV / O linear projections: NOT selective-scope; only block.
            return {"attn_block", "attn"}
        if "up_proj" in name or "gate_proj" in name or "down_proj" in name:
            return {"ffn_swiglu"}
        return set()
    if op.kind in ("compressor_pool", "indexer_topk"):
        return {"attn_core", "attn_block", "attn"}
    if op.kind == "swiglu":
        return {"ffn_swiglu"}
    if op.kind == "ln":
        return {"ln"}
    if op.kind in ("mhc_pre", "mhc_post", "mhc_head"):
        return {"hc"}
    return set()
