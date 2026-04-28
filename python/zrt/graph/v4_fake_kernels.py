"""DeepSeek-V4 inference/kernel.py torch-only replacements.

Step 1 (modeling_deepseek.py::_install_kernel_stubs) installs minimal stubs in
``sys.modules['kernel']`` so the inference module can import.  Step 3 (this
file) replaces those stubs with real torch implementations whose dtype/shape
exactly match the TileLang kernels and whose body emits a representative
sequence of aten ops, so the FakeTensorMode capture sees the right operator
mix.

Numerical fidelity is *not* a goal — under FakeTensorMode no values are
computed.  The goals are:
  1. Output dtype/shape matches the real kernel exactly (FLOPs / fusion / TP
     analysis depend on it).
  2. Trace contains the dominant aten ops a real implementation would issue
     (amax / clamp / cast / matmul / softmax / sigmoid / reduce_sum / gather)
     so the cost model can categorise nodes correctly.

Op signatures match ``inference/kernel.py``:

| function           | input dtypes              | output                                                                                                        |
|--------------------|---------------------------|---------------------------------------------------------------------------------------------------------------|
| act_quant          | x: BF16 (..., N)          | (y: FP8E4M3 (..., N), s: scale_dtype (..., N//block))                                                         |
| act_quant inplace  | x: BF16 (..., N)          | x in-place (quant→dequant)                                                                                    |
| fp4_act_quant      | x: BF16 (..., N)          | (y: float4_e2m1fn_x2 (..., N//2), s: FP8E8M0 (..., N//block))                                                 |
| fp8_gemm           | x: FP8E4M3, w: FP8E4M3    | BF16 (..., N)                                                                                                 |
| fp4_gemm           | x: FP8E4M3, w: float4_x2  | BF16 (..., N)                                                                                                 |
| sparse_attn        | q: (b,s,h,d) BF16         | o: (b,s,h,d) same dtype as q                                                                                  |
| hc_split_sinkhorn  | mixes: (b,s,(2+hc)*hc)    | (pre: (b,s,hc) FP32, post: (b,s,hc) FP32, comb: (b,s,hc,hc) FP32)                                             |
"""
from __future__ import annotations

import sys
import types
from typing import Optional

import torch


_FP8_MAX = 448.0
_FP4_MAX = 6.0


# ── Quantisation kernels ─────────────────────────────────────────────────────


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    """Block-wise FP8 quantisation matching inference/kernel.py:act_quant."""
    N = x.size(-1)
    shape = x.shape
    x_blocks = x.reshape(*shape[:-1], N // block_size, block_size)
    amax = x_blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    s = amax / _FP8_MAX
    y_blocks = (x_blocks / s).clamp(-_FP8_MAX, _FP8_MAX)

    if inplace:
        # Round-trip through fp8 to model precision loss, then cast back.
        x.copy_((y_blocks * s).reshape(shape).to(x.dtype))
        return x

    y = y_blocks.to(torch.float8_e4m3fn).reshape(shape)
    s_out = s.squeeze(-1).to(scale_dtype)
    return y, s_out


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
):
    """Block-wise FP4 quantisation matching inference/kernel.py:fp4_act_quant."""
    N = x.size(-1)
    shape = x.shape
    x_blocks = x.reshape(*shape[:-1], N // block_size, block_size)
    amax = x_blocks.abs().amax(dim=-1, keepdim=True).clamp_min(_FP4_MAX * (2.0 ** -126))
    s = amax / _FP4_MAX
    y_blocks = (x_blocks / s).clamp(-_FP4_MAX, _FP4_MAX)

    if inplace:
        x.copy_((y_blocks * s).reshape(shape).to(x.dtype))
        return x

    # Output last dim halves under float4_e2m1fn_x2 (2 fp4 packed per byte).
    y = x.new_empty(*shape[:-1], N // 2, dtype=torch.float4_e2m1fn_x2)
    s_out = s.squeeze(-1).to(torch.float8_e8m0fnu)
    return y, s_out


# ── GEMM kernels ─────────────────────────────────────────────────────────────


def _quantised_gemm(x: torch.Tensor, sx: torch.Tensor, w: torch.Tensor,
                    sw: torch.Tensor, scale_dtype) -> torch.Tensor:
    """Common path for fp8/fp4 gemm: dequant + matmul + cast."""
    out_shape = (*x.shape[:-1], w.shape[0])
    x_bf = x.float().to(torch.bfloat16)
    w_bf = w.float().to(torch.bfloat16)
    x_2d = x_bf.reshape(-1, x.shape[-1])
    out_2d = x_2d @ w_bf.t()
    return out_2d.reshape(out_shape)


def fp8_gemm(x, sx, w, sw, scale_dtype):
    return _quantised_gemm(x, sx, w, sw, scale_dtype)


def fp4_gemm(x, sx, w, sw, scale_dtype):
    return _quantised_gemm(x, sx, w, sw, scale_dtype)


# ── Sparse attention ─────────────────────────────────────────────────────────


def sparse_attn(
    q: torch.Tensor,        # (b, s, h, d)
    kv: torch.Tensor,       # (b, T, d) — KV cache or current-step kv
    attn_sink: torch.Tensor,  # (h,)
    topk_idxs: torch.Tensor,  # (b, s, k)  int
    softmax_scale: float,
) -> torch.Tensor:
    """Top-k sparse attention; trace is gather → bmm → softmax → bmm.

    Real TileLang kernel fuses gather + attn into one pass; capture only needs
    the equivalent aten ops to be present.
    """
    b, s, h, d = q.shape
    # Gather KV slice per query position: (b, s, k, d)
    idx = topk_idxs.long().clamp(min=0)
    kv_expanded = kv.unsqueeze(1).expand(b, s, -1, -1)  # (b, s, T, d)
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, d)  # (b, s, k, d)
    kv_gathered = torch.gather(kv_expanded, 2, idx_expanded)

    # Attention: q (b,s,h,d) · k (b,s,k,d).T  → (b,s,h,k)
    scores = torch.einsum("bshd,bskd->bshk", q, kv_gathered) * softmax_scale
    # attn_sink: per-head additive bias on the sink position
    scores = scores + attn_sink.view(1, 1, h, 1).to(scores.dtype)
    weights = scores.softmax(dim=-1).to(kv_gathered.dtype)
    out = torch.einsum("bshk,bskd->bshd", weights, kv_gathered)
    return out.to(q.dtype)


# ── Hyper-Connections sinkhorn split ─────────────────────────────────────────


def hc_split_sinkhorn(
    mixes: torch.Tensor,    # (b, s, (2+hc)*hc) FP32
    hc_scale: torch.Tensor,  # (3,) FP32
    hc_base: torch.Tensor,   # ((2+hc)*hc,) FP32
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Reference torch implementation of inference/kernel.py:hc_split_sinkhorn.

    Splits the projected `mixes` tensor into (pre, post, comb) factors and
    runs `sinkhorn_iters` of row/column normalisation on `comb`.  Math mirrors
    kernel.py:372-427 line-for-line.
    """
    b, s, _ = mixes.shape
    hc = hc_mult

    flat = mixes.reshape(-1, (2 + hc) * hc)              # (B, mix_hc)
    pre_logits  = flat[..., :hc]                          # (B, hc)
    post_logits = flat[..., hc:2 * hc]                    # (B, hc)
    comb_logits = flat[..., 2 * hc:].reshape(-1, hc, hc)  # (B, hc, hc)

    pre  = torch.sigmoid(pre_logits  * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(post_logits * hc_scale[1] + hc_base[hc:2 * hc])

    comb = comb_logits * hc_scale[2] + hc_base[2 * hc:].reshape(hc, hc)
    # row softmax + col-norm (init iter)
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    # remaining sinkhorn iterations: alternate row / col normalisation
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return (
        pre.reshape(b, s, hc),
        post.reshape(b, s, hc),
        comb.reshape(b, s, hc, hc),
    )


# ── Registry / install ──────────────────────────────────────────────────────


_INSTALLED = False


def install() -> None:
    """Replace ``sys.modules['kernel']`` stub bodies with the real torch fakes.

    Idempotent.  Called by ``modeling_deepseek.py`` after the basic stubs are
    in place; safe to call standalone (it creates the module if missing).
    """
    global _INSTALLED
    kernel = sys.modules.get("kernel")
    if kernel is None:
        kernel = types.ModuleType("kernel")
        sys.modules["kernel"] = kernel

    kernel.act_quant = act_quant
    kernel.fp4_act_quant = fp4_act_quant
    kernel.fp8_gemm = fp8_gemm
    kernel.fp4_gemm = fp4_gemm
    kernel.sparse_attn = sparse_attn
    kernel.hc_split_sinkhorn = hc_split_sinkhorn
    kernel._zrt_v4_fake = True

    # fast_hadamard_transform is a function-level import inside
    # rotate_activation; only stub if not already present.
    if "fast_hadamard_transform" not in sys.modules:
        fht = types.ModuleType("fast_hadamard_transform")

        def hadamard_transform(x, scale=1.0):
            return x * scale

        fht.hadamard_transform = hadamard_transform
        fht._zrt_v4_fake = True
        sys.modules["fast_hadamard_transform"] = fht

    _INSTALLED = True


__all__ = [
    "act_quant",
    "fp4_act_quant",
    "fp8_gemm",
    "fp4_gemm",
    "sparse_attn",
    "hc_split_sinkhorn",
    "install",
]
