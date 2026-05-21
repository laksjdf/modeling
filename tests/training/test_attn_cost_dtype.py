"""Tests for _attn_cost per-operand byte accounting under quantization (Stage C2)."""
from __future__ import annotations

import pytest

from zrt.training.ir.training_graph import Op, Tensor
from zrt.training.models.flops import _attn_cost
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec


def _make_attn_op(seq: int = 256, heads: int = 4, head_dim: int = 32,
                   in_dtype: Dtype = Dtype.BF16, component: str = "attention"):
    return Op(
        name="L0.attn_core", kind="attn_core",
        inputs=[
            Tensor(name="q", shape_logical=(seq, heads, head_dim),
                   shape_local=(seq, heads, head_dim), dtype=in_dtype,
                   is_activation=True),
            Tensor(name="k", shape_logical=(seq, heads, head_dim),
                   shape_local=(seq, heads, head_dim), dtype=in_dtype,
                   is_activation=True),
            Tensor(name="v", shape_logical=(seq, heads, head_dim),
                   shape_local=(seq, heads, head_dim), dtype=in_dtype,
                   is_activation=True),
        ],
        outputs=[Tensor(name="o", shape_logical=(seq, heads, head_dim),
                        shape_local=(seq, heads, head_dim), dtype=in_dtype,
                        is_activation=True)],
        meta={"b": 1, "s": seq, "heads": heads, "head_dim": head_dim,
              "causal": True},
        layer_id=0, layer_kind=LayerKind.MOE, component=component,
    )


def _model(**kw):
    return ModelSpec(
        hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
        vocab=1000, seq_len=256, layers=[LayerKind.DENSE], **kw,
    )


def test_bf16_baseline_unchanged():
    """When all attention dtypes are BF16, the result equals the v1 formula."""
    m = _model()
    op = _make_attn_op(in_dtype=Dtype.BF16)
    cost = _attn_cost(op, m, system=None)
    # FA tile sizing falls back to (128, 64) when sram_bytes=0
    # All Q/K/V/O bytes at bpe=2; v1 used uniform bpe = inputs[0].dtype.bytes = 2
    assert cost.fwd_bytes > 0
    # Compute the same way for sanity — bpe identical → v2 formula == v1 formula
    import math
    Br, Bc = 128, 64
    Tr = math.ceil(256 / Br)
    Tc = math.ceil(256 / Bc)
    tc_eff = (Tc + 1) / 2  # causal
    q_elems = 1 * 4 * 256 * 32
    kv_elems = 1 * 4 * Tr * tc_eff * Bc * 32
    o_elems = q_elems
    expected_fwd = (q_elems + 2 * kv_elems + o_elems) * 2  # all BF16
    assert cost.fwd_bytes == pytest.approx(expected_fwd)


def test_fp8_attn_halves_fwd_bytes():
    """FP8 attention activations → Q/K/V/O bytes halve."""
    m_bf16 = _model()
    m_fp8 = _model(attn_act_dtype=Dtype.FP8_E4M3)
    op_bf16 = _make_attn_op(in_dtype=Dtype.BF16)
    op_fp8 = _make_attn_op(in_dtype=Dtype.FP8_E4M3)
    c_bf16 = _attn_cost(op_bf16, m_bf16, system=None)
    c_fp8 = _attn_cost(op_fp8, m_fp8, system=None)
    assert c_fp8.fwd_bytes == pytest.approx(c_bf16.fwd_bytes / 2)
    assert c_fp8.dx_bytes == pytest.approx(c_bf16.dx_bytes / 2)


def test_attn_flops_unchanged_by_dtype():
    """Quantization changes peak selection, not arithmetic count."""
    m_bf16 = _model()
    m_fp8 = _model(attn_act_dtype=Dtype.FP8_E4M3)
    op = _make_attn_op()
    c_bf16 = _attn_cost(op, m_bf16, system=None)
    c_fp8 = _attn_cost(op, m_fp8, system=None)
    assert c_bf16.fwd_cube_flops == c_fp8.fwd_cube_flops
    assert c_bf16.fwd_vector_flops == c_fp8.fwd_vector_flops


def test_attn_cost_with_explicit_compression_ratio():
    """Sparse-attn meta override works without consulting model.attn_compression_ratio."""
    m = _model()
    op = _make_attn_op(in_dtype=Dtype.BF16)
    op.meta["sparse_topk"] = 64   # bypasses model.attn_compression_ratio path
    op.meta["swa_window"] = 0
    cost = _attn_cost(op, m, system=None)
    assert cost.fwd_bytes > 0
    assert cost.fwd_cube_flops > 0
