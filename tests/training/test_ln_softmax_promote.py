"""Tests for FP32-promote-aware bytes on softmax / LN / RMSNorm (Stage C3)."""
from __future__ import annotations

import pytest

from zrt.training.ir.training_graph import Op, Tensor
from zrt.training.models.flops import op_cost
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec


def _model(act_dtype=Dtype.BF16):
    return ModelSpec(
        hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
        vocab=1000, seq_len=64, layers=[LayerKind.DENSE], act_dtype=act_dtype,
    )


def _ln_op(seq: int, h: int, in_dtype: Dtype, out_dtype: Dtype, kind: str = "rmsnorm"):
    return Op(
        name="L0.ln1", kind=kind,
        inputs=[Tensor(name="x", shape_logical=(seq, h), shape_local=(seq, h),
                       dtype=in_dtype, is_activation=True)],
        outputs=[Tensor(name="y", shape_logical=(seq, h), shape_local=(seq, h),
                        dtype=out_dtype, is_activation=True)],
        meta={},
        layer_id=0, layer_kind=LayerKind.DENSE, component="norm",
    )


def _softmax_op(b: int, h: int, s: int, in_dtype: Dtype):
    return Op(
        name="L0.softmax", kind="softmax",
        inputs=[Tensor(name="scores", shape_logical=(b, h, s, s),
                       shape_local=(b, h, s, s), dtype=in_dtype,
                       is_activation=True)],
        outputs=[Tensor(name="probs", shape_logical=(b, h, s, s),
                        shape_local=(b, h, s, s), dtype=in_dtype,
                        is_activation=True)],
        meta={},
        layer_id=0, layer_kind=LayerKind.DENSE, component=None,
    )


# ── BF16 baseline: no promote ────────────────────────────────────────────


def test_bf16_ln_no_promote():
    m = _model()
    op = _ln_op(seq=128, h=64, in_dtype=Dtype.BF16, out_dtype=Dtype.BF16,
                kind="rmsnorm")
    cost = op_cost(op, m)
    # input + output bytes = 2 * 128*64*2 = 32768
    assert cost.fwd_bytes == 128 * 64 * 2 + 128 * 64 * 2


def test_bf16_softmax_no_promote():
    m = _model()
    op = _softmax_op(b=1, h=4, s=64, in_dtype=Dtype.BF16)
    cost = op_cost(op, m)
    # No extra reads
    assert cost.fwd_bytes == 1 * 4 * 64 * 64 * 2 + 1 * 4 * 64 * 64 * 2


# ── FP8 input triggers FP32 promote: extra reads ─────────────────────────


def test_fp8_ln_adds_one_extra_read():
    m = _model()
    op = _ln_op(seq=128, h=64, in_dtype=Dtype.FP8_E4M3, out_dtype=Dtype.BF16,
                kind="rmsnorm")
    cost = op_cost(op, m)
    # input(FP8=1) + output(BF16=2) = 128*64*3 = 24576
    base_bytes = 128 * 64 * 1 + 128 * 64 * 2
    # plus extra read for FP32 reduce: 1× input elements at FP8 dtype
    extra = 1 * 128 * 64 * 1
    assert cost.fwd_bytes == base_bytes + extra


def test_fp8_softmax_adds_two_extra_reads():
    m = _model()
    op = _softmax_op(b=1, h=4, s=64, in_dtype=Dtype.FP8_E4M3)
    cost = op_cost(op, m)
    n = 1 * 4 * 64 * 64
    base_bytes = n * 1 + n * 1  # input + output FP8
    extra = 2 * n * 1  # max + sum-of-exp passes
    assert cost.fwd_bytes == base_bytes + extra


def test_fp4_ln_adds_one_extra_read():
    m = _model()
    op = _ln_op(seq=128, h=64, in_dtype=Dtype.FP4, out_dtype=Dtype.BF16,
                kind="rmsnorm")
    cost = op_cost(op, m)
    # input(FP4=0.5) + output(BF16=2)
    base_bytes = 128 * 64 * 0.5 + 128 * 64 * 2
    extra = 1 * 128 * 64 * 0.5
    assert cost.fwd_bytes == pytest.approx(base_bytes + extra)


def test_fp32_input_no_promote():
    """FP32 input means promote is already done — no extra reads."""
    m = _model()
    op = _ln_op(seq=128, h=64, in_dtype=Dtype.FP32, out_dtype=Dtype.BF16,
                kind="rmsnorm")
    cost = op_cost(op, m)
    base_bytes = 128 * 64 * 4 + 128 * 64 * 2
    assert cost.fwd_bytes == base_bytes  # no extra
