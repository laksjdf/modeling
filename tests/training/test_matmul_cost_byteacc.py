"""Tests for per-operand byte accounting in _matmul_cost (Stage C1).

Verifies:
  - BF16 baseline numerically matches v1 formula
  - FP4 weight reduces weight HBM bytes ~3.56× (BF16=2 → FP4 stored=0.5625)
  - FP8 activation halves A/C bytes; FP4 weight reduces W bytes; together they
    drop fwd_bytes substantially
  - Backward dx/dw byte formulas are correct under mixed dtype
  - Legacy ``model=None`` path falls back to v1 formula via _bpe
"""
from __future__ import annotations

import pytest

from zrt.training.ir.training_graph import Op, Tensor
from zrt.training.models.flops import _matmul_cost
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec


def _matmul_op(m: int, n: int, k: int, in_dtype: Dtype = Dtype.BF16,
               out_dtype: Dtype = Dtype.BF16, component: str | None = None,
               name: str = "L0.test_matmul"):
    return Op(
        name=name, kind="matmul",
        inputs=[Tensor(name="x", shape_logical=(m, k), shape_local=(m, k),
                       dtype=in_dtype, is_activation=True)],
        outputs=[Tensor(name="y", shape_logical=(m, n), shape_local=(m, n),
                        dtype=out_dtype, is_activation=True)],
        meta={"m": m, "n": n, "k": k},
        layer_id=0, layer_kind=LayerKind.MOE, component=component,
    )


def _moe_model(**dtype_kwargs) -> ModelSpec:
    return ModelSpec(
        hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
        vocab=1000, seq_len=64, layers=[LayerKind.MOE],
        num_experts=8, moe_ffn=256, top_k=2,
        **dtype_kwargs,
    )


# ── Legacy path (model=None) preserves v1 behaviour ──────────────────────


def test_legacy_path_matches_v1_formula():
    op = _matmul_op(m=128, n=64, k=32, in_dtype=Dtype.BF16)
    cost = _matmul_cost(op, model=None)
    # v1: total_bytes = (m*k + k*n + m*n) * 2  (BF16)
    expected = (128 * 32 + 32 * 64 + 128 * 64) * 2
    assert cost.fwd_bytes == expected
    assert cost.dx_bytes == expected
    assert cost.dw_bytes == expected


# ── BF16 baseline with model: same as v1 ─────────────────────────────────


def test_bf16_baseline_with_model_matches_v1():
    m = _moe_model()  # all dtypes BF16 by default (except routed_expert_grad = FP32)
    op = _matmul_op(m=128, n=64, k=32, in_dtype=Dtype.BF16,
                    out_dtype=Dtype.BF16, component=None)
    cost = _matmul_cost(op, m)
    expected = (128 * 32 + 32 * 64 + 128 * 64) * 2
    assert cost.fwd_bytes == expected
    assert cost.dx_bytes == expected
    assert cost.dw_bytes == expected


def test_bf16_baseline_routed_expert_matches_v1():
    """Even routed_expert (which has routed_expert_grad_dtype=FP32 default)
    keeps BF16 dw_bytes — grad_dtype field is for memory only, not op-cost."""
    m = _moe_model()  # routed_expert_weight_dtype defaults to BF16
    op = _matmul_op(m=128, n=64, k=32, component="routed_expert")
    cost = _matmul_cost(op, m)
    expected = (128 * 32 + 32 * 64 + 128 * 64) * 2
    assert cost.fwd_bytes == expected
    assert cost.dw_bytes == expected


# ── FP4 weight reduces fwd_bytes for the weight matrix only ──────────────


def test_fp4_weight_reduces_only_kn_term():
    m = _moe_model(routed_expert_weight_dtype=Dtype.FP4)
    op = _matmul_op(m=128, n=64, k=32, component="routed_expert")
    cost = _matmul_cost(op, m)
    # A (m*k) and C (m*n) still BF16 = 2 bytes
    # W (k*n) now FP4 stored = 0.5625 bytes (incl. block scale)
    expected_fwd = 128 * 32 * 2 + 32 * 64 * 0.5625 + 128 * 64 * 2
    assert cost.fwd_bytes == pytest.approx(expected_fwd)
    # dx also reads W → also reduced
    expected_dx = 128 * 64 * 2 + 32 * 64 * 0.5625 + 128 * 32 * 2
    assert cost.dx_bytes == pytest.approx(expected_dx)


def test_fp4_weight_reduction_ratio():
    """Sanity: FP4 weight bytes ≈ BF16 weight bytes / 3.56."""
    m_bf16 = _moe_model()
    m_fp4 = _moe_model(routed_expert_weight_dtype=Dtype.FP4)
    op_bf16 = _matmul_op(m=128, n=64, k=32, component="routed_expert")
    op_fp4 = _matmul_op(m=128, n=64, k=32, component="routed_expert")
    cost_bf16 = _matmul_cost(op_bf16, m_bf16)
    cost_fp4 = _matmul_cost(op_fp4, m_fp4)
    # Only k*n term differs: (k*n)*2 vs (k*n)*0.5625
    delta = 32 * 64 * (2 - 0.5625)
    assert cost_fp4.fwd_bytes == pytest.approx(cost_bf16.fwd_bytes - delta)


# ── FP8 activation + FP4 weight (DeepSeek-V4) ────────────────────────────


def test_v4_fp8_act_fp4_weight_routed_expert():
    m = _moe_model(
        moe_act_dtype=Dtype.FP8_E4M3,
        routed_expert_weight_dtype=Dtype.FP4,
        routed_expert_compute_dtype=Dtype.FP4,
    )
    # When op.inputs[0].dtype is BF16 but model says FP8, the bundle wins.
    # In a real graph the builder sets the tensor dtype to FP8 already.
    op = _matmul_op(m=128, n=64, k=32, in_dtype=Dtype.FP8_E4M3,
                    out_dtype=Dtype.FP8_E4M3, component="routed_expert")
    cost = _matmul_cost(op, m)
    expected_fwd = 128 * 32 * 1.0 + 32 * 64 * 0.5625 + 128 * 64 * 1.0
    assert cost.fwd_bytes == pytest.approx(expected_fwd)


# ── Attention region uses attn_weight_dtype ──────────────────────────────


def test_attn_matmul_uses_attn_weight_dtype():
    m = _moe_model(
        attn_act_dtype=Dtype.FP8_E4M3,
        attn_weight_dtype=Dtype.BF16,
    )
    op = _matmul_op(m=128, n=64, k=32, in_dtype=Dtype.FP8_E4M3,
                    out_dtype=Dtype.FP8_E4M3, component="attention")
    cost = _matmul_cost(op, m)
    # A (m*k) FP8 = 1 byte, W BF16 = 2 bytes, C FP8 = 1 byte
    expected_fwd = 128 * 32 * 1.0 + 32 * 64 * 2.0 + 128 * 64 * 1.0
    assert cost.fwd_bytes == pytest.approx(expected_fwd)


# ── FLOPs are unchanged by quantization (peak selection is separate) ────


def test_flops_independent_of_dtype():
    m_bf16 = _moe_model()
    m_fp4 = _moe_model(routed_expert_weight_dtype=Dtype.FP4)
    op = _matmul_op(m=128, n=64, k=32, component="routed_expert")
    c_bf16 = _matmul_cost(op, m_bf16)
    c_fp4 = _matmul_cost(op, m_fp4)
    assert c_bf16.fwd_cube_flops == c_fp4.fwd_cube_flops
    assert c_bf16.dx_cube_flops == c_fp4.dx_cube_flops
    assert c_bf16.dw_cube_flops == c_fp4.dw_cube_flops


# ── fwd_multiplier still applied ─────────────────────────────────────────


def test_fwd_multiplier_scales_flops():
    m = _moe_model()
    op = _matmul_op(m=128, n=64, k=32, component="routed_expert")
    op.meta["fwd_multiplier"] = 3.0
    cost = _matmul_cost(op, m)
    base_flops = 2.0 * 128 * 64 * 32
    assert cost.fwd_cube_flops == pytest.approx(base_flops * 3.0)


# ── dw_bytes formula sanity ──────────────────────────────────────────────


def test_dw_bytes_formula_under_fp8_act():
    """dw reads dC (grad_in) + A (in_act), writes dW (grad_weight). All three
    default to in_act = moe_act_dtype under mixed quant."""
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3,
                   routed_expert_weight_dtype=Dtype.FP4)
    op = _matmul_op(m=128, n=64, k=32, in_dtype=Dtype.FP8_E4M3,
                    component="routed_expert")
    cost = _matmul_cost(op, m)
    # dC=FP8, A=FP8, dW=FP8 (grad_weight defaults to in_act)
    expected = 128 * 64 * 1.0 + 128 * 32 * 1.0 + 32 * 64 * 1.0
    assert cost.dw_bytes == pytest.approx(expected)
