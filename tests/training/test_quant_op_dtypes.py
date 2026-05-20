"""Tests for OpDtypeBundle / resolve_op_dtypes (Stage A3, v2 mixed quant).

Coverage:
  - per-component dispatch (attention / routed_expert / shared_expert /
    embedding / norm / unknown)
  - fallback chain (None sentinels resolve to param_dtype / grad_dtype)
  - expected_input_dtype special-cases (residual add, dispatch/combine)
  - cast op picks up dtypes from meta
"""
from __future__ import annotations

import pytest

from zrt.training.ir.training_graph import Op
from zrt.training.models.quant import (
    OpDtypeBundle, expected_input_dtype, resolve_op_dtypes,
)
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec


def _model(**dtype_kwargs) -> ModelSpec:
    return ModelSpec(
        hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
        vocab=1000, seq_len=64, layers=[LayerKind.MOE], num_experts=8,
        moe_ffn=128, top_k=2,
        **dtype_kwargs,
    )


def _op(name: str, kind: str = "matmul", component: str | None = None, **meta) -> Op:
    return Op(name=name, kind=kind, component=component, meta=dict(meta))


# ── Component dispatch ────────────────────────────────────────────────────


def test_attention_bundle_picks_attn_fields():
    m = _model(
        attn_compute_dtype=Dtype.FP8_E4M3,
        attn_weight_dtype=Dtype.BF16,
        attn_act_dtype=Dtype.FP8_E4M3,
        attn_grad_dtype=Dtype.BF16,
    )
    b = resolve_op_dtypes(_op("L0.qkv", component="attention"), m)
    assert b.in_act is Dtype.FP8_E4M3
    assert b.weight is Dtype.BF16
    assert b.out_act is Dtype.FP8_E4M3
    assert b.compute is Dtype.FP8_E4M3
    # Bwd byte slots track in_act (region activation dtype) — matches v1's
    # implicit "op_cost reads inputs[0].dtype" assumption.
    assert b.grad_in is Dtype.FP8_E4M3
    assert b.grad_weight is Dtype.FP8_E4M3
    assert b.grad_act is Dtype.FP8_E4M3


def test_routed_expert_bundle_uses_routed_fields():
    m = _model(
        routed_expert_compute_dtype=Dtype.FP4,
        routed_expert_weight_dtype=Dtype.FP4,
        moe_act_dtype=Dtype.FP8_E4M3,
        routed_expert_grad_dtype=Dtype.FP32,
        grad_dtype=Dtype.BF16,
    )
    b = resolve_op_dtypes(_op("L0.routed_expert_ffn", component="routed_expert"), m)
    assert b.in_act is Dtype.FP8_E4M3
    assert b.weight is Dtype.FP4
    assert b.compute is Dtype.FP4
    # Bwd activation/grad bytes track moe_act_dtype (= FP8 here).
    # routed_expert_grad_dtype (= FP32) is for memory_breakdown.grads, NOT
    # op-cost dW bytes; v1 baseline preserved.
    assert b.grad_in is Dtype.FP8_E4M3
    assert b.grad_weight is Dtype.FP8_E4M3
    assert b.grad_act is Dtype.FP8_E4M3


def test_shared_expert_bundle_uses_shared_fields():
    m = _model(
        shared_expert_compute_dtype=Dtype.FP8_E4M3,
        shared_expert_weight_dtype=Dtype.BF16,
        moe_act_dtype=Dtype.FP8_E4M3,
        shared_expert_grad_dtype=Dtype.BF16,
    )
    b = resolve_op_dtypes(_op("L0.shared_up", component="shared_expert"), m)
    assert b.in_act is Dtype.FP8_E4M3
    assert b.weight is Dtype.BF16
    assert b.compute is Dtype.FP8_E4M3
    assert b.grad_weight is Dtype.FP8_E4M3


def test_embedding_forced_bf16_regardless_of_act_dtype():
    m = _model(act_dtype=Dtype.FP8_E4M3)
    b = resolve_op_dtypes(_op("embed", kind="embed", component="embedding"), m)
    assert b.in_act is Dtype.BF16
    assert b.weight is Dtype.BF16
    assert b.compute is Dtype.BF16


def test_norm_forced_bf16():
    m = _model(act_dtype=Dtype.FP8_E4M3)
    b = resolve_op_dtypes(_op("L0.ln", kind="ln", component="norm"), m)
    assert b.compute is Dtype.BF16


def test_unknown_component_falls_back_to_global():
    m = _model(
        act_dtype=Dtype.BF16,
        param_dtype=Dtype.BF16,
        grad_dtype=Dtype.FP32,
    )
    b = resolve_op_dtypes(_op("L0.something", component=None), m)
    assert b.in_act is Dtype.BF16
    assert b.weight is Dtype.BF16
    # grad_weight in default bundle tracks act_dtype, not grad_dtype.
    # grad_dtype is for memory_breakdown.grads (stored grad footprint).
    assert b.grad_weight is Dtype.BF16


# ── Fallback chain (None sentinels) ──────────────────────────────────────


def test_attn_weight_dtype_defaults_to_param_dtype():
    m = _model(param_dtype=Dtype.BF16)
    # attn_weight_dtype was not set; __post_init__ should fill it with param_dtype.
    assert m.attn_weight_dtype is Dtype.BF16
    b = resolve_op_dtypes(_op("L0.qkv", component="attention"), m)
    assert b.weight is Dtype.BF16


def test_shared_expert_weight_dtype_defaults_to_param_dtype():
    m = _model(param_dtype=Dtype.BF16)
    assert m.shared_expert_weight_dtype is Dtype.BF16


def test_attn_grad_dtype_defaults_to_grad_dtype():
    m = _model(grad_dtype=Dtype.BF16)
    assert m.attn_grad_dtype is Dtype.BF16


def test_shared_expert_grad_dtype_defaults_to_grad_dtype():
    m = _model(grad_dtype=Dtype.BF16)
    assert m.shared_expert_grad_dtype is Dtype.BF16


def test_explicit_override_wins_over_sentinel():
    m = _model(
        param_dtype=Dtype.BF16,
        attn_weight_dtype=Dtype.FP8_E4M3,        # explicit
    )
    assert m.attn_weight_dtype is Dtype.FP8_E4M3


# ── expected_input_dtype: multi-input quirks ──────────────────────────────


def test_residual_add_pulls_to_residual_dtype():
    m = _model(
        act_dtype=Dtype.BF16,
        moe_act_dtype=Dtype.FP8_E4M3,
        routed_expert_compute_dtype=Dtype.FP4,
    )
    # expert_agg is the residual-add boundary
    add_op = _op("L0.expert_agg", kind="add", component="routed_expert")
    assert expected_input_dtype(add_op, ti=0, model=m) is Dtype.BF16
    assert expected_input_dtype(add_op, ti=1, model=m) is Dtype.BF16


def test_non_residual_add_uses_in_act():
    m = _model(act_dtype=Dtype.BF16)
    add_op = _op("L0.hc_post.add", kind="add", component=None)
    assert expected_input_dtype(add_op, ti=0, model=m) is Dtype.BF16


def test_dispatch_combine_uses_moe_act_dtype():
    m = _model(
        act_dtype=Dtype.BF16,
        moe_act_dtype=Dtype.FP8_E4M3,
    )
    dispatch = _op("L0.dispatch", kind="dispatch", component="routed_expert")
    combine = _op("L0.combine", kind="combine", component="routed_expert")
    assert expected_input_dtype(dispatch, ti=0, model=m) is Dtype.FP8_E4M3
    assert expected_input_dtype(combine, ti=0, model=m) is Dtype.FP8_E4M3


def test_matmul_input_uses_in_act():
    m = _model(act_dtype=Dtype.BF16, moe_act_dtype=Dtype.FP8_E4M3)
    op = _op("L0.routed_expert_ffn", component="routed_expert")
    assert expected_input_dtype(op, ti=0, model=m) is Dtype.FP8_E4M3


# ── Cast op bundle pulls from meta ────────────────────────────────────────


def test_cast_op_bundle_reads_meta():
    m = _model()
    cast_op = _op(
        "L0.x_ln2.cast_to_fp8_e4m3", kind="cast", component="cast",
        src_dtype=Dtype.BF16, dst_dtype=Dtype.FP8_E4M3,
    )
    b = resolve_op_dtypes(cast_op, m)
    assert b.in_act is Dtype.BF16
    assert b.out_act is Dtype.FP8_E4M3
    assert b.compute is Dtype.FP8_E4M3      # for mfu_native peak selection


def test_cast_op_with_no_meta_falls_back():
    m = _model(act_dtype=Dtype.BF16)
    cast_op = _op("L0.cast", kind="cast", component="cast")
    b = resolve_op_dtypes(cast_op, m)
    # both default to act_dtype
    assert b.in_act is Dtype.BF16
    assert b.out_act is Dtype.BF16


# ── Bundle is hashable & frozen ───────────────────────────────────────────


def test_bundle_is_frozen():
    b = OpDtypeBundle(
        in_act=Dtype.BF16, weight=Dtype.BF16, out_act=Dtype.BF16,
        compute=Dtype.BF16, grad_in=Dtype.BF16, grad_weight=Dtype.BF16,
        grad_act=Dtype.BF16,
    )
    with pytest.raises(Exception):
        b.in_act = Dtype.FP8_E4M3
