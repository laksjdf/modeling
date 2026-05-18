"""Tests for ModelSpec dtype field extension."""
import pytest

from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec


def _minimal_kwargs(**overrides):
    base = dict(
        hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
        vocab=1000, seq_len=64, layers=[LayerKind.DENSE],
    )
    base.update(overrides)
    return base


def test_default_per_component_dtypes_are_bf16():
    m = ModelSpec(**_minimal_kwargs())
    assert m.attn_compute_dtype is Dtype.BF16
    assert m.routed_expert_compute_dtype is Dtype.BF16
    assert m.shared_expert_compute_dtype is Dtype.BF16
    assert m.routed_expert_weight_dtype is Dtype.BF16
    assert m.routed_expert_grad_dtype is Dtype.FP32


def test_per_region_act_dtypes_default_to_none():
    m = ModelSpec(**_minimal_kwargs())
    assert m.attn_act_dtype is None
    assert m.moe_act_dtype is None


def test_effective_attn_act_dtype_falls_back_to_act_dtype():
    m = ModelSpec(**_minimal_kwargs(act_dtype=Dtype.FP16))
    assert m.effective_attn_act_dtype() is Dtype.FP16
    # Explicit override wins
    m2 = ModelSpec(**_minimal_kwargs(act_dtype=Dtype.FP16, attn_act_dtype=Dtype.BF16))
    assert m2.effective_attn_act_dtype() is Dtype.BF16


def test_effective_moe_act_dtype_falls_back_to_routed_compute():
    """When moe_act_dtype unset, default to routed_expert_compute_dtype."""
    m = ModelSpec(**_minimal_kwargs(routed_expert_compute_dtype=Dtype.FP8_E4M3))
    assert m.effective_moe_act_dtype() is Dtype.FP8_E4M3
    # Explicit override wins
    m2 = ModelSpec(**_minimal_kwargs(
        routed_expert_compute_dtype=Dtype.FP8_E4M3,
        moe_act_dtype=Dtype.BF16,
    ))
    assert m2.effective_moe_act_dtype() is Dtype.BF16


def test_legacy_routed_expert_dtype_string_fp4_syncs_to_weight_dtype():
    """For back-compat: routed_expert_dtype='fp4' should populate
    routed_expert_weight_dtype when the new field is at its default."""
    m = ModelSpec(**_minimal_kwargs(routed_expert_dtype="fp4"))
    assert m.routed_expert_weight_dtype is Dtype.FP4


def test_explicit_routed_expert_weight_dtype_wins_over_legacy_string():
    m = ModelSpec(**_minimal_kwargs(
        routed_expert_dtype="fp4",
        routed_expert_weight_dtype=Dtype.BF16,
    ))
    assert m.routed_expert_weight_dtype is Dtype.BF16
