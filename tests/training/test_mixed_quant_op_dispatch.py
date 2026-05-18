"""Tests for Op.component tagging + compute-dtype dispatch."""
import pytest

from zrt.training.compose.stage import _resolve_compute_dtype
from zrt.training.ir.training_graph import Op
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec


def _model(**dtype_kwargs):
    return ModelSpec(
        hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
        vocab=1000, seq_len=64, layers=[LayerKind.MOE], **dtype_kwargs,
    )


def _op(name, kind="matmul", component=None):
    return Op(name=name, kind=kind, component=component)


def test_attention_op_uses_attn_compute_dtype():
    m = _model(attn_compute_dtype=Dtype.FP8_E4M3)
    op = _op("layer0.qkv_proj", component="attention")
    assert _resolve_compute_dtype(op, m) is Dtype.FP8_E4M3


def test_routed_expert_op_uses_routed_compute_dtype():
    m = _model(routed_expert_compute_dtype=Dtype.FP8_E4M3)
    op = _op("layer0.routed_expert_ffn", component="routed_expert")
    assert _resolve_compute_dtype(op, m) is Dtype.FP8_E4M3


def test_shared_expert_op_uses_shared_dtype():
    m = _model(shared_expert_compute_dtype=Dtype.FP8_E4M3)
    op = _op("layer0.shared_up_proj", component="shared_expert")
    assert _resolve_compute_dtype(op, m) is Dtype.FP8_E4M3


def test_embedding_op_forced_bf16():
    m = _model(act_dtype=Dtype.FP8_E4M3)
    op = _op("embed", kind="embed", component="embedding")
    assert _resolve_compute_dtype(op, m) is Dtype.BF16


def test_norm_op_forced_bf16():
    m = _model()
    op = _op("layer0.ln", kind="ln", component="norm")
    assert _resolve_compute_dtype(op, m) is Dtype.BF16


def test_unset_component_falls_back_to_act_dtype():
    m = _model(act_dtype=Dtype.FP16)
    op = _op("anonymous", component=None)
    assert _resolve_compute_dtype(op, m) is Dtype.FP16


def test_op_dataclass_has_component_field_default_none():
    op = Op(name="x", kind="matmul")
    assert op.component is None


def test_op_dataclass_accepts_component_kwarg():
    op = Op(name="x", kind="matmul", component="attention")
    assert op.component == "attention"
