"""Tests for mixed-quant memory accounting."""
import pytest

from zrt.training.models.memory import memory_breakdown
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import OptKind, Strategy
from zrt.training.spec.system import GPU, SystemSpec
from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.ir.training_graph import Graph


def _make_system():
    gpu = GPU(name="h100", flops_bf16=989, flops_fp8=3958, flops_fp4=0,
              hbm_gb=80, hbm_bw_gbps=3350)
    link = LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1,
                    topology="all_to_all", num_devices=8)
    return SystemSpec(gpu=gpu, host_mem_gb=2048,
                      interconnect=InterconnectSpec(intra_node=link, inter_node=link),
                      nodes=1, gpus_per_node=8)


def _moe_model(**kwargs):
    base = dict(
        hidden=512, ffn=2048, num_heads=8, num_kv_heads=8, head_dim=64,
        vocab=4096, seq_len=128,
        layers=[LayerKind.MOE, LayerKind.MOE],
        num_experts=8, moe_ffn=1024, top_k=2,
    )
    base.update(kwargs)
    return ModelSpec(**base)


def test_legacy_string_fp4_and_new_enum_produce_same_weight_bytes():
    """Back-compat: routed_expert_dtype='fp4' must match routed_expert_weight_dtype=Dtype.FP4."""
    g, sys_, st = Graph(), _make_system(), Strategy(optimizer=OptKind.ADAM)
    m_legacy = _moe_model(routed_expert_dtype="fp4")
    m_new    = _moe_model(routed_expert_weight_dtype=Dtype.FP4)
    mb_legacy = memory_breakdown(g, m_legacy, sys_, st)
    mb_new    = memory_breakdown(g, m_new, sys_, st)
    assert mb_legacy.weights == mb_new.weights


def test_fp4_routed_expert_smaller_than_bf16():
    """FP4 routed expert weight should be ~3.5× smaller than BF16."""
    g, sys_, st = Graph(), _make_system(), Strategy(optimizer=OptKind.ADAM)
    m_bf16 = _moe_model()
    m_fp4 = _moe_model(routed_expert_weight_dtype=Dtype.FP4)
    mb_bf16 = memory_breakdown(g, m_bf16, sys_, st)
    mb_fp4 = memory_breakdown(g, m_fp4, sys_, st)
    # FP4 stored 0.5625 vs BF16 2.0 → expert weight ratio ≈ 0.281, but
    # non-expert params (attn/embed) keep BF16 weight bytes, so total
    # ratio is somewhere between 0.281 and 1.0.
    assert mb_fp4.weights < mb_bf16.weights
    # Expert weight is a substantial fraction of total → expect ≥ 20% saving overall
    saving = (mb_bf16.weights - mb_fp4.weights) / mb_bf16.weights
    assert saving > 0.2, f"FP4 should save >20% weight memory, got {saving:.2%}"


def test_fp8_routed_expert_weight_halves_routed_bytes_vs_bf16():
    g, sys_, st = Graph(), _make_system(), Strategy(optimizer=OptKind.ADAM)
    m_bf16 = _moe_model()
    m_fp8 = _moe_model(routed_expert_weight_dtype=Dtype.FP8_E4M3)
    mb_bf16 = memory_breakdown(g, m_bf16, sys_, st)
    mb_fp8 = memory_breakdown(g, m_fp8, sys_, st)
    assert mb_fp8.weights < mb_bf16.weights


def test_dense_model_unaffected_by_routed_dtype():
    """Dense model (no MoE layers) → routed_expert_weight_dtype has no effect."""
    g, sys_, st = Graph(), _make_system(), Strategy(optimizer=OptKind.ADAM)
    base = dict(
        hidden=512, ffn=2048, num_heads=8, num_kv_heads=8, head_dim=64,
        vocab=4096, seq_len=128, layers=[LayerKind.DENSE, LayerKind.DENSE],
    )
    m_bf16 = ModelSpec(**base)
    m_fp4 = ModelSpec(**base, routed_expert_weight_dtype=Dtype.FP4)
    mb_bf16 = memory_breakdown(g, m_bf16, sys_, st)
    mb_fp4 = memory_breakdown(g, m_fp4, sys_, st)
    assert mb_bf16.weights == mb_fp4.weights


def test_ep_a2a_buffer_uses_routed_compute_dtype():
    """EP A2A staging buffer should scale with routed_expert_compute_dtype."""
    g = Graph()
    sys_ = _make_system()
    st = Strategy(ep=4, dp=1, optimizer=OptKind.ADAM)
    m_bf16 = _moe_model()
    m_fp8 = _moe_model(routed_expert_compute_dtype=Dtype.FP8_E4M3)
    mb_bf16 = memory_breakdown(g, m_bf16, sys_, st)
    mb_fp8  = memory_breakdown(g, m_fp8,  sys_, st)
    # Comm buffers shrink when routed compute dtype shrinks
    # (4× staging × seq_cp × hidden × act_bytes × n_moe).
    assert mb_fp8.comm_buffers < mb_bf16.comm_buffers


def test_cp_a2a_buffer_uses_attn_act_dtype():
    g = Graph()
    sys_ = _make_system()
    st = Strategy(cp=4, dp=1, optimizer=OptKind.ADAM)
    m_bf16 = _moe_model()
    m_fp8_attn = _moe_model(attn_act_dtype=Dtype.FP8_E4M3)
    mb_bf16 = memory_breakdown(g, m_bf16, sys_, st)
    mb_fp8  = memory_breakdown(g, m_fp8_attn, sys_, st)
    assert mb_fp8.comm_buffers < mb_bf16.comm_buffers


def test_qk_score_matrix_uses_attn_act_dtype():
    """Activations: QK^T score matrix term (~5·a·s²·bytes) should
    scale with attn_act_dtype when present."""
    g = Graph()
    sys_ = _make_system()
    st = Strategy(optimizer=OptKind.ADAM)
    # Force long sequence so QK^T dominates
    m_bf16 = _moe_model(seq_len=1024, num_heads=16)
    m_fp8 = _moe_model(seq_len=1024, num_heads=16, attn_act_dtype=Dtype.FP8_E4M3)
    mb_bf16 = memory_breakdown(g, m_bf16, sys_, st)
    mb_fp8 = memory_breakdown(g, m_fp8, sys_, st)
    assert mb_fp8.activations < mb_bf16.activations


def test_routed_expert_grad_dtype_affects_grad_bytes_in_moe_models():
    g = Graph()
    sys_ = _make_system()
    st = Strategy(optimizer=OptKind.ADAM)
    m_fp32 = _moe_model()
    m_bf16 = _moe_model(routed_expert_grad_dtype=Dtype.BF16)
    mb_fp32 = memory_breakdown(g, m_fp32, sys_, st)
    mb_bf16 = memory_breakdown(g, m_bf16, sys_, st)
    # BF16 expert grad smaller than FP32 expert grad → total grads smaller
    assert mb_bf16.grads < mb_fp32.grads


def test_dense_model_grad_unaffected_by_routed_expert_grad_dtype():
    g, sys_, st = Graph(), _make_system(), Strategy(optimizer=OptKind.ADAM)
    base = dict(hidden=512, ffn=2048, num_heads=8, num_kv_heads=8, head_dim=64,
                vocab=4096, seq_len=128, layers=[LayerKind.DENSE, LayerKind.DENSE])
    m_fp32 = ModelSpec(**base)
    m_bf16 = ModelSpec(**base, routed_expert_grad_dtype=Dtype.BF16)
    mb_fp32 = memory_breakdown(g, m_fp32, sys_, st)
    mb_bf16 = memory_breakdown(g, m_bf16, sys_, st)
    assert mb_fp32.grads == mb_bf16.grads
