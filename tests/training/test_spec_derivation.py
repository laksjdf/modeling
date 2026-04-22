"""Test spec derivation — param count matches known model cards."""

import pytest
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import ModelSpec, LayerKind


def test_dtype_bytes():
    assert Dtype.FP32.bytes == 4
    assert Dtype.BF16.bytes == 2
    assert Dtype.FP16.bytes == 2
    assert Dtype.FP8.bytes == 1


def test_llama3_70b_param_count():
    """Llama-3 70B should have ~70.6B parameters."""
    model = ModelSpec(
        hidden=8192, ffn=28672, num_heads=64, num_kv_heads=8,
        head_dim=128, vocab=128256, seq_len=4096,
        layers=[LayerKind.DENSE] * 80,
    )
    params = model.total_params()
    assert 70e9 < params < 71e9, f"Expected ~70.6B, got {params/1e9:.2f}B"


def test_llama3_70b_per_layer():
    """Each dense layer should have reasonable param count."""
    model = ModelSpec(
        hidden=8192, ffn=28672, num_heads=64, num_kv_heads=8,
        head_dim=128, vocab=128256, seq_len=4096,
        layers=[LayerKind.DENSE] * 80,
    )
    per_layer = model.params_per_dense_layer()
    assert per_layer > 0
    # QKV + O_proj + FFN(up+gate+down) + 2*LN
    # Should be roughly 800M-900M for Llama-3 70B
    assert 800e6 < per_layer < 1e9, f"Expected ~856M, got {per_layer/1e6:.1f}M"


def test_gpt3_175b_param_count():
    """GPT-3 175B: hidden=12288, ffn=49152, 96 layers, 96 heads.

    Note: our dense block assumes SwiGLU (3 projections: up+gate+down),
    while GPT-3 uses standard FFN (2 projections: up+down). So our
    param count will be higher by ~num_layers * hidden * ffn.
    With SwiGLU: ~233B instead of ~175B.
    """
    model = ModelSpec(
        hidden=12288, ffn=49152, num_heads=96, num_kv_heads=96,
        head_dim=128, vocab=50257, seq_len=2048,
        layers=[LayerKind.DENSE] * 96,
    )
    params = model.total_params()
    # SwiGLU model: ~233B (standard would be ~175B)
    assert 230e9 < params < 240e9, f"Expected ~233B (SwiGLU), got {params/1e9:.2f}B"


def test_small_model_sanity():
    """Small model: 2 layers, hidden=512, should have reasonable params."""
    model = ModelSpec(
        hidden=512, ffn=2048, num_heads=8, num_kv_heads=8,
        head_dim=64, vocab=32000, seq_len=512,
        layers=[LayerKind.DENSE] * 2,
    )
    params = model.total_params()
    assert params > 0
    # Embedding alone: 32000 * 512 = ~16M
    assert params > 30e6, f"Too few params: {params/1e6:.1f}M"


def test_head_dim_total():
    model = ModelSpec(
        hidden=8192, ffn=28672, num_heads=64, num_kv_heads=8,
        head_dim=128, vocab=128256, seq_len=4096,
        layers=[LayerKind.DENSE] * 80,
    )
    assert model.head_dim_total == 64 * 128  # 8192
    assert model.kv_dim == 8 * 128  # 1024
