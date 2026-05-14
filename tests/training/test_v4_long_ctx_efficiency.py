"""Long-context efficiency regression tests for current training specs.

The DeepSeek-V4 paper claims inference-time ratios at 1M context:
  V4-Pro  single-token FLOPs / V3.2 ≈ 0.27
  V4-Pro  KV cache size / V3.2       ≈ 0.10
  V4-Flash single-token FLOPs / V3.2 ≈ 0.10
  V4-Flash KV cache size / V3.2      ≈ 0.07

This module intentionally does not claim those inference ratios are modeled.
The current training configs use effective training parameters and a simple
per-token KV estimate, so the tests pin that implemented behavior tightly.
When inference CSA/HCA KV compression is modeled, replace these expected
ratios with paper-aligned invariants.
"""

import pytest
from zrt.training.io.config_loader import load_specs


def _per_token_flops(model) -> float:
    """Approximate per-token FLOPs (forward only): 6 * P_eff * 1."""
    return 6.0 * model.effective_params_for_flops()


def _kv_cache_bytes_per_token(model) -> float:
    """KV cache bytes per token per layer (single KV head group).

    V3/V3.2 MLA: kv_lora_rank * (2 * dtype_bytes) — compressed KV
    V4 CSA:      kv_lora_rank / compression_ratio * dtype_bytes
    V4 HCA:      kv_lora_rank / hca_ratio * dtype_bytes
    """
    kv_rank = getattr(model, "kv_lora_rank", 0)
    if kv_rank == 0:
        # Standard MHA
        return 2 * model.num_kv_heads * model.head_dim * model.param_dtype.bytes

    dtype_bytes = model.param_dtype.bytes
    # MLA compressed KV: kv_lora_rank per token
    return kv_rank * dtype_bytes


def _kv_cache_total(model) -> float:
    """Total KV cache bytes per token across all layers."""
    per_layer = _kv_cache_bytes_per_token(model)
    return per_layer * len(model.layers)


@pytest.fixture
def v32():
    model, _, _ = load_specs(
        "python/zrt/training/configs/deepseek_v3_2_3d_h100.yaml"
    )
    return model


@pytest.fixture
def v4_pro():
    model, _, _ = load_specs(
        "python/zrt/training/configs/deepseek_v4_pro_3d_h100.yaml"
    )
    return model


@pytest.fixture
def v4_flash():
    model, _, _ = load_specs(
        "python/zrt/training/configs/deepseek_v4_flash_3d_h100.yaml"
    )
    return model


class TestV4ProLongCtxEfficiency:
    """V4-Pro vs V3.2 ratios under the current training-spec model."""

    def test_per_token_flops_ratio(self, v4_pro, v32):
        """V4-Pro per-token training FLOPs ratio is pinned to current specs."""
        ratio = _per_token_flops(v4_pro) / _per_token_flops(v32)
        assert ratio == pytest.approx(1.3450563709363128, rel=1e-6)

    def test_kv_cache_ratio(self, v4_pro, v32):
        """V4-Pro simple KV estimate ratio is pinned to current specs."""
        ratio = _kv_cache_total(v4_pro) / _kv_cache_total(v32)
        assert ratio == pytest.approx(2.0327868852459017, rel=1e-6)


class TestV4FlashLongCtxEfficiency:
    """V4-Flash vs V3.2 ratios under the current training-spec model."""

    def test_per_token_flops_ratio(self, v4_flash, v32):
        """V4-Flash per-token training FLOPs ratio is pinned to current specs."""
        ratio = _per_token_flops(v4_flash) / _per_token_flops(v32)
        assert ratio == pytest.approx(0.3744235509136079, rel=1e-6)

    def test_kv_cache_ratio(self, v4_flash, v32):
        """V4-Flash simple KV estimate ratio is pinned to current specs."""
        ratio = _kv_cache_total(v4_flash) / _kv_cache_total(v32)
        assert ratio == pytest.approx(1.4426229508196722, rel=1e-6)


class TestKVCacheBasic:
    """Basic KV cache sanity checks."""

    def test_v32_kv_cache_nonzero(self, v32):
        assert _kv_cache_total(v32) > 0

    def test_v4_pro_kv_cache_nonzero(self, v4_pro):
        assert _kv_cache_total(v4_pro) > 0

    def test_v4_flash_kv_cache_nonzero(self, v4_flash):
        assert _kv_cache_total(v4_flash) > 0
