"""Tests for YAML quant_preset expansion + extended _parse_dtype."""
import pytest

from zrt.training.io.config_loader import _expand_quant_preset, _parse_dtype
from zrt.training.spec.dtype import Dtype


def test_parse_dtype_accepts_fp8_e4m3():
    assert _parse_dtype("fp8_e4m3") is Dtype.FP8_E4M3


def test_parse_dtype_accepts_fp8_e5m2():
    assert _parse_dtype("fp8_e5m2") is Dtype.FP8_E5M2


def test_parse_dtype_accepts_fp4():
    assert _parse_dtype("fp4") is Dtype.FP4


def test_parse_dtype_fp8_alias_still_works():
    """Legacy YAML using 'fp8' → FP8_E4M3 (alias)."""
    assert _parse_dtype("fp8") is Dtype.FP8_E4M3


def test_expand_preset_deepseek_v4_fp8_fp4():
    d = {"quant_preset": "deepseek_v4_fp8_fp4"}
    out = _expand_quant_preset(d)
    assert "quant_preset" not in out
    assert out["routed_expert_compute_dtype"] == "fp8_e4m3"
    assert out["routed_expert_weight_dtype"] == "fp4"
    assert out["attn_compute_dtype"] == "bf16"
    assert out["moe_act_dtype"] == "fp8_e4m3"


def test_expand_preset_explicit_field_wins():
    """Explicit dtype in YAML overrides preset value."""
    d = {
        "quant_preset": "deepseek_v4_fp8_fp4",
        "attn_compute_dtype": "fp8_e4m3",   # override
    }
    out = _expand_quant_preset(d)
    assert out["attn_compute_dtype"] == "fp8_e4m3"
    # other preset values still applied
    assert out["routed_expert_weight_dtype"] == "fp4"


def test_expand_preset_noop_when_absent():
    d = {"param_dtype": "bf16"}
    out = _expand_quant_preset(d)
    assert out == {"param_dtype": "bf16"}


def test_expand_preset_unknown_raises():
    with pytest.raises(KeyError, match="unknown.*quant_preset"):
        _expand_quant_preset({"quant_preset": "nonsense_preset"})


def test_preset_bf16_baseline_is_pure_bf16():
    d = {"quant_preset": "bf16_baseline"}
    out = _expand_quant_preset(d)
    assert out["routed_expert_compute_dtype"] == "bf16"
    assert out["routed_expert_weight_dtype"] == "bf16"
    assert out["attn_compute_dtype"] == "bf16"
