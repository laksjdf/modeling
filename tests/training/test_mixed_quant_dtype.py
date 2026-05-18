"""Tests for Dtype enum extension (FP4, FP8 subtypes)."""
import pytest

from zrt.training.spec.dtype import Dtype


def test_dtype_legacy_values_preserved():
    """Existing Dtype values must still parse and have correct .bytes."""
    assert Dtype.FP32.bytes == 4.0
    assert Dtype.BF16.bytes == 2.0
    assert Dtype.FP16.bytes == 2.0
    assert Dtype.FP8.bytes == 1.0


def test_dtype_new_fp8_subtypes():
    assert Dtype.FP8_E4M3.bytes == 1.0
    assert Dtype.FP8_E5M2.bytes == 1.0


def test_dtype_fp4_byte_size_is_half():
    assert Dtype.FP4.bytes == 0.5


def test_dtype_fp4_block_overhead():
    """FP4 uses block=32 with one BF16 (2B) scale per block → 2/32 = 0.0625 B/elem."""
    assert Dtype.FP4.block_overhead_bytes_per_elem == pytest.approx(0.0625)
    # Non-FP4 dtypes have no block overhead.
    assert Dtype.BF16.block_overhead_bytes_per_elem == 0.0
    assert Dtype.FP8_E4M3.block_overhead_bytes_per_elem == 0.0


def test_dtype_stored_bytes_includes_overhead():
    assert Dtype.FP4.stored_bytes == pytest.approx(0.5625)
    assert Dtype.BF16.stored_bytes == 2.0
    assert Dtype.FP32.stored_bytes == 4.0


def test_dtype_fp8_alias_is_e4m3():
    """Existing code/YAML using FP8 must map to FP8_E4M3 (DeepSeek-V4 forward GEMM)."""
    assert Dtype.FP8 is Dtype.FP8_E4M3
