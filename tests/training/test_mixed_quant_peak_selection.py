"""Tests for peak TFLOPS routing by dtype."""
import pytest

from zrt.training.io.perf_tables import peak_tflops_for
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.system import GPU


def _gpu(name="h100", *, bf16=989.0, fp8=3958.0, fp4=0.0):
    return GPU(name=name, flops_bf16=bf16, flops_fp8=fp8, flops_fp4=fp4,
               hbm_gb=80.0, hbm_bw_gbps=3350.0)


def test_bf16_returns_bf16_peak():
    gpu = _gpu()
    assert peak_tflops_for(gpu, Dtype.BF16) == pytest.approx(989.0e12)


def test_fp16_falls_back_to_bf16_peak():
    gpu = _gpu()
    assert peak_tflops_for(gpu, Dtype.FP16) == pytest.approx(989.0e12)


def test_fp8_e4m3_returns_fp8_peak():
    gpu = _gpu()
    assert peak_tflops_for(gpu, Dtype.FP8_E4M3) == pytest.approx(3958.0e12)


def test_fp8_e5m2_returns_fp8_peak():
    gpu = _gpu()
    assert peak_tflops_for(gpu, Dtype.FP8_E5M2) == pytest.approx(3958.0e12)


def test_fp4_returns_fp4_peak_when_supported():
    gpu = _gpu(fp4=30000.0)
    assert peak_tflops_for(gpu, Dtype.FP4) == pytest.approx(30000.0e12)


def test_fp4_falls_back_to_fp8_when_unsupported():
    # H100 has fp4=0 → fallback to fp8.
    gpu = _gpu(fp4=0.0)
    assert peak_tflops_for(gpu, Dtype.FP4) == pytest.approx(3958.0e12)


def test_fp8_falls_back_to_bf16_when_unsupported():
    # A100-like: no FP8 hardware
    gpu = _gpu(fp8=0.0, fp4=0.0)
    assert peak_tflops_for(gpu, Dtype.FP8_E4M3) == pytest.approx(989.0e12)


def test_b300_yaml_loads_with_fp4_tops():
    """B300 spec declares native FP4."""
    from zrt.hardware.registry import load
    hw = load("nvidia_b300")
    assert hw.compute.fp4_tops > 0
    assert hw.compute.fp8_tops > hw.compute.bf16_tflops  # FP8 >= 2x BF16


def test_h100_yaml_declares_fp4_tops_zero():
    """H100 lacks native FP4 hardware -> fp4_tops must be 0 in spec."""
    from zrt.hardware.registry import load
    hw = load("nvidia_h100_sxm")
    assert hw.compute.fp4_tops == 0.0


def test_op_to_time_fp8_is_faster_than_bf16_on_h100():
    """On H100 (BF16=989, FP8=3958), FP8 compute should be ~4× faster."""
    from zrt.training.compose.stage import op_to_time
    from zrt.training.spec.system import SystemSpec
    from zrt.hardware.spec import InterconnectSpec, LinkSpec

    gpu = _gpu(name="H100", bf16=989.0, fp8=3958.0, fp4=0.0)
    link = LinkSpec(type="NVLink4", bandwidth_gbps=900, latency_us=1,
                    topology="all_to_all", num_devices=8)
    system = SystemSpec(gpu=gpu, host_mem_gb=2048,
                        interconnect=InterconnectSpec(intra_node=link, inter_node=link),
                        nodes=1, gpus_per_node=8)
    flops = 1e12  # 1 TFLOP of compute
    bytes_ = 0    # ignore memory bound
    t_bf16 = op_to_time(flops, bytes_, system, gpu.name, Dtype.BF16)
    t_fp8  = op_to_time(flops, bytes_, system, gpu.name, Dtype.FP8_E4M3)
    # FP8 peak is 4× BF16 → t_fp8 should be roughly t_bf16 / 4 (efficiency
    # curve is currently dtype-blind so ratio = peak ratio).
    assert t_fp8 < t_bf16
    assert t_fp8 == pytest.approx(t_bf16 * 989.0 / 3958.0, rel=0.05)
