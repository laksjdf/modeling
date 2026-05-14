"""Tests for performance efficiency curves in perf_tables.py.

These tests verify *physical correctness and calibration*, not implementation
details.  A test that merely checks ``result == 0.50`` is useless; the tests
below check that the curves encode accurate engineering knowledge.

Key calibration sources (cited in the module docstring):
  - NVIDIA MLPerf Training v4.0: BF16 ~452 TFLOPs/s/GPU on H100 (512 GPUs)
    → efficiency = 452 / 989 ≈ 0.457
  - NVIDIA DGXC LLaMA-3.1 70B: ~461 TFLOPs/s/GPU BF16 (1024 H100 SXM GPUs)
    → efficiency = 461 / 989 ≈ 0.466

Roofline ridge-point for H100 BF16:
  peak_compute / peak_bw = 989e12 / 3350e9 ≈ 295 FLOPs/byte
"""
from __future__ import annotations

import pytest

from zrt.training.io.perf_tables import achieved_bandwidth_efficiency, achieved_flops_efficiency
from zrt.training.spec.dtype import Dtype


# ── achieved_flops_efficiency ─────────────────────────────────────────────────

class TestFLOPsEfficiency:
    def test_zero_flops_returns_zero(self):
        """Zero and negative FLOPs: no work done, efficiency is zero by convention."""
        assert achieved_flops_efficiency("h100", Dtype.BF16, 0.0) == 0.0
        assert achieved_flops_efficiency("h100", Dtype.BF16, -1.0) == 0.0

    def test_efficiency_is_a_valid_fraction(self):
        """Efficiency must be in (0, 1]: it is a fraction of peak throughput."""
        for flops in (1e6, 1e9, 5e9, 1e10, 5e10, 1e11, 1e12, 1e15):
            eff = achieved_flops_efficiency("h100", Dtype.BF16, flops)
            assert 0.0 < eff <= 1.0, f"efficiency {eff:.3f} out of (0,1] for flops={flops:.0e}"

    def test_training_scale_efficiency_calibrated_to_mlperf(self):
        """At full-model training scale, achieved efficiency must match published benchmarks.

        Sources:
          - MLPerf Training v4.0: ~452 TFLOPs/s/GPU BF16 → eff = 452/989 ≈ 0.457
          - DGXC LLaMA-3.1 70B:   ~461 TFLOPs/s/GPU BF16 → eff = 461/989 ≈ 0.466

        Acceptable calibrated range: [0.40, 0.60].  A value of 0.85 (common
        single-kernel microbenchmark) would overestimate throughput by ~70%
        and make all step-time predictions systematically wrong.
        """
        large_training_flops = 1e15  # typical full-model forward pass
        eff = achieved_flops_efficiency("h100", Dtype.BF16, large_training_flops)
        assert 0.40 <= eff <= 0.60, (
            f"Training-scale efficiency {eff:.3f} is outside calibrated range [0.40, 0.60]. "
            "MLPerf v4.0 H100 BF16: 0.457; DGXC LLaMA-3.1 70B: 0.466."
        )

    def test_training_scale_not_microbenchmark(self):
        """Sustained training MFU ~50% — NOT the 0.85 microbenchmark peak.

        The 0.85 figure reflects single-kernel roofline; sustained training
        throughput is ~40% lower due to pipeline bubbles, optimizer steps, and
        gradient accumulation.  Using 0.85 would cause a systematic ~70% error
        in step-time predictions.
        """
        eff = achieved_flops_efficiency("h100", Dtype.BF16, 1e15)
        assert eff < 0.70, (
            f"Got {eff:.2f}, but training-scale MFU is ~0.50 on H100, "
            "not the 0.85 single-kernel microbenchmark value."
        )

    def test_medium_matmuls_more_efficient_than_tiny(self):
        """Medium-size matmuls saturate tensor cores better than tiny ops.

        A 1-MFLOPs op (e.g. a tiny bias add) is overhead-dominated;
        a 10-GFLOPs op (e.g. a large FFN layer) sustains higher utilisation.
        """
        eff_tiny = achieved_flops_efficiency("h100", Dtype.BF16, 1e6)
        eff_medium = achieved_flops_efficiency("h100", Dtype.BF16, 1e10)
        assert eff_medium >= eff_tiny

    def test_training_scale_efficiency_not_higher_than_medium_kernel(self):
        """At full-model training scale, efficiency is lower than single-kernel medium ops.

        Single large-kernel benchmarks (10 GFLOPs) show higher efficiency
        than full-step training (1 PFLOPs), because training includes
        pipeline bubbles, all-reduce overhead, and optimizer steps.
        The efficiency curve must encode this drop.
        """
        eff_kernel = achieved_flops_efficiency("h100", Dtype.BF16, 1e10)
        eff_training = achieved_flops_efficiency("h100", Dtype.BF16, 1e15)
        assert eff_training <= eff_kernel, (
            "Training-scale efficiency should not exceed single-kernel efficiency. "
            "Full-step training has overheads that reduce effective MFU."
        )


# ── achieved_bandwidth_efficiency ────────────────────────────────────────────

class TestBandwidthEfficiency:
    def test_zero_bytes_returns_zero(self):
        """Zero and negative byte counts: no transfer, efficiency is zero."""
        assert achieved_bandwidth_efficiency("h100", 0.0) == 0.0
        assert achieved_bandwidth_efficiency("h100", -1.0) == 0.0

    def test_efficiency_is_a_valid_fraction(self):
        """Bandwidth efficiency must be in (0, 1]."""
        for bytes_ in (1e3, 1e5, 1e6, 1e7, 1e8, 1e9, 1e12):
            eff = achieved_bandwidth_efficiency("h100", bytes_)
            assert 0.0 < eff <= 1.0, f"bandwidth efficiency {eff:.3f} out of (0,1] for bytes={bytes_:.0e}"

    def test_larger_transfers_achieve_higher_efficiency(self):
        """Larger HBM transfers amortise startup latency and approach peak bandwidth.

        Physical law: large contiguous transfers saturate the memory controller
        and achieve close to advertised peak bandwidth (e.g. 3350 GB/s on H100).
        Small transfers (< 1 MB) are penalised by cache-line overhead and
        controller startup cost, achieving lower utilisation (~40%).

        If this test fails it means small and large transfers have the same (or
        inverted) efficiency, which would cause memory-bound operators to have
        the same estimated latency regardless of tensor size — clearly wrong.
        """
        eff_10kb = achieved_bandwidth_efficiency("h100", 1e4)    # 10 KB
        eff_10mb = achieved_bandwidth_efficiency("h100", 1e7)    # 10 MB
        eff_1gb = achieved_bandwidth_efficiency("h100", 1e9)     # 1 GB

        assert eff_10kb < eff_10mb, "10 MB transfer should be more efficient than 10 KB"
        assert eff_10mb < eff_1gb,  "1 GB transfer should be more efficient than 10 MB"

    def test_large_transfer_approaches_peak_bandwidth(self):
        """Transfers > 100 MB should achieve ≥ 75% of peak HBM bandwidth.

        H100 advertised peak: 3350 GB/s.  In practice, large sustained transfers
        achieve 80–90% of this.  A value below 0.75 would indicate over-penalising
        large transfers, causing memory-bound operators to appear slower than reality.
        """
        eff = achieved_bandwidth_efficiency("h100", 1e9)
        assert eff >= 0.75, (
            f"Got bandwidth efficiency {eff:.2f} for large (1 GB) transfer; "
            "expected ≥ 0.75 for sustained large HBM transfers."
        )


# ── Roofline correctness via op_to_time ──────────────────────────────────────
#
# op_to_time uses both efficiency tables together.  These tests verify that
# the roofline model correctly identifies the dominant bottleneck, which is
# essential for predicting whether an operator is compute-bound or
# memory-bandwidth-bound.

class TestRooflineDiscrimination:
    """op_to_time ridge-point discrimination on H100 BF16.

    H100 ridge point ≈ 989e12 / 3350e9 ≈ 295 FLOPs/byte.
    At arithmetic intensity >> 295: compute-bound (time ~ FLOPs / peak_compute).
    At arithmetic intensity << 295: memory-bound (time ~ bytes / peak_bw).
    """

    def _make_h100_system(self):
        from zrt.hardware.spec import InterconnectSpec, LinkSpec
        from zrt.training.spec.system import GPU, SystemSpec
        return SystemSpec(
            gpu=GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80, hbm_bw_gbps=3350),
            host_mem_gb=256,
            interconnect=InterconnectSpec(
                intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0,
                                   topology="all_to_all", num_devices=8),
                inter_node=LinkSpec(type="IB", bandwidth_gbps=400, latency_us=5.0,
                                   topology="fat_tree"),
            ),
            nodes=1, gpus_per_node=8,
        )

    def test_compute_bound_insensitive_to_memory_increase(self):
        """At very high arithmetic intensity, time is determined by FLOPs, not bytes.

        Op: 1 PFLOPs, 1 MB → AI = 1e9 FLOPs/byte >> 295 (H100 ridge).
        Multiplying bytes by 10× should barely change the result (<5%)
        because memory time is negligible compared to compute time.

        If this fails, the roofline model is not correctly identifying the
        compute-bound regime, and all large matmul estimates are unreliable.
        """
        from zrt.training.compose.stage import op_to_time
        system = self._make_h100_system()
        t_base = op_to_time(1e15, 1e6, system)
        t_10x_bytes = op_to_time(1e15, 1e7, system)
        assert t_base > 0
        assert abs(t_10x_bytes - t_base) / t_base < 0.05, (
            "Compute-bound op: 10× more bytes should change time by < 5%"
        )

    def test_memory_bound_insensitive_to_flops_increase(self):
        """At very low arithmetic intensity, time is determined by bytes, not FLOPs.

        Op: 1 GFLOPs, 1 GB → AI = 1 FLOPs/byte << 295 (H100 ridge).
        Multiplying FLOPs by 10× should barely change the result (<5%)
        because compute time is negligible compared to memory time.

        If this fails, memory-bound ops (elementwise, LayerNorm, etc.) will be
        assigned compute-limited latencies — a significant accuracy error.
        """
        from zrt.training.compose.stage import op_to_time
        system = self._make_h100_system()
        t_base = op_to_time(1e9, 1e9, system)
        t_10x_flops = op_to_time(1e10, 1e9, system)
        assert t_base > 0
        assert abs(t_10x_flops - t_base) / t_base < 0.05, (
            "Memory-bound op: 10× more FLOPs should change time by < 5%"
        )

    def test_zero_work_gives_zero_time(self):
        """An op with zero FLOPs and zero bytes takes no time."""
        from zrt.training.compose.stage import op_to_time
        system = self._make_h100_system()
        assert op_to_time(0.0, 0.0, system) == 0.0

    def test_single_layer_matmul_in_physically_reasonable_range(self):
        """1 TFLOPs on H100 BF16 should complete in a physically plausible time.

        H100 BF16 peak: 989 TFLOPS = 989e12 FLOPS/s.
        At efficiency=1.0 (theoretical): 1e12 / 989e12 ≈ 1.01 ms (lower bound).
        At efficiency=0.3 (very conservative): ~3.4 ms (upper bound).
        Outside [0.5 ms, 20 ms] = [5e-4 s, 2e-2 s] indicates a unit or scaling bug.

        A common mistake: omitting the ×1e12 factor for TFLOPS, which would
        produce a 1e12× larger time value (~1e9 seconds) — clearly wrong.
        """
        from zrt.training.compose.stage import op_to_time
        system = self._make_h100_system()
        t_s = op_to_time(1e12, 1e9, system)   # 1 TFLOPs, 1 GB
        t_ms = t_s * 1000
        assert 0.5 < t_ms < 20, (
            f"1 TFLOPs on H100 gave {t_ms:.3f} ms; expected 0.5–20 ms. "
            "Likely a unit/scaling bug (e.g. missing ×1e12 for TFLOPS)."
        )
