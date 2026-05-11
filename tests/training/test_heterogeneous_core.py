"""Tests for heterogeneous-core (Cube/Vector) compute time modeling.

Validates:
1. OpCost cube/vector FLOPs split per op kind
2. Heterogeneous timing formula correctness
3. Backward compatibility (homogeneous hardware produces identical results)
4. Ascend 910B and H100 YAMLs have cube/vector data
"""
from __future__ import annotations

import math

import pytest

from zrt.hardware import load as hw_load
from zrt.hardware.spec import ComputeSpec
from zrt.training.compose.stage import op_to_time, op_to_time_hetero
from zrt.training.io.perf_tables import achieved_flops_efficiency
from zrt.training.ir.training_graph import Op
from zrt.training.models.flops import OpCost, op_cost
from zrt.training.spec.dtype import Dtype
from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.spec.system import GPU, SystemSpec


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_system(
    flops_bf16=320.0,
    cube_tflops=None,
    vector_tflops=None,
    overlap_ratio=None,
) -> SystemSpec:
    gpu = GPU(
        name="TestGPU",
        flops_bf16=flops_bf16,
        flops_fp8=flops_bf16 * 2,
        hbm_gb=64,
        hbm_bw_gbps=1600,
        cube_tflops=cube_tflops,
        vector_tflops=vector_tflops,
        overlap_ratio=overlap_ratio or {},
    )
    return SystemSpec(
        gpu=gpu,
        host_mem_gb=256,
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="HCCS", bandwidth_gbps=400, latency_us=3, topology="full_mesh", num_devices=1),
            inter_node=LinkSpec(type="RoCE", bandwidth_gbps=100, latency_us=10, topology="fat_tree"),
        ),
        nodes=1,
        gpus_per_node=1,
    )


def _make_910b_system() -> SystemSpec:
    return _make_system(
        flops_bf16=320,
        cube_tflops=280,
        vector_tflops=40,
        overlap_ratio={"attn_core": 0.6},
    )


# ── 1. YAML + ComputeSpec ───────────────────────────────────────────────────

class TestHardwareYAML:
    def test_ascend_910b_has_hetero_fields(self):
        hw = hw_load("ascend_910b")
        assert hw.compute.cube_bf16_tflops is not None
        assert hw.compute.vector_bf16_tflops is not None
        assert hw.compute.cube_bf16_tflops > 0
        assert hw.compute.vector_bf16_tflops > 0
        assert "attn_core" in hw.compute.overlap_ratio

    def test_ascend_910c_has_hetero_fields(self):
        hw = hw_load("ascend_910c")
        assert hw.compute.cube_bf16_tflops is not None
        assert hw.compute.vector_bf16_tflops is not None
        assert hw.compute.cube_bf16_tflops > 0
        assert hw.compute.vector_bf16_tflops > 0

    def test_h100_has_hetero_fields(self):
        hw = hw_load("nvidia_h100_sxm")
        assert hw.compute.cube_bf16_tflops == 989
        assert hw.compute.vector_bf16_tflops == 66.9
        assert "attn_core" in hw.compute.overlap_ratio

    def test_compute_spec_defaults(self):
        c = ComputeSpec()
        assert c.cube_bf16_tflops is None
        assert c.vector_bf16_tflops is None
        assert c.overlap_ratio == {}


# ── 2. OpCost cube/vector split ─────────────────────────────────────────────

class TestOpCostSplit:
    def _mock_model(self):
        from unittest.mock import MagicMock
        m = MagicMock()
        m.attn_compression_ratio = 1.0
        return m

    def test_matmul_pure_cube(self):
        op = Op(name="qkv_proj", kind="matmul", inputs=[], outputs=[],
                meta={"m": 4, "n": 4096, "k": 4096})
        cost = op_cost(op, self._mock_model())
        assert cost.fwd_cube_flops == cost.fwd_flops
        assert cost.fwd_vector_flops == 0.0
        assert cost.dx_cube_flops == cost.dx_flops
        assert cost.dx_vector_flops == 0.0

    def test_lm_head_pure_cube(self):
        op = Op(name="lm_head", kind="lm_head", inputs=[], outputs=[],
                meta={"m": 4, "n": 32000, "k": 4096})
        cost = op_cost(op, self._mock_model())
        assert cost.fwd_cube_flops == cost.fwd_flops
        assert cost.fwd_vector_flops == 0.0

    def test_attn_core_explicit_split(self):
        b, s, h, d = 2, 1024, 32, 128
        op = Op(name="attn", kind="attn_core", inputs=[], outputs=[],
                meta={"b": b, "s": s, "heads": h, "head_dim": d, "causal": True})
        cost = op_cost(op, self._mock_model())

        expected_cube = 4 * b * s * s * h * d  # dense QK+AV (Cube core always does full matmul)
        expected_vector = 5 * b * h * s * s

        assert cost.fwd_cube_flops == expected_cube
        assert cost.fwd_vector_flops == expected_vector
        assert cost.fwd_flops == 2 * b * s * s * h * d  # causal total for 6P accounting
        # dx uses 2.5x multiplier for both
        assert cost.dx_cube_flops == 2.5 * expected_cube
        assert cost.dx_vector_flops == 2.5 * expected_vector

    def test_elementwise_pure_vector(self):
        op = Op(name="layernorm", kind="ln", inputs=[], outputs=[],
                meta={"bytes_fwd": 8192})
        cost = op_cost(op, self._mock_model())
        assert cost.fwd_cube_flops == 0.0
        # fwd_vector_flops matches fwd_flops
        assert cost.fwd_vector_flops == cost.fwd_flops

    def test_unknown_op_pure_cube_fallback(self):
        op = Op(name="custom_op", kind="unknown_kind", inputs=[], outputs=[],
                meta={})
        cost = op_cost(op, self._mock_model())
        # Unknown ops return OpCost() with all zeros
        assert cost.fwd_flops == 0.0
        assert cost.fwd_cube_flops == 0.0
        assert cost.fwd_vector_flops == 0.0


# ── 3. Heterogeneous timing formula ─────────────────────────────────────────

class TestHeteroTiming:
    def test_pure_cube_op_uses_cube_peak(self):
        """Pure cube op: compute_us = cube_flops / (cube_tflops * eff)."""
        system = _make_910b_system()
        flops = 1e12
        eff = achieved_flops_efficiency("TestGPU", Dtype.BF16, flops)
        expected = flops / (280e12 * eff)

        result = op_to_time_hetero(flops, 0.0, 0.0, system, overlap_ratio=0.0)
        assert abs(result - expected) < 1e-15

    def test_pure_vector_op_uses_vector_peak(self):
        """Pure vector op: compute_us = vector_flops / (vector_tflops * eff)."""
        system = _make_910b_system()
        flops = 1e10
        eff = achieved_flops_efficiency("TestGPU", Dtype.BF16, flops)
        expected = flops / (40e12 * eff)

        result = op_to_time_hetero(0.0, flops, 0.0, system, overlap_ratio=0.0)
        assert abs(result - expected) < 1e-15

    def test_attn_core_hand_calculated(self):
        """Synthetic attn_core on 910B: verify compute_us within 1e-6."""
        b, s, h, d = 2, 1024, 32, 128
        cube_flops = 4 * b * s * s * h * d   # dense QK+AV = 34359738368
        vector_flops = 5 * b * h * s * s     # = 335544320

        system = _make_910b_system()
        total = cube_flops + vector_flops
        eff = achieved_flops_efficiency("TestGPU", Dtype.BF16, total)

        cube_t = cube_flops / (280e12 * eff)
        vector_t = vector_flops / (40e12 * eff)
        overlap_ratio = 0.6
        expected = max(cube_t, vector_t) + (1 - overlap_ratio) * min(cube_t, vector_t)

        result = op_to_time_hetero(cube_flops, vector_flops, 0.0, system,
                                    overlap_ratio=overlap_ratio)
        assert abs(result - expected) < 1e-18, f"got {result}, expected {expected}"

    def test_overlap_zero_is_serial_sum(self):
        """overlap_ratio=0: compute = max + min = cube_t + vector_t."""
        system = _make_910b_system()
        cube_flops = 1e12
        vector_flops = 1e10
        total = cube_flops + vector_flops
        eff = achieved_flops_efficiency("TestGPU", Dtype.BF16, total)

        cube_t = cube_flops / (280e12 * eff)
        vector_t = vector_flops / (40e12 * eff)
        expected = cube_t + vector_t  # max + 1.0 * min = sum

        result = op_to_time_hetero(cube_flops, vector_flops, 0.0, system,
                                    overlap_ratio=0.0)
        assert abs(result - expected) < 1e-18

    def test_overlap_one_is_max_only(self):
        """overlap_ratio=1: compute = max(cube_t, vector_t)."""
        system = _make_910b_system()
        cube_flops = 1e12
        vector_flops = 1e10
        total = cube_flops + vector_flops
        eff = achieved_flops_efficiency("TestGPU", Dtype.BF16, total)

        cube_t = cube_flops / (280e12 * eff)
        vector_t = vector_flops / (40e12 * eff)
        expected = max(cube_t, vector_t)

        result = op_to_time_hetero(cube_flops, vector_flops, 0.0, system,
                                    overlap_ratio=1.0)
        assert abs(result - expected) < 1e-18

    def test_zero_flops_gives_zero_time(self):
        system = _make_910b_system()
        result = op_to_time_hetero(0.0, 0.0, 0.0, system)
        assert result == 0.0


# ── 4. Backward compatibility ───────────────────────────────────────────────

class TestBackwardCompat:
    def test_homogeneous_uses_standard_path(self):
        """When cube_tflops is None, op_to_time produces identical results."""
        system = _make_system(flops_bf16=320, cube_tflops=None)

        flops = 1e12
        bytes_ = 1e9
        result = op_to_time(flops, bytes_, system)
        assert result > 0

        # Manually compute expected
        peak = 320e12
        eff = achieved_flops_efficiency("TestGPU", Dtype.BF16, flops)
        compute_t = flops / (peak * eff)
        bw = 1600e9
        eff_bw = 0.85  # bytes > 1e8
        memory_t = bytes_ / (bw * eff_bw)
        expected = max(compute_t, memory_t)
        assert abs(result - expected) < 1e-18

    def test_partial_hetero_config_falls_back_to_standard_path(self):
        """One missing hetero peak must not make that work class free."""
        system = _make_system(flops_bf16=320, cube_tflops=280, vector_tflops=None)

        cube_flops = 1e12
        vector_flops = 5e11
        bytes_ = 1e9

        result = op_to_time_hetero(cube_flops, vector_flops, bytes_, system)
        expected = op_to_time(cube_flops + vector_flops, bytes_, system)

        assert result == expected

    def test_hetero_produces_different_result_than_homogeneous(self):
        """On Ascend 910B with attn_core, hetero gives different compute_us."""
        b, s, h, d = 2, 1024, 32, 128
        cube_flops = 2 * b * s * s * h * d
        vector_flops = 5 * b * h * s * s

        # Homogeneous path (single peak)
        homo_system = _make_system(flops_bf16=320, cube_tflops=None)
        homo_time = op_to_time(cube_flops + vector_flops, 0.0, homo_system)

        # Heterogeneous path
        hetero_system = _make_910b_system()
        hetero_time = op_to_time_hetero(
            cube_flops, vector_flops, 0.0, hetero_system, overlap_ratio=0.6
        )

        # They must differ (different peak rates and overlap)
        assert hetero_time != homo_time
        # Hetero should be slower for pure-cube ops (280 < 320)
        # But with vector overlap it may differ — just verify non-zero delta
        assert abs(hetero_time - homo_time) > 0
