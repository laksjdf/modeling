"""Tests for EP wave-overlap savings in compose/stage.py.

_wave_overlap_saved() models how much EP all-to-all communication time can be
hidden by overlapping with expert GEMM computation.

The model:
  - Split both comm and GEMM into K equal waves
  - Wave 0: comm is fully exposed (no prior GEMM to overlap with)
  - Waves 1..K-1: comm can be hidden behind the previous wave's GEMM
  - Per wave: exposed = max(comm_per_wave - gemm_per_wave, 0)
  - saved = comm_time - exposed_total

Why this matters for accuracy:
  - For MoE models like DeepSeek-V3 (EP=8, topK=8), EP A2A is a significant
    fraction (often 20–40%) of per-layer time.
  - If _wave_overlap_saved() is incorrect, MoE training step-time estimates
    are off by the same fraction — a systematic error on all MoE benchmarks.
  - In particular: if saved > comm_time, the model predicts negative exposed
    communication, making MoE appear faster than compute-only models.
"""
from __future__ import annotations

import pytest

from zrt.training.compose.stage import _wave_overlap_saved


# ── Boundary conditions ───────────────────────────────────────────────────────

class TestBoundaryConditions:
    def test_zero_comm_saves_nothing(self):
        """No communication to hide when there is no comm time."""
        assert _wave_overlap_saved(comm_time=0.0, gemm_time=1.0, K=4) == 0.0

    def test_zero_gemm_saves_nothing(self):
        """No GEMM to overlap with — comm is fully exposed."""
        assert _wave_overlap_saved(comm_time=1.0, gemm_time=0.0, K=4) == 0.0

    def test_negative_inputs_save_nothing(self):
        """Negative inputs are guarded; result is 0."""
        assert _wave_overlap_saved(comm_time=-1.0, gemm_time=1.0, K=4) == 0.0
        assert _wave_overlap_saved(comm_time=1.0, gemm_time=-1.0, K=4) == 0.0

    def test_k1_saves_nothing(self):
        """With K=1 wave there is no prior GEMM wave to overlap with.

        The model's first wave is always fully exposed; with K=1 there are
        no subsequent waves.  Result: zero savings regardless of GEMM size.
        """
        saved = _wave_overlap_saved(comm_time=5.0, gemm_time=5.0, K=1)
        assert saved == 0.0, f"K=1 should save nothing; got saved={saved}"

    def test_k0_saves_nothing(self):
        """K=0 is degenerate (no waves); result should be 0."""
        assert _wave_overlap_saved(comm_time=1.0, gemm_time=1.0, K=0) == 0.0


# ── Invariants ────────────────────────────────────────────────────────────────

class TestInvariants:
    @pytest.mark.parametrize("comm,gemm,K", [
        (0.1, 0.5, 4),
        (1.0, 1.0, 4),
        (10.0, 0.5, 4),
        (0.5, 10.0, 8),
        (3.0, 1.0, 2),
        (1.0, 3.0, 2),
    ])
    def test_saved_is_non_negative(self, comm, gemm, K):
        """Saved time can never be negative: you cannot save more than nothing."""
        saved = _wave_overlap_saved(comm, gemm, K)
        assert saved >= 0.0, f"saved={saved} < 0 for comm={comm}, gemm={gemm}, K={K}"

    @pytest.mark.parametrize("comm,gemm,K", [
        (0.1, 0.5, 4),
        (1.0, 1.0, 4),
        (10.0, 0.5, 4),
        (0.5, 10.0, 8),
        (3.0, 1.0, 2),
    ])
    def test_saved_cannot_exceed_total_comm(self, comm, gemm, K):
        """Cannot hide more communication than exists.

        saved > comm_time would mean exposed communication is negative —
        a physically impossible result that would make the model predict
        negative latency contributions for EP collectives.
        """
        saved = _wave_overlap_saved(comm, gemm, K)
        assert saved <= comm + 1e-12, (
            f"saved={saved:.6f} > comm={comm} for gemm={gemm}, K={K}"
        )


# ── Quantitative correctness ──────────────────────────────────────────────────

class TestQuantitativeCorrectness:
    def test_gemm_dominates_comm_k4_hides_75_percent(self):
        """When GEMM >> comm with K=4, exactly 75% of comm is hidden.

        Derivation:
          comm_per_wave = comm / 4
          gemm_per_wave = gemm / 4  (>> comm_per_wave)
          exposed_per_wave = max(comm_per_wave - gemm_per_wave, 0) = 0
          exposed_total = comm/4  (first wave only)
          saved = comm - comm/4 = 3/4 * comm = 75%

        This is the maximum possible savings for K=4.  If this test fails,
        the formula is wrong and all EP overlap estimates are incorrect.
        """
        comm = 1.0
        gemm = 1000.0  # gemm >> comm
        K = 4
        saved = _wave_overlap_saved(comm, gemm, K)
        expected = comm * (K - 1) / K   # = 0.75
        assert saved == pytest.approx(expected, rel=1e-9), (
            f"Expected {expected:.4f} (75%), got {saved:.6f}"
        )

    def test_gemm_dominates_comm_k8_hides_87_5_percent(self):
        """When GEMM >> comm with K=8, exactly 87.5% is hidden (= (K-1)/K)."""
        comm = 1.0
        gemm = 1000.0
        K = 8
        saved = _wave_overlap_saved(comm, gemm, K)
        expected = comm * (K - 1) / K   # = 0.875
        assert saved == pytest.approx(expected, rel=1e-9)

    def test_more_waves_hide_more_when_gemm_dominates(self):
        """Increasing K (more waves) increases the fraction of comm hidden,
        when GEMM dominates.  This is a monotone property of the model.
        """
        comm, gemm = 1.0, 100.0
        saved_k2 = _wave_overlap_saved(comm, gemm, K=2)
        saved_k4 = _wave_overlap_saved(comm, gemm, K=4)
        saved_k8 = _wave_overlap_saved(comm, gemm, K=8)
        assert saved_k2 < saved_k4 < saved_k8, (
            f"K savings not monotone: K=2→{saved_k2:.3f}, K=4→{saved_k4:.3f}, K=8→{saved_k8:.3f}"
        )

    def test_comm_dominates_gemm_saves_little(self):
        """When comm >> GEMM, each wave's comm exceeds available GEMM overlap.

        Derivation (comm=100, gemm=1, K=4):
          comm_per_wave = 25, gemm_per_wave = 0.25
          exposed_per_wave = 25 - 0.25 = 24.75
          exposed_total = 25 + 3 × 24.75 = 99.25
          saved = 100 - 99.25 = 0.75  (< 1% of comm)

        In MoE models where A2A >> expert GEMM (e.g. very fast NVLink vs slow
        GPU), almost all communication is exposed.  Incorrectly reporting large
        savings here would produce optimistic estimates.
        """
        comm = 100.0
        gemm = 1.0
        K = 4
        saved = _wave_overlap_saved(comm, gemm, K)
        # Verify analytically:
        comm_pw = comm / K
        gemm_pw = gemm / K
        exposed_pw = max(comm_pw - gemm_pw, 0)
        exposed_total = comm_pw + (K - 1) * exposed_pw
        expected_saved = max(0.0, comm - exposed_total)
        assert saved == pytest.approx(expected_saved, rel=1e-9)
        # And it should be small relative to comm:
        assert saved < comm * 0.02, (
            f"comm >> GEMM: saved={saved:.3f} should be < 2% of comm={comm}"
        )

    def test_equal_comm_and_gemm_analytical(self):
        """comm == GEMM: verify the formula matches hand calculation.

        comm=4, gemm=4, K=4:
          comm_per_wave=1, gemm_per_wave=1
          exposed_per_wave = max(1-1, 0) = 0
          exposed_total = 1 + 3×0 = 1
          saved = 4 - 1 = 3  (75% → same as gemm-dominated case)

        This confirms the formula is correct for balanced comm/GEMM.
        """
        comm = 4.0
        gemm = 4.0
        K = 4
        saved = _wave_overlap_saved(comm, gemm, K)
        assert saved == pytest.approx(3.0, rel=1e-9), (
            f"comm=gemm=4, K=4: expected saved=3.0, got {saved}"
        )

    def test_deepseek_v3_realistic_scenario(self):
        """Realistic DeepSeek-V3 EP scenario: comm ≈ gemm, K=4.

        For a typical MoE layer on H100 (8 experts/GPU with EP=8):
          - EP A2A comm: ~0.5 ms per phase (fwd or bwd)
          - Expert GEMM: ~1.0 ms (topK=8 means 8/8=1 full expert per GPU)
          - K = 4 waves

        Expected savings: GEMM (1.0) > comm (0.5) → exposed = comm/4 = 0.125 ms
        Saved = 0.5 - 0.125 = 0.375 ms (75% hidden, same as GEMM-dominated).

        This test guards against over- or under-estimating EP overlap for
        real-world MoE models.
        """
        comm = 0.5e-3    # 0.5 ms A2A per phase
        gemm = 1.0e-3    # 1.0 ms expert GEMM
        K = 4
        saved = _wave_overlap_saved(comm, gemm, K)
        # gemm (1.0ms) > comm (0.5ms) → 75% hidden
        expected = comm * (K - 1) / K   # 0.375 ms
        assert saved == pytest.approx(expected, rel=1e-9)
        assert 0.35e-3 < saved < 0.40e-3, (
            f"DeepSeek-V3 scenario: saved={saved*1e3:.3f} ms, expected ~0.375 ms"
        )
