"""Tests for unified DP overlap window across all composers + dualbatch path."""

import pytest
from zrt.training.compose.schedules import (
    OneF1BComposer, Interleaved1F1BComposer, ZeroBubbleComposer,
    DualPipeComposer, DualPipeVComposer, pipeline_step_time,
)
from zrt.training.compose.stage import StageTime
from zrt.training.ir.builders import build_graph
from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.strategy import Strategy, PPSched
from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.spec.system import SystemSpec, GPU


def _make_system():
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


class TestSteadyBwdOverlap:
    """dp_steady_overlap_ratio lets DP hide in steady-state backward compute
    (FSDP / Megatron behavior), not only in the explicit cooldown phase."""

    def test_default_ratio_is_0p5(self):
        s = Strategy(tp=1, pp=1, dp=8, micro_batch=1, global_batch=8)
        assert s.dp_steady_overlap_ratio == 0.5

    def test_ratio_extends_hide_window_in_dualpipe(self):
        """DualPipe pp=2 has zero bubble (reference formula: (PP/2-1)*(...)=0),
        so cooldown=0. With ratio>0, steady_bwd contributes and dp_exposed shrinks."""
        # Two equal stages, fwd=10, bwd=20 (bwd_dw=0 → W=0)
        # pp=2 DualPipe: factor = (2/2-1) = 0 → bubble=0, cooldown=0
        # steady_bwd_total = M * t_bwd_max = 4 * 20 = 80
        stages = [StageTime(fwd=10.0, bwd=20.0), StageTime(fwd=10.0, bwd=20.0)]
        s_no = Strategy(tp=1, pp=2, dp=4, micro_batch=1, global_batch=4,
                        pp_schedule=PPSched.DUALPIPE, dp_steady_overlap_ratio=0.0)
        s_half = Strategy(tp=1, pp=2, dp=4, micro_batch=1, global_batch=4,
                          pp_schedule=PPSched.DUALPIPE, dp_steady_overlap_ratio=0.5)

        r_no = DualPipeComposer().compose(stages, M=4, pp=2, dp_ar_time=100.0, strategy=s_no)
        r_half = DualPipeComposer().compose(stages, M=4, pp=2, dp_ar_time=100.0, strategy=s_half)

        # ratio=0: window = 0 + 0*80 = 0 → hide = 0 → exposed = 100.0
        # ratio=0.5: window = 0 + 0.5*80 = 40 → hide = 40 → exposed = 60.0
        assert r_no.dp_exposed == pytest.approx(100.0)
        assert r_half.dp_exposed == pytest.approx(60.0)

    def test_ratio_zero_matches_legacy_behavior(self):
        """ratio=0 should reproduce the previous cooldown-only window exactly,
        keeping backward-compat with anchor fixtures."""
        stages = [StageTime(fwd=5.0, bwd=10.0), StageTime(fwd=5.0, bwd=10.0)]
        s = Strategy(tp=1, pp=2, dp=4, micro_batch=1, global_batch=4,
                     pp_schedule=PPSched.ONE_F_ONE_B, dp_steady_overlap_ratio=0.0)
        # pp=2 1F1B: cooldown = (pp-1)*max(bwd) = 10
        r = OneF1BComposer().compose(stages, M=4, pp=2, dp_ar_time=20.0, strategy=s)
        # hide = min(10, 20) = 10 → exposed = 10
        assert r.dp_exposed == pytest.approx(10.0)


class TestDualbatchRecoversWithSteadyOverlap:
    """dualbatch + DP overlap interaction.

    Post-redesign: dualbatch on DualPipe/DualPipeV is a no-op for the
    pipeline bubble (the antiparallel reduction is already baked into the
    composer's (pp-1)/(2V) formula). The natural cooldown therefore
    remains available for DP hiding even when dualbatch=True.
    """

    def test_dualbatch_dualpipe_preserves_cooldown_dp_hide(self):
        """DualPipe pp=2 + dualbatch: reference formula gives zero bubble at pp=2,
        so cooldown=0. DP hiding at ratio=0 is therefore also zero."""
        model = ModelSpec(hidden=2048, ffn=8192, num_heads=16, num_kv_heads=16,
                          head_dim=128, vocab=32000, seq_len=1024,
                          layers=[LayerKind.DENSE]*4)
        s = Strategy(tp=1, pp=2, dp=4, micro_batch=1, global_batch=4,
                     pp_schedule=PPSched.DUALPIPE, dualbatch=True,
                     dp_steady_overlap_ratio=0.0)
        graph = build_graph(model, s)
        result = pipeline_step_time(graph, model, _make_system(), s)
        # pp=2 DualPipe: (PP/2-1)*(F&B+B-3W) = 0 → zero bubble → cooldown=0
        assert result.cooldown == pytest.approx(0.0, abs=1e-12)
        assert result.dp_hidden >= 0.0

    def test_dualbatch_with_ratio_half_recovers_dp_hide(self):
        model = ModelSpec(hidden=2048, ffn=8192, num_heads=16, num_kv_heads=16,
                          head_dim=128, vocab=32000, seq_len=1024,
                          layers=[LayerKind.DENSE]*4)
        s = Strategy(tp=1, pp=2, dp=4, micro_batch=1, global_batch=4,
                     pp_schedule=PPSched.DUALPIPE, dualbatch=True,
                     dp_steady_overlap_ratio=0.5)
        graph = build_graph(model, s)
        result = pipeline_step_time(graph, model, _make_system(), s)
        # With ratio>0, dp_hidden > 0 even though dualbatch zeroed cooldown
        assert result.dp_hidden > 0.0, \
            f"steady-bwd window should recover some DP hide, got dp_hidden={result.dp_hidden}"


class TestHideWindowConsistency:
    """All composers should use the same hide-window formula:
    hide = cooldown + ratio * steady_bwd."""

    def _stages(self):
        return [StageTime(fwd=5.0, bwd=10.0), StageTime(fwd=5.0, bwd=10.0)]

    @pytest.mark.parametrize("composer_cls,sched", [
        (OneF1BComposer, PPSched.ONE_F_ONE_B),
        (DualPipeComposer, PPSched.DUALPIPE),
        (ZeroBubbleComposer, PPSched.ZERO_BUBBLE),
    ])
    def test_hide_window_includes_steady_bwd(self, composer_cls, sched):
        stages = self._stages()
        s = Strategy(tp=1, pp=2, dp=4, micro_batch=1, global_batch=4,
                     pp_schedule=sched, dp_steady_overlap_ratio=0.5)
        # M=4 → steady_bwd ≈ 4 * 10 = 40, plus per-composer cooldown
        # Set dp_ar_time deliberately larger than cooldown alone to detect that
        # the steady_bwd term contributes.
        r = composer_cls().compose(stages, M=4, pp=2, dp_ar_time=200.0, strategy=s)
        assert r.dp_exposed < 200.0, \
            f"{sched}: no steady-bwd contribution detected (dp_exposed={r.dp_exposed})"
        # Sanity: with ratio=0 it should be larger (more exposed).
        s0 = Strategy(tp=1, pp=2, dp=4, micro_batch=1, global_batch=4,
                      pp_schedule=sched, dp_steady_overlap_ratio=0.0)
        r0 = composer_cls().compose(stages, M=4, pp=2, dp_ar_time=200.0, strategy=s0)
        assert r0.dp_exposed > r.dp_exposed, \
            f"{sched}: ratio=0 should expose more DP than ratio=0.5"

    def test_onef1b_pp1_keeps_last_bucket_residual(self):
        """pp=1: even when the steady-bwd hide window dwarfs dp_ar_time, the
        last gradient bucket cannot overlap, so a residual of
        dp_ar_time / dp_grad_buckets stays exposed (never exactly 0)."""
        stages = [StageTime(fwd=1.0, bwd=2.0)]
        s = Strategy(tp=1, pp=1, dp=4, micro_batch=1, global_batch=4,
                     dp_steady_overlap_ratio=1.0, dp_grad_buckets=25)
        # window = bwd*M = 8 >> dp_ar_time; max_hidable = 3*(1-1/25) = 2.88
        r = OneF1BComposer().compose(stages, M=4, pp=1, dp_ar_time=3.0, strategy=s)
        assert r.dp_exposed == pytest.approx(3.0 / 25)
        assert r.dp_exposed > 0.0

    def test_dualbatch_uses_same_helper(self):
        """After fix #3, dualbatch's new_dp_exposed should go through the same
        helper (cooldown + ratio*steady_bwd), so it can still hide DP in steady_bwd
        even when cooldown collapses to 0."""
        model = ModelSpec(hidden=2048, ffn=8192, num_heads=16, num_kv_heads=16,
                          head_dim=128, vocab=32000, seq_len=1024,
                          layers=[LayerKind.DENSE]*4)
        s = Strategy(tp=1, pp=2, dp=4, micro_batch=1, global_batch=16,
                     pp_schedule=PPSched.DUALPIPE, dualbatch=True,
                     dp_steady_overlap_ratio=1.0)
        graph = build_graph(model, s)
        result = pipeline_step_time(graph, model, _make_system(), s)
        # With ratio=1.0 and steady_bwd >> dp_total, DP should be ~fully hidden.
        assert result.dp_hidden > 0.5 * (result.dp_exposed + result.dp_hidden), \
            f"dualbatch+ratio=1.0 should hide most DP; got exposed={result.dp_exposed}, hidden={result.dp_hidden}"


class TestLastBucketResidual:
    """Regression for the dp_exposed≈0 bug: gradient bucketing leaves the last
    bucket's collective non-overlappable, so dp_exposed must stay > 0 whenever
    dp>1, regardless of how large the hide window is."""

    def _model(self):
        return ModelSpec(hidden=2048, ffn=8192, num_heads=16, num_kv_heads=16,
                         head_dim=128, vocab=32000, seq_len=1024,
                         layers=[LayerKind.DENSE] * 4)

    @pytest.mark.parametrize("pp,sched", [
        (1, PPSched.ONE_F_ONE_B),
        (2, PPSched.ONE_F_ONE_B),
        (2, PPSched.DUALPIPE),
    ])
    def test_dp_exposed_strictly_positive(self, pp, sched):
        model = self._model()
        s = Strategy(tp=1, pp=pp, dp=8, micro_batch=1, global_batch=32,
                     zero_stage=1, pp_schedule=sched,
                     dp_overlap_in_bubble=True, dp_steady_overlap_ratio=0.5,
                     dp_grad_buckets=25)
        r = pipeline_step_time(build_graph(model, s), model, _make_system(), s)
        total_dp = r.dp_exposed + r.dp_hidden
        assert total_dp > 0.0
        assert r.dp_exposed > 0.0, \
            f"DP fully hidden (bug): exposed={r.dp_exposed}, total={total_dp}"
        # Bucket-residual is a *floor* on exposed: at least 1/n of total DP
        # comm always escapes the hide window. Exact equality only when the
        # window dominates over max_hidable; otherwise window is binding and
        # exposed = dp_ar_time - window (≥ total/n).
        assert r.dp_exposed >= total_dp / 25 * (1 - 1e-6), (
            f"exposed={r.dp_exposed} below bucket floor total/25={total_dp/25}"
        )

    def test_more_buckets_smaller_residual(self):
        stages = [StageTime(fwd=1.0, bwd=2.0)]
        s_few = Strategy(tp=1, pp=1, dp=4, micro_batch=1, global_batch=4,
                         dp_steady_overlap_ratio=1.0, dp_grad_buckets=4)
        s_many = Strategy(tp=1, pp=1, dp=4, micro_batch=1, global_batch=4,
                          dp_steady_overlap_ratio=1.0, dp_grad_buckets=100)
        r_few = OneF1BComposer().compose(stages, M=4, pp=1, dp_ar_time=3.0, strategy=s_few)
        r_many = OneF1BComposer().compose(stages, M=4, pp=1, dp_ar_time=3.0, strategy=s_many)
        assert r_few.dp_exposed == pytest.approx(3.0 / 4)
        assert r_many.dp_exposed == pytest.approx(3.0 / 100)
        assert r_many.dp_exposed < r_few.dp_exposed

    def test_overlap_disabled_still_fully_exposed(self):
        """dp_overlap_in_bubble=False is unchanged: DP fully exposed."""
        stages = [StageTime(fwd=1.0, bwd=2.0)]
        s = Strategy(tp=1, pp=1, dp=4, micro_batch=1, global_batch=4,
                     dp_overlap_in_bubble=False, dp_grad_buckets=25)
        r = OneF1BComposer().compose(stages, M=4, pp=1, dp_ar_time=3.0, strategy=s)
        assert r.dp_exposed == pytest.approx(3.0)


class TestDualbatchBubbleFloor:
    """dualbatch cannot zero the pipeline bubble — there is always an
    irreducible (pp-1)/(2V) · t_stage fill/drain even with two antiparallel
    streams (no third stream exists to fill that region).
    """

    def test_1f1b_dualbatch_residual_bubble_is_dualpipe_floor(self):
        """1F1B + dualbatch should reduce bubble to the DualPipe floor
        (pp-1)/2 · t_stage_max, NOT to zero, regardless of M."""
        model = ModelSpec(hidden=2048, ffn=8192, num_heads=16, num_kv_heads=16,
                          head_dim=128, vocab=32000, seq_len=1024,
                          layers=[LayerKind.DENSE] * 4)
        # Large M to confirm bubble doesn't get "filled" by steady work.
        s = Strategy(tp=1, pp=4, dp=2, micro_batch=1, global_batch=256,
                     pp_schedule=PPSched.ONE_F_ONE_B, dualbatch=True)
        graph = build_graph(model, s)
        result = pipeline_step_time(graph, model, _make_system(), s)
        assert result.bubble_fraction > 0.0, (
            "dualbatch must not zero the pipeline bubble; got "
            f"bubble_fraction={result.bubble_fraction}"
        )
        assert (result.warmup + result.cooldown) > 0.0

    def test_dualpipev_dualbatch_skips_pipeline_adjustment(self):
        """DualPipeV already bakes the antiparallel floor into the
        composer's (pp-1)/(2V) bubble. dualbatch must not double-count
        and zero it; the composer's bubble should pass through unchanged."""
        model = ModelSpec(hidden=2048, ffn=8192, num_heads=16, num_kv_heads=16,
                          head_dim=128, vocab=32000, seq_len=1024,
                          layers=[LayerKind.DENSE] * 4)
        s = Strategy(tp=1, pp=4, dp=2, micro_batch=1, global_batch=256,
                     pp_schedule=PPSched.DUALPIPE_V, vpp_chunks=2,
                     dualbatch=True)
        graph = build_graph(model, s)
        result = pipeline_step_time(graph, model, _make_system(), s)
        # Composer gives bubble = (pp-1)/(2V) · t_stage > 0; dualbatch must
        # leave it alone.
        assert result.bubble_fraction > 0.0
        assert (result.warmup + result.cooldown) > 0.0


class TestZeroBubbleFloor:
    """ZB-1P / ZB-V still has a minimal per-transition bubble equal to 2*pp_p2p.
    Without a floor, balanced t_w configs collapse bubble to 0 incorrectly."""

    def test_balanced_tw_does_not_collapse_to_zero(self):
        # bottleneck.bwd_dw equal to fwd+bwd → naive formula gives 0 cooldown.
        stages = [StageTime(fwd=4.0, bwd=6.0, bwd_dw=10.0),
                  StageTime(fwd=4.0, bwd=6.0, bwd_dw=10.0)]
        s = Strategy(tp=1, pp=2, dp=4, micro_batch=1, global_batch=4,
                     pp_schedule=PPSched.ZERO_BUBBLE)
        r = ZeroBubbleComposer().compose(stages, M=4, pp=2, dp_ar_time=0.0, strategy=s)
        # Floor should be 2 * pp_p2p; with no system passed to the composer
        # directly, we rely on a small constant floor (≥1e-6) keeping it strictly > 0.
        assert r.cooldown > 0.0, \
            f"ZB cooldown collapsed to zero; got cooldown={r.cooldown}"

    def test_unbalanced_tw_uses_natural_value(self):
        """When t_stage > 2*t_w, the natural formula dominates and the floor is
        not applied (floor < natural)."""
        stages = [StageTime(fwd=4.0, bwd=6.0, bwd_dw=2.0),  # t_w small
                  StageTime(fwd=4.0, bwd=6.0, bwd_dw=2.0)]
        s = Strategy(tp=1, pp=2, dp=4, micro_batch=1, global_batch=4,
                     pp_schedule=PPSched.ZERO_BUBBLE)
        r = ZeroBubbleComposer().compose(stages, M=4, pp=2, dp_ar_time=0.0, strategy=s)
        # Natural: bubble = (pp-1)*(t_stage - 2*t_w) = 1*(10-4) = 6; cooldown=3
        assert r.cooldown == pytest.approx(3.0)
