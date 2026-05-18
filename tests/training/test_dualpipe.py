"""Test DualPipe and DualPipeV composers."""
from __future__ import annotations

import pytest

from zrt.training.compose.schedules import (
    DualPipeComposer,
    DualPipeVComposer,
    OneF1BComposer,
    StepResult,
    ZeroBubbleComposer,
)
from zrt.training.compose.stage import StageTime
from zrt.training.spec.strategy import PPSched, Strategy


def _make_stage_times(pp, fwd=0.01, bwd=0.02):
    return [StageTime(fwd=fwd, bwd=bwd) for _ in range(pp)]


def _make_strategy(
    pp=4, vpp_chunks=1, dp=1, micro_batch=1, global_batch=32,
    schedule=PPSched.DUALPIPE,
):
    return Strategy(
        pp=pp, vpp_chunks=vpp_chunks, dp=dp,
        micro_batch=micro_batch, global_batch=global_batch,
        pp_schedule=schedule,
    )


def test_dualpipe_bubble_less_than_1f1b():
    st = [
        StageTime(fwd=0.02, bwd=0.01),
        StageTime(fwd=0.01, bwd=0.02),
        StageTime(fwd=0.02, bwd=0.01),
        StageTime(fwd=0.01, bwd=0.02),
    ]
    s = _make_strategy(pp=4)
    M = s.num_microbatches()

    dp_result = DualPipeComposer().compose(st, M, 4, 0.0, s)
    s1f1b = Strategy(pp=4, dp=1, micro_batch=1, global_batch=32)
    f1b_result = OneF1BComposer().compose(st, M, 4, 0.0, s1f1b)

    assert dp_result.bubble_fraction < f1b_result.bubble_fraction


def test_dualpipe_pp1_no_pipeline():
    st = _make_stage_times(1)
    s = _make_strategy(pp=1)
    M = s.num_microbatches()

    dp_result = DualPipeComposer().compose(st, M, 1, 0.0, s)
    f1b_result = OneF1BComposer().compose(st, M, 1, 0.0, s)

    assert dp_result.step_time == pytest.approx(f1b_result.step_time, rel=1e-9)


def test_dualpipev_bubble_less_than_dualpipe():
    st = _make_stage_times(4)
    s_v = _make_strategy(pp=4, vpp_chunks=2, schedule=PPSched.DUALPIPE_V)
    s_dp = _make_strategy(pp=4, vpp_chunks=1, schedule=PPSched.DUALPIPE)
    M = s_v.num_microbatches()

    dpv_result = DualPipeVComposer().compose(st, M, 4, 0.0, s_v)
    dp_result = DualPipeComposer().compose(st, M, 4, 0.0, s_dp)

    assert dpv_result.bubble_fraction <= dp_result.bubble_fraction + 1e-9


def test_dualpipev_vpp1_equals_dualpipe():
    st = _make_stage_times(4)
    s = _make_strategy(pp=4, vpp_chunks=1, schedule=PPSched.DUALPIPE_V)
    M = s.num_microbatches()

    dpv_result = DualPipeVComposer().compose(st, M, 4, 0.0, s)
    dp_result = DualPipeComposer().compose(st, M, 4, 0.0, s)

    assert dpv_result.step_time == pytest.approx(dp_result.step_time, rel=1e-9)


def test_dualpipe_schedule_identity():
    """Regression: DualPipeComposer should return schedule_name='dualpipe'."""
    st = _make_stage_times(4)
    s = _make_strategy(pp=4)
    M = s.num_microbatches()

    result = DualPipeComposer().compose(st, M, 4, 0.0, s)

    assert result.schedule_name == "dualpipe"


def test_dualpipev_schedule_identity():
    """Regression: DualPipeVComposer should return schedule_name='dualpipev'."""
    st = _make_stage_times(4)
    s = _make_strategy(pp=4, vpp_chunks=2, schedule=PPSched.DUALPIPE_V)
    M = s.num_microbatches()

    result = DualPipeVComposer().compose(st, M, 4, 0.0, s)

    assert result.schedule_name == "dualpipev"


def test_standard_1f1b_schedule_identity():
    """Regression: OneF1BComposer should return schedule_name='1f1b'."""
    st = _make_stage_times(4)
    s = Strategy(pp=4, dp=1, micro_batch=1, global_batch=32)
    M = s.num_microbatches()

    result = OneF1BComposer().compose(st, M, 4, 0.0, s)

    assert result.schedule_name == "1f1b"


def test_zero_bubble_uses_weight_grad_to_reduce_bubble():
    """ZB-1P uses W (weight gradient) to fill pipeline bubble.

    Formula: warmup = (PP-1) * (F - W), cooldown = (PP-1) * (B - W)
    where F = fwd, B = bwd_dx (activation gradient only).
    """
    st = [
        StageTime(fwd=0.01, bwd=0.02, bwd_dx=0.01, bwd_dw=0.01),
        StageTime(fwd=0.01, bwd=0.02, bwd_dx=0.01, bwd_dw=0.01),
        StageTime(fwd=0.01, bwd=0.02, bwd_dx=0.01, bwd_dw=0.01),
        StageTime(fwd=0.01, bwd=0.02, bwd_dx=0.01, bwd_dw=0.01),
    ]
    s = _make_strategy(pp=4, schedule=PPSched.ZERO_BUBBLE)
    M = s.num_microbatches()

    zb = ZeroBubbleComposer().compose(st, M, 4, 0.0, s)
    f1b = OneF1BComposer().compose(st, M, 4, 0.0, _make_strategy(pp=4))

    t_fwd = 0.01
    t_b_dx = 0.01
    t_stage = 0.03
    warmup_expected = 3 * max(t_fwd - 0.01, 2e-6)
    cooldown_expected = 3 * max(t_b_dx - 0.01, 2e-6)
    step_expected = warmup_expected + M * t_stage + cooldown_expected
    assert zb.step_time == pytest.approx(step_expected, rel=1e-6)
    assert zb.bubble_fraction < f1b.bubble_fraction
    assert zb.schedule_name == "zb"


def test_zero_bubble_uses_bottleneck_stage_weight_grad():
    st = [
        StageTime(fwd=0.01, bwd=0.03, bwd_dx=0.03, bwd_dw=0.00),
        StageTime(fwd=0.01, bwd=0.02, bwd_dx=0.00, bwd_dw=0.02),
        StageTime(fwd=0.01, bwd=0.02, bwd_dx=0.01, bwd_dw=0.01),
        StageTime(fwd=0.01, bwd=0.02, bwd_dx=0.01, bwd_dw=0.01),
    ]
    s = _make_strategy(pp=4, schedule=PPSched.ZERO_BUBBLE)
    M = s.num_microbatches()

    zb = ZeroBubbleComposer().compose(st, M, 4, 0.0, s)

    bottleneck_stage = 0.04
    bottleneck_dw = 0.0
    expected = M * bottleneck_stage + (4 - 1) * max(bottleneck_stage - 2 * bottleneck_dw, 0.0)
    assert zb.step_time == pytest.approx(expected, rel=1e-9)


def test_dualpipe_pp2_zero_bubble():
    """DualPipe pp=2: two anti-parallel streams perfectly fill each other — zero bubble."""
    st = _make_stage_times(2)
    s = _make_strategy(pp=2)
    M = s.num_microbatches()

    result = DualPipeComposer().compose(st, M, 2, 0.0, s)

    assert result.bubble_fraction == pytest.approx(0.0, abs=1e-12)
    assert result.warmup == pytest.approx(0.0, abs=1e-12)
    assert result.cooldown == pytest.approx(0.0, abs=1e-12)


def test_dualpipev_pp2_zero_bubble():
    """DualPipeV pp=2: same zero-bubble property holds regardless of vpp_chunks."""
    st = _make_stage_times(2)
    s = _make_strategy(pp=2, vpp_chunks=2, schedule=PPSched.DUALPIPE_V)
    M = s.num_microbatches()

    result = DualPipeVComposer().compose(st, M, 2, 0.0, s)

    assert result.bubble_fraction == pytest.approx(0.0, abs=1e-12)
    assert result.warmup == pytest.approx(0.0, abs=1e-12)
    assert result.cooldown == pytest.approx(0.0, abs=1e-12)


def test_dualpipe_pp3_half_stage_bubble():
    """DualPipe pp=3: odd pipeline degree produces half-stage bubble.

    Formula: warmup = (PP/2-1) * (F&B - 2W), cooldown = (PP/2-1) * (B - W)
    For pp=3 with no W: warmup = cooldown = 0.5 * F&B
    F&B = max(F, B) = max(0.01, 0.02) = 0.02
    """
    st = _make_stage_times(3, fwd=0.01, bwd=0.02)
    s = _make_strategy(pp=3)
    M = s.num_microbatches()

    result = DualPipeComposer().compose(st, M, 3, 0.0, s)

    F = 0.01
    B_dx = 0.02
    F_and_B = max(F, B_dx)
    warmup_expected = 0.5 * F_and_B
    cooldown_expected = 0.5 * B_dx
    assert result.bubble_fraction > 0.0
    assert result.warmup == pytest.approx(warmup_expected, abs=1e-6)
    assert result.cooldown == pytest.approx(cooldown_expected, abs=1e-6)


def test_dualpipev_pp3_half_stage_bubble():
    """DualPipeV pp=3, V=2: bubble same as DualPipe (V only affects device count).

    Formula unchanged from DualPipe: warmup = cooldown = 0.5 * F&B
    F&B = max(F, B) = 0.02
    """
    st = _make_stage_times(3, fwd=0.01, bwd=0.02)
    s = _make_strategy(pp=3, vpp_chunks=2, schedule=PPSched.DUALPIPE_V)
    M = s.num_microbatches()

    result = DualPipeVComposer().compose(st, M, 3, 0.0, s)

    F = 0.01
    B_dx = 0.02
    F_and_B = max(F, B_dx)
    warmup_expected = 0.5 * F_and_B
    cooldown_expected = 0.5 * B_dx
    assert result.bubble_fraction > 0.0
    assert result.warmup == pytest.approx(warmup_expected, abs=1e-6)
    assert result.cooldown == pytest.approx(cooldown_expected, abs=1e-6)
