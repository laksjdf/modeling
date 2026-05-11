"""Test Interleaved 1F1B composer."""
from __future__ import annotations

import pytest

from zrt.training.compose.schedules import Interleaved1F1BComposer, OneF1BComposer, StepResult
from zrt.training.compose.stage import StageTime
from zrt.training.spec.strategy import PPSched, Strategy


def _make_strategy(pp=4, vpp_chunks=2, dp=1, micro_batch=1, global_batch=32):
    return Strategy(
        pp=pp, vpp_chunks=vpp_chunks, dp=dp,
        micro_batch=micro_batch, global_batch=global_batch,
        pp_schedule=PPSched.INTERLEAVED,
    )


def _make_stage_times(pp, fwd=0.01, bwd=0.02):
    return [StageTime(fwd=fwd, bwd=bwd) for _ in range(pp)]


def test_interleaved_vpp1_equals_standard_1f1b():
    st = _make_stage_times(4)
    s = _make_strategy(pp=4, vpp_chunks=1)
    M = s.num_microbatches()

    interleaved = Interleaved1F1BComposer().compose(st, M, 4, 0.0, s)
    standard = OneF1BComposer().compose(st, M, 4, 0.0, s)

    assert interleaved.step_time == pytest.approx(standard.step_time, rel=1e-9)
    assert interleaved.bubble_fraction == pytest.approx(standard.bubble_fraction, rel=1e-9)


def test_interleaved_bubble_less_than_standard():
    st = _make_stage_times(4)
    s_vpp = _make_strategy(pp=4, vpp_chunks=2)
    s_std = Strategy(pp=4, vpp_chunks=1, dp=1, micro_batch=1, global_batch=32)
    M = s_vpp.num_microbatches()

    interleaved = Interleaved1F1BComposer().compose(st, M, 4, 0.0, s_vpp)
    standard = OneF1BComposer().compose(st, M, 4, 0.0, s_std)

    assert interleaved.bubble_fraction <= standard.bubble_fraction + 1e-9


def test_interleaved_pp1_no_pipeline():
    st = _make_stage_times(1)
    s = _make_strategy(pp=1, vpp_chunks=2)
    M = s.num_microbatches()

    result = Interleaved1F1BComposer().compose(st, M, 1, 0.0, s)
    standard = OneF1BComposer().compose(st, M, 1, 0.0, s)

    assert result.step_time == pytest.approx(standard.step_time, rel=1e-9)


def test_interleaved_schedule_identity():
    """Regression: Interleaved1F1BComposer should return schedule_name='i1f1b'."""
    st = _make_stage_times(4)
    s = _make_strategy(pp=4, vpp_chunks=2)
    M = s.num_microbatches()

    result = Interleaved1F1BComposer().compose(st, M, 4, 0.0, s)

    assert result.schedule_name == "i1f1b"


def test_interleaved_smaller_bubble_than_standard():
    """Regression: VPP should reduce bubble compared to standard 1F1B."""
    st = _make_stage_times(4)
    s_vpp = _make_strategy(pp=4, vpp_chunks=2)
    s_std = Strategy(pp=4, vpp_chunks=1, dp=1, micro_batch=1, global_batch=32)
    M = s_vpp.num_microbatches()

    interleaved = Interleaved1F1BComposer().compose(st, M, 4, 0.0, s_vpp)
    standard = OneF1BComposer().compose(st, M, 4, 0.0, s_std)

    assert interleaved.bubble_fraction < standard.bubble_fraction
