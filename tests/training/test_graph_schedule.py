"""Test graph-path schedule dispatch in TrainingPipelinePass."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.transform.analysis.training import TrainingPipelinePass
from python.zrt.transform.context import (
    ParallelConfig,
    TrainingConfig,
    TransformContext,
)


def _make_graph(metadata=None):
    return OpGraph(name="test", phase="prefill", metadata=metadata or {})


def _make_ctx(tp=1, pp=4, dp=1, pp_schedule="1f1b", vpp_chunks=1,
              micro_batch=1, global_batch=8):
    from python.zrt.hardware.spec import (
        ComputeSpec, HardwareSpec, InterconnectSpec, LinkSpec, MemorySpec,
    )
    hw = HardwareSpec(
        name="test_h100", vendor="nvidia", device_type="gpu",
        compute=ComputeSpec(bf16_tflops=989, fp8_tops=1979),
        memory=MemorySpec(capacity_gb=80, hbm_bandwidth_gbps=3350),
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=100, latency_us=10.0),
        ),
    )
    return TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=tp, pp=pp, dp=dp),
        training=TrainingConfig(
            pp_schedule=pp_schedule,
            vpp_chunks=vpp_chunks,
            micro_batch=micro_batch,
            global_batch=global_batch,
        ),
    )


def _run_pass(pp, pp_schedule, vpp_chunks=1, per_stage_us=1000.0):
    g = _make_graph(metadata={
        "num_layers": 4, "num_layers_traced": 4,
        "training_flops": 1e12,
    })
    ctx = _make_ctx(pp=pp, pp_schedule=pp_schedule, vpp_chunks=vpp_chunks)

    mock_timeline = MagicMock()
    mock_timeline.total_latency_us = per_stage_us * pp

    with patch("python.zrt.executor.scheduler.DAGScheduler") as MockSched:
        MockSched.return_value.schedule.return_value = mock_timeline
        result = TrainingPipelinePass().run(g, ctx)

    return result.metadata["pipeline_metrics"]


def _run_stage_pass(pp_schedule, bwd_dw_share=0.5):
    nodes = {}
    for stage_id in range(4):
        dx = 1.0 - bwd_dw_share
        dw = bwd_dw_share
        nodes[f"s{stage_id}"] = OpNode(
            id=f"s{stage_id}",
            op_type="aten.mm.default",
            annotations={
                "stage_id": stage_id,
                "phase": "bwd",
                "flops_dx": dx,
                "flops_dw": dw,
            },
        )
    g = OpGraph(
        name="stage_split",
        phase="train",
        nodes=nodes,
        edges=[],
        metadata={
            "num_layers": 4,
            "num_layers_traced": 4,
            "training_flops": 1e12,
        },
    )
    ctx = _make_ctx(pp=4, pp_schedule=pp_schedule)

    mock_timeline = MagicMock()
    mock_timeline.total_latency_us = 3000.0
    mock_timeline.phase_latency.side_effect = lambda phase: {
        "fwd": 1000.0,
        "bwd": 2000.0,
    }.get(phase, 0.0)

    with patch("python.zrt.executor.scheduler.DAGScheduler") as MockSched:
        MockSched.return_value.schedule.return_value = mock_timeline
        result = TrainingPipelinePass().run(g, ctx)

    return result.metadata["pipeline_metrics"], result.metadata


def _run_heterogeneous_stage_pass(pp_schedule):
    nodes = {
        "s0": OpNode(
            id="s0",
            op_type="aten.mm.default",
            annotations={
                "stage_id": 0,
                "phase": "bwd",
                "flops_dx": 1.0,
                "flops_dw": 0.0,
            },
        ),
        "s1": OpNode(
            id="s1",
            op_type="aten.mm.default",
            annotations={
                "stage_id": 1,
                "phase": "bwd",
                "flops_dx": 0.0,
                "flops_dw": 1.0,
            },
        ),
    }
    g = OpGraph(
        name="heterogeneous_stage_split",
        phase="train",
        nodes=nodes,
        edges=[],
        metadata={
            "num_layers": 2,
            "num_layers_traced": 2,
            "training_flops": 1e12,
        },
    )
    ctx = _make_ctx(pp=2, pp_schedule=pp_schedule, global_batch=4)

    timelines = [
        {"fwd": 1000.0, "bwd": 3000.0},
        {"fwd": 1000.0, "bwd": 1000.0},
    ]

    with patch("python.zrt.executor.scheduler.DAGScheduler") as MockSched:
        scheduler = MockSched.return_value
        scheduler.schedule.side_effect = [
            _mock_timeline(timelines[0]),
            _mock_timeline(timelines[1]),
        ]
        result = TrainingPipelinePass().run(g, ctx)

    return result.metadata["pipeline_metrics"], result.metadata


def _mock_timeline(values):
    mock_timeline = MagicMock()
    mock_timeline.total_latency_us = values["fwd"] + values["bwd"]
    mock_timeline.phase_latency.side_effect = lambda phase: values.get(phase, 0.0)
    return mock_timeline


def test_vpp_reduces_step_time_vs_1f1b():
    f1b = _run_pass(pp=4, pp_schedule="1f1b")
    vpp = _run_pass(pp=4, pp_schedule="interleaved", vpp_chunks=2)
    assert vpp.step_time_ms < f1b.step_time_ms


def test_dualpipe_reduces_step_time_vs_1f1b():
    f1b = _run_pass(pp=4, pp_schedule="1f1b")
    dp = _run_pass(pp=4, pp_schedule="dualpipe")
    assert dp.step_time_ms < f1b.step_time_ms


def test_dualpipev_reduces_step_time_vs_dualpipe():
    dp = _run_pass(pp=4, pp_schedule="dualpipe")
    dpv = _run_pass(pp=4, pp_schedule="dualpipev", vpp_chunks=2)
    assert dpv.step_time_ms <= dp.step_time_ms


def test_zero_bubble_uses_dw_split_to_reduce_dualpipe_bubble():
    dualpipe, _ = _run_stage_pass("dualpipe")
    zero_bubble, metadata = _run_stage_pass("zb")

    pp = 4
    M = 8
    t_stage_ms = 3.0
    t_w_ms = 1.0
    expected_step_ms = M * t_stage_ms + (pp - 1) * (t_stage_ms - t_w_ms)

    assert metadata["stage_timelines_bwd_dw"] == {
        0: pytest.approx(1000.0),
        1: pytest.approx(1000.0),
        2: pytest.approx(1000.0),
        3: pytest.approx(1000.0),
    }
    assert zero_bubble.step_time_ms == pytest.approx(expected_step_ms)
    assert zero_bubble.bubble_fraction < dualpipe.bubble_fraction


def test_zero_bubble_uses_bottleneck_stage_dw_split():
    zero_bubble, metadata = _run_heterogeneous_stage_pass("zb")

    M = 4
    t_stage_ms = 4.0
    bottleneck_t_w_ms = 0.0
    expected_step_ms = M * t_stage_ms + (2 - 1) * (t_stage_ms - bottleneck_t_w_ms)

    assert metadata["stage_timelines_bwd_dw"][0] == pytest.approx(0.0)
    assert metadata["stage_timelines_bwd_dw"][1] == pytest.approx(1000.0)
    assert zero_bubble.step_time_ms == pytest.approx(expected_step_ms)


def test_zero_bubble_logs_when_dw_annotations_missing(caplog):
    caplog.set_level("DEBUG", logger="python.zrt.transform.analysis.training")

    metrics = _run_pass(pp=4, pp_schedule="zb")

    assert metrics.step_time_ms > 0
    assert "ZeroBubble selected without stage/phase or flops_dw annotations" in caplog.text


def test_1f1b_bubble_fraction():
    metrics = _run_pass(pp=4, pp_schedule="1f1b")
    pp = 4
    M = 8
    expected_bubble = (pp - 1) / (M + pp - 1)
    assert metrics.bubble_fraction == pytest.approx(expected_bubble, abs=0.01)


def test_pp1_no_pipeline():
    metrics = _run_pass(pp=1, pp_schedule="1f1b")
    assert metrics.step_time_ms > 0
