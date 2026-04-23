"""Phase-1 bugfix regression tests.

Bug 1.1: 1F1B step-time formula
Bug 1.2: Activation memory (Korthikanti + recompute + ZeRO metadata)
Bug 1.3: TrainingFlopsPass per-node annotation priority
"""
from __future__ import annotations

import pytest

from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.transform.analysis.training import (
    TrainingFlopsPass,
    TrainingMemoryPass,
    TrainingPipelinePass,
)
from python.zrt.transform.context import (
    ParallelConfig,
    TrainingConfig,
    TransformContext,
)


def _make_graph(nodes=None, metadata=None):
    g = OpGraph(name="test", phase="prefill", metadata=metadata or {})
    if nodes:
        for n in nodes:
            g.nodes[n.id] = n
    return g


def _make_ctx(tp=1, pp=1, dp=1, cp=1, zero_stage=0, micro_batch=1,
              global_batch=32, recompute_policy="none", optimizer="adam"):
    from python.zrt.hardware.spec import (
        ComputeSpec,
        HardwareSpec,
        InterconnectSpec,
        LinkSpec,
        MemorySpec,
    )
    hw = HardwareSpec(
        name="test_h100",
        vendor="nvidia",
        device_type="gpu",
        compute=ComputeSpec(bf16_tflops=989, fp8_tops=1979),
        memory=MemorySpec(capacity_gb=80, hbm_bandwidth_gbps=3350),
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=100, latency_us=10.0),
        ),
    )
    return TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=tp, pp=pp, dp=dp, cp=cp),
        training=TrainingConfig(
            optimizer=optimizer,
            zero_stage=zero_stage,
            micro_batch=micro_batch,
            global_batch=global_batch,
            recompute_policy=recompute_policy,
        ),
    )


# ── Bug 1.3: TrainingFlopsPass ────────────────────────────────────────────────

def test_training_flops_uses_per_node_annotations():
    n1 = OpNode(id="mm", op_type="aten.mm.default",
                annotations={"flops_fwd": 100, "flops_dx": 50, "flops_dw": 50})
    n2 = OpNode(id="comm", op_type="comm.all_reduce",
                annotations={"flops_fwd": 0, "flops_dx": 0, "flops_dw": 0})
    g = _make_graph([n1, n2], {"num_layers": 4, "num_layers_traced": 4})
    ctx = _make_ctx(micro_batch=1, global_batch=4)

    result = TrainingFlopsPass().run(g, ctx)
    assert result.metadata["forward_flops"] == 100
    assert result.metadata["backward_flops"] == 100
    assert result.metadata["training_flops"] == 200


def test_training_flops_6p_fallback_when_no_annotations():
    g = _make_graph(metadata={
        "num_layers": 4, "num_layers_traced": 4,
        "seq_len": 128, "total_params": 1000,
    })
    ctx = _make_ctx(micro_batch=1, global_batch=4)

    result = TrainingFlopsPass().run(g, ctx)
    tokens = 128 * 1
    assert result.metadata["forward_flops"] == 2 * 1000 * tokens
    assert result.metadata["backward_flops"] == 4 * 1000 * tokens


# ── Bug 1.2: TrainingMemoryPass ───────────────────────────────────────────────

def test_activation_memory_applies_recompute_policy():
    metadata = {
        "seq_len": 2048, "hidden": 4096, "num_layers": 4,
        "num_layers_traced": 4,
    }

    ctx_none = _make_ctx(tp=1, pp=1, recompute_policy="none")
    ctx_sel = _make_ctx(tp=1, pp=1, recompute_policy="selective")
    ctx_full = _make_ctx(tp=1, pp=1, recompute_policy="full")

    g_none = _make_graph(metadata=metadata)
    g_sel = _make_graph(metadata=metadata)
    g_full = _make_graph(metadata=metadata)

    mem_none = TrainingMemoryPass().run(g_none, ctx_none).metadata["memory_breakdown"]
    mem_sel = TrainingMemoryPass().run(g_sel, ctx_sel).metadata["memory_breakdown"]
    mem_full = TrainingMemoryPass().run(g_full, ctx_full).metadata["memory_breakdown"]

    assert mem_sel.activations == pytest.approx(mem_none.activations * 0.5, rel=0.01)
    assert mem_full.activations == pytest.approx(mem_none.activations * 0.1, rel=0.01)


def test_activation_memory_reads_zero_metadata():
    g = _make_graph(metadata={
        "seq_len": 2048, "hidden": 4096, "num_layers": 4,
        "num_layers_traced": 4, "total_params": 100_000_000,
        "zero": {"weight_shard": 4, "grad_shard": 4, "optstate_shard": 4, "stage": 3},
    })
    ctx = _make_ctx(tp=1, dp=4, zero_stage=3)

    result = TrainingMemoryPass().run(g, ctx)
    mem = result.metadata["memory_breakdown"]

    assert mem.weights > 0
    assert mem.grads > 0
    assert mem.opt_state > 0


def test_activation_memory_pp_inflight():
    metadata = {
        "seq_len": 2048, "hidden": 4096, "num_layers": 4,
        "num_layers_traced": 4,
    }
    ctx_pp1 = _make_ctx(tp=1, pp=1, recompute_policy="none")
    ctx_pp4 = _make_ctx(tp=1, pp=4, recompute_policy="none")

    mem_pp1 = TrainingMemoryPass().run(
        _make_graph(metadata=metadata), ctx_pp1
    ).metadata["memory_breakdown"]
    mem_pp4 = TrainingMemoryPass().run(
        _make_graph(metadata=metadata), ctx_pp4
    ).metadata["memory_breakdown"]

    assert mem_pp4.activations == pytest.approx(mem_pp1.activations * 4, rel=0.01)


# ── Bug 1.1: 1F1B step-time formula ──────────────────────────────────────────

def test_step_time_matches_1f1b_formula():
    pp = 4
    M = 8
    per_stage_us = 1000.0
    stage_time_us = per_stage_us * pp

    metadata = {
        "num_layers": 4, "num_layers_traced": 4,
        "training_flops": 1e12,
    }
    g = _make_graph(metadata=metadata)
    ctx = _make_ctx(pp=pp, micro_batch=1, global_batch=M)

    from unittest.mock import patch, MagicMock
    mock_timeline = MagicMock()
    mock_timeline.total_latency_us = stage_time_us

    with patch("python.zrt.executor.scheduler.DAGScheduler") as MockSched:
        MockSched.return_value.schedule.return_value = mock_timeline
        result = TrainingPipelinePass().run(g, ctx)

    metrics = result.metadata["pipeline_metrics"]
    expected_step_ms = (M + pp - 1) * per_stage_us / 1000.0
    assert metrics.step_time_ms == pytest.approx(expected_step_ms, rel=0.01)


def test_bubble_fraction_correct():
    pp = 4
    M = 8
    per_stage_us = 1000.0
    stage_time_us = per_stage_us * pp

    metadata = {
        "num_layers": 4, "num_layers_traced": 4,
        "training_flops": 1e12,
    }
    g = _make_graph(metadata=metadata)
    ctx = _make_ctx(pp=pp, micro_batch=1, global_batch=M)

    from unittest.mock import patch, MagicMock
    mock_timeline = MagicMock()
    mock_timeline.total_latency_us = stage_time_us

    with patch("python.zrt.executor.scheduler.DAGScheduler") as MockSched:
        MockSched.return_value.schedule.return_value = mock_timeline
        result = TrainingPipelinePass().run(g, ctx)

    metrics = result.metadata["pipeline_metrics"]
    expected_bubble = (pp - 1) / (M + pp - 1)
    assert metrics.bubble_fraction == pytest.approx(expected_bubble, abs=0.01)


def test_steady_steps_equals_num_microbatches():
    pp = 4
    M = 8
    per_stage_us = 1000.0
    stage_time_us = per_stage_us * pp

    metadata = {
        "num_layers": 4, "num_layers_traced": 4,
        "training_flops": 1e12,
    }
    g = _make_graph(metadata=metadata)
    ctx = _make_ctx(pp=pp, micro_batch=1, global_batch=M)

    from unittest.mock import patch, MagicMock
    mock_timeline = MagicMock()
    mock_timeline.total_latency_us = stage_time_us

    with patch("python.zrt.executor.scheduler.DAGScheduler") as MockSched:
        MockSched.return_value.schedule.return_value = mock_timeline
        result = TrainingPipelinePass().run(g, ctx)

    metrics = result.metadata["pipeline_metrics"]
    assert metrics.steady_steps == M - pp + 1  # standard 1F1B: F+B pairs on stage 0
    assert metrics.warmup_steps == pp - 1
    assert metrics.cooldown_steps == pp - 1
