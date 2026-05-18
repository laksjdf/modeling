"""Test 1F1B pipeline schedule — bubble ratio matches Megatron paper."""

import pytest
from zrt.training.compose.schedules import OneF1BComposer, pipeline_step_time
from zrt.training.compose.stage import StageTime
from zrt.training.ir.builders import build_graph
from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.strategy import Strategy
from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.spec.system import SystemSpec, GPU


def _make_system():
    return SystemSpec(
        gpu=GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80, hbm_bw_gbps=3350),
        host_mem_gb=256,
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0, topology="all_to_all", num_devices=8),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=400, latency_us=5.0, topology="fat_tree"),
        ),
        nodes=1, gpus_per_node=8,
    )


def test_single_stage_no_bubble():
    """PP=1 should have zero bubble."""
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 4,
    )
    system = _make_system()
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=4)
    graph = build_graph(model, strategy)

    result = pipeline_step_time(graph, model, system, strategy)
    assert result.bubble_fraction == 0.0
    assert result.warmup == 0.0
    assert result.cooldown == 0.0


def test_single_stage_dp_allreduce_overlaps_backward_when_enabled():
    """PP=1: DP grad reduce overlaps backward via steady-BWD window.
    bwd=2.0, M=4 → steady_bwd_total=8.0. With default ratio=0.5,
    window=0+0.5*8.0=4.0. dp_ar_time=3.0 < 4.0 → fully hidden."""
    stage = [StageTime(fwd=1.0, bwd=2.0)]
    strategy = Strategy(tp=1, pp=1, dp=4, micro_batch=1, global_batch=4)

    result = OneF1BComposer().compose(stage, M=4, pp=1, dp_ar_time=3.0, strategy=strategy)

    assert result.step_time == 12.0
    assert result.dp_exposed == 0.0


def test_single_stage_dp_allreduce_exposed_when_overlap_disabled():
    stage = [StageTime(fwd=1.0, bwd=2.0)]
    strategy = Strategy(
        tp=1, pp=1, dp=4, micro_batch=1, global_batch=4,
        dp_overlap_in_bubble=False,
    )

    result = OneF1BComposer().compose(stage, M=4, pp=1, dp_ar_time=3.0, strategy=strategy)

    assert result.step_time == 15.0
    assert result.dp_exposed == 3.0


def test_pp2_bubble_ratio():
    """PP=2 should have bubble ≈ (pp-1)/(pp-1+M) ≈ 1/(1+M)."""
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 4,
    )
    system = _make_system()
    M = 4
    strategy = Strategy(tp=1, pp=2, dp=1, micro_batch=1, global_batch=M)
    graph = build_graph(model, strategy)

    result = pipeline_step_time(graph, model, system, strategy)
    # Bubble = warmup + cooldown, roughly (pp-1)/(pp-1+M) fraction
    # For pp=2, M=4: expected bubble ≈ 1/(1+4) = 0.2
    expected_bubble = (strategy.pp - 1) / (strategy.pp - 1 + M)
    assert result.bubble_fraction == pytest.approx(expected_bubble, rel=0.05)


def test_step_time_increases_with_pp():
    """More PP stages → larger bubble fraction (for same M, same dp)."""
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 4,
    )
    system = SystemSpec(
        gpu=GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80, hbm_bw_gbps=3350),
        host_mem_gb=256,
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0, topology="all_to_all", num_devices=8),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=400, latency_us=5.0, topology="fat_tree"),
        ),
        nodes=2, gpus_per_node=8,
    )

    # Keep dp constant to isolate the effect of PP
    s1 = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=1)
    s2 = Strategy(tp=1, pp=2, dp=1, micro_batch=1, global_batch=1)

    g1 = build_graph(model, s1)
    g2 = build_graph(model, s2)

    r1 = pipeline_step_time(g1, model, system, s1)
    r2 = pipeline_step_time(g2, model, system, s2)

    # With pipeline, bubble fraction increases (more idle time)
    # while step time may stay similar due to microbatch pipelining
    assert r2.bubble_fraction > r1.bubble_fraction
    assert r1.bubble_fraction == 0.0  # PP=1 has no bubble


def test_mfu_positive_and_bounded():
    """MFU should be between 0 and 1."""
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 4,
    )
    system = _make_system()
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=4)
    graph = build_graph(model, strategy)

    result = pipeline_step_time(graph, model, system, strategy)
    assert 0 < result.mfu < 1.0
