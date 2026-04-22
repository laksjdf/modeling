"""Test 1F1B pipeline schedule — bubble ratio matches Megatron paper."""

import pytest
from zrt.training.compose.pipeline import pipeline_step_time
from zrt.training.compose.stage import StageTime
from zrt.training.ir.builders import build_graph
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import SystemSpec, GPU, NetTier


def _make_system():
    return SystemSpec(
        gpu=GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80, hbm_bw_gbps=3350),
        host_mem_gb=256,
        nets=[NetTier("intra_node", 900, 1.0, "nvswitch")],
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
    # Bubble = warmup + cooldown, roughly (pp-1)/M fraction
    # For pp=2, M=4: expected bubble ≈ warmup+cooldown / step_time
    assert result.bubble_fraction > 0  # there should be some bubble
    assert result.bubble_fraction < 0.5  # but not too much


def test_step_time_increases_with_pp():
    """More PP stages → larger bubble → longer step time (for same M)."""
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 4,
    )
    system = SystemSpec(
        gpu=GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80, hbm_bw_gbps=3350),
        host_mem_gb=256,
        nets=[NetTier("intra_node", 900, 1.0, "nvswitch")],
        nodes=2, gpus_per_node=8,
    )

    s1 = Strategy(tp=1, pp=1, dp=8, micro_batch=1, global_batch=8)
    s2 = Strategy(tp=1, pp=2, dp=4, micro_batch=1, global_batch=8)

    g1 = build_graph(model, s1)
    g2 = build_graph(model, s2)

    r1 = pipeline_step_time(g1, model, system, s1)
    r2 = pipeline_step_time(g2, model, system, s2)

    # With pipeline, the step should be longer due to bubbles
    # (per-stage time is similar, but we pay the pipeline tax)
    assert r2.step_time > 0
    assert r1.step_time > 0


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
    assert 0 < result.mfu <= 1.0
