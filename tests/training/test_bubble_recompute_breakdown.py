"""Bubble absolute time + recompute time as a first-class step-time term.

RED→GREEN regression for:
  1. StepResult.bubble = warmup + cooldown (absolute seconds), 0 when pp=1.
  2. StepResult.recompute_time = 0 with no recompute policy, > 0 when
     full/partial recompute is enabled.
  3. recompute attributed OUT of bwd_compute:
     compute_time == fwd_compute + bwd_compute + recompute_time
  4. step_time identity preserved (attribution does not change totals).
"""

import pytest

from zrt.training.compose.schedules import OneF1BComposer, pipeline_step_time
from zrt.training.compose.stage import StageTime
from zrt.training.ir.builders import build_graph
from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.strategy import Strategy, RecomputePolicy
from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.spec.system import SystemSpec, GPU


def _system():
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


def _model(n_layers=4):
    return ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * n_layers,
    )


# ── Bubble absolute time ─────────────────────────────────────────────────

def test_bubble_absolute_equals_warmup_plus_cooldown():
    stage = [StageTime(fwd=1.0, bwd=2.0) for _ in range(2)]
    strategy = Strategy(tp=1, pp=2, dp=1, micro_batch=1, global_batch=4)

    r = OneF1BComposer().compose(stage, M=4, pp=2, dp_ar_time=0.0, strategy=strategy)

    assert r.bubble == pytest.approx(r.warmup + r.cooldown)
    assert r.bubble > 0.0


def test_bubble_zero_when_single_stage():
    stage = [StageTime(fwd=1.0, bwd=2.0)]
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=4)

    r = OneF1BComposer().compose(stage, M=4, pp=1, dp_ar_time=0.0, strategy=strategy)

    assert r.bubble == 0.0


# ── Recompute as a separate term ─────────────────────────────────────────

def test_recompute_time_zero_without_policy():
    model, system = _model(), _system()
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=4)
    graph = build_graph(model, strategy)

    step = pipeline_step_time(graph, model, system, strategy)

    assert step.recompute_time == 0.0


def test_recompute_time_positive_with_full_recompute():
    model, system = _model(), _system()
    strategy = Strategy(
        tp=1, pp=1, dp=1, micro_batch=1, global_batch=4,
        recompute=RecomputePolicy(per_layer={"dense": {"full"}}),
    )
    graph = build_graph(model, strategy)

    step = pipeline_step_time(graph, model, system, strategy)

    assert step.recompute_time > 0.0


def test_recompute_excluded_from_bwd_compute_invariant():
    model, system = _model(), _system()
    strategy = Strategy(
        tp=1, pp=1, dp=1, micro_batch=1, global_batch=4,
        recompute=RecomputePolicy(per_layer={"dense": {"full"}}),
    )
    graph = build_graph(model, strategy)

    step = pipeline_step_time(graph, model, system, strategy)

    # New compute decomposition: recompute is its own term, not in bwd.
    assert step.compute_time == pytest.approx(
        step.fwd_compute + step.bwd_compute + step.recompute_time, rel=1e-6
    )
    # Top-level step identity still holds.
    assert step.step_time == pytest.approx(
        step.pipeline_time + step.optimizer_time + step.optimizer_comm, rel=1e-6
    )


def test_recompute_raw_zero_without_policy():
    model, system = _model(), _system()
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=4)
    graph = build_graph(model, strategy)

    step = pipeline_step_time(graph, model, system, strategy)

    assert step.recompute_time_raw == 0.0


def _moe_bottleneck_model():
    """3 dense + 13 heavy-MoE layers; small dense attention so a MoE stage
    gates the pipeline (mirrors the DeepSeek-style report the user hit)."""
    return ModelSpec(
        hidden=4096, ffn=8192, num_heads=16, num_kv_heads=4, head_dim=128,
        vocab=32000, seq_len=512,
        layers=[LayerKind.DENSE] * 3 + [LayerKind.MOE] * 13,
        num_experts=256, top_k=8, moe_ffn=8192, n_shared_experts=2,
    )


def test_dense_recompute_pipeline_hidden_raw_visible():
    """User-reported case: PP=16 MoE model, recompute only on dense.

    Dense recompute runs inside a non-bottleneck stage → critical-path
    recompute_time == 0 (correctly, step_time is unchanged), but the raw
    magnitude must still be > 0 so the user can see recompute is active.
    """
    model, system = _moe_bottleneck_model(), _system()
    common = dict(tp=4, cp=1, pp=16, ep=1, dp=2,
                  micro_batch=1, global_batch=32)
    rc = Strategy(**common,
                  recompute=RecomputePolicy(per_layer={"dense": {"attn_block"}}))
    no_rc = Strategy(**common)

    g_rc = build_graph(model, rc)
    g_no = build_graph(model, no_rc)
    s_rc = pipeline_step_time(g_rc, model, system, rc)
    s_no = pipeline_step_time(g_no, model, system, no_rc)

    # Some non-bottleneck (dense) stage actually did the recompute work.
    assert max(st.recompute for st in s_rc.per_stage) > 0.0
    # It is hidden behind the heavier MoE stage → 0 on the critical path,
    # and step_time is unchanged vs. no recompute.
    assert s_rc.recompute_time == 0.0
    assert s_rc.step_time == pytest.approx(s_no.step_time, rel=1e-9)
    # But the raw magnitude is visible and positive.
    assert s_rc.recompute_time_raw > 0.0
    # Invariant still holds with the critical-path term only.
    assert s_rc.compute_time == pytest.approx(
        s_rc.fwd_compute + s_rc.bwd_compute + s_rc.recompute_time, rel=1e-6
    )


def test_recompute_attribution_preserves_step_time():
    """Turning the attribution on must not change step_time vs. the value
    the composer timeline produces (recompute stays on the bwd critical
    path; only its *reporting* moves out of bwd_compute)."""
    model, system = _model(), _system()
    strategy = Strategy(
        tp=1, pp=2, dp=1, micro_batch=1, global_batch=8,
        recompute=RecomputePolicy(per_layer={"dense": {"full"}}),
    )
    graph = build_graph(model, strategy)

    step = pipeline_step_time(graph, model, system, strategy)

    # pipeline_time = compute_time + exposed_comm, with compute_time now
    # carrying the recompute term explicitly.
    assert step.pipeline_time == pytest.approx(
        step.compute_time + step.exposed_comm, rel=1e-6
    )
    assert step.recompute_time > 0.0
    assert step.bubble == pytest.approx(step.warmup + step.cooldown)
