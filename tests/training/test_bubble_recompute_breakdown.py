"""Bubble absolute time + recompute time as a first-class step-time term.

RED→GREEN regression for:
  1. StepResult.bubble = warmup + cooldown (absolute seconds), 0 when pp=1.
  2. StepResult.recompute_time = 0 with no recompute policy, > 0 when
     full/partial recompute is enabled.
  3. bubble attributed OUT of compute_time:
     pipeline_time == compute_time + exposed_comm + bubble
  4. recompute attributed OUT of bwd_compute:
     compute_time == fwd_compute + bwd_compute + recompute_time
  5. step_time identity preserved (attribution does not change totals).
"""

import json
import shutil
from pathlib import Path

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


def test_pipeline_bubble_excluded_from_compute_time_invariant():
    model, system = _model(), _system()
    strategy = Strategy(tp=1, pp=2, dp=4, micro_batch=1, global_batch=8)
    graph = build_graph(model, strategy)

    step = pipeline_step_time(graph, model, system, strategy)

    assert step.bubble > 0.0
    assert step.pipeline_time == pytest.approx(
        step.compute_time + step.exposed_comm + step.bubble, rel=1e-6
    )
    assert step.compute_time == pytest.approx(
        step.fwd_compute + step.bwd_compute + step.recompute_time, rel=1e-6
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

    # pipeline_time = useful compute + exposed comm + pipeline bubble.
    assert step.pipeline_time == pytest.approx(
        step.compute_time + step.exposed_comm + step.bubble, rel=1e-6
    )
    assert step.recompute_time > 0.0
    assert step.bubble == pytest.approx(step.warmup + step.cooldown)


def test_recompute_critical_path_does_not_include_pipeline_bubble():
    """Critical recompute is actual bottleneck-stage recompute work.

    Pipeline schedule/bubble amplification belongs in bubble/schedule terms;
    it must not make recompute_time exceed raw recompute work.
    """
    model, system = _model(), _system()
    strategy = Strategy(
        tp=1, pp=2, dp=1, micro_batch=1, global_batch=8,
        recompute=RecomputePolicy(per_layer={"dense": {"full"}}),
    )
    graph = build_graph(model, strategy)

    step = pipeline_step_time(graph, model, system, strategy)
    s_bot = max(step.per_stage, key=lambda st: st.fwd + st.bwd)

    assert step.recompute_time_raw > 0.0
    assert step.recompute_time == pytest.approx(8 * s_bot.recompute)
    assert step.recompute_time <= step.recompute_time_raw + 1e-9


def test_html_export_surfaces_recompute_and_bubble():
    """The HTML report must visibly carry recompute + bubble: JS constants,
    the metric cards, and the step-time breakdown section."""
    from zrt.training.io.html_exporter import export_estimate_html
    from zrt.training.models.flops import op_cost

    model, system = _model(), _system()
    # TP*CP*PP*DP must equal world_size (8) for estimate()'s validate().
    strategy = Strategy(
        tp=1, cp=1, pp=2, ep=1, dp=4, micro_batch=1, global_batch=8,
        recompute=RecomputePolicy(per_layer={"dense": {"full"}}),
    )
    graph = build_graph(model, strategy)
    from zrt.training.search.estimator import estimate
    report = estimate(model, system, strategy, graph=graph)
    op_costs = {op.name: op_cost(op, model, system) for op in graph.ops}

    out_dir = Path("output") / "test_html_bubble_recompute"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    try:
        out = out_dir / "r.html"
        export_estimate_html(report=report, graph=graph, model=model,
                             system=system, strategy=strategy,
                             op_costs=op_costs, output_path=out)
        html = out.read_text(encoding="utf-8")

        assert "recompute_time_ms" in html
        assert "recompute_time_raw_ms" in html
        assert "Recompute (crit. path)" in html
        assert "step time breakdown" in html
        assert "pipeline bubble" in html
        assert report.recompute_time_ms > 0.0  # this config does recompute
    finally:
        if out_dir.exists():
            shutil.rmtree(out_dir)


def test_html_json_literal_escapes_script_breakout_and_control_chars():
    from zrt.training.io.html_exporter import _json_parse_literal_for_script

    data = {"name": "</SCRIPT>\n'x'", "template": "${not_executed}"}

    literal = _json_parse_literal_for_script(data)
    payload = json.loads(literal)

    assert "</script" not in literal.lower()
    assert json.loads(payload) == data


def test_search_report_surfaces_bubble_and_recompute():
    """grid-search / estimate result report (search/report.py) must carry
    bubble absolute time and recompute pre/post-hide times."""
    from zrt.training.search.estimator import estimate
    from zrt.training.search.report import report_to_dict, report_summary

    model, system = _model(), _system()
    strategy = Strategy(
        tp=1, cp=1, pp=2, ep=1, dp=4, micro_batch=1, global_batch=8,
        recompute=RecomputePolicy(per_layer={"dense": {"full"}}),
    )
    report = estimate(model, system, strategy)

    d = report_to_dict(report)
    for key in ("bubble_time_ms", "recompute_time_ms", "recompute_time_raw_ms"):
        assert key in d, f"{key} missing from report_to_dict"
    assert d["bubble_time_ms"] > 0.0
    assert d["recompute_time_raw_ms"] > 0.0

    txt = report_summary(report)
    assert "Recompute (critical path)" in txt
    assert "Pipeline Bubble" in txt
    assert "Recompute (pre/post pipeline-hide)" in txt
    assert "raw (pre-hide, NOT in step)" in txt
    assert f"({report.bubble_time_ms:.1f} ms)" in txt
    # The recompute/bubble lines we added must not introduce CJK (the
    # pre-existing table uses box-drawing '─' but no wide CJK that would
    # break <NNs> column alignment / crash GBK consoles).
    added = [l for l in txt.splitlines()
             if "Recompute" in l or "pre-hide" in l or "Bubble:" in l]
    for line in added:
        assert all(ord(c) < 0x2E80 for c in line), f"CJK in: {line!r}"


def test_search_results_table_has_recompute_columns():
    """grid-search results DataFrame (training_search_util.format_results)
    must expose recompute critical + raw alongside bubble.

    Regression guard for the columns added in PR #109 — keeps the search
    results table (results_summary.csv / printed top-5 / best-config Excel
    grouping) from silently dropping recompute time again.
    """
    from zrt.training.search.estimator import estimate
    from zrt.training.search.training_search_util import format_results

    model, system = _model(), _system()
    strategy = Strategy(
        tp=1, cp=1, pp=2, ep=1, dp=4, micro_batch=1, global_batch=8,
        recompute=RecomputePolicy(per_layer={"dense": {"full"}}),
    )
    report = estimate(model, system, strategy)
    df = format_results([report], [{"model": "m"}])

    for col in ("compute_time_ms", "recompute_time_ms", "recompute_time_raw_ms",
                "bubble_time_ms", "bubble_fraction"):
        assert col in df.columns, f"{col} missing from results table"
    row = df.iloc[0]
    assert row["pipeline_time_ms"] == pytest.approx(
        row["compute_time_ms"] + row["exposed_comm_ms"] + row["bubble_time_ms"],
        abs=0.01,
    )
    assert df.iloc[0]["recompute_time_raw_ms"] > 0.0
