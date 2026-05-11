"""Tests for python.zrt.report.summary — AC-5: E2ESummary extension integration."""

import pytest
from unittest.mock import MagicMock

from python.zrt.ir.node import OpNode
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.types import DType, TensorMeta
from python.zrt.simulator.result import SimResult
from python.zrt.report.summary import E2ESummary, build_summary


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_node(nid: str, op_type: str, scope: str,
               shape_in=((128, 7168),), shape_out=((128, 7168),),
               category="compute") -> OpNode:
    inputs = [TensorMeta.from_shape_dtype(f"t_{nid}_in{i}", s, DType.BF16)
              for i, s in enumerate(shape_in)]
    outputs = [TensorMeta.from_shape_dtype(f"t_{nid}_out{i}", s, DType.BF16)
               for i, s in enumerate(shape_out)]
    return OpNode(
        id=nid, op_type=op_type, inputs=inputs, outputs=outputs,
        attrs={}, scope=scope, category=category, layer="",
        component="", op_short=op_type.split(".")[-1],
    )


def _sr(nid: str, bound="compute", latency_us=100.0) -> SimResult:
    return SimResult(
        op_node_id=nid, latency_us=latency_us, compute_us=latency_us * 0.8,
        memory_us=latency_us * 0.2, flops=1000000,
        read_bytes=4096, write_bytes=4096, arithmetic_intensity=100.0,
        bound=bound, hw_utilization=0.5, backend="roofline", confidence=0.3,
    )


def _mock_timeline(latency_us=1400.0) -> MagicMock:
    t = MagicMock()
    t.total_latency_us = latency_us
    t.compute_time_us = latency_us * 0.8
    t.comm_time_us = 0.0
    t.overlap_us = 0.0
    return t


def _mock_hw_spec() -> MagicMock:
    hw = MagicMock()
    hw.nodes = 1
    hw.gpus_per_node = 8
    hw.peak_flops.return_value = 200e12  # 200 TFLOPS
    hw.hbm_bandwidth.return_value = 3.35e12  # 3.35 TB/s
    return hw


def _mock_ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.parallel.describe.return_value = "TP8"
    ctx.parallel.tp = 8
    ctx.parallel.pp = 1
    ctx.training = None
    return ctx


def _build_minimal_graph():
    """Minimal graph: 1 layer with 2 mm ops."""
    nodes = {}
    nid = 0
    for scope in ["model.layers.0.self_attn.q_proj",
                   "model.layers.0.self_attn.v_proj",
                   "model.norm"]:
        nodes[f"n{nid}"] = _make_node(f"n{nid}",
            "aten.rms_norm.default" if "norm" in scope else "aten.mm.default",
            scope, shape_in=((128, 7168), (7168, 2048)),
            shape_out=((128, 2048),))
        nid += 1

    graph = OpGraph(name="test_min", phase="decode", nodes=nodes)
    sim_results = {nid: _sr(nid) for nid in nodes}
    return graph, sim_results


# ═══════════════════════════════════════════════════════════════════════════════
# AC-5: E2ESummary extension
# ═══════════════════════════════════════════════════════════════════════════════

class TestAC5E2ESummaryExtended:
    """AC-5: E2ESummary with Phase 1 fields."""

    def test_backward_compatible_construction(self):
        """E2ESummary can be constructed without new fields (backward compat)."""
        s = E2ESummary(
            model="test", hardware="h100", phase="decode",
            parallel_desc="TP8", batch_size=1, seq_len=1,
            latency_ms=1.0, tokens_per_sec=1.0,
            ttft_ms=None, tpot_ms=1.0,
            compute_ms=1.0, comm_ms=0.0,
            exposed_comm_ms=0.0, overlap_ratio=1.0,
            mfu=0.5, hbm_bandwidth_util=0.3,
            total_flops=1000, total_bytes=2000,
            read_bytes=1000, write_bytes=1000,
            arithmetic_intensity=0.5,
            by_component={}, by_layer=[], top_bottleneck_ops=[],
        )
        assert s.report_context is None
        assert s.bound_compute_pct == 0.0
        assert s.model_blocks == 0
        assert s.mtp_depth == 1

    def test_new_fields_have_defaults(self):
        """All Phase 1 fields should have sensible defaults."""
        fields = E2ESummary.__dataclass_fields__
        assert "report_context" in fields
        assert "bound_compute_pct" in fields
        assert "bound_memory_pct" in fields
        assert "bound_comm_pct" in fields
        assert "memory_per_gpu_gb" in fields
        assert "model_blocks" in fields
        assert "mtp_depth" in fields
        assert "effective_tpot_ms" in fields

    def test_new_fields_assignable(self):
        s = E2ESummary(
            model="test", hardware="h100", phase="decode",
            parallel_desc="TP8", batch_size=1, seq_len=1,
            latency_ms=1.0, tokens_per_sec=1.0,
            ttft_ms=None, tpot_ms=1.0,
            compute_ms=1.0, comm_ms=0.0,
            exposed_comm_ms=0.0, overlap_ratio=1.0,
            mfu=0.5, hbm_bandwidth_util=0.3,
            total_flops=1000, total_bytes=2000,
            read_bytes=1000, write_bytes=1000,
            arithmetic_intensity=0.5,
            by_component={}, by_layer=[], top_bottleneck_ops=[],
            report_context=None, bound_compute_pct=70.0,
            bound_memory_pct=20.0, bound_comm_pct=10.0,
            memory_per_gpu_gb=52.0, model_blocks=61,
            mtp_depth=3, mtp_acceptance_rate=0.65,
            effective_tpot_ms=5.4,
        )
        assert s.bound_compute_pct == 70.0
        assert s.memory_per_gpu_gb == 52.0
        assert s.model_blocks == 61
        assert s.mtp_depth == 3
        assert s.effective_tpot_ms == 5.4


class TestBuildSummaryExtended:
    """AC-5: build_summary with build_report parameter."""

    def test_build_summary_without_report(self):
        """build_summary with build_report=False (default) — no report_context."""
        graph, sim_results = _build_minimal_graph()
        s = build_summary(
            model="test", hardware="h100", phase="decode",
            batch_size=1, seq_len=1,
            graph=graph, sim_results=sim_results,
            timeline=_mock_timeline(), hw_spec=_mock_hw_spec(),
        )
        assert s.report_context is None
        assert s.model == "test"
        assert s.phase == "decode"
        assert s.bound_compute_pct >= 0
        assert s.bound_compute_pct + s.bound_memory_pct + s.bound_comm_pct > 90.0
        assert s.model_blocks >= 1  # has model.layers.N

    def test_build_summary_bound_decomposition(self):
        """Bound decomposition should be computed from sim_results."""
        graph, sim_results = _build_minimal_graph()
        s = build_summary(
            model="test", hardware="h100", phase="decode",
            batch_size=1, seq_len=1,
            graph=graph, sim_results=sim_results,
            timeline=_mock_timeline(), hw_spec=_mock_hw_spec(),
        )
        total_pct = s.bound_compute_pct + s.bound_memory_pct + s.bound_comm_pct
        assert abs(total_pct - 100.0) < 1.0

    def test_build_summary_model_blocks_count(self):
        """model_blocks should count numeric depth-3 nodes."""
        graph, sim_results = _build_minimal_graph()
        s = build_summary(
            model="test", hardware="h100", phase="decode",
            batch_size=1, seq_len=1,
            graph=graph, sim_results=sim_results,
            timeline=_mock_timeline(), hw_spec=_mock_hw_spec(),
        )
        # graph has "model.layers.0" → 1 block
        assert s.model_blocks == 1

    def test_build_summary_with_report(self):
        """build_summary with build_report=True creates report_context."""
        graph, sim_results = _build_minimal_graph()
        s = build_summary(
            model="test", hardware="h100", phase="decode",
            batch_size=1, seq_len=1,
            graph=graph, sim_results=sim_results,
            timeline=_mock_timeline(), hw_spec=_mock_hw_spec(),
            build_report=True, ctx=_mock_ctx(),
        )
        assert s.report_context is not None
        assert s.report_context.model == "test"
        assert len(s.report_context.blocks) > 0

    def test_build_summary_report_without_ctx_skips(self):
        """build_report=True without ctx should not crash."""
        graph, sim_results = _build_minimal_graph()
        s = build_summary(
            model="test", hardware="h100", phase="decode",
            batch_size=1, seq_len=1,
            graph=graph, sim_results=sim_results,
            timeline=_mock_timeline(), hw_spec=_mock_hw_spec(),
            build_report=True, ctx=None,
        )
        assert s.report_context is None  # skipped because ctx is None

    def test_build_summary_old_signature_still_works(self):
        """Callers using old signature (no build_report) should work."""
        graph, sim_results = _build_minimal_graph()
        s = build_summary(
            model="test", hardware="h100", phase="decode",
            batch_size=1, seq_len=1,
            graph=graph, sim_results=sim_results,
            timeline=_mock_timeline(), hw_spec=_mock_hw_spec(),
        )
        assert s is not None

    def test_build_summary_tpot_decode(self):
        """decode phase should populate tpot_ms."""
        graph, sim_results = _build_minimal_graph()
        s = build_summary(
            model="test", hardware="h100", phase="decode",
            batch_size=1, seq_len=1,
            graph=graph, sim_results=sim_results,
            timeline=_mock_timeline(1000), hw_spec=_mock_hw_spec(),
        )
        assert s.tpot_ms is not None and s.tpot_ms > 0
        # decode: tokens_per_sec = batch_size / latency_s
        assert s.tokens_per_sec > 0

    def test_build_summary_ttft_prefill(self):
        """prefill phase should populate ttft_ms."""
        graph, sim_results = _build_minimal_graph()
        s = build_summary(
            model="test", hardware="h100", phase="prefill",
            batch_size=1, seq_len=128,
            graph=graph, sim_results=sim_results,
            timeline=_mock_timeline(5000), hw_spec=_mock_hw_spec(),
        )
        assert s.ttft_ms is not None and s.ttft_ms > 0
        assert s.tpot_ms is None
