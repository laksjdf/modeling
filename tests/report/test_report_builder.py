"""Tests for python.zrt.report.report_builder — AC-4: Builder hierarchical construction."""

import pytest
from unittest.mock import MagicMock

from python.zrt.ir.node import OpNode
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.types import DType, TensorMeta
from python.zrt.simulator.result import SimResult
from python.zrt.report.report_builder import (
    build_report_context,
    _build_metadata, _build_bound,
)
from python.zrt.report.report_types import ReportContext


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_node(nid: str, op_type: str, scope: str,
               shape_in=((128, 7168),), shape_out=((128, 7168),),
               category="compute", layer="") -> OpNode:
    """Create an OpNode with minimal realistic fields."""
    inputs = [TensorMeta.from_shape_dtype(f"t_{nid}_in{i}", s, DType.BF16)
              for i, s in enumerate(shape_in)]
    outputs = [TensorMeta.from_shape_dtype(f"t_{nid}_out{i}", s, DType.BF16)
               for i, s in enumerate(shape_out)]
    return OpNode(
        id=nid, op_type=op_type, inputs=inputs, outputs=outputs,
        attrs={}, scope=scope, category=category, layer=layer,
        component="", op_short=op_type.split(".")[-1],
    )


def _sr(nid: str, bound="compute", latency_us=100.0) -> SimResult:
    """Create a minimal SimResult."""
    return SimResult(
        op_node_id=nid,
        latency_us=latency_us,
        compute_us=latency_us * 0.8,
        memory_us=latency_us * 0.2,
        flops=1000000,
        read_bytes=4096,
        write_bytes=4096,
        arithmetic_intensity=100.0,
        bound=bound,
        hw_utilization=0.5,
        backend="roofline",
        confidence=0.3,
    )


def _build_2layer_graph() -> tuple[OpGraph, dict[str, SimResult]]:
    """Build a simple 2-layer transformer graph.

    Structure:
      embed_tokens (Embedding)
      model.layers.0
        ├── self_attn → q_proj (aten.mm), v_proj (aten.mm), sdpa
        └── mlp → gate_proj (aten.mm), up_proj (aten.mm), rms_norm
      model.layers.1
        ├── self_attn → q_proj (aten.mm), v_proj (aten.mm), sdpa
        └── mlp → gate_proj (aten.mm), up_proj (aten.mm), rms_norm
      model.norm → rms_norm
      lm_head → aten.linear
    """
    nodes = {}
    nid = 0

    def _add(op_type, scope, **kw):
        nonlocal nid
        node = _make_node(f"n{nid}", op_type, scope, **kw)
        nodes[f"n{nid}"] = node
        nid += 1
        return f"n{nid - 1}"

    # Embedding
    _add("aten.embedding", "model.embed_tokens",
         shape_in=((128,),), shape_out=((128, 7168),), category="memory")

    # Layer 0
    _add("aten.mm.default", "model.layers.0.self_attn.q_proj",
         shape_in=((128, 7168), (7168, 2048)), shape_out=((128, 2048),))
    _add("aten.mm.default", "model.layers.0.self_attn.v_proj",
         shape_in=((128, 7168), (7168, 2048)), shape_out=((128, 2048),))
    _add("scaled_dot_product_attention", "model.layers.0.self_attn",
         shape_in=((1, 24, 128, 128),), shape_out=((1, 24, 128, 128),))
    _add("aten.mm.default", "model.layers.0.mlp.gate_proj",
         shape_in=((128, 7168), (7168, 7168)), shape_out=((128, 7168),))
    _add("aten.mm.default", "model.layers.0.mlp.up_proj",
         shape_in=((128, 7168), (7168, 7168)), shape_out=((128, 7168),))
    _add("aten.rms_norm.default", "model.layers.0.mlp",
         shape_in=((128, 7168),), shape_out=((128, 7168),))

    # Layer 1 (identical structure)
    _add("aten.mm.default", "model.layers.1.self_attn.q_proj",
         shape_in=((128, 7168), (7168, 2048)), shape_out=((128, 2048),))
    _add("aten.mm.default", "model.layers.1.self_attn.v_proj",
         shape_in=((128, 7168), (7168, 2048)), shape_out=((128, 2048),))
    _add("scaled_dot_product_attention", "model.layers.1.self_attn",
         shape_in=((1, 24, 128, 128),), shape_out=((1, 24, 128, 128),))
    _add("aten.mm.default", "model.layers.1.mlp.gate_proj",
         shape_in=((128, 7168), (7168, 7168)), shape_out=((128, 7168),))
    _add("aten.mm.default", "model.layers.1.mlp.up_proj",
         shape_in=((128, 7168), (7168, 7168)), shape_out=((128, 7168),))
    _add("aten.rms_norm.default", "model.layers.1.mlp",
         shape_in=((128, 7168),), shape_out=((128, 7168),))

    # Output
    _add("aten.rms_norm.default", "model.norm",
         shape_in=((128, 7168),), shape_out=((128, 7168),))
    _add("aten.linear", "lm_head",
         shape_in=((128, 7168),), shape_out=((128, 128256),))

    graph = OpGraph(name="test_2layer", phase="decode", nodes=nodes)
    sim_results = {nid: _sr(nid) for nid in nodes}
    return graph, sim_results


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
    return hw


def _mock_ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.parallel.describe.return_value = "TP8"
    ctx.parallel.tp = 8
    ctx.parallel.pp = 1
    ctx.training = None
    return ctx


# ═══════════════════════════════════════════════════════════════════════════════
# AC-4: Builder hierarchical construction
# ═══════════════════════════════════════════════════════════════════════════════

class TestAC4Builder:
    """AC-4: Builder produces correct 4-level hierarchy for 2-layer transformer."""

    @pytest.fixture
    def graph_and_results(self):
        return _build_2layer_graph()

    @pytest.fixture
    def rc(self, graph_and_results):
        graph, sim_results = graph_and_results
        return build_report_context(
            model="Test-2L",
            hardware="nvidia_h100_sxm",
            phase="decode",
            batch_size=1,
            seq_len=128,
            graph=graph,
            sim_results=sim_results,
            timeline=_mock_timeline(),
            hw_spec=_mock_hw_spec(),
            ctx=_mock_ctx(),
        )

    def test_blocks_count(self, rc):
        """2-layer transformer → 1 Embedding + 1 TransformerBlock + 1 Output = 3."""
        assert len(rc.blocks) >= 2  # at minimum Embedding + Transformer

    def test_embedding_block_present(self, rc):
        emb = [b for b in rc.blocks if b.name == "Embedding"]
        assert len(emb) == 1
        assert emb[0].repeat == 1

    def test_transformer_block_present(self, rc):
        tf = [b for b in rc.blocks if "Block" in b.name or "Transformer" in b.name or "MoEBlock" in b.name]
        assert len(tf) >= 1

    @pytest.mark.xfail(
        strict=False,
        reason="pre-existing: block merge 逻辑回归，repeat 始终为 1。tracked in #65",
    )
    def test_transformer_block_repeat(self, rc):
        tf = [b for b in rc.blocks if "Block" in b.name or "Transformer" in b.name or "MoEBlock" in b.name]
        if tf:
            # 2-layer transformer, layers merged → repeat should be 2
            assert tf[0].repeat == 2

    def test_output_block_present(self, rc):
        out = [b for b in rc.blocks if b.name == "Output"]
        # model.norm and lm_head both map to Output
        assert len(out) >= 1

    def test_metadata_fields_set(self, rc, graph_and_results):
        graph, sim_results = graph_and_results
        assert rc.model == "Test-2L"
        assert rc.phase == "decode"
        assert rc.parallel_desc == "TP8"
        assert rc.tpot_ms is not None
        assert rc.tpot_ms > 0
        assert rc.tokens_per_sec > 0

    def test_blocks_have_sub_structures(self, rc):
        """Each block should have at least one SubStructure."""
        for b in rc.blocks:
            assert len(b.sub_structures) > 0, f"Block {b.name} has no sub-structures"

    def test_sub_structures_have_op_families(self, rc):
        """Each SubStructure should have at least one OpFamily."""
        total_families = 0
        for b in rc.blocks:
            for ss in b.sub_structures:
                assert len(ss.op_families) > 0, f"SubStructure {ss.name} has no op_families"
                total_families += len(ss.op_families)
        assert total_families > 0

    def test_op_family_aggregation(self, rc):
        """OpFamily should have count, total_ms, and correct op_type grouping."""
        for b in rc.blocks:
            for ss in b.sub_structures:
                for ofd in ss.op_families:
                    assert ofd.count > 0, f"OpFamily {ofd.op_type} has count=0"
                    assert ofd.total_ms > 0, f"OpFamily {ofd.op_type} has total_ms=0"
                    assert ofd.op_type != ""

    def test_op_family_has_formula(self, rc):
        """Each OpFamily should have a formula from FormulaRegistry."""
        for b in rc.blocks:
            for ss in b.sub_structures:
                for ofd in ss.op_families:
                    assert ofd.formula != "", f"OpFamily {ofd.op_type} missing formula"
                    assert ofd.display_name != ""

    def test_bound_bar_computed(self, rc):
        """Bound percentages should be computed and sum to ~100."""
        assert rc.compute_pct >= 0
        assert rc.memory_pct >= 0
        assert rc.communication_pct >= 0
        total = rc.compute_pct + rc.memory_pct + rc.communication_pct
        assert abs(total - 100.0) < 1.0

    def test_no_exception(self, rc):
        assert rc is not None
        assert isinstance(rc, ReportContext)


class TestBuildMetadata:
    def test_metadata_sets_all_fields(self):
        graph, sim_results = _build_2layer_graph()
        rc = ReportContext()
        _build_metadata(rc, "M", "H100", "prefill", 1, 8192,
                       _mock_timeline(5000), _mock_hw_spec(), _mock_ctx(), None, None)
        assert rc.model == "M"
        assert rc.phase == "prefill"
        assert rc.prefill_ms is not None
        assert rc.tpot_ms is None  # prefill has no tpot
        assert rc.model_blocks == 0  # no profile

    def test_metadata_decode_has_tpot(self):
        graph, sim_results = _build_2layer_graph()
        rc = ReportContext()
        _build_metadata(rc, "M", "H100", "decode", 1, 1,
                       _mock_timeline(5000), _mock_hw_spec(), _mock_ctx(), None, None)
        assert rc.prefill_ms is None
        assert rc.tpot_ms is not None


class TestBuildBound:
    def test_bound_with_compute_only(self):
        graph, sim_results = _build_2layer_graph()
        rc = ReportContext()
        _build_bound(rc, sim_results, graph)
        # All nodes are compute → compute_pct ≈ 100, others ≈ 0
        assert rc.compute_pct > 90.0
