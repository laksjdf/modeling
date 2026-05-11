"""Tests for python.zrt.report.dot_exporter — DOT graph visualization."""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.ir.edge import Edge
from python.zrt.ir.types import DType, TensorMeta
from python.zrt.report.dot_exporter import export_dot, render_dot


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tm(shape: tuple[int, ...], idx: int = 0) -> TensorMeta:
    return TensorMeta(id=f"t{idx}", shape=shape, dtype=DType.FP32,
                      mem_bytes=1)


def _make_graph() -> OpGraph:
    """Build a small 3-node graph spanning 2 layers + 1 ungrouped."""
    g = OpGraph(name="test_model", phase="prefill")
    n0 = OpNode(id="op_0", op_type="RMSNorm", layer="0",
                category="compute", component="RMSNorm",
                inputs=[_tm((128, 4096))], outputs=[_tm((128, 4096))])
    n1 = OpNode(id="op_1", op_type="SelfAttention", layer="0",
                category="compute", component="SelfAttention",
                inputs=[_tm((128, 4096))], outputs=[_tm((128, 64))])
    n2 = OpNode(id="op_2", op_type="comm.all_reduce", layer="1",
                category="communication", component="AllReduce",
                inputs=[_tm((128, 4096))], outputs=[_tm((128, 4096))])
    n3 = OpNode(id="op_3", op_type="aten.view.default",
                category="memory",
                inputs=[_tm((128, 4096))], outputs=[_tm((128, 4096))])
    for n in (n0, n1, n2, n3):
        g.add_node(n)
    g.add_edge(Edge(src="op_0", src_idx=0, dst="op_1", dst_idx=0, tensor=_tm((128, 4096))))
    g.add_edge(Edge(src="op_1", src_idx=0, dst="op_2", dst_idx=0, tensor=_tm((128, 4096))))
    g.add_edge(Edge(src="op_2", src_idx=0, dst="op_3", dst_idx=0, tensor=_tm((128, 4096))))
    return g


# ── Tests ────────────────────────────────────────────────────────────────────

class TestExportDot:

    def test_dot_valid_syntax(self, tmp_path: Path):
        g = _make_graph()
        dot = export_dot(g, tmp_path / "out.dot")
        text = dot.read_text()
        assert "digraph {" in text
        assert "subgraph cluster_" in text
        assert "->" in text

    def test_cluster_count_matches_layers(self, tmp_path: Path):
        g = _make_graph()
        text = export_dot(g, tmp_path / "out.dot").read_text()
        # layers: "0", "1", and __other__ → 3 clusters
        clusters = re.findall(r"subgraph (cluster_\w+)", text)
        assert len(clusters) == 3

    def test_node_count_matches_graph(self, tmp_path: Path):
        g = _make_graph()
        text = export_dot(g, tmp_path / "out.dot").read_text()
        # Each node appears as "node_id" [label=...
        for nid in g.nodes:
            assert f'"{nid}"' in text

    def test_category_colors(self, tmp_path: Path):
        g = _make_graph()
        text = export_dot(g, tmp_path / "out.dot").read_text()
        assert "#4A90D9" in text   # compute
        assert "#E05252" in text   # communication
        assert "#E09A52" in text   # memory

    def test_empty_graph_no_crash(self, tmp_path: Path):
        g = OpGraph(name="empty", phase="forward")
        dot = export_dot(g, tmp_path / "empty.dot")
        text = dot.read_text()
        assert "digraph {" in text
        assert text.count("subgraph") == 0  # no clusters for empty graph


class TestRenderDot:

    def test_render_dot_missing_graphviz(self, tmp_path: Path):
        dot_file = tmp_path / "fake.dot"
        dot_file.write_text("digraph { a -> b }\n")
        with patch("python.zrt.report.dot_exporter.shutil.which", return_value=None):
            assert render_dot(dot_file) is None


class TestGraphDumpPass:

    def test_passthrough(self, tmp_path: Path):
        from python.zrt.transform.debug_pass import GraphDumpPass
        g = _make_graph()
        p = GraphDumpPass("test_dump", tmp_path, render=False)
        assert p.name == "dump_test_dump"
        ctx = None  # not used by GraphDumpPass
        result = p.run(g, ctx)
        # passthrough: same graph returned
        assert result is g
        # DOT file created
        assert (tmp_path / "test_dump.dot").exists()
        text = (tmp_path / "test_dump.dot").read_text()
        assert "digraph {" in text
