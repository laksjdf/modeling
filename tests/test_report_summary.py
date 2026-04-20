"""Tests for python.zrt.report.E2ESummary and build_summary()."""
import pytest
from python.zrt.ir.node import OpNode
from python.zrt.ir.edge import Edge
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.types import TensorMeta, DType
from python.zrt.executor.scheduler import DAGScheduler, Timeline, ScheduledOp
from python.zrt.simulator.result import SimResult
from python.zrt.report import E2ESummary, build_summary
import python.zrt.hardware.registry as hw_registry


# ── helpers ───────────────────────────────────────────────────────────────────

def _t(tid, shape=(1, 128)):
    return TensorMeta.from_shape_dtype(tid, shape, DType.BF16)


def _node(nid, op_type="aten.mm.default", scope="", layer="0",
          module_class="", category="compute", stream_id=0, latency_us=10.0):
    n = OpNode(
        id=nid, op_type=op_type,
        inputs=[_t(f"{nid}_in", (128, 512))],
        outputs=[_t(f"{nid}_out", (128, 512))],
        scope=scope, layer=layer, module_class=module_class, category=category,
    )
    n.annotations["stream_id"]   = stream_id
    n.annotations["stream_type"] = "comm" if category == "communication" else "compute"
    n.annotations["latency_us"]  = latency_us
    return n


def _edge(src, dst):
    return Edge(src=src, src_idx=0, dst=dst, dst_idx=0, tensor=_t("e"))


def _graph(nodes, edges, name="test", phase="prefill"):
    return OpGraph(name=name, phase=phase,
                   nodes={n.id: n for n in nodes},
                   edges=edges)


def _sim(nid, latency_us=10.0, flops=1024, read=512, write=512):
    return SimResult(
        op_node_id=nid, latency_us=latency_us,
        compute_us=latency_us * 0.8, memory_us=latency_us * 0.2,
        flops=flops, read_bytes=read, write_bytes=write,
        arithmetic_intensity=flops / (read + write),
        bound="compute", hw_utilization=0.5,
        backend="roofline", confidence=0.3,
    )


def _hw():
    return hw_registry.load("nvidia_h100_sxm")


# ── prefill metrics ───────────────────────────────────────────────────────────

def test_prefill_ttft_set_tpot_none():
    n = _node("a", latency_us=10_000.0)  # 10 ms
    g = _graph([n], [])
    tl = DAGScheduler().schedule(g)
    s = build_summary("M", "H", "prefill", 1, 128, g, {"a": _sim("a", 10_000.0)}, tl, _hw())
    assert s.ttft_ms == pytest.approx(10.0)
    assert s.tpot_ms is None


def test_prefill_tokens_per_sec():
    n = _node("a", latency_us=1_000.0)  # 1 ms
    g = _graph([n], [])
    tl = DAGScheduler().schedule(g)
    s = build_summary("M", "H", "prefill", 1, 128, g, {"a": _sim("a", 1_000.0)}, tl, _hw())
    # 128 tokens / 0.001 s = 128_000 tokens/s
    assert s.tokens_per_sec == pytest.approx(128_000.0)


# ── decode metrics ────────────────────────────────────────────────────────────

def test_decode_tpot_set_ttft_none():
    n = _node("a", latency_us=5_000.0)  # 5 ms
    g = _graph([n], [], phase="decode")
    tl = DAGScheduler().schedule(g)
    s = build_summary("M", "H", "decode", 1, 1, g, {"a": _sim("a", 5_000.0)}, tl, _hw())
    assert s.tpot_ms == pytest.approx(5.0)
    assert s.ttft_ms is None


def test_decode_tokens_per_sec_batch4():
    n = _node("a", latency_us=2_000.0)  # 2 ms
    g = _graph([n], [], phase="decode")
    tl = DAGScheduler().schedule(g)
    s = build_summary("M", "H", "decode", 4, 1, g, {"a": _sim("a", 2_000.0)}, tl, _hw())
    # 4 tokens / 0.002 s = 2000 tokens/s
    assert s.tokens_per_sec == pytest.approx(2_000.0)


# ── comm decomposition ────────────────────────────────────────────────────────

def test_no_comm_exposed_comm_is_zero():
    n = _node("a", latency_us=10_000.0)
    g = _graph([n], [])
    tl = DAGScheduler().schedule(g)
    s = build_summary("M", "H", "prefill", 1, 128, g, {"a": _sim("a", 10_000.0)}, tl, _hw())
    assert s.comm_ms == pytest.approx(0.0)
    assert s.exposed_comm_ms == pytest.approx(0.0)
    assert s.overlap_ratio == pytest.approx(1.0)  # no comm → ratio=1 by convention


def test_comm_fully_hidden_overlap_ratio_100pct():
    """Compute(8ms) and comm(5ms) start together after root(10ms)."""
    root  = _node("root",  latency_us=10_000.0, stream_id=0)
    left  = _node("left",  latency_us=8_000.0,  stream_id=0)
    right = _node("right", latency_us=5_000.0,  stream_id=1,
                  category="communication", op_type="comm.all_reduce")
    right.annotations["stream_type"] = "comm"
    g  = _graph([root, left, right], [_edge("root", "left"), _edge("root", "right")])
    tl = DAGScheduler().schedule(g)
    sr = {
        "root":  _sim("root",  10_000.0),
        "left":  _sim("left",  8_000.0),
        "right": _sim("right", 5_000.0),
    }
    s = build_summary("M", "H", "prefill", 1, 128, g, sr, tl, _hw())
    assert s.exposed_comm_ms == pytest.approx(0.0, abs=1e-3)
    assert s.overlap_ratio   == pytest.approx(1.0, abs=1e-3)


def test_comm_not_hidden_linear_chain():
    """compute → comm in sequence: no overlap."""
    a = _node("a", latency_us=10_000.0, stream_id=0)
    b = _node("b", latency_us=4_000.0,  stream_id=1,
              category="communication", op_type="comm.all_reduce")
    b.annotations["stream_type"] = "comm"
    g  = _graph([a, b], [_edge("a", "b")])
    tl = DAGScheduler().schedule(g)
    sr = {"a": _sim("a", 10_000.0), "b": _sim("b", 4_000.0)}
    s  = build_summary("M", "H", "prefill", 1, 128, g, sr, tl, _hw())
    assert s.exposed_comm_ms == pytest.approx(4.0, abs=1e-3)
    assert s.overlap_ratio   == pytest.approx(0.0, abs=1e-3)


# ── hw efficiency ─────────────────────────────────────────────────────────────

def test_mfu_is_between_0_and_1():
    n = _node("a", latency_us=10_000.0)
    g = _graph([n], [])
    tl = DAGScheduler().schedule(g)
    s = build_summary("M", "H", "prefill", 1, 128, g, {"a": _sim("a", 10_000.0)}, tl, _hw())
    assert 0.0 <= s.mfu <= 1.0
    assert 0.0 <= s.hbm_bandwidth_util <= 1.0


def test_mfu_nonzero_for_matmul():
    n = _node("a", latency_us=1_000.0)
    g = _graph([n], [])
    tl = DAGScheduler().schedule(g)
    big_flops = int(1e12)  # 1 TFLOP
    sr = {"a": _sim("a", 1_000.0, flops=big_flops)}
    s = build_summary("M", "H", "prefill", 1, 128, g, sr, tl, _hw())
    assert s.mfu > 0.0
    assert s.total_flops == big_flops


# ── hierarchical decomposition ────────────────────────────────────────────────

def test_by_component_groups_by_last_scope_segment():
    """Nodes from the same component type across layers should be grouped."""
    nodes = [
        _node("q0", scope="model.layers.0.self_attn", latency_us=8_000.0),
        _node("m0", scope="model.layers.0.mlp",       latency_us=6_000.0),
        _node("q1", scope="model.layers.1.self_attn", latency_us=8_000.0),
        _node("m1", scope="model.layers.1.mlp",       latency_us=6_000.0),
    ]
    g  = _graph(nodes, [])
    tl = DAGScheduler().schedule(g)
    sr = {n.id: _sim(n.id, n.annotations["latency_us"]) for n in nodes}
    s  = build_summary("M", "H", "prefill", 1, 128, g, sr, tl, _hw())
    assert "self_attn" in s.by_component
    assert "mlp"       in s.by_component
    # self_attn: 16ms out of 28ms total serial ≈ 57%
    assert s.by_component["self_attn"] == pytest.approx(
        (8_000.0 + 8_000.0) / (8_000.0 + 6_000.0 + 8_000.0 + 6_000.0) * 100.0,
        abs=0.1
    )


def test_by_layer_ordered_by_index():
    nodes = [
        _node("n0", scope="model.layers.0.mlp", latency_us=5_000.0),
        _node("n1", scope="model.layers.1.mlp", latency_us=7_000.0),
        _node("n2", scope="model.layers.2.mlp", latency_us=6_000.0),
    ]
    g  = _graph(nodes, [])
    tl = DAGScheduler().schedule(g)
    sr = {n.id: _sim(n.id, n.annotations["latency_us"]) for n in nodes}
    s  = build_summary("M", "H", "prefill", 1, 128, g, sr, tl, _hw())
    assert len(s.by_layer) == 3
    assert s.by_layer[0] == pytest.approx(5.0, abs=0.01)
    assert s.by_layer[1] == pytest.approx(7.0, abs=0.01)
    assert s.by_layer[2] == pytest.approx(6.0, abs=0.01)


def test_no_scope_nodes_produce_empty_breakdown():
    n = _node("a", scope="", latency_us=10_000.0)
    g = _graph([n], [])
    tl = DAGScheduler().schedule(g)
    s  = build_summary("M", "H", "prefill", 1, 128, g, {"a": _sim("a", 10_000.0)}, tl, _hw())
    assert s.by_component == {}
    assert s.by_layer == []


# ── top bottleneck ops ────────────────────────────────────────────────────────

def test_top_bottleneck_ops_sorted_by_latency():
    nodes = [_node(f"n{i}", latency_us=float(i + 1) * 1_000.0) for i in range(5)]
    g  = _graph(nodes, [])
    tl = DAGScheduler().schedule(g)
    sr = {n.id: _sim(n.id, n.annotations["latency_us"]) for n in nodes}
    s  = build_summary("M", "H", "prefill", 1, 128, g, sr, tl, _hw(), top_n=3)
    assert len(s.top_bottleneck_ops) == 3
    lats = [lat for _, lat in s.top_bottleneck_ops]
    assert lats == sorted(lats, reverse=True)


def test_top_bottleneck_ops_includes_op_type():
    n = _node("a", op_type="gated_mlp", scope="model.layers.0.mlp", latency_us=20_000.0)
    g = _graph([n], [])
    tl = DAGScheduler().schedule(g)
    s  = build_summary("M", "H", "prefill", 1, 128, g, {"a": _sim("a", 20_000.0)}, tl, _hw())
    desc = s.top_bottleneck_ops[0][0]
    assert "gated_mlp" in desc


# ── parallel_desc ─────────────────────────────────────────────────────────────

def test_parallel_desc_stored():
    n = _node("a", latency_us=1_000.0)
    g = _graph([n], [])
    tl = DAGScheduler().schedule(g)
    s  = build_summary("M", "H", "prefill", 1, 128, g, {"a": _sim("a")}, tl, _hw(),
                       parallel_desc="TP8-EP8")
    assert s.parallel_desc == "TP8-EP8"


# ── __str__ sanity ────────────────────────────────────────────────────────────

def test_str_contains_key_fields():
    n = _node("a", latency_us=5_000.0)
    g = _graph([n], [])
    tl = DAGScheduler().schedule(g)
    s  = build_summary("DeepSeek-V3", "nvidia_h100_sxm", "prefill", 1, 128,
                       g, {"a": _sim("a", 5_000.0)}, tl, _hw())
    text = str(s)
    assert "PREFILL" in text
    assert "TTFT" in text
    assert "MFU" in text
    assert "HBM BW util" in text
    assert "Throughput" in text


# ── memory budget integration ────────────────────────────────────────────────

def test_build_summary_with_memory_budget():
    """Verify that memory_budget can be passed and stored in E2ESummary."""
    from python.zrt.memory import MemoryBudget

    n = _node("a", latency_us=1_000.0)
    g = _graph([n], [])
    tl = DAGScheduler().schedule(g)

    budget = MemoryBudget(
        weights_mb=100.0,
        kv_cache_mb=50.0,
        activation_peak_mb=30.0,
        comm_buffer_mb=10.0,
        framework_overhead_mb=5.0,
        total_mb=195.0,
        capacity_mb=80000.0,
        is_feasible=True,
    )

    s = build_summary(
        "DeepSeek-V3", "nvidia_h100_sxm", "prefill", 1, 128,
        g, {"a": _sim("a", 1_000.0)}, tl, _hw(),
        memory_budget=budget,
    )

    assert s.memory_budget is not None
    assert s.memory_budget.total_mb == 195.0
    assert s.memory_budget.is_feasible is True


def test_build_summary_without_memory_budget():
    """Verify backward compatibility: memory_budget is optional and defaults to None."""
    n = _node("a", latency_us=1_000.0)
    g = _graph([n], [])
    tl = DAGScheduler().schedule(g)

    s = build_summary(
        "DeepSeek-V3", "nvidia_h100_sxm", "prefill", 1, 128,
        g, {"a": _sim("a", 1_000.0)}, tl, _hw(),
    )

    assert s.memory_budget is None
