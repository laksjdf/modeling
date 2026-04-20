"""Tests for python.zrt.executor.DAGScheduler."""
import pytest
from python.zrt.ir.node import OpNode
from python.zrt.ir.edge import Edge
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.types import TensorMeta, DType
from python.zrt.executor import DAGScheduler, Timeline, ScheduledOp
from python.zrt.transform import TransformContext, ParallelConfig, StreamConfig, build_default_pipeline


# ── helpers ───────────────────────────────────────────────────────────────────

def _t(tid, shape=(1,), dtype=DType.BF16):
    return TensorMeta.from_shape_dtype(tid, shape, dtype)


def _node(nid, op_type="aten.mm.default", stream_id=0,
          stream_type="compute", latency_us=10.0, category=None):
    cat = category or ("communication" if stream_type == "comm" else "compute")
    n = OpNode(id=nid, op_type=op_type,
               inputs=[_t(f"{nid}_in")], outputs=[_t(f"{nid}_out")],
               category=cat)
    n.annotations["stream_id"]   = stream_id
    n.annotations["stream_type"] = stream_type
    n.annotations["latency_us"]  = latency_us
    return n


def _edge(src, dst):
    return Edge(src=src, src_idx=0, dst=dst, dst_idx=0, tensor=_t("e"))


def _graph(nodes, edges, name="test"):
    return OpGraph(name=name, phase="prefill",
                   nodes={n.id: n for n in nodes},
                   edges=edges)


# ── basic scheduling ──────────────────────────────────────────────────────────

def test_single_node():
    g = _graph([_node("a", latency_us=5.0)], [])
    tl = DAGScheduler().schedule(g)
    assert len(tl.scheduled_ops) == 1
    op = tl.scheduled_ops[0]
    assert op.start_us == 0.0
    assert op.end_us == 5.0
    assert tl.total_latency_us == 5.0


def test_linear_chain_total_latency():
    """A → B → C on the same stream: total = sum of latencies."""
    a = _node("a", latency_us=10.0)
    b = _node("b", latency_us=8.0)
    c = _node("c", latency_us=6.0)
    g = _graph([a, b, c], [_edge("a", "b"), _edge("b", "c")])
    tl = DAGScheduler().schedule(g)
    assert tl.total_latency_us == pytest.approx(24.0)


def test_dependency_ordering():
    """B must start after A ends, regardless of stream availability."""
    a = _node("a", latency_us=10.0, stream_id=0)
    b = _node("b", latency_us=5.0,  stream_id=1)  # different stream, still depends on A
    g = _graph([a, b], [_edge("a", "b")])
    tl = DAGScheduler().schedule(g)

    sched = {op.node_id: op for op in tl.scheduled_ops}
    assert sched["b"].start_us == pytest.approx(10.0)
    assert sched["b"].end_us   == pytest.approx(15.0)


def test_stream_serialization():
    """Two independent nodes on the same stream cannot overlap."""
    a = _node("a", latency_us=10.0, stream_id=0)
    b = _node("b", latency_us=5.0,  stream_id=0)  # same stream, no edge
    g = _graph([a, b], [])
    tl = DAGScheduler().schedule(g)
    # whichever is scheduled first occupies stream 0; the other must wait
    sched = {op.node_id: op for op in tl.scheduled_ops}
    # Nodes are in insertion order: a first
    assert sched["b"].start_us >= sched["a"].end_us


def test_parallel_independent_different_streams():
    """Two independent nodes on different streams run concurrently after shared pred."""
    root  = _node("root", latency_us=10.0, stream_id=0)
    left  = _node("left", latency_us=8.0,  stream_id=0)
    right = _node("right", latency_us=5.0,  stream_id=1)
    g = _graph([root, left, right],
               [_edge("root", "left"), _edge("root", "right")])
    tl = DAGScheduler().schedule(g)

    sched = {op.node_id: op for op in tl.scheduled_ops}
    # Both start at 10 (after root)
    assert sched["left"].start_us  == pytest.approx(10.0)
    assert sched["right"].start_us == pytest.approx(10.0)
    # Wall-clock = 10 + max(8, 5) = 18
    assert tl.total_latency_us == pytest.approx(18.0)


# ── timeline properties ───────────────────────────────────────────────────────

def test_timeline_total_latency_is_max_end():
    a = _node("a", latency_us=10.0, stream_id=0)
    b = _node("b", latency_us=7.0,  stream_id=1)
    g = _graph([a, b], [])
    tl = DAGScheduler().schedule(g)
    max_end = max(op.end_us for op in tl.scheduled_ops)
    assert tl.total_latency_us == pytest.approx(max_end)


def test_timeline_compute_comm_time():
    compute = _node("c0", stream_id=0, stream_type="compute", latency_us=10.0)
    comm    = _node("c1", stream_id=1, stream_type="comm",    latency_us=4.0,
                    op_type="comm.all_reduce", category="communication")
    g = _graph([compute, comm], [])
    tl = DAGScheduler().schedule(g)
    assert tl.compute_time_us == pytest.approx(10.0)
    assert tl.comm_time_us    == pytest.approx(4.0)


def test_overlap_comm_hidden_behind_compute():
    """
    Graph:  root → (left: compute, stream 0, lat=8)
                 → (right: comm,    stream 1, lat=5)

    Both start at root_end=10, run in parallel.
    Wall-clock = 10 + max(8, 5) = 18.
    overlap = compute(8) + comm(5) - total(8) = 5   [comm fully hidden]
    """
    root  = _node("root",  stream_id=0, stream_type="compute", latency_us=10.0)
    left  = _node("left",  stream_id=0, stream_type="compute", latency_us=8.0)
    right = _node("right", stream_id=1, stream_type="comm",    latency_us=5.0,
                  op_type="comm.all_reduce", category="communication")
    g = _graph([root, left, right],
               [_edge("root", "left"), _edge("root", "right")])
    tl = DAGScheduler().schedule(g)

    assert tl.total_latency_us == pytest.approx(18.0)   # 10 + 8
    assert tl.overlap_us       == pytest.approx(5.0)    # comm fully masked
    assert tl.overlap_us >= 0.0


def test_no_overlap_linear_chain():
    """Linear chain compute → comm: no overlap possible."""
    a = _node("a", stream_id=0, stream_type="compute", latency_us=10.0)
    b = _node("b", stream_id=1, stream_type="comm",    latency_us=4.0,
              op_type="comm.all_reduce", category="communication")
    g = _graph([a, b], [_edge("a", "b")])
    tl = DAGScheduler().schedule(g)
    assert tl.overlap_us == pytest.approx(0.0)


def test_timeline_ops_on_stream():
    a = _node("a", stream_id=0, latency_us=5.0)
    b = _node("b", stream_id=0, latency_us=3.0)
    c = _node("c", stream_id=1, latency_us=7.0)
    g = _graph([a, b, c], [_edge("a", "b")])
    tl = DAGScheduler().schedule(g)
    s0 = tl.ops_on_stream(0)
    s1 = tl.ops_on_stream(1)
    assert len(s0) == 2
    assert len(s1) == 1
    assert s0[0].node_id == "a"
    assert s0[1].node_id == "b"


# ── latency fallback ──────────────────────────────────────────────────────────

def test_latency_fallback_without_annotation():
    """Nodes without latency_us annotation get 1 µs default."""
    n = OpNode(id="x", op_type="aten.mm.default",
               inputs=[_t("xi")], outputs=[_t("xo")])
    n.annotations["stream_id"]   = 0
    n.annotations["stream_type"] = "compute"
    # no latency_us annotation
    g = _graph([n], [])
    tl = DAGScheduler().schedule(g)
    assert tl.scheduled_ops[0].latency_us == pytest.approx(1.0)


def test_latency_from_hw_spec():
    """When hw_spec is provided, Roofline estimates latency for un-annotated nodes."""
    import python.zrt.hardware.registry as hw_registry
    hw = hw_registry.load("nvidia_h100_sxm")
    from python.zrt.ir.types import TensorMeta, DType
    n = OpNode(
        id="mm",
        op_type="aten.mm.default",
        inputs=[_t("a", (128, 4096)), _t("b", (4096, 4096))],
        outputs=[_t("out", (128, 4096))],
    )
    n.annotations["stream_id"]   = 0
    n.annotations["stream_type"] = "compute"
    g = _graph([n], [])
    tl = DAGScheduler(hw_spec=hw).schedule(g)
    assert tl.scheduled_ops[0].latency_us > 0.0


# ── integration with transform pipeline ───────────────────────────────────────

def test_schedule_after_full_pipeline():
    """Full pipeline (TP=4) + DAGScheduler produces a valid Timeline."""
    from python.zrt.ir.edge import Edge
    from python.zrt.transform import build_default_pipeline, TransformContext, ParallelConfig, StreamConfig
    import python.zrt.hardware.registry as hw_registry

    # build a simple 2-node graph
    from python.zrt.ir.types import TensorMeta, DType
    def _ln(nid, scope, in_sh, out_sh):
        return OpNode(id=nid, op_type="aten.mm.default",
                      inputs=[TensorMeta.from_shape_dtype(f"{nid}_in", in_sh, DType.BF16)],
                      outputs=[TensorMeta.from_shape_dtype(f"{nid}_out", out_sh, DType.BF16)],
                      scope=scope, category="compute")

    q = _ln("q", "model.layers.0.self_attn.q_proj", (128, 4096), (128, 4096))
    o = _ln("o", "model.layers.0.self_attn.o_proj", (128, 4096), (128, 4096))
    edge = Edge(src="q", src_idx=0, dst="o", dst_idx=0,
                tensor=TensorMeta.from_shape_dtype("e0", (128, 4096), DType.BF16))
    g = OpGraph(name="test", phase="prefill",
                nodes={"q": q, "o": o}, edges=[edge])

    hw  = hw_registry.load("nvidia_h100_sxm")
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=4),
        stream_config=StreamConfig(num_compute_streams=1, num_comm_streams=1),
    )
    transformed = build_default_pipeline().run(g, ctx)
    tl = DAGScheduler(hw_spec=hw).schedule(transformed)

    assert tl.total_latency_us > 0.0
    assert len(tl.scheduled_ops) == transformed.num_nodes()
    for op in tl.scheduled_ops:
        assert op.end_us >= op.start_us
        assert op.latency_us >= 0.0


def test_schedule_single_device_no_comm():
    """TP=1 graph has no comm nodes; all ops on compute stream."""
    from python.zrt.ir.edge import Edge
    from python.zrt.transform import build_default_pipeline, TransformContext, ParallelConfig, StreamConfig
    import python.zrt.hardware.registry as hw_registry
    from python.zrt.ir.types import TensorMeta, DType

    def _ln(nid, scope):
        return OpNode(id=nid, op_type="aten.mm.default",
                      inputs=[TensorMeta.from_shape_dtype(f"{nid}_in", (128, 4096), DType.BF16)],
                      outputs=[TensorMeta.from_shape_dtype(f"{nid}_out", (128, 4096), DType.BF16)],
                      scope=scope, category="compute")

    q = _ln("q", "model.layers.0.self_attn.q_proj")
    o = _ln("o", "model.layers.0.self_attn.o_proj")
    g = OpGraph(name="test", phase="prefill",
                nodes={"q": q, "o": o},
                edges=[Edge("q", 0, "o", 0,
                            TensorMeta.from_shape_dtype("e0", (128, 4096), DType.BF16))])

    hw  = hw_registry.load("nvidia_h100_sxm")
    ctx = TransformContext(hw_spec=hw, parallel=ParallelConfig(tp=1),
                           stream_config=StreamConfig())
    tl = DAGScheduler(hw_spec=hw).schedule(build_default_pipeline().run(g, ctx))

    assert tl.comm_time_us == pytest.approx(0.0)
    assert tl.overlap_us   == pytest.approx(0.0)
    assert tl.total_latency_us > 0.0


# ── OverlapAnalyzer tests ─────────────────────────────────────────────────────

def test_overlap_analyzer_no_comm():
    """No communication nodes => no overlap."""
    from python.zrt.executor import OverlapAnalyzer
    a = _node("a", stream_id=0, stream_type="compute", latency_us=10.0)
    b = _node("b", stream_id=0, stream_type="compute", latency_us=5.0)
    g = _graph([a, b], [_edge("a", "b")])
    tl = DAGScheduler().schedule(g)

    report = OverlapAnalyzer().analyze(tl)
    assert report.compute_us == pytest.approx(15.0)
    assert report.comm_us == pytest.approx(0.0)
    assert report.overlap_us == pytest.approx(0.0)
    assert report.exposed_comm_us == pytest.approx(0.0)
    assert report.critical_path_us == pytest.approx(15.0)


def test_overlap_analyzer_fully_hidden_comm():
    """Communication on separate stream, fully masked by compute."""
    from python.zrt.executor import OverlapAnalyzer
    root = _node("root", stream_id=0, stream_type="compute", latency_us=10.0)
    compute = _node("comp", stream_id=0, stream_type="compute", latency_us=8.0)
    comm = _node("c", stream_id=1, stream_type="comm", latency_us=5.0,
                 op_type="comm.all_reduce", category="communication")
    g = _graph([root, compute, comm],
               [_edge("root", "compute"), _edge("root", "comm")])
    tl = DAGScheduler().schedule(g)

    report = OverlapAnalyzer().analyze(tl)
    assert report.compute_us == pytest.approx(18.0)  # 10 + 8
    assert report.comm_us == pytest.approx(5.0)
    assert report.overlap_us == pytest.approx(5.0)  # fully overlapped
    assert report.exposed_comm_us == pytest.approx(0.0)


def test_overlap_analyzer_partial_overlap():
    """Communication partially overlapped."""
    from python.zrt.executor import OverlapAnalyzer
    compute = _node("comp", stream_id=0, stream_type="compute", latency_us=3.0)
    comm = _node("c", stream_id=1, stream_type="comm", latency_us=5.0,
                 op_type="comm.all_reduce", category="communication")
    g = _graph([compute, comm], [])
    tl = DAGScheduler().schedule(g)

    report = OverlapAnalyzer().analyze(tl)
    assert report.compute_us == pytest.approx(3.0)
    assert report.comm_us == pytest.approx(5.0)
    # Both start at time 0, compute ends at 3, comm ends at 5
    # Overlap interval: [0, 3], duration 3
    assert report.overlap_us == pytest.approx(3.0)
    assert report.exposed_comm_us == pytest.approx(2.0)
    assert abs(report.overlap_ratio - 0.6) < 0.01


# ── CommLatencyPass tests ──────────────────────────────────────────────────────

def test_comm_latency_pass_intra_node():
    """Single-node multi-card: use intra_node bandwidth."""
    from python.zrt.transform.analysis.comm_latency import CommLatencyPass
    from python.zrt.ir.types import TensorMeta, DType

    # Create a comm node
    comm_node = OpNode(
        id="allreduce_0",
        op_type="comm.all_reduce",
        inputs=[TensorMeta.from_shape_dtype("in0", (1024, 1024), DType.BF16)],
        outputs=[TensorMeta.from_shape_dtype("out0", (1024, 1024), DType.BF16)],
        attrs={"group_size": 4, "collective": "all_reduce"},
        category="communication",
    )
    g = OpGraph(name="test", phase="prefill",
                nodes={"allreduce_0": comm_node}, edges=[])

    import python.zrt.hardware.registry as hw_registry
    hw = hw_registry.load("nvidia_h100_sxm")  # 8 GPUs per node
    ctx = TransformContext(hw_spec=hw, parallel=ParallelConfig(tp=4))

    # Run the pass
    result = CommLatencyPass().run(g, ctx)
    result_node = result.nodes["allreduce_0"]

    # Should use intra_node bandwidth (NVLink)
    assert result_node.annotations.get("cross_node") is False
    latency = result_node.annotations.get("latency_us")
    assert latency is not None and latency > 0.0


def test_comm_latency_pass_cross_node():
    """Multi-node: use inter_node bandwidth when group > intra_node devices."""
    from python.zrt.transform.analysis.comm_latency import CommLatencyPass
    from python.zrt.ir.types import TensorMeta, DType

    comm_node = OpNode(
        id="allreduce_0",
        op_type="comm.all_reduce",
        inputs=[TensorMeta.from_shape_dtype("in0", (1024, 1024), DType.BF16)],
        outputs=[TensorMeta.from_shape_dtype("out0", (1024, 1024), DType.BF16)],
        attrs={"group_size": 16, "collective": "all_reduce"},  # Exceed 8 GPUs per node
        category="communication",
    )
    g = OpGraph(name="test", phase="prefill",
                nodes={"allreduce_0": comm_node}, edges=[])

    import python.zrt.hardware.registry as hw_registry
    hw = hw_registry.load("nvidia_h100_sxm")
    ctx = TransformContext(hw_spec=hw, parallel=ParallelConfig(tp=16))

    result = CommLatencyPass().run(g, ctx)
    result_node = result.nodes["allreduce_0"]

    # Should use inter_node bandwidth (RoCE/IB)
    assert result_node.annotations.get("cross_node") is True
    latency = result_node.annotations.get("latency_us")
    assert latency is not None and latency > 0.0


def test_comm_latency_pass_zero_group_size():
    """group_size=1: no communication needed."""
    from python.zrt.transform.analysis.comm_latency import CommLatencyPass
    from python.zrt.ir.types import TensorMeta, DType

    comm_node = OpNode(
        id="allreduce_0",
        op_type="comm.all_reduce",
        inputs=[TensorMeta.from_shape_dtype("in0", (1024, 1024), DType.BF16)],
        outputs=[TensorMeta.from_shape_dtype("out0", (1024, 1024), DType.BF16)],
        attrs={"group_size": 1, "collective": "all_reduce"},
        category="communication",
    )
    g = OpGraph(name="test", phase="prefill",
                nodes={"allreduce_0": comm_node}, edges=[])

    import python.zrt.hardware.registry as hw_registry
    hw = hw_registry.load("nvidia_h100_sxm")
    ctx = TransformContext(hw_spec=hw)

    result = CommLatencyPass().run(g, ctx)
    result_node = result.nodes["allreduce_0"]

    # Should be zero latency
    assert result_node.annotations.get("latency_us") == pytest.approx(0.0)


def test_comm_latency_pass_alltoall():
    """All-to-all uses different latency formula."""
    from python.zrt.transform.analysis.comm_latency import CommLatencyPass
    from python.zrt.ir.types import TensorMeta, DType

    comm_node = OpNode(
        id="a2a_0",
        op_type="comm.all_to_all",
        inputs=[TensorMeta.from_shape_dtype("in0", (512, 512), DType.BF16)],
        outputs=[TensorMeta.from_shape_dtype("out0", (512, 512), DType.BF16)],
        attrs={"group_size": 8, "collective": "all_to_all"},
        category="communication",
    )
    g = OpGraph(name="test", phase="prefill",
                nodes={"a2a_0": comm_node}, edges=[])

    import python.zrt.hardware.registry as hw_registry
    hw = hw_registry.load("nvidia_h100_sxm")
    ctx = TransformContext(hw_spec=hw, parallel=ParallelConfig(ep=8))

    result = CommLatencyPass().run(g, ctx)
    result_node = result.nodes["a2a_0"]

    latency = result_node.annotations.get("latency_us")
    assert latency is not None and latency > 0.0
    assert result_node.annotations.get("comm_algorithm") == "all_to_all"
