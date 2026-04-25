"""Test stream overlap: CoC, MC2, Ring-CP exposed comm-time formulas."""

import pytest
from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.types import TensorMeta, DType
from zrt.ir.edge import Edge
from zrt.transform.context import TransformContext, ParallelConfig, TrainingConfig
from zrt.transform.analysis.passes import StreamAssignPass
from zrt.transform.analysis.training import compute_exposed_comm_time, TrainingPipelinePass


def _make_hardware_spec():
    from zrt.hardware.spec import HardwareSpec, ComputeSpec, MemorySpec, InterconnectSpec, LinkSpec
    return HardwareSpec(
        name="test_gpu",
        vendor="test",
        device_type="gpu",
        compute=ComputeSpec(bf16_tflops=1000),
        memory=MemorySpec(capacity_gb=80, hbm_bandwidth_gbps=3000),
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="nvlink", num_devices=8,
                                bandwidth_gbps=900, latency_us=1.0),
            inter_node=LinkSpec(type="ib", num_devices=1000,
                                bandwidth_gbps=400, latency_us=5.0),
        ),
    )


def _make_graph_with_comm(seq_len=2048, hidden=4096):
    """Create a graph with compute + comm nodes for overlap testing."""
    nodes = {}
    edges = []

    inp = TensorMeta(id="in_0", shape=(1, seq_len, hidden), dtype=DType.BF16,
                     mem_bytes=seq_len * hidden * 2)
    mid = TensorMeta(id="mid_0", shape=(1, seq_len, hidden), dtype=DType.BF16,
                     mem_bytes=seq_len * hidden * 2)
    out = TensorMeta(id="out_0", shape=(1, seq_len, hidden), dtype=DType.BF16,
                     mem_bytes=seq_len * hidden * 2)

    compute_node = OpNode(
        id="matmul_0",
        op_type="aten.mm",
        inputs=[inp],
        outputs=[mid],
        scope="model.layers.0.mlp.gate_proj",
        category="compute",
    )
    comm_node = OpNode(
        id="comm_ar_0",
        op_type="comm.all_reduce",
        inputs=[mid],
        outputs=[out],
        scope="model.layers.0.mlp",
        category="communication",
    )

    nodes["matmul_0"] = compute_node
    nodes["comm_ar_0"] = comm_node
    edges.append(Edge(src="matmul_0", src_idx=0, dst="comm_ar_0", dst_idx=0, tensor=mid))

    return OpGraph(
        name="test_overlap",
        phase="forward",
        nodes=nodes,
        edges=edges,
        metadata={"seq_len": seq_len, "hidden": hidden, "num_layers": 1},
    )


class TestComputeExposedCommTime:
    """Unit tests for compute_exposed_comm_time() helper."""

    def test_none_overlap_full_exposure(self):
        """No overlap: entire comm time is exposed."""
        exposed = compute_exposed_comm_time(100.0, "none")
        assert exposed == 100.0

    def test_mc2_zero_exposure(self):
        """MC2: fused AG+matmul, zero exposed comm."""
        exposed = compute_exposed_comm_time(200.0, "mc2")
        assert exposed == 0.0

    def test_coc_fully_hidden(self):
        """CoC: comm fits inside overlap window → exposed = 0."""
        # t_matmul = 100, k = 4 → overlap window = 100 * 3/4 = 75
        # t_comm = 50 < 75 → fully hidden
        exposed = compute_exposed_comm_time(50.0, "coc", target_latency_us=100.0, coc_tile_k=4)
        assert exposed == 0.0

    def test_coc_partially_exposed(self):
        """CoC: comm exceeds overlap window → partial exposure."""
        # overlap window = 100 * 3/4 = 75
        # t_comm = 100 → exposed = 100 - 75 = 25
        exposed = compute_exposed_comm_time(100.0, "coc", target_latency_us=100.0, coc_tile_k=4)
        assert exposed == pytest.approx(25.0)

    def test_coc_reduces_relative_to_no_overlap(self):
        """CoC always produces less exposed time than no overlap."""
        t_comm, t_matmul, k = 80.0, 100.0, 4
        exposed_coc = compute_exposed_comm_time(t_comm, "coc", t_matmul, k)
        exposed_none = compute_exposed_comm_time(t_comm, "none")
        assert exposed_coc <= exposed_none

    def test_ring_cp_fully_hidden(self):
        """Ring-CP: P2P fits inside FA tile → exposed = 0."""
        exposed = compute_exposed_comm_time(30.0, "ring_cp", target_latency_us=50.0)
        assert exposed == 0.0

    def test_ring_cp_partially_exposed(self):
        """Ring-CP: P2P exceeds FA tile → partial exposure."""
        exposed = compute_exposed_comm_time(80.0, "ring_cp", target_latency_us=50.0)
        assert exposed == pytest.approx(30.0)

    def test_ring_cp_reduces_exposure(self):
        """Ring-CP always produces less or equal exposed time than no overlap."""
        t_p2p, t_fa = 60.0, 40.0
        exposed_rcp = compute_exposed_comm_time(t_p2p, "ring_cp", t_fa)
        exposed_none = compute_exposed_comm_time(t_p2p, "none")
        assert exposed_rcp < exposed_none


class TestStreamAssignOverlapDetection:
    """Tests for StreamAssignPass overlap type detection."""

    def test_ring_cp_detected_from_overlap_target(self):
        """Comm node with fa_tile: overlap_target gets overlap_type=ring_cp."""
        graph = _make_graph_with_comm()
        comm = graph.nodes["comm_ar_0"]
        comm.annotations["overlap_target"] = "fa_tile:matmul_0"

        ctx = TransformContext(hw_spec=_make_hardware_spec())
        result = StreamAssignPass().run(graph, ctx)

        assert result.nodes["comm_ar_0"].annotations["overlap_type"] == "ring_cp"

    def test_mc2_detected_from_fused_attr(self):
        """Comm node with fused_ag_matmul attr gets overlap_type=mc2."""
        graph = _make_graph_with_comm()
        comm = graph.nodes["comm_ar_0"]
        comm.attrs["fused_ag_matmul"] = True

        ctx = TransformContext(hw_spec=_make_hardware_spec())
        result = StreamAssignPass().run(graph, ctx)

        assert result.nodes["comm_ar_0"].annotations["overlap_type"] == "mc2"

    def test_coc_detected_from_tile_k(self):
        """Comm node with coc_tile_k attr gets overlap_type=coc."""
        graph = _make_graph_with_comm()
        comm = graph.nodes["comm_ar_0"]
        comm.attrs["coc_tile_k"] = 4

        ctx = TransformContext(hw_spec=_make_hardware_spec())
        result = StreamAssignPass().run(graph, ctx)

        assert result.nodes["comm_ar_0"].annotations["overlap_type"] == "coc"

    def test_no_overlap_default(self):
        """Comm node without overlap markers gets overlap_type=none."""
        graph = _make_graph_with_comm()
        ctx = TransformContext(hw_spec=_make_hardware_spec())
        result = StreamAssignPass().run(graph, ctx)

        assert result.nodes["comm_ar_0"].annotations["overlap_type"] == "none"

    def test_compute_nodes_no_overlap_type(self):
        """Compute nodes don't get overlap_type annotation."""
        graph = _make_graph_with_comm()
        ctx = TransformContext(hw_spec=_make_hardware_spec())
        result = StreamAssignPass().run(graph, ctx)

        assert "overlap_type" not in result.nodes["matmul_0"].annotations


class TestOverlapIntegration:
    """Integration: overlap reduces step_time in TrainingPipelinePass."""

    def test_overlap_reduces_step_time(self):
        """Step time with overlap should be less than without."""
        seq_len, hidden = 2048, 4096
        hw = _make_hardware_spec()

        def _make_ctx():
            return TransformContext(
                hw_spec=hw,
                parallel=ParallelConfig(tp=1, pp=1),
                training=TrainingConfig(micro_batch=1, global_batch=8),
            )

        # Graph without overlap
        g_no = _make_graph_with_comm(seq_len, hidden)
        for n in g_no.nodes.values():
            n.annotations["latency_us"] = 100.0
        ctx = _make_ctx()
        r_no = TrainingPipelinePass().run(g_no, ctx)
        step_no = r_no.metadata["pipeline_metrics"].step_time_ms

        # Graph with ring-CP overlap on the comm node
        g_ov = _make_graph_with_comm(seq_len, hidden)
        for n in g_ov.nodes.values():
            n.annotations["latency_us"] = 100.0
        comm = g_ov.nodes["comm_ar_0"]
        comm.annotations["overlap_type"] = "ring_cp"
        comm.annotations["overlap_target"] = "fa_tile:matmul_0"
        r_ov = TrainingPipelinePass().run(g_ov, ctx)
        step_ov = r_ov.metadata["pipeline_metrics"].step_time_ms

        assert step_ov < step_no, (
            f"Overlap should reduce step time: {step_ov} >= {step_no}"
        )

    def test_coc_overlap_uses_predecessor_latency_without_explicit_target(self):
        """CoC overlap should still reduce step time using predecessor compute latency."""
        seq_len, hidden = 2048, 4096
        hw = _make_hardware_spec()
        ctx = TransformContext(
            hw_spec=hw,
            parallel=ParallelConfig(tp=1, pp=1),
            training=TrainingConfig(micro_batch=1, global_batch=8),
        )

        g_no = _make_graph_with_comm(seq_len, hidden)
        for n in g_no.nodes.values():
            n.annotations["latency_us"] = 100.0
        r_no = TrainingPipelinePass().run(g_no, ctx)
        step_no = r_no.metadata["pipeline_metrics"].step_time_ms

        g_coc = _make_graph_with_comm(seq_len, hidden)
        for n in g_coc.nodes.values():
            n.annotations["latency_us"] = 100.0
        comm = g_coc.nodes["comm_ar_0"]
        comm.annotations["overlap_type"] = "coc"
        comm.attrs["coc_tile_k"] = 4
        # Intentionally no overlap_target: pipeline should use predecessor latency.

        r_coc = TrainingPipelinePass().run(g_coc, ctx)
        step_coc = r_coc.metadata["pipeline_metrics"].step_time_ms

        assert step_coc < step_no, (
            f"CoC fallback should reduce step time: {step_coc} >= {step_no}"
        )
