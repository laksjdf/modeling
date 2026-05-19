"""Test data parallel pass: global comm insertion, ZeRO staging, overlap."""

import pytest
from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.types import TensorMeta, DType
from zrt.ir.edge import Edge
from zrt.transform.context import TransformContext, ParallelConfig, TrainingConfig
from zrt.transform.parallel.data_parallel import DataParallelPass
from zrt.transform.analysis import TrainingPipelinePass


def _make_backward_graph(num_layers=2, hidden=4096, seq_len=2048):
    """Create a backward-phase graph with gradient-producing nodes."""
    nodes = {}
    edges = []

    for i in range(num_layers):
        grad_out = TensorMeta(
            id=f"grad_out_{i}",
            shape=(1, seq_len, hidden),
            dtype=DType.BF16,
            mem_bytes=seq_len * hidden * 2,
        )
        grad_node = OpNode(
            id=f"grad_node_{i}",
            op_type=f"aten.mm_backward",
            inputs=[TensorMeta(id=f"grad_in_{i}", shape=(1, seq_len, hidden),
                               dtype=DType.BF16, mem_bytes=seq_len * hidden * 2)],
            outputs=[grad_out],
            scope=f"model.layers.{i}.self_attn.q_proj",
            layer=str(i),
            category="compute",
        )
        nodes[grad_node.id] = grad_node

        if i > 0:
            edges.append(Edge(
                src=f"grad_node_{i-1}", src_idx=0,
                dst=f"grad_node_{i}", dst_idx=0,
                tensor=grad_out,
            ))

    return OpGraph(
        name="test_dp_model",
        phase="train_backward",
        nodes=nodes,
        edges=edges,
        metadata={"seq_len": seq_len, "hidden": hidden, "num_layers": num_layers},
    )


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


class TestDPZero0:
    """ZeRO-0: single global all_reduce at end of backward pass."""

    def test_single_global_all_reduce(self):
        graph = _make_backward_graph(num_layers=3)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
                dp_overlap_in_bubble=False,
            ),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values()
                    if n.annotations.get("dp_comm")]
        assert len(dp_nodes) == 1  # single global comm node

        node = dp_nodes[0]
        assert node.op_type == "comm.all_reduce"
        assert node.attrs["group_size"] == 4
        assert node.attrs["collective"] == "all_reduce"
        assert node.attrs["bucket_bytes"] > 0
        assert node.id == "comm_dp_grad_reduce"

    def test_dp_comm_annotation_present(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=0),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values()
                    if n.annotations.get("dp_comm")]
        for node in dp_nodes:
            assert node.annotations["dp_comm"] is True
            assert node.annotations["inserted_by"] == "data_parallel_pass"


class TestDPZero1:
    """ZeRO-1: single global all_reduce at end of backward pass."""

    def test_all_reduce_for_zero1(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=1),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values()
                    if n.annotations.get("dp_comm")]
        assert len(dp_nodes) == 1
        for node in dp_nodes:
            assert node.op_type == "comm.all_reduce"
            assert node.attrs["collective"] == "all_reduce"


class TestDPZero2:
    """ZeRO-2: single global reduce_scatter at end of backward pass."""

    def test_reduce_scatter_for_zero2(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=2),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values()
                    if n.annotations.get("dp_comm")]
        assert len(dp_nodes) == 1
        for node in dp_nodes:
            assert node.op_type == "comm.reduce_scatter"
            assert node.attrs["collective"] == "reduce_scatter"

    def test_zero3_skipped_by_dp_pass(self):
        """ZeRO-3 should be skipped by DataParallelPass (handled by ZeroFSDPPass)."""
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=3),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values()
                    if n.annotations.get("dp_comm")]
        assert len(dp_nodes) == 0

    def test_reduce_scatter_has_lower_modeled_time_than_all_reduce(self):
        graph = _make_backward_graph(num_layers=1)
        ctx_z0 = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=0),
        )
        ctx_z2 = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=2),
        )

        ar_graph = DataParallelPass().run(graph, ctx_z0)
        rs_graph = DataParallelPass().run(graph, ctx_z2)

        ar_time = TrainingPipelinePass._compute_dp_ar_time(ar_graph, _make_hardware_spec(), ctx_z0)
        rs_time = TrainingPipelinePass._compute_dp_ar_time(rs_graph, _make_hardware_spec(), ctx_z2)

        assert rs_time == pytest.approx(ar_time / 2)


class TestDPOverlap:
    """Tests for DP overlap-in-bubble behavior."""

    def test_overlap_annotation_set(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
                dp_overlap_in_bubble=True,
            ),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values()
                    if n.annotations.get("dp_comm")]
        for node in dp_nodes:
            assert node.annotations.get("overlap_in_bubble") is True

    def test_no_overlap_annotation_when_disabled(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
                dp_overlap_in_bubble=False,
            ),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values()
                    if n.annotations.get("dp_comm")]
        for node in dp_nodes:
            assert "overlap_in_bubble" not in node.annotations

    def test_dp_skip_when_dp1(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=1),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=0),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values()
                    if n.annotations.get("dp_comm")]
        assert len(dp_nodes) == 0


class TestDPGlobalComm:
    """Tests for global communication node placement."""

    def test_comm_inserted_after_last_bwd_node(self):
        graph = _make_backward_graph(num_layers=3)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=0),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values() if n.annotations.get("dp_comm")]
        assert len(dp_nodes) == 1
        comm_node = dp_nodes[0]

        # Comm node should be connected from the last backward node
        out_edges_from_comm = [e for e in result.edges if e.src == comm_node.id]
        assert len(out_edges_from_comm) == 0  # Leaf node after insertion

    def test_total_gradient_bytes_accumulated(self):
        graph = _make_backward_graph(num_layers=3, hidden=4096, seq_len=2048)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=2),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values() if n.annotations.get("dp_comm")]
        assert len(dp_nodes) == 1
        bucket_bytes = dp_nodes[0].attrs["bucket_bytes"]
        assert bucket_bytes > 0
        # Should be sum of all backward node outputs
        expected = sum(
            o.mem_bytes for n in graph.nodes.values() for o in n.outputs
            if hasattr(o, 'mem_bytes')
        )
        assert bucket_bytes == expected
