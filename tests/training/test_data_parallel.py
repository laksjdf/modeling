"""Test data parallel pass: per-group comm insertion, ZeRO staging, overlap."""

import pytest
from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.types import TensorMeta, DType
from zrt.ir.edge import Edge
from zrt.transform.context import (
    TransformContext, ParallelConfig, TrainingConfig, OffloadConfig,
)
from zrt.transform.parallel.data_parallel import DataParallelPass
from zrt.transform.analysis import TrainingPipelinePass
from zrt.transform.training.offload import OffloadPass


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
    """ZeRO-0: all_reduce comm nodes."""

    def test_all_reduce_created_single_group(self):
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
        assert len(dp_nodes) == 1  # single group (aligned with estimate path)

        for node in dp_nodes:
            assert node.op_type == "comm.all_reduce"
            assert node.attrs["group_size"] == 4
            assert node.attrs["collective"] == "all_reduce"
            assert node.attrs["bucket_bytes"] > 0

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


class TestDPZero2:
    """ZeRO-2/3: reduce_scatter comm nodes."""

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
        for node in dp_nodes:
            assert node.op_type == "comm.reduce_scatter"
            assert node.attrs["collective"] == "reduce_scatter"

    def test_reduce_scatter_for_zero3(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=3),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values()
                    if n.annotations.get("dp_comm")]
        for node in dp_nodes:
            assert node.op_type == "comm.reduce_scatter"

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

    def test_dp_ar_time_treats_interconnect_bandwidth_as_gb_per_second(self):
        dp = 4
        bucket_bytes = 900_000
        comm_node = OpNode(
            id="comm_grad_reduce",
            op_type="comm.all_reduce",
            inputs=[], outputs=[],
            attrs={
                "group_size": dp,
                "collective": "all_reduce",
                "bucket_bytes": bucket_bytes,
                "role": "dp_grad_reduce",
            },
            scope="data_parallel.grad_reduce",
            category="communication",
        )
        comm_node.annotations["dp_comm"] = True

        graph = OpGraph(
            name="test_dp_bw_units",
            phase="train_backward",
            nodes={"comm_grad_reduce": comm_node},
            metadata={"seq_len": 4096, "hidden": 4096},
        )
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=dp),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=0),
        )

        ar_time = TrainingPipelinePass._compute_dp_ar_time(
            graph, _make_hardware_spec(), ctx,
        )

        expected = 2.0 * (dp - 1) / dp * bucket_bytes / (900e9 / 1e6)
        assert ar_time == pytest.approx(expected)


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


class TestDPGroupIdx:
    """Tests for group index assignment (single group: always 0)."""

    def test_group_idx_is_zero(self):
        graph = _make_backward_graph(num_layers=3)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=0),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = sorted(
            [n for n in result.nodes.values() if n.annotations.get("dp_comm")],
            key=lambda n: n.attrs["dp_grad_group_idx"],
        )
        indices = [n.attrs["dp_grad_group_idx"] for n in dp_nodes]
        assert indices == [0]


class TestDPDivScale:
    """Tests for aten.div.Scalar gradient averaging nodes.

    DataParallelPass inserts a single div/scale node after the DP comm node
    so that all_reduce / reduce_scatter SUM is averaged by dp.
    """

    def test_div_scale_node_created(self):
        graph = _make_backward_graph(num_layers=3)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=0),
        )

        result = DataParallelPass().run(graph, ctx)

        scale_nodes = [
            n for n in result.nodes.values()
            if n.annotations.get("inserted_by") == "data_parallel_pass"
            and n.op_type == "aten.div.Scalar"
        ]
        assert len(scale_nodes) == 1  # single group

    def test_div_scale_divisor_equals_dp(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=8),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=0),
        )

        result = DataParallelPass().run(graph, ctx)

        scale_nodes = [
            n for n in result.nodes.values()
            if n.op_type == "aten.div.Scalar"
        ]
        for node in scale_nodes:
            assert node.attrs["divisor"] == 8
            assert node.attrs["role"] == "dp_grad_average"

    def test_div_scale_inserted_after_comm_node(self):
        """Verify div/scale is a successor of the corresponding comm node."""
        graph = _make_backward_graph(num_layers=1)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=0),
        )

        result = DataParallelPass().run(graph, ctx)

        comm_node = next(
            n for n in result.nodes.values()
            if n.annotations.get("dp_comm")
        )
        scale_node = next(
            n for n in result.nodes.values()
            if n.op_type == "aten.div.Scalar"
        )

        comm_succs = result.successors(comm_node.id)
        assert scale_node.id in comm_succs, (
            f"div/scale {scale_node.id} not a successor of "
            f"comm node {comm_node.id}"
        )

    def test_div_scale_has_correct_annotations(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=0),
        )

        result = DataParallelPass().run(graph, ctx)

        scale_nodes = [
            n for n in result.nodes.values()
            if n.op_type == "aten.div.Scalar"
        ]
        for node in scale_nodes:
            assert node.annotations["inserted_by"] == "data_parallel_pass"
            assert node.annotations["phase"] == "bwd"
            assert node.category == "compute"

    def test_div_scale_not_inserted_when_dp_1(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=1),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=0),
        )

        result = DataParallelPass().run(graph, ctx)

        scale_nodes = [
            n for n in result.nodes.values()
            if n.op_type == "aten.div.Scalar"
        ]
        assert len(scale_nodes) == 0

    def test_div_scale_divisor_matches_reduce_scatter(self):
        """div/scale divisor equals dp regardless of collective type."""
        graph = _make_backward_graph(num_layers=1)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=2),
        )

        result = DataParallelPass().run(graph, ctx)

        comm_node = next(
            n for n in result.nodes.values()
            if n.annotations.get("dp_comm")
        )
        assert comm_node.op_type == "comm.reduce_scatter"

        scale_node = next(
            n for n in result.nodes.values()
            if n.op_type == "aten.div.Scalar"
        )
        assert scale_node.attrs["divisor"] == 4


class TestDPOffloadChain:
    """Tests for OffloadPass _find_dp_chain_end with div/scale nodes.

    When DataParallelPass inserts a div/scale node after the DP comm node,
    OffloadPass must walk past it so that D2H is inserted after the averaged
    gradients, not before.
    """

    def test_find_dp_chain_end_skips_div_scale(self):
        """_find_dp_chain_end from comm node should return scale_node."""
        graph = _make_backward_graph(num_layers=1)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
                offload=OffloadConfig(pct=1.0, grads=True),
            ),
        )

        dp_graph = DataParallelPass().run(graph, ctx)

        comm_node = next(
            n for n in dp_graph.nodes.values()
            if n.annotations.get("dp_comm")
        )

        offload_pass = OffloadPass()
        chain_end = offload_pass._find_dp_chain_end(dp_graph, comm_node)

        assert chain_end is not comm_node, (
            "_find_dp_chain_end should skip past div/scale to the chain end"
        )
        assert chain_end.op_type == "aten.div.Scalar", (
            f"expected div/scale at chain end, got {chain_end.op_type}"
        )
        assert chain_end.annotations["inserted_by"] == "data_parallel_pass"

    def test_d2h_inserted_after_div_scale(self):
        """After OffloadPass runs, D2H should be a successor of div/scale,
        not of the comm node directly."""
        graph = _make_backward_graph(num_layers=1)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
                offload=OffloadConfig(pct=1.0, grads=True),
            ),
        )

        dp_graph = DataParallelPass().run(graph, ctx)
        result = OffloadPass().run(dp_graph, ctx)

        comm_node = next(
            n for n in result.nodes.values()
            if n.annotations.get("dp_comm")
        )
        scale_node = next(
            n for n in result.nodes.values()
            if n.op_type == "aten.div.Scalar"
        )
        d2h_node = next(
            n for n in result.nodes.values()
            if n.annotations.get("inserted_by") == "offload_pass"
        )

        scale_succs = result.successors(scale_node.id)
        assert d2h_node.id in scale_succs, (
            f"D2H {d2h_node.id} should be a successor of "
            f"div/scale {scale_node.id}, got succs={scale_succs}"
        )

        comm_succs = result.successors(comm_node.id)
        assert scale_node.id in comm_succs, (
            "comm node should still point to div/scale"
        )

    def test_no_offload_when_grads_disabled(self):
        """D2H should not be inserted when offload.grads is False."""
        graph = _make_backward_graph(num_layers=1)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
                offload=OffloadConfig(pct=1.0, grads=False),
            ),
        )

        dp_graph = DataParallelPass().run(graph, ctx)
        result = OffloadPass().run(dp_graph, ctx)

        d2h_nodes = [
            n for n in result.nodes.values()
            if n.annotations.get("inserted_by") == "offload_pass"
        ]
        assert len(d2h_nodes) == 0

    def test_find_dp_chain_end_stops_at_non_dp_node(self):
        """When div/scale is NOT present (legacy graph), chain end is the
        comm node itself."""
        graph = _make_backward_graph(num_layers=1)

        dp = 4
        comm_node = OpNode(
            id="comm_grad_reduce",
            op_type="comm.all_reduce",
            inputs=[], outputs=[],
            attrs={"group_size": dp, "collective": "all_reduce",
                   "bucket_bytes": 1000, "role": "dp_grad_reduce"},
            scope="data_parallel.grad_reduce",
            category="communication",
        )
        comm_node.annotations["dp_comm"] = True
        comm_node.annotations["inserted_by"] = "data_parallel_pass"
        comm_node.annotations["phase"] = "bwd"

        g = OpGraph(
            name="test_no_scale",
            phase="train_backward",
            nodes={"comm_grad_reduce": comm_node},
            metadata={"seq_len": 4096, "hidden": 4096},
        )

        offload_pass = OffloadPass()
        chain_end = offload_pass._find_dp_chain_end(g, comm_node)

        assert chain_end is comm_node, (
            "without div/scale, chain end should be the comm node itself"
        )
