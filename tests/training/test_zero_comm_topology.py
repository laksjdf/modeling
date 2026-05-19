"""Test ZeRO communication topology: global vs per-layer insertion."""

import pytest
from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.types import TensorMeta, DType
from zrt.ir.edge import Edge
from zrt.transform.context import TransformContext, ParallelConfig, TrainingConfig
from zrt.transform.parallel.data_parallel import DataParallelPass
from zrt.transform.training.zero_fsdp import ZeroFSDPPass


def _make_stitched_graph(num_layers=3, hidden=4096, seq_len=128):
    """Create a stitched graph with forward and backward nodes for multiple layers."""
    nodes = {}
    edges = []

    # Create forward nodes (chain: layer0 -> layer1 -> layer2)
    for i in range(num_layers):
        fwd_out = TensorMeta(
            id=f"fwd_out_{i}", shape=(1, seq_len, hidden),
            dtype=DType.BF16, mem_bytes=seq_len * hidden * 2,
        )
        fwd_node = OpNode(
            id=f"fused_{i}_op_fwd",
            op_type="aten.mm",
            inputs=[TensorMeta(id=f"fwd_in_{i}", shape=(1, seq_len, hidden),
                               dtype=DType.BF16, mem_bytes=seq_len * hidden * 2)],
            outputs=[fwd_out],
            scope=f"model.layers.{i}.self_attn",
            layer=str(i),
            category="compute",
        )
        fwd_node.annotations["phase"] = "fwd"
        nodes[fwd_node.id] = fwd_node

        if i > 0:
            edges.append(Edge(
                src=f"fused_{i-1}_op_fwd", src_idx=0,
                dst=fwd_node.id, dst_idx=0, tensor=fwd_out,
            ))

    # Create backward nodes (chain: layer0 -> layer1 -> layer2)
    for i in range(num_layers):
        bwd_out = TensorMeta(
            id=f"bwd_out_{i}", shape=(1, seq_len, hidden),
            dtype=DType.BF16, mem_bytes=seq_len * hidden * 2,
        )
        bwd_node = OpNode(
            id=f"bwd_op_{i}",
            op_type="aten.mm_backward",
            inputs=[TensorMeta(id=f"bwd_in_{i}", shape=(1, seq_len, hidden),
                               dtype=DType.BF16, mem_bytes=seq_len * hidden * 2)],
            outputs=[bwd_out],
            scope=f"model.layers.{i}.self_attn",
            layer=str(i),
            category="compute",
        )
        bwd_node.annotations["phase"] = "bwd"
        nodes[bwd_node.id] = bwd_node

        if i > 0:
            edges.append(Edge(
                src=f"bwd_op_{i-1}", src_idx=0,
                dst=bwd_node.id, dst_idx=0, tensor=bwd_out,
            ))

    # Connect forward tail to backward head to simulate stitched graph
    if num_layers > 0:
        edges.append(Edge(
            src=f"fused_{num_layers-1}_op_fwd", src_idx=0,
            dst="bwd_op_0", dst_idx=0,
            tensor=TensorMeta(id="stitch_edge", shape=(1,), dtype=DType.BF16, mem_bytes=2),
        ))

    return OpGraph(
        name="test_stitched_model",
        phase="train",
        nodes=nodes,
        edges=edges,
        metadata={"num_layers": num_layers},
    )


def _make_ctx(zero_stage, dp=4):
    """Create a TransformContext for testing."""
    from zrt.hardware.spec import HardwareSpec, ComputeSpec, MemorySpec, InterconnectSpec, LinkSpec
    hw = HardwareSpec(
        name="test", vendor="test", device_type="gpu",
        compute=ComputeSpec(bf16_tflops=1000),
        memory=MemorySpec(capacity_gb=80, hbm_bandwidth_gbps=3000),
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="nvlink", num_devices=8, bandwidth_gbps=900, latency_us=1.0),
            inter_node=LinkSpec(type="ib", num_devices=1000, bandwidth_gbps=400, latency_us=5.0),
        ),
    )
    return TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, dp=dp),
        training=TrainingConfig(micro_batch=1, global_batch=dp, zero_stage=zero_stage),
    )


class TestZeROCommunicationTopology:
    """Verify exact communication node insertion per ZeRO stage."""

    def test_zero1_global_allreduce(self):
        """ZeRO-1: Exactly one global all_reduce at the end of backward pass."""
        graph = _make_stitched_graph(num_layers=3)
        graph.metadata["phase"] = "train_backward"  # DP pass expects backward phase
        ctx = _make_ctx(zero_stage=1, dp=4)

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values() if n.annotations.get("dp_comm")]
        assert len(dp_nodes) == 1, f"Expected 1 DP comm node, got {len(dp_nodes)}"
        node = dp_nodes[0]
        assert node.op_type == "comm.all_reduce"
        assert node.id == "comm_dp_grad_reduce"
        assert node.attrs["collective"] == "all_reduce"
        assert node.attrs["group_size"] == 4

    def test_zero2_global_reducescatter(self):
        """ZeRO-2: Exactly one global reduce_scatter at the end of backward pass."""
        graph = _make_stitched_graph(num_layers=3)
        graph.metadata["phase"] = "train_backward"
        ctx = _make_ctx(zero_stage=2, dp=4)

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values() if n.annotations.get("dp_comm")]
        assert len(dp_nodes) == 1, f"Expected 1 DP comm node, got {len(dp_nodes)}"
        node = dp_nodes[0]
        assert node.op_type == "comm.reduce_scatter"
        assert node.id == "comm_dp_grad_reduce"
        assert node.attrs["collective"] == "reduce_scatter"
        assert node.attrs["group_size"] == 4

    def test_zero3_skips_dp_pass(self):
        """ZeRO-3: DataParallelPass should insert NO nodes (handled by ZeroFSDPPass)."""
        graph = _make_stitched_graph(num_layers=3)
        ctx = _make_ctx(zero_stage=3, dp=4)

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values() if n.annotations.get("dp_comm")]
        assert len(dp_nodes) == 0, "DataParallelPass should skip ZeRO-3"

    def test_zero3_per_layer_fsdp_comm(self):
        """ZeRO-3: ZeroFSDPPass inserts per-layer AllGather + ReduceScatter."""
        graph = _make_stitched_graph(num_layers=3)
        ctx = _make_ctx(zero_stage=3, dp=4)

        # DP pass skips ZeRO-3, pass result to ZeroFSDPPass
        after_dp = DataParallelPass().run(graph, ctx)
        result = ZeroFSDPPass().run(after_dp, ctx)

        fsdp_nodes = [n for n in result.nodes.values()
                      if n.annotations.get("inserted_by") == "zero_fsdp_pass"]

        # Verify per-layer insertion: at least 1 AG and 1 RS per layer
        ag_count = sum(1 for n in fsdp_nodes if n.op_type == "comm.all_gather")
        rs_count = sum(1 for n in fsdp_nodes if n.op_type == "comm.reduce_scatter")

        assert ag_count == 3, f"Expected ==3 AllGather nodes (1 per layer), got {ag_count}"
        assert rs_count == 3, f"Expected ==3 ReduceScatter nodes (1 per layer), got {rs_count}"
        assert len(fsdp_nodes) >= 6, f"Expected >=6 FSDP comm nodes, got {len(fsdp_nodes)}"

    def test_zero3_fsdp_comm_per_layer_scope(self):
        """ZeRO-3: Each layer should have its own AG/RS nodes."""
        graph = _make_stitched_graph(num_layers=3)
        ctx = _make_ctx(zero_stage=3, dp=4)

        after_dp = DataParallelPass().run(graph, ctx)
        result = ZeroFSDPPass().run(after_dp, ctx)

        fsdp_nodes = [n for n in result.nodes.values()
                      if n.annotations.get("inserted_by") == "zero_fsdp_pass"]

        # Verify nodes are distributed across layers
        layer_scopes = set()
        for n in fsdp_nodes:
            # scope format: model.layers.{i}.self_attn
            parts = n.scope.split(".")
            if len(parts) >= 3 and parts[1] == "layers":
                layer_scopes.add(parts[2])

        assert len(layer_scopes) == 3, f"Expected comm nodes in 3 layers, found in {len(layer_scopes)}"

    def test_zero1_vs_zero2_comm_node_count(self):
        """ZeRO-1 and ZeRO-2 should both insert exactly 1 global comm node."""
        for stage in [1, 2]:
            graph = _make_stitched_graph(num_layers=4)
            graph.metadata["phase"] = "train_backward"
            ctx = _make_ctx(zero_stage=stage, dp=4)

            result = DataParallelPass().run(graph, ctx)
            dp_nodes = [n for n in result.nodes.values() if n.annotations.get("dp_comm")]
            assert len(dp_nodes) == 1, f"ZeRO-{stage} should have exactly 1 global comm node"
