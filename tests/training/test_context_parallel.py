"""Test context parallel pass: Ulysses A2A, Ring P2P, and metadata completeness."""

import pytest
from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.types import TensorMeta, DType
from zrt.ir.edge import Edge
from zrt.transform.context import TransformContext, ParallelConfig, TrainingConfig
from zrt.transform.parallel.context_parallel import ContextParallelPass
from zrt.transform.parallel.comm_inserter import CommInserterPass


def _make_attn_graph(seq_len=2048, hidden=4096, num_layers=2):
    """Create a graph with attention ops that CP pass targets."""
    nodes = {}
    edges = []

    inp = TensorMeta(id="input_0", shape=(1, seq_len, hidden), dtype=DType.BF16,
                     mem_bytes=seq_len * hidden * 2)
    qkv_out = TensorMeta(id="qkv_0", shape=(1, seq_len, hidden * 3), dtype=DType.BF16,
                         mem_bytes=seq_len * hidden * 3 * 2)
    attn_out = TensorMeta(id="attn_out_0", shape=(1, seq_len, hidden), dtype=DType.BF16,
                          mem_bytes=seq_len * hidden * 2)

    for i in range(num_layers):
        qkv = OpNode(
            id=f"qkv_proj_{i}",
            op_type="aten.linear",
            inputs=[inp],
            outputs=[qkv_out],
            scope=f"model.layers.{i}.self_attn.qkv_proj",
            category="compute",
        )
        attn = OpNode(
            id=f"sdpa_{i}",
            op_type="aten._scaled_dot_product_attention",
            inputs=[qkv_out],
            outputs=[attn_out],
            scope=f"model.layers.{i}.self_attn",
            category="compute",
        )
        o_proj = OpNode(
            id=f"o_proj_{i}",
            op_type="aten.linear",
            inputs=[attn_out],
            outputs=[inp],
            scope=f"model.layers.{i}.self_attn.o_proj",
            category="compute",
        )
        nodes[qkv.id] = qkv
        nodes[attn.id] = attn
        nodes[o_proj.id] = o_proj

        edges.append(Edge(src=qkv.id, src_idx=0, dst=attn.id, dst_idx=0, tensor=qkv_out))
        edges.append(Edge(src=attn.id, src_idx=0, dst=o_proj.id, dst_idx=0, tensor=attn_out))

    return OpGraph(
        name="test_cp_model",
        phase="forward",
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


class TestUlyssesCP:
    """Tests for Ulysses-style context parallel (A2A around attention)."""

    def test_ulysses_inserts_pre_post_a2a(self):
        graph = _make_attn_graph()
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, cp_kind="ulysses"),
        )

        g = ContextParallelPass().run(graph, ctx)
        g = CommInserterPass().run(g, ctx)

        attn_nodes = [n for n in g.nodes.values()
                      if n.op_type == "aten._scaled_dot_product_attention"]
        assert len(attn_nodes) == 2

        for attn in attn_nodes:
            pre_id = f"comm_a2a_cp_pre_{attn.id}"
            post_id = f"comm_a2a_cp_post_{attn.id}"
            assert pre_id in g.nodes, f"Missing pre-A2A node for {attn.id}"
            assert post_id in g.nodes, f"Missing post-A2A node for {attn.id}"
            assert g.nodes[pre_id].op_type == "comm.all_to_all"
            assert g.nodes[post_id].op_type == "comm.all_to_all"
            assert g.nodes[pre_id].attrs["role"] == "cp_ulysses_pre"
            assert g.nodes[post_id].attrs["role"] == "cp_ulysses_post"

    def test_ulysses_message_size(self):
        seq_len, hidden, cp = 2048, 4096, 4
        graph = _make_attn_graph(seq_len=seq_len, hidden=hidden)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(micro_batch=2, global_batch=8, cp_kind="ulysses"),
        )

        g = ContextParallelPass().run(graph, ctx)
        g = CommInserterPass().run(g, ctx)

        # Expected: micro_batch * (seq_len / cp) * hidden * 2
        expected_bytes = 2 * (seq_len // cp) * hidden * 2

        for n in g.nodes.values():
            if n.op_type == "comm.all_to_all" and "cp_ulysses" in n.attrs.get("role", ""):
                assert n.attrs["message_size_bytes"] == expected_bytes, (
                    f"Expected {expected_bytes}, got {n.attrs['message_size_bytes']}"
                )
                assert n.attrs["msg_bytes"] == expected_bytes

    def test_ulysses_group_size(self):
        graph = _make_attn_graph()
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, cp_kind="ulysses"),
        )
        g = ContextParallelPass().run(graph, ctx)
        g = CommInserterPass().run(g, ctx)

        for n in g.nodes.values():
            if n.op_type == "comm.all_to_all" and "cp_ulysses" in n.attrs.get("role", ""):
                assert n.attrs["group_size"] == 4


class TestRingCP:
    """Tests for Ring-style context parallel (P2P rounds around attention)."""

    def test_ring_inserts_p2p_rounds(self):
        cp = 4
        graph = _make_attn_graph()
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(micro_batch=1, global_batch=8, cp_kind="ring"),
        )

        g = ContextParallelPass().run(graph, ctx)
        g = CommInserterPass().run(g, ctx)

        attn_nodes = [n for n in g.nodes.values()
                      if n.op_type == "aten._scaled_dot_product_attention"]
        assert len(attn_nodes) == 2

        for attn in attn_nodes:
            # Should have exactly cp P2P rounds per attention op
            p2p_nodes = [
                n for n in g.nodes.values()
                if n.op_type == "comm.send_recv"
                and n.attrs.get("role") == "cp_ring"
                and f"ring_{attn.id}_" in n.id
            ]
            assert len(p2p_nodes) == cp, (
                f"Expected {cp} P2P rounds for {attn.id}, got {len(p2p_nodes)}"
            )

    def test_ring_message_size(self):
        seq_len, hidden, cp = 2048, 4096, 4
        graph = _make_attn_graph(seq_len=seq_len, hidden=hidden)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(micro_batch=2, global_batch=8, cp_kind="ring"),
        )

        g = ContextParallelPass().run(graph, ctx)
        g = CommInserterPass().run(g, ctx)

        expected_bytes = 2 * (seq_len // cp) * hidden * 2

        p2p_nodes = [n for n in g.nodes.values()
                     if n.op_type == "comm.send_recv" and n.attrs.get("role") == "cp_ring"]
        for p2p in p2p_nodes:
            assert p2p.attrs["message_size_bytes"] == expected_bytes
            assert p2p.attrs["msg_bytes"] == expected_bytes

    def test_ring_overlap_target_format(self):
        graph = _make_attn_graph()
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, cp_kind="ring"),
        )

        g = ContextParallelPass().run(graph, ctx)
        g = CommInserterPass().run(g, ctx)

        p2p_nodes = [n for n in g.nodes.values()
                     if n.op_type == "comm.send_recv" and n.attrs.get("role") == "cp_ring"]
        for p2p in p2p_nodes:
            target = p2p.annotations.get("overlap_target", "")
            assert target.startswith("fa_tile:"), (
                f"Expected overlap_target 'fa_tile:...', got '{target}'"
            )

    def test_ring_round_and_scope_attrs(self):
        graph = _make_attn_graph()
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, cp_kind="ring"),
        )

        g = ContextParallelPass().run(graph, ctx)
        g = CommInserterPass().run(g, ctx)

        p2p_nodes = [n for n in g.nodes.values()
                     if n.op_type == "comm.send_recv" and n.attrs.get("role") == "cp_ring"]
        for p2p in p2p_nodes:
            assert "round" in p2p.attrs, f"P2P node {p2p.id} missing round attr"
            assert isinstance(p2p.attrs["round"], int)
            assert "scope" in p2p.attrs
            assert "layer" in p2p.attrs


class TestHybridAndCompressedCP:
    """Tests for graph-native CP annotations that must be consumed by comm insertion."""

    def test_hybrid_inserts_ulysses_a2a_and_ring_p2p(self):
        graph = _make_attn_graph(num_layers=1)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, cp_kind="hybrid"),
        )

        g = ContextParallelPass().run(graph, ctx)
        g = CommInserterPass().run(g, ctx)

        roles = [n.attrs.get("role") for n in g.nodes.values()]
        assert "cp_hybrid_ulysses_pre" in roles
        assert "cp_hybrid_ulysses_post" in roles
        # The graph fixture marks qkv/attention/o_proj scopes as attention-like.
        assert roles.count("cp_hybrid_ring") == 3 * 4

    def test_compressed_inserts_stage1_p2p_and_stage2_allgather(self):
        graph = _make_attn_graph(num_layers=1)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, cp_kind="compressed"),
        )

        g = ContextParallelPass().run(graph, ctx)
        g = CommInserterPass().run(g, ctx)

        roles = {n.attrs.get("role") for n in g.nodes.values()}
        assert "cp_compressed_stage1" in roles
        assert "cp_compressed_stage2" in roles


class TestCPSkip:
    """Tests that CP pass is skipped when cp <= 1."""

    def test_cp1_no_insertion(self):
        graph = _make_attn_graph()
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=1),
            training=TrainingConfig(micro_batch=1, global_batch=8),
        )

        g = ContextParallelPass().run(graph, ctx)
        g = CommInserterPass().run(g, ctx)

        cp_nodes = [n for n in g.nodes.values()
                    if "cp_ulysses" in n.attrs.get("role", "")
                    or n.attrs.get("role") == "cp_ring"]
        assert len(cp_nodes) == 0
