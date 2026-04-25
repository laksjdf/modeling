"""Test graph-native memory: activation liveness, recompute, ZeRO sharding, stage-aware inflight."""

import pytest
from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.types import TensorMeta, DType
from zrt.ir.edge import Edge
from zrt.transform.context import TransformContext, ParallelConfig, TrainingConfig
from zrt.transform.analysis.training import TrainingMemoryPass


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


def _make_stitched_graph(num_layers=2, hidden=4096, seq_len=2048):
    """Create a stitched forward+backward graph with fwd_bwd_stitched metadata.

    Structure: fwd_0 → fwd_1 → bwd_0 → bwd_1
    Edges from fwd to bwd represent saved activations.
    """
    nodes = {}
    edges = []
    tensor_bytes = seq_len * hidden * 2  # BF16

    for i in range(num_layers):
        out_t = TensorMeta(id=f"fwd_out_{i}", shape=(1, seq_len, hidden),
                           dtype=DType.BF16, mem_bytes=tensor_bytes)
        fwd_node = OpNode(
            id=f"fwd_{i}",
            op_type="aten.linear",
            inputs=[TensorMeta(id=f"fwd_in_{i}", shape=(1, seq_len, hidden),
                               dtype=DType.BF16, mem_bytes=tensor_bytes)],
            outputs=[out_t],
            scope=f"model.layers.{i}.mlp",
            layer=str(i),
            category="compute",
            annotations={"phase": "forward"},
        )
        nodes[fwd_node.id] = fwd_node

    for i in range(num_layers):
        bwd_out = TensorMeta(id=f"bwd_out_{i}", shape=(1, seq_len, hidden),
                             dtype=DType.BF16, mem_bytes=tensor_bytes)
        bwd_node = OpNode(
            id=f"bwd_{i}",
            op_type="aten.mm_backward",
            inputs=[TensorMeta(id=f"bwd_in_{i}", shape=(1, seq_len, hidden),
                               dtype=DType.BF16, mem_bytes=tensor_bytes)],
            outputs=[bwd_out],
            scope=f"model.layers.{i}.mlp",
            layer=str(i),
            category="compute",
            annotations={"phase": "backward"},
        )
        nodes[bwd_node.id] = bwd_node

    # Chain forward nodes: fwd_0 → fwd_1
    for i in range(1, num_layers):
        edges.append(Edge(
            src=f"fwd_{i-1}", src_idx=0, dst=f"fwd_{i}", dst_idx=0,
            tensor=nodes[f"fwd_{i}"].inputs[0],
        ))

    # Chain backward nodes: bwd_0 → bwd_1
    for i in range(1, num_layers):
        edges.append(Edge(
            src=f"bwd_{i-1}", src_idx=0, dst=f"bwd_{i}", dst_idx=0,
            tensor=nodes[f"bwd_{i}"].inputs[0],
        ))

    # Fwd→Bwd edges (saved activations): fwd_i → bwd_i for each layer
    for i in range(num_layers):
        fwd_out = nodes[f"fwd_{i}"].outputs[0]
        edges.append(Edge(src=f"fwd_{i}", src_idx=0, dst=f"bwd_{i}", dst_idx=0, tensor=fwd_out))

    return OpGraph(
        name="test_stitched",
        phase="train",
        nodes=nodes,
        edges=edges,
        metadata={
            "seq_len": seq_len,
            "hidden": hidden,
            "num_layers": num_layers,
            "fwd_bwd_stitched": True,
        },
    )


class TestGraphNativeActivation:
    """Tests for graph-native activation memory from fwd→bwd liveness."""

    def test_stitched_uses_graph_native_path(self):
        """When fwd_bwd_stitched=True, activations come from edge liveness."""
        graph = _make_stitched_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, pp=1, dp=1),
            training=TrainingConfig(micro_batch=1, global_batch=8),
        )

        result = TrainingMemoryPass().run(graph, ctx)
        breakdown = result.metadata["memory_breakdown"]

        # 2 fwd→bwd edges, each carrying tensor_bytes
        expected_activation = 2 * (2048 * 4096 * 2)  # 2 layers * tensor_bytes
        assert breakdown.activations == pytest.approx(expected_activation, rel=0.01)

    def test_stitched_scales_with_layers(self):
        """More layers → more fwd→bwd edges → more activation memory."""
        g2 = _make_stitched_graph(num_layers=2)
        g4 = _make_stitched_graph(num_layers=4)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, pp=1, dp=1),
            training=TrainingConfig(micro_batch=1, global_batch=8),
        )

        mem_2 = TrainingMemoryPass().run(g2, ctx).metadata["memory_breakdown"].activations
        mem_4 = TrainingMemoryPass().run(g4, ctx).metadata["memory_breakdown"].activations

        assert mem_4 == pytest.approx(mem_2 * 2, rel=0.01)

    def test_stitched_accepts_fwd_bwd_phase_aliases(self):
        """Graph-native path should accept real stitch labels: fwd/bwd."""
        graph = _make_stitched_graph(num_layers=2)
        for n in graph.nodes.values():
            phase = n.annotations.get("phase")
            if phase == "forward":
                n.annotations["phase"] = "fwd"
            elif phase == "backward":
                n.annotations["phase"] = "bwd"

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, pp=1, dp=1),
            training=TrainingConfig(micro_batch=1, global_batch=8),
        )

        result = TrainingMemoryPass().run(graph, ctx)
        breakdown = result.metadata["memory_breakdown"]

        expected_activation = 2 * (2048 * 4096 * 2)
        assert breakdown.activations == pytest.approx(expected_activation, rel=0.01)


class TestRecomputeReducesActivation:
    """Tests for recompute annotations reducing saved activations."""

    def test_recompute_excludes_node_outputs(self):
        """Nodes with recompute=True have their outputs excluded from activation memory."""
        graph = _make_stitched_graph(num_layers=2)

        # Mark fwd_0 as recomputed — its output should not count
        graph.nodes["fwd_0"].annotations["recompute"] = True

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, pp=1, dp=1),
            training=TrainingConfig(micro_batch=1, global_batch=8),
        )

        result = TrainingMemoryPass().run(graph, ctx)
        breakdown = result.metadata["memory_breakdown"]

        # Only fwd_1 → bwd_1 edge counts (fwd_0 is recomputed)
        expected = 1 * (2048 * 4096 * 2)
        assert breakdown.activations == pytest.approx(expected, rel=0.01)

    def test_no_recompute_full_activation(self):
        """Without recompute, all fwd→bwd edges contribute."""
        graph = _make_stitched_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, pp=1, dp=1),
            training=TrainingConfig(micro_batch=1, global_batch=8),
        )

        result = TrainingMemoryPass().run(graph, ctx)
        breakdown = result.metadata["memory_breakdown"]

        expected = 2 * (2048 * 4096 * 2)
        assert breakdown.activations == pytest.approx(expected, rel=0.01)


class TestZeroShardFactors:
    """Tests for ZeRO sharding factors on weights/grads/opt-state."""

    @staticmethod
    def _make_graph_with_weights(hidden=4096, seq_len=2048):
        """Graph with weight tensors so count_params > 0."""
        nodes = {
            "mm_0": OpNode(
                id="mm_0",
                op_type="aten.linear",
                inputs=[
                    TensorMeta(id="in_0", shape=(1, seq_len, hidden), dtype=DType.BF16,
                               mem_bytes=seq_len * hidden * 2),
                    TensorMeta(id="w_0", shape=(hidden, hidden), dtype=DType.BF16,
                               mem_bytes=hidden * hidden * 2),
                ],
                outputs=[
                    TensorMeta(id="out_0", shape=(1, seq_len, hidden), dtype=DType.BF16,
                               mem_bytes=seq_len * hidden * 2),
                ],
                scope="model.layers.0.self_attn.q_proj",
                category="compute",
            ),
        }
        return OpGraph(name="weighted", phase="forward", nodes=nodes, edges=[],
                       metadata={"seq_len": seq_len, "hidden": hidden, "num_layers": 1})

    def test_zero1_reduces_opt_state(self):
        graph = self._make_graph_with_weights()
        hw = _make_hardware_spec()

        ctx_z0 = TransformContext(
            hw_spec=hw,
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=0),
        )
        ctx_z1 = TransformContext(
            hw_spec=hw,
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=1),
        )

        mem_z0 = TrainingMemoryPass().run(graph, ctx_z0).metadata["memory_breakdown"]
        mem_z1 = TrainingMemoryPass().run(graph, ctx_z1).metadata["memory_breakdown"]

        # ZeRO-1 should shard opt_state by dp=4
        assert mem_z1.opt_state == pytest.approx(mem_z0.opt_state / 4, rel=0.01)

    def test_zero2_reduces_grads(self):
        graph = self._make_graph_with_weights()
        hw = _make_hardware_spec()

        ctx_z1 = TransformContext(
            hw_spec=hw,
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=1),
        )
        ctx_z2 = TransformContext(
            hw_spec=hw,
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=2),
        )

        mem_z1 = TrainingMemoryPass().run(graph, ctx_z1).metadata["memory_breakdown"]
        mem_z2 = TrainingMemoryPass().run(graph, ctx_z2).metadata["memory_breakdown"]

        # ZeRO-2 should shard grads by dp=4
        assert mem_z2.grads < mem_z1.grads


class TestStageAwareInflight:
    """Tests for stage-aware inflight depth using stage_id annotations."""

    def test_stage_id_reduces_inflight(self):
        """With stage_id annotations, inflight = pp - min(stage_id)."""
        graph = _make_stitched_graph(num_layers=2)
        # Don't use stitched path — test the Korthikanti path with stage_id
        del graph.metadata["fwd_bwd_stitched"]

        # Add stage_id annotations to nodes
        for n in graph.nodes.values():
            n.annotations["stage_id"] = 1

        hw = _make_hardware_spec()
        ctx = TransformContext(
            hw_spec=hw,
            parallel=ParallelConfig(tp=1, pp=4, dp=1),
            training=TrainingConfig(micro_batch=1, global_batch=8),
        )

        result = TrainingMemoryPass().run(graph, ctx)
        breakdown_stage = result.metadata["memory_breakdown"]

        # Without stage_id, inflight = pp = 4
        graph_no = _make_stitched_graph(num_layers=2)
        del graph_no.metadata["fwd_bwd_stitched"]

        result_no = TrainingMemoryPass().run(graph_no, ctx)
        breakdown_no = result_no.metadata["memory_breakdown"]

        # With stage_id=1, inflight = pp - 1 = 3
        # With no stage_id, inflight = pp = 4
        assert breakdown_stage.activations < breakdown_no.activations

    def test_different_stages_different_inflight(self):
        """Different stages should yield different peak activation memory."""
        hw = _make_hardware_spec()

        mems = {}
        for stage_id in [0, 1, 2]:
            graph = _make_stitched_graph(num_layers=2)
            del graph.metadata["fwd_bwd_stitched"]
            for n in graph.nodes.values():
                n.annotations["stage_id"] = stage_id

            ctx = TransformContext(
                hw_spec=hw,
                parallel=ParallelConfig(tp=1, pp=4, dp=1),
                training=TrainingConfig(micro_batch=1, global_batch=8),
            )

            result = TrainingMemoryPass().run(graph, ctx)
            mems[stage_id] = result.metadata["memory_breakdown"].activations

        # stage 0 has highest inflight (pp - 0 = 4)
        # stage 1 has inflight pp - 1 = 3
        # stage 2 has inflight pp - 2 = 2
        assert mems[0] > mems[1] > mems[2]
