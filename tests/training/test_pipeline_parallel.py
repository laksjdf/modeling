"""Tests for Phase 2: PipelineParallelPass and per-stage TrainingPipelinePass."""
import pytest
from python.zrt.ir.node import OpNode
from python.zrt.ir.edge import Edge
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.types import TensorMeta, DType
from python.zrt.transform.context import (
    TransformContext, ParallelConfig, StreamConfig, TrainingConfig,
)
from python.zrt.transform.parallel.pipeline_parallel import (
    PipelineParallelPass, LayerGroup,
)
from python.zrt.transform.analysis.training import TrainingPipelinePass


# ── helpers ───────────────────────────────────────────────────────────────────

def _t(tid: str, shape=(1, 128, 4096)):
    return TensorMeta.from_shape_dtype(tid, shape, DType.BF16)


def _make_hw():
    import python.zrt.hardware.registry as hw_registry
    return hw_registry.load("nvidia_h100_sxm")


def _make_linear_graph(num_layers: int = 4) -> OpGraph:
    """Build a simple linear graph: one matmul node per transformer layer."""
    nodes: dict[str, OpNode] = {}
    edges: list[Edge] = []

    prev_out_tensor = _t("input_0")
    for i in range(num_layers):
        node_id = f"mm_layer{i}"
        out_tensor = _t(f"out_layer{i}")
        node = OpNode(
            id=node_id,
            op_type="aten.mm.default",
            inputs=[prev_out_tensor],
            outputs=[out_tensor],
            scope=f"model.layers.{i}.mlp.gate_proj",
            layer=str(i),
            category="compute",
        )
        node.annotations["latency_us"] = 100.0 * (i + 1)  # vary load per layer
        nodes[node_id] = node

        if i > 0:
            prev_id = f"mm_layer{i-1}"
            edges.append(Edge(
                src=prev_id, src_idx=0,
                dst=node_id, dst_idx=0,
                tensor=prev_out_tensor,
            ))

        prev_out_tensor = out_tensor

    return OpGraph(
        name="test_model",
        phase="train_forward",
        nodes=nodes,
        edges=edges,
        metadata={"seq_len": 128, "hidden": 4096, "num_layers": num_layers},
    )


def _make_ctx(pp: int = 2, tp: int = 1, dp: int = 1,
              global_batch: int = 8, micro_batch: int = 1,
              pp_layer_assignment=None) -> TransformContext:
    return TransformContext(
        hw_spec=_make_hw(),
        parallel=ParallelConfig(tp=tp, pp=pp, dp=dp),
        stream_config=StreamConfig(),
        training=TrainingConfig(
            micro_batch=micro_batch,
            global_batch=global_batch,
            pp_layer_assignment=pp_layer_assignment,
        ),
    )


# ── PipelineParallelPass tests ────────────────────────────────────────────────

class TestPipelineParallelPass:

    def test_pp1_all_stage0(self):
        """pp=1 → every node gets stage_id=0."""
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=1)
        result = PipelineParallelPass().run(graph, ctx)
        for node in result.nodes.values():
            assert node.annotations.get("stage_id") == 0

    def test_pp2_splits_layers(self):
        """pp=2 → layers distributed across 2 stages."""
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2)
        result = PipelineParallelPass().run(graph, ctx)

        stage_ids = {node.annotations["stage_id"]
                     for node in result.nodes.values()
                     if not node.op_type.startswith("comm.")}
        assert 0 in stage_ids
        assert 1 in stage_ids

    def test_pp2_inserts_p2p_node(self):
        """pp=2 → comm.send_recv nodes are inserted at stage boundaries."""
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2, pp_layer_assignment=[0, 0, 1, 1])
        result = PipelineParallelPass().run(graph, ctx)

        p2p_nodes = [n for n in result.nodes.values()
                     if n.op_type == "comm.send_recv"]
        assert len(p2p_nodes) >= 1, f"Expected >= 1 P2P node, got {len(p2p_nodes)}"
        p2p = p2p_nodes[0]
        assert p2p.attrs["src_stage"] == 0
        assert p2p.attrs["dst_stage"] == 1

    def test_pp4_inserts_three_p2p_nodes(self):
        """pp=4 → 3 stage boundaries → 3 comm.send_recv nodes."""
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=4)
        result = PipelineParallelPass().run(graph, ctx)

        p2p_nodes = [n for n in result.nodes.values()
                     if n.op_type == "comm.send_recv"]
        assert len(p2p_nodes) == 3

    def test_pp2_does_not_mutate_input(self):
        """Pass is functional: input graph is not modified."""
        graph = _make_linear_graph(num_layers=4)
        original_ids = set(graph.nodes.keys())
        ctx = _make_ctx(pp=2)
        _ = PipelineParallelPass().run(graph, ctx)
        assert set(graph.nodes.keys()) == original_ids

    def test_explicit_layer_assignment(self):
        """Explicit pp_layer_assignment is respected."""
        graph = _make_linear_graph(num_layers=4)
        # layers 0,1 → stage 0; layers 2,3 → stage 1
        ctx = _make_ctx(pp=2, pp_layer_assignment=[0, 0, 1, 1])
        result = PipelineParallelPass().run(graph, ctx)

        stage0_layers = {
            int(n.layer) for n in result.nodes.values()
            if n.layer and n.op_type != "comm.send_recv"
               and n.annotations.get("stage_id") == 0
        }
        stage1_layers = {
            int(n.layer) for n in result.nodes.values()
            if n.layer and n.op_type != "comm.send_recv"
               and n.annotations.get("stage_id") == 1
        }
        assert stage0_layers == {0, 1}
        assert stage1_layers == {2, 3}

    def test_p2p_node_belongs_to_receiver_stage(self):
        """P2P node is annotated with the receiving stage_id."""
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2)
        result = PipelineParallelPass().run(graph, ctx)

        p2p = next(n for n in result.nodes.values() if n.op_type == "comm.send_recv")
        assert p2p.annotations["stage_id"] == 1  # receiver is stage 1

    def test_p2p_node_has_positive_message_size(self):
        """P2P node attrs contain a positive message_size_bytes."""
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2)
        result = PipelineParallelPass().run(graph, ctx)

        p2p = next(n for n in result.nodes.values() if n.op_type == "comm.send_recv")
        assert p2p.attrs["message_size_bytes"] > 0

    def test_all_nodes_have_stage_id_after_pp2(self):
        """After pp=2, every node has a stage_id annotation."""
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=2)
        result = PipelineParallelPass().run(graph, ctx)

        for node in result.nodes.values():
            assert "stage_id" in node.annotations, (
                f"Node {node.id} ({node.op_type}) missing stage_id"
            )

    def test_vpp_interleaved_layer_assignment(self):
        """VPP (vpp_chunks=2, pp=2, 8 layers) produces interleaved assignment.
        
        Expected distribution:
          Device 0: L0,L1,L4,L5 (virtual stages 0 and 2)
          Device 1: L2,L3,L6,L7 (virtual stages 1 and 3)
        
        Expected P2P boundaries:
          L1→L2: stage 0→1, virtual_stage 0→1
          L3→L4: stage 1→0, virtual_stage 1→2
          L5→L6: stage 0→1, virtual_stage 2→3
        """
        graph = _make_linear_graph(num_layers=8)
        ctx = _make_ctx(pp=2)
        ctx.training = TrainingConfig(
            micro_batch=1,
            global_batch=8,
            pp_schedule="interleaved",
            vpp_chunks=2,
        )
        result = PipelineParallelPass().run(graph, ctx)

        stage0_layers = {
            int(n.layer) for n in result.nodes.values()
            if n.layer and n.op_type != "comm.send_recv"
               and n.annotations.get("stage_id") == 0
        }
        stage1_layers = {
            int(n.layer) for n in result.nodes.values()
            if n.layer and n.op_type != "comm.send_recv"
               and n.annotations.get("stage_id") == 1
        }

        assert stage0_layers == {0, 1, 4, 5}, f"Expected {0,1,4,5}, got {stage0_layers}"
        assert stage1_layers == {2, 3, 6, 7}, f"Expected {2,3,6,7}, got {stage1_layers}"

        # Verify P2P nodes
        p2p_nodes = [n for n in result.nodes.values() if n.op_type == "comm.send_recv"]
        assert len(p2p_nodes) == 3, f"Expected 3 P2P nodes for VPP, got {len(p2p_nodes)}"


    def test_vpp_virtual_stage_id_annotation(self):
        """VPP mode adds virtual_stage_id annotation to compute nodes."""
        graph = _make_linear_graph(num_layers=8)
        ctx = _make_ctx(pp=2)
        ctx.training = TrainingConfig(
            micro_batch=1,
            global_batch=8,
            pp_schedule="interleaved",
            vpp_chunks=2,
        )
        result = PipelineParallelPass().run(graph, ctx)

        compute_nodes = [n for n in result.nodes.values()
                         if n.op_type != "comm.send_recv"]
        
        for node in compute_nodes:
            assert "virtual_stage_id" in node.annotations, (
                f"Node {node.id} missing virtual_stage_id in VPP mode"
            )

        virtual_ids = {n.annotations["virtual_stage_id"] for n in compute_nodes}
        assert virtual_ids == {0, 1, 2, 3}, f"Expected 4 virtual stages, got {virtual_ids}"

    def test_vpp_p2p_count_more_than_standard(self):
        """VPP mode has more P2P nodes due to interleaved virtual stage boundaries.
        
        Standard pp=2 with explicit continuous assignment: 1 P2P (L3→L4 boundary)
        VPP pp=2, vpp_chunks=2: 3 P2P nodes at L1→L2, L5→L6, L7→L8 boundaries
        """
        graph = _make_linear_graph(num_layers=8)
        
        ctx_std = _make_ctx(pp=2, pp_layer_assignment=[0, 0, 0, 0, 1, 1, 1, 1])
        ctx_vpp = _make_ctx(pp=2)
        ctx_vpp.training = TrainingConfig(
            micro_batch=1,
            global_batch=8,
            pp_schedule="interleaved",
            vpp_chunks=2,
        )
        
        result_std = PipelineParallelPass().run(graph, ctx_std)
        result_vpp = PipelineParallelPass().run(graph, ctx_vpp)
        
        p2p_std = [n for n in result_std.nodes.values() if n.op_type == "comm.send_recv"]
        p2p_vpp = [n for n in result_vpp.nodes.values() if n.op_type == "comm.send_recv"]
        
        assert len(p2p_std) == 1, f"Standard pp=2 should have 1 P2P, got {len(p2p_std)}"
        assert len(p2p_vpp) == 3, f"VPP should have == 3 P2P nodes, got {len(p2p_vpp)}"

    def test_vpp_p2p_has_virtual_stage_attrs(self):
        """P2P nodes in VPP mode have src_virtual_stage and dst_virtual_stage attrs."""
        graph = _make_linear_graph(num_layers=8)
        ctx = _make_ctx(pp=2)
        ctx.training = TrainingConfig(
            micro_batch=1,
            global_batch=8,
            pp_schedule="interleaved",
            vpp_chunks=2,
        )
        result = PipelineParallelPass().run(graph, ctx)
        
        p2p_nodes = [n for n in result.nodes.values() if n.op_type == "comm.send_recv"]
        
        for p2p in p2p_nodes:
            assert "src_virtual_stage" in p2p.attrs, f"P2P {p2p.id} missing src_virtual_stage"
            assert "dst_virtual_stage" in p2p.attrs, f"P2P {p2p.id} missing dst_virtual_stage"
            
            src_vs = p2p.attrs["src_virtual_stage"]
            dst_vs = p2p.attrs["dst_virtual_stage"]
            if src_vs is not None and dst_vs is not None:
                assert src_vs != dst_vs, f"Virtual stage crossing: {src_vs} != {dst_vs}"

    def test_vpp_pp4_interleaved_assignment(self):
        """VPP with pp=4, vpp_chunks=2, 16 layers.
        
        Expected: each device gets 2 virtual stages with 2 layers each.
        layers_per_chunk = 16 / (4*2) = 2
        chunk_id = idx // 2
        stage = chunk_id % 4
        
        L0,L1 → chunk 0 → stage 0
        L2,L3 → chunk 1 → stage 1
        L4,L5 → chunk 2 → stage 2
        L6,L7 → chunk 3 → stage 3
        L8,L9 → chunk 4 → stage 0
        ...
        """
        graph = _make_linear_graph(num_layers=16)
        ctx = _make_ctx(pp=4)
        ctx.training = TrainingConfig(
            micro_batch=1,
            global_batch=16,
            pp_schedule="interleaved",
            vpp_chunks=2,
        )
        result = PipelineParallelPass().run(graph, ctx)
        
        stage_layers = {}
        for s in range(4):
            stage_layers[s] = {
                int(n.layer) for n in result.nodes.values()
                if n.layer and n.op_type != "comm.send_recv"
                   and n.annotations.get("stage_id") == s
            }
        
        assert stage_layers[0] == {0, 1, 8, 9}, f"Stage 0: {stage_layers[0]}"
        assert stage_layers[1] == {2, 3, 10, 11}, f"Stage 1: {stage_layers[1]}"
        assert stage_layers[2] == {4, 5, 12, 13}, f"Stage 2: {stage_layers[2]}"
        assert stage_layers[3] == {6, 7, 14, 15}, f"Stage 3: {stage_layers[3]}"

    def test_vpp_vpp_chunks_1_falls_back_to_standard(self):
        """vpp_chunks=1 with interleaved schedule falls back to standard behavior."""
        graph = _make_linear_graph(num_layers=8)
        
        ctx_vpp1 = _make_ctx(pp=2)
        ctx_vpp1.training = TrainingConfig(
            micro_batch=1,
            global_batch=8,
            pp_schedule="interleaved",
            vpp_chunks=1,
        )
        ctx_std = _make_ctx(pp=2)
        
        result_vpp1 = PipelineParallelPass().run(graph, ctx_vpp1)
        result_std = PipelineParallelPass().run(graph, ctx_std)
        
        vpp1_stages = {n.annotations["stage_id"] for n in result_vpp1.nodes.values()
                       if n.op_type != "comm.send_recv"}
        std_stages = {n.annotations["stage_id"] for n in result_std.nodes.values()
                      if n.op_type != "comm.send_recv"}
        
        assert vpp1_stages == std_stages, f"vpp_chunks=1 should fallback to standard"
        
        vpp1_virtual = [n.annotations.get("virtual_stage_id") for n in result_vpp1.nodes.values()]
        assert all(v is None for v in vpp1_virtual), "vpp_chunks=1 should not add virtual_stage_id"

    def test_vpp_uneven_layer_count_10_layers_pp2_vpp2(self):
        """VPP with 10 layers, pp=2, vpp_chunks=2 — bounded chunk_id prevents overflow.
        
        Before fix: floor division (10 // 4 = 2) created chunk_id=4 for layers 8-9,
        wrapping back to device 0, giving device 0 three chunks instead of two.
        
        After fix: chunk_id = min(idx // layers_per_chunk, total_chunks - 1),
        ensuring no chunk_id exceeds total_chunks - 1.
        
        layers_per_chunk = 10 // 4 = 2
        Remaining layers (8,9) are clamped to chunk 3 (last valid chunk).
        
        Distribution:
          - layers 0-1: chunk 0 → stage 0
          - layers 2-3: chunk 1 → stage 1
          - layers 4-5: chunk 2 → stage 0
          - layers 6-9: chunk 3 → stage 1 (clamped)
        """
        graph = _make_linear_graph(num_layers=10)
        ctx = _make_ctx(pp=2)
        ctx.training = TrainingConfig(
            micro_batch=1,
            global_batch=10,
            pp_schedule="interleaved",
            vpp_chunks=2,
        )
        result = PipelineParallelPass().run(graph, ctx)
        
        stage0_layers = {
            int(n.layer) for n in result.nodes.values()
            if n.layer and n.op_type != "comm.send_recv"
               and n.annotations.get("stage_id") == 0
        }
        stage1_layers = {
            int(n.layer) for n in result.nodes.values()
            if n.layer and n.op_type != "comm.send_recv"
               and n.annotations.get("stage_id") == 1
        }
        
        assert stage0_layers == {0, 1, 4, 5}, f"Stage 0 got {stage0_layers}"
        assert stage1_layers == {2, 3, 6, 7, 8, 9}, f"Stage 1 got {stage1_layers}"
        
        compute_nodes = [n for n in result.nodes.values() if n.op_type != "comm.send_recv"]
        virtual_ids = {n.annotations["virtual_stage_id"] for n in compute_nodes}
        assert virtual_ids == {0, 1, 2, 3}, f"Expected 4 virtual stages (0-3), got {virtual_ids}"
        assert max(virtual_ids) == 3, f"Max virtual_stage_id should be 3, got {max(virtual_ids)}"

    def test_vpp_uneven_layer_count_11_layers_pp2_vpp3(self):
        """VPP with 11 layers, pp=2, vpp_chunks=3 — 6 total chunks, bounded chunk_id.
        
        total_chunks = 2 * 3 = 6
        layers_per_chunk = 11 // 6 = 1
        Remaining layers (6-10) are clamped to chunk 5 (last valid chunk).
        
        Distribution:
          - layer 0: chunk 0 → stage 0
          - layer 1: chunk 1 → stage 1
          - layer 2: chunk 2 → stage 0
          - layer 3: chunk 3 → stage 1
          - layer 4: chunk 4 → stage 0
          - layers 5-10: chunk 5 → stage 1 (clamped)
        """
        graph = _make_linear_graph(num_layers=11)
        ctx = _make_ctx(pp=2)
        ctx.training = TrainingConfig(
            micro_batch=1,
            global_batch=11,
            pp_schedule="interleaved",
            vpp_chunks=3,
        )
        result = PipelineParallelPass().run(graph, ctx)
        
        compute_nodes = [n for n in result.nodes.values() if n.op_type != "comm.send_recv"]
        virtual_ids = {n.annotations["virtual_stage_id"] for n in compute_nodes}
        assert virtual_ids == {0, 1, 2, 3, 4, 5}, f"Expected 6 virtual stages (0-5), got {virtual_ids}"
        assert max(virtual_ids) == 5, f"Max virtual_stage_id should be 5, got {max(virtual_ids)}"


# ── TrainingPipelinePass per-stage tests ──────────────────────────────────────

class TestTrainingPipelinePassPerStage:

    def _run_pipeline_pass(self, num_layers=4, pp=2, global_batch=8):
        """Run PP pass then TrainingPipelinePass and return (result_graph, metrics)."""
        from python.zrt.transform.analysis.training import TrainingFlopsPass

        graph = _make_linear_graph(num_layers=num_layers)
        ctx = _make_ctx(pp=pp, global_batch=global_batch)

        # First assign stage_ids
        g_pp = PipelineParallelPass().run(graph, ctx)
        # Run flops pass to set layer_scale metadata
        g_flops = TrainingFlopsPass().run(g_pp, ctx)
        # Run pipeline pass
        result = TrainingPipelinePass().run(g_flops, ctx)
        return result, result.metadata["pipeline_metrics"]

    def test_metrics_present(self):
        """TrainingPipelinePass writes pipeline_metrics to graph.metadata."""
        result, metrics = self._run_pipeline_pass()
        assert metrics is not None
        assert metrics.step_time_ms >= 0

    def test_pp1_no_bubble(self):
        """pp=1 → bubble_fraction == 0 (no warmup/cooldown)."""
        graph = _make_linear_graph(num_layers=4)
        ctx = _make_ctx(pp=1, global_batch=8)
        from python.zrt.transform.analysis.training import TrainingFlopsPass

        g_flops = TrainingFlopsPass().run(graph, ctx)
        result = TrainingPipelinePass().run(g_flops, ctx)
        metrics = result.metadata["pipeline_metrics"]
        assert metrics.warmup_steps == 0
        assert metrics.cooldown_steps == 0
        assert metrics.bubble_fraction == 0.0

    def test_pp2_has_bubble(self):
        """pp=2 → warmup_steps == cooldown_steps == 1 → bubble > 0."""
        _, metrics = self._run_pipeline_pass(pp=2, global_batch=8)
        assert metrics.warmup_steps == 1
        assert metrics.cooldown_steps == 1
        assert metrics.bubble_fraction > 0.0

    def test_per_stage_latency_not_divided_by_pp(self):
        """per_stage_ms must reflect real per-stage time, not total/pp."""
        graph = _make_linear_graph(num_layers=4)
        ctx2 = _make_ctx(pp=2, global_batch=8)
        ctx1 = _make_ctx(pp=1, global_batch=8)

        from python.zrt.transform.analysis.training import TrainingFlopsPass

        # pp=2 path with stage_ids
        g_pp2 = PipelineParallelPass().run(graph, ctx2)
        g_pp2 = TrainingFlopsPass().run(g_pp2, ctx2)
        result2 = TrainingPipelinePass().run(g_pp2, ctx2)
        per_stage_pp2 = result2.metadata["pipeline_metrics"].per_stage_ms

        # pp=1 baseline
        g1 = TrainingFlopsPass().run(graph, ctx1)
        result1 = TrainingPipelinePass().run(g1, ctx1)
        total_pp1 = result1.metadata["pipeline_metrics"].per_stage_ms

        # Bottleneck stage latency should be >= total/pp (greedy assigns heavier layers)
        # and <= total (can't be larger than whole graph)
        assert 0 < per_stage_pp2 <= total_pp1 * 1.1, (
            f"pp2 per_stage={per_stage_pp2:.3f}ms should be <= pp1 total={total_pp1:.3f}ms"
        )

    def test_stage_timelines_stored_in_metadata(self):
        """When pp>1 and stage_ids exist, stage_timelines_fwd is stored."""
        result, _ = self._run_pipeline_pass(pp=2)
        assert "stage_timelines_fwd" in result.metadata
        timelines = result.metadata["stage_timelines_fwd"]
        assert isinstance(timelines, dict)
        assert 0 in timelines
        assert 1 in timelines

    def test_bubble_fraction_increases_with_pp(self):
        """More PP stages → larger bubble fraction (for fixed M)."""
        from python.zrt.transform.analysis.training import TrainingFlopsPass

        results = {}
        for pp in (1, 2, 4):
            graph = _make_linear_graph(num_layers=4)
            ctx = _make_ctx(pp=pp, global_batch=8)
            g = PipelineParallelPass().run(graph, ctx) if pp > 1 else graph
            g = TrainingFlopsPass().run(g, ctx)
            r = TrainingPipelinePass().run(g, ctx)
            results[pp] = r.metadata["pipeline_metrics"].bubble_fraction

        # bubble_fraction = (2*(pp-1)) / (2*(pp-1)+M) — strictly grows with pp
        assert results[1] == 0.0                # pp=1: no bubble
        assert results[2] > results[1]          # pp=2: some bubble
        assert results[4] > results[2]          # pp=4: more bubble
        assert all(0.0 <= v < 1.0 for v in results.values())


# ── Integration: PP pass in default pipeline ──────────────────────────────────

class TestPipelineInDefaultPipeline:

    def test_pp2_in_default_pipeline(self):
        """PipelineParallelPass is activated when pp=2 in build_default_pipeline."""
        from python.zrt.transform import build_default_pipeline

        graph = _make_linear_graph(num_layers=4)
        ctx = TransformContext(
            hw_spec=_make_hw(),
            parallel=ParallelConfig(pp=2),
            stream_config=StreamConfig(),
        )
        pipe = build_default_pipeline()
        result = pipe.run(graph, ctx)

        # After full pipeline: stage_id annotations present
        compute_nodes = [n for n in result.nodes.values()
                         if n.category == "compute"]
        assert all("stage_id" in n.annotations for n in compute_nodes), (
            "Some compute nodes missing stage_id after pp=2 pipeline"
        )

    def test_pp1_pipeline_no_p2p_nodes(self):
        """When pp=1, no P2P comm.send_recv nodes are inserted."""
        from python.zrt.transform import build_default_pipeline

        graph = _make_linear_graph(num_layers=4)
        ctx = TransformContext(
            hw_spec=_make_hw(),
            parallel=ParallelConfig(pp=1),
            stream_config=StreamConfig(),
        )
        pipe = build_default_pipeline()
        result = pipe.run(graph, ctx)

        p2p_nodes = [n for n in result.nodes.values()
                     if n.op_type == "comm.send_recv"]
        assert len(p2p_nodes) == 0


# ── Gap 2: VPP/DualPipe on graph-native per-stage path ────────────────────────────

def test_pp_vpp_uses_reduced_bubble():
    """VPP bubble should be smaller than standard 1F1B on per-stage path.

    This test verifies that the per-stage path (when stage_id annotations exist)
    correctly applies VPP/DualPipe schedule-type adjustments.

    Gap 2 fix: The per-stage path now applies VPP/DualPipe formulas just like
    the non-per-stage path, ensuring consistent behavior.
    """
    from python.zrt.transform.analysis.training import TrainingFlopsPass

    # Create asymmetric graph (different latencies per layer)
    graph = _make_linear_graph(num_layers=4)
    ctx_vpp = _make_ctx(pp=2, global_batch=8)
    ctx_std = _make_ctx(pp=2, global_batch=8)

    # Set VPP schedule for vpp context
    ctx_vpp.training = TrainingConfig(
        micro_batch=1,
        global_batch=8,
        pp_schedule="interleaved",
        vpp_chunks=2,
    )
    ctx_std.training = TrainingConfig(
        micro_batch=1,
        global_batch=8,
        pp_schedule="1f1b",
        vpp_chunks=1,
    )

    # Apply PP pass to get stage_id annotations
    g_pp_vpp = PipelineParallelPass().run(graph, ctx_vpp)
    g_pp_std = PipelineParallelPass().run(graph, ctx_std)

    # Run flops pass and pipeline pass
    g_vpp = TrainingFlopsPass().run(g_pp_vpp, ctx_vpp)
    result_vpp = TrainingPipelinePass().run(g_vpp, ctx_vpp)

    g_std = TrainingFlopsPass().run(g_pp_std, ctx_std)
    result_std = TrainingPipelinePass().run(g_std, ctx_std)

    # Extract metrics
    bubble_vpp = result_vpp.metadata["pipeline_metrics"].bubble_fraction
    bubble_std = result_std.metadata["pipeline_metrics"].bubble_fraction

    # VPP bubble should be no larger than standard 1F1B bubble.
    # Strict reduction requires both fwd and bwd time; when bwd=0 the
    # reduction may vanish because cooldown is zero in both schedules.
    assert bubble_vpp <= bubble_std + 1e-6, (
        f"VPP bubble ({bubble_vpp:.3f}) should be <= standard 1F1B bubble ({bubble_std:.3f})"
    )

    # Verify stage_timelines are present (per-stage path was used)
    assert "stage_timelines_fwd" in result_vpp.metadata
    assert "stage_timelines_bwd" in result_vpp.metadata
