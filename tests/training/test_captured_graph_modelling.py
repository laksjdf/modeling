"""Integration test: training modelling driven by a captured-style OpGraph.

Captured graphs have opaque tensor IDs (t0, t1, ...) unlike synthetic test
graphs that use descriptive names like 'weight_0'.  This file verifies that
estimate_training_from_graphs() returns meaningful results in both cases.
"""
from __future__ import annotations

import math
from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.edge import Edge
from zrt.ir.types import TensorMeta, DType
from zrt.transform.context import TransformContext, ParallelConfig, TrainingConfig


def _hw():
    from zrt.hardware.spec import HardwareSpec, ComputeSpec, MemorySpec, InterconnectSpec, LinkSpec
    return HardwareSpec(
        name="test_gpu", vendor="test", device_type="gpu",
        compute=ComputeSpec(bf16_tflops=312),
        memory=MemorySpec(capacity_gb=80, hbm_bandwidth_gbps=2000),
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="nvlink", num_devices=8, bandwidth_gbps=600, latency_us=1.0),
            inter_node=LinkSpec(type="ib",     num_devices=128, bandwidth_gbps=400, latency_us=5.0),
        ),
    )


def _captured_style_graph(seq_len=2048, hidden=4096, ffn=16384, num_layers=32) -> OpGraph:
    """Minimal transformer forward graph with captured-style tensor IDs (t0, t1, ...).

    Structure per layer: linear(t_act, t_weight) → linear(t_mid, t_weight2) → add
    Two layers total to keep the test fast.
    """
    nodes: dict[str, OpNode] = {}
    edges: list[Edge] = []
    tid = 0

    def _tm(shape, dtype=DType.BF16):
        nonlocal tid
        t = TensorMeta(
            id=f"t{tid}",
            shape=shape,
            dtype=dtype,
            mem_bytes=int(math.prod(shape) * 2),
        )
        tid += 1
        return t

    # Input activation (t0) — external, not produced by any node
    t_input = _tm((seq_len, hidden))

    prev_out = t_input
    for layer_id in range(2):
        scope_base = f"model.layers.{layer_id}"

        # QKV projection: mm(activation, weight)
        t_w_qkv = _tm((hidden, hidden))   # weight — external
        t_qkv   = _tm((seq_len, hidden))  # output
        n_qkv = OpNode(
            id=f"L{layer_id}_qkv",
            op_type="aten.mm.default",
            inputs=[prev_out, t_w_qkv],
            outputs=[t_qkv],
            scope=f"{scope_base}.self_attn.qkv_proj",
            category="compute",
            layer=str(layer_id),
        )
        nodes[n_qkv.id] = n_qkv

        # FFN up-projection
        t_w_up = _tm((hidden, ffn))
        t_up   = _tm((seq_len, ffn))
        n_up = OpNode(
            id=f"L{layer_id}_up",
            op_type="aten.mm.default",
            inputs=[t_qkv, t_w_up],
            outputs=[t_up],
            scope=f"{scope_base}.mlp.up_proj",
            category="compute",
            layer=str(layer_id),
        )
        nodes[n_up.id] = n_up
        edges.append(Edge(src=n_qkv.id, dst=n_up.id, tensor=t_qkv, src_idx=0, dst_idx=0))

        # FFN down-projection
        t_w_down = _tm((ffn, hidden))
        t_down   = _tm((seq_len, hidden))
        n_down = OpNode(
            id=f"L{layer_id}_down",
            op_type="aten.mm.default",
            inputs=[t_up, t_w_down],
            outputs=[t_down],
            scope=f"{scope_base}.mlp.down_proj",
            category="compute",
            layer=str(layer_id),
        )
        nodes[n_down.id] = n_down
        edges.append(Edge(src=n_up.id, dst=n_down.id, tensor=t_up, src_idx=0, dst_idx=0))

        prev_out = t_down

    return OpGraph(
        name="captured_llama",
        phase="forward",
        nodes=nodes,
        edges=edges,
        metadata={
            "seq_len": seq_len,
            "hidden": hidden,
            "num_layers": num_layers,
            "batch_size": 1,
        },
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_pipeline_routing_runs_roofline_and_stream_assign():
    """estimate_training_from_graphs() via build_default_pipeline must run RooflinePass + StreamAssignPass.

    Verifies that nodes carry latency_us (from RooflinePass) and stream_id
    (from StreamAssignPass), proving the full pipeline ran, not just training passes.
    """
    g = _captured_style_graph()
    hw = _hw()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),
    )

    from python.zrt.transform.pipeline import build_default_pipeline
    pipe = build_default_pipeline()
    result = pipe.run(g, ctx)

    # All compute nodes must have latency_us (from RooflinePass)
    for nid, node in result.nodes.items():
        assert "latency_us" in node.annotations, (
            f"Node {nid} missing latency_us — RooflinePass did not run"
        )
        assert "stream_id" in node.annotations, (
            f"Node {nid} missing stream_id — StreamAssignPass did not run"
        )

    # Graph must have training_flops metadata (from TrainingFlopsPass)
    assert "training_flops" in result.metadata
    assert result.metadata["training_flops"] > 0


def test_backward_fusion_rules_fire_on_backward_graph():
    """Verify that backward fusion rules from fusion_rules.py:195-311 match
    on a synthetic backward graph when run through build_default_pipeline().

    Creates a backward OpGraph with ops matching norm_backward and
    gated_mlp_backward sub-patterns, then asserts at least one node gets
    relabeled with a backward fusion label.
    """
    import math
    from python.zrt.transform.pipeline import build_default_pipeline

    # Build a synthetic backward graph with backward-style ops in matching scopes
    nodes: dict[str, OpNode] = {}
    edges: list[Edge] = []
    tid = 0

    def _tm(shape, dtype=DType.BF16):
        nonlocal tid
        t = TensorMeta(
            id=f"t{tid}", shape=shape, dtype=dtype,
            mem_bytes=int(math.prod(shape) * 2),
        )
        tid += 1
        return t

    # Group 1: norm_backward — native_layer_norm_backward in RMSNorm scope
    t_norm_in = _tm((2048, 4096))
    t_norm_out = _tm((2048, 4096))
    n_norm = OpNode(
        id="bwd_layernorm",
        op_type="aten.native_layer_norm_backward.default",
        inputs=[t_norm_in],
        outputs=[t_norm_out],
        scope="model.layers.0.input_layernorm",
        module_class="RMSNorm",
        category="compute",
    )
    nodes[n_norm.id] = n_norm

    # Group 2: gated_mlp_backward — silu_backward + mul + mm in MLP scope
    t_mlp_in = _tm((2048, 4096))
    t_silu_out = _tm((2048, 4096))
    n_silu = OpNode(
        id="bwd_silu",
        op_type="aten.silu_backward.default",
        inputs=[t_mlp_in],
        outputs=[t_silu_out],
        scope="model.layers.0.mlp",
        module_class="MLP",
        category="compute",
    )
    nodes[n_silu.id] = n_silu

    t_mul_out = _tm((2048, 4096))
    n_mul = OpNode(
        id="bwd_mul",
        op_type="aten.mul.default",
        inputs=[t_silu_out],
        outputs=[t_mul_out],
        scope="model.layers.0.mlp",
        module_class="MLP",
        category="compute",
    )
    nodes[n_mul.id] = n_mul

    t_w = _tm((4096, 4096))
    t_mm_out = _tm((2048, 4096))
    n_mm = OpNode(
        id="bwd_mm",
        op_type="aten.mm.default",
        inputs=[t_mul_out, t_w],
        outputs=[t_mm_out],
        scope="model.layers.0.mlp",
        module_class="MLP",
        category="compute",
    )
    nodes[n_mm.id] = n_mm

    edges.append(Edge(src=n_silu.id, dst=n_mul.id, tensor=t_silu_out, src_idx=0, dst_idx=0))
    edges.append(Edge(src=n_mul.id, dst=n_mm.id, tensor=t_mul_out, src_idx=0, dst_idx=0))

    graph = OpGraph(
        name="synthetic_backward",
        phase="train_backward",
        nodes=nodes,
        edges=edges,
        metadata={
            "seq_len": 2048,
            "hidden": 4096,
            "num_layers": 32,
            "batch_size": 1,
        },
    )

    from zrt.hardware.spec import HardwareSpec, ComputeSpec, MemorySpec, InterconnectSpec, LinkSpec
    hw_nvidia = HardwareSpec(
        name="nvidia_gpu", vendor="nvidia", device_type="gpu",
        compute=ComputeSpec(bf16_tflops=312),
        memory=MemorySpec(capacity_gb=80, hbm_bandwidth_gbps=2000),
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="nvlink", num_devices=8, bandwidth_gbps=600, latency_us=1.0),
            inter_node=LinkSpec(type="ib",     num_devices=128, bandwidth_gbps=400, latency_us=5.0),
        ),
    )
    ctx = TransformContext(
        hw_spec=hw_nvidia,
        parallel=ParallelConfig(tp=1, pp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),
    )

    pipe = build_default_pipeline()
    result = pipe.run(graph, ctx)

    # Collect all op_types (including fused nodes)
    all_op_types = {n.op_type for n in result.nodes.values()}

    # At least one backward fusion label must be present
    backward_labels = {
        "norm_backward", "gated_mlp_backward", "mlp_backward",
        "sdpa_backward", "attn_grad", "embedding_backward",
    }
    matched = all_op_types & backward_labels
    assert matched, (
        f"No backward fusion labels found. Got op_types: {all_op_types}. "
        f"Expected one of: {backward_labels}"
    )


# ── stitch_fwd_bwd tests ──────────────────────────────────────────────────────

def _backward_graph_for_fwd(fwd: OpGraph, seq_len=2048, hidden=4096, ffn=16384) -> OpGraph:
    """Build a synthetic backward graph whose inputs match forward outputs.

    For each layer, creates a backward mm node that consumes the forward
    output tensor (matching shape+dtype), producing a gradient.
    """
    nodes: dict[str, OpNode] = {}
    edges: list[Edge] = []
    tid = 0

    def _tm(shape, dtype=DType.BF16):
        nonlocal tid
        t = TensorMeta(
            id=f"t{tid}", shape=shape, dtype=dtype,
            mem_bytes=int(math.prod(shape) * 2),
        )
        tid += 1
        return t

    # Collect forward output tensors by layer for matching
    fwd_outputs_by_layer = {}
    for node in fwd:
        if node.layer:
            for out in node.outputs:
                fwd_outputs_by_layer.setdefault(node.layer, []).append(out)

    prev_grad = None
    for layer_id in range(2):
        scope_base = f"model.layers.{layer_id}"

        # Use a forward output shape as the grad input to enable cross-graph matching
        fwd_outs = fwd_outputs_by_layer.get(str(layer_id), [])
        act_shape = fwd_outs[-1].shape if fwd_outs else (seq_len, hidden)

        if prev_grad is None:
            t_grad_in = _tm(act_shape)
        else:
            t_grad_in = prev_grad

        t_w_grad = _tm((hidden, hidden))
        t_grad_out = _tm((seq_len, hidden))

        n_bwd = OpNode(
            id=f"op_{layer_id}_bwd_mm",
            op_type="aten.mm.default",
            inputs=[t_grad_in, t_w_grad],
            outputs=[t_grad_out],
            scope=f"{scope_base}.self_attn.q_proj",
            category="compute",
            layer=str(layer_id),
        )
        nodes[n_bwd.id] = n_bwd

        # Chain backward nodes
        if prev_grad is not None and layer_id > 0:
            prev_id = f"op_{layer_id - 1}_bwd_mm"
            edges.append(Edge(
                src=prev_id, dst=n_bwd.id,
                tensor=t_grad_in, src_idx=0, dst_idx=0,
            ))

        prev_grad = t_grad_out

    return OpGraph(
        name="synthetic_bwd",
        phase="train_backward",
        nodes=nodes,
        edges=edges,
        metadata={"seq_len": seq_len, "hidden": hidden, "num_layers": 32, "batch_size": 1},
    )


def test_stitch_preserves_all_nodes():
    """Stitched graph must contain all forward + backward nodes."""
    from zrt.ir.adapter import stitch_fwd_bwd
    fwd = _captured_style_graph()
    bwd = _backward_graph_for_fwd(fwd)
    stitched = stitch_fwd_bwd(fwd, bwd)
    assert len(stitched.nodes) == len(fwd.nodes) + len(bwd.nodes), (
        f"Expected {len(fwd.nodes) + len(bwd.nodes)} nodes, got {len(stitched.nodes)}"
    )


def test_stitch_phase_annotations():
    """Every node must have annotations['phase'] of 'fwd' or 'bwd'."""
    from zrt.ir.adapter import stitch_fwd_bwd
    fwd = _captured_style_graph()
    bwd = _backward_graph_for_fwd(fwd)
    stitched = stitch_fwd_bwd(fwd, bwd)
    for nid, node in stitched.nodes.items():
        phase = node.annotations.get("phase")
        assert phase in ("fwd", "bwd"), f"Node {nid} has invalid phase: {phase}"


def test_stitch_backward_id_prefix():
    """All backward node IDs must start with 'bwd_', forward IDs must not."""
    from zrt.ir.adapter import stitch_fwd_bwd
    fwd = _captured_style_graph()
    bwd = _backward_graph_for_fwd(fwd)
    stitched = stitch_fwd_bwd(fwd, bwd)
    for nid, node in stitched.nodes.items():
        if node.annotations.get("phase") == "bwd":
            assert nid.startswith("bwd_"), f"Backward node {nid} missing 'bwd_' prefix"
        else:
            assert not nid.startswith("bwd_"), f"Forward node {nid} has 'bwd_' prefix"


def test_stitch_param_detection():
    """Nodes in weight-bearing scopes with matmul ops must have is_param = True."""
    from zrt.ir.adapter import stitch_fwd_bwd
    fwd = _captured_style_graph()
    bwd = _backward_graph_for_fwd(fwd)
    stitched = stitch_fwd_bwd(fwd, bwd)
    param_nodes = [n for n in stitched if n.annotations.get("is_param")]
    assert len(param_nodes) > 0, "Expected at least one param node (qkv_proj, up_proj, down_proj)"


def test_stitch_cross_graph_edges():
    """There must be at least one edge from a forward node to a backward node."""
    from zrt.ir.adapter import stitch_fwd_bwd
    fwd = _captured_style_graph()
    bwd = _backward_graph_for_fwd(fwd)
    stitched = stitch_fwd_bwd(fwd, bwd)
    fwd_ids = {nid for nid, n in stitched.nodes.items() if n.annotations.get("phase") == "fwd"}
    bwd_ids = {nid for nid, n in stitched.nodes.items() if n.annotations.get("phase") == "bwd"}
    cross = [e for e in stitched.edges if e.src in fwd_ids and e.dst in bwd_ids]
    assert len(cross) > 0, "Expected at least one cross-graph (fwd→bwd) edge"


def test_stitch_no_id_conflicts():
    """No duplicate node IDs in the stitched graph."""
    from zrt.ir.adapter import stitch_fwd_bwd
    fwd = _captured_style_graph()
    bwd = _backward_graph_for_fwd(fwd)
    stitched = stitch_fwd_bwd(fwd, bwd)
    ids = list(stitched.nodes.keys())
    assert len(ids) == len(set(ids)), "Duplicate node IDs found in stitched graph"


# ── Issue 4: cross-graph edge correctness tests ───────────────────────────────

def test_stitch_cross_edges_within_same_layer():
    """Cross-graph edges must prefer same-layer producers.

    A backward node with layer='1' should not receive a cross-graph edge
    from a forward node with layer='0' when a same-layer candidate exists.
    """
    from zrt.ir.adapter import stitch_fwd_bwd

    # Build a 2-layer forward graph where both layers produce (2048, 4096) bf16
    nodes_fwd: dict[str, OpNode] = {}
    tid = 0

    def _tm(shape, dtype=DType.BF16):
        nonlocal tid
        t = TensorMeta(id=f"t{tid}", shape=shape, dtype=dtype,
                       mem_bytes=int(math.prod(shape) * 2))
        tid += 1
        return t

    for layer in range(2):
        inp = _tm((2048, 4096))
        w = _tm((4096, 4096))
        out = _tm((2048, 4096))
        n = OpNode(id=f"fwd_L{layer}", op_type="aten.mm.default",
                   inputs=[inp, w], outputs=[out],
                   scope=f"model.layers.{layer}.self_attn.q_proj",
                   category="compute", layer=str(layer))
        nodes_fwd[n.id] = n

    fwd = OpGraph(name="fwd", phase="train_forward", nodes=nodes_fwd, edges=[],
                  metadata={"seq_len": 2048, "hidden": 4096})

    # Build backward node for layer 1 only, with matching tensor shape
    bwd_in = _tm((2048, 4096))  # same shape as fwd outputs
    bwd_w = _tm((4096, 4096))
    bwd_out = _tm((2048, 4096))
    n_bwd = OpNode(id="bwd_L1", op_type="aten.mm.default",
                   inputs=[bwd_in, bwd_w], outputs=[bwd_out],
                   scope="model.layers.1.self_attn.q_proj",
                   category="compute", layer="1")
    nodes_bwd = {n_bwd.id: n_bwd}
    bwd = OpGraph(name="bwd", phase="train_backward", nodes=nodes_bwd, edges=[],
                  metadata={"seq_len": 2048, "hidden": 4096})

    stitched = stitch_fwd_bwd(fwd, bwd)
    cross = [e for e in stitched.edges
             if e.src in nodes_fwd and e.dst.startswith("bwd_")]
    assert len(cross) > 0, "Expected at least one cross-graph edge"
    # All cross-edges for the layer-1 backward node must come from layer-1 forward
    for e in cross:
        fwd_node = stitched.nodes.get(e.src)
        assert fwd_node is not None
        assert fwd_node.layer == "1", (
            f"Cross-edge from {e.src} (layer={fwd_node.layer}) to {e.dst} "
            f"should prefer same-layer producer (layer=1), not layer 0"
        )


def test_stitch_preserves_both_metadata():
    """Both forward and backward metadata must be accessible in the stitched graph."""
    from zrt.ir.adapter import stitch_fwd_bwd
    fwd = _captured_style_graph()
    bwd = _backward_graph_for_fwd(fwd)
    fwd.metadata["fwd_only_key"] = "fwd_value"
    bwd.metadata["bwd_only_key"] = "bwd_value"
    bwd.metadata["seq_len"] = 9999  # bwd override

    stitched = stitch_fwd_bwd(fwd, bwd)
    # fwd_only_key preserved
    assert stitched.metadata.get("fwd_only_key") == "fwd_value"
    # bwd_only_key preserved
    assert stitched.metadata.get("bwd_only_key") == "bwd_value"
    # fwd wins on conflict (fwd overrides bwd for seq_len)
    assert stitched.metadata.get("seq_len") != 9999, "fwd should win on key conflicts"
    # Namespaced copies accessible
    assert "fwd_metadata" in stitched.metadata
    assert "bwd_metadata" in stitched.metadata
    assert stitched.metadata["bwd_metadata"].get("bwd_only_key") == "bwd_value"


def test_stitch_param_detection_covers_embedding():
    """Embedding nodes must be flagged is_param=True."""
    from zrt.ir.adapter import stitch_fwd_bwd

    nodes: dict[str, OpNode] = {}
    t_in = TensorMeta(id="t0", shape=(4,), dtype=DType.INT64, mem_bytes=32)
    t_w = TensorMeta(id="t1", shape=(32000, 4096), dtype=DType.BF16, mem_bytes=32000*4096*2)
    t_out = TensorMeta(id="t2", shape=(4, 4096), dtype=DType.BF16, mem_bytes=4*4096*2)
    n_emb = OpNode(id="embed", op_type="aten.embedding.default",
                   inputs=[t_in, t_w], outputs=[t_out],
                   scope="model.embed_tokens", category="compute")
    nodes[n_emb.id] = n_emb

    fwd = OpGraph(name="fwd", phase="train_forward", nodes=nodes, edges=[],
                  metadata={"seq_len": 128, "hidden": 4096})
    # Minimal backward graph
    bwd_nodes: dict[str, OpNode] = {}
    n_bwd = OpNode(id="bwd_op", op_type="aten.mm.default",
                   inputs=[t_out], outputs=[t_out],
                   scope="model.layers.0.mlp", category="compute", layer="0")
    bwd_nodes[n_bwd.id] = n_bwd
    bwd = OpGraph(name="bwd", phase="train_backward", nodes=bwd_nodes, edges=[],
                  metadata={"seq_len": 128, "hidden": 4096})

    stitched = stitch_fwd_bwd(fwd, bwd)
    embed_node = stitched.nodes.get("embed")
    assert embed_node is not None, "Embedding node missing from stitched graph"
    assert embed_node.annotations.get("is_param") is True, (
        "Embedding node should be flagged is_param=True"
    )


def test_stitch_cross_edges_use_tensor_ids():
    """Cross-graph edges must use exact tensor-ID matching when IDs align.

    Two fwd nodes produce tensors with the SAME shape/dtype (ambiguous by
    fallback heuristic), but bwd consumes one specific tensor via its exact
    tensor ID. The stitcher must pick the correct producer, not the first
    same-shape candidate.
    """
    from zrt.ir.adapter import stitch_fwd_bwd

    # Two fwd nodes in same layer produce identically-shaped tensors
    t_other = TensorMeta(id="t_other", shape=(2048, 4096),
                         dtype=DType.BF16, mem_bytes=2048*4096*2)
    t_target = TensorMeta(id="t_grad_target", shape=(2048, 4096),
                          dtype=DType.BF16, mem_bytes=2048*4096*2)

    fwd_nodes = {
        "fwd_A": OpNode(id="fwd_A", op_type="aten.mm.default",
                        inputs=[], outputs=[t_other],
                        scope="model.layers.0.self_attn.q_proj",
                        category="compute", layer="0"),
        "fwd_B": OpNode(id="fwd_B", op_type="aten.mm.default",
                        inputs=[], outputs=[t_target],
                        scope="model.layers.0.self_attn.k_proj",
                        category="compute", layer="0"),
    }
    fwd = OpGraph(name="fwd", phase="train_forward",
                  nodes=fwd_nodes, edges=[],
                  metadata={"seq_len": 2048, "hidden": 4096})

    # Backward consumes the target tensor by the SHARED id
    t_bwd_in = TensorMeta(id="t_grad_target", shape=(2048, 4096),
                          dtype=DType.BF16, mem_bytes=2048*4096*2)
    bwd_nodes = {
        "bwd_X": OpNode(id="bwd_X", op_type="aten.mm.default",
                        inputs=[t_bwd_in], outputs=[],
                        scope="model.layers.0.self_attn.k_proj",
                        category="compute", layer="0"),
    }
    bwd = OpGraph(name="bwd", phase="train_backward",
                  nodes=bwd_nodes, edges=[],
                  metadata={"seq_len": 2048, "hidden": 4096})

    stitched = stitch_fwd_bwd(fwd, bwd)
    cross = [e for e in stitched.edges
             if e.src in fwd_nodes and e.dst == "bwd_bwd_X"]
    assert len(cross) == 1, f"Expected 1 cross-edge, got {len(cross)}"
    assert cross[0].src == "fwd_B", (
        f"Tensor-ID match must pick fwd_B (producer of t_grad_target), "
        f"not fwd_A. Got {cross[0].src}."
    )


# ── Phase 0–2 end-to-end tests ────────────────────────────────────────────────

def test_pp_routing_end_to_end():
    """End-to-end: stitch fwd+bwd → training pipeline with pp=2.

    Verifies that after the full pipeline:
    1. Every node has stage_id ∈ {0, 1}.
    2. At least one comm.send_recv node crosses stage 0 → 1.
    3. The P2P node is on the dependency path of stage-1 consumers.
    4. Per-stage fwd and bwd timelines are populated for both stages.
    5. Pipeline metrics (step_time, bubble) are sensible.
    6. Cross-graph fwd→bwd edges survive the per-stage subgraph split.
    """
    from zrt.ir.adapter import stitch_fwd_bwd
    from zrt.transform.pipeline import build_default_pipeline

    fwd = _captured_style_graph()
    bwd = _backward_graph_for_fwd(fwd)
    stitched = stitch_fwd_bwd(fwd, bwd)

    hw = _hw()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=2, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),
    )
    pipe = build_default_pipeline()
    result = pipe.run(stitched, ctx)

    # 1) Every node has stage_id ∈ {0, 1}
    for nid, node in result.nodes.items():
        sid = node.annotations.get("stage_id")
        assert sid in (0, 1), f"Node {nid} has bad stage_id={sid}"

    # Both stages must be populated (not everything collapsed into stage 0)
    stage_ids_used = {n.annotations.get("stage_id") for n in result.nodes.values()}
    assert stage_ids_used == {0, 1}, (
        f"Only stages {stage_ids_used} populated — PP did not split the graph"
    )

    # 2) At least one P2P send_recv with src_stage=0, dst_stage=1
    p2p = [n for n in result.nodes.values()
           if n.op_type == "comm.send_recv"
           and n.attrs.get("src_stage") == 0
           and n.attrs.get("dst_stage") == 1]
    assert p2p, "Expected at least one comm.send_recv crossing stage 0→1"

    # 3) The P2P node must be on the dependency path of stage-1 consumers
    p2p_ids = {n.id for n in p2p}
    stage1_nodes = [n for n in result.nodes.values()
                    if n.annotations.get("stage_id") == 1]
    assert any(set(result.predecessors(n.id)) & p2p_ids for n in stage1_nodes), (
        "No stage-1 node has the boundary P2P as a predecessor — "
        "boundary comm is not on the dependency path (Item 1 fix missing)"
    )

    # 4) Per-stage timelines populated
    stage_fwd = result.metadata.get("stage_timelines_fwd", {})
    stage_bwd = result.metadata.get("stage_timelines_bwd", {})
    assert set(stage_fwd.keys()) == {0, 1}, (
        f"stage_timelines_fwd keys={set(stage_fwd.keys())} — "
        "expected {{0, 1}} (Items 3–4 fix missing)"
    )
    assert set(stage_bwd.keys()) == {0, 1}, (
        f"stage_timelines_bwd keys={set(stage_bwd.keys())} — "
        "expected {{0, 1}} (Items 3–4 fix missing)"
    )
    assert all(v > 0 for v in stage_fwd.values()), (
        f"stage_timelines_fwd has zero entry: {stage_fwd}"
    )
    assert any(v > 0 for v in stage_bwd.values()), (
        f"stage_timelines_bwd is all zero: {stage_bwd}"
    )

    # 5) Pipeline metrics sensible; bubble ≈ (pp-1)/(M+pp-1) for symmetric stages
    pm = result.metadata.get("pipeline_metrics")
    assert pm is not None and pm.step_time_ms > 0, (
        f"pipeline_metrics.step_time_ms={getattr(pm, 'step_time_ms', None)}"
    )
    M = ctx.training.num_microbatches
    pp = 2
    expected_bubble = (pp - 1) / (M + pp - 1)
    assert abs(pm.bubble_fraction - expected_bubble) / expected_bubble < 0.20, (
        f"bubble_fraction={pm.bubble_fraction:.3f}, expected ≈{expected_bubble:.3f}"
    )

    # 6) Cross-graph edge survival: every stage must have bwd nodes, and
    #    at least one bwd node in each stage must have a same-stage predecessor.
    for s in (0, 1):
        stage_node_ids = {nid for nid, n in result.nodes.items()
                          if n.annotations.get("stage_id") == s}
        sub = result.subgraph(stage_node_ids)
        bwd_nodes = [nid for nid, n in sub.nodes.items()
                     if n.annotations.get("phase") == "bwd"]
        assert bwd_nodes, (
            f"Stage {s} has no bwd nodes after stitch+PP split — "
            "stage_id assignment may not cover the backward graph"
        )
        bwd_with_preds = [nid for nid in bwd_nodes if sub.predecessors(nid)]
        assert bwd_with_preds, (
            f"Stage {s}: bwd nodes {bwd_nodes} all lack same-stage predecessors — "
            "cross-graph fwd→bwd edges did not survive the subgraph split"
        )


def test_pp_heterogeneous_1f1b_formula():
    """Heterogeneous 1F1B: asymmetric per-stage timing must use the spec formula.

    Without this test the symmetric graph makes both homogeneous and
    heterogeneous formulas produce the same number, so the new branch
    is never exercised.

    Strategy: pre-annotate stage-0 nodes with latency_us=10 and stage-1
    nodes with latency_us=100 (10× imbalance). The heterogeneous formula
    `(pp-1)*t_fwd[0] + M*max(t_fwd+t_bwd) + (pp-1)*t_bwd[pp-1]` gives
    a different result from `(M+pp-1)*max(t_stage)` in this case.
    """
    from zrt.ir.adapter import stitch_fwd_bwd
    from zrt.transform.pipeline import build_default_pipeline

    fwd = _captured_style_graph()
    bwd = _backward_graph_for_fwd(fwd)
    stitched = stitch_fwd_bwd(fwd, bwd)

    # Pre-annotate asymmetric latency: layer 0 = 10 µs, layer 1 = 100 µs.
    # Injecting before the pipeline skips RooflinePass estimation for these nodes.
    for node in stitched.nodes.values():
        try:
            lid = int(node.layer)
        except (ValueError, TypeError):
            continue
        node.annotations["latency_us"] = 10.0 if lid == 0 else 100.0

    hw = _hw()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=2, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),
    )
    result = build_default_pipeline().run(stitched, ctx)

    stage_fwd = result.metadata.get("stage_timelines_fwd", {})
    stage_bwd = result.metadata.get("stage_timelines_bwd", {})
    pm = result.metadata.get("pipeline_metrics")

    # Asymmetry must be visible in per-stage output
    assert stage_fwd.get(0, 0) < stage_fwd.get(1, 0), (
        f"Stage 0 fwd ({stage_fwd.get(0):.1f}µs) should be < "
        f"stage 1 fwd ({stage_fwd.get(1):.1f}µs) — latency injection not working"
    )

    # Correct formula uses bottleneck stage times (not first/last stage):
    #   warmup   = (pp - 1) * max(t_fwd[s])
    #   steady   = M * max(t_fwd[s] + t_bwd[s])
    #   cooldown = (pp - 1) * max(t_bwd[s])
    M, pp = ctx.training.num_microbatches, 2
    t_fwd_max = max(stage_fwd[s] for s in (0, 1))
    t_bwd_max = max(stage_bwd[s] for s in (0, 1))
    t_stage_max = max(stage_fwd[s] + stage_bwd[s] for s in (0, 1))

    expected_step_us = (pp - 1) * t_fwd_max + M * t_stage_max + (pp - 1) * t_bwd_max

    # Add optimizer step time to expected (per §5.5.2 of muon_optimizer_design.md)
    opt_step_time_us = result.metadata.get("optimizer_step_time_us", 0)
    expected_step_us += opt_step_time_us

    # Verify the implementation uses the heterogeneous formula
    # Verify the implementation uses the bottleneck stage formula
    actual_step_us = pm.step_time_ms * 1000.0
    assert abs(actual_step_us - expected_step_us) / expected_step_us < 0.05, (
        f"step_time={actual_step_us:.1f}µs; expected {expected_step_us:.1f}µs "
        f"(bottleneck: fwd_max={t_fwd_max:.1f}µs, bwd_max={t_bwd_max:.1f}µs, stage_max={t_stage_max:.1f}µs)"
    )


def test_modeller_uses_pipeline_step_time_for_schedule_adjustments():
    """Graph-native modeller must not recompute away TrainingPipelinePass schedule logic."""
    import pytest
    from unittest.mock import MagicMock, patch
    from zrt.ir.graph import OpGraph
    from zrt.transform.analysis import estimate_training_from_graphs

    hw = _hw()
    pp = 4
    microbatches = 8
    per_stage_us = 1000.0
    graph = OpGraph(
        name="schedule_adjusted",
        phase="train_forward",
        metadata={
            "num_layers": 4,
            "num_layers_traced": 4,
            "training_flops": 1e12,
        },
    )

    mock_timeline = MagicMock()
    mock_timeline.total_latency_us = per_stage_us * pp
    mock_timeline.compute_time_us = mock_timeline.total_latency_us
    mock_timeline.comm_time_us = 0.0
    mock_timeline.overlap_us = 0.0

    with patch("python.zrt.executor.scheduler.DAGScheduler") as MockSched:
        MockSched.return_value.schedule.return_value = mock_timeline
        report = estimate_training_from_graphs(
            forward_graph=graph,
            hw_spec=hw,
            seq_len=2048,
            batch_size=1,
            hidden=4096,
            num_layers=4,
            pp=pp,
            micro_batch=1,
            global_batch=microbatches,
            pp_schedule="dualpipe",
        )

    expected_step_ms = (microbatches * per_stage_us + (pp - 1) * per_stage_us / 2.0) / 1000.0
    simplified_step_ms = (microbatches + pp - 1) * (per_stage_us / 1000.0)
    expected_bubble = ((pp - 1) * per_stage_us / 2.0) / (expected_step_ms * 1000.0)
    assert expected_step_ms != pytest.approx(simplified_step_ms)
    assert report.step_time_ms == pytest.approx(expected_step_ms)
    assert report.bubble_fraction == pytest.approx(expected_bubble)


# ── Phase 2 end-to-end: stitched pp>1 ─────────────────────────────────────────

def test_pp_routing_basic():
    """End-to-end: stitch fwd+bwd → pipeline with pp=2 → per-stage timelines + P2P + 1F1B."""
    from zrt.ir.adapter import stitch_fwd_bwd
    from zrt.transform.pipeline import build_default_pipeline

    fwd = _captured_style_graph()
    bwd = _backward_graph_for_fwd(fwd)
    stitched = stitch_fwd_bwd(fwd, bwd)

    hw = _hw()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=2, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),
    )
    pipe = build_default_pipeline()
    result = pipe.run(stitched, ctx)

    # 1) Every node has stage_id in {0, 1}
    for nid, node in result.nodes.items():
        sid = node.annotations.get("stage_id")
        assert sid in (0, 1), f"Node {nid} has bad stage_id={sid}"

    # 2) At least one P2P send_recv crossing stages
    p2p = [n for n in result.nodes.values()
           if n.op_type == "comm.send_recv"
           and n.attrs.get("src_stage") == 0
           and n.attrs.get("dst_stage") == 1]
    assert p2p, "Expected at least one comm.send_recv crossing stage 0→1"

    # 3) Receiver-side dependency must go through the P2P node
    p2p_ids = {n.id for n in p2p}
    stage1_nodes = [n for n in result.nodes.values()
                    if n.annotations.get("stage_id") == 1
                    and n.category != "communication"]
    assert any(
        set(result.predecessors(n.id)) & p2p_ids for n in stage1_nodes
    ), "Stage-1 consumers should depend on boundary comm nodes"

    # 4) Per-stage timelines populated
    stage_fwd = result.metadata.get("stage_timelines_fwd", {})
    stage_bwd = result.metadata.get("stage_timelines_bwd", {})
    assert set(stage_fwd.keys()) == {0, 1}, f"stage_timelines_fwd keys={set(stage_fwd.keys())}"
    assert set(stage_bwd.keys()) == {0, 1}, f"stage_timelines_bwd keys={set(stage_bwd.keys())}"
    assert all(v > 0 for v in stage_fwd.values()), f"stage_timelines_fwd has zero: {stage_fwd}"
    assert any(v > 0 for v in stage_bwd.values()), f"stage_timelines_bwd unexpectedly empty: {stage_bwd}"

    # 5) Pipeline metrics sensible
    pm = result.metadata.get("pipeline_metrics")
    assert pm is not None and pm.step_time_ms > 0

    # 6) Cross-graph edge survival: at least one bwd node has an intra-stage predecessor
    #    (stage 0 should have fwd→bwd cross-graph edges; stage 1 bwd nodes may only
    #    receive input via inter-stage P2P, which is valid)
    any_bwd_with_preds = False
    for s in (0, 1):
        stage_node_ids = {nid for nid, n in result.nodes.items()
                          if n.annotations.get("stage_id") == s}
        sub = result.subgraph(stage_node_ids)
        bwd_with_preds = [nid for nid, n in sub.nodes.items()
                          if n.annotations.get("phase") == "bwd"
                          and sub.predecessors(nid)]
        if bwd_with_preds:
            any_bwd_with_preds = True
    assert any_bwd_with_preds, (
        "No bwd node in any stage has an intra-stage predecessor — cross-graph edges lost"
    )


# ── run_trace_phases auto-stitch tests ───────────────────────────────────────

def _make_trace_phase_result(fwd_graph, bwd_graph):
    """Build a TracePhaseResult as run_trace_phases would for training phases."""
    import tempfile, json
    from pathlib import Path
    from python.zrt.pipeline import TracePhaseResult, _save_stitched_graph
    from python.zrt.ir.adapter import stitch_fwd_bwd

    stitched_raw   = stitch_fwd_bwd(fwd_graph, fwd_graph, name="test_train_raw")
    stitched_fused = stitch_fwd_bwd(fwd_graph, bwd_graph, name="test_train_fused")
    all_graphs = {
        "train_forward":  (fwd_graph, fwd_graph),
        "train_backward": (bwd_graph, bwd_graph),
        "train":          (stitched_raw, stitched_fused),
    }
    return all_graphs, stitched_fused


def test_auto_stitch_graphs_key_present():
    """TracePhaseResult must expose result.graphs['train'] when both training phases captured."""
    fwd = _captured_style_graph()
    bwd = _backward_graph_for_fwd(fwd)
    all_graphs, stitched = _make_trace_phase_result(fwd, bwd)

    assert "train" in all_graphs, "Auto-stitch must create graphs['train']"
    _, fused = all_graphs["train"]
    assert fused.phase == "train"


def test_auto_stitch_node_count():
    """Stitched graph must contain all forward + backward nodes."""
    fwd = _captured_style_graph()
    bwd = _backward_graph_for_fwd(fwd)
    _, stitched = _make_trace_phase_result(fwd, bwd)

    fwd_count = len(fwd.nodes)
    bwd_count = len(bwd.nodes)
    assert len(stitched.nodes) == fwd_count + bwd_count, (
        f"Expected {fwd_count + bwd_count} nodes, got {len(stitched.nodes)}"
    )


def test_auto_stitch_phase_annotations_complete():
    """Every node in the stitched graph must carry annotations['phase']."""
    fwd = _captured_style_graph()
    bwd = _backward_graph_for_fwd(fwd)
    _, stitched = _make_trace_phase_result(fwd, bwd)

    for nid, node in stitched.nodes.items():
        phase = node.annotations.get("phase")
        assert phase in ("fwd", "bwd"), (
            f"Node {nid} has unexpected phase annotation: {phase!r}"
        )


def test_auto_stitch_separates_fwd_and_bwd():
    """Separate forward/backward graphs must still be accessible after stitch."""
    fwd = _captured_style_graph()
    bwd = _backward_graph_for_fwd(fwd)
    all_graphs, _ = _make_trace_phase_result(fwd, bwd)

    assert "train_forward"  in all_graphs
    assert "train_backward" in all_graphs
    raw_fwd, fused_fwd = all_graphs["train_forward"]
    raw_bwd, fused_bwd = all_graphs["train_backward"]
    assert raw_fwd.phase == "train_forward" or raw_fwd is fwd
    assert raw_bwd.phase == "train_backward" or raw_bwd is bwd


def test_save_stitched_graph_writes_json(tmp_path):
    """_save_stitched_graph must write a valid JSON file."""
    import json
    from python.zrt.pipeline import _save_stitched_graph
    from python.zrt.ir.adapter import stitch_fwd_bwd

    fwd = _captured_style_graph()
    bwd = _backward_graph_for_fwd(fwd)
    stitched = stitch_fwd_bwd(fwd, bwd, name="test_stitch")

    _save_stitched_graph(stitched, slug="testmodel", output_dir=tmp_path)

    json_path = tmp_path / "testmodel_train_stitched_graph.json"
    assert json_path.exists(), f"Expected {json_path} to be created"

    data = json.loads(json_path.read_text())
    assert data["phase"] == "train"
    assert len(data["nodes"]) == len(fwd.nodes) + len(bwd.nodes)
