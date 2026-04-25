"""Test training modeller integration with transform infrastructure."""

import pytest
from dataclasses import dataclass
from zrt.ir.graph import OpGraph, OpNode
from zrt.ir.types import TensorMeta, DType
from zrt.transform.context import (
    TransformContext, ParallelConfig, TrainingConfig,
)
from zrt.transform.analysis import (
    TrainFlopsPass, TrainingFlopsPass, TrainingMemoryPass, TrainingPipelinePass,
)


def _make_simple_graph(seq_len=2048, hidden=4096, num_layers=32):
    """Create a simple OpGraph with basic transformer ops."""
    nodes = {}
    edges = []

    # Create a simple matmul node
    matmul_node = OpNode(
        id="matmul_0",
        op_type="aten.linear",
        inputs=[
            TensorMeta(id="input_0", shape=(1, seq_len, hidden), dtype=DType.BF16, mem_bytes=1*seq_len*hidden*2),
            TensorMeta(id="weight_0", shape=(hidden, hidden * 4), dtype=DType.BF16, mem_bytes=hidden*(hidden*4)*2),
        ],
        outputs=[
            TensorMeta(id="output_0", shape=(1, seq_len, hidden * 4), dtype=DType.BF16, mem_bytes=1*seq_len*(hidden*4)*2),
        ],
        scope="model.layers.0.mlp.gate_proj",
        category="compute",
    )

    nodes["matmul_0"] = matmul_node

    graph = OpGraph(
        name="test_model",
        phase="forward",
        nodes=nodes,
        edges=edges,
        metadata={
            "seq_len": seq_len,
            "hidden": hidden,
            "num_layers": num_layers,
        },
    )

    return graph


def _make_hardware_spec():
    """Create a mock hardware spec."""
    from zrt.hardware.spec import HardwareSpec, ComputeSpec, MemorySpec, InterconnectSpec, LinkSpec

    compute = ComputeSpec(
        bf16_tflops=1000,  # 1 TFLOPS
    )

    memory = MemorySpec(
        capacity_gb=80,
        hbm_bandwidth_gbps=3000,
    )

    interconnect = InterconnectSpec(
        intra_node=LinkSpec(
            type="nvlink",
            num_devices=8,
            bandwidth_gbps=900,
            latency_us=1.0,
        ),
        inter_node=LinkSpec(
            type="ib",
            num_devices=1000,
            bandwidth_gbps=400,
            latency_us=5.0,
        ),
    )

    return HardwareSpec(
        name="test_gpu",
        vendor="test",
        device_type="gpu",
        compute=compute,
        memory=memory,
        interconnect=interconnect,
    )


def test_training_flops_pass():
    """Test TrainingFlopsPass annotates graph with training FLOPs."""
    graph = _make_simple_graph()
    hw = _make_hardware_spec()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=1),
        training=TrainingConfig(
            micro_batch=1,
            global_batch=32,
        ),
    )

    pass_ = TrainingFlopsPass()
    result = pass_.run(graph, ctx)

    assert "training_flops" in result.metadata
    assert "forward_flops" in result.metadata
    assert "backward_flops" in result.metadata
    assert result.metadata["training_flops"] > 0
    assert result.metadata["backward_flops"] == 2 * result.metadata["forward_flops"]


def test_train_flops_pass_scales_attention_with_graph_compression_ratio():
    """Graph-native attention FLOPs use graph-level compressed attention ratio."""
    attn_node = OpNode(
        id="attn_0",
        op_type="aten.scaled_dot_product_attention",
        inputs=[
            TensorMeta(id="q", shape=(1, 1024, 4096), dtype=DType.BF16, mem_bytes=1*1024*4096*2),
        ],
        outputs=[
            TensorMeta(id="out", shape=(1, 1024, 4096), dtype=DType.BF16, mem_bytes=1*1024*4096*2),
        ],
        scope="model.layers.0.self_attn",
        category="compute",
    )
    graph = OpGraph(
        name="test_model",
        phase="train_forward",
        nodes={"attn_0": attn_node},
        edges=[],
        metadata={"attn_compression_ratio": 0.27},
    )
    hw = _make_hardware_spec()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),
    )

    result = TrainFlopsPass().run(graph, ctx)

    dense_fwd = 2 * 1 * 1024 * 1024 * 64 * 64
    expected_fwd = dense_fwd * 0.27
    node = result.nodes["attn_0"]
    assert node.annotations["flops_fwd"] == int(expected_fwd)
    assert node.annotations["flops_dx"] == int(2.5 * expected_fwd)
    assert node.annotations["flops_dw"] == 0


def test_train_flops_pass_node_compression_ratio_overrides_graph_ratio():
    """Node annotations can override graph-level compressed attention ratio."""
    attn_node = OpNode(
        id="attn_0",
        op_type="aten.scaled_dot_product_attention",
        inputs=[
            TensorMeta(id="q", shape=(1, 512, 1024), dtype=DType.BF16, mem_bytes=1*512*1024*2),
        ],
        outputs=[
            TensorMeta(id="out", shape=(1, 512, 1024), dtype=DType.BF16, mem_bytes=1*512*1024*2),
        ],
        scope="model.layers.0.self_attn",
        category="compute",
        annotations={"attn_compression_ratio": 0.5},
    )
    graph = OpGraph(
        name="test_model",
        phase="train_forward",
        nodes={"attn_0": attn_node},
        edges=[],
        metadata={"attn_compression_ratio": 0.27},
    )
    hw = _make_hardware_spec()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),
    )

    result = TrainFlopsPass().run(graph, ctx)

    dense_fwd = 2 * 1 * 512 * 512 * 16 * 64
    assert result.nodes["attn_0"].annotations["flops_fwd"] == int(dense_fwd * 0.5)


def test_train_flops_pass_reads_attention_head_dim_from_attrs():
    """Graph-native attention dimension inference should not hardcode 64."""
    attn_node = OpNode(
        id="attn_0",
        op_type="aten.scaled_dot_product_attention",
        inputs=[
            TensorMeta(id="q", shape=(1, 40, 512, 80), dtype=DType.BF16, mem_bytes=1*40*512*80*2),
        ],
        outputs=[
            TensorMeta(id="out", shape=(1, 512, 3200), dtype=DType.BF16, mem_bytes=1*512*3200*2),
        ],
        attrs={"heads": 40, "head_dim": 80},
        scope="model.layers.0.self_attn",
        category="compute",
    )
    graph = OpGraph(
        name="test_model",
        phase="train_forward",
        nodes={"attn_0": attn_node},
        edges=[],
        metadata={},
    )
    hw = _make_hardware_spec()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),
    )

    result = TrainFlopsPass().run(graph, ctx)

    expected_fwd = 2 * 1 * 512 * 512 * 40 * 80
    assert result.nodes["attn_0"].annotations["flops_fwd"] == expected_fwd
    assert TrainFlopsPass()._attention_dims(attn_node, graph) == (1, 512, 40, 80)


def test_train_flops_pass_invalid_compression_ratio_warns_and_uses_dense(caplog):
    """Bad graph-native ratio annotations should not abort the whole pass."""
    attn_node = OpNode(
        id="attn_0",
        op_type="aten.scaled_dot_product_attention",
        inputs=[
            TensorMeta(id="q", shape=(1, 512, 1024), dtype=DType.BF16, mem_bytes=1*512*1024*2),
        ],
        outputs=[
            TensorMeta(id="out", shape=(1, 512, 1024), dtype=DType.BF16, mem_bytes=1*512*1024*2),
        ],
        scope="model.layers.0.self_attn",
        category="compute",
        annotations={"attn_compression_ratio": 1.5},
    )
    graph = OpGraph(
        name="test_model",
        phase="train_forward",
        nodes={"attn_0": attn_node},
        edges=[],
        metadata={},
    )
    hw = _make_hardware_spec()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),
    )

    with caplog.at_level("WARNING"):
        result = TrainFlopsPass().run(graph, ctx)

    dense_fwd = 2 * 1 * 512 * 512 * 16 * 64
    assert result.nodes["attn_0"].annotations["flops_fwd"] == dense_fwd
    assert "Invalid attn_compression_ratio=1.5 on node attn_0" in caplog.text


def test_training_memory_pass_zero_1():
    """Test TrainingMemoryPass with ZeRO-1."""
    graph = _make_simple_graph()
    hw = _make_hardware_spec()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=4),
        training=TrainingConfig(
            micro_batch=1,
            global_batch=32,
            zero_stage=1,
        ),
    )

    pass_ = TrainingMemoryPass()
    result = pass_.run(graph, ctx)

    assert "memory_breakdown" in result.metadata
    breakdown = result.metadata["memory_breakdown"]
    assert breakdown.weights > 0
    assert breakdown.grads > 0
    assert breakdown.opt_state > 0
    assert breakdown.activations > 0
    assert breakdown.total > 0


def test_training_memory_pass_zero_2():
    """Test TrainingMemoryPass with ZeRO-2 (gradients sharded)."""
    graph = _make_simple_graph()
    hw = _make_hardware_spec()
    ctx_zero1 = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=4),
        training=TrainingConfig(
            micro_batch=1,
            global_batch=32,
            zero_stage=1,
        ),
    )
    ctx_zero2 = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=4),
        training=TrainingConfig(
            micro_batch=1,
            global_batch=32,
            zero_stage=2,
        ),
    )

    pass_ = TrainingMemoryPass()
    result_z1 = pass_.run(graph, ctx_zero1)
    result_z2 = pass_.run(graph, ctx_zero2)

    # ZeRO-2 should have smaller gradients than ZeRO-1
    assert result_z2.metadata["memory_breakdown"].grads < result_z1.metadata["memory_breakdown"].grads


def test_training_pipeline_pass():
    """Test TrainingPipelinePass computes pipeline metrics."""
    graph = _make_simple_graph()
    hw = _make_hardware_spec()

    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=2, dp=1),
        training=TrainingConfig(
            micro_batch=1,
            global_batch=8,
        ),
    )

    # Manually add some latency annotations for testing
    for node in graph.nodes.values():
        node.annotations["latency_us"] = 100.0  # 100us per op

    pipeline_pass = TrainingPipelinePass()
    result = pipeline_pass.run(graph, ctx)

    assert "pipeline_metrics" in result.metadata
    metrics = result.metadata["pipeline_metrics"]
    assert metrics.warmup_steps == 1  # pp - 1
    assert metrics.cooldown_steps == 1  # pp - 1
    assert metrics.steady_steps == 8  # M (num_microbatches)
    assert 0 < metrics.bubble_fraction < 1


def test_training_config_num_microbatches():
    """Test TrainingConfig.num_microbatches property."""
    config = TrainingConfig(micro_batch=2, global_batch=32)
    assert config.num_microbatches == 16

    config = TrainingConfig(micro_batch=1, global_batch=8)
    assert config.num_microbatches == 8


# ── Layer scaling tests (Change 4) ───────────────────────────────────────────

def test_layer_scaling_flops():
    """TrainingFlopsPass scales FLOPs by num_layers/num_layers_traced when they differ."""
    graph = _make_simple_graph(num_layers=8)  # full model: 8 layers
    graph.metadata["num_layers_traced"] = 2   # only 2 were traced
    hw = _make_hardware_spec()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),
    )
    pass_ = TrainingFlopsPass()
    result = pass_.run(graph, ctx)

    # Without total_params override, graph-counted params scale by 8/2=4
    assert result.metadata["layer_scale"] == 4.0
    # total_params should be 4x the single matmul weight
    single_weight = 4096 * (4096 * 4)  # hidden * (hidden*4)
    assert result.metadata["total_params"] == single_weight * 4


def test_layer_scaling_no_effect_when_equal():
    """When num_layers == num_layers_traced, layer_scale is 1.0 (no scaling)."""
    graph = _make_simple_graph(num_layers=32)
    hw = _make_hardware_spec()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),
    )
    pass_ = TrainingFlopsPass()
    result = pass_.run(graph, ctx)
    assert result.metadata["layer_scale"] == 1.0


def test_layer_scaling_step_time():
    """TrainingPipelinePass scales stage_time by layer_scale."""
    graph = _make_simple_graph(num_layers=8)
    graph.metadata["num_layers_traced"] = 2
    # Add latency so DAGScheduler has something to schedule
    for node in graph.nodes.values():
        node.annotations["latency_us"] = 100.0
    hw = _make_hardware_spec()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),
    )
    # Run FlopsPass first to set layer_scale
    from zrt.transform.analysis import TrainingFlopsPass as TFP
    graph = TFP().run(graph, ctx)

    result = TrainingPipelinePass().run(graph, ctx)
    # stage_time should be 100us * 4 (layer_scale) = 400us per microbatch
    metrics = result.metadata["pipeline_metrics"]
    assert metrics.step_time_ms > 0


# ── Adam optimizer-state tests (Change 5) ────────────────────────────────────

def test_adam_opt_state_12_bytes_per_param():
    """Adam under mixed precision: master(FP32) + m(FP32) + v(FP32) = 12 B/P."""
    graph = _make_simple_graph()
    hw = _make_hardware_spec()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=1),
        training=TrainingConfig(
            optimizer="adam",
            zero_stage=0,
            micro_batch=1,
            global_batch=8,
        ),
    )
    pass_ = TrainingMemoryPass()
    result = pass_.run(graph, ctx)
    breakdown = result.metadata["memory_breakdown"]
    total_params = result.metadata.get("total_params", 0)
    if total_params == 0:
        total_params = 4096 * (4096 * 4)  # weight_0 shape
    # opt_state should be 12 * total_params bytes (not 4 * total_params)
    expected_opt_bytes = total_params * 12
    assert breakdown.opt_state == pytest.approx(expected_opt_bytes, rel=0.01), (
        f"Expected opt_state ≈ {expected_opt_bytes} (12 B/P), got {breakdown.opt_state}"
    )


def test_adamw_opt_state_same_as_adam():
    """AdamW should use same 12 B/P as Adam."""
    graph = _make_simple_graph()
    hw = _make_hardware_spec()
    ctx_adam = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=1),
        training=TrainingConfig(optimizer="adam", zero_stage=0, micro_batch=1, global_batch=8),
    )
    ctx_adamw = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=1),
        training=TrainingConfig(optimizer="adamw", zero_stage=0, micro_batch=1, global_batch=8),
    )
    pass_ = TrainingMemoryPass()
    opt_adam = pass_.run(graph, ctx_adam).metadata["memory_breakdown"].opt_state
    opt_adamw = pass_.run(graph, ctx_adamw).metadata["memory_breakdown"].opt_state
    assert opt_adam == pytest.approx(opt_adamw, rel=0.01)
