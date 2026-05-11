"""Test training modeller integration with transform infrastructure."""

import pytest
from zrt.ir.graph import OpGraph, OpNode
from zrt.ir.types import TensorMeta, DType
from zrt.transform.context import (
    TransformContext, ParallelConfig, TrainingConfig,
)
from zrt.transform.analysis import (
    FlopsPass,
    TrainingFlopsPass, TrainingMemoryPass, TrainingPipelinePass,
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


def test_training_memory_pass_zero_1():
    """Test TrainingMemoryPass with ZeRO-1 (optimizer state sharded by dp)."""
    graph = _make_simple_graph()
    hw = _make_hardware_spec()
    dp = 4

    def _ctx(zero_stage):
        return TransformContext(
            hw_spec=hw,
            parallel=ParallelConfig(tp=1, pp=1, dp=dp),
            training=TrainingConfig(
                micro_batch=1,
                global_batch=32,
                zero_stage=zero_stage,
            ),
        )

    pass_ = TrainingMemoryPass()
    z0 = pass_.run(graph, _ctx(0)).metadata["memory_breakdown"]
    z1 = pass_.run(graph, _ctx(1)).metadata["memory_breakdown"]

    # ZeRO-1: optimizer state sharded by dp; weights and grads unchanged
    assert z1.opt_state == pytest.approx(z0.opt_state / dp, rel=0.01)
    assert z1.weights == pytest.approx(z0.weights, rel=0.01)
    assert z1.grads == pytest.approx(z0.grads, rel=0.01)


def test_training_memory_pass_zero_2():
    """Test TrainingMemoryPass with ZeRO-2 (optimizer state + gradients sharded by dp)."""
    graph = _make_simple_graph()
    hw = _make_hardware_spec()
    dp = 4

    def _ctx(zero_stage):
        return TransformContext(
            hw_spec=hw,
            parallel=ParallelConfig(tp=1, pp=1, dp=dp),
            training=TrainingConfig(
                micro_batch=1,
                global_batch=32,
                zero_stage=zero_stage,
            ),
        )

    pass_ = TrainingMemoryPass()
    z0 = pass_.run(graph, _ctx(0)).metadata["memory_breakdown"]
    z2 = pass_.run(graph, _ctx(2)).metadata["memory_breakdown"]

    # ZeRO-2: both optimizer state AND gradients sharded by dp; weights unchanged
    assert z2.opt_state == pytest.approx(z0.opt_state / dp, rel=0.01)
    assert z2.grads == pytest.approx(z0.grads / dp, rel=0.01)
    assert z2.weights == pytest.approx(z0.weights, rel=0.01)


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
    # opt_state should be 12 * total_params bytes for Adam (FP32: master + m + v = 3 × 4 bytes)
    expected_opt_bytes = total_params * 12
    assert breakdown.opt_state == pytest.approx(expected_opt_bytes, rel=0.01), (
        f"Expected opt_state ≈ {expected_opt_bytes} (12 B/P for FP32 Adam), got {breakdown.opt_state}"
    )


def test_adamw_opt_state_same_as_adam():
    """AdamW should use same 12 B/P as Adam (FP32: master + m + v = 3 × 4 bytes)."""
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
