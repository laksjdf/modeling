"""Test training modeller integration with transform infrastructure."""

import pytest
from dataclasses import dataclass
from zrt.ir.graph import OpGraph, OpNode
from zrt.ir.types import TensorMeta, DType
from zrt.transform.context import (
    TransformContext, ParallelConfig, TrainingConfig,
)
from zrt.transform.analysis import (
    TrainingFlopsPass, TrainingMemoryPass, TrainingPipelinePass,
    estimate_training,
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
    assert metrics.steady_steps == 7  # M - pp + 1
    assert 0 < metrics.bubble_fraction < 1


def test_estimate_training():
    """Test estimate_training() function returns a complete report."""
    graph = _make_simple_graph()
    hw = _make_hardware_spec()

    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=2, pp=2, dp=4),
        training=TrainingConfig(
            optimizer="adam",
            zero_stage=1,
            micro_batch=1,
            global_batch=32,
        ),
    )

    # Manually add some latency annotations for testing
    for node in graph.nodes.values():
        node.annotations["latency_us"] = 100.0  # 100us per op

    report = estimate_training(graph, ctx)

    assert report.config_summary != ""
    assert "TP2" in report.config_summary
    assert "PP2" in report.config_summary
    assert "DP4" in report.config_summary
    assert "ZeRO-1" in report.config_summary

    assert report.step_time_ms >= 0
    assert report.mfu >= 0
    assert report.training_flops >= 0
    assert report.total_params >= 0
    assert len(report.memory_breakdown) > 0

    # Test to_dict()
    d = report.to_dict()
    assert "step_time_ms" in d
    assert "mfu" in d
    assert "memory_breakdown_gb" in d

    # Test summary()
    summary = report.summary()
    assert "Training Estimation Report" in summary
    assert "Step time" in summary
    assert "MFU" in summary


def test_training_config_num_microbatches():
    """Test TrainingConfig.num_microbatches property."""
    config = TrainingConfig(micro_batch=2, global_batch=32)
    assert config.num_microbatches == 16

    config = TrainingConfig(micro_batch=1, global_batch=8)
    assert config.num_microbatches == 8
