from python.zrt.hardware import load
from python.zrt.ir.edge import Edge
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.ir.types import DType, TensorMeta
from python.zrt.memory import ActivationAnalysis, MemoryBudget, MemoryModel, analyze_activation
from python.zrt.transform import ParallelConfig, QuantConfig


def test_memory_budget_breakdown_contains_expected_keys():
    budget = MemoryBudget(
        weights_mb=1.0,
        kv_cache_mb=2.0,
        activation_peak_mb=3.0,
        comm_buffer_mb=4.0,
        framework_overhead_mb=0.5,
        total_mb=10.5,
        capacity_mb=80.0,
        is_feasible=True,
    )
    data = budget.breakdown()
    assert data["weights_mb"] == 1.0
    assert data["total_mb"] == 10.5
    assert budget.utilization == 10.5 / 80.0


def test_memory_model_scales_weights_with_tp():
    profile = {
        "total_params": 8_000_000_000,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
    }
    model = MemoryModel(overhead_ratio=0.0)
    hw = load("nvidia_h100_sxm")

    single = model.estimate(profile, hw, ParallelConfig(tp=1))
    tp4 = model.estimate(profile, hw, ParallelConfig(tp=4))

    assert tp4.weights_mb < single.weights_mb
    assert tp4.weights_mb == single.weights_mb / 4


def test_memory_model_uses_graph_metadata_and_quantized_kv():
    graph = OpGraph(
        name="g",
        phase="decode",
        metadata={
            "hidden_size": 7168,
            "intermediate_size": 18432,
            "num_hidden_layers (traced)": 16,
            "num_attention_heads": 56,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 129280,
            "total_params": 14_000_000_000,
        },
    )
    model = MemoryModel(overhead_ratio=0.0)
    hw = load("nvidia_a100_80g")
    budget = model.estimate(
        graph,
        hw,
        ParallelConfig(tp=4, pp=2, sp=True),
        quant=QuantConfig(weight="int8", activation="bf16", kv_cache="fp8"),
        batch_size=2,
        seq_len=4096,
    )

    assert budget.kv_cache_mb > 0
    assert budget.activation_peak_mb > 0
    assert budget.comm_buffer_mb > 0
    assert budget.weights_mb > budget.kv_cache_mb


def test_memory_model_marks_oom_when_capacity_is_too_small():
    profile = {
        "total_params": 70_000_000_000,
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "vocab_size": 128000,
    }
    model = MemoryModel()
    hw = load("ascend_910b")
    budget = model.estimate(profile, hw, ParallelConfig(tp=1), batch_size=8, seq_len=8192)

    assert budget.total_mb > budget.capacity_mb
    assert budget.is_feasible is False


def test_sequence_parallel_reduces_activation_peak():
    profile = {
        "total_params": 13_000_000_000,
        "hidden_size": 5120,
        "intermediate_size": 13824,
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "vocab_size": 64000,
    }
    model = MemoryModel(overhead_ratio=0.0)
    hw = load("nvidia_h800")

    no_sp = model.estimate(profile, hw, ParallelConfig(tp=4, sp=False), batch_size=1, seq_len=4096)
    sp = model.estimate(profile, hw, ParallelConfig(tp=4, sp=True), batch_size=1, seq_len=4096)

    assert sp.activation_peak_mb < no_sp.activation_peak_mb


def test_activation_analysis_simple():
    """Linear 3-node graph: A → B → C, with B producing 2 outputs."""
    t_a_out = TensorMeta.from_shape_dtype("t_a_out", (2, 4, 1024), DType.BF16)
    t_b_out1 = TensorMeta.from_shape_dtype("t_b_out1", (2, 4, 1024), DType.BF16)
    t_b_out2 = TensorMeta.from_shape_dtype("t_b_out2", (2, 4, 1024), DType.BF16)
    t_c_out = TensorMeta.from_shape_dtype("t_c_out", (2, 4, 1024), DType.BF16)

    node_a = OpNode(id="op_a", op_type="aten.linear.default", outputs=[t_a_out])
    node_b = OpNode(id="op_b", op_type="aten.split.Tensor", inputs=[t_a_out], outputs=[t_b_out1, t_b_out2])
    node_c = OpNode(id="op_c", op_type="aten.add.Tensor", inputs=[t_b_out1, t_b_out2], outputs=[t_c_out])

    graph = OpGraph(
        name="test_linear",
        phase="prefill",
        nodes={"op_a": node_a, "op_b": node_b, "op_c": node_c},
        edges=[
            Edge(src="op_a", src_idx=0, dst="op_b", dst_idx=0, tensor=t_a_out),
            Edge(src="op_b", src_idx=0, dst="op_c", dst_idx=0, tensor=t_b_out1),
            Edge(src="op_b", src_idx=1, dst="op_c", dst_idx=1, tensor=t_b_out2),
        ],
    )

    analysis = analyze_activation(graph)
    assert analysis.peak_bytes > 0
    assert analysis.peak_mb > 0.0
    assert analysis.peak_node_id == "op_b"
    assert "op_a" in analysis.per_node_live_mb
    assert "op_b" in analysis.per_node_live_mb
    assert "op_c" in analysis.per_node_live_mb
    assert analysis.per_node_live_mb["op_b"] > analysis.per_node_live_mb["op_a"]


def test_activation_analysis_peak_node_id():
    """Verify peak_node_id is the node after which peak occurs."""
    t1 = TensorMeta.from_shape_dtype("t1", (1, 1024), DType.BF16)
    t2 = TensorMeta.from_shape_dtype("t2", (1, 2048), DType.BF16)

    node1 = OpNode(id="n1", op_type="aten.linear.default", outputs=[t1])
    node2 = OpNode(id="n2", op_type="aten.linear.default", inputs=[t1], outputs=[t2])

    graph = OpGraph(
        name="test_peak",
        phase="prefill",
        nodes={"n1": node1, "n2": node2},
        edges=[Edge(src="n1", src_idx=0, dst="n2", dst_idx=0, tensor=t1)],
    )

    analysis = analyze_activation(graph)
    assert analysis.peak_node_id == "n2"
    assert analysis.per_node_live_mb["n2"] > 0.0


def test_memory_model_uses_graph_activation():
    """OpGraph with nodes should use analyze_activation, not formula."""
    t_out = TensorMeta.from_shape_dtype("t_out", (1, 256), DType.BF16)
    node = OpNode(id="op1", op_type="aten.linear.default", outputs=[t_out])

    graph = OpGraph(
        name="test_graph_activation",
        phase="prefill",
        nodes={"op1": node},
        metadata={
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
        },
    )

    model = MemoryModel(overhead_ratio=0.0, activation_slack=1.0)
    hw = load("nvidia_h100_sxm")

    budget = model.estimate(graph, hw, ParallelConfig(tp=1), batch_size=1, seq_len=128)
    analysis = analyze_activation(graph)

    assert budget.activation_peak_mb == analysis.peak_mb
    assert budget.activation_peak_mb > 0.0


def test_memory_model_mla_architecture():
    """MLA (Multi-head Latent Attention) 架构：使用 kv_lora_rank + qk_rope_head_dim"""
    profile = {
        "total_params": 8_000_000_000,
        "hidden_size": 5120,
        "intermediate_size": 13824,
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "vocab_size": 64000,
        # MLA 特殊字段（DeepSeek-V2, Qwen-2.5 等）
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
    }
    model = MemoryModel(overhead_ratio=0.0)
    hw = load("nvidia_h100_sxm")

    # 不使用 TP，不使用 PP（便于对比）
    budget = model.estimate(
        profile, hw, ParallelConfig(tp=1, pp=1), batch_size=1, seq_len=4096
    )

    # MLA 下 kv_dim = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576
    # kv_cache = 2 * layers * kv_dim * seq_len * batch_size * kv_bytes
    #         = 2 * 40 * 576 * 4096 * 1 * 2 / (1024*1024)
    expected_kv_mb = 2 * 40 * 576 * 4096 * 1 * 2.0 / (1024.0 ** 2)

    assert budget.kv_cache_mb > 0
    assert abs(budget.kv_cache_mb - expected_kv_mb) < 0.1  # 容差 0.1 MB


def test_memory_model_ep_shards_weights():
    """Expert Parallel 应该按 EP 因子分片权重"""
    profile = {
        "total_params": 70_000_000_000,
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "vocab_size": 128000,
        "num_experts": 256,
        "num_shared_experts": 2,
        "moe_topk": 6,
    }
    model = MemoryModel(overhead_ratio=0.0)
    hw = load("nvidia_h100_sxm")

    single = model.estimate(profile, hw, ParallelConfig(tp=1, ep=1))
    ep8 = model.estimate(profile, hw, ParallelConfig(tp=1, ep=8))

    # EP 分片应该减少权重显存
    assert ep8.weights_mb < single.weights_mb
    assert ep8.weights_mb == single.weights_mb / 8
