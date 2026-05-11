"""Test memory model — ZeRO stages, activation scaling."""

import pytest
from zrt.training.ir.builders import build_graph
from zrt.training.models.memory import memory_breakdown
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.strategy import Strategy
from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.spec.system import SystemSpec, GPU


def _make_system():
    return SystemSpec(
        gpu=GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80, hbm_bw_gbps=3350),
        host_mem_gb=256,
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0, topology="all_to_all", num_devices=8),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=400, latency_us=5.0, topology="fat_tree"),
        ),
        nodes=1, gpus_per_node=8,
    )


def _make_model():
    return ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 4,
    )


def test_zero_0_no_sharding():
    """ZeRO-0: all optimizer state on each rank."""
    model = _make_model()
    system = _make_system()
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, zero_stage=0)
    graph = build_graph(model, strategy)
    mem = memory_breakdown(graph, model, system, strategy)

    assert mem.opt_state > 0
    assert mem.weights > 0
    assert mem.grads > 0


def test_zero_1_divides_opt_state():
    """ZeRO-1: optimizer state divided by DP."""
    model = _make_model()
    system = _make_system()
    strategy_z0 = Strategy(tp=1, pp=1, dp=4, micro_batch=1, zero_stage=0)
    strategy_z1 = Strategy(tp=1, pp=1, dp=4, micro_batch=1, zero_stage=1)

    graph = build_graph(model, strategy_z0)
    mem_z0 = memory_breakdown(graph, model, system, strategy_z0)
    graph1 = build_graph(model, strategy_z1)
    mem_z1 = memory_breakdown(graph1, model, system, strategy_z1)

    # ZeRO-1 should reduce opt_state by ~4x
    ratio = mem_z1.opt_state / mem_z0.opt_state if mem_z0.opt_state > 0 else 0
    assert ratio == pytest.approx(0.25, abs=0.01), f"ZeRO-1 opt_state ratio: {ratio}"


def test_zero_2_divides_grads_and_opt():
    """ZeRO-2: optimizer state + grads divided by DP."""
    model = _make_model()
    system = _make_system()
    strategy_z0 = Strategy(tp=1, pp=1, dp=4, micro_batch=1, zero_stage=0)
    strategy_z2 = Strategy(tp=1, pp=1, dp=4, micro_batch=1, zero_stage=2)

    graph_z0 = build_graph(model, strategy_z0)
    mem_z0 = memory_breakdown(graph_z0, model, system, strategy_z0)
    graph_z2 = build_graph(model, strategy_z2)
    mem_z2 = memory_breakdown(graph_z2, model, system, strategy_z2)

    # ZeRO-2 should reduce opt_state and grads by ~4x
    opt_ratio = mem_z2.opt_state / mem_z0.opt_state if mem_z0.opt_state > 0 else 0
    grad_ratio = mem_z2.grads / mem_z0.grads if mem_z0.grads > 0 else 0
    assert opt_ratio == pytest.approx(0.25, abs=0.01), f"ZeRO-2 opt_state ratio: {opt_ratio}"
    assert grad_ratio == pytest.approx(0.25, abs=0.01), f"ZeRO-2 grads ratio: {grad_ratio}"


def test_tp_reduces_weights():
    """TP should shard weights across ranks."""
    model = _make_model()
    system = _make_system()

    strategy_1 = Strategy(tp=1, pp=1, dp=1, micro_batch=1)
    strategy_2 = Strategy(tp=2, pp=1, dp=1, micro_batch=1)

    graph1 = build_graph(model, strategy_1)
    graph2 = build_graph(model, strategy_2)
    mem1 = memory_breakdown(graph1, model, system, strategy_1)
    mem2 = memory_breakdown(graph2, model, system, strategy_2)

    # TP=2 should halve weights
    ratio = mem2.weights / mem1.weights if mem1.weights > 0 else 0
    assert ratio == pytest.approx(0.5, abs=0.01)


def test_memory_total_positive():
    """Total memory should be positive and reasonable."""
    model = _make_model()
    system = _make_system()
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1)
    graph = build_graph(model, strategy)
    mem = memory_breakdown(graph, model, system, strategy)

    assert mem.total > 0
    gb = mem.to_gb()
    assert gb["total_gb"] > 0
    assert gb["total_gb"] < system.gpu.hbm_gb  # should fit in HBM for small model


def test_activation_memory_scales_with_microbatch():
    """Activation memory should scale linearly with micro_batch."""
    model = _make_model()
    system = _make_system()

    s1 = Strategy(tp=1, pp=1, dp=1, micro_batch=1)
    s2 = Strategy(tp=1, pp=1, dp=1, micro_batch=2)

    g1 = build_graph(model, s1)
    g2 = build_graph(model, s2)
    m1 = memory_breakdown(g1, model, system, s1)
    m2 = memory_breakdown(g2, model, system, s2)

    assert m2.activations == pytest.approx(m1.activations * 2, rel=0.01)
