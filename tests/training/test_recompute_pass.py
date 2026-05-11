"""Test RecomputePass selective policy annotation accuracy."""
from __future__ import annotations

import pytest

from zrt.hardware.spec import (
    ComputeSpec, HardwareSpec, InterconnectSpec, LinkSpec, MemorySpec,
)
from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.types import DType, TensorMeta
from zrt.transform.context import TransformContext, TrainingConfig
from zrt.transform.training.recompute import RecomputePass


def _hw():
    return HardwareSpec(
        name="test_h100", vendor="nvidia", device_type="gpu",
        compute=ComputeSpec(bf16_tflops=989, fp8_tops=1979),
        memory=MemorySpec(capacity_gb=80, hbm_bandwidth_gbps=3350),
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=100, latency_us=10.0),
        ),
    )


def _ctx(recompute_policy="selective"):
    return TransformContext(
        hw_spec=_hw(),
        training=TrainingConfig(
            micro_batch=1, global_batch=8,
            recompute_policy=recompute_policy,
        ),
    )


def _tensor(shape=(1, 128, 4096)):
    return TensorMeta(
        id="t", shape=shape, dtype=DType.BF16,
        mem_bytes=shape[0] * shape[1] * shape[2] * 2 if len(shape) == 3 else shape[0] * shape[1] * 2
    )


def _softmax_node():
    return OpNode(
        id="softmax_0",
        op_type="aten.softmax",
        inputs=[_tensor()],
        outputs=[_tensor()],
        scope="model.layers.0.self_attn",
        category="compute",
    )


def _attn_proj_node():
    return OpNode(
        id="o_proj_0",
        op_type="aten.linear",
        inputs=[_tensor()],
        outputs=[_tensor()],
        scope="model.layers.0.self_attn.o_proj",
        category="compute",
    )


def _flash_attn_node():
    return OpNode(
        id="flash_attn_0",
        op_type="flash_attn.flash_attn_func",
        inputs=[_tensor((1, 128, 4096)), _tensor((1, 128, 4096))],
        outputs=[_tensor((1, 128, 4096))],
        scope="model.layers.0.self_attn",
        category="compute",
    )


def _layer_norm_node():
    return OpNode(
        id="ln_0",
        op_type="aten.layer_norm",
        inputs=[_tensor()],
        outputs=[_tensor()],
        scope="model.layers.0.post_attention_layernorm",
        category="compute",
    )


def _ffn_swiglu_node():
    return OpNode(
        id="swiglu_0",
        op_type="aten.silu",
        inputs=[_tensor()],
        outputs=[_tensor()],
        scope="model.layers.0.mlp",
        category="compute",
    )


def _matmul_node():
    return OpNode(
        id="mm_0",
        op_type="aten.mm",
        inputs=[_tensor()],
        outputs=[_tensor()],
        scope="model.layers.0.mlp.gate_proj",
        category="compute",
    )


def _make_graph(nodes, phase="forward", metadata=None):
    return OpGraph(
        name="test",
        phase=phase,
        nodes={n.id: n for n in nodes},
        metadata=metadata or {},
    )


def test_selective_recomputes_softmax():
    """Selective policy: softmax is the primary target."""
    g = _make_graph([_softmax_node()])
    ctx = _ctx("selective")
    result = RecomputePass().run(g, ctx)
    assert result.nodes["softmax_0"].annotations.get("recompute") is True
    assert result.nodes["softmax_0"].annotations.get("recompute_policy") == "selective"


def test_selective_recomputes_attn_output_projection():
    """Selective policy: O_proj is targeted."""
    g = _make_graph([_attn_proj_node()])
    ctx = _ctx("selective")
    result = RecomputePass().run(g, ctx)
    assert result.nodes["o_proj_0"].annotations.get("recompute") is True


def test_selective_recomputes_flash_attn():
    """Selective policy: flash_attn is targeted."""
    g = _make_graph([_flash_attn_node()])
    ctx = _ctx("selective")
    result = RecomputePass().run(g, ctx)
    assert result.nodes["flash_attn_0"].annotations.get("recompute") is True


def test_selective_excludes_layer_norm():
    """Selective policy: layer_norm is excluded."""
    g = _make_graph([_layer_norm_node()])
    ctx = _ctx("selective")
    result = RecomputePass().run(g, ctx)
    assert result.nodes["ln_0"].annotations.get("recompute") is not True


def test_selective_excludes_ffn_swiglu():
    """Selective policy: SwiGLU in FFN is excluded."""
    g = _make_graph([_ffn_swiglu_node()])
    ctx = _ctx("selective")
    result = RecomputePass().run(g, ctx)
    assert result.nodes["swiglu_0"].annotations.get("recompute") is not True


def test_selective_excludes_matmul():
    """Selective policy: generic matmul is excluded."""
    g = _make_graph([_matmul_node()])
    ctx = _ctx("selective")
    result = RecomputePass().run(g, ctx)
    assert result.nodes["mm_0"].annotations.get("recompute") is not True


def test_full_recomputes_all_forward_ops():
    """Full policy: all forward ops are marked."""
    g = _make_graph([
        _softmax_node(),
        _layer_norm_node(),
        _ffn_swiglu_node(),
        _matmul_node(),
    ])
    ctx = _ctx("full")
    result = RecomputePass().run(g, ctx)
    for nid in ("softmax_0", "ln_0", "swiglu_0", "mm_0"):
        assert result.nodes[nid].annotations.get("recompute") is True
        assert result.nodes[nid].annotations.get("recompute_policy") == "full"


def test_none_recomputes_nothing():
    """None policy: no ops are marked."""
    g = _make_graph([
        _softmax_node(),
        _attn_proj_node(),
    ])
    ctx = _ctx("none")
    result = RecomputePass().run(g, ctx)
    for nid in ("softmax_0", "o_proj_0"):
        assert result.nodes[nid].annotations.get("recompute") is not True


def test_backward_phase_nodes_skipped():
    """RecomputePass should skip backward-phase nodes in stitched graphs."""
    softmax = _softmax_node()
    softmax.annotations["phase"] = "bwd"
    g = _make_graph([softmax])
    g.metadata["phase"] = "train_backward"
    ctx = _ctx("selective")
    result = RecomputePass().run(g, ctx)
    assert result.nodes["softmax_0"].annotations.get("recompute") is not True


def test_mixed_graph_selective_vs_full():
    """Verify selective marks subset, full marks all."""
    nodes = [
        _softmax_node(),
        _layer_norm_node(),
        _ffn_swiglu_node(),
    ]
    g_selective = _make_graph(nodes)
    ctx_selective = _ctx("selective")
    r_selective = RecomputePass().run(g_selective, ctx_selective)

    g_full = _make_graph(nodes)
    ctx_full = _ctx("full")
    r_full = RecomputePass().run(g_full, ctx_full)

    # Selective marks fewer than full
    selective_count = sum(
        1 for n in r_selective.nodes.values() if n.annotations.get("recompute")
    )
    full_count = sum(
        1 for n in r_full.nodes.values() if n.annotations.get("recompute")
    )
    assert selective_count < full_count
    assert selective_count == 1  # Only softmax
    assert full_count == 3  # All nodes
