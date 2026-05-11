"""Test IR builders — dense block op count and shapes."""

import pytest
from zrt.training.ir.builders import dense_block, build_graph
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.strategy import Strategy


def test_dense_block_op_count():
    """Dense block should produce 12 ops (LN, QKV, RoPE, attn, O_proj, add, LN, up, gate, swiglu, down, add)."""
    ops = dense_block(
        hidden=8192, ffn=28672, seq=4096,
        num_heads=64, num_kv_heads=8, head_dim=128,
        layer_id=0,
    )
    assert len(ops) == 12
    kinds = [op.kind for op in ops]
    assert kinds == ["ln", "matmul", "rope", "attn_core", "matmul", "add",
                     "ln", "matmul", "matmul", "swiglu", "matmul", "add"]


def test_dense_block_matmul_meta():
    """Matmul ops should have m, n, k meta."""
    ops = dense_block(
        hidden=4096, ffn=16384, seq=2048,
        num_heads=32, num_kv_heads=32, head_dim=128,
        layer_id=0,
    )
    qkv = ops[1]  # QKV projection
    assert qkv.kind == "matmul"
    assert qkv.meta["m"] == 2048
    assert qkv.meta["k"] == 4096
    assert qkv.meta["n"] == 32 * 128 + 2 * 32 * 128  # 3 * h_kv for MHA


def test_build_graph_basic():
    """Build graph with 2 dense layers, no parallelism."""
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 2,
    )
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1)
    graph = build_graph(model, strategy)

    # 2 layers * 12 ops + embed + final_ln + lm_head = 27 ops
    assert len(graph.ops) == 2 * 12 + 3
    assert len(graph.collectives) == 0  # no TP, no collectives
    assert 0 in graph.layer_index
    assert 1 in graph.layer_index


def test_build_graph_with_tp():
    """Build graph with TP=2, should have collectives."""
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 2,
    )
    strategy = Strategy(tp=2, pp=1, dp=1, micro_batch=1)
    graph = build_graph(model, strategy)

    # 4 collectives per layer: AG before QKV, RS after O_proj, AG before up, RS after down
    assert len(graph.collectives) == 4 * 2

    # Check collective kinds
    kinds = [c.kind for c in graph.collectives]
    ag_count = kinds.count("AG")
    rs_count = kinds.count("RS")
    assert ag_count == 4  # 2 AG per layer
    assert rs_count == 4  # 2 RS per layer

    layer_ops = graph.ops_for_layer(0)
    attn = next(op for op in layer_ops if op.kind == "attn_core")
    swiglu = next(op for op in layer_ops if op.kind == "swiglu")
    assert attn.meta["heads"] == 16
    assert attn.inputs[0].shape_local == (2048, 2048)
    assert swiglu.meta["bytes_fwd"] == 2048 * 16384 * Dtype.BF16.bytes * 3 // 2
    assert isinstance(swiglu.meta["bytes_fwd"], int)


def test_ops_for_layer():
    """ops_for_layer should return the correct ops."""
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 3,
    )
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1)
    graph = build_graph(model, strategy)

    for lid in range(3):
        layer_ops = graph.ops_for_layer(lid)
        assert len(layer_ops) == 12
        assert all(op.layer_id == lid for op in layer_ops)


def test_attn_core_meta():
    """Attention core should have b, s, heads, head_dim, causal."""
    ops = dense_block(
        hidden=4096, ffn=16384, seq=2048,
        num_heads=32, num_kv_heads=8, head_dim=128,
        layer_id=0,
    )
    attn = ops[3]
    assert attn.kind == "attn_core"
    assert attn.meta["s"] == 2048
    assert attn.meta["heads"] == 32
    assert attn.meta["head_dim"] == 128
    assert attn.meta["causal"] is True


def test_tp_ag_rs_phase_both():
    """TP AG/RS collectives should have phase='both' (forward + backward communication)."""
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 2,
    )
    strategy = Strategy(tp=2, pp=1, dp=1, micro_batch=1)
    graph = build_graph(model, strategy)

    for c in graph.collectives:
        if c.group == "TP":
            assert c.phase == "both", f"TP collective {c.name} should have phase='both'"


def test_ring_cp_cp_tiles_marker():
    """Ring CP should set cp_tiles marker on attn_core for FLOPs calculation."""
    from zrt.training.spec.strategy import CPKind

    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 2,
    )
    strategy = Strategy(tp=1, pp=1, dp=1, cp=4, cp_kind=CPKind.RING, micro_batch=1)
    graph = build_graph(model, strategy)

    for op in graph.ops:
        if op.kind == "attn_core":
            assert op.meta.get("cp_tiles") == 4, (
                f"Ring CP should set cp_tiles=4, got {op.meta.get('cp_tiles')}"
            )
            assert op.meta.get("heads_gathered_by_cp") is False


def test_ulysses_cp_heads_gathered():
    """Ulysses CP should set heads_gathered_by_cp=True and multiply heads by cp."""
    from zrt.training.spec.strategy import CPKind

    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 2,
    )
    strategy = Strategy(tp=2, pp=1, dp=1, cp=4, cp_kind=CPKind.ULYSSES, micro_batch=1)
    graph = build_graph(model, strategy)

    for op in graph.ops:
        if op.kind == "attn_core":
            heads_tp = 32 // 2  # TP shards heads
            heads_after_cp = heads_tp * 4  # CP gathers heads
            assert op.meta.get("heads") == heads_after_cp, (
                f"Ulysses CP should multiply heads to {heads_after_cp}, got {op.meta.get('heads')}"
            )
            assert op.meta.get("heads_gathered_by_cp") is True


def test_ep_a2a_phase_both():
    """EP A2A collectives should have phase='both' (forward + backward communication)."""
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.MOE],
        moe_ffn=16384, num_experts=8, top_k=2,
    )
    strategy = Strategy(tp=1, pp=1, dp=1, ep=4, micro_batch=1)
    graph = build_graph(model, strategy)

    for c in graph.collectives:
        if c.group == "EP":
            assert c.phase == "both", f"EP collective {c.name} should have phase='both'"


def test_ep_a2a_bytes_with_tp():
    """EP A2A bytes should consider TP sharding (hidden/tp)."""
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.MOE],
        moe_ffn=16384, num_experts=8, top_k=2,
    )
    strategy = Strategy(tp=4, pp=1, dp=1, ep=2, micro_batch=1)
    graph = build_graph(model, strategy)

    expected_bytes = model.seq_len * (model.hidden // 4) * model.act_dtype.bytes

    for c in graph.collectives:
        if c.group == "EP" and c.kind == "A2A":
            assert c.bytes_ == expected_bytes, (
                f"EP A2A bytes should be {expected_bytes} (with TP=4), got {c.bytes_}"
            )


def test_cp_ep_a2a_bytes_considers_cp():
    """EP A2A bytes should consider CP sharding (seq/cp)."""
    from zrt.training.spec.strategy import CPKind

    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.MOE],
        moe_ffn=16384, num_experts=8, top_k=2,
    )
    strategy = Strategy(tp=1, pp=1, dp=1, ep=2, cp=4, cp_kind=CPKind.ULYSSES, micro_batch=1)
    graph = build_graph(model, strategy)

    expected_seq = model.seq_len // 4
    expected_bytes = expected_seq * model.hidden * model.act_dtype.bytes

    for c in graph.collectives:
        if c.group == "EP" and c.kind == "A2A":
            assert c.bytes_ == expected_bytes, (
                f"EP A2A bytes should be {expected_bytes} (with CP=4), got {c.bytes_}"
            )


def test_tp_ag_rs_bytes_with_cp():
    """TP AG/RS bytes should consider CP sharding (seq/cp)."""
    from zrt.training.spec.strategy import CPKind

    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 2,
    )
    strategy = Strategy(tp=2, pp=1, dp=1, cp=4, cp_kind=CPKind.ULYSSES, micro_batch=1)
    graph = build_graph(model, strategy)

    expected_seq = model.seq_len // 4
    expected_bytes = expected_seq * model.hidden * model.act_dtype.bytes

    for c in graph.collectives:
        if c.group == "TP" and c.kind in ("AG", "RS"):
            assert c.bytes_ == expected_bytes, (
                f"TP AG/RS bytes should be {expected_bytes} (with CP=4), got {c.bytes_}"
            )


def test_comm_buffer_memory_with_cp():
    """Communication buffer memory should include CP A2A buffers."""
    from zrt.training.models.memory import _comm_buffer_memory

    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 2,
    )
    strategy = Strategy(tp=1, pp=1, dp=1, cp=4, micro_batch=1)

    comm_buf = _comm_buffer_memory(model, strategy)
    expected_per_layer = 4 * (2048 // 4) * 4096 * 2
    expected_total = expected_per_layer * 2 * 1

    assert comm_buf == expected_total, (
        f"Comm buffer with CP should be {expected_total}, got {comm_buf}"
    )


def test_comm_buffer_memory_with_ep():
    """Communication buffer memory should include EP A2A buffers for MoE."""
    from zrt.training.models.memory import _comm_buffer_memory

    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.MOE],
        moe_ffn=16384, num_experts=8, top_k=2,
    )
    strategy = Strategy(tp=1, pp=1, dp=1, ep=2, micro_batch=1)

    comm_buf = _comm_buffer_memory(model, strategy)
    expected_per_layer = 4 * 2048 * 4096 * 2
    expected_total = expected_per_layer * 1 * 1

    assert comm_buf == expected_total, (
        f"Comm buffer with EP should be {expected_total}, got {comm_buf}"
    )
