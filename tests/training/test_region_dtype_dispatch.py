"""Tests for region-level activation dtype dispatch in builders (Stage B).

Verifies that:
  - attention sub-block tensors carry ``model.effective_attn_act_dtype()``
  - MoE FFN sub-block tensors carry ``model.effective_moe_act_dtype()``
  - residual stream tensors stay at ``model.act_dtype``
  - LN ops at region boundaries have mismatched input/output dtype (the
    boundary the v2 cast_pass will fire on in Stage D)
  - EP A2A / CP A2A / TP AG-RS bytes scale with the region dtype
  - BF16 baseline (all dtypes default) is numerically unchanged from v1
"""
from __future__ import annotations

import pytest

from zrt.training.ir.builders import build_graph
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import Strategy


def _moe_model(**dtype_kwargs) -> ModelSpec:
    return ModelSpec(
        hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
        vocab=1000, seq_len=64, layers=[LayerKind.MOE],
        num_experts=8, moe_ffn=256, top_k=2, n_shared_experts=1,
        **dtype_kwargs,
    )


def _dense_model(**dtype_kwargs) -> ModelSpec:
    return ModelSpec(
        hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
        vocab=1000, seq_len=64, layers=[LayerKind.DENSE],
        **dtype_kwargs,
    )


def _find_op(graph, name_substring: str):
    """Return the first non-cast op whose name contains the substring.

    Cast ops (kind="cast") inherit their consumer's name as a prefix, so a
    naive substring match would also match them. We skip cast ops to give
    callers the canonical "main" op they are looking for.
    """
    for op in graph.ops:
        if op.kind == "cast":
            continue
        if name_substring in op.name:
            return op
    raise AssertionError(f"non-cast op containing {name_substring!r} not found")


def _find_collective(graph, name_substring: str):
    for c in graph.collectives:
        if name_substring in c.name:
            return c
    raise AssertionError(f"collective containing {name_substring!r} not found")


# ── BF16 baseline must be unchanged ──────────────────────────────────────


def test_bf16_baseline_all_attn_tensors_bf16():
    m = _moe_model()
    g = build_graph(m, Strategy())
    qkv = _find_op(g, "qkv_proj")
    assert qkv.inputs[0].dtype is Dtype.BF16
    assert qkv.outputs[0].dtype is Dtype.BF16
    o_proj = _find_op(g, "o_proj")
    assert o_proj.outputs[0].dtype is Dtype.BF16


def test_bf16_baseline_all_moe_tensors_bf16():
    m = _moe_model()
    g = build_graph(m, Strategy())
    rmoe = _find_op(g, "routed_expert_ffn")
    assert rmoe.inputs[0].dtype is Dtype.BF16
    assert rmoe.outputs[0].dtype is Dtype.BF16
    agg = _find_op(g, "expert_agg")
    assert agg.outputs[0].dtype is Dtype.BF16


def test_bf16_baseline_ln_dtype_uniform():
    m = _moe_model()
    g = build_graph(m, Strategy())
    ln1 = _find_op(g, "L0.ln1")
    # input and output should both be BF16 in baseline
    assert ln1.inputs[0].dtype is Dtype.BF16
    assert ln1.outputs[0].dtype is Dtype.BF16


# ── FP8 MoE region splits attention region ───────────────────────────────


def test_fp8_moe_region_split_attention_stays_bf16():
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3)
    g = build_graph(m, Strategy())
    # Attention path unchanged
    qkv = _find_op(g, "qkv_proj")
    assert qkv.inputs[0].dtype is Dtype.BF16
    assert qkv.outputs[0].dtype is Dtype.BF16
    # MoE path uses FP8
    rmoe = _find_op(g, "routed_expert_ffn")
    assert rmoe.inputs[0].dtype is Dtype.FP8_E4M3
    assert rmoe.outputs[0].dtype is Dtype.FP8_E4M3
    # x_ln2 produced by ln2 epilog = FP8
    ln2 = _find_op(g, "L0.ln2")
    assert ln2.outputs[0].dtype is Dtype.FP8_E4M3
    # But ln2 input is residual (BF16) → boundary will be picked up by cast_pass
    assert ln2.inputs[0].dtype is Dtype.BF16


def test_fp8_attn_region_split_moe_stays_bf16():
    m = _moe_model(attn_act_dtype=Dtype.FP8_E4M3)
    g = build_graph(m, Strategy())
    qkv = _find_op(g, "qkv_proj")
    assert qkv.inputs[0].dtype is Dtype.FP8_E4M3
    # MoE region untouched
    rmoe = _find_op(g, "routed_expert_ffn")
    assert rmoe.inputs[0].dtype is Dtype.BF16
    # ln1 boundary: residual BF16 in → FP8 out
    ln1 = _find_op(g, "L0.ln1")
    assert ln1.inputs[0].dtype is Dtype.BF16
    assert ln1.outputs[0].dtype is Dtype.FP8_E4M3


def test_residual_add_inputs_cast_to_residual_dtype_under_moe_quant():
    """After cast_pass, residual2 add inputs are both in residual dtype.

    The original ffn_out was FP8 (MoE region); cast_pass inserts a
    ``cast`` op that converts it to BF16 before the residual add. Both
    sides of the add end up at residual dtype. The cast op proves the
    boundary was detected.
    """
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3)
    g = build_graph(m, Strategy())
    res2 = _find_op(g, "residual2")
    # Both inputs are now in residual dtype (BF16) thanks to cast_pass.
    for t in res2.inputs:
        assert t.dtype is Dtype.BF16
    # Output goes back to residual stream → BF16
    assert res2.outputs[0].dtype is Dtype.BF16
    # A cast op was spliced — it converts FP8 → BF16 for the consumer
    # ``residual2``.
    found_cast = False
    for op in g.ops:
        if op.kind == "cast" and "residual2" in op.name:
            found_cast = True
            assert op.meta["src_dtype"] is Dtype.FP8_E4M3
            assert op.meta["dst_dtype"] is Dtype.BF16
            break
    assert found_cast, "expected a cast op spliced before residual2"


# ── Dense block: attention region only ──────────────────────────────────


def test_dense_block_attn_region_splits_ffn():
    m = _dense_model(attn_act_dtype=Dtype.FP8_E4M3)
    g = build_graph(m, Strategy())
    qkv = _find_op(g, "qkv_proj")
    assert qkv.outputs[0].dtype is Dtype.FP8_E4M3
    # Dense FFN stays on act_dtype
    up = _find_op(g, "up_proj")
    assert up.inputs[0].dtype is Dtype.BF16


# ── Collective payload scales with region dtype ──────────────────────────


def test_ep_a2a_bytes_use_moe_act_dtype():
    base = _moe_model()
    fp8 = _moe_model(moe_act_dtype=Dtype.FP8_E4M3)
    strat = Strategy(ep=8, dp=1, tp=1, cp=1, pp=1, micro_batch=1)

    # build_graph requires world_size match for validate(), but
    # insert_collectives is called from build_graph itself. We bypass
    # validate by going through build_graph directly.
    g_base = build_graph(base, strat)
    g_fp8 = build_graph(fp8, strat)

    a2a_base = _find_collective(g_base, "a2a_before_L0.routed_expert_ffn")
    a2a_fp8 = _find_collective(g_fp8, "a2a_before_L0.routed_expert_ffn")
    # FP8 = half the bytes of BF16
    assert a2a_fp8.bytes_ == a2a_base.bytes_ // 2


def test_cp_a2a_bytes_use_attn_act_dtype():
    from zrt.training.spec.strategy import CPKind
    base = _moe_model()
    fp8 = _moe_model(attn_act_dtype=Dtype.FP8_E4M3)
    strat = Strategy(cp=4, dp=1, tp=1, pp=1, ep=1, cp_kind=CPKind.ULYSSES,
                     micro_batch=1)
    g_base = build_graph(base, strat)
    g_fp8 = build_graph(fp8, strat)
    a2a_base = _find_collective(g_base, "a2a_fwd_before_L0.attn_core")
    a2a_fp8 = _find_collective(g_fp8, "a2a_fwd_before_L0.attn_core")
    assert a2a_fp8.bytes_ == a2a_base.bytes_ // 2


def test_tp_ag_qkv_uses_attn_act_dtype():
    base = _moe_model()
    fp8 = _moe_model(attn_act_dtype=Dtype.FP8_E4M3)
    strat = Strategy(tp=2, dp=1, cp=1, pp=1, ep=1, micro_batch=1)
    g_base = build_graph(base, strat)
    g_fp8 = build_graph(fp8, strat)
    ag_base = _find_collective(g_base, "ag_L0.qkv_proj")
    ag_fp8 = _find_collective(g_fp8, "ag_L0.qkv_proj")
    assert ag_fp8.bytes_ == ag_base.bytes_ // 2


def test_tp_ag_in_moe_layer_uses_moe_act_dtype_for_ffn():
    """In an MoE layer the FFN-side TP collectives wrap shared experts →
    moe_act_dtype controls AG bytes for shared up/down proj."""
    base = _moe_model()
    fp8 = _moe_model(moe_act_dtype=Dtype.FP8_E4M3)
    strat = Strategy(tp=2, dp=1, cp=1, pp=1, ep=1, micro_batch=1)
    g_base = build_graph(base, strat)
    g_fp8 = build_graph(fp8, strat)
    # FFN-side TP AG fires only when matmul name contains 'up_proj'.
    # In MoE block, that op is shared_up_proj.
    ag_base = _find_collective(g_base, "ag_L0.shared_up_proj")
    ag_fp8 = _find_collective(g_fp8, "ag_L0.shared_up_proj")
    assert ag_fp8.bytes_ == ag_base.bytes_ // 2
