"""Tests for cast_pass (Stage D2/D4) and QuantPolicy fusion toggle.

Verifies:
  - BF16 baseline: no cast ops inserted
  - Region quant: cast op inserted at residual-add boundary (FP8/FP4 →
    residual stream)
  - cast op metadata fields populated correctly
  - cast op carries layer_id from consumer (PP-aware)
  - layer_index rebuilt correctly after insertion
  - QuantPolicy fused vs unfused changes cast cost only
  - `assume_all_casts_fused=true` (default) returns zero cost
  - `assume_all_casts_fused=false` materializes cast HBM bytes
"""
from __future__ import annotations

import pytest

from zrt.training.ir.builders import build_graph
from zrt.training.ir.cast_pass import insert_cast_pass
from zrt.training.ir.training_graph import Graph, Op, Tensor
from zrt.training.models.flops import _cast_cost, op_cost
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import QuantPolicy, Strategy


def _moe_model(**kw) -> ModelSpec:
    return ModelSpec(
        hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
        vocab=1000, seq_len=64, layers=[LayerKind.MOE],
        num_experts=8, moe_ffn=256, top_k=2, n_shared_experts=1,
        **kw,
    )


def _dense_model(**kw) -> ModelSpec:
    return ModelSpec(
        hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
        vocab=1000, seq_len=64, layers=[LayerKind.DENSE], **kw,
    )


def _count_casts(g: Graph) -> int:
    return sum(1 for op in g.ops if op.kind == "cast")


# ── BF16 baseline: zero casts ─────────────────────────────────────────────


def test_bf16_baseline_inserts_no_casts():
    m = _moe_model()
    g = build_graph(m, Strategy())
    assert _count_casts(g) == 0


def test_dense_block_baseline_inserts_no_casts():
    m = _dense_model()
    g = build_graph(m, Strategy())
    assert _count_casts(g) == 0


# ── Region quant triggers cast insertion ─────────────────────────────────


def test_moe_quant_inserts_cast_before_residual_add():
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3)
    g = build_graph(m, Strategy())
    casts = [op for op in g.ops if op.kind == "cast"]
    assert len(casts) >= 1
    # At least one of them targets residual2 add
    assert any("residual2" in c.name for c in casts)


def test_attn_quant_inserts_cast_before_residual1():
    m = _moe_model(attn_act_dtype=Dtype.FP8_E4M3)
    g = build_graph(m, Strategy())
    assert any(
        op.kind == "cast" and "residual1" in op.name for op in g.ops
    ), "expected cast before residual1 when attn region is FP8"


# ── Cast op metadata is correct ──────────────────────────────────────────


def test_cast_metadata_records_src_dst_and_amax():
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3)
    g = build_graph(m, Strategy())
    cast = next(op for op in g.ops
                if op.kind == "cast" and "residual2" in op.name)
    assert cast.meta["src_dtype"] is Dtype.FP8_E4M3
    assert cast.meta["dst_dtype"] is Dtype.BF16
    # FP8 → BF16 is dequant: no amax needed
    assert cast.meta["needs_amax"] is False
    assert cast.meta["num_elements"] > 0


def test_cast_needs_amax_when_quantizing_to_low_precision():
    """LN1 epilog produces moe_act dtype when MoE quantization is on.
    If we artificially force a quantize cast (BF16 → FP8 somewhere), the
    cast meta should set needs_amax=True.
    """
    g = Graph(
        ops=[
            Op(name="prod", kind="rmsnorm",
               inputs=[Tensor(name="x", shape_logical=(64, 128),
                              shape_local=(64, 128), dtype=Dtype.BF16,
                              is_activation=True)],
               outputs=[Tensor(name="y", shape_logical=(64, 128),
                               shape_local=(64, 128), dtype=Dtype.BF16,
                               is_activation=True)],
               meta={}, layer_id=0, layer_kind=LayerKind.MOE, component="norm"),
            Op(name="cons", kind="matmul",
               inputs=[Tensor(name="y", shape_logical=(64, 128),
                              shape_local=(64, 128), dtype=Dtype.BF16,
                              is_activation=True)],
               outputs=[Tensor(name="z", shape_logical=(64, 256),
                               shape_local=(64, 256), dtype=Dtype.FP8_E4M3,
                               is_activation=True)],
               meta={"m": 64, "n": 256, "k": 128},
               layer_id=0, layer_kind=LayerKind.MOE, component="routed_expert"),
        ],
        layer_index={0: (0, 2)},
    )
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3,
                   routed_expert_compute_dtype=Dtype.FP8_E4M3)
    insert_cast_pass(g, m, QuantPolicy(assume_all_casts_fused=False))
    casts = [op for op in g.ops if op.kind == "cast"]
    assert len(casts) == 1
    c = casts[0]
    assert c.meta["src_dtype"] is Dtype.BF16
    assert c.meta["dst_dtype"] is Dtype.FP8_E4M3
    assert c.meta["needs_amax"] is True


# ── Layer attribution: cast inherits from consumer ───────────────────────


def test_cast_op_inherits_consumer_layer_id():
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3)
    g = build_graph(m, Strategy())
    for cast in (op for op in g.ops if op.kind == "cast"):
        # No cast op should have layer_id = -1 (those are global ops)
        assert cast.layer_id >= 0


def test_layer_index_rebuilt_after_cast_insertion():
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3)
    g = build_graph(m, Strategy())
    # layer 0 range should still cover its first transformer op through
    # its last op (incl. spliced casts).
    assert 0 in g.layer_index
    start, end = g.layer_index[0]
    assert end > start


# ── _cast_cost: fused vs unfused ─────────────────────────────────────────


def test_fused_cast_is_zero_cost():
    cast = Op(
        name="L0.cast_fused", kind="cast", component="cast",
        inputs=[], outputs=[],
        meta={"num_elements": 100_000, "src_dtype": Dtype.BF16,
              "dst_dtype": Dtype.FP8_E4M3, "fused": True,
              "needs_amax": True},
        layer_id=0, layer_kind=LayerKind.MOE,
    )
    cost = _cast_cost(cast)
    assert cost.fwd_bytes == 0
    assert cost.dx_bytes == 0
    assert cost.dw_bytes == 0


def test_unfused_cast_fp8_to_bf16_no_amax():
    cast = Op(
        name="L0.cast_unfused", kind="cast", component="cast",
        inputs=[], outputs=[],
        meta={"num_elements": 100_000, "src_dtype": Dtype.FP8_E4M3,
              "dst_dtype": Dtype.BF16, "fused": False,
              "needs_amax": False},
        layer_id=0, layer_kind=LayerKind.MOE,
    )
    cost = _cast_cost(cast)
    # n * (FP8 + BF16) = 100_000 * (1 + 2) = 300_000
    assert cost.fwd_bytes == 300_000
    assert cost.dx_bytes == 300_000
    assert cost.dw_bytes == 0


def test_unfused_cast_with_amax_adds_extra_read():
    cast = Op(
        name="L0.cast_amax", kind="cast", component="cast",
        inputs=[], outputs=[],
        meta={"num_elements": 100_000, "src_dtype": Dtype.BF16,
              "dst_dtype": Dtype.FP8_E4M3, "fused": False,
              "needs_amax": True},
        layer_id=0, layer_kind=LayerKind.MOE,
    )
    cost = _cast_cost(cast)
    # main = 100_000 * (BF16 + FP8) = 100_000 * (2 + 1) = 300_000
    # amax = 100_000 * BF16 = 200_000
    assert cost.fwd_bytes == 500_000


# ── QuantPolicy toggle wired through build_graph ─────────────────────────


def test_quant_policy_assume_fused_makes_all_casts_zero():
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3)
    strat = Strategy(quant=QuantPolicy(assume_all_casts_fused=True))
    g = build_graph(m, strat)
    casts = [op for op in g.ops if op.kind == "cast"]
    assert len(casts) > 0
    for c in casts:
        assert c.meta["fused"] is True
        assert _cast_cost(c).fwd_bytes == 0


def test_quant_policy_unfused_materializes_cast_bytes():
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3)
    strat = Strategy(quant=QuantPolicy(
        assume_all_casts_fused=False,
        fuse_ln_epilog=False,
        fuse_gemm_epilog=False,
        fuse_attn_internal=False,
    ))
    g = build_graph(m, strat)
    casts = [op for op in g.ops if op.kind == "cast"]
    assert len(casts) > 0
    total = sum(_cast_cost(c).fwd_bytes for c in casts)
    assert total > 0


def test_quant_policy_partial_fusion():
    """fuse_gemm_epilog=True absorbs GEMM-output cast but not residual-add."""
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3)
    strat = Strategy(quant=QuantPolicy(
        assume_all_casts_fused=False,
        fuse_ln_epilog=True,
        fuse_gemm_epilog=True,
        fuse_attn_internal=True,
    ))
    g = build_graph(m, strat)
    # residual2's cast comes from a "other" site → unfused
    cast = next(op for op in g.ops
                if op.kind == "cast" and "residual2" in op.name)
    assert cast.meta["fused"] is False


# ── Idempotency: running cast_pass again is a no-op ──────────────────────


def test_cast_pass_idempotent():
    m = _moe_model(moe_act_dtype=Dtype.FP8_E4M3)
    g = build_graph(m, Strategy())
    n_casts_first = _count_casts(g)
    # Re-run cast_pass: existing casts should pass through unchanged.
    insert_cast_pass(g, m, QuantPolicy())
    assert _count_casts(g) == n_casts_first
