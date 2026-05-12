"""Tests for python/zrt/transform/fusion/semantics.py.

Each test builds a *minimal* set of OpNode + ModuleFusionRule objects and
exercises one slice of the derivation pipeline (resolve_io_views,
derive_shape_axes, compute_flops, compute_memory, annotate_fused_node).
No real OpGraph capture is required.
"""
from __future__ import annotations

import logging

import pytest

from python.zrt.ir.node import OpNode
from python.zrt.ir.types import DType, TensorMeta
from python.zrt.transform.fusion.core.io_role import IORole, ShapeDerivation
from python.zrt.transform.fusion.core.rule import ModuleFusionRule
from python.zrt.transform.fusion.semantics import (
    TensorView,
    annotate_fused_node,
    compute_flops,
    compute_memory,
    derive_shape_axes,
    resolve_io_views,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _t(tid: str, shape: tuple[int, ...], dtype: DType = DType.BF16) -> TensorMeta:
    return TensorMeta.from_shape_dtype(tid, shape, dtype)


def _make_rmsnorm_group():
    """Mimic an RMSNorm fusion group: 2 inputs (activation [B,S,H] +
    weight [H]) coming in on the first child op, 1 output on the last
    child op.  Two child ops is enough."""
    op0 = OpNode(
        id="op0",
        op_type="aten.pow.Tensor_Scalar",
        inputs=[_t("a", (2, 128, 4096)), _t("w", (4096,))],
        outputs=[_t("o0", (2, 128, 4096))],
    )
    op1 = OpNode(
        id="op1",
        op_type="aten.mul.Tensor",
        inputs=[_t("o0", (2, 128, 4096)), _t("w", (4096,))],
        outputs=[_t("y", (2, 128, 4096))],
    )
    return [op0, op1]


def _rmsnorm_rule(**overrides) -> ModuleFusionRule:
    base = dict(
        target_class="RMSNorm",
        op_type="rms_norm",
        io_roles=(
            IORole(role="activation", source_kind="input",
                   source_op_index=0, source_arg_index=0,
                   shape_role="[B,S,H]"),
            IORole(role="weight", source_kind="input",
                   source_op_index=-1, source_arg_index=1,
                   shape_role="[H]"),
            IORole(role="output", source_kind="output",
                   source_op_index=-1, source_arg_index=-1,
                   shape_role="[B,S,H]"),
        ),
        shape_derivation=ShapeDerivation(
            batch_size="activation.shape[0]",
            seq_len="activation.shape[1]",
            hidden_in="activation.shape[-1]",
        ),
    )
    base.update(overrides)
    return ModuleFusionRule(**base)


def _fused_node() -> OpNode:
    return OpNode(id="fused_0", op_type="rms_norm")


# ─────────────────────────────────────────────────────────────────────────────
# 1. resolve_io_views — basic
# ─────────────────────────────────────────────────────────────────────────────

def test_resolve_io_views_basic():
    child_ops = _make_rmsnorm_group()
    rule = _rmsnorm_rule()
    views = resolve_io_views(child_ops, rule)

    assert set(views) == {"activation", "weight", "output"}
    assert views["activation"].shape == (2, 128, 4096)
    assert views["activation"].dtype == "bf16"
    assert views["activation"].numel == 2 * 128 * 4096
    # bf16 is 2 bytes per element.
    assert views["activation"].itemsize == 2.0
    assert views["activation"].bytes == 2 * 128 * 4096 * 2

    assert views["weight"].shape == (4096,)
    assert views["output"].shape == (2, 128, 4096)


# ─────────────────────────────────────────────────────────────────────────────
# 2. resolve_io_views — out-of-bounds index is skipped, not raised
# ─────────────────────────────────────────────────────────────────────────────

def test_resolve_io_views_index_oob_skipped(caplog):
    child_ops = _make_rmsnorm_group()
    rule = ModuleFusionRule(
        target_class="RMSNorm",
        op_type="rms_norm",
        io_roles=(
            IORole(role="activation", source_op_index=0, source_arg_index=0),
            IORole(role="weight",     source_op_index=42, source_arg_index=0),  # OOB op
            IORole(role="bias",       source_op_index=0,  source_arg_index=99), # OOB arg
        ),
    )
    with caplog.at_level(logging.WARNING, logger="python.zrt.transform.fusion.semantics"):
        views = resolve_io_views(child_ops, rule)

    assert "activation" in views
    assert "weight" not in views
    assert "bias" not in views
    # Two warnings should have been emitted (one per OOB role).
    msgs = [r.getMessage() for r in caplog.records]
    assert any("source_op_index" in m for m in msgs)
    assert any("source_arg_index" in m for m in msgs)


# ─────────────────────────────────────────────────────────────────────────────
# 3. derive_shape_axes — basic
# ─────────────────────────────────────────────────────────────────────────────

def test_derive_shape_axes_basic():
    child_ops = _make_rmsnorm_group()
    rule = _rmsnorm_rule()
    views = resolve_io_views(child_ops, rule)
    axes = derive_shape_axes(views, rule)

    assert axes["batch_size"] == 2
    assert axes["seq_len"] == 128
    assert axes["hidden_in"] == 4096


# ─────────────────────────────────────────────────────────────────────────────
# 4. derive_shape_axes — extra fields and forward references
# ─────────────────────────────────────────────────────────────────────────────

def test_derive_shape_axes_extra():
    child_ops = _make_rmsnorm_group()
    rule = _rmsnorm_rule(
        shape_derivation=ShapeDerivation(
            batch_size="activation.shape[0]",
            seq_len="activation.shape[1]",
            hidden_in="activation.shape[-1]",
            extra=(
                ("tokens", "batch_size * seq_len"),
                # Reference earlier axis + an io_view simultaneously.
                ("hidden_total", "tokens * hidden_in"),
            ),
        ),
    )
    views = resolve_io_views(child_ops, rule)
    axes = derive_shape_axes(views, rule)

    assert axes["tokens"] == 2 * 128
    assert axes["hidden_total"] == 2 * 128 * 4096


# ─────────────────────────────────────────────────────────────────────────────
# 5. compute_flops — formula path
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_flops_formula():
    child_ops = _make_rmsnorm_group()
    # Pretend Linear-like rule: 2*B*S*Hin*Hout
    rule = ModuleFusionRule(
        target_class="Linear",
        op_type="linear",
        io_roles=(
            IORole(role="activation", source_op_index=0, source_arg_index=0),
            IORole(role="output", source_kind="output",
                   source_op_index=-1, source_arg_index=-1),
        ),
        shape_derivation=ShapeDerivation(
            batch_size="activation.shape[0]",
            seq_len="activation.shape[1]",
            hidden_in="activation.shape[-1]",
            hidden_out="output.shape[-1]",
        ),
        flops_formula="2 * batch_size * seq_len * hidden_in * hidden_out",
    )
    views = resolve_io_views(child_ops, rule)
    axes = derive_shape_axes(views, rule)
    ns = {**views, **axes}

    flops = compute_flops(rule, ns, child_ops)
    expected = 2 * 2 * 128 * 4096 * 4096
    assert flops == float(expected)


# ─────────────────────────────────────────────────────────────────────────────
# 6. compute_flops — callable beats formula
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_flops_callable():
    seen: dict = {}

    def my_flops(ns: dict) -> float:
        seen["called"] = True
        return 12345.0

    child_ops = _make_rmsnorm_group()
    rule = ModuleFusionRule(
        target_class="RMSNorm",
        op_type="rms_norm",
        io_roles=(IORole(role="activation",
                         source_op_index=0, source_arg_index=0),),
        flops_callable=my_flops,
        flops_formula="999 * 999",   # should be ignored
    )
    views = resolve_io_views(child_ops, rule)
    flops = compute_flops(rule, dict(views), child_ops)

    assert seen.get("called") is True
    assert flops == 12345.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. compute_memory — from_io path
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_memory_from_io():
    child_ops = _make_rmsnorm_group()
    # No formula; flops_kind = "from_io" drives both flops AND memory.
    rule = ModuleFusionRule(
        target_class="RMSNorm",
        op_type="rms_norm",
        io_roles=(IORole(role="activation",
                         source_op_index=0, source_arg_index=0),),
        flops_kind="from_io",
    )

    views = resolve_io_views(child_ops, rule)
    mem = compute_memory(rule, dict(views), child_ops)

    # Sum of all child input bytes + output bytes.
    expected = 0
    for op in child_ops:
        for t in op.inputs:
            expected += t.mem_bytes
        for t in op.outputs:
            expected += t.mem_bytes
    assert mem == float(expected)
    assert expected > 0


# ─────────────────────────────────────────────────────────────────────────────
# 8. annotate_fused_node — end-to-end shape of annotations
# ─────────────────────────────────────────────────────────────────────────────

def test_annotate_fused_node_writes_all_keys():
    child_ops = _make_rmsnorm_group()
    rule = _rmsnorm_rule(
        flops_formula="4 * batch_size * seq_len * hidden_in",
        memory_formula="activation.bytes + weight.bytes + output.bytes",
    )
    fused = _fused_node()
    annotate_fused_node(fused, child_ops, rule)

    a = fused.annotations
    # sem_* keys
    assert "sem_io" in a and set(a["sem_io"]) == {"activation", "weight", "output"}
    assert a["sem_io"]["activation"]["shape"] == [2, 128, 4096]
    assert a["sem_io"]["activation"]["dtype"] == "bf16"

    assert "sem_shape" in a
    assert a["sem_shape"]["batch_size"] == 2
    assert a["sem_shape"]["seq_len"] == 128
    assert a["sem_shape"]["hidden_in"] == 4096

    assert a["sem_flops"] == float(4 * 2 * 128 * 4096)
    assert a["sem_memory_bytes"] is not None and a["sem_memory_bytes"] > 0
    assert a["sem_dtype"] == "bf16"

    # Flat layer
    assert a["batch_size"] == 2
    assert a["seq_len"] == 128
    assert a["hidden_in"] == 4096


# ─────────────────────────────────────────────────────────────────────────────
# 9. annotate_fused_node — broken formula does not raise
# ─────────────────────────────────────────────────────────────────────────────

def test_annotate_fused_node_handles_bad_formula_gracefully(caplog):
    child_ops = _make_rmsnorm_group()
    rule = _rmsnorm_rule(
        flops_formula="undefined_var * 2",   # undefined name → FormulaError
    )
    fused = _fused_node()
    with caplog.at_level(logging.WARNING, logger="python.zrt.transform.fusion.semantics"):
        # Must not raise.
        annotate_fused_node(fused, child_ops, rule)

    a = fused.annotations
    # sem_flops should be None because the formula failed.
    assert a.get("sem_flops") is None
    # The other fields still got populated.
    assert "sem_io" in a
    assert a["sem_shape"]["batch_size"] == 2
    # A warning was emitted.
    assert any("flops" in r.getMessage() for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# 10. annotate_fused_node — pre-existing flat annotations are preserved
# ─────────────────────────────────────────────────────────────────────────────

def test_annotate_fused_node_preserves_existing_flat_annotations():
    child_ops = _make_rmsnorm_group()
    rule = _rmsnorm_rule()
    fused = _fused_node()
    # Pretend rule.annotations or an upstream pass already set this.
    fused.annotations["batch_size"] = "from_upstream"

    annotate_fused_node(fused, child_ops, rule)

    # Flat layer must NOT have overwritten the pre-existing value.
    assert fused.annotations["batch_size"] == "from_upstream"
    # But sem_shape (the nested truth) DOES contain the derived value.
    assert fused.annotations["sem_shape"]["batch_size"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# 11. TensorView.from_tensor_meta — sanity
# ─────────────────────────────────────────────────────────────────────────────

def test_tensorview_from_tensor_meta():
    m = _t("x", (4, 8), DType.FP32)
    v = TensorView.from_tensor_meta(m)
    assert v.shape == (4, 8)
    assert v.dtype == "fp32"
    assert v.itemsize == 4.0
    assert v.numel == 32
    assert v.bytes == 32 * 4
