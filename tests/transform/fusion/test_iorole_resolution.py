"""Tests for rule-based IO resolution and child_ops-based fallback.

Verifies that ``node.inputs`` / ``node.outputs`` on fused nodes carry
placeholder tensors (model weights, ``input_ids``) that have no producer
``OpNode`` — the bug fixed by switching ``build_fused_node`` from
edge-based ``_external_io`` to ``resolve_io_tensors`` /
``_child_ops_external_io``.
"""
from __future__ import annotations

from python.zrt.ir.node import OpNode
from python.zrt.ir.types import DType, TensorMeta
from python.zrt.transform.fusion.building.io_resolver import (
    _child_ops_external_io,
    resolve_io_tensors,
)
from python.zrt.transform.fusion.core.io_role import IORole
from python.zrt.transform.fusion.core.rule import ModuleFusionRule


def _t(tid: str, shape=(7168,), dtype=DType.BF16) -> TensorMeta:
    return TensorMeta.from_shape_dtype(tid, tuple(shape), dtype)


def _op(nid: str, op_type: str, inputs, outputs) -> OpNode:
    return OpNode(id=nid, op_type=op_type, inputs=list(inputs),
                  outputs=list(outputs))


def _rule(name: str, io_roles) -> ModuleFusionRule:
    return ModuleFusionRule(
        target_class="*",
        op_type=name,
        name=name,
        io_roles=tuple(io_roles),
    )


# ─── resolve_io_tensors ───────────────────────────────────────────────────


def test_resolve_io_tensors_picks_correct_slots():
    """rule.io_roles drives slot picks; placeholders are captured."""
    weight = _t("w", (7168,))
    x = _t("x", (1, 128, 7168), DType.FP32)
    y = _t("y", (1, 128, 7168), DType.FP32)
    pow_op = _op("op_0", "aten.pow.Tensor_Scalar", [x], [_t("p", (1,128,7168), DType.FP32)])
    mul_op = _op("op_5", "aten.mul.Tensor", [weight, x], [y])
    child_ops = [pow_op, mul_op]

    rule = _rule("rms_norm", [
        IORole(role="activation", source_kind="input",
               source_op_index=0, source_arg_index=0),
        IORole(role="weight", source_kind="input",
               source_op_index=-1, source_arg_index=0),
        IORole(role="output", source_kind="output",
               source_op_index=-1, source_arg_index=-1),
    ])
    ins, outs = resolve_io_tensors(child_ops, rule)
    assert [t.shape for t in ins] == [(1, 128, 7168), (7168,)]
    assert [t.shape for t in outs] == [(1, 128, 7168)]


def test_resolve_io_tensors_dedups_by_id():
    """Same tensor.id appearing in multiple roles is deduped."""
    shared = _t("shared", (4, 8))
    other = _t("other", (4,))
    op = _op("op_0", "aten.mul.Tensor", [shared, other],
             [_t("out", (4, 8))])
    rule = _rule("rule", [
        IORole(role="a", source_kind="input",
               source_op_index=0, source_arg_index=0),
        IORole(role="b", source_kind="input",
               source_op_index=0, source_arg_index=0),  # same tensor
    ])
    ins, _ = resolve_io_tensors([op], rule)
    assert len(ins) == 1
    assert ins[0].id == "shared"


def test_resolve_io_tensors_handles_out_of_range_gracefully():
    """Bad indices log a warning, do not raise."""
    op = _op("op_0", "aten.mm.default", [_t("a", (1, 8))], [_t("out", (1, 4))])
    rule = _rule("rule", [
        IORole(role="weight", source_kind="input",
               source_op_index=-1, source_arg_index=5),  # out of range
    ])
    ins, outs = resolve_io_tensors([op], rule)
    assert ins == [] and outs == []


# ─── _child_ops_external_io fallback ──────────────────────────────────────


def test_child_ops_external_io_captures_placeholders():
    """Tensors not produced by any child op (weights, input_ids) are external inputs."""
    weight = _t("w", (129280, 7168))
    input_ids = _t("ids", (1, 128), DType.INT64)
    emb_out = _t("emb_out", (1, 128, 7168))
    op = _op("op_0", "aten.embedding.default",
             [weight, input_ids], [emb_out])

    ins, outs = _child_ops_external_io([op])
    # Both placeholders captured; no producer edge required.
    assert {t.id for t in ins} == {"w", "ids"}
    assert {t.id for t in outs} == {"emb_out"}


def test_child_ops_external_io_excludes_internal_flow():
    """Tensors produced and consumed inside the group are internal."""
    a = _t("a", (1, 8))
    b = _t("b", (1, 8))
    c = _t("c", (1, 8))
    d = _t("d", (1, 8))
    op1 = _op("op_0", "aten.add.Tensor", [a, b], [c])  # a, b external; c internal
    op2 = _op("op_1", "aten.mul.Tensor", [c, a], [d])  # c internal; a re-used; d external

    ins, outs = _child_ops_external_io([op1, op2])
    assert {t.id for t in ins} == {"a", "b"}     # c excluded — produced internally
    assert {t.id for t in outs} == {"d"}         # c excluded — consumed internally


def test_child_ops_external_io_terminal_product_fallback():
    """All outputs consumed internally → fall back to last child's outputs."""
    a = _t("a", (1, 8))
    b = _t("b", (1, 8))
    op1 = _op("op_0", "aten.pow.Tensor_Scalar", [a], [b])
    op2 = _op("op_1", "aten.rsqrt.default", [b], [_t("coef", (1, 8))])

    # Synthetic case: pretend op_1's coef is consumed by op_0 (cyclical, just
    # to force "all outputs consumed internally").  Easier path: re-use a
    # group where every output is also a child's input.
    op3 = _op("op_2", "aten.identity.default", [_t("coef", (1, 8))], [a])
    ins, outs = _child_ops_external_io([op1, op2, op3])
    # `a` is produced internally by op_2 AND consumed by op_0 → not external.
    # `coef` produced by op_2 → consumed by op_3.  Terminal-product fallback
    # surfaces op_2 (last child)'s outputs.
    assert outs, "outputs must not be empty after terminal-product fallback"


def test_child_ops_external_io_empty_input_group():
    """Empty group returns empty lists."""
    ins, outs = _child_ops_external_io([])
    assert ins == [] and outs == []
