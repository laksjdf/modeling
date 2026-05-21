"""Per-op dtype resolution for byte and FLOPs accounting.

See ``docs/docs-pri/mixed_quant_v2_op_bytes_zh.md`` §3 for the full design.

This module is the single source of truth for "what dtype does each operand
of an op live in" — replacing the ad-hoc ``inputs[0].dtype`` lookups and
the ``_resolve_compute_dtype`` helper in ``compose/stage.py``.

Downstream callers:
  - ``models/flops.py::_matmul_cost`` / ``_attn_cost`` (per-operand bytes)
  - ``compose/stage.py::_resolve_compute_dtype`` (thin wrapper → ``.compute``)
  - ``ir/cast_pass.py`` (decides dtype-boundary insertion sites)
"""

from __future__ import annotations

from dataclasses import dataclass

from zrt.training.ir.training_graph import Op
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import ModelSpec


@dataclass(frozen=True)
class OpDtypeBundle:
    """Seven-slot dtype resolution for one op.

    Slots:
      in_act      activation read from HBM (A in fwd, dC in bwd-dx input)
      weight      parameter matrix storage form (FP4 includes block-scale via stored_bytes)
      out_act     activation written back to HBM (C in fwd)
      compute     math-unit tile dtype (selects peak_tflops_for)
      grad_in     dC — output-side gradient flowing into bwd
      grad_weight dW write-back dtype (master grad)
      grad_act    dA — input-side gradient produced by bwd-dx
    """
    in_act:      Dtype
    weight:      Dtype
    out_act:     Dtype
    compute:     Dtype
    grad_in:     Dtype
    grad_weight: Dtype
    grad_act:    Dtype


def _attention_bundle(model: ModelSpec) -> OpDtypeBundle:
    # Bwd activation/grad bytes default to in_act — matches v1's implicit
    # assumption (op_cost used ``inputs[0].dtype`` for all six byte slots).
    # The FP32 master grad write traffic is amortized into optimizer time,
    # not counted per-op here. ``attn_grad_dtype`` controls the stored grad
    # footprint in ``memory_breakdown``, NOT op-level dW bytes.
    a = model.effective_attn_act_dtype()
    return OpDtypeBundle(
        in_act=a,
        weight=model.attn_weight_dtype,
        out_act=a,
        compute=model.attn_compute_dtype,
        grad_in=a,
        grad_weight=a,
        grad_act=a,
    )


def _routed_expert_bundle(model: ModelSpec) -> OpDtypeBundle:
    a = model.effective_moe_act_dtype()
    return OpDtypeBundle(
        in_act=a,
        weight=model.routed_expert_weight_dtype,
        out_act=a,
        compute=model.routed_expert_compute_dtype,
        # Bwd activations (dC, dA) flow at the MoE region's activation
        # dtype — under mixed FP8 this is FP8, halving bwd activation
        # bytes. dW write also tracks in_act (FP32 master accumulator is
        # modeled separately by ``memory_breakdown``).
        grad_in=a,
        grad_weight=a,
        grad_act=a,
    )


def _shared_expert_bundle(model: ModelSpec) -> OpDtypeBundle:
    a = model.effective_moe_act_dtype()
    return OpDtypeBundle(
        in_act=a,
        weight=model.shared_expert_weight_dtype,
        out_act=a,
        compute=model.shared_expert_compute_dtype,
        grad_in=a,
        grad_weight=a,
        grad_act=a,
    )


def _embedding_norm_bundle(model: ModelSpec) -> OpDtypeBundle:
    # Embedding / norm are forced to BF16 — matches the hard-coding in
    # the v1 ``_resolve_compute_dtype`` (stage.py:149-150). They have
    # negligible FLOPs and quantizing them is numerically unstable.
    return OpDtypeBundle(
        in_act=Dtype.BF16,
        weight=Dtype.BF16,
        out_act=Dtype.BF16,
        compute=Dtype.BF16,
        grad_in=Dtype.BF16,
        grad_weight=Dtype.BF16,
        grad_act=Dtype.BF16,
    )


def _cast_bundle(op: Op, model: ModelSpec) -> OpDtypeBundle:
    """Cast op's dtypes live entirely in ``op.meta``.

    ``compute`` is set to ``dst_dtype`` so ``peak_tflops_for`` picks the
    realistic unit; in practice cast op has 0 FLOPs so this only affects
    diagnostics in ``mfu_native``.
    """
    src = op.meta.get("src_dtype", model.act_dtype)
    dst = op.meta.get("dst_dtype", model.act_dtype)
    return OpDtypeBundle(
        in_act=src,
        weight=src,                 # unused; cast has no weight
        out_act=dst,
        compute=dst,
        grad_in=dst,
        grad_weight=dst,
        grad_act=src,
    )


def _default_bundle(model: ModelSpec) -> OpDtypeBundle:
    a = model.act_dtype
    return OpDtypeBundle(
        in_act=a,
        weight=model.param_dtype,
        out_act=a,
        compute=a,
        grad_in=a,
        grad_weight=a,
        grad_act=a,
    )


def resolve_op_dtypes(op: Op, model: ModelSpec) -> OpDtypeBundle:
    """Return the dtype bundle for one op.

    Dispatches by ``op.component`` — the tag that ``build_graph`` assigns
    to every op in builders.py. Unknown / unset components fall back to the
    model's global ``act_dtype`` / ``param_dtype`` / ``grad_dtype``.
    """
    comp = getattr(op, "component", None)

    if op.kind == "cast":
        return _cast_bundle(op, model)
    if comp == "attention":
        return _attention_bundle(model)
    if comp == "routed_expert":
        return _routed_expert_bundle(model)
    if comp == "shared_expert":
        return _shared_expert_bundle(model)
    if comp in ("embedding", "norm"):
        return _embedding_norm_bundle(model)
    return _default_bundle(model)


def expected_input_dtype(op: Op, ti: int, model: ModelSpec) -> Dtype:
    """Dtype that ``op.inputs[ti]`` must arrive in.

    Used by ``cast_pass`` to decide where to insert cast ops. Defaults to
    ``bundle.in_act`` but special-cases multi-input ops where operand
    dtypes differ.
    """
    bundle = resolve_op_dtypes(op, model)

    if op.kind == "add":
        # Residual-add boundary: both sides must arrive in the residual
        # dtype (BF16/FP32), regardless of which region produced them.
        # ``component`` is set to "routed_expert" on expert_agg adds in
        # builders.py:528, so we treat any add inside an expert region as
        # a residual-add boundary.
        if op.component in {"routed_expert", "shared_expert"}:
            return model.effective_residual_dtype()
        return bundle.in_act

    if op.kind in ("dispatch", "combine"):
        # Token routing happens in MoE-region activation dtype.
        return model.effective_moe_act_dtype()

    if op.kind in ("mhc_pre", "mhc_post", "mhc_head"):
        # Hyper-Connection internals: mhc_pre emits sinkhorn intermediates
        # (hc_post, hc_comb) in FP32 deliberately; mhc_post consumes them
        # in FP32. These tensors stay inside the HC sub-network and never
        # cross a region boundary. Trust the tensor's producer-defined
        # dtype rather than forcing it back to the bundle's act dtype.
        if 0 <= ti < len(op.inputs):
            return op.inputs[ti].dtype
        return bundle.in_act

    # matmul / attn_core / softmax / ln / rmsnorm / rope / swiglu read
    # activations in their region's act dtype.
    return bundle.in_act
