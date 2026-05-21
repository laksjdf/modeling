"""Cast op insertion pass — splice dtype-boundary cast ops into the IR.

Scans ``graph.ops`` in order; for every input tensor whose dtype does not
match the consumer's expected input dtype (resolved via
``models.quant.expected_input_dtype``), inserts a ``kind="cast"`` op between
producer and consumer.

The cast op carries:
  ``op.meta = {
      "num_elements":    <int>,
      "src_dtype":       <Dtype>,
      "dst_dtype":       <Dtype>,
      "fused":           <bool>,   # decided per-site by QuantPolicy
      "needs_amax":      <bool>,   # True when going from high → quantized
      "site":            <str>,    # for diagnostics: ln_epilog / gemm_epilog / ...
      "adjacent_op_name": <str>,
  }``

Layer attribution: cast inherits ``layer_id`` and ``layer_kind`` from the
*consumer* — the cast physically happens at the consumer's HBM read, so
this is the correct stage assignment for PP slicing. After insertion the
graph's ``layer_index`` is rebuilt.

cast_pass runs AFTER builders construct the transformer ops and BEFORE
``insert_collectives`` runs, because:
  - collectives bind to matmul names (``inserted_after=<matmul.name>``);
    cast ops don't conflict with that, so adding casts before this step
    is harmless;
  - if cast_pass ran after, casts spliced between matmul and collective
    insertion sites could disturb the (AG → matmul → RS) wrapping
    invariant the collective inserter depends on.

The single entry point is :func:`insert_cast_pass`.
"""

from __future__ import annotations

from zrt.training.ir.training_graph import Graph, Op, Tensor
from zrt.training.models.quant import expected_input_dtype, resolve_op_dtypes
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.strategy import QuantPolicy


# Dtypes that require an amax reduction when used as cast destination.
_QUANT_DTYPES = {Dtype.FP4, Dtype.FP8_E4M3, Dtype.FP8_E5M2}


def insert_cast_pass(
    graph: Graph, model: ModelSpec, quant: QuantPolicy | None = None,
) -> None:
    """Splice cast ops at dtype boundaries IN-PLACE.

    ``quant`` defaults to ``QuantPolicy()`` (all casts fused → 0 cost) so
    callers that don't yet supply a Strategy get v1 behaviour.
    """
    if quant is None:
        quant = QuantPolicy()

    # Track each tensor's actual current dtype (producer-defined).
    producer_dtype: dict[str, Dtype] = {}
    producer_kind: dict[str, str] = {}
    for op in graph.ops:
        for t in op.outputs:
            producer_dtype[t.name] = t.dtype
            producer_kind[t.name] = op.kind

    new_ops: list[Op] = []
    cast_counter = 0
    for op in graph.ops:
        if op.kind == "cast":
            # Idempotent: skip if cast_pass has already run on this graph.
            new_ops.append(op)
            continue
        new_inputs: list[Tensor] = []
        for ti, t in enumerate(op.inputs):
            need = expected_input_dtype(op, ti, model)
            have = producer_dtype.get(t.name, t.dtype)
            if have == need:
                new_inputs.append(t)
                continue

            # dtype boundary → splice a cast.
            cast_counter += 1
            site = _classify_site(op, producer_kind.get(t.name, ""))
            fused = quant.is_fused_at(site)
            needs_amax = need in _QUANT_DTYPES and have not in _QUANT_DTYPES
            cast = _make_cast_op(
                src_name=t.name, src_dtype=have, dst_dtype=need,
                shape_logical=t.shape_logical, shape_local=t.shape_local,
                consumer=op, site=site, fused=fused, needs_amax=needs_amax,
                cast_id=cast_counter,
            )
            new_ops.append(cast)
            new_inputs.append(cast.outputs[0])
            producer_dtype[cast.outputs[0].name] = need
            producer_kind[cast.outputs[0].name] = "cast"

        op.inputs = new_inputs
        new_ops.append(op)
        # Update producer_dtype map for downstream ops in case this op
        # already updated some tensor dtypes.
        for t in op.outputs:
            producer_dtype[t.name] = t.dtype
            producer_kind[t.name] = op.kind

    graph.ops = new_ops
    _rebuild_layer_index(graph)


def _classify_site(consumer: Op, producer_kind: str) -> str:
    """Decide which QuantPolicy fuse_* flag applies to this boundary.

    The classification is conservative: when in doubt, treat the cast as
    "other" (unfused unless ``assume_all_casts_fused``).
    """
    # GEMM input prefaced by LN/RMSNorm → LN epilog typically fuses the
    # cast right into its output stage.
    if consumer.kind == "matmul" and producer_kind in {"ln", "rmsnorm"}:
        return "ln_epilog"
    # GEMM output going into LN/swiglu/etc → GEMM epilog fuses the cast.
    if producer_kind == "matmul":
        return "gemm_epilog"
    # Internal attention chain (Q/K/V/O all attention component) →
    # FlashAttention tile-local.
    if consumer.component == "attention" and consumer.kind in {
        "attn_core", "sparse_attn", "hca_attn", "swa_attn",
        "rope", "ln", "rmsnorm",
    }:
        return "attn_internal"
    return "other"


def _make_cast_op(
    src_name: str, src_dtype: Dtype, dst_dtype: Dtype,
    shape_logical, shape_local,
    consumer: Op, site: str, fused: bool, needs_amax: bool, cast_id: int,
) -> Op:
    out_name = f"{src_name}__cast_{dst_dtype.value}_{cast_id}"
    out_t = Tensor(
        name=out_name, shape_logical=shape_logical, shape_local=shape_local,
        dtype=dst_dtype, is_activation=True,
    )
    # Reuse the producer's tensor as input — share by name so downstream
    # passes that match on tensor identity still work.
    in_t = Tensor(
        name=src_name, shape_logical=shape_logical, shape_local=shape_local,
        dtype=src_dtype, is_activation=True,
    )
    n = 1
    for dim in shape_local:
        n *= int(dim) if dim else 1

    return Op(
        name=f"{consumer.name}.cast_{cast_id}_{src_dtype.value}_to_{dst_dtype.value}",
        kind="cast",
        inputs=[in_t],
        outputs=[out_t],
        meta={
            "num_elements": int(n),
            "src_dtype": src_dtype,
            "dst_dtype": dst_dtype,
            "fused": bool(fused),
            "needs_amax": bool(needs_amax),
            "site": site,
            "adjacent_op_name": consumer.name,
        },
        # Cast happens at the consumer's HBM read → inherit layer_id from
        # consumer so PP stage assignment is right.
        layer_id=consumer.layer_id,
        layer_kind=consumer.layer_kind,
        component="cast",
    )


def _rebuild_layer_index(graph: Graph) -> None:
    """Recompute ``graph.layer_index`` from current op order.

    Each layer's [start, end) range covers ALL ops whose ``layer_id``
    matches — including freshly spliced cast ops. Global ops (embed,
    final_ln, lm_head, hc_expand, mhc_head) keep ``layer_id == -1`` and
    are excluded from the index, matching v1 behaviour.
    """
    new_idx: dict[int, tuple[int, int]] = {}
    current_lid: int | None = None
    start = 0
    for i, op in enumerate(graph.ops):
        lid = op.layer_id
        if lid < 0:
            # Flush current run and reset.
            if current_lid is not None:
                new_idx[current_lid] = (start, i)
                current_lid = None
            continue
        if current_lid is None:
            current_lid = lid
            start = i
        elif lid != current_lid:
            new_idx[current_lid] = (start, i)
            current_lid = lid
            start = i
    if current_lid is not None:
        new_idx[current_lid] = (start, len(graph.ops))
    graph.layer_index = new_idx
