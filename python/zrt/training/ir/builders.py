"""IR block builders — construct per-layer op lists from ModelSpec geometry."""

from __future__ import annotations

from zrt.training.ir.graph import Graph, Op, Tensor
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import Strategy
from zrt.training.ir.shard import ShardPlan, insert_collectives


def _tensor(name: str, shape: tuple[int, ...], dtype: Dtype,
            is_activation: bool = True, is_param: bool = False) -> Tensor:
    return Tensor(name=name, shape_logical=shape, shape_local=shape,
                  dtype=dtype, is_activation=is_activation, is_param=is_param)


def dense_block(
    hidden: int,
    ffn: int,
    seq: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    layer_id: int,
    act_dtype: Dtype = Dtype.BF16,
) -> list[Op]:
    """Build ops for one dense transformer block (pre-norm / RMSNorm style).

    Produces ~13 ops:
      LN, QKV_proj, RoPE, attn_core, O_proj, add(residual),
      LN, up_proj, gate_proj, swiglu, down_proj, add(residual)
    """
    ops: list[Op] = []
    b = 1  # batch handled at tensor level; micro_batch applied in memory/flops
    h = hidden
    h_attn = num_heads * head_dim
    h_kv = num_kv_heads * head_dim
    prefix = f"L{layer_id}"

    # ── Pre-attention RMSNorm ──────────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.ln1", kind="ln",
        inputs=[_tensor("x", (seq, h), act_dtype)],
        outputs=[_tensor("x_ln1", (seq, h), act_dtype)],
        meta={"bytes_fwd": seq * h * act_dtype.bytes * 2},  # read + write
        layer_id=layer_id, layer_kind=LayerKind.DENSE,
    ))

    # ── QKV projection ────────────────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.qkv_proj", kind="matmul",
        inputs=[_tensor("x_ln1", (seq, h), act_dtype)],
        outputs=[_tensor("qkv", (seq, h_attn + 2 * h_kv), act_dtype)],
        meta={"m": seq, "n": h_attn + 2 * h_kv, "k": h},
        layer_id=layer_id, layer_kind=LayerKind.DENSE,
    ))

    # ── RoPE ───────────────────────────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.rope", kind="rope",
        inputs=[_tensor("q", (seq, h_attn), act_dtype),
                _tensor("k", (seq, h_kv), act_dtype)],
        outputs=[_tensor("q_rope", (seq, h_attn), act_dtype),
                 _tensor("k_rope", (seq, h_kv), act_dtype)],
        meta={"bytes_fwd": seq * (h_attn + h_kv) * act_dtype.bytes * 2},
        layer_id=layer_id, layer_kind=LayerKind.DENSE,
    ))

    # ── Attention core (flash-attn, causal) ───────────────────────────────
    ops.append(Op(
        name=f"{prefix}.attn_core", kind="attn_core",
        inputs=[_tensor("q_rope", (seq, h_attn), act_dtype),
                _tensor("k_rope", (seq, h_kv), act_dtype),
                _tensor("v", (seq, h_kv), act_dtype)],
        outputs=[_tensor("attn_out", (seq, h_attn), act_dtype)],
        meta={
            "b": b, "s": seq,
            "heads": num_heads, "head_dim": head_dim,
            "causal": True,
            "h_kv": h_kv,
        },
        layer_id=layer_id, layer_kind=LayerKind.DENSE,
    ))

    # ── O projection ──────────────────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.o_proj", kind="matmul",
        inputs=[_tensor("attn_out", (seq, h_attn), act_dtype)],
        outputs=[_tensor("attn_proj", (seq, h), act_dtype)],
        meta={"m": seq, "n": h, "k": h_attn},
        layer_id=layer_id, layer_kind=LayerKind.DENSE,
    ))

    # ── Residual add ──────────────────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.residual1", kind="add",
        inputs=[_tensor("attn_proj", (seq, h), act_dtype),
                _tensor("x", (seq, h), act_dtype)],
        outputs=[_tensor("x_attn", (seq, h), act_dtype)],
        meta={"bytes_fwd": seq * h * act_dtype.bytes * 3},  # 2 read + 1 write
        layer_id=layer_id, layer_kind=LayerKind.DENSE,
    ))

    # ── Post-attention RMSNorm ────────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.ln2", kind="ln",
        inputs=[_tensor("x_attn", (seq, h), act_dtype)],
        outputs=[_tensor("x_ln2", (seq, h), act_dtype)],
        meta={"bytes_fwd": seq * h * act_dtype.bytes * 2},
        layer_id=layer_id, layer_kind=LayerKind.DENSE,
    ))

    # ── FFN up projection ─────────────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.up_proj", kind="matmul",
        inputs=[_tensor("x_ln2", (seq, h), act_dtype)],
        outputs=[_tensor("up", (seq, ffn), act_dtype)],
        meta={"m": seq, "n": ffn, "k": h},
        layer_id=layer_id, layer_kind=LayerKind.DENSE,
    ))

    # ── FFN gate projection ───────────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.gate_proj", kind="matmul",
        inputs=[_tensor("x_ln2", (seq, h), act_dtype)],
        outputs=[_tensor("gate", (seq, ffn), act_dtype)],
        meta={"m": seq, "n": ffn, "k": h},
        layer_id=layer_id, layer_kind=LayerKind.DENSE,
    ))

    # ── SwiGLU activation ─────────────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.swiglu", kind="swiglu",
        inputs=[_tensor("up", (seq, ffn), act_dtype),
                _tensor("gate", (seq, ffn), act_dtype)],
        outputs=[_tensor("swiglu_out", (seq, ffn), act_dtype)],
        meta={"bytes_fwd": seq * ffn * act_dtype.bytes * 3},
        layer_id=layer_id, layer_kind=LayerKind.DENSE,
    ))

    # ── FFN down projection ───────────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.down_proj", kind="matmul",
        inputs=[_tensor("swiglu_out", (seq, ffn), act_dtype)],
        outputs=[_tensor("ffn_out", (seq, h), act_dtype)],
        meta={"m": seq, "n": h, "k": ffn},
        layer_id=layer_id, layer_kind=LayerKind.DENSE,
    ))

    # ── Residual add ──────────────────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.residual2", kind="add",
        inputs=[_tensor("ffn_out", (seq, h), act_dtype),
                _tensor("x_attn", (seq, h), act_dtype)],
        outputs=[_tensor("y", (seq, h), act_dtype)],
        meta={"bytes_fwd": seq * h * act_dtype.bytes * 3},
        layer_id=layer_id, layer_kind=LayerKind.DENSE,
    ))

    return ops


def _embed_op(vocab: int, hidden: int, seq: int, act_dtype: Dtype) -> Op:
    return Op(
        name="embed", kind="embed",
        inputs=[_tensor("input_ids", (seq,), act_dtype)],
        outputs=[_tensor("x_embed", (seq, hidden), act_dtype)],
        meta={"m": seq, "n": hidden, "k": vocab},
        layer_id=-1, layer_kind=LayerKind.DENSE,
    )


def _lm_head_op(vocab: int, hidden: int, seq: int, act_dtype: Dtype) -> Op:
    return Op(
        name="lm_head", kind="lm_head",
        inputs=[_tensor("x_final", (seq, hidden), act_dtype)],
        outputs=[_tensor("logits", (seq, vocab), act_dtype)],
        meta={"m": seq, "n": vocab, "k": hidden},
        layer_id=-1, layer_kind=LayerKind.DENSE,
    )


def _final_ln_op(hidden: int, seq: int, act_dtype: Dtype) -> Op:
    return Op(
        name="final_ln", kind="ln",
        inputs=[_tensor("x_final_raw", (seq, hidden), act_dtype)],
        outputs=[_tensor("x_final", (seq, hidden), act_dtype)],
        meta={"bytes_fwd": seq * hidden * act_dtype.bytes * 2},
        layer_id=-1, layer_kind=LayerKind.DENSE,
    )


def build_graph(model: ModelSpec, strategy: Strategy) -> Graph:
    """Build the full IR from ModelSpec + Strategy.

    Iterates over model.layers, calls the appropriate block builder,
    then applies sharding and inserts collectives.
    """
    all_ops: list[Op] = []
    layer_index: dict[int, tuple[int, int]] = {}
    h = model.hidden
    s = model.seq_len
    act_dtype = model.act_dtype

    # Embedding
    all_ops.append(_embed_op(model.vocab, h, s, act_dtype))

    # Transformer blocks
    for i, lk in enumerate(model.layers):
        start = len(all_ops)
        if lk == LayerKind.DENSE:
            block_ops = dense_block(
                hidden=h, ffn=model.ffn, seq=s,
                num_heads=model.num_heads,
                num_kv_heads=model.num_kv_heads,
                head_dim=model.head_dim,
                layer_id=i, act_dtype=act_dtype,
            )
        elif lk == LayerKind.MOE:
            # Phase 2: moe_block()
            block_ops = dense_block(
                hidden=h, ffn=model.ffn, seq=s,
                num_heads=model.num_heads,
                num_kv_heads=model.num_kv_heads,
                head_dim=model.head_dim,
                layer_id=i, act_dtype=act_dtype,
            )
        elif lk == LayerKind.MTP:
            # Phase 2: mtp_block()
            block_ops = dense_block(
                hidden=h, ffn=model.ffn, seq=s,
                num_heads=model.num_heads,
                num_kv_heads=model.num_kv_heads,
                head_dim=model.head_dim,
                layer_id=i, act_dtype=act_dtype,
            )
        else:
            raise ValueError(f"Unknown LayerKind: {lk}")

        all_ops.extend(block_ops)
        layer_index[i] = (start, len(all_ops))

    # Final LN + lm_head
    all_ops.append(_final_ln_op(h, s, act_dtype))
    all_ops.append(_lm_head_op(model.vocab, h, s, act_dtype))

    graph = Graph(ops=all_ops, collectives=[], layer_index=layer_index)

    # Apply sharding and insert collectives
    shard = ShardPlan(strategy)
    insert_collectives(graph, shard, model)

    return graph
