"""IR block builders — construct per-layer op lists from ModelSpec geometry."""

from __future__ import annotations

from zrt.training.ir.training_graph import Graph, Op, Tensor
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import Strategy
from zrt.training.ir.shard import ShardPlan, insert_collectives


def _tensor(name: str, shape: tuple[int, ...], dtype: Dtype,
            is_activation: bool = True, is_param: bool = False) -> Tensor:
    return Tensor(name=name, shape_logical=shape, shape_local=shape,
                  dtype=dtype, is_activation=is_activation, is_param=is_param)


def _mhc_pre_op(seq: int, hidden: int, hc_mult: int, sinkhorn_iters: int,
                layer_id: int, layer_kind: LayerKind, prefix: str, suffix: str,
                act_dtype: Dtype) -> Op:
    """One Hyper-Connections pre-mix op.

    Inputs:  x_hc (seq, hc, h)
    Outputs: x_pre (seq, h) + post (seq, hc) + comb (seq, hc, hc)
    """
    mix_hc = (2 + hc_mult) * hc_mult
    return Op(
        name=f"{prefix}.mhc_pre_{suffix}", kind="mhc_pre",
        inputs=[_tensor(f"x_hc_{suffix}", (seq, hc_mult, hidden), act_dtype)],
        outputs=[
            _tensor(f"x_pre_{suffix}", (seq, hidden), act_dtype),
            _tensor(f"hc_post_{suffix}", (seq, hc_mult), Dtype.FP32),
            _tensor(f"hc_comb_{suffix}", (seq, hc_mult, hc_mult), Dtype.FP32),
        ],
        meta={"b": 1, "s": seq, "h": hidden, "hc": hc_mult,
              "mix_hc": mix_hc, "sinkhorn_iters": sinkhorn_iters},
        layer_id=layer_id, layer_kind=layer_kind,
    )


def _mhc_post_op(seq: int, hidden: int, hc_mult: int, layer_id: int,
                 layer_kind: LayerKind, prefix: str, suffix: str,
                 act_dtype: Dtype) -> Op:
    """One Hyper-Connections post-mix op.

    Inputs:  x_sub (seq, h) + residual (seq, hc, h) + post (seq, hc) + comb (seq, hc, hc)
    Outputs: x_out (seq, hc, h)
    """
    return Op(
        name=f"{prefix}.mhc_post_{suffix}", kind="mhc_post",
        inputs=[
            _tensor(f"x_sub_{suffix}", (seq, hidden), act_dtype),
            _tensor(f"x_res_{suffix}", (seq, hc_mult, hidden), act_dtype),
            _tensor(f"hc_post_{suffix}", (seq, hc_mult), Dtype.FP32),
            _tensor(f"hc_comb_{suffix}", (seq, hc_mult, hc_mult), Dtype.FP32),
        ],
        outputs=[_tensor(f"x_hc_out_{suffix}", (seq, hc_mult, hidden), act_dtype)],
        meta={"b": 1, "s": seq, "h": hidden, "hc": hc_mult},
        layer_id=layer_id, layer_kind=layer_kind,
    )


def dense_block(
    hidden: int,
    ffn: int,
    seq: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    layer_id: int,
    act_dtype: Dtype = Dtype.BF16,
    hc_mult: int = 1,
    hc_sinkhorn_iters: int = 20,
) -> list[Op]:
    """Build ops for one dense transformer block (pre-norm / RMSNorm style).

    Without HC (hc_mult <= 1): ~13 ops
      LN, QKV_proj, RoPE, attn_core, O_proj, add(residual),
      LN, up_proj, gate_proj, swiglu, down_proj, add(residual)

    With HC (hc_mult > 1): residual adds are replaced by (mhc_pre, mhc_post)
    pairs, and the block's input/output is (seq, hc, h) instead of (seq, h).
    """
    use_hc = hc_mult > 1
    ops: list[Op] = []
    b = 1  # batch handled at tensor level; micro_batch applied in memory/flops
    h = hidden
    h_attn = num_heads * head_dim
    h_kv = num_kv_heads * head_dim
    prefix = f"L{layer_id}"

    # ── HC pre-attn (hc_mult > 1 only) ─────────────────────────────────────
    # Replaces the implicit "block input is the residual" assumption: when HC
    # is on, the block input has shape (seq, hc, h) and gets mixed down to
    # (seq, h) here before the attn pre-norm.
    if use_hc:
        ops.append(_mhc_pre_op(
            seq, h, hc_mult, hc_sinkhorn_iters,
            layer_id, LayerKind.DENSE, prefix, "attn", act_dtype,
        ))
        ln1_in = _tensor(f"x_pre_attn", (seq, h), act_dtype)
    else:
        ln1_in = _tensor("x", (seq, h), act_dtype)

    # ── Pre-attention RMSNorm ──────────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.ln1", kind="ln",
        inputs=[ln1_in],
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

    # ── Residual add OR mhc_post_attn ─────────────────────────────────────
    if use_hc:
        ops.append(_mhc_post_op(
            seq, h, hc_mult, layer_id, LayerKind.DENSE, prefix, "attn", act_dtype,
        ))
        # Output is (seq, hc, h) — fed directly into mhc_pre_ffn next.
        ops.append(_mhc_pre_op(
            seq, h, hc_mult, hc_sinkhorn_iters,
            layer_id, LayerKind.DENSE, prefix, "ffn", act_dtype,
        ))
        ln2_in = _tensor(f"x_pre_ffn", (seq, h), act_dtype)
    else:
        ops.append(Op(
            name=f"{prefix}.residual1", kind="add",
            inputs=[_tensor("attn_proj", (seq, h), act_dtype),
                    _tensor("x", (seq, h), act_dtype)],
            outputs=[_tensor("x_attn", (seq, h), act_dtype)],
            meta={"bytes_fwd": seq * h * act_dtype.bytes * 3},  # 2 read + 1 write
            layer_id=layer_id, layer_kind=LayerKind.DENSE,
        ))
        ln2_in = _tensor("x_attn", (seq, h), act_dtype)

    # ── Post-attention RMSNorm ────────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.ln2", kind="ln",
        inputs=[ln2_in],
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

    # ── Residual add OR mhc_post_ffn ──────────────────────────────────────
    if use_hc:
        ops.append(_mhc_post_op(
            seq, h, hc_mult, layer_id, LayerKind.DENSE, prefix, "ffn", act_dtype,
        ))
    else:
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


def _hc_expand_op(seq: int, hidden: int, hc_mult: int, act_dtype: Dtype) -> Op:
    """Expand embed output (seq, h) → (seq, hc, h) by replication.

    Pure reshape/repeat; FLOPs = 0 but activation memory grows by hc_mult.
    Mirrors ``inference.Transformer.forward``::

        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
    """
    return Op(
        name="hc_expand", kind="hc_expand",
        inputs=[_tensor("x_embed", (seq, hidden), act_dtype)],
        outputs=[_tensor("x_hc_init", (seq, hc_mult, hidden), act_dtype)],
        meta={"b": 1, "s": seq, "h": hidden, "hc": hc_mult,
              "bytes_fwd": seq * hidden * hc_mult * act_dtype.bytes},
        layer_id=-1, layer_kind=LayerKind.DENSE,
    )


def _moe_block(
    hidden: int,
    ffn: int,
    moe_ffn: int,
    num_experts: int,
    top_k: int,
    n_shared_experts: int,
    seq: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    layer_id: int,
    act_dtype: Dtype = Dtype.BF16,
    hc_mult: int = 1,
    hc_sinkhorn_iters: int = 20,
) -> list[Op]:
    """Build ops for one MoE (Mixture-of-Experts) transformer block.

    MoE block structure:
      - Attention (same as dense)
      - Router: h -> num_experts (gating logits)
      - Shared expert FFN: runs on all tokens (if n_shared_experts > 0)
      - Routed expert FFN: top-k experts per token
      - Expert output aggregation

    Without HC (hc_mult <= 1): ~18 ops
      LN, QKV_proj, RoPE, attn_core, O_proj, add(residual),
      LN, router, shared FFN (if n_shared > 0),
      routed expert FFN (num_experts × top_k),
      expert aggregation, add(residual)

    With HC (hc_mult > 1): residual adds are replaced by (mhc_pre, mhc_post)
    pairs, similar to dense_block.
    """
    use_hc = hc_mult > 1
    ops: list[Op] = []
    b = 1  # batch handled at tensor level
    h = hidden
    h_attn = num_heads * head_dim
    h_kv = num_kv_heads * head_dim
    prefix = f"L{layer_id}"

    # ── HC pre-attn (hc_mult > 1 only) ─────────────────────────────────────
    if use_hc:
        ops.append(_mhc_pre_op(
            seq, h, hc_mult, hc_sinkhorn_iters,
            layer_id, LayerKind.MOE, prefix, "attn", act_dtype,
        ))
        ln1_in = _tensor(f"x_pre_attn", (seq, h), act_dtype)
    else:
        ln1_in = _tensor("x", (seq, h), act_dtype)

    # ── Pre-attention RMSNorm ──────────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.ln1", kind="ln",
        inputs=[ln1_in],
        outputs=[_tensor("x_ln1", (seq, h), act_dtype)],
        meta={"bytes_fwd": seq * h * act_dtype.bytes * 2},
        layer_id=layer_id, layer_kind=LayerKind.MOE,
    ))

    # ── QKV projection ────────────────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.qkv_proj", kind="matmul",
        inputs=[_tensor("x_ln1", (seq, h), act_dtype)],
        outputs=[_tensor("qkv", (seq, h_attn + 2 * h_kv), act_dtype)],
        meta={"m": seq, "n": h_attn + 2 * h_kv, "k": h},
        layer_id=layer_id, layer_kind=LayerKind.MOE,
    ))

    # ── RoPE ───────────────────────────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.rope", kind="rope",
        inputs=[_tensor("q", (seq, h_attn), act_dtype),
                _tensor("k", (seq, h_kv), act_dtype)],
        outputs=[_tensor("q_rope", (seq, h_attn), act_dtype),
                 _tensor("k_rope", (seq, h_kv), act_dtype)],
        meta={"bytes_fwd": seq * (h_attn + h_kv) * act_dtype.bytes * 2},
        layer_id=layer_id, layer_kind=LayerKind.MOE,
    ))

    # ── Attention core ─────────────────────────────────────────────────────
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
        layer_id=layer_id, layer_kind=LayerKind.MOE,
    ))

    # ── O projection ──────────────────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.o_proj", kind="matmul",
        inputs=[_tensor("attn_out", (seq, h_attn), act_dtype)],
        outputs=[_tensor("attn_proj", (seq, h), act_dtype)],
        meta={"m": seq, "n": h, "k": h_attn},
        layer_id=layer_id, layer_kind=LayerKind.MOE,
    ))

    # ── Residual add OR mhc_post_attn ─────────────────────────────────────
    if use_hc:
        ops.append(_mhc_post_op(
            seq, h, hc_mult, layer_id, LayerKind.MOE, prefix, "attn", act_dtype,
        ))
        ops.append(_mhc_pre_op(
            seq, h, hc_mult, hc_sinkhorn_iters,
            layer_id, LayerKind.MOE, prefix, "ffn", act_dtype,
        ))
        ln2_in = _tensor(f"x_pre_ffn", (seq, h), act_dtype)
    else:
        ops.append(Op(
            name=f"{prefix}.residual1", kind="add",
            inputs=[_tensor("attn_proj", (seq, h), act_dtype),
                    _tensor("x", (seq, h), act_dtype)],
            outputs=[_tensor("x_attn", (seq, h), act_dtype)],
            meta={"bytes_fwd": seq * h * act_dtype.bytes * 3},
            layer_id=layer_id, layer_kind=LayerKind.MOE,
        ))
        ln2_in = _tensor("x_attn", (seq, h), act_dtype)

    # ── Post-attention RMSNorm ────────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.ln2", kind="ln",
        inputs=[ln2_in],
        outputs=[_tensor("x_ln2", (seq, h), act_dtype)],
        meta={"bytes_fwd": seq * h * act_dtype.bytes * 2},
        layer_id=layer_id, layer_kind=LayerKind.MOE,
    ))

    # ── Router: h -> num_experts (gating logits) ─────────────────────────────
    ops.append(Op(
        name=f"{prefix}.router", kind="matmul",
        inputs=[_tensor("x_ln2", (seq, h), act_dtype)],
        outputs=[_tensor("router_logits", (seq, num_experts), act_dtype)],
        meta={"m": seq, "n": num_experts, "k": h},
        layer_id=layer_id, layer_kind=LayerKind.MOE,
    ))

    # ── Top-k selection (modeled as memory-bound op) ───────────────────────
    ops.append(Op(
        name=f"{prefix}.topk_select", kind="softmax",
        inputs=[_tensor("router_logits", (seq, num_experts), act_dtype)],
        outputs=[_tensor("topk_weights", (seq, top_k), act_dtype),
                 _tensor("topk_indices", (seq, top_k), act_dtype)],
        meta={
            "bytes_fwd": seq * num_experts * act_dtype.bytes * 2,
            "num_experts": num_experts,
            "top_k": top_k,
        },
        layer_id=layer_id, layer_kind=LayerKind.MOE,
    ))

    # ── Shared expert FFN (if n_shared_experts > 0) ────────────────────────
    if n_shared_experts > 0:
        ops.append(Op(
            name=f"{prefix}.shared_up_proj", kind="matmul",
            inputs=[_tensor("x_ln2", (seq, h), act_dtype)],
            outputs=[_tensor("shared_up", (seq, moe_ffn), act_dtype)],
            meta={"m": seq, "n": moe_ffn, "k": h},
            layer_id=layer_id, layer_kind=LayerKind.MOE,
        ))
        ops.append(Op(
            name=f"{prefix}.shared_gate_proj", kind="matmul",
            inputs=[_tensor("x_ln2", (seq, h), act_dtype)],
            outputs=[_tensor("shared_gate", (seq, moe_ffn), act_dtype)],
            meta={"m": seq, "n": moe_ffn, "k": h},
            layer_id=layer_id, layer_kind=LayerKind.MOE,
        ))
        ops.append(Op(
            name=f"{prefix}.shared_swiGLU", kind="swiglu",
            inputs=[_tensor("shared_up", (seq, moe_ffn), act_dtype),
                    _tensor("shared_gate", (seq, moe_ffn), act_dtype)],
            outputs=[_tensor("shared_swiglu_out", (seq, moe_ffn), act_dtype)],
            meta={"bytes_fwd": seq * moe_ffn * act_dtype.bytes * 3},
            layer_id=layer_id, layer_kind=LayerKind.MOE,
        ))
        ops.append(Op(
            name=f"{prefix}.shared_down_proj", kind="matmul",
            inputs=[_tensor("shared_swiglu_out", (seq, moe_ffn), act_dtype)],
            outputs=[_tensor("shared_ffn_out", (seq, h), act_dtype)],
            meta={"m": seq, "n": h, "k": moe_ffn},
            layer_id=layer_id, layer_kind=LayerKind.MOE,
        ))

    # ── Routed expert FFN (modeled as single op with fwd_multiplier) ─────────────
    # Each token is routed to top_k experts, each expert runs a full FFN.
    # Total FLOPs = top_k × (up + gate + down) expert FFNs
    # We model this as a single matmul with fwd_multiplier = 3 * top_k
    # (3 for up_proj + gate_proj + down_proj per expert)
    ops.append(Op(
        name=f"{prefix}.routed_expert_ffn", kind="matmul",
        inputs=[_tensor("x_ln2", (seq, h), act_dtype)],
        outputs=[_tensor("routed_ffn_out", (seq, h), act_dtype)],
        meta={
            "m": seq,
            "n": h,
            "k": moe_ffn,
            "fwd_multiplier": 3 * top_k,  # 3 matmuls per active expert
        },
        layer_id=layer_id, layer_kind=LayerKind.MOE,
    ))

    # ── Expert aggregation (combine shared + routed outputs) ─────────────────
    if n_shared_experts > 0:
        ops.append(Op(
            name=f"{prefix}.expert_agg", kind="add",
            inputs=[_tensor("shared_ffn_out", (seq, h), act_dtype),
                    _tensor("routed_ffn_out", (seq, h), act_dtype)],
            outputs=[_tensor("ffn_out", (seq, h), act_dtype)],
            meta={"bytes_fwd": seq * h * act_dtype.bytes * 3},
            layer_id=layer_id, layer_kind=LayerKind.MOE,
        ))
    else:
        # No shared experts: routed output is the FFN output
        ops.append(Op(
            name=f"{prefix}.expert_agg", kind="add",
            inputs=[_tensor("routed_ffn_out", (seq, h), act_dtype)],
            outputs=[_tensor("ffn_out", (seq, h), act_dtype)],
            meta={"bytes_fwd": seq * h * act_dtype.bytes * 2},
            layer_id=layer_id, layer_kind=LayerKind.MOE,
        ))

    # ── Residual add OR mhc_post_ffn ──────────────────────────────────────
    if use_hc:
        ops.append(_mhc_post_op(
            seq, h, hc_mult, layer_id, LayerKind.MOE, prefix, "ffn", act_dtype,
        ))
    else:
        ops.append(Op(
            name=f"{prefix}.residual2", kind="add",
            inputs=[_tensor("ffn_out", (seq, h), act_dtype),
                    _tensor("x_attn", (seq, h), act_dtype)],
            outputs=[_tensor("y", (seq, h), act_dtype)],
            meta={"bytes_fwd": seq * h * act_dtype.bytes * 3},
            layer_id=layer_id, layer_kind=LayerKind.MOE,
        ))

    return ops


def _mtp_block(
    hidden: int,
    ffn: int,
    seq: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    layer_id: int,
    act_dtype: Dtype = Dtype.BF16,
    hc_mult: int = 1,
    hc_sinkhorn_iters: int = 20,
) -> list[Op]:
    """Build ops for one MTP (Multimodal Projection) transformer block.

    MTP block structure:
      - Dense transformer block (same as dense_block)
      - Plus an additional embedding projection layer

    This is used for multimodal models that need to project embeddings
    between different modalities (e.g., vision-language models).

    Without HC (hc_mult <= 1): ~15 ops
      Embedding projection, LN, QKV_proj, RoPE, attn_core, O_proj, add,
      LN, up_proj, gate_proj, swiglu, down_proj, add

    With HC (hc_mult > 1): residual adds replaced by mhc_pre/mhc_post.
    """
    # MTP is essentially a dense block with an extra embedding projection
    dense_ops = dense_block(
        hidden=hidden,
        ffn=ffn,
        seq=seq,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        layer_id=layer_id,
        act_dtype=act_dtype,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=hc_sinkhorn_iters,
    )

    # Prepend embedding projection for MTP
    prefix = f"L{layer_id}"
    h = hidden

    # Embedding projection: h -> h (modal fusion)
    embed_proj = Op(
        name=f"{prefix}.mtp_embed_proj", kind="matmul",
        inputs=[_tensor("x_mtp_in", (seq, h), act_dtype)],
        outputs=[_tensor("x_proj", (seq, h), act_dtype)],
        meta={"m": seq, "n": h, "k": h},
        layer_id=layer_id, layer_kind=LayerKind.MTP,
    )

    return [embed_proj] + dense_ops


def _embed_op(vocab: int, hidden: int, seq: int, act_dtype: Dtype) -> Op:
    return Op(
        name="embed", kind="embed",
        inputs=[_tensor("input_ids", (seq,), act_dtype)],
        outputs=[_tensor("x_embed", (seq, hidden), act_dtype)],
        meta={"m": seq, "n": hidden, "k": vocab},
        layer_id=-1, layer_kind=LayerKind.DENSE,
    )


def _mhc_head_op(seq: int, hidden: int, hc_mult: int, act_dtype: Dtype) -> Op:
    """Final HC mix-down before final_ln + lm_head.

    Mirrors ``ParallelHead.hc_head``: project (seq, hc, h) → mixes (seq, hc),
    sigmoid, weighted sum to (seq, h).  No sinkhorn, no comb output (used only
    for routing into the head, not back into a residual stream).
    """
    return Op(
        name="mhc_head", kind="mhc_head",
        inputs=[_tensor("x_hc_final", (seq, hc_mult, hidden), act_dtype)],
        outputs=[_tensor("x_collapsed", (seq, hidden), act_dtype)],
        meta={"b": 1, "s": seq, "h": hidden, "hc": hc_mult, "mix_hc": hc_mult},
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

    hc_mult = max(1, getattr(model, "hc_mult", 1))
    hc_iters = getattr(model, "hc_sinkhorn_iters", 20)

    # Embedding
    all_ops.append(_embed_op(model.vocab, h, s, act_dtype))

    # Optional HC expansion: (seq, h) → (seq, hc, h)
    if hc_mult > 1:
        all_ops.append(_hc_expand_op(s, h, hc_mult, act_dtype))

    # Transformer blocks
    block_kwargs = dict(
        hidden=h, ffn=model.ffn, seq=s,
        num_heads=model.num_heads,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
        act_dtype=act_dtype,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=hc_iters,
    )
    for i, lk in enumerate(model.layers):
        start = len(all_ops)
        if lk == LayerKind.DENSE:
            block_ops = dense_block(layer_id=i, **block_kwargs)
        elif lk == LayerKind.MOE:
            block_ops = _moe_block(
                hidden=h, ffn=model.ffn, moe_ffn=model.moe_ffn,
                num_experts=model.num_experts, top_k=model.top_k,
                n_shared_experts=model.n_shared_experts,
                seq=s, num_heads=model.num_heads,
                num_kv_heads=model.num_kv_heads,
                head_dim=model.head_dim,
                layer_id=i, act_dtype=act_dtype,
                hc_mult=hc_mult, hc_sinkhorn_iters=hc_iters,
            )
        elif lk == LayerKind.MTP:
            block_ops = _mtp_block(
                hidden=h, ffn=model.ffn, seq=s,
                num_heads=model.num_heads,
                num_kv_heads=model.num_kv_heads,
                head_dim=model.head_dim,
                layer_id=i, act_dtype=act_dtype,
                hc_mult=hc_mult, hc_sinkhorn_iters=hc_iters,
            )
        else:
            raise ValueError(f"Unknown LayerKind: {lk}")

        all_ops.extend(block_ops)
        layer_index[i] = (start, len(all_ops))

    # Optional HC head mix-down before final_ln (only when HC active)
    if hc_mult > 1:
        all_ops.append(_mhc_head_op(s, h, hc_mult, act_dtype))

    # Final LN + lm_head
    all_ops.append(_final_ln_op(h, s, act_dtype))
    all_ops.append(_lm_head_op(model.vocab, h, s, act_dtype))

    graph = Graph(ops=all_ops, collectives=[], layer_index=layer_index)

    # Apply sharding and insert collectives
    shard = ShardPlan(strategy)
    insert_collectives(graph, model, strategy)

    return graph
