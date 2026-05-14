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


def _norm_kind(model: ModelSpec | None) -> str:
    """Return the norm op kind based on model's norm_kind field.

    Defaults to "rmsnorm" for DeepSeek models, "ln" (LayerNorm) for others.
    """
    if model is None:
        return "ln"  # legacy default
    return getattr(model, "norm_kind", "rmsnorm")


# ── Hyper-Connection ops ───────────────────────────────────────────────

def _mhc_pre_op(seq: int, hidden: int, hc_mult: int, sinkhorn_iters: int,
                layer_id: int, layer_kind: LayerKind, prefix: str, suffix: str,
                act_dtype: Dtype) -> Op:
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


# ── Attention sub-block builders ───────────────────────────────────────

def _build_attn_ops(model: ModelSpec, layer_id: int, seq: int,
                    layer_kind: LayerKind, prefix: str,
                    act_dtype: Dtype) -> list[Op]:
    """Dispatch attention sub-block based on model architecture.

    Returns ops that produce `attn_proj` tensor with shape (seq, hidden).
    """
    if model.use_mla:
        return _build_mla_attn(model, layer_id, seq, layer_kind, prefix, act_dtype)
    elif model.use_v4_attn:
        return _build_v4_attn(model, layer_id, seq, layer_kind, prefix, act_dtype)
    else:
        return _build_standard_attn(model, layer_id, seq, layer_kind, prefix, act_dtype)


def _build_standard_attn(model: ModelSpec, layer_id: int, seq: int,
                         layer_kind: LayerKind, prefix: str,
                         act_dtype: Dtype) -> list[Op]:
    """Standard MHA: fused QKV → RoPE → flash-attn → O projection."""
    h = model.hidden
    h_attn = model.num_heads * model.head_dim
    h_kv = model.num_kv_heads * model.head_dim

    return [
        Op(name=f"{prefix}.qkv_proj", kind="matmul",
           inputs=[_tensor("x_ln1", (seq, h), act_dtype)],
           outputs=[_tensor("qkv", (seq, h_attn + 2 * h_kv), act_dtype)],
           meta={"m": seq, "n": h_attn + 2 * h_kv, "k": h},
           layer_id=layer_id, layer_kind=layer_kind),

        Op(name=f"{prefix}.rope", kind="rope",
           inputs=[_tensor("q", (seq, h_attn), act_dtype),
                   _tensor("k", (seq, h_kv), act_dtype)],
           outputs=[_tensor("q_rope", (seq, h_attn), act_dtype),
                    _tensor("k_rope", (seq, h_kv), act_dtype)],
           meta={"bytes_fwd": seq * (h_attn + h_kv) * act_dtype.bytes * 2},
           layer_id=layer_id, layer_kind=layer_kind),

        Op(name=f"{prefix}.attn_core", kind="attn_core",
           inputs=[_tensor("q_rope", (seq, h_attn), act_dtype),
                   _tensor("k_rope", (seq, h_kv), act_dtype),
                   _tensor("v", (seq, h_kv), act_dtype)],
           outputs=[_tensor("attn_out", (seq, h_attn), act_dtype)],
           meta={"b": 1, "s": seq, "heads": model.num_heads,
                 "head_dim": model.head_dim, "causal": True},
           layer_id=layer_id, layer_kind=layer_kind),

        Op(name=f"{prefix}.o_proj", kind="matmul",
           inputs=[_tensor("attn_out", (seq, h_attn), act_dtype)],
           outputs=[_tensor("attn_proj", (seq, h), act_dtype)],
           meta={"m": seq, "n": h, "k": h_attn},
           layer_id=layer_id, layer_kind=layer_kind),
    ]


def _build_mla_attn(model: ModelSpec, layer_id: int, seq: int,
                    layer_kind: LayerKind, prefix: str,
                    act_dtype: Dtype) -> list[Op]:
    """DeepSeek-V3 MLA: low-rank Q/KV projections → split → attn → o_proj."""
    h = model.hidden
    h_q = model.num_heads * (model.qk_nope_head_dim + model.qk_rope_head_dim)
    h_kv = model.kv_lora_rank + model.qk_rope_head_dim
    h_kv_out = model.num_heads * (model.qk_nope_head_dim + model.v_head_dim)
    h_attn_out = model.num_heads * model.v_head_dim
    n_h = model.num_heads
    qk_rope = model.qk_rope_head_dim
    qk_nope = model.qk_nope_head_dim
    v_hd = model.v_head_dim

    ops: list[Op] = []

    # q_a_proj: h → q_lora_rank
    ops.append(Op(name=f"{prefix}.q_a_proj", kind="matmul",
        inputs=[_tensor("x_ln1", (seq, h), act_dtype)],
        outputs=[_tensor("q_a", (seq, model.q_lora_rank), act_dtype)],
        meta={"m": seq, "n": model.q_lora_rank, "k": h},
        layer_id=layer_id, layer_kind=layer_kind))

    # q_a_layernorm
    ops.append(Op(name=f"{prefix}.q_a_norm", kind=_norm_kind(model),
        inputs=[_tensor("q_a", (seq, model.q_lora_rank), act_dtype)],
        outputs=[_tensor("q_a_normed", (seq, model.q_lora_rank), act_dtype)],
        meta={"bytes_fwd": seq * model.q_lora_rank * act_dtype.bytes * 2},
        layer_id=layer_id, layer_kind=layer_kind))

    # q_b_proj: q_lora_rank → num_heads × (qk_nope + qk_rope)
    ops.append(Op(name=f"{prefix}.q_b_proj", kind="matmul",
        inputs=[_tensor("q_a_normed", (seq, model.q_lora_rank), act_dtype)],
        outputs=[_tensor("q_full", (seq, h_q), act_dtype)],
        meta={"m": seq, "n": h_q, "k": model.q_lora_rank},
        layer_id=layer_id, layer_kind=layer_kind))

    # kv_a_proj_with_mqa: h → kv_lora_rank + qk_rope_head_dim
    ops.append(Op(name=f"{prefix}.kv_a_proj", kind="matmul",
        inputs=[_tensor("x_ln1", (seq, h), act_dtype)],
        outputs=[_tensor("kv_a", (seq, h_kv), act_dtype)],
        meta={"m": seq, "n": h_kv, "k": h},
        layer_id=layer_id, layer_kind=layer_kind))

    # kv_a_layernorm (only on the kv_lora_rank part, not rope part)
    ops.append(Op(name=f"{prefix}.kv_a_norm", kind=_norm_kind(model),
        inputs=[_tensor("kv_a_latent", (seq, model.kv_lora_rank), act_dtype)],
        outputs=[_tensor("kv_a_normed", (seq, model.kv_lora_rank), act_dtype)],
        meta={"bytes_fwd": seq * model.kv_lora_rank * act_dtype.bytes * 2},
        layer_id=layer_id, layer_kind=layer_kind))

    # kv_b_proj: kv_lora_rank → num_heads × (qk_nope + v_head)
    ops.append(Op(name=f"{prefix}.kv_b_proj", kind="matmul",
        inputs=[_tensor("kv_a_normed", (seq, model.kv_lora_rank), act_dtype)],
        outputs=[_tensor("kv_full", (seq, h_kv_out), act_dtype)],
        meta={"m": seq, "n": h_kv_out, "k": model.kv_lora_rank},
        layer_id=layer_id, layer_kind=layer_kind))

    # RoPE on rope dims of Q and K
    h_rope = n_h * qk_rope
    ops.append(Op(name=f"{prefix}.rope", kind="rope",
        inputs=[_tensor("q_rope_part", (seq, h_rope), act_dtype),
                _tensor("k_rope_part", (seq, qk_rope), act_dtype)],
        outputs=[_tensor("q_rope", (seq, h_rope), act_dtype),
                 _tensor("k_rope", (seq, qk_rope), act_dtype)],
        meta={"bytes_fwd": seq * (h_rope + qk_rope) * act_dtype.bytes * 2},
        layer_id=layer_id, layer_kind=layer_kind))

# Optional: Lightning Indexer for V3.2 DSA (all attention layers when index_topk > 0)
    if model.index_topk > 0 and not model.use_v4_attn:
        ops.extend(_build_indexer_ops(model, layer_id, seq,
                                       prefix, layer_kind, act_dtype,
                                       q_input_name="q_a_normed"))

# Attention core — MLA uses qk_nope + qk_rope for QK, v_head for V
    effective_head_dim = qk_nope + qk_rope
    attn_meta = {"b": 1, "s": seq, "heads": n_h,
                 "head_dim": effective_head_dim, "causal": True,
                 "v_head_dim": v_hd}
    # V3.2: sparse attention over indexer topk KV (all attention layers)
    if model.index_topk > 0 and not model.use_v4_attn:
        attn_meta["sparse_topk"] = model.index_topk
    ops.append(Op(name=f"{prefix}.attn_core", kind="attn_core",
        inputs=[_tensor("q_final", (seq, h_q), act_dtype),
                _tensor("k_final", (seq, n_h * qk_nope + qk_rope), act_dtype),
                _tensor("v_final", (seq, n_h * v_hd), act_dtype)],
        outputs=[_tensor("attn_out", (seq, h_attn_out), act_dtype)],
        meta=attn_meta,
        layer_id=layer_id, layer_kind=layer_kind))

    # o_proj: num_heads × v_head_dim → hidden
    ops.append(Op(name=f"{prefix}.o_proj", kind="matmul",
        inputs=[_tensor("attn_out", (seq, h_attn_out), act_dtype)],
        outputs=[_tensor("attn_proj", (seq, h), act_dtype)],
        meta={"m": seq, "n": h, "k": h_attn_out},
        layer_id=layer_id, layer_kind=layer_kind))

    return ops


def _build_v4_attn(model: ModelSpec, layer_id: int, seq: int,
                   layer_kind: LayerKind, prefix: str,
                   act_dtype: Dtype) -> list[Op]:
    """V4 attention: low-rank Q + single-KV MQA + optional compressor/indexer + grouped O."""
    h = model.hidden
    d = model.head_dim
    h_attn = model.num_heads * d
    cp_type = model.get_layer_cp_type(layer_id)

    ops: list[Op] = []

    # wq_a: h → q_lora_rank
    ops.append(Op(name=f"{prefix}.wq_a", kind="matmul",
        inputs=[_tensor("x_ln1", (seq, h), act_dtype)],
        outputs=[_tensor("qr", (seq, model.q_lora_rank), act_dtype)],
        meta={"m": seq, "n": model.q_lora_rank, "k": h},
        layer_id=layer_id, layer_kind=layer_kind))

    # q_norm (RMSNorm)
    ops.append(Op(name=f"{prefix}.q_norm", kind=_norm_kind(model),
        inputs=[_tensor("qr", (seq, model.q_lora_rank), act_dtype)],
        outputs=[_tensor("qr_normed", (seq, model.q_lora_rank), act_dtype)],
        meta={"bytes_fwd": seq * model.q_lora_rank * act_dtype.bytes * 2},
        layer_id=layer_id, layer_kind=layer_kind))

    # wq_b: q_lora_rank → num_heads × head_dim
    ops.append(Op(name=f"{prefix}.wq_b", kind="matmul",
        inputs=[_tensor("qr_normed", (seq, model.q_lora_rank), act_dtype)],
        outputs=[_tensor("q_raw", (seq, h_attn), act_dtype)],
        meta={"m": seq, "n": h_attn, "k": model.q_lora_rank},
        layer_id=layer_id, layer_kind=layer_kind))

    # rsqrt normalization (memory-bound)
    ops.append(Op(name=f"{prefix}.q_rsqrt_norm", kind=_norm_kind(model),
        inputs=[_tensor("q_raw", (seq, h_attn), act_dtype)],
        outputs=[_tensor("q_norm2", (seq, h_attn), act_dtype)],
        meta={"bytes_fwd": seq * h_attn * act_dtype.bytes * 2},
        layer_id=layer_id, layer_kind=layer_kind))

    # wkv: h → head_dim (single KV head, MQA)
    ops.append(Op(name=f"{prefix}.wkv", kind="matmul",
        inputs=[_tensor("x_ln1", (seq, h), act_dtype)],
        outputs=[_tensor("kv_raw", (seq, d), act_dtype)],
        meta={"m": seq, "n": d, "k": h},
        layer_id=layer_id, layer_kind=layer_kind))

    # kv_norm
    ops.append(Op(name=f"{prefix}.kv_norm", kind=_norm_kind(model),
        inputs=[_tensor("kv_raw", (seq, d), act_dtype)],
        outputs=[_tensor("kv_normed", (seq, d), act_dtype)],
        meta={"bytes_fwd": seq * d * act_dtype.bytes * 2},
        layer_id=layer_id, layer_kind=layer_kind))

    # Optional: compressor for CSA/HCA
    if cp_type in ('csa', 'hca'):
        ops.extend(_build_compressor_ops(model, layer_id, seq, cp_type,
                                          prefix, layer_kind, act_dtype))

    # Optional: indexer for CSA
    if cp_type == 'csa':
        ops.extend(_build_indexer_ops(model, layer_id, seq,
                                       prefix, layer_kind, act_dtype))

    # RoPE on rope dims
    rope_d = model.qk_rope_head_dim
    ops.append(Op(name=f"{prefix}.rope", kind="rope",
        inputs=[_tensor("q_rope_part", (seq, model.num_heads * rope_d), act_dtype),
                _tensor("k_rope_part", (seq, rope_d), act_dtype)],
        outputs=[_tensor("q_rope", (seq, model.num_heads * rope_d), act_dtype),
                 _tensor("k_rope", (seq, rope_d), act_dtype)],
        meta={"bytes_fwd": seq * (model.num_heads * rope_d + rope_d) * act_dtype.bytes * 2},
        layer_id=layer_id, layer_kind=layer_kind))

    # Attention core — varies by layer type
    attn_meta = {"b": 1, "s": seq, "heads": model.num_heads,
                 "head_dim": d, "causal": True}

    if cp_type == 'csa':
        attn_kind = "sparse_attn"
        attn_meta["sparse_topk"] = model.index_topk
        attn_meta["swa_window"] = model.swa_window
    elif cp_type == 'hca':
        attn_kind = "hca_attn"
        ratio = model.compress_ratios[layer_id] if model.compress_ratios else 128
        attn_meta["compress_ratio"] = ratio
        attn_meta["swa_window"] = model.swa_window
    elif cp_type == 'swa':
        attn_kind = "swa_attn"
        attn_meta["swa_window"] = model.swa_window
    else:
        attn_kind = "attn_core"

    ops.append(Op(name=f"{prefix}.{attn_kind}", kind=attn_kind,
        inputs=[_tensor("q_final", (seq, h_attn), act_dtype),
                _tensor("k_all", (seq, d), act_dtype),
                _tensor("v_all", (seq, d), act_dtype)],
        outputs=[_tensor("attn_out", (seq, h_attn), act_dtype)],
        meta=attn_meta,
        layer_id=layer_id, layer_kind=layer_kind))

    # Grouped output projection: wo_a → wo_b
    # wo_a: per-group matmul, k=h_per_group gives correct total FLOPs
    # (equivalent to o_groups parallel matmuls summed as one op)
    h_per_group = h_attn // model.o_groups
    d_g = model.o_groups * model.o_lora_rank

    ops.append(Op(name=f"{prefix}.wo_a", kind="matmul",
        inputs=[_tensor("attn_out", (seq, h_attn), act_dtype)],
        outputs=[_tensor("o_a", (seq, d_g), act_dtype)],
        meta={"m": seq, "n": d_g, "k": h_per_group},
        layer_id=layer_id, layer_kind=layer_kind))

    ops.append(Op(name=f"{prefix}.wo_b", kind="matmul",
        inputs=[_tensor("o_a", (seq, d_g), act_dtype)],
        outputs=[_tensor("attn_proj", (seq, h), act_dtype)],
        meta={"m": seq, "n": h, "k": d_g},
        layer_id=layer_id, layer_kind=layer_kind))

    return ops


def _build_compressor_ops(model: ModelSpec, layer_id: int, seq: int,
                          cp_type: str, prefix: str,
                          layer_kind: LayerKind, act_dtype: Dtype) -> list[Op]:
    """KV compressor: wkv + wgate + gated pooling → compressed KV."""
    d = model.head_dim
    m = 4 if cp_type == 'csa' else 128
    coff = 2 if cp_type == 'csa' else 1
    h = model.hidden
    ops: list[Op] = []

    ops.append(Op(name=f"{prefix}.comp_wkv", kind="matmul",
        inputs=[_tensor("x_ln1", (seq, h), act_dtype)],
        outputs=[_tensor("comp_kv_raw", (seq, coff * d), act_dtype)],
        meta={"m": seq, "n": coff * d, "k": h},
        layer_id=layer_id, layer_kind=layer_kind))

    ops.append(Op(name=f"{prefix}.comp_wgate", kind="matmul",
        inputs=[_tensor("x_ln1", (seq, h), act_dtype)],
        outputs=[_tensor("comp_score_raw", (seq, coff * d), act_dtype)],
        meta={"m": seq, "n": coff * d, "k": h},
        layer_id=layer_id, layer_kind=layer_kind))

    ops.append(Op(name=f"{prefix}.comp_pool", kind="compressor_pool",
        inputs=[_tensor("comp_kv_raw", (seq, coff * d), act_dtype),
                _tensor("comp_score_raw", (seq, coff * d), act_dtype)],
        outputs=[_tensor("kv_compressed", (seq, d), act_dtype)],
        meta={"s": seq, "m": m, "coff": coff, "d": d, "bytes_fwd": seq * d * act_dtype.bytes},
        layer_id=layer_id, layer_kind=layer_kind))

    return ops


def _build_indexer_ops(model: ModelSpec, layer_id: int, seq: int,
                       prefix: str, layer_kind: LayerKind,
                       act_dtype: Dtype, q_input_name: str = "qr") -> list[Op]:
    """Lightning Indexer: index queries + scoring + top-k selection."""
    h = model.hidden
    ih = model.index_n_heads
    id_ = model.index_head_dim
    coff = 2  # overlapping for m=4
    ops: list[Op] = []

    # Indexer compressor pools m=4 raw tokens into 1 compressed block (both V3.2 and V4-CSA).
    # Scoring einsum is (seq, kv_len) where kv_len = seq // 4 after compression.
    cp_type = model.get_layer_cp_type(layer_id) if model.use_v4_attn else 'none'
    kv_len = max(1, seq // 4)

    # Indexer queries from q_lora_rank latent
    ops.append(Op(name=f"{prefix}.idx_wq_b", kind="matmul",
        inputs=[_tensor(q_input_name, (seq, model.q_lora_rank), act_dtype)],
        outputs=[_tensor("idx_q", (seq, ih * id_), act_dtype)],
        meta={"m": seq, "n": ih * id_, "k": model.q_lora_rank},
        layer_id=layer_id, layer_kind=layer_kind))

    # Per-head weights projection
    ops.append(Op(name=f"{prefix}.idx_weights", kind="matmul",
        inputs=[_tensor("x_ln1", (seq, h), act_dtype)],
        outputs=[_tensor("idx_w", (seq, ih), act_dtype)],
        meta={"m": seq, "n": ih, "k": h},
        layer_id=layer_id, layer_kind=layer_kind))

    # Indexer's own compressor for building index KV
    ops.append(Op(name=f"{prefix}.idx_comp_wkv", kind="matmul",
        inputs=[_tensor("x_ln1", (seq, h), act_dtype)],
        outputs=[_tensor("idx_kv_raw", (seq, coff * id_), act_dtype)],
        meta={"m": seq, "n": coff * id_, "k": h},
        layer_id=layer_id, layer_kind=layer_kind))

    ops.append(Op(name=f"{prefix}.idx_comp_wgate", kind="matmul",
        inputs=[_tensor("x_ln1", (seq, h), act_dtype)],
        outputs=[_tensor("idx_score_raw", (seq, coff * id_), act_dtype)],
        meta={"m": seq, "n": coff * id_, "k": h},
        layer_id=layer_id, layer_kind=layer_kind))

    ops.append(Op(name=f"{prefix}.idx_comp_pool", kind="compressor_pool",
        inputs=[_tensor("idx_kv_raw", (seq, coff * id_), act_dtype),
                _tensor("idx_score_raw", (seq, coff * id_), act_dtype)],
        outputs=[_tensor("idx_kv", (kv_len, id_), act_dtype)],
        meta={"s": seq, "m": 4, "coff": coff, "d": id_, "bytes_fwd": kv_len * id_ * act_dtype.bytes},
        layer_id=layer_id, layer_kind=layer_kind))

    # Scoring: einsum(q[seq,ih,id_], kv[kv_len,id_]) → ReLU × weights → topk
    # idx_kv has kv_len = seq//4 entries after compression
    idx_q_bytes = seq * ih * id_ * act_dtype.bytes
    idx_kv_bytes = kv_len * id_ * act_dtype.bytes
    idx_w_bytes = seq * ih * act_dtype.bytes
    idx_out_bytes = seq * model.index_topk * act_dtype.bytes
    total_bytes = idx_q_bytes + idx_kv_bytes + idx_w_bytes + idx_out_bytes
    ops.append(Op(name=f"{prefix}.idx_score_topk", kind="indexer_topk",
        inputs=[_tensor("idx_q", (seq, ih * id_), act_dtype),
                _tensor("idx_kv", (kv_len, id_), act_dtype),
                _tensor("idx_w", (seq, ih), act_dtype)],
        outputs=[_tensor("topk_indices", (seq, model.index_topk), act_dtype)],
        meta={"s": seq, "ih": ih, "id": id_, "topk": model.index_topk,
              "kv_len": kv_len,
              "bytes_fwd": total_bytes},
        layer_id=layer_id, layer_kind=layer_kind))

    return ops


# ── MoE FFN sub-block ──────────────────────────────────────────────────

def _build_moe_ffn_ops(model: ModelSpec, layer_id: int, seq: int,
                       prefix: str, layer_kind: LayerKind,
                       act_dtype: Dtype) -> list[Op]:
    """MoE FFN: router + shared expert + routed experts + aggregation."""
    h = model.hidden
    ops: list[Op] = []
    is_hash = layer_id < model.n_hash_routed_layers

    # Router
    if is_hash:
        ops.append(Op(name=f"{prefix}.router", kind="hash_route",
            inputs=[_tensor("x_ln2", (seq, h), act_dtype)],
            outputs=[_tensor("topk_weights", (seq, model.top_k), act_dtype),
                     _tensor("topk_indices", (seq, model.top_k), act_dtype)],
            meta={"num_experts": model.num_experts, "top_k": model.top_k,
                  "bytes_fwd": 0},
            layer_id=layer_id, layer_kind=layer_kind))
    else:
        ops.append(Op(name=f"{prefix}.router", kind="matmul",
            inputs=[_tensor("x_ln2", (seq, h), act_dtype)],
            outputs=[_tensor("router_logits", (seq, model.num_experts), act_dtype)],
            meta={"m": seq, "n": model.num_experts, "k": h},
            layer_id=layer_id, layer_kind=layer_kind))
        ops.append(Op(name=f"{prefix}.topk_select", kind="softmax",
            inputs=[_tensor("router_logits", (seq, model.num_experts), act_dtype)],
            outputs=[_tensor("topk_weights", (seq, model.top_k), act_dtype),
                     _tensor("topk_indices", (seq, model.top_k), act_dtype)],
            meta={"bytes_fwd": seq * model.num_experts * act_dtype.bytes * 2,
                  "num_experts": model.num_experts, "top_k": model.top_k,
                  "scoring_func": model.scoring_func},
            layer_id=layer_id, layer_kind=layer_kind))

    # Shared expert FFN
    if model.n_shared_experts > 0:
        if model.moe_ffn == 0:
            raise ValueError(
                f"Layer {layer_id}: moe_ffn=0 but n_shared_experts={model.n_shared_experts}>0. "
                "Set moe_ffn to the per-expert FFN hidden size in the model spec."
            )
        m = model.moe_ffn
        ops.append(Op(name=f"{prefix}.shared_up_proj", kind="matmul",
            inputs=[_tensor("x_ln2", (seq, h), act_dtype)],
            outputs=[_tensor("shared_up", (seq, m), act_dtype)],
            meta={"m": seq, "n": m, "k": h},
            layer_id=layer_id, layer_kind=layer_kind))
        ops.append(Op(name=f"{prefix}.shared_gate_proj", kind="matmul",
            inputs=[_tensor("x_ln2", (seq, h), act_dtype)],
            outputs=[_tensor("shared_gate", (seq, m), act_dtype)],
            meta={"m": seq, "n": m, "k": h},
            layer_id=layer_id, layer_kind=layer_kind))
        ops.append(Op(name=f"{prefix}.shared_swiGLU", kind="swiglu",
            inputs=[_tensor("shared_up", (seq, m), act_dtype),
                    _tensor("shared_gate", (seq, m), act_dtype)],
            outputs=[_tensor("shared_swiglu_out", (seq, m), act_dtype)],
            meta={"bytes_fwd": seq * m * act_dtype.bytes * 3,
                  "swiglu_clamp": model.swiglu_clamp},
            layer_id=layer_id, layer_kind=layer_kind))
        ops.append(Op(name=f"{prefix}.shared_down_proj", kind="matmul",
            inputs=[_tensor("shared_swiglu_out", (seq, m), act_dtype)],
            outputs=[_tensor("shared_ffn_out", (seq, h), act_dtype)],
            meta={"m": seq, "n": h, "k": m},
            layer_id=layer_id, layer_kind=layer_kind))

    # Routed expert FFN (single op with fwd_multiplier)
    ops.append(Op(name=f"{prefix}.routed_expert_ffn", kind="matmul",
        inputs=[_tensor("x_ln2", (seq, h), act_dtype)],
        outputs=[_tensor("routed_ffn_out", (seq, h), act_dtype)],
        meta={"m": seq, "n": h, "k": model.moe_ffn,
              "fwd_multiplier": 3 * model.top_k,
              "swiglu_clamp": model.swiglu_clamp,
              "fused_weight_dims": True},
        layer_id=layer_id, layer_kind=layer_kind))

    # Expert aggregation
    if model.n_shared_experts > 0:
        ops.append(Op(name=f"{prefix}.expert_agg", kind="add",
            inputs=[_tensor("shared_ffn_out", (seq, h), act_dtype),
                    _tensor("routed_ffn_out", (seq, h), act_dtype)],
            outputs=[_tensor("ffn_out", (seq, h), act_dtype)],
            meta={"bytes_fwd": seq * h * act_dtype.bytes * 3},
            layer_id=layer_id, layer_kind=layer_kind))
    else:
        ops.append(Op(name=f"{prefix}.expert_agg", kind="add",
            inputs=[_tensor("routed_ffn_out", (seq, h), act_dtype)],
            outputs=[_tensor("ffn_out", (seq, h), act_dtype)],
            meta={"bytes_fwd": seq * h * act_dtype.bytes * 2},
            layer_id=layer_id, layer_kind=layer_kind))

    return ops


# ── Block builders ─────────────────────────────────────────────────────

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
    model: ModelSpec | None = None,
) -> list[Op]:
    """Build ops for one dense transformer block.

    When `model` is provided, uses architecture-aware attention (MLA / V4 / standard).
    Without `model`, falls back to standard MHA for backward compatibility.
    """
    use_hc = hc_mult > 1
    ops: list[Op] = []
    h = hidden
    prefix = f"L{layer_id}"

    # ── HC pre-attn ────────────────────────────────────────────────────
    if use_hc:
        ops.append(_mhc_pre_op(
            seq, h, hc_mult, hc_sinkhorn_iters,
            layer_id, LayerKind.DENSE, prefix, "attn", act_dtype,
        ))
        ln1_in = _tensor(f"x_pre_attn", (seq, h), act_dtype)
    else:
        ln1_in = _tensor("x", (seq, h), act_dtype)

    # ── Pre-attention RMSNorm ──────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.ln1", kind=_norm_kind(model),
        inputs=[ln1_in],
        outputs=[_tensor("x_ln1", (seq, h), act_dtype)],
        meta={"bytes_fwd": seq * h * act_dtype.bytes * 2},
        layer_id=layer_id, layer_kind=LayerKind.DENSE,
    ))

    # ── Attention sub-block ────────────────────────────────────────────
    if model is not None:
        attn_ops = _build_attn_ops(model, layer_id, seq, LayerKind.DENSE,
                                    prefix, act_dtype)
    else:
        # Legacy path — standard MHA
        h_attn = num_heads * head_dim
        h_kv = num_kv_heads * head_dim
        attn_ops = [
            Op(name=f"{prefix}.qkv_proj", kind="matmul",
               inputs=[_tensor("x_ln1", (seq, h), act_dtype)],
               outputs=[_tensor("qkv", (seq, h_attn + 2 * h_kv), act_dtype)],
               meta={"m": seq, "n": h_attn + 2 * h_kv, "k": h},
               layer_id=layer_id, layer_kind=LayerKind.DENSE),
            Op(name=f"{prefix}.rope", kind="rope",
               inputs=[_tensor("q", (seq, h_attn), act_dtype),
                       _tensor("k", (seq, h_kv), act_dtype)],
               outputs=[_tensor("q_rope", (seq, h_attn), act_dtype),
                        _tensor("k_rope", (seq, h_kv), act_dtype)],
               meta={"bytes_fwd": seq * (h_attn + h_kv) * act_dtype.bytes * 2},
               layer_id=layer_id, layer_kind=LayerKind.DENSE),
            Op(name=f"{prefix}.attn_core", kind="attn_core",
               inputs=[_tensor("q_rope", (seq, h_attn), act_dtype),
                       _tensor("k_rope", (seq, h_kv), act_dtype),
                       _tensor("v", (seq, h_kv), act_dtype)],
               outputs=[_tensor("attn_out", (seq, h_attn), act_dtype)],
               meta={"b": 1, "s": seq, "heads": num_heads,
                     "head_dim": head_dim, "causal": True},
               layer_id=layer_id, layer_kind=LayerKind.DENSE),
            Op(name=f"{prefix}.o_proj", kind="matmul",
               inputs=[_tensor("attn_out", (seq, h_attn), act_dtype)],
               outputs=[_tensor("attn_proj", (seq, h), act_dtype)],
               meta={"m": seq, "n": h, "k": h_attn},
               layer_id=layer_id, layer_kind=LayerKind.DENSE),
        ]
    ops.extend(attn_ops)

    # ── Residual add OR mhc_post_attn ──────────────────────────────────
    if use_hc:
        ops.append(_mhc_post_op(
            seq, h, hc_mult, layer_id, LayerKind.DENSE, prefix, "attn", act_dtype,
        ))
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
            meta={"bytes_fwd": seq * h * act_dtype.bytes * 3},
            layer_id=layer_id, layer_kind=LayerKind.DENSE,
        ))
        ln2_in = _tensor("x_attn", (seq, h), act_dtype)

    # ── Post-attention RMSNorm ─────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.ln2", kind=_norm_kind(model),
        inputs=[ln2_in],
        outputs=[_tensor("x_ln2", (seq, h), act_dtype)],
        meta={"bytes_fwd": seq * h * act_dtype.bytes * 2},
        layer_id=layer_id, layer_kind=LayerKind.DENSE,
    ))

    # ── Dense FFN (SwiGLU) ─────────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.up_proj", kind="matmul",
        inputs=[_tensor("x_ln2", (seq, h), act_dtype)],
        outputs=[_tensor("up", (seq, ffn), act_dtype)],
        meta={"m": seq, "n": ffn, "k": h},
        layer_id=layer_id, layer_kind=LayerKind.DENSE,
    ))
    ops.append(Op(
        name=f"{prefix}.gate_proj", kind="matmul",
        inputs=[_tensor("x_ln2", (seq, h), act_dtype)],
        outputs=[_tensor("gate", (seq, ffn), act_dtype)],
        meta={"m": seq, "n": ffn, "k": h},
        layer_id=layer_id, layer_kind=LayerKind.DENSE,
    ))
    ops.append(Op(
        name=f"{prefix}.swiglu", kind="swiglu",
        inputs=[_tensor("up", (seq, ffn), act_dtype),
                _tensor("gate", (seq, ffn), act_dtype)],
        outputs=[_tensor("swiglu_out", (seq, ffn), act_dtype)],
        meta={"bytes_fwd": seq * ffn * act_dtype.bytes * 3},
        layer_id=layer_id, layer_kind=LayerKind.DENSE,
    ))
    ops.append(Op(
        name=f"{prefix}.down_proj", kind="matmul",
        inputs=[_tensor("swiglu_out", (seq, ffn), act_dtype)],
        outputs=[_tensor("ffn_out", (seq, h), act_dtype)],
        meta={"m": seq, "n": h, "k": ffn},
        layer_id=layer_id, layer_kind=LayerKind.DENSE,
    ))

    # ── Residual add OR mhc_post_ffn ───────────────────────────────────
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
    model: ModelSpec | None = None,
) -> list[Op]:
    """Build ops for one MoE transformer block."""
    use_hc = hc_mult > 1
    ops: list[Op] = []
    h = hidden
    prefix = f"L{layer_id}"

    # ── HC pre-attn ────────────────────────────────────────────────────
    if use_hc:
        ops.append(_mhc_pre_op(
            seq, h, hc_mult, hc_sinkhorn_iters,
            layer_id, LayerKind.MOE, prefix, "attn", act_dtype,
        ))
        ln1_in = _tensor(f"x_pre_attn", (seq, h), act_dtype)
    else:
        ln1_in = _tensor("x", (seq, h), act_dtype)

    # ── Pre-attention RMSNorm ──────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.ln1", kind=_norm_kind(model),
        inputs=[ln1_in],
        outputs=[_tensor("x_ln1", (seq, h), act_dtype)],
        meta={"bytes_fwd": seq * h * act_dtype.bytes * 2},
        layer_id=layer_id, layer_kind=LayerKind.MOE,
    ))

    # ── Attention sub-block ────────────────────────────────────────────
    if model is not None:
        attn_ops = _build_attn_ops(model, layer_id, seq, LayerKind.MOE,
                                    prefix, act_dtype)
    else:
        h_attn = num_heads * head_dim
        h_kv = num_kv_heads * head_dim
        attn_ops = [
            Op(name=f"{prefix}.qkv_proj", kind="matmul",
               inputs=[_tensor("x_ln1", (seq, h), act_dtype)],
               outputs=[_tensor("qkv", (seq, h_attn + 2 * h_kv), act_dtype)],
               meta={"m": seq, "n": h_attn + 2 * h_kv, "k": h},
               layer_id=layer_id, layer_kind=LayerKind.MOE),
            Op(name=f"{prefix}.rope", kind="rope",
               inputs=[_tensor("q", (seq, h_attn), act_dtype),
                       _tensor("k", (seq, h_kv), act_dtype)],
               outputs=[_tensor("q_rope", (seq, h_attn), act_dtype),
                        _tensor("k_rope", (seq, h_kv), act_dtype)],
               meta={"bytes_fwd": seq * (h_attn + h_kv) * act_dtype.bytes * 2},
               layer_id=layer_id, layer_kind=LayerKind.MOE),
            Op(name=f"{prefix}.attn_core", kind="attn_core",
               inputs=[_tensor("q_rope", (seq, h_attn), act_dtype),
                       _tensor("k_rope", (seq, h_kv), act_dtype),
                       _tensor("v", (seq, h_kv), act_dtype)],
               outputs=[_tensor("attn_out", (seq, h_attn), act_dtype)],
               meta={"b": 1, "s": seq, "heads": num_heads,
                     "head_dim": head_dim, "causal": True},
               layer_id=layer_id, layer_kind=LayerKind.MOE),
            Op(name=f"{prefix}.o_proj", kind="matmul",
               inputs=[_tensor("attn_out", (seq, h_attn), act_dtype)],
               outputs=[_tensor("attn_proj", (seq, h), act_dtype)],
               meta={"m": seq, "n": h, "k": h_attn},
               layer_id=layer_id, layer_kind=LayerKind.MOE),
        ]
    ops.extend(attn_ops)

    # ── Residual add OR mhc_post_attn ──────────────────────────────────
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

    # ── Post-attention RMSNorm ─────────────────────────────────────────
    ops.append(Op(
        name=f"{prefix}.ln2", kind=_norm_kind(model),
        inputs=[ln2_in],
        outputs=[_tensor("x_ln2", (seq, h), act_dtype)],
        meta={"bytes_fwd": seq * h * act_dtype.bytes * 2},
        layer_id=layer_id, layer_kind=LayerKind.MOE,
    ))

    # ── MoE FFN ────────────────────────────────────────────────────────
    if model is not None:
        ops.extend(_build_moe_ffn_ops(model, layer_id, seq, prefix,
                                       LayerKind.MOE, act_dtype))
    else:
        # Legacy path
        ops.append(Op(
            name=f"{prefix}.router", kind="matmul",
            inputs=[_tensor("x_ln2", (seq, h), act_dtype)],
            outputs=[_tensor("router_logits", (seq, num_experts), act_dtype)],
            meta={"m": seq, "n": num_experts, "k": h},
            layer_id=layer_id, layer_kind=LayerKind.MOE,
        ))
        ops.append(Op(
            name=f"{prefix}.topk_select", kind="softmax",
            inputs=[_tensor("router_logits", (seq, num_experts), act_dtype)],
            outputs=[_tensor("topk_weights", (seq, top_k), act_dtype),
                     _tensor("topk_indices", (seq, top_k), act_dtype)],
            meta={"bytes_fwd": seq * num_experts * act_dtype.bytes * 2,
                  "num_experts": num_experts, "top_k": top_k},
            layer_id=layer_id, layer_kind=LayerKind.MOE,
        ))
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
        ops.append(Op(
            name=f"{prefix}.routed_expert_ffn", kind="matmul",
            inputs=[_tensor("x_ln2", (seq, h), act_dtype)],
            outputs=[_tensor("routed_ffn_out", (seq, h), act_dtype)],
            meta={"m": seq, "n": h, "k": moe_ffn, "fwd_multiplier": 3 * top_k,
                  "fused_weight_dims": True},
            layer_id=layer_id, layer_kind=LayerKind.MOE,
        ))
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
            ops.append(Op(
                name=f"{prefix}.expert_agg", kind="add",
                inputs=[_tensor("routed_ffn_out", (seq, h), act_dtype)],
                outputs=[_tensor("ffn_out", (seq, h), act_dtype)],
                meta={"bytes_fwd": seq * h * act_dtype.bytes * 2},
                layer_id=layer_id, layer_kind=LayerKind.MOE,
            ))

    # ── Residual add OR mhc_post_ffn ───────────────────────────────────
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
    model: ModelSpec | None = None,
) -> list[Op]:
    """Build ops for one MTP block: embedding projection + transformer block."""
    prefix = f"L{layer_id}"
    h = hidden

    embed_proj = Op(
        name=f"{prefix}.mtp_embed_proj", kind="matmul",
        inputs=[_tensor("x_mtp_in", (seq, h), act_dtype)],
        outputs=[_tensor("x_proj", (seq, h), act_dtype)],
        meta={"m": seq, "n": h, "k": h},
        layer_id=layer_id, layer_kind=LayerKind.MTP,
    )

    # MTP uses MoE block if any MoE layers exist, else dense
    if model is not None and any(lk == LayerKind.MOE for lk in model.layers):
        block_ops = _moe_block(
            hidden=hidden, ffn=ffn, moe_ffn=model.moe_ffn,
            num_experts=model.num_experts, top_k=model.top_k,
            n_shared_experts=model.n_shared_experts,
            seq=seq, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, layer_id=layer_id, act_dtype=act_dtype,
            hc_mult=hc_mult, hc_sinkhorn_iters=hc_sinkhorn_iters,
            model=model,
        )
    else:
        block_ops = dense_block(
            hidden=hidden, ffn=ffn, seq=seq,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, layer_id=layer_id, act_dtype=act_dtype,
            hc_mult=hc_mult, hc_sinkhorn_iters=hc_sinkhorn_iters,
            model=model,
        )

    return [embed_proj] + block_ops


# ── Embedding / head / HC ops ──────────────────────────────────────────

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
    return Op(
        name="hc_expand", kind="hc_expand",
        inputs=[_tensor("x_embed", (seq, hidden), act_dtype)],
        outputs=[_tensor("x_hc_init", (seq, hc_mult, hidden), act_dtype)],
        meta={"b": 1, "s": seq, "h": hidden, "hc": hc_mult,
              "bytes_fwd": seq * hidden * hc_mult * act_dtype.bytes},
        layer_id=-1, layer_kind=LayerKind.DENSE,
    )


def _mhc_head_op(seq: int, hidden: int, hc_mult: int, act_dtype: Dtype) -> Op:
    return Op(
        name="mhc_head", kind="mhc_head",
        inputs=[_tensor("x_hc_final", (seq, hc_mult, hidden), act_dtype)],
        outputs=[_tensor("x_collapsed", (seq, hidden), act_dtype)],
        meta={"b": 1, "s": seq, "h": hidden, "hc": hc_mult, "mix_hc": hc_mult},
        layer_id=-1, layer_kind=LayerKind.DENSE,
    )


def _final_ln_op(model: ModelSpec, hidden: int, seq: int, act_dtype: Dtype) -> Op:
    return Op(
        name="final_ln", kind=_norm_kind(model),
        inputs=[_tensor("x_final_raw", (seq, hidden), act_dtype)],
        outputs=[_tensor("x_final", (seq, hidden), act_dtype)],
        meta={"bytes_fwd": seq * hidden * act_dtype.bytes * 2},
        layer_id=-1, layer_kind=LayerKind.DENSE,
    )


# ── Main graph builder ─────────────────────────────────────────────────

def build_graph(model: ModelSpec, strategy: Strategy) -> Graph:
    """Build the full IR from ModelSpec + Strategy."""
    all_ops: list[Op] = []
    layer_index: dict[int, tuple[int, int]] = {}
    h = model.hidden
    s = model.seq_len
    act_dtype = model.act_dtype

    hc_mult = max(1, model.hc_mult)
    hc_iters = model.hc_sinkhorn_iters

    # Embedding
    all_ops.append(_embed_op(model.vocab, h, s, act_dtype))

    # Optional HC expansion
    if hc_mult > 1:
        all_ops.append(_hc_expand_op(s, h, hc_mult, act_dtype))

    # Transformer blocks — pass ModelSpec for architecture-aware dispatch
    for i, lk in enumerate(model.layers):
        start = len(all_ops)
        if lk == LayerKind.DENSE:
            block_ops = dense_block(
                hidden=h, ffn=model.ffn, seq=s,
                num_heads=model.num_heads, num_kv_heads=model.num_kv_heads,
                head_dim=model.head_dim, layer_id=i, act_dtype=act_dtype,
                hc_mult=hc_mult, hc_sinkhorn_iters=hc_iters,
                model=model,
            )
        elif lk == LayerKind.MOE:
            block_ops = _moe_block(
                hidden=h, ffn=model.ffn, moe_ffn=model.moe_ffn,
                num_experts=model.num_experts, top_k=model.top_k,
                n_shared_experts=model.n_shared_experts,
                seq=s, num_heads=model.num_heads, num_kv_heads=model.num_kv_heads,
                head_dim=model.head_dim, layer_id=i, act_dtype=act_dtype,
                hc_mult=hc_mult, hc_sinkhorn_iters=hc_iters,
                model=model,
            )
        elif lk == LayerKind.MTP:
            block_ops = _mtp_block(
                hidden=h, ffn=model.ffn, seq=s,
                num_heads=model.num_heads, num_kv_heads=model.num_kv_heads,
                head_dim=model.head_dim, layer_id=i, act_dtype=act_dtype,
                hc_mult=hc_mult, hc_sinkhorn_iters=hc_iters,
                model=model,
            )
        else:
            raise ValueError(f"Unknown LayerKind: {lk}")

        all_ops.extend(block_ops)
        layer_index[i] = (start, len(all_ops))

    # Optional HC head mix-down
    if hc_mult > 1:
        all_ops.append(_mhc_head_op(s, h, hc_mult, act_dtype))

# Final LN + lm_head
    all_ops.append(_final_ln_op(model, h, s, act_dtype))
    all_ops.append(_lm_head_op(model.vocab, h, s, act_dtype))

    graph = Graph(ops=all_ops, collectives=[], layer_index=layer_index)

    # Apply sharding and insert collectives
    shard = ShardPlan(strategy)
    insert_collectives(graph, model, strategy)

    return graph
