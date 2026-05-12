"""Per-class fusion-rule templates.

Each template is a dict that mirrors the YAML schema from
``docs/fusion_v2_rich_rules_zh.md`` §3.1.  ``joiner.join_rules`` consumes
these templates, fills in any runtime-observed op sequence and stamps
the final ``target_class`` / ``op_type``.

Coverage (16 patterns, intentionally focused — extend as needed):

  * Norms:       RMSNorm, LayerNorm
  * Linear-like: Linear, ColumnParallelLinear, RowParallelLinear,
                 ParallelEmbedding
  * MoE:         Gate, Expert, MoE, MLP
  * Attention:   MLA, Indexer, Compressor, Attention
  * Top-level helpers: apply_rotary_emb, sparse_attn
  * Block:       Block (transformer block — class_only marker)

A ``_default`` entry catches anything else; the joiner uses it to emit a
``class_only`` rule with empty formulas so a human can finish the spec.
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Optional


# ─── Shared shape derivations ─────────────────────────────────────────────────

_BS_H_INOUT = {
    "batch_size": "activation.shape[0]",
    "seq_len": "activation.shape[1]",
    "hidden_in": "activation.shape[-1]",
    "hidden_out": "output.shape[-1]",
}

_BS_H = {
    "batch_size": "activation.shape[0]",
    "seq_len": "activation.shape[1]",
    "hidden_in": "activation.shape[-1]",
}


# ─── Templates ────────────────────────────────────────────────────────────────

TEMPLATES: dict[str, dict] = {
    # ── Norms ───────────────────────────────────────────────────────────────
    "RMSNorm": {
        "op_type": "rms_norm",
        "priority": 20,
        "match": {"kind": "ordered_regex", "min_ops": 5},
        "io_roles": [
            {"role": "activation", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 0,
             "shape_role": "[B,S,H]"},
            {"role": "weight", "source_kind": "input",
             "source_op_index": -1, "source_arg_index": 0,
             "shape_role": "[H]"},
            {"role": "output", "source_kind": "output",
             "source_op_index": -1, "source_arg_index": -1,
             "shape_role": "[B,S,H]"},
        ],
        "shape_derivation": {
            "batch_size": "activation.shape[0]",
            "seq_len": "activation.shape[1]",
            "hidden_in": "activation.shape[-1]",
            "hidden_out": "output.shape[-1]",
        },
        "flops_formula": "4 * batch_size * seq_len * hidden_in",
        "memory_formula": "activation.bytes + weight.bytes + output.bytes",
        "annotations": {"layer_norm_kind": "rms"},
    },

    "LayerNorm": {
        "op_type": "layer_norm",
        "priority": 20,
        "match": {"kind": "ordered_regex", "min_ops": 4},
        "io_roles": [
            {"role": "activation", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 0,
             "shape_role": "[B,S,H]"},
            {"role": "weight", "source_kind": "input",
             "source_op_index": -1, "source_arg_index": 0,
             "shape_role": "[H]"},
            {"role": "output", "source_kind": "output",
             "source_op_index": -1, "source_arg_index": -1,
             "shape_role": "[B,S,H]"},
        ],
        "shape_derivation": dict(_BS_H_INOUT),
        "flops_formula": "8 * batch_size * seq_len * hidden_in",
        "memory_formula": "activation.bytes + weight.bytes + output.bytes",
        "annotations": {"layer_norm_kind": "layer"},
    },

    # ── Linear family ───────────────────────────────────────────────────────
    "Linear": {
        "op_type": "linear",
        "priority": 25,
        "match": {
            "kind": "dag_signature",
            "op_multiset": [
                ["aten\\.mm\\.default|aten\\.matmul\\..*|aten\\.linear\\.default", 1],
            ],
        },
        "io_roles": [
            {"role": "activation", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 0,
             "shape_role": "[B,S,Hin]"},
            {"role": "weight", "source_kind": "input",
             "source_op_index": -1, "source_arg_index": 1,
             "shape_role": "[Hout,Hin]"},
            {"role": "output", "source_kind": "output",
             "source_op_index": -1, "source_arg_index": -1,
             "shape_role": "[B,S,Hout]"},
        ],
        "shape_derivation": dict(_BS_H_INOUT),
        "flops_formula": "2 * batch_size * seq_len * hidden_in * hidden_out",
        "memory_formula": "activation.bytes + weight.bytes + output.bytes",
        "annotations": {"quant": "auto"},
    },

    "ColumnParallelLinear": {
        "op_type": "column_parallel_linear",
        "priority": 25,
        "match": {"kind": "class_only"},
        "io_roles": [
            {"role": "activation", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 0,
             "shape_role": "[B,S,Hin]"},
            {"role": "weight", "source_kind": "input",
             "source_op_index": -1, "source_arg_index": 1,
             "shape_role": "[Hout/TP,Hin]"},
            {"role": "output", "source_kind": "output",
             "source_op_index": -1, "source_arg_index": -1,
             "shape_role": "[B,S,Hout/TP]"},
        ],
        "shape_derivation": dict(_BS_H_INOUT),
        "flops_formula": "2 * batch_size * seq_len * hidden_in * hidden_out",
        "memory_formula": "activation.bytes + weight.bytes + output.bytes",
        "annotations": {"parallel": "column"},
    },

    "RowParallelLinear": {
        "op_type": "row_parallel_linear",
        "priority": 25,
        "match": {"kind": "class_only"},
        "io_roles": [
            {"role": "activation", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 0,
             "shape_role": "[B,S,Hin/TP]"},
            {"role": "weight", "source_kind": "input",
             "source_op_index": -1, "source_arg_index": 1,
             "shape_role": "[Hout,Hin/TP]"},
            {"role": "output", "source_kind": "output",
             "source_op_index": -1, "source_arg_index": -1,
             "shape_role": "[B,S,Hout]"},
        ],
        "shape_derivation": dict(_BS_H_INOUT),
        "flops_formula": "2 * batch_size * seq_len * hidden_in * hidden_out",
        "memory_formula": "activation.bytes + weight.bytes + output.bytes",
        "annotations": {"parallel": "row"},
    },

    "ParallelEmbedding": {
        "op_type": "parallel_embedding",
        "priority": 20,
        "match": {"kind": "class_only"},
        "io_roles": [
            {"role": "activation", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 0,
             "shape_role": "[B,S]"},
            {"role": "weight", "source_kind": "input",
             "source_op_index": -1, "source_arg_index": 0,
             "shape_role": "[V,H]"},
            {"role": "output", "source_kind": "output",
             "source_op_index": -1, "source_arg_index": -1,
             "shape_role": "[B,S,H]"},
        ],
        "shape_derivation": {
            "batch_size": "activation.shape[0]",
            "seq_len": "activation.shape[1]",
            "vocab_size": "weight.shape[0]",
            "hidden_out": "output.shape[-1]",
        },
        "flops_formula": "0",
        "memory_formula": "activation.bytes + output.bytes",
        "annotations": {"parallel": "row"},
    },

    # ── MoE ─────────────────────────────────────────────────────────────────
    "Gate": {
        "op_type": "moe_gate",
        "priority": 25,
        "match": {"kind": "class_only"},
        "io_roles": [
            {"role": "activation", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 0,
             "shape_role": "[B,S,H]"},
            {"role": "router_weight", "source_kind": "input",
             "source_op_index": -1, "source_arg_index": 0,
             "shape_role": "[E,H]"},
            {"role": "output", "source_kind": "output",
             "source_op_index": -1, "source_arg_index": -1,
             "shape_role": "[B*S,topk]"},
        ],
        "shape_derivation": {
            "batch_size": "activation.shape[0]",
            "seq_len": "activation.shape[1]",
            "hidden_in": "activation.shape[-1]",
            "num_experts": "router_weight.shape[0]",
            "topk": "output.shape[-1]",
        },
        "flops_formula": "2 * batch_size * seq_len * hidden_in * num_experts",
        "memory_formula": "activation.bytes + router_weight.bytes + output.bytes",
        "annotations": {"moe_role": "gate"},
    },

    "Expert": {
        "op_type": "moe_expert",
        "priority": 25,
        "match": {"kind": "class_only"},
        "io_roles": [
            {"role": "activation", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 0,
             "shape_role": "[T,H]"},
            {"role": "output", "source_kind": "output",
             "source_op_index": -1, "source_arg_index": -1,
             "shape_role": "[T,H]"},
        ],
        "shape_derivation": {
            "batch_size": "activation.shape[0]",
            "hidden_in": "activation.shape[-1]",
            "hidden_out": "output.shape[-1]",
            "extra": [
                ["intermediate_dim", "hidden_in * 4"],
            ],
        },
        # SwiGLU: 3 GEMMs (w1, w3, w2), all of size [T,H]·[H,I]
        "flops_formula": "6 * activation.shape[0] * hidden_in * intermediate_dim",
        "memory_formula": "activation.bytes + output.bytes * 2",
        "annotations": {"moe_role": "expert", "ffn_kind": "swiglu"},
    },

    "MoE": {
        "op_type": "moe_layer",
        "priority": 30,
        "match": {"kind": "class_only"},
        "io_roles": [
            {"role": "activation", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 0,
             "shape_role": "[B,S,H]"},
            {"role": "output", "source_kind": "output",
             "source_op_index": -1, "source_arg_index": -1,
             "shape_role": "[B,S,H]"},
        ],
        "shape_derivation": dict(_BS_H_INOUT),
        # Outer flops_kind=sum_children — each Expert / Gate contributes.
        "flops_kind": "sum_children",
        "memory_formula": "activation.bytes + output.bytes",
        "annotations": {"moe_role": "layer"},
    },

    "MLP": {
        "op_type": "mlp",
        "priority": 25,
        "match": {"kind": "class_only"},
        "io_roles": [
            {"role": "activation", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 0,
             "shape_role": "[B,S,H]"},
            {"role": "output", "source_kind": "output",
             "source_op_index": -1, "source_arg_index": -1,
             "shape_role": "[B,S,H]"},
        ],
        "shape_derivation": {
            **_BS_H_INOUT,
            "extra": [["intermediate_dim", "hidden_in * 4"]],
        },
        "flops_formula": "6 * batch_size * seq_len * hidden_in * intermediate_dim",
        "memory_formula": "activation.bytes + output.bytes",
        "annotations": {"ffn_kind": "swiglu"},
    },

    # ── Attention family ────────────────────────────────────────────────────
    "MLA": {
        "op_type": "mla_attention",
        "priority": 30,
        "match": {"kind": "class_only"},
        "io_roles": [
            {"role": "activation", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 0,
             "shape_role": "[B,S,H]"},
            {"role": "output", "source_kind": "output",
             "source_op_index": -1, "source_arg_index": -1,
             "shape_role": "[B,S,H]"},
        ],
        "shape_derivation": dict(_BS_H_INOUT),
        # Approximation: 4 projections + QK^T + AV  (kept conservative).
        "flops_formula": "8 * batch_size * seq_len * hidden_in * hidden_out + 2 * batch_size * seq_len * seq_len * hidden_out",
        "memory_formula": "activation.bytes + output.bytes",
        "annotations": {"attention_kind": "mla"},
    },

    "Attention": {
        "op_type": "attention",
        "priority": 30,
        "match": {"kind": "class_only"},
        "io_roles": [
            {"role": "activation", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 0,
             "shape_role": "[B,S,H]"},
            {"role": "output", "source_kind": "output",
             "source_op_index": -1, "source_arg_index": -1,
             "shape_role": "[B,S,H]"},
        ],
        "shape_derivation": dict(_BS_H_INOUT),
        "flops_formula": "8 * batch_size * seq_len * hidden_in * hidden_out + 2 * batch_size * seq_len * seq_len * hidden_out",
        "memory_formula": "activation.bytes + output.bytes",
        "annotations": {"attention_kind": "mha"},
    },

    "Indexer": {
        "op_type": "sparse_indexer",
        "priority": 25,
        "match": {"kind": "class_only"},
        "io_roles": [
            {"role": "activation", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 0,
             "shape_role": "[B,S,H]"},
            {"role": "output", "source_kind": "output",
             "source_op_index": -1, "source_arg_index": -1,
             "shape_role": "[B,S,topk]"},
        ],
        "shape_derivation": {
            **_BS_H,
            "topk": "output.shape[-1]",
        },
        "flops_formula": "4 * batch_size * seq_len * hidden_in * topk",
        "memory_formula": "activation.bytes + output.bytes",
        "annotations": {"sparse_role": "indexer"},
    },

    "Compressor": {
        "op_type": "kv_compressor",
        "priority": 30,
        "match": {"kind": "class_only"},
        "io_roles": [
            {"role": "activation", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 0,
             "shape_role": "[B,S,H]"},
            {"role": "output", "source_kind": "output",
             "source_op_index": -1, "source_arg_index": -1,
             "shape_role": "[B,S/R,D]"},
        ],
        "shape_derivation": {
            "batch_size": "activation.shape[0]",
            "seq_len": "activation.shape[1]",
            "hidden_in": "activation.shape[-1]",
            "head_dim": "output.shape[-1]",
            "extra": [
                ["compress_ratio",
                 "max(1, activation.shape[1] // output.shape[1])"],
            ],
        },
        "flops_formula": "2 * batch_size * seq_len * hidden_in * head_dim",
        "memory_formula": "activation.bytes + output.bytes * 2",
        "annotations": {"sparse_role": "compressor"},
    },

    # ── Top-level helpers ───────────────────────────────────────────────────
    "apply_rotary_emb": {
        "op_type": "apply_rotary_emb",
        "priority": 15,
        "match": {"kind": "ordered_regex", "min_ops": 1},
        "io_roles": [
            {"role": "activation", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 0,
             "shape_role": "[B,S,N,D]"},
            {"role": "freqs_cis", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 1,
             "shape_role": "[S,D/2]"},
            {"role": "output", "source_kind": "output",
             "source_op_index": -1, "source_arg_index": -1,
             "shape_role": "[B,S,N,D]"},
        ],
        "shape_derivation": {
            "batch_size": "activation.shape[0]",
            "seq_len": "activation.shape[1]",
            "num_heads": "activation.shape[2]",
            "head_dim": "activation.shape[3]",
        },
        "flops_formula": "6 * batch_size * seq_len * num_heads * head_dim",
        "memory_formula": "activation.bytes + output.bytes",
        "annotations": {"helper_kind": "rope"},
    },

    "sparse_attn": {
        "op_type": "sparse_attention",
        "priority": 30,
        "match": {"kind": "class_only"},
        "io_roles": [
            {"role": "activation", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 0,
             "shape_role": "[B,S,N,D]"},
            {"role": "kv_cache", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 1,
             "shape_role": "[B,S,N,D]"},
            {"role": "attn_sink", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 2,
             "shape_role": "[N]"},
            {"role": "expert_indices", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 3,
             "shape_role": "[B,S,topk]"},
            {"role": "output", "source_kind": "output",
             "source_op_index": -1, "source_arg_index": -1,
             "shape_role": "[B,S,N,D]"},
        ],
        "shape_derivation": {
            "batch_size": "activation.shape[0]",
            "seq_len": "activation.shape[1]",
            "num_heads": "activation.shape[2]",
            "head_dim": "activation.shape[3]",
            "topk": "expert_indices.shape[-1]",
        },
        "flops_formula": "4 * batch_size * seq_len * num_heads * head_dim * topk",
        "memory_formula": "activation.bytes + kv_cache.bytes + output.bytes",
        "annotations": {"helper_kind": "sparse_attn"},
    },

    # ── Block ───────────────────────────────────────────────────────────────
    "Block": {
        "op_type": "transformer_block",
        "priority": 5,  # very low: defer to inner module rules first
        "match": {"kind": "class_only"},
        "io_roles": [
            {"role": "activation", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 0,
             "shape_role": "[B,S,H]"},
            {"role": "output", "source_kind": "output",
             "source_op_index": -1, "source_arg_index": -1,
             "shape_role": "[B,S,H]"},
        ],
        "shape_derivation": dict(_BS_H_INOUT),
        "flops_kind": "sum_children",
        "memory_formula": "activation.bytes + output.bytes",
        "annotations": {"layer_role": "transformer_block"},
    },

    # ── Default fallback ────────────────────────────────────────────────────
    "_default": {
        "op_type": None,  # filled by joiner
        "priority": 10,
        "match": {"kind": "class_only"},
        "io_roles": [
            {"role": "activation", "source_kind": "input",
             "source_op_index": 0, "source_arg_index": 0,
             "shape_role": "[B,S,H]"},
            {"role": "output", "source_kind": "output",
             "source_op_index": -1, "source_arg_index": -1,
             "shape_role": "[B,S,H]"},
        ],
        # No shape_derivation / flops_formula — joiner records a
        # review_note asking the developer to fill these in.
        "annotations": {"discover_status": "fallback"},
    },
}


# Aliases — class-name aliases share a template
_ALIASES: dict[str, str] = {
    "LlamaRMSNorm": "RMSNorm",
    "Qwen2RMSNorm": "RMSNorm",
    "MistralRMSNorm": "RMSNorm",
}


def get_template(class_name: str) -> Optional[dict]:
    """Return a deep-copied template dict for *class_name*, or ``None``.

    Match precedence:
      1. exact name in ``TEMPLATES``,
      2. alias mapping,
      3. ``None`` (caller falls back to ``TEMPLATES['_default']``).
    """
    if class_name in TEMPLATES:
        return deepcopy(TEMPLATES[class_name])
    target = _ALIASES.get(class_name)
    if target and target in TEMPLATES:
        return deepcopy(TEMPLATES[target])
    return None
