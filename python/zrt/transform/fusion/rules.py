"""Fusion rule definitions: transparent ops, semantic labels, platform sub-patterns.

Architecture (three layers):
  Layer 1 – Capture (dispatch.py):  records ALL aten ops, never filters.
  Layer 2 – Fusion  (fusion.py):    skips transparent ops during pattern
                                    matching; absorbed ops stay in _children.
  Layer 3 – Display (excel_writer): optional filtering for readability,
                                    independent of fusion result.

Changing the display-layer filter has zero effect on fusion decisions because
pattern matching uses the skip sets here as *wildcards*, not as removal filters.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Transparent / absorbed ops
# ─────────────────────────────────────────────────────────────────────────────

# Always transparent: pure metadata / autograd book-keeping, zero compute.
ALWAYS_TRANSPARENT: Set[str] = {
    "aten.detach.default",
    "aten.alias.default",
    "aten.is_same_size.default",
    "prim.device.default",
    # MOVED:  "aten.lift_fresh_copy.default"  -> LIFT_OPS (performs memory copy)
    # REMOVED: "aten.detach_.default"   (in-place, not dispatched)
    # REMOVED: "aten._version.default"  (internal metadata, not dispatched)
}

# Shape-only ops: change stride/size metadata but do not move data.
# Treated as wildcards during pattern matching; absorbed into adjacent group.
# NOTE: some of these CAN carry semantic meaning in specific modules (e.g.
# transpose/view in MLA weight absorption).  The matching function skips them
SHAPE_OPS: Set[str] = {
    "aten.view.default",
    "aten._unsafe_view.default",
    # MOVED:  "aten.reshape.default"             -> POTENTIAL_COPY_OPS (may trigger copy)
    "aten.expand.default",
    "aten.expand_as.default",
    "aten.squeeze.default",
    "aten.squeeze.dim",
    "aten.unsqueeze.default",
    "aten.permute.default",
    "aten.transpose.int",
    # MOVED:  "aten.contiguous.memory_format"   -> POTENTIAL_COPY_OPS (triggers copy on non-contiguous input)
    # REMOVED: "aten.flatten.using_ints"        (C-level only)
    "aten.as_strided.default",
    "aten.select.int",
    "aten.slice.Tensor",
    "aten.t.default",
    # REMOVED: "aten.chunk.default"             (C-level only)
    "aten.split.Tensor",
    "aten.split_with_sizes.default",
    "aten.unbind.int",
    # ADDED: common view/shape ops confirmed dispatched
    "aten.diagonal.default",
    "aten.slice_backward.default",  # ADDED: backward of slice (reshape metadata only)
    # NOTE: aten.narrow.default is NOT added — narrow decomposes to aten.slice.Tensor
    #       before dispatch, so aten.narrow.default never appears in traces.
}

# Memory-initialisation ops: real (tiny) compute; kept in records but do not
# participate as pattern anchors.
INIT_OPS: Set[str] = {
    "aten.zeros_like.default",
    "aten.ones_like.default",
    "aten.full_like.default",
    "aten.empty_like.default",
    "aten.zeros.default",        # FIXED: was .memory_format (wrong dispatch name)
    "aten.ones.default",         # FIXED: was .memory_format (wrong dispatch name)
    "aten.full.default",
    "aten.empty.memory_format",
    "aten.new_empty.default",    # ADDED: allocates fresh memory on the same device as input
    "aten.new_empty_strided.default",  # ADDED: allocates fresh memory with explicit strides
    "aten.arange.start",
    "aten.arange.default",
    "aten.arange.start_step",    # ADDED: torch.arange(start, end, step)
    "aten.scalar_tensor.default",
}

# Constant-lifting ops: perform memory copy (Inductor lowers to clone) but
# should still be skipped as wildcards during pattern matching.  They are NOT
# fully transparent (they do move data) so they are kept separate from
# ALWAYS_TRANSPARENT for correct FLOP accounting.
LIFT_OPS: Set[str] = {
    "aten.lift_fresh_copy.default",
    "aten.lift_fresh.default",
}

# Ops that always allocate new memory and copy data.
# Kept separate from SHAPE_OPS (guaranteed view-only) so that FLOP/memory
# accounting can distinguish the two classes.  Still skipped as wildcards
# during pattern matching.
#
# DEAD KEYS (kept as documentation, never appear in dispatch traces):
#   "aten.reshape.default"          — contiguous input → dispatches as view.default;
#                                     non-contiguous → clone.default + _unsafe_view.default
#   "aten.contiguous.memory_format" — dispatches as clone.default on non-contiguous input;
#                                     no-op (returns self) on already-contiguous input
# The real copy cost of reshape/contiguous is captured via aten.clone.default.
POTENTIAL_COPY_OPS: Set[str] = {
    "aten.repeat.default",     # always copies: allocates new memory + copies data
    "aten.flip.default",       # always copies: allocates new memory + reorders data
    "aten.clone.default",      # always copies: allocates new memory + copies data
    "aten._to_copy.default",   # device/dtype conversion: allocates new memory + copies data
    "aten.copy_.default",      # in-place copy: allocates new memory + copies data into target
}
PATTERN_SKIP: Set[str] = ALWAYS_TRANSPARENT | SHAPE_OPS | INIT_OPS | LIFT_OPS | POTENTIAL_COPY_OPS


# ─────────────────────────────────────────────────────────────────────────────
# Semantic labels  (module class name → human-readable fusion label)
# ─────────────────────────────────────────────────────────────────────────────
# Ordered list of (regex, label).  First full-match wins (case-insensitive).
# Covers all common transformer architectures without per-model customisation.

SEMANTIC_LABELS: List[Tuple[str, str]] = [
    # ── DeepSeek-V4 Hyper-Connections (must precede attn / mlp matchers) ───
    # Class names come from python.zrt.graph.patches.patch_hc_for_capture():
    # HCPreAttn / HCPostAttn / HCPreFfn / HCPostFfn / HCHead.
    (r".*HCPreAttn.*",                                          "mhc_pre_attn"),
    (r".*HCPostAttn.*",                                         "mhc_post_attn"),
    (r".*HCPreFfn.*",                                           "mhc_pre_ffn"),
    (r".*HCPostFfn.*",                                          "mhc_post_ffn"),
    (r".*HCHead.*",                                             "mhc_head"),
    # ── Norm variants ────────────────────────────────────────────────────────
    (r".*RMSNorm.*|.*RmsNorm.*|.*NormHead.*",                  "rms_norm"),
    (r".*LayerNorm.*",                                          "layer_norm"),
    (r".*L2Norm.*",                                            "rms_norm"),
    # ── Position encoding ────────────────────────────────────────────────────
    (r".*RotaryEmb.*|.*RoPE.*|.*RotaryPosition.*|.*YarnRotary.*"
     r"|.*LlamaRotary.*|.*DynamicNTKScaling.*",                "rope"),
    # ── Attention (after MLA so specific projections matched above) ─────────
    (r".*MLA.*|.*MultiLatentAttn.*",                           "mla_attn"),
    (r".*Attention.*|.*SelfAttn.*|.*MultiHead.*Attn.*",        "attn"),
    # ── MoE gate / router ────────────────────────────────────────────────────
    # DeepSeek-V4: Gate class is named exactly "Gate" (no MoE/Expert prefix).
    # Must precede the compound regex so it wins on exact match.
    (r"Gate",                                                      "moe_gate"),
    (r".*(MoE|Moe|Expert|TopK|Top1|Top2|Sparse|Switch).*(Gate|Router).*", "moe_gate"),
    # ── MoE container / shared expert ────────────────────────────────────────
    (r".*SparseMoeBlock.*|.*MoEBlock.*",                        "moe_block"),
    (r".*MoE.*",                                                 "moe_block"),  # ADDED: DeepseekV2MoE 等纯 MoE 类名
    (r".*SharedExpert.*",                                       "moe_shared"),
    (r".*Expert.*",                                             "moe_expert"),
    # ── Dense FFN / MLP ──────────────────────────────────────────────────────
    (r".*MLP.*|.*FFN.*|.*FeedForward.*|.*PointwiseFF.*",       "mlp"),
    # ── Embedding / head ─────────────────────────────────────────────────────
    (r".*Embed.*",                                              "embedding"),
    (r".*LMHead.*|.*LmHead.*",                                  "lm_head"),
]


def get_semantic_label(module_class: str) -> Optional[str]:
    """Return the semantic fusion label for *module_class*, or ``None``."""
    for pattern, label in SEMANTIC_LABELS:
        if re.fullmatch(pattern, module_class, re.IGNORECASE):
            return label
    return None


# Semantics for "container" modules (attention, MLP, MoE blocks) whose ops
# should only appear as fused_op when a platform subpattern explicitly matches
# them (e.g. "npu_fusion_attention", "flash_attn", "sdpa").  Without a
# subpattern match the display falls back to the actual aten op names so that
# unfused ops are never hidden behind a generic module label.
CONTAINER_SEMANTICS: Set[str] = {
    "attn", "mla_attn",
    "mlp",
    "moe_block", "moe_shared", "moe_expert",
}


# ─────────────────────────────────────────────────────────────────────────────
# Sub-patterns  (platform-specific op-sequence matching within a fused group)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubPattern:
    """A platform-specific fusion-op pattern.

    Matching is two-step:
      1. ``module_re``  — the group's ``module_class`` must full-match this.
      2. ``op_seq``     — the group's effective aten ops (PATTERN_SKIP removed)
                         must contain *op_seq* as an ordered subsequence.
    Both conditions must hold.  An empty ``op_seq`` means "match by class only".
    """
    name: str                      # Fused-op display label, e.g. "flash_attn"
    module_re: str                 # Regex matched against module_class
    op_seq: List[str] = field(default_factory=list)  # aten op regexes (re.search)
    priority: int = 10             # Higher wins when multiple patterns match

    def matches_class(self, module_class: str) -> bool:
        if not module_class:
            return False
        return bool(re.fullmatch(self.module_re, module_class, re.IGNORECASE))

    def matches_ops(self, op_names: List[str]) -> bool:
        if not self.op_seq:
            return True
        return match_subsequence(op_names, self.op_seq)


def match_subsequence(op_names: List[str], pattern: List[str]) -> bool:
    """Return True if *pattern* appears as an *ordered* subsequence.

    Ops in PATTERN_SKIP are excluded from *op_names* before matching.
    Pattern elements are matched in order but need not be contiguous —
    intermediate compute ops (e.g. the QK scaling mul or mask add inside
    scaled dot-product attention) are allowed between anchors.

    This is safe because Pass 1 already groups ops by scope, so all ops in a
    group are from the same module's computation.  Within-module ordering is
    sufficient to uniquely identify kernels (e.g. attention's bmm→softmax→bmm
    vs. MLP's mm→silu→mm).
    """
    effective = [op for op in op_names if op not in PATTERN_SKIP]
    i = 0
    for pat in pattern:
        found = False
        while i < len(effective):
            if re.search(pat, effective[i], re.IGNORECASE):
                i += 1
                found = True
                break
            i += 1
        if not found:
            return False
    return True


# ── Shared module regexes used in pattern definitions ────────────────────────

_ATTN_RE    = r".*Attention.*|.*SelfAttn.*|.*MultiHead.*|.*MLA.*"
_GATE_RE    = r".*Gate.*|.*Router.*|.*MoEGate.*|.*MoeGate.*|.*TopkRouter.*"
_MLP_RE     = r".*MLP.*|.*FFN.*|.*FeedForward.*|.*PointwiseFF.*"
_MOE_RE     = r".*MoE.*|.*SparseMoe.*|.*Expert.*"
_NORM_RE    = r".*RMSNorm.*|.*LayerNorm.*|.*RmsNorm.*"
_EMBED_RE   = r".*Embed.*"

# Broader regex used only for backward patterns: attention backward ops may be
# attributed to parent decoder/encoder/block layers when inner attention module
# backward hooks don't fire (e.g. non-Tensor outputs from the attention module).
# The backward-specific aten ops (_softmax_backward_data, scaled_dot_product_attention_backward)
# are unique to attention backward and serve as discriminators instead.
_ATTN_BWD_RE = (
    r".*Attention.*|.*SelfAttn.*|.*MultiHead.*|.*MLA.*"
    r"|.*DecoderLayer.*|.*EncoderLayer.*|.*TransformerLayer.*|.*Block.*"
)

# ── CUDA patterns (H100 / A100 / A10 family) ─────────────────────────────────
_CUDA_PATTERNS: List[SubPattern] = [
    # ── Backward patterns (priority 50+) ─────────────────────────────────────
    # SDPA backward (single composite op) — use broad regex, op is uniquely backward
    SubPattern("sdpa_backward", _ATTN_BWD_RE,
               [r"_scaled_dot_product.*_backward"],
               priority=50),
    # Attention backward: dQ/dK/dV via mm + softmax_backward + mm
    # Use broad regex: _softmax_backward_data is uniquely diagnostic of attn backward
    SubPattern("attn_grad", _ATTN_BWD_RE,
               [r"\b(mm|bmm|matmul)\b", r"_softmax_backward_data", r"\b(mm|bmm|matmul)\b"],
               priority=42),
    # Native norm backward (fused kernel: LayerNorm / GroupNorm)
    SubPattern("norm_backward", _NORM_RE,
               [r"native_layer_norm_backward|native_group_norm_backward|_fused_rms_norm_backward"],
               priority=38),
    # Embedding backward
    SubPattern("embedding_backward", _EMBED_RE,
               [r"embedding_dense_backward"],
               priority=35),
    # Gated MLP backward (SwiGLU / GeGLU): silu/gelu_backward → mul → mm
    SubPattern("gated_mlp_backward", _MLP_RE,
               [r"\bmul\b", r"silu_backward|gelu_backward", r"\b(mm|addmm)\b"],
               priority=28),
    # Dense MLP backward: activation_backward → mm
    SubPattern("mlp_backward", _MLP_RE,
               [r"threshold_backward|silu_backward|gelu_backward", r"\b(mm|addmm)\b"],
               priority=24),
    # ── Forward patterns ─────────────────────────────────────────────────────
    # DeepSeek-V4 inline q second-norm + RoPE (at Attention scope, not a sub-module).
    # square→rsqrt (inline RMSNorm on q) followed by view_as_complex (RoPE) is unique to V4.
    SubPattern("v4_q_norm", _ATTN_RE,
               [r"square", r"rsqrt", r"view_as_complex"],
               priority=46),
    # DeepSeek-V4 kv RoPE + act_quant + window-topk + cache write (Attention scope).
    # view_as_complex (RoPE on kv) followed by amax/clamp_min (act_quant block-scale) is unique.
    SubPattern("v4_kv_quant", _ATTN_RE,
               [r"view_as_complex", r"(amax|clamp_min)"],
               priority=46),
    # DeepSeek-V4 sparse attention: gather(topk KV) → bmm(QK) → softmax → bmm(AV)
    # 'gather' before the first bmm distinguishes it from dense flash_attn.
    SubPattern("v4_sparse_attn", _ATTN_RE,
               [r"\bgather\b", r"\b(mm|bmm|matmul)\b", r"softmax", r"\b(mm|bmm|matmul)\b"],
               priority=45),
    # SDPA aten 级展开: QK mm → softmax → AV mm (展示为融合后的 SDPA 大算子)
    SubPattern("flash_attn", _ATTN_RE,
               [r"\b(mm|bmm|matmul)\b", r"softmax", r"\b(mm|bmm|matmul)\b"],
               priority=40),
    # Scaled-dot-product-attention (torch.nn.functional.scaled_dot_product_attention)
    SubPattern("sdpa", _ATTN_RE,
               [r"scaled_dot_product_attention"],
               priority=35),
    # MoE gating with top-k selection
    # softplus covers DeepSeek-V4 default score_func="sqrtsoftplus" (F.softplus().sqrt())
    SubPattern("moe_gate_topk", _GATE_RE,
               [r"\b(mm|linear)\b", r"softmax|sigmoid|softplus", r"topk"],
               priority=30),
    # MoE gating without top-k (just scoring; also covers hash-routing layers)
    SubPattern("moe_gate", _GATE_RE,
               [r"\b(mm|linear)\b", r"softmax|sigmoid|softplus"],
               priority=25),
    # SwiGLU / GeGLU gated MLP
    SubPattern("gated_mlp", _MLP_RE,
               [r"\bmm\b", r"silu|gelu|relu", r"\bmul\b", r"\bmm\b"],
               priority=20),
    # MoE expert dispatch (scatter/gather routing)
    SubPattern("moe_dispatch", _MOE_RE,
               [r"topk", r"index_select|gather|scatter"],
               priority=15),
]

# ── Ascend NPU / CANN patterns ────────────────────────────────────────────────
_ASCEND_PATTERNS: List[SubPattern] = [
    # ── Backward patterns ─────────────────────────────────────────────────────
    SubPattern("sdpa_backward", _ATTN_BWD_RE,
               [r"_scaled_dot_product.*_backward"],
               priority=55),
    SubPattern("attn_grad", _ATTN_BWD_RE,
               [r"\b(mm|bmm|matmul)\b", r"_softmax_backward_data", r"\b(mm|bmm|matmul)\b"],
               priority=45),
    SubPattern("norm_backward", _NORM_RE,
               [r"native_layer_norm_backward|native_group_norm_backward"],
               priority=42),
    SubPattern("embedding_backward", _EMBED_RE,
               [r"embedding_dense_backward"],
               priority=38),
    SubPattern("gated_mlp_backward", _MLP_RE,
               [r"\bmul\b", r"silu_backward|gelu_backward", r"\b(mm|addmm)\b"],
               priority=32),
    SubPattern("mlp_backward", _MLP_RE,
               [r"threshold_backward|silu_backward|gelu_backward", r"\b(mm|addmm)\b"],
               priority=28),
    # ── Forward patterns ─────────────────────────────────────────────────────
    # AddRMSNorm: residual add fused into norm (cross-boundary, see fusion.py)
    SubPattern("npu_add_rms_norm", _NORM_RE,
               [r"\badd\b", r"pow|mean|rsqrt|mul"],
               priority=50),
    # DeepSeek-V4 SparseAttnSharedKV → npu_sparse_attn_sharedkv kernel (Ascend)
    # Inference: sparse_attn kernel emits gather before the QK bmm
    # SubPattern("npu_sas", _ATTN_RE,
    #            [r"\bgather\b", r"\b(mm|bmm|matmul)\b", r"softmax",
    #             r"\b(mm|bmm|matmul)\b"],
    #            priority=46),
    # Training: V4 Compressor module performs gated KV pooling — always maps to
    # npu_sparse_attn_sharedkv on Ascend.  The ops are split into multiple groups
    # by topo-sort interleaving with norm sub-module, so class-only matching is
    # used (empty op_seq = match any ops in a Compressor-class group).
    # SubPattern("npu_sas", r".*Compressor.*",
    #            [],
    #            priority=46),
    # DeepSeek-V4 sparse attention fallback label for CUDA (same pattern, lower priority)
    SubPattern("v4_sparse_attn", _ATTN_RE,
               [r"\bgather\b", r"\b(mm|bmm|matmul)\b", r"softmax", r"\b(mm|bmm|matmul)\b"],
               priority=45),
    # 同 CUDA 平台: SDPA aten 级展开 → 不映射到 NPU 融合核，只匹配直达 sdpa 调用
    SubPattern("sdpa", _ATTN_RE,
               [r"scaled_dot_product_attention"],
               priority=35),
    SubPattern("npu_moe_gate_topk", _GATE_RE,
               [r"\b(mm|linear)\b", r"softmax|sigmoid|softplus", r"topk"],
               priority=30),
    SubPattern("npu_moe_gate", _GATE_RE,
               [r"\b(mm|linear)\b", r"softmax|sigmoid|softplus"],
               priority=25),
    SubPattern("gated_mlp", _MLP_RE,
               [r"\bmm\b", r"silu|gelu|relu", r"\bmul\b", r"\b(mm|addmm)\b"],
               priority=20),
    SubPattern("npu_moe_dispatch", _MOE_RE,
               [r"topk", r"index_select|scatter|gather"],
               priority=15),
]

# ── CPU patterns ──────────────────────────────────────────────────────────────
_CPU_PATTERNS: List[SubPattern] = [
    # ── Backward patterns ─────────────────────────────────────────────────────
    SubPattern("sdpa_backward", _ATTN_BWD_RE,
               [r"_scaled_dot_product.*_backward"],
               priority=50),
    SubPattern("attn_grad", _ATTN_BWD_RE,
               [r"\b(mm|bmm|matmul)\b", r"_softmax_backward_data", r"\b(mm|bmm|matmul)\b"],
               priority=30),
    SubPattern("norm_backward", _NORM_RE,
               [r"native_layer_norm_backward|native_group_norm_backward"],
               priority=38),
    SubPattern("embedding_backward", _EMBED_RE,
               [r"embedding_dense_backward"],
               priority=35),
    SubPattern("gated_mlp_backward", _MLP_RE,
               [r"\bmul\b", r"silu_backward|gelu_backward", r"\b(mm|addmm)\b"],
               priority=28),
    SubPattern("mlp_backward", _MLP_RE,
               [r"threshold_backward|silu_backward|gelu_backward", r"\b(mm|addmm)\b"],
               priority=24),
    # ── Forward patterns ─────────────────────────────────────────────────────
    SubPattern("sdpa", _ATTN_RE,
               [r"scaled_dot_product_attention"],
               priority=25),
    SubPattern("moe_gate_topk", _GATE_RE,
               [r"\b(mm|linear)\b", r"softmax|sigmoid|softplus", r"topk"],
               priority=25),
    SubPattern("moe_gate", _GATE_RE,
               [r"\b(mm|linear)\b", r"softmax|sigmoid|softplus"],
               priority=20),
]

PLATFORM_SUBPATTERNS: Dict[str, List[SubPattern]] = {
    "cuda":        _CUDA_PATTERNS,
    "ascend_npu":  _ASCEND_PATTERNS,
    "cpu":         _CPU_PATTERNS,
    "generic":     [],
}

# ─────────────────────────────────────────────────────────────────────────────
# Platform settings
# ─────────────────────────────────────────────────────────────────────────────
# max_parent_ops:  refuse parent-merge when total child ops exceed this.
# max_children:    refuse parent-merge when unique child paths exceed this.
# add_norm_fusion: whether to detect cross-boundary Add+Norm → AddRMSNorm.

PLATFORM_SETTINGS: Dict[str, Dict] = {
    "cuda":        {"max_parent_ops": 60, "max_children": 8,  "add_norm_fusion": True},
    "ascend_npu":  {"max_parent_ops": 50, "max_children": 8,  "add_norm_fusion": True},
    "cpu":         {"max_parent_ops": 20, "max_children": 6,  "add_norm_fusion": False},
    "generic":     {"max_parent_ops": 30, "max_children": 5,  "add_norm_fusion": False},
}


def get_subpatterns(platform: str) -> List[SubPattern]:
    """Return sub-patterns for *platform* sorted by priority (descending)."""
    patterns = PLATFORM_SUBPATTERNS.get(platform, [])
    return sorted(patterns, key=lambda p: -p.priority)


def get_platform_settings(platform: str) -> Dict:
    return PLATFORM_SETTINGS.get(platform, PLATFORM_SETTINGS["generic"])
