"""FormulaRegistry — maps operator type to display formula strings.

Extracted from python/zrt/simulator/backends/roofline.py (comprehensive
operator formula table, lines 12-200+).

The registry provides:
  - display_name:  human-readable, e.g. "Matrix Multiply"
  - flops_formula: FLOPs formula string, e.g. "2·M·K·N"
  - io_formula:    I/O pattern string,   e.g. "R=(M·K+K·N)·dtype  W=M·N·dtype"
  - notes:         optional commentary
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# FormulaEntry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FormulaEntry:
    """One entry in the formula registry."""

    op_pattern: str         # regex applied to op_type (re.search)
    display_name: str       # "Matrix Multiply" | "AllReduce" | "RMS Norm"
    category: str           # "compute" | "communication" | "memory"
    flops_formula: str      # e.g. "2·M·K·N"
    io_formula: str         # e.g. "R=(M·K+K·N)·dtype  W=M·N·dtype"
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Registry (ordered — first match wins)
# ─────────────────────────────────────────────────────────────────────────────

# fmt: off
_FORMULA_ENTRIES: list[FormulaEntry] = [
    # ── Matrix Multiply (GEMM) ──────────────────────────────────────────────

    FormulaEntry(
        op_pattern=r"aten\.addmm",
        display_name="AddMM",
        category="compute",
        flops_formula="2·M·K·N + M·N",
        io_formula="R=(M·K+K·N+β)·dtype  W=M·N·dtype",
        notes="mm + bias add",
    ),
    FormulaEntry(
        op_pattern=r"aten\.bmm",
        display_name="Batch Matrix Multiply",
        category="compute",
        flops_formula="2·B·M·K·N",
        io_formula="R=(B·M·K+B·K·N)·dtype  W=B·M·N·dtype",
    ),
    FormulaEntry(
        op_pattern=r"aten\.linear",
        display_name="Linear",
        category="compute",
        flops_formula="2·batch·I·O [+ batch·O if bias]",
        io_formula="R=(batch·I+O·I)·dtype  W=batch·O·dtype",
    ),
    FormulaEntry(
        op_pattern=r"aten\.(mm|matmul)",
        display_name="Matrix Multiply",
        category="compute",
        flops_formula="2·M·K·N",
        io_formula="R=(M·K+K·N)·dtype  W=M·N·dtype",
    ),

    # ── Convolution ─────────────────────────────────────────────────────────

    FormulaEntry(
        op_pattern=r"aten\.(convolution|_convolution|conv2d|conv3d)",
        display_name="Convolution",
        category="compute",
        flops_formula="2·N·Cout·Hout·Wout·Cin·Kh·Kw",
        io_formula="R=input+weight+bias  W=output",
        notes="groups → Cin/groups",
    ),

    # ── Attention ───────────────────────────────────────────────────────────

    FormulaEntry(
        op_pattern=r"(aten\.)?_?scaled_dot_product_attention",
        display_name="Scaled Dot-Product Attention",
        category="compute",
        flops_formula="4·N·H·Sq·Sk·D + 5·N·H·Sq·Sk",
        io_formula="R=(Q+K+V)·dtype  W=output·dtype",
        notes="QK matmul + softmax + AV matmul",
    ),
    FormulaEntry(
        op_pattern=r"(flash_attn|sdpa|npu_sdpa_decomposed)",
        display_name="Flash Attention",
        category="compute",
        flops_formula="4·N·H·Sq·Sk·D + 5·N·H·Sq·Sk",
        io_formula="R=(Q+K+V)·dtype  W=output·dtype",
    ),
    FormulaEntry(
        op_pattern=r"sparse_attn|v4_sparse_attn",
        display_name="Sparse Attention",
        category="compute",
        flops_formula="(4·N·H·Sq·Sk·D)·sparsity_ratio",
        io_formula="R=(Q+ratio·K+ratio·V)·dtype  W=output·dtype",
    ),
    FormulaEntry(
        op_pattern=r"mla_attn",
        display_name="MLA Attention",
        category="compute",
        flops_formula="(4·N·H·Sq·Sk·D)·sparsity_ratio",
        io_formula="ratio from annotations",
    ),
    FormulaEntry(
        op_pattern=r"(sdpa_backward|attn_grad)",
        display_name="Attention Backward",
        category="compute",
        flops_formula="4·N·H·Sq·Sk·D + 5·N·H·Sq·Sk  (×2–3)",
        io_formula="grad shapes",
    ),

    # ── Norm ────────────────────────────────────────────────────────────────

    FormulaEntry(
        op_pattern=r"add_rms_norm|add_layer_norm|npu_add_rms",
        display_name="Add + RMS Norm",
        category="compute",
        flops_formula="6·N",
        io_formula="R=(2·N+|weight|)·dtype  W=N·dtype",
        notes="residual add (1N) + rms_norm (5N)",
    ),
    FormulaEntry(
        op_pattern=r"rms_norm|gemma_rms_norm",
        display_name="RMS Norm",
        category="compute",
        flops_formula="4·N",
        io_formula="R=(N+|weight|)·dtype  W=N·dtype",
        notes="sq+mean+rsqrt+scale",
    ),
    FormulaEntry(
        op_pattern=r"rms_gated|rms_norm_gated",
        display_name="Gated RMS Norm",
        category="compute",
        flops_formula="9·N",
        io_formula="R=(N+|weight|+|gate|)·dtype  W=N·dtype",
        notes="rms_norm (4N) + sigmoid (4N) + mul (1N)",
    ),
    FormulaEntry(
        op_pattern=r"layer_norm|aten\.(layer_norm|native_layer_norm)",
        display_name="Layer Norm",
        category="compute",
        flops_formula="5·N",
        io_formula="R=(N+2·|weight|)·dtype  W=N·dtype",
        notes="mean+var+norm+scale+shift",
    ),
    FormulaEntry(
        op_pattern=r"norm_backward",
        display_name="Norm Backward",
        category="compute",
        flops_formula="6·N",
        io_formula="R=grad+input+weight  W=grad",
    ),

    # ── Softmax ─────────────────────────────────────────────────────────────

    FormulaEntry(
        op_pattern=r"_?softmax",
        display_name="Softmax",
        category="compute",
        flops_formula="5·N",
        io_formula="R=N·dtype  W=N·dtype",
        notes="max+sub+exp+sum+div",
    ),

    # ── TopK / Sort ─────────────────────────────────────────────────────────

    FormulaEntry(
        op_pattern=r"aten\.topk",
        display_name="TopK",
        category="compute",
        flops_formula="2·N·log₂(N)",
        io_formula="R=N·dtype  W=(values+indices)·dtype",
        notes="compare + swap",
    ),
    FormulaEntry(
        op_pattern=r"aten\.sort",
        display_name="Sort",
        category="compute",
        flops_formula="2·N·log₂(N)",
        io_formula="R=N·dtype  W=(values+indices)·dtype",
    ),

    # ── Element-wise (1 op/elem) ────────────────────────────────────────────

    FormulaEntry(
        op_pattern=r"aten\.(add|sub|rsub|mul|div|neg)[^m_]",
        display_name="Element-wise Arithmetic",
        category="compute",
        flops_formula="1·N",
        io_formula="R=Σ|inputs|·dtype  W=|output|·dtype",
    ),
    FormulaEntry(
        op_pattern=r"aten\.(abs|relu|tanh|exp|log|sqrt|rsqrt|pow|masked_fill)",
        display_name="Element-wise Unary",
        category="compute",
        flops_formula="1·N",
        io_formula="R=|input|·dtype  W=|output|·dtype",
    ),
    FormulaEntry(
        op_pattern=r"aten\.(mean|sum|amax|amin)\b",
        display_name="Reduction",
        category="compute",
        flops_formula="1·N",
        io_formula="R=N·dtype  W=scalar·dtype",
    ),
    FormulaEntry(
        op_pattern=r"aten\.cumsum|cumprod",
        display_name="Cumulative Scan",
        category="compute",
        flops_formula="1·N",
        io_formula="R=N·dtype  W=N·dtype",
    ),
    FormulaEntry(
        op_pattern=r"aten\.copy_",
        display_name="Copy",
        category="memory",
        flops_formula="0",
        io_formula="R=input·dtype  W=output·dtype",
    ),

    # ── Element-wise (2 op/elem) ───────────────────────────────────────────

    FormulaEntry(
        op_pattern=r"aten\.(reciprocal|clamp|clamp_min|clamp_max)",
        display_name="Element-wise Clamp",
        category="compute",
        flops_formula="2·N",
        io_formula="R=N·dtype  W=N·dtype",
    ),
    FormulaEntry(
        op_pattern=r"(rope|rotary)",
        display_name="RoPE",
        category="compute",
        flops_formula="2·N",
        io_formula="R=input·dtype  W=output·dtype",
        notes="cos·x + sin·x_rot",
    ),
    FormulaEntry(
        op_pattern=r"aten\.var",
        display_name="Variance",
        category="compute",
        flops_formula="3·N",
        io_formula="R=N·dtype  W=scalar·dtype",
        notes="sq+mean+sub+sq+mean",
    ),

    # ── Activation (4 op/elem) ─────────────────────────────────────────────

    FormulaEntry(
        op_pattern=r"(aten\.)?silu|SiLU",
        display_name="SiLU",
        category="compute",
        flops_formula="4·N",
        io_formula="R=N·dtype  W=N·dtype",
        notes="x·σ(x), σ≈4 ops",
    ),
    FormulaEntry(
        op_pattern=r"(aten\.)?gelu($|[^_\w])",
        display_name="GELU",
        category="compute",
        flops_formula="4·N",
        io_formula="R=N·dtype  W=N·dtype",
        notes="~x·Φ(x), ≈4 ops",
    ),
    FormulaEntry(
        op_pattern=r"aten\.sigmoid",
        display_name="Sigmoid",
        category="compute",
        flops_formula="4·N",
        io_formula="R=N·dtype  W=N·dtype",
        notes="1/(1+e^-x)",
    ),

    # ── Embedding ───────────────────────────────────────────────────────────

    FormulaEntry(
        op_pattern=r"(aten\.)?embedding|embedding_or_input|Embedding",
        display_name="Embedding Lookup",
        category="memory",
        flops_formula="0",
        io_formula="R=V·D·dtype  W=T·D·dtype",
        notes="gather from weight matrix",
    ),

    # ── Communication ──────────────────────────────────────────────────────

    FormulaEntry(
        op_pattern=r"comm\.all_reduce",
        display_name="AllReduce",
        category="communication",
        flops_formula="—",
        io_formula="2·(n-1)/n·D",
        notes="Ring algorithm (default)",
    ),
    FormulaEntry(
        op_pattern=r"comm\.all_gather",
        display_name="AllGather",
        category="communication",
        flops_formula="—",
        io_formula="(n-1)/n·D",
        notes="Ring or Tree",
    ),
    FormulaEntry(
        op_pattern=r"comm\.reduce_scatter",
        display_name="ReduceScatter",
        category="communication",
        flops_formula="—",
        io_formula="(n-1)/n·D",
        notes="Ring algorithm",
    ),
    FormulaEntry(
        op_pattern=r"comm\.all_to_all",
        display_name="All-to-All",
        category="communication",
        flops_formula="—",
        io_formula="(n-1)/n²·D",
        notes="pairwise exchange",
    ),
    FormulaEntry(
        op_pattern=r"comm\.send_recv",
        display_name="Send/Recv (P2P)",
        category="communication",
        flops_formula="—",
        io_formula="D / BW",
        notes="point-to-point activation transfer",
    ),
    FormulaEntry(
        op_pattern=r"comm\.broadcast",
        display_name="Broadcast",
        category="communication",
        flops_formula="—",
        io_formula="D / BW · log₂(n)",
        notes="tree-based",
    ),

    # ── Fused / composite (semantic labels from FusionPass) ─────────────────

    FormulaEntry(
        op_pattern=r"gated_mlp",
        display_name="Gated MLP (SwiGLU)",
        category="compute",
        flops_formula="8·I·O",
        io_formula="R=3·I·O·dtype  W=I·O·dtype",
        notes="gate_proj + up_proj + silu + mul + down_proj",
    ),
    FormulaEntry(
        op_pattern=r"moe_gate_topk",
        display_name="MoE Gate + TopK",
        category="compute",
        flops_formula="2·T·H·E + T·E·log₂(E)",
        io_formula="R=T·H·dtype  W=T·K·dtype",
        notes="linear scoring + topk selection",
    ),
    FormulaEntry(
        op_pattern=r"moe_gate",
        display_name="MoE Gate",
        category="compute",
        flops_formula="2·T·H·E",
        io_formula="R=T·H·dtype  W=T·E·dtype",
    ),
    FormulaEntry(
        op_pattern=r"moe_dispatch|npu_moe_dispatch",
        display_name="MoE Dispatch",
        category="communication",
        flops_formula="—",
        io_formula="T·H·dtype / EP",
        notes="token routing to expert devices",
    ),
    FormulaEntry(
        op_pattern=r"moe_shared|shared_expert",
        display_name="Shared Expert",
        category="compute",
        flops_formula="8·I·O",
        io_formula="R=3·I·O·dtype  W=I·O·dtype",
    ),

    # ── lm_head / output projection ─────────────────────────────────────────

    FormulaEntry(
        op_pattern=r"lm_head",
        display_name="LM Head",
        category="compute",
        flops_formula="2·batch·H·V",
        io_formula="R=(batch·H+H·V)·dtype  W=batch·V·dtype",
    ),

    # ── Catch-all (must be last) ────────────────────────────────────────────

    FormulaEntry(
        op_pattern=r"aten\.",
        display_name="ATen Operator",
        category="compute",
        flops_formula="?",
        io_formula="?",
        notes="generic aten op — no formula registered",
    ),
]
# fmt: on


# ─────────────────────────────────────────────────────────────────────────────
# FormulaRegistry
# ─────────────────────────────────────────────────────────────────────────────

class FormulaRegistry:
    """Lookup display formulas by operator type string.

    Usage::

        reg = FormulaRegistry()
        entry = reg.lookup("aten.mm.default")
        print(entry.display_name)   # "Matrix Multiply"
        print(entry.flops_formula)  # "2·M·K·N"
    """

    def __init__(self) -> None:
        self._entries = _FORMULA_ENTRIES

    def lookup(self, op_type: str) -> FormulaEntry | None:
        """Return the first FormulaEntry whose pattern matches ``op_type``.

        ``None`` if no entry matches (should not happen due to catch-all).
        """
        for entry in self._entries:
            if re.search(entry.op_pattern, op_type, re.IGNORECASE):
                return entry
        return None

    def display_info(self, op_type: str) -> dict[str, str]:
        """Convenience: return {display_name, category, flops_formula, io_formula}."""
        entry = self.lookup(op_type)
        if entry is None:
            return {
                "display_name": op_type.split(".")[-1],
                "category": "compute",
                "flops_formula": "?",
                "io_formula": "?",
            }
        return {
            "display_name": entry.display_name,
            "category": entry.category,
            "flops_formula": entry.flops_formula,
            "io_formula": entry.io_formula,
        }
