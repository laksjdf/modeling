"""Shape description generator — converts TensorMeta lists to human-readable strings.

Produces compact shape descriptors for display in report tables.
Supports all operator categories: mm, norm, attention, softmax, elemwise,
communication, embedding, and generic fallback.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.ir.node import OpNode
    from python.zrt.ir.types import TensorMeta


def describe_shapes(node: "OpNode") -> str:
    """Generate a human-readable shape descriptor for an OpNode.

    Examples
    --------
    aten.mm.default        → "M=128, K=7168, N=2048"
    aten.addmm.default     → "M=128, K=7168, N=2048"
    aten.bmm.default       → "B=32, M=128, K=7168, N=2048"
    aten.rms_norm / fused  → "N=7168"
    aten._softmax          → "N=128"
    aten.silu / gelu       → "N=128"
    comm.all_reduce        → "data=7.2MB, group=4"
    aten.embedding         → "tokens=128, hidden=7168"
    flash_attn / sdpa      → "B=1, H=24, Sq=8192, Sk=8192, D=128"
    unknown                 → "in=[128,7168] out=[128,2048]"
    """

    op_type = node.op_type

    # ── Communication ────────────────────────────────────────────────────────
    if op_type.startswith("comm.") or node.category == "communication":
        return _desc_comm(node)

    # ── Fused semantic labels ────────────────────────────────────────────────
    if is_attention_like(op_type):
        return _desc_attention(node)
    if "embedding" in op_type.lower() or "embed" in op_type.lower():
        return _desc_embedding(node)

    # ── aten ops by pattern ──────────────────────────────────────────────────
    op_short = _op_short(op_type)

    if op_short in ("mm", "matmul"):
        return _desc_mm(node)
    if op_short in ("addmm",):
        return _desc_mm(node)
    if op_short in ("bmm",):
        return _desc_bmm(node)
    if op_short in ("linear",):
        return _desc_linear(node)
    if _is_norm(op_short, op_type):
        return _desc_norm(node)
    if "softmax" in op_short:
        return _desc_scalar(node, "N")
    if _is_elemwise(op_short, op_type):
        return _desc_scalar(node, "N")
    if "rope" in op_short.lower() or "rotary" in op_type.lower():
        return _desc_scalar(node, "N")

    # ── Fallback: show input/output shapes ───────────────────────────────────
    return _desc_fallback(node)


def describe_shapes_from_tensors(
    inputs: list["TensorMeta"],
    outputs: list["TensorMeta"],
    op_type: str = "",
) -> str:
    """Generate shape descriptor directly from tensor lists (no OpNode needed)."""
    in_shapes = ", ".join(_fmt_shape(t) for t in inputs)
    out_shapes = ", ".join(_fmt_shape(t) for t in outputs)
    if not in_shapes and not out_shapes:
        return ""
    return f"in=[{in_shapes}] out=[{out_shapes}]"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

_MM_RE = re.compile(r"(mm|matmul|addmm)", re.IGNORECASE)
_BMM_RE = re.compile(r"(bmm|batch_matmul)", re.IGNORECASE)
_NORM_RE = re.compile(
    r"(rms_norm|layer_norm|norm|add_rms|npu_add_rms)", re.IGNORECASE,
)
_ELEM_RE = re.compile(
    r"^(add|sub|mul|div|neg|abs|relu|tanh|exp|log|sqrt|rsqrt|pow|silu|gelu|sigmoid|"
    r"reciprocal|clamp|mean|sum|amax|amin|masked_fill|cumsum|cumprod|var|"
    r"copy_|clone|view|reshape|transpose|permute|slice|select|expand|cat|stack|split)$",
    re.IGNORECASE,
)
_ATTN_RE = re.compile(
    r"(flash_attn|sdpa|sparse_attn|mla_attn|v4_sparse|npu_sdpa|attention|attn)",
    re.IGNORECASE,
)


def _op_short(op_type: str) -> str:
    """Extract the short op name from a full aten op_type."""
    # "aten.mm.default" → "mm"
    parts = op_type.split(".")
    if len(parts) >= 2 and parts[0] == "aten":
        return parts[1]
    return parts[-1]


def _fmt_shape(t: "TensorMeta") -> str:
    """Format a TensorMeta as a short shape string."""
    shape = getattr(t, "shape", None)
    if shape is None:
        return "?"
    return "[" + ",".join(str(d) for d in shape) + "]"


def _first_input(node: "OpNode") -> "TensorMeta | None":
    return node.inputs[0] if node.inputs else None


def _first_output(node: "OpNode") -> "TensorMeta | None":
    return node.outputs[0] if node.outputs else None


def _second_input(node: "OpNode") -> "TensorMeta | None":
    return node.inputs[1] if len(node.inputs) > 1 else None


def _get_shape(t: "TensorMeta | None") -> tuple[int, ...]:
    if t is None:
        return ()
    return getattr(t, "shape", ())


def _mem_bytes(t: "TensorMeta | None") -> int:
    if t is None:
        return 0
    return getattr(t, "mem_bytes", 0)


def is_attention_like(op_type: str) -> bool:
    return bool(_ATTN_RE.search(op_type))


def _is_norm(op_short: str, op_type: str) -> bool:
    return bool(_NORM_RE.search(op_short)) or bool(_NORM_RE.search(op_type))


def _is_elemwise(op_short: str, op_type: str) -> bool:
    return bool(_ELEM_RE.match(op_short)) or bool(_ELEM_RE.match(op_type))


# ── Descriptor generators per op category ────────────────────────────────────


def _desc_mm(node: "OpNode") -> str:
    """aten.mm / aten.matmul → M=..., K=..., N=..."""
    a = _first_input(node)
    b = _second_input(node)
    o = _first_output(node)
    ashape = _get_shape(a)
    bshape = _get_shape(b)
    oshape = _get_shape(o)

    # M×K @ K×N → M×N
    if len(ashape) >= 2 and len(bshape) >= 2:
        m = ashape[-2] if len(ashape) >= 2 else "?"
        k = ashape[-1] if len(ashape) >= 1 else "?"
        n = bshape[-1] if len(bshape) >= 1 else "?"
        return f"M={m}, K={k}, N={n}"
    return _desc_fallback(node)


def _desc_bmm(node: "OpNode") -> str:
    """aten.bmm → B=..., M=..., K=..., N=..."""
    a = _first_input(node)
    b = _second_input(node)
    ashape = _get_shape(a)
    bshape = _get_shape(b)

    if len(ashape) >= 3 and len(bshape) >= 3:
        b_ = ashape[-3] if len(ashape) >= 3 else "?"
        m = ashape[-2]
        k = ashape[-1]
        n = bshape[-1]
        return f"B={b_}, M={m}, K={k}, N={n}"
    return _desc_mm(node)  # fallback to mm


def _desc_linear(node: "OpNode") -> str:
    """aten.linear → batch=..., I=..., O=..."""
    a = _first_input(node)
    o = _first_output(node)
    ashape = _get_shape(a)
    oshape = _get_shape(o)

    if len(ashape) >= 2 and len(oshape) >= 2:
        batch = ashape[0]
        inp = ashape[-1]
        out = oshape[-1]
        return f"batch={batch}, I={inp}, O={out}"
    return _desc_fallback(node)


def _desc_norm(node: "OpNode") -> str:
    """RMSNorm / LayerNorm → N=..."""
    a = _first_input(node)
    ashape = _get_shape(a)
    if ashape:
        return f"N={ashape[-1]}"
    return _desc_fallback(node)


def _desc_scalar(node: "OpNode", dim_label: str = "N") -> str:
    """Element-wise / softmax → N=..."""
    a = _first_input(node)
    ashape = _get_shape(a)
    if ashape:
        return f"{dim_label}={ashape[-1]}"
    return _desc_fallback(node)


def _desc_attention(node: "OpNode") -> str:
    """Flash Attention / SDPA → B=..., H=..., Sq=..., Sk=..., D=..."""
    # Try to extract from first input (Q tensor: [B, H, Sq, D])
    a = _first_input(node)
    ashape = _get_shape(a)

    if len(ashape) >= 4:
        return f"B={ashape[-4]}, H={ashape[-3]}, Sq={ashape[-2]}, D={ashape[-1]}"
    if len(ashape) >= 3:
        return f"B={ashape[0]}, H={ashape[1]}, Sq={ashape[2]}"
    # Check annotations for shape hints
    hints = getattr(node, "annotations", {})
    if hints:
        parts = []
        for k in ("batch", "heads", "seq_q", "seq_k", "head_dim"):
            if k in hints:
                parts.append(f"{k}={hints[k]}")
        if parts:
            return ", ".join(parts)
    return _desc_fallback(node)


def _desc_embedding(node: "OpNode") -> str:
    """Embedding lookup → tokens=..., hidden=..."""
    a = _first_input(node)
    o = _first_output(node)
    ashape = _get_shape(a)
    oshape = _get_shape(o)

    # Input: [tokens], Output: [tokens, hidden]
    if ashape and oshape:
        tokens = ashape[-1] if len(ashape) >= 1 else "?"
        hidden = oshape[-1] if len(oshape) >= 2 else "?"
        return f"tokens={tokens}, hidden={hidden}"
    return _desc_fallback(node)


def _desc_comm(node: "OpNode") -> str:
    """Communication op → data=X MB, group=N"""
    total_in = sum(_mem_bytes(t) for t in node.inputs)
    total_out = sum(_mem_bytes(t) for t in node.outputs)
    data_bytes = max(total_in, total_out)

    group_size = node.attrs.get("group_size", "?")

    if data_bytes >= 1024 * 1024:
        return f"data={data_bytes / (1024 * 1024):.1f}MB, group={group_size}"
    elif data_bytes >= 1024:
        return f"data={data_bytes / 1024:.1f}KB, group={group_size}"
    return f"data={data_bytes}B, group={group_size}"


def _desc_fallback(node: "OpNode") -> str:
    """Generic fallback: in=[...] out=[...]."""
    in_str = ", ".join(_fmt_shape(t) for t in node.inputs)
    out_str = ", ".join(_fmt_shape(t) for t in node.outputs)
    return f"in=[{in_str}] out=[{out_str}]"
