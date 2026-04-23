"""Roofline model simulator — universal fallback backend.

Latency model
-------------
    latency = max(FLOPs / peak_flops,  bytes / hbm_bandwidth,  1e-3 µs)

The bound column in SimResult tells you which term dominates.

算子全景计算公式
================

┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 分类           │ 代表算子 / op_type                     │ FLOPs 公式                            │
│                │                                        │ 读带宽 (R) / 写带宽 (W)               │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 矩阵乘 (GEMM)                                                                                   │
│   mm           │ aten.mm / aten.matmul                  │ 2·M·K·N                               │
│                │                                        │ R=(M·K+K·N)·b  W=M·N·b               │
│   addmm        │ aten.addmm                             │ 2·M·K·N + M·N  (mm + bias add)        │
│                │                                        │ R=(M·K+K·N+|bias|)·b  W=M·N·b        │
│   bmm          │ aten.bmm                               │ 2·B·M·K·N                             │
│                │                                        │ R=(B·M·K+B·K·N)·b  W=B·M·N·b         │
│   linear       │ aten.linear                            │ 2·batch·I·O [+ batch·O if bias]       │
│                │ (input=(*,I), weight=(O,I))             │ R=(batch·I+O·I)·b  W=batch·O·b        │
│   Linear       │ FusionPass 融合的 nn.Linear            │ 2·batch·I·N [+ batch·N if bias]       │
│   lm_head      │ (input=(*,I), weight=(I,N) transposed) │ R=(batch·I+I·N)·b  W=batch·N·b        │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 注意力 (Attention)                                                                              │
│   sdpa         │ aten.scaled_dot_product_attention       │ 4·N·H·Sq·Sk·D + 5·N·H·Sq·Sk          │
│   flash_attn   │ aten._scaled_dot_product_flash_attn    │   (QK matmul + AV matmul + softmax)   │
│   mla_attn     │ flash_attn / sdpa / mla_attn           │ R=(Q+K+V)·b  W=output·b               │
│   sdpa_backward│ attn_grad                              │ 同上 (backward 代入 grad shape)        │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 归一化 (Norm)                                                                                   │
│   rms_norm     │ rms_norm                               │ 4·N  (sq+mean+rsqrt+scale)            │
│                │                                        │ R=(N+|weight|)·b  W=N·b               │
│   layer_norm   │ aten.layer_norm / layer_norm           │ 5·N  (mean+var+norm+scale+shift)       │
│                │                                        │ R=(N+2·|weight|)·b  W=N·b             │
│   add_rms_norm │ add_rms_norm / add_layer_norm          │ 6·N  (norm×5 + residual add)           │
│   npu_add_rms  │ npu_add_rms_norm                       │ R=(2·N+|weight|)·b  W=N·b             │
│   norm_backward│ norm_backward                          │ 同 add_rms_norm                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Softmax                                                                                         │
│                │ aten._softmax / aten.softmax.int        │ 5·N  (max+sub+exp+sum+div)            │
│                │                                        │ R=N·b  W=N·b                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 逐元素 — 1 op/elem                                                                              │
│                │ add / sub / rsub / mul / div / neg /   │ 1·N                                   │
│                │ abs / relu / tanh / exp / log /        │ R=sum(|inputs|)·b                     │
│                │ sqrt / rsqrt / pow / masked_fill        │ W=|output|·b                          │
│                │ mean / sum / amax / amin (reduction)    │                                       │
│                │ copy_                                   │ 0 FLOPs, R=input·b, W=output·b        │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 逐元素 — 2 ops/elem                                                                             │
│                │ reciprocal / clamp / clamp_min/max     │ 2·N                                   │
│                │ var (reduce: sq+mean+sub+sq+mean ≈ 3N) │ 3·N                                   │
│                │ rope (cos*x + sin*x_rot)               │ 2·N                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 激活 — 4 ops/elem                                                                               │
│                │ silu (x·σ(x), σ≈4 ops)                │ 4·N                                   │
│                │ gelu  (~x·Φ(x), ≈4 ops)               │ 4·N                                   │
│                │ sigmoid (1/(1+e^-x))                   │ 4·N                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 超越函数 — 10 ops/elem (CORDIC / polynomial)                                                   │
│                │ sin / cos / atan2  (用于 RoPE)          │ 10·N                                  │
│                │                                        │ R=input·b  W=output·b                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ MLP / 专家层 (MLP / MoE expert)                                                                 │
│   gated_mlp    │ gated_mlp / mlp                        │ Σ 2·batch·H·Oᵢ + 4·N_act/2           │
│   moe_block    │ gated_mlp_backward / mlp_backward      │   (按权重矩阵累加 GEMM FLOPs          │
│   moe_expert   │ moe_expert / moe_shared / moe_block    │    + gated activation 代价)           │
│                │                                        │ R=hidden+Σweights  W=output           │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ MoE 路由 (MoE gate / router)                                                                   │
│   moe_gate     │ moe_gate / npu_moe_gate                │ linear FLOPs + 5·N (softmax)          │
│   moe_gate_topk│ moe_gate_topk / npu_moe_gate_topk      │   + N (topk, if with_topk=True)       │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ MoE Dispatch (scatter / gather 路由)                                                            │
│   moe_dispatch │ moe_dispatch / npu_moe_dispatch         │ 0 FLOPs (索引操作)                   │
│                │ aten.index / gather / scatter           │ R=sum(inputs)·b  W=|output|·b         │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Embedding / 查表                                                                                │
│   embedding    │ aten.embedding / embedding              │ 0 FLOPs (随机 HBM 读取)              │
│   embedding_bwd│ embedding_backward                     │ R=|output|·b  W=|output|·b            │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Dtype 转换                                                                                      │
│                │ aten._to_copy (cast / device copy)     │ 0 FLOPs                               │
│                │                                        │ R=|input|·b_in  W=|output|·b_out      │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 分配 / 填充 (write-only)                                                                        │
│                │ new_empty / new_empty_strided           │ 0 FLOPs, R=0                          │
│                │ fill_ / zero_                          │ W=|output|·b                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Shape / View (透明算子)                                                                         │
│                │ view / reshape / expand / squeeze /    │ ≈ 0 FLOPs                             │
│                │ permute / transpose / contiguous /     │ R≈|output|·b  W≈|output|·b            │
│                │ flatten / select / slice / cat / stack │ (视内存是否连续, 实际可能为0)           │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 兜底 (fallback)                                                                                 │
│                │ 任意未覆盖算子                          │ 1·N_out (保守估计)                    │
│                │                                        │ R=total_input_bytes  W=total_output   │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

符号说明
--------
  N         = numel(tensor)  元素总数
  b / b_in / b_out = dtype.itemsize  字节宽度 (bf16=2, fp32=4, ...)
  batch     = numel(input.shape[:-1])  最后一维以外的所有维度乘积
  M,K,N     = 矩阵维度 (rows, common, cols)
  B,H,Sq,Sk,D = attention 批次/头数/Query长/Key长/头维度
  H,I       = MLP hidden_size / intermediate_size
  Oᵢ        = 第 i 个权重矩阵的输出维度

分类覆盖说明
------------
  _EXACT_FORMULAS  精确匹配表覆盖 ~108 个 op_type 字符串 (aten 原始算子 + FusionPass 语义标签)
  _fused_decompose 对 is_fused=True 且无精确匹配的节点按 fused_from 子算子累加 FLOPs
  _SHAPE_OP_PREFIXES 前缀表, 透明算子跳过 FLOPs 计算
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from python.zrt.ir.types import DType
from python.zrt.simulator.base import OpSimulator
from python.zrt.simulator.result import SimResult

if TYPE_CHECKING:
    from python.zrt.ir.node import OpNode
    from python.zrt.hardware.spec import HardwareSpec


# ── helpers ───────────────────────────────────────────────────────────────────

def _numel(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    n = 1
    for d in shape:
        n *= max(d, 0)
    return n


def _primary_dtype(node: "OpNode") -> DType:
    """Return the dominant dtype for compute-throughput lookup."""
    if node.outputs:
        return node.outputs[0].dtype
    if node.inputs:
        return node.inputs[0].dtype
    return DType.BF16


def _itemsize(node: "OpNode") -> float:
    return _primary_dtype(node).itemsize


# ── per-op formula functions ──────────────────────────────────────────────────
# Each returns (flops: float, read_bytes: float, write_bytes: float)

FMR = tuple[float, float, float]   # (flops, read_bytes, write_bytes)


def _mm(node: "OpNode") -> FMR:
    """aten.mm.default: A=(M,K) @ B=(K,N) → (M,N)
    FLOPs = 2·M·K·N   R=(M·K+K·N)·b   W=M·N·b
    """
    if len(node.inputs) < 2:
        return _default(node)
    a, b = node.inputs[0], node.inputs[1]
    if len(a.shape) < 2 or len(b.shape) < 2:
        return _default(node)
    M, K = a.shape[-2], a.shape[-1]
    N = b.shape[-1]
    it = a.dtype.itemsize
    flops = 2.0 * M * N * K
    read  = (M * K + K * N) * it
    write = M * N * it
    return flops, read, write


def _addmm(node: "OpNode") -> FMR:
    """aten.addmm.default: bias + mat1=(M,K) @ mat2=(K,N) → (M,N)
    FLOPs = 2·M·K·N + M·N   R=(M·K+K·N+|bias|)·b   W=M·N·b
    """
    if len(node.inputs) < 3:
        return _default(node)
    # inputs: [bias, mat1, mat2]
    mat1, mat2 = node.inputs[1], node.inputs[2]
    bias = node.inputs[0]
    if len(mat1.shape) < 2 or len(mat2.shape) < 2:
        return _default(node)
    M, K = mat1.shape[0], mat1.shape[1]
    N = mat2.shape[1]
    it = mat1.dtype.itemsize
    flops = 2.0 * M * N * K + M * N   # mm + bias add
    read  = (M * K + K * N + _numel(bias.shape)) * it
    write = M * N * it
    return flops, read, write


def _bmm(node: "OpNode") -> FMR:
    """aten.bmm.default: (B,M,K) @ (B,K,N) → (B,M,N)
    FLOPs = 2·B·M·K·N   R=(B·M·K+B·K·N)·b   W=B·M·N·b
    """
    if len(node.inputs) < 2:
        return _default(node)
    a, b = node.inputs[0], node.inputs[1]
    if len(a.shape) < 3 or len(b.shape) < 3:
        return _mm(node)   # fallback to 2-D mm
    B, M, K = a.shape[0], a.shape[1], a.shape[2]
    N = b.shape[2]
    it = a.dtype.itemsize
    flops = 2.0 * B * M * N * K
    read  = (B * M * K + B * K * N) * it
    write = B * M * N * it
    return flops, read, write


def _linear(node: "OpNode") -> FMR:
    """aten.linear.default: input=(*,I), weight=(O,I), optional bias=(O,)
    FLOPs = 2·batch·I·O [+ batch·O if bias]
    R=(batch·I + O·I [+ O])·b   W=batch·O·b
    """
    if len(node.inputs) < 2:
        return _default(node)
    inp, weight = node.inputs[0], node.inputs[1]
    if len(inp.shape) < 1 or len(weight.shape) < 2:
        return _default(node)
    I = inp.shape[-1]
    O = weight.shape[0]
    batch = _numel(inp.shape[:-1])
    it = inp.dtype.itemsize
    flops = 2.0 * batch * O * I
    read  = (batch * I + O * I) * it
    write = batch * O * it
    if len(node.inputs) >= 3:   # bias
        bias = node.inputs[2]
        flops += batch * O
        read  += _numel(bias.shape) * it
    return flops, read, write


def _scaled_dot_product_attention(node: "OpNode") -> FMR:
    """aten._scaled_dot_product_flash_attention / scaled_dot_product_attention.

    Input layout assumed: Q=(N,H,Sq,D), K=(N,H,Sk,D), V=(N,H,Sk,Dv)
    FLOPs = 4·N·H·Sq·Sk·D        (QK + AV matmuls)
          + 5·N·H·Sq·Sk           (softmax: max+sub+exp+sum+div)
    R = (Q+K+V)·b    W = output·b
    """
    if len(node.inputs) < 3:
        return _default(node)
    q, k, v = node.inputs[0], node.inputs[1], node.inputs[2]
    if len(q.shape) < 4 or len(k.shape) < 4:
        return _default(node)
    # Assume (N, H, Sq, D) layout
    N, H, Sq, D = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
    Sk = k.shape[2]
    it = q.dtype.itemsize
    # QK matmul: 2*N*H*Sq*Sk*D,  AV matmul: 2*N*H*Sq*Sk*D
    flops = 4.0 * N * H * Sq * Sk * D
    # Softmax ops ~ 5*N*H*Sq*Sk (sub-dominant, included for completeness)
    flops += 5.0 * N * H * Sq * Sk
    read  = (N*H*Sq*D + N*H*Sk*D + N*H*Sk*D) * it   # Q + K + V
    write = (N*H*Sq*D) * it                           # output
    return flops, read, write


def _layer_norm(node: "OpNode") -> FMR:
    """aten.layer_norm.default / aten.native_layer_norm.default
    FLOPs ≈ 5·N  (mean + variance + normalize + scale + shift)
    R=(N + 2·|weight|)·b    W=N·b
    """
    if not node.inputs:
        return _default(node)
    inp = node.inputs[0]
    n = _numel(inp.shape)
    it = inp.dtype.itemsize
    # mean(N) + var(2N) + norm(N) + scale(N) + shift(N) ≈ 5N flops
    flops = 5.0 * n
    # read: input + weight + bias (last dim)
    weight_size = inp.shape[-1] if inp.shape else 1
    read  = (n + 2 * weight_size) * it
    write = n * it
    return flops, read, write


def _rms_norm(node: "OpNode") -> FMR:
    """Fused rms_norm: fewer ops than layer_norm (no mean subtraction).
    FLOPs ≈ 4·N  (pow + mean + rsqrt + mul + scale)
    R=(N + |weight|)·b    W=N·b
    """
    if not node.inputs:
        return _default(node)
    inp = node.inputs[0]
    n = _numel(inp.shape)
    it = inp.dtype.itemsize
    # pow(N) + mean(N) + rsqrt(1) + mul(N) + scale(N) ≈ 4N flops
    flops = 4.0 * n
    weight_size = inp.shape[-1] if inp.shape else 1
    read  = (n + weight_size) * it
    write = n * it
    return flops, read, write


def _softmax(node: "OpNode") -> FMR:
    """aten._softmax.default / aten.softmax.int
    FLOPs ≈ 5·N  (max + sub + exp + sum + div)
    R=N·b    W=N·b
    """
    if not node.inputs:
        return _default(node)
    inp = node.inputs[0]
    n = _numel(inp.shape)
    it = inp.dtype.itemsize
    # max(N) + sub(N) + exp(N) + sum(N) + div(N) ≈ 5N
    flops = 5.0 * n
    read  = n * it
    write = n * it
    return flops, read, write


def _elementwise(node: "OpNode", ops_per_elem: float = 1.0) -> FMR:
    """Generic elementwise op.
    FLOPs = ops_per_elem · N_out
    R=sum(|inputs|)·b    W=|output|·b

    ops_per_elem 取值参考:
      1.0  — add/sub/mul/div/neg/abs/relu/exp/log/sqrt/rsqrt/pow/masked_fill/reduction
      2.0  — reciprocal/clamp/rope  (比较 + 赋值)
      3.0  — var  (sq+mean+sub+sq+mean ≈ 3步)
      4.0  — silu/gelu/sigmoid  (含指数/多项式近似)
      10.0 — sin/cos/atan2  (CORDIC / 多项式展开)
    """
    if not node.outputs:
        return _default(node)
    out = node.outputs[0]
    n_out = _numel(out.shape)
    it = out.dtype.itemsize
    flops = ops_per_elem * n_out
    read  = float(node.total_input_bytes())
    write = float(node.total_output_bytes())
    return flops, read, write


def _embedding(node: "OpNode") -> FMR:
    """aten.embedding.default / embedding / embedding_backward
    FLOPs = 0  (纯查表, 无算术运算)
    R=|output|·b   W=|output|·b   (cache-miss dominated random reads)
    """
    if not node.outputs:
        return _default(node)
    out = node.outputs[0]
    n = _numel(out.shape)
    it = out.dtype.itemsize
    flops = 0.0
    read  = n * it
    write = n * it
    return flops, read, write


def _dtype_cast(node: "OpNode") -> FMR:
    """aten._to_copy.default: dtype cast / device copy.
    FLOPs = 0  (无算术运算, 纯数据搬运)
    R=|input|·b_in    W=|output|·b_out   (src/dst 字节宽可能不同, 如 bf16→fp32)
    """
    if node.inputs and node.outputs:
        inp = node.inputs[0]
        out = node.outputs[0]
        read  = _numel(inp.shape) * inp.dtype.itemsize
        write = _numel(out.shape) * out.dtype.itemsize
    else:
        read  = float(node.total_input_bytes())
        write = float(node.total_output_bytes())
    return 0.0, read, write


def _gather(node: "OpNode") -> FMR:
    """aten.index.Tensor / index_select / gather / scatter / scatter_add
       moe_dispatch / npu_moe_dispatch
    FLOPs = 0  (索引/路由操作, 无算术)
    R=sum(|inputs|)·b    W=|output|·b
    """
    if not node.outputs:
        return _default(node)
    out = node.outputs[0]
    n = _numel(out.shape)
    it = out.dtype.itemsize
    read  = float(node.total_input_bytes())
    write = n * it
    return 0.0, read, write


def _linear_proj(node: "OpNode") -> FMR:
    """Fused Linear projection: input(*,I) @ weight(I,O) → (*,O).

    Used for op_type='Linear' nodes produced by FusionPass when a single
    nn.Linear module's ops (view + mm + view) are grouped together.
    Weight is stored in (I, O) layout after transpose in aten.mm.
    """
    if len(node.inputs) < 2:
        return _default(node)
    inp, weight = node.inputs[0], node.inputs[1]
    if len(inp.shape) < 1 or len(weight.shape) < 2:
        return _default(node)
    I     = inp.shape[-1]
    N     = weight.shape[-1]        # weight layout: (I, N) after mm's transpose
    batch = _numel(inp.shape[:-1])  # product of all dims except last
    it    = inp.dtype.itemsize
    flops = 2.0 * batch * I * N
    read  = (batch * I + I * N) * it
    write = batch * N * it
    if len(node.inputs) >= 3:       # optional bias
        bias   = node.inputs[2]
        flops += batch * N
        read  += _numel(bias.shape) * bias.dtype.itemsize
    return flops, read, write


def _write_only(node: "OpNode") -> FMR:
    """Allocation / fill ops: new_empty / new_empty_strided / fill_ / zero_
    FLOPs = 0    R=0    W=|output|·b
    """
    return 0.0, 0.0, float(node.total_output_bytes())


def _default(node: "OpNode") -> FMR:
    """Conservative fallback for unrecognized ops.
    FLOPs = N_out  (1 flop / output element)
    R=total_input_bytes    W=total_output_bytes
    """
    n_out = sum(_numel(o.shape) for o in node.outputs) if node.outputs else 1
    it = _itemsize(node)
    flops = float(n_out)
    read  = float(node.total_input_bytes())
    write = float(node.total_output_bytes())
    return flops, read, write


# ── fused op formulas ─────────────────────────────────────────────────────────

def _fused_attention(node: "OpNode") -> FMR:
    """flash_attn / sdpa / sdpa_backward / npu_fusion_attention / attn / attn_grad / mla_attn
    外部输入按 [Q, K, V, ...] 顺序, 调用 _scaled_dot_product_attention.
    FLOPs = 4·N·H·Sq·Sk·D + 5·N·H·Sq·Sk
    R=(Q+K+V)·b    W=output·b
    """
    if len(node.inputs) >= 3:
        return _scaled_dot_product_attention(node)
    # single-tensor attention (e.g., compact fused node)
    return _default(node)


def _fused_norm(node: "OpNode") -> FMR:
    """add_rms_norm / add_layer_norm / npu_add_rms_norm / norm_backward
    在 rms_norm(5N) 基础上加 residual add(1N), 共 6N FLOPs.
    FLOPs = 6·N    R=(2·N + |weight|)·b  (input + residual + weight)    W=N·b
    """
    if not node.inputs:
        return _default(node)
    # For add_norm variants: FLOPs = 4-5 * N + N (for the add)
    inp = node.inputs[0]
    n = _numel(inp.shape)
    it = inp.dtype.itemsize
    flops = 6.0 * n     # 5 for norm + 1 for residual add
    weight_size = inp.shape[-1] if inp.shape else 1
    read  = (n * 2 + weight_size) * it   # input + residual + weight
    write = n * it
    return flops, read, write


def _fused_mlp(node: "OpNode") -> FMR:
    """gated_mlp / mlp / gated_mlp_backward / mlp_backward
       moe_block / moe_expert / moe_shared
    输入典型布局: [hidden=(B,S,H), gate_w=(I,H), up_w=(I,H), down_w=(H,I)]
    FLOPs = Σᵢ 2·batch·H·Oᵢ  +  4·N_act/2   (各 GEMM FLOPs + gated activation)
    R = hidden·b + Σ|weight_i|·b    W = output·b
    """
    if len(node.inputs) < 2:
        return _default(node)

    hidden = node.inputs[0]
    it = hidden.dtype.itemsize

    if len(hidden.shape) < 2:
        return _default(node)

    batch = _numel(hidden.shape[:-1])   # B*S
    H = hidden.shape[-1]

    # Collect intermediate sizes from weight tensors
    mm_flops = 0.0
    mm_read  = batch * H * it   # hidden state read once
    mm_write = 0.0

    # Each weight matrix contributes one matmul
    for w in node.inputs[1:]:
        if len(w.shape) < 2:
            continue
        # weight shape: (out_features, in_features) or (in, out)
        s0, s1 = w.shape[0], w.shape[1]
        # Infer which dim matches H
        if s1 == H:
            O = s0
        elif s0 == H:
            O = s1
        else:
            O = max(s0, s1)
        mm_flops += 2.0 * batch * H * O
        mm_read  += s0 * s1 * it        # read weight
        mm_write += batch * O * it

    # Elementwise (activation + mul for gated MLP)
    n_out = _numel(node.outputs[0].shape) if node.outputs else batch * H
    elem_flops = 4.0 * (n_out // 2 if n_out > batch * H else n_out)

    flops = mm_flops + elem_flops
    read  = mm_read
    write = (node.outputs[0].mem_bytes if node.outputs
             else batch * H * it)
    return flops, read, write


def _fused_moe_gate(node: "OpNode", with_topk: bool = False) -> FMR:
    """moe_gate / npu_moe_gate / moe_gate_topk / npu_moe_gate_topk
    一个小 GEMM (hidden→num_experts) + softmax (+ topk 比较)
    FLOPs = linear_FLOPs + 5·N_out [+ N_out if with_topk]
    R/W   = same as _linear for the gate matmul
    """
    # Dominant cost: one matmul to compute gate scores
    if len(node.inputs) >= 2:
        flops, read, write = _linear(node) if len(node.inputs[1].shape) >= 2 else _default(node)
        # Add softmax cost
        n_out = sum(_numel(o.shape) for o in node.outputs) if node.outputs else 1
        flops += 5.0 * n_out
        if with_topk:
            flops += n_out   # topk comparison ops
        return flops, read, write
    return _default(node)


# ── op → formula dispatch table ──────────────────────────────────────────────

# Maps op_type (exact match) → formula function.
# Check is "op_type starts with <key>" for prefix entries.

_EXACT_FORMULAS: dict[str, "callable"] = {
    # ── matmul family ─────────────────────────────────────────────────────────
    "aten.mm.default":                  _mm,
    "aten.mm":                          _mm,
    "aten.addmm.default":               _addmm,
    "aten.addmm":                       _addmm,
    "aten.bmm.default":                 _bmm,
    "aten.bmm":                         _bmm,
    "aten.matmul.default":              _mm,
    "aten.matmul":                      _mm,
    "aten.linear.default":              _linear,
    "aten.linear":                      _linear,
    # ── attention ─────────────────────────────────────────────────────────────
    "aten._scaled_dot_product_flash_attention.default":    _scaled_dot_product_attention,
    "aten.scaled_dot_product_attention.default":           _scaled_dot_product_attention,
    "aten._scaled_dot_product_efficient_attention.default":_scaled_dot_product_attention,
    # ── norm ──────────────────────────────────────────────────────────────────
    "aten.layer_norm.default":          _layer_norm,
    "aten.layer_norm":                  _layer_norm,
    "aten.native_layer_norm.default":   _layer_norm,
    # ── softmax ───────────────────────────────────────────────────────────────
    "aten._softmax.default":            _softmax,
    "aten.softmax.int":                 _softmax,
    "aten.special_softmax.int":         _softmax,
    # ── elementwise — 1 op/elem ───────────────────────────────────────────────
    "aten.add.Tensor":                  _elementwise,
    "aten.add_.Tensor":                 _elementwise,
    "aten.add.Scalar":                  _elementwise,
    "aten.sub.Tensor":                  _elementwise,
    "aten.sub.Scalar":                  _elementwise,
    "aten.rsub.Scalar":                 _elementwise,
    "aten.rsub.default":                _elementwise,
    "aten.mul.Tensor":                  _elementwise,
    "aten.mul.Scalar":                  _elementwise,
    "aten.div.Tensor":                  _elementwise,
    "aten.div.Scalar":                  _elementwise,
    "aten.neg.default":                 _elementwise,
    "aten.abs.default":                 _elementwise,
    "aten.relu.default":                _elementwise,
    "aten.relu_.default":               _elementwise,
    "aten.tanh.default":                _elementwise,
    "aten.exp.default":                 _elementwise,
    "aten.log.default":                 _elementwise,
    "aten.sqrt.default":                _elementwise,
    "aten.rsqrt.default":               _elementwise,
    "aten.pow.Tensor_Scalar":           _elementwise,
    "aten.pow.Tensor_Tensor":           _elementwise,
    "aten.masked_fill.Scalar":          _elementwise,
    "aten.masked_fill_.Scalar":         _elementwise,
    "aten.masked_fill.Tensor":          _elementwise,
    # ── elementwise — ~2 ops/elem ─────────────────────────────────────────────
    "aten.reciprocal.default":          lambda n: _elementwise(n, 2.0),
    "aten.clamp.default":               lambda n: _elementwise(n, 2.0),
    "aten.clamp.Scalar":                lambda n: _elementwise(n, 2.0),
    "aten.clamp.Tensor":                lambda n: _elementwise(n, 2.0),
    "aten.clamp_min.default":           lambda n: _elementwise(n, 2.0),
    "aten.clamp_max.default":           lambda n: _elementwise(n, 2.0),
    # ── activation — ~4 ops/elem ─────────────────────────────────────────────
    "aten.silu.default":                lambda n: _elementwise(n, 4.0),
    "aten.silu_.default":               lambda n: _elementwise(n, 4.0),
    "aten.gelu.default":                lambda n: _elementwise(n, 4.0),
    "aten.sigmoid.default":             lambda n: _elementwise(n, 4.0),
    # ── transcendental — ~10 ops/elem (CORDIC / polynomial approx) ───────────
    "aten.sin.default":                 lambda n: _elementwise(n, 10.0),
    "aten.cos.default":                 lambda n: _elementwise(n, 10.0),
    "aten.atan2.default":               lambda n: _elementwise(n, 10.0),
    # ── embedding / gather ────────────────────────────────────────────────────
    "aten.embedding.default":           _embedding,
    "aten.index.Tensor":                _gather,
    "aten.index_select.default":        _gather,
    "aten.gather.default":              _gather,
    "aten.scatter.src":                 _gather,
    "aten.scatter_.src":                _gather,
    "aten.scatter_add.default":         _gather,
    # ── reduction ─────────────────────────────────────────────────────────────
    "aten.mean.dim":                    lambda n: _elementwise(n, 1.0),
    "aten.mean.default":                lambda n: _elementwise(n, 1.0),
    "aten.sum.dim_IntList":             lambda n: _elementwise(n, 1.0),
    "aten.sum.default":                 lambda n: _elementwise(n, 1.0),
    "aten.var.correction":              lambda n: _elementwise(n, 3.0),
    "aten.amax.default":                lambda n: _elementwise(n, 1.0),
    "aten.amin.default":                lambda n: _elementwise(n, 1.0),
    # ── dtype cast ────────────────────────────────────────────────────────────
    "aten._to_copy.default":            _dtype_cast,
    # ── memory / shape — trivial compute ──────────────────────────────────────
    "aten.copy_.default":               lambda n: (0.0, float(n.total_input_bytes()), float(n.total_output_bytes())),
    # ── write-only allocation ops (0 compute) ─────────────────────────────────
    "aten.new_empty.default":           _write_only,
    "aten.new_empty_strided.default":   _write_only,
    "aten.fill_.Scalar":                _write_only,
    "aten.zero_.default":               _write_only,
    # ── fused semantic labels from FusionEngine / FusionPass ──────────────────
    # norm
    "rms_norm":                         _rms_norm,
    "layer_norm":                       _layer_norm,
    "add_rms_norm":                     _fused_norm,
    "add_layer_norm":                   _fused_norm,
    "npu_add_rms_norm":                 _fused_norm,
    # attention
    "flash_attn":                       _fused_attention,
    "sdpa":                             _fused_attention,
    "sdpa_backward":                    _fused_attention,
    "npu_fusion_attention":             _fused_attention,
    "attn":                             _fused_attention,
    "attn_grad":                        _fused_attention,
    "mla_attn":                         _fused_attention,
    # MLP
    "gated_mlp":                        _fused_mlp,
    "gated_mlp_backward":               _fused_mlp,
    "mlp":                              _fused_mlp,
    "mlp_backward":                     _fused_mlp,
    # MoE gate / router
    "moe_gate":                         lambda n: _fused_moe_gate(n, with_topk=False),
    "moe_gate_topk":                    lambda n: _fused_moe_gate(n, with_topk=True),
    "npu_moe_gate":                     lambda n: _fused_moe_gate(n, with_topk=False),
    "npu_moe_gate_topk":                lambda n: _fused_moe_gate(n, with_topk=True),
    # MoE dispatch (scatter/gather routing)
    "moe_dispatch":                     _gather,
    "npu_moe_dispatch":                 _gather,
    # MoE block / expert
    "moe_block":                        _fused_mlp,
    "moe_expert":                       _fused_mlp,
    "moe_shared":                       _fused_mlp,
    # RoPE
    "rope":                             lambda n: _elementwise(n, 2.0),
    # Linear projection (single nn.Linear module grouped by FusionPass)
    "Linear":                           _linear_proj,
    # Norm backward (native fused kernel)
    "norm_backward":                    _fused_norm,
    # Embedding / lm_head
    "embedding":                        _embedding,
    "lm_head":                          _linear_proj,
    "embedding_backward":               _embedding,
}


def _shape_ops_fmr(node: "OpNode") -> FMR:
    """Shape/view/permute ops: view/reshape/expand/squeeze/permute/transpose
       contiguous/flatten/as_strided/select/slice/clone/cat/stack/chunk/split
    FLOPs ≈ 0  (元数据重解释, 实际可能零内存移动)
    R≈|output|·b    W≈|output|·b  (保守估计, 连续内存实际为0)
    """
    it = _itemsize(node)
    n = _numel(node.outputs[0].shape) if node.outputs else 1
    return 0.0, n * it, n * it


_SHAPE_OP_PREFIXES: tuple[str, ...] = (
    "aten.view", "aten._unsafe_view", "aten.reshape",
    "aten.expand", "aten.squeeze", "aten.unsqueeze",
    "aten.permute", "aten.transpose", "aten.contiguous",
    "aten.flatten", "aten.as_strided", "aten.select",
    "aten.slice", "aten.clone", "aten.t.", "aten.chunk",
    "aten.split", "aten.unbind", "aten.detach", "aten.alias",
    "aten.cat", "aten.stack",
)


# ── RooflineSimulator ─────────────────────────────────────────────────────────

class RooflineSimulator(OpSimulator):
    """Theoretical Roofline model — universal fallback backend.

    Uses pre-registered analytic formulas keyed by op_type.
    Any op without a formula uses the default fallback (1 flop / output elem).

    This backend always returns True from ``can_simulate()``, making it the
    guaranteed last resort in ``SimulatorHub``.
    """

    name = "roofline"
    priority = 0

    def can_simulate(self, node: "OpNode", hw: "HardwareSpec") -> bool:
        return True

    def simulate(self, node: "OpNode", hw: "HardwareSpec") -> SimResult:
        flops, read_bytes, write_bytes = self._fmr(node)
        total_bytes = read_bytes + write_bytes

        dtype = _primary_dtype(node)
        peak  = hw.peak_flops(dtype)        # ops/s
        bw    = hw.hbm_bandwidth()          # bytes/s

        compute_us = (flops / peak * 1e6)   if peak > 0 else 0.0
        memory_us  = (total_bytes / bw * 1e6) if bw > 0  else 0.0

        # Latency bound: kernel launch overhead (minimum ~1 µs for GPUs/NPUs)
        latency_us = max(compute_us, memory_us, 1e-3)

        ai = flops / total_bytes if total_bytes > 0 else math.inf

        if compute_us > 0 or memory_us > 0:
            bound = "compute" if compute_us >= memory_us else "memory"
        else:
            bound = "latency"

        hw_util = 0.0
        if peak > 0 and latency_us > 0:
            actual_rate = flops / (latency_us * 1e-6)
            hw_util = min(1.0, actual_rate / peak)

        return SimResult(
            op_node_id        = node.id,
            latency_us        = latency_us,
            compute_us        = compute_us,
            memory_us         = memory_us,
            flops             = int(flops),
            read_bytes        = int(read_bytes),
            write_bytes       = int(write_bytes),
            arithmetic_intensity = ai,
            bound             = bound,
            hw_utilization    = hw_util,
            backend           = self.name,
            confidence        = 0.3,
        )

    # ── FLOPs / Memory formula dispatch ──────────────────────────────────────

    def _fmr(self, node: "OpNode") -> FMR:
        op = node.op_type

        # 1. Exact match
        fn = _EXACT_FORMULAS.get(op)
        if fn is not None:
            return fn(node)

        # 2. Shape / transparent ops
        for prefix in _SHAPE_OP_PREFIXES:
            if op.startswith(prefix):
                return _shape_ops_fmr(node)

        # 3. Fused node: sum sub-op estimates if fused_from is available
        if node.is_fused and node.fused_from:
            return self._fused_decompose(node)

        # 4. Fallback
        return _default(node)

    def _fused_decompose(self, node: "OpNode") -> FMR:
        """Sum up FLOPs/memory for all sub-ops listed in fused_from.

        Since we don't have intermediate tensor shapes, we use the node's
        external inputs and outputs to estimate the dominant matmul costs.
        Shape/transparent ops in fused_from are skipped (0 compute).
        """
        total_flops = 0.0
        total_read  = float(node.total_input_bytes())
        total_write = float(node.total_output_bytes())

        for sub_op in node.fused_from:
            # Skip shape / transparent ops — they contribute no FLOPs
            if any(sub_op.startswith(p) for p in _SHAPE_OP_PREFIXES):
                continue
            if sub_op in ("aten.detach.default", "aten.alias.default",
                          "aten.lift_fresh_copy.default"):
                continue

            fn = _EXACT_FORMULAS.get(sub_op)
            if fn is not None:
                # Reuse the node's shapes as a proxy for the dominant op
                f, _r, _w = fn(node)
                total_flops += f
            else:
                # Unknown sub-op: 1 flop / output elem (conservative)
                total_flops += sum(_numel(o.shape) for o in node.outputs)

        # Clamp read/write to at least actual tensor bytes
        total_read  = max(total_read,  float(node.total_input_bytes()))
        total_write = max(total_write, float(node.total_output_bytes()))

        return total_flops, total_read, total_write


# ── Op formula string generation for Excel export ────────────────────────────
# Returns dicts with keys: flops_sym, flops_num, read_sym, read_num, write_sym, write_num

def _bw(node: "OpNode") -> int:
    """Return dtype itemsize as int (avoids '2.0' in formula strings)."""
    if node.inputs:
        return int(node.inputs[0].dtype.itemsize)
    if node.outputs:
        return int(node.outputs[0].dtype.itemsize)
    return 2


def _mk(fs: str, fn: str, rs: str, rn: str, ws: str, wn: str) -> dict:
    return {"flops_sym": fs, "flops_num": fn,
            "read_sym": rs, "read_num": rn,
            "write_sym": ws, "write_num": wn}


def _fs_mm(node: "OpNode") -> dict:
    if len(node.inputs) < 2: return _fs_default(node)
    a, b = node.inputs[0], node.inputs[1]
    if len(a.shape) < 2 or len(b.shape) < 2: return _fs_default(node)
    M, K, N, bw = a.shape[-2], a.shape[-1], b.shape[-1], _bw(node)
    return _mk("2·M·K·N", f"2·{M}·{K}·{N}",
               "(M·K+K·N)·b", f"({M}·{K}+{K}·{N})·{bw}",
               "M·N·b", f"{M}·{N}·{bw}")


def _fs_addmm(node: "OpNode") -> dict:
    if len(node.inputs) < 3: return _fs_default(node)
    bias, mat1, mat2 = node.inputs[0], node.inputs[1], node.inputs[2]
    if len(mat1.shape) < 2 or len(mat2.shape) < 2: return _fs_default(node)
    M, K, N, bw = mat1.shape[0], mat1.shape[1], mat2.shape[1], _bw(node)
    Nb = _numel(bias.shape)
    return _mk("2·M·K·N+M·N", f"2·{M}·{K}·{N}+{M}·{N}",
               "(M·K+K·N+|bias|)·b", f"({M}·{K}+{K}·{N}+{Nb})·{bw}",
               "M·N·b", f"{M}·{N}·{bw}")


def _fs_bmm(node: "OpNode") -> dict:
    if len(node.inputs) < 2: return _fs_default(node)
    a, b = node.inputs[0], node.inputs[1]
    if len(a.shape) < 3 or len(b.shape) < 3: return _fs_mm(node)
    B, M, K, N, bw = a.shape[0], a.shape[1], a.shape[2], b.shape[2], _bw(node)
    return _mk("2·B·M·K·N", f"2·{B}·{M}·{K}·{N}",
               "(B·M·K+B·K·N)·b", f"({B}·{M}·{K}+{B}·{K}·{N})·{bw}",
               "B·M·N·b", f"{B}·{M}·{N}·{bw}")


def _fs_linear(node: "OpNode") -> dict:
    if len(node.inputs) < 2: return _fs_default(node)
    inp, w = node.inputs[0], node.inputs[1]
    if len(inp.shape) < 1 or len(w.shape) < 2: return _fs_default(node)
    I, O, batch, bw = inp.shape[-1], w.shape[0], _numel(inp.shape[:-1]), _bw(node)
    if len(node.inputs) >= 3:
        Nb = _numel(node.inputs[2].shape)
        return _mk("2·batch·I·O+batch·O", f"2·{batch}·{I}·{O}+{batch}·{O}",
                   "(batch·I+O·I+O)·b", f"({batch}·{I}+{O}·{I}+{Nb})·{bw}",
                   "batch·O·b", f"{batch}·{O}·{bw}")
    return _mk("2·batch·I·O", f"2·{batch}·{I}·{O}",
               "(batch·I+O·I)·b", f"({batch}·{I}+{O}·{I})·{bw}",
               "batch·O·b", f"{batch}·{O}·{bw}")


def _fs_linear_proj(node: "OpNode") -> dict:
    if len(node.inputs) < 2: return _fs_default(node)
    inp, w = node.inputs[0], node.inputs[1]
    if len(inp.shape) < 1 or len(w.shape) < 2: return _fs_default(node)
    I, N, batch, bw = inp.shape[-1], w.shape[-1], _numel(inp.shape[:-1]), _bw(node)
    return _mk("2·batch·I·N", f"2·{batch}·{I}·{N}",
               "(batch·I+I·N)·b", f"({batch}·{I}+{I}·{N})·{bw}",
               "batch·N·b", f"{batch}·{N}·{bw}")


def _fs_sdpa(node: "OpNode") -> dict:
    if len(node.inputs) < 3: return _fs_default(node)
    q, k = node.inputs[0], node.inputs[1]
    if len(q.shape) < 4: return _fs_default(node)
    N, H, Sq, D = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
    Sk, bw = (k.shape[2] if len(k.shape) >= 3 else Sq), _bw(node)
    return _mk("4·N·H·Sq·Sk·D+5·N·H·Sq·Sk",
               f"4·{N}·{H}·{Sq}·{Sk}·{D}+5·{N}·{H}·{Sq}·{Sk}",
               "(Q+K+V)·b",
               f"({N}·{H}·{Sq}·{D}+{N}·{H}·{Sk}·{D}+{N}·{H}·{Sk}·{D})·{bw}",
               "N·H·Sq·D·b", f"{N}·{H}·{Sq}·{D}·{bw}")


def _fs_rms_norm(node: "OpNode") -> dict:
    if not node.inputs: return _fs_default(node)
    inp = node.inputs[0]
    N, W, bw = _numel(inp.shape), (inp.shape[-1] if inp.shape else 1), _bw(node)
    return _mk("4·N", f"4·{N}", "(N+|W|)·b", f"({N}+{W})·{bw}", "N·b", f"{N}·{bw}")


def _fs_layer_norm(node: "OpNode") -> dict:
    if not node.inputs: return _fs_default(node)
    inp = node.inputs[0]
    N, W, bw = _numel(inp.shape), (inp.shape[-1] if inp.shape else 1), _bw(node)
    return _mk("5·N", f"5·{N}", "(N+2·|W|)·b", f"({N}+2·{W})·{bw}", "N·b", f"{N}·{bw}")


def _fs_add_norm(node: "OpNode") -> dict:
    if not node.inputs: return _fs_default(node)
    inp = node.inputs[0]
    N, W, bw = _numel(inp.shape), (inp.shape[-1] if inp.shape else 1), _bw(node)
    return _mk("6·N", f"6·{N}", "(2·N+|W|)·b", f"(2·{N}+{W})·{bw}", "N·b", f"{N}·{bw}")


def _fs_softmax(node: "OpNode") -> dict:
    if not node.inputs: return _fs_default(node)
    inp = node.inputs[0]
    N, bw = _numel(inp.shape), _bw(node)
    return _mk("5·N", f"5·{N}", "N·b", f"{N}·{bw}", "N·b", f"{N}·{bw}")


def _fs_elementwise(node: "OpNode", k: float, ks: str) -> dict:
    if not node.outputs: return _fs_default(node)
    out = node.outputs[0]
    N, bw = _numel(out.shape), _bw(node)
    ri = node.total_input_bytes()
    return _mk(f"{ks}·N", f"{ks}·{N}", "Σ|inputs|·b", str(ri), "|output|·b", f"{N}·{bw}")


def _fs_embedding(node: "OpNode") -> dict:
    if not node.outputs: return _fs_default(node)
    out = node.outputs[0]
    N, bw = _numel(out.shape), _bw(node)
    return _mk("0 (lookup)", "0", "|output|·b", f"{N}·{bw}", "|output|·b", f"{N}·{bw}")


def _fs_gather(node: "OpNode") -> dict:
    if not node.outputs: return _fs_default(node)
    out = node.outputs[0]
    N, bw = _numel(out.shape), _bw(node)
    ri = node.total_input_bytes()
    return _mk("0 (index)", "0", "Σ|inputs|·b", str(ri), "|output|·b", f"{N}·{bw}")


def _fs_shape(node: "OpNode") -> dict:
    if not node.outputs: return _mk("0", "0", "0", "0", "0", "0")
    out = node.outputs[0]
    N, bw = _numel(out.shape), _bw(node)
    return _mk("0 (shape)", "0", "~|output|·b", f"~{N}·{bw}", "~|output|·b", f"~{N}·{bw}")


def _fs_mlp(node: "OpNode") -> dict:
    if len(node.inputs) < 2: return _fs_default(node)
    h = node.inputs[0]
    if len(h.shape) < 2: return _fs_default(node)
    batch, H, bw = _numel(h.shape[:-1]), h.shape[-1], h.dtype.itemsize
    ws = [w for w in node.inputs[1:] if len(w.shape) >= 2]
    ri = h.mem_bytes + sum(w.mem_bytes for w in ws)
    wo = node.outputs[0].mem_bytes if node.outputs else 0
    return _mk("Σᵢ(2·batch·H·Oᵢ)+4·N_act",
               f"Σ(2·{batch}·{H}·Oᵢ) [{len(ws)} weights]",
               "hidden+Σweights·b", str(ri),
               "|output|·b", str(wo))


def _fs_comm(node: "OpNode") -> dict:
    vol = sum(t.mem_bytes for t in node.outputs)
    return _mk("0 (comm)", "0", "comm_vol·b", str(vol), "comm_vol·b", str(vol))


def _fs_default(node: "OpNode") -> dict:
    n_out = sum(_numel(o.shape) for o in node.outputs) if node.outputs else 1
    ri, wo = node.total_input_bytes(), node.total_output_bytes()
    return _mk("N_out", str(n_out), "Σ|inputs|·b", str(ri), "Σ|outputs|·b", str(wo))


_EW_DISPATCH: dict[str, tuple] = {
    "aten.add.Tensor": (1.0, "1"), "aten.add_.Tensor": (1.0, "1"),
    "aten.add.Scalar": (1.0, "1"), "aten.sub.Tensor": (1.0, "1"),
    "aten.sub.Scalar": (1.0, "1"), "aten.rsub.Scalar": (1.0, "1"),
    "aten.rsub.default": (1.0, "1"), "aten.mul.Tensor": (1.0, "1"),
    "aten.mul.Scalar": (1.0, "1"), "aten.div.Tensor": (1.0, "1"),
    "aten.div.Scalar": (1.0, "1"), "aten.neg.default": (1.0, "1"),
    "aten.abs.default": (1.0, "1"), "aten.relu.default": (1.0, "1"),
    "aten.relu_.default": (1.0, "1"), "aten.tanh.default": (1.0, "1"),
    "aten.exp.default": (1.0, "1"), "aten.log.default": (1.0, "1"),
    "aten.sqrt.default": (1.0, "1"), "aten.rsqrt.default": (1.0, "1"),
    "aten.pow.Tensor_Scalar": (1.0, "1"), "aten.pow.Tensor_Tensor": (1.0, "1"),
    "aten.masked_fill.Scalar": (1.0, "1"), "aten.masked_fill_.Scalar": (1.0, "1"),
    "aten.masked_fill.Tensor": (1.0, "1"),
    "aten.mean.dim": (1.0, "1"), "aten.mean.default": (1.0, "1"),
    "aten.sum.dim_IntList": (1.0, "1"), "aten.sum.default": (1.0, "1"),
    "aten.amax.default": (1.0, "1"), "aten.amin.default": (1.0, "1"),
    "aten.reciprocal.default": (2.0, "2"), "aten.clamp.default": (2.0, "2"),
    "aten.clamp.Scalar": (2.0, "2"), "aten.clamp.Tensor": (2.0, "2"),
    "aten.clamp_min.default": (2.0, "2"), "aten.clamp_max.default": (2.0, "2"),
    "aten.var.correction": (3.0, "3"),
    "aten.silu.default": (4.0, "4"), "aten.silu_.default": (4.0, "4"),
    "aten.gelu.default": (4.0, "4"), "aten.sigmoid.default": (4.0, "4"),
    "aten.sin.default": (10.0, "10"), "aten.cos.default": (10.0, "10"),
    "aten.atan2.default": (10.0, "10"),
}

_FORMULA_DISPATCH: dict[str, "callable"] = {
    "aten.mm.default": _fs_mm, "aten.mm": _fs_mm,
    "aten.addmm.default": _fs_addmm, "aten.addmm": _fs_addmm,
    "aten.bmm.default": _fs_bmm, "aten.bmm": _fs_bmm,
    "aten.matmul.default": _fs_mm, "aten.matmul": _fs_mm,
    "aten.linear.default": _fs_linear, "aten.linear": _fs_linear,
    "aten._scaled_dot_product_flash_attention.default": _fs_sdpa,
    "aten.scaled_dot_product_attention.default": _fs_sdpa,
    "aten._scaled_dot_product_efficient_attention.default": _fs_sdpa,
    "aten.layer_norm.default": _fs_layer_norm, "aten.layer_norm": _fs_layer_norm,
    "aten.native_layer_norm.default": _fs_layer_norm,
    "aten._softmax.default": _fs_softmax, "aten.softmax.int": _fs_softmax,
    "aten.special_softmax.int": _fs_softmax,
    "aten.embedding.default": _fs_embedding,
    "aten.index.Tensor": _fs_gather, "aten.index_select.default": _fs_gather,
    "aten.gather.default": _fs_gather, "aten.scatter.src": _fs_gather,
    "aten.scatter_.src": _fs_gather, "aten.scatter_add.default": _fs_gather,
    "aten._to_copy.default": lambda n: _mk(
        "0 (cast)", "0", "|input|·b_in", str(n.total_input_bytes()),
        "|output|·b_out", str(n.total_output_bytes())),
    "aten.copy_.default": lambda n: _mk(
        "0 (copy)", "0", "Σ|inputs|·b", str(n.total_input_bytes()),
        "Σ|outputs|·b", str(n.total_output_bytes())),
    "aten.new_empty.default": lambda n: _mk(
        "0 (alloc)", "0", "0", "0", "|output|·b", str(n.total_output_bytes())),
    "aten.new_empty_strided.default": lambda n: _mk(
        "0 (alloc)", "0", "0", "0", "|output|·b", str(n.total_output_bytes())),
    "aten.fill_.Scalar": lambda n: _mk(
        "0 (fill)", "0", "0", "0", "|output|·b", str(n.total_output_bytes())),
    "aten.zero_.default": lambda n: _mk(
        "0 (zero)", "0", "0", "0", "|output|·b", str(n.total_output_bytes())),
    # fused semantic labels
    "rms_norm": _fs_rms_norm, "layer_norm": _fs_layer_norm,
    "add_rms_norm": _fs_add_norm, "add_layer_norm": _fs_add_norm,
    "npu_add_rms_norm": _fs_add_norm, "norm_backward": _fs_add_norm,
    "flash_attn": _fs_sdpa, "sdpa": _fs_sdpa, "sdpa_backward": _fs_sdpa,
    "npu_fusion_attention": _fs_sdpa, "attn": _fs_sdpa, "attn_grad": _fs_sdpa,
    "mla_attn": _fs_sdpa,
    "gated_mlp": _fs_mlp, "gated_mlp_backward": _fs_mlp,
    "mlp": _fs_mlp, "mlp_backward": _fs_mlp,
    "moe_block": _fs_mlp, "moe_expert": _fs_mlp, "moe_shared": _fs_mlp,
    "moe_gate": _fs_linear, "moe_gate_topk": _fs_linear,
    "npu_moe_gate": _fs_linear, "npu_moe_gate_topk": _fs_linear,
    "moe_dispatch": _fs_gather, "npu_moe_dispatch": _fs_gather,
    "embedding": _fs_embedding, "embedding_backward": _fs_embedding,
    "rope": lambda n: _fs_elementwise(n, 2.0, "2"),
    "Linear": _fs_linear_proj, "lm_head": _fs_linear_proj,
}


def get_op_formulas(node: "OpNode") -> dict[str, str]:
    """Return symbolic and numeric formula strings for a node (used for Excel export).

    Returns a dict with keys:
      flops_sym, flops_num  — compute formula (symbolic / with actual numbers)
      read_sym,  read_num   — read bytes formula
      write_sym, write_num  — write bytes formula
    """
    op = node.op_type

    if node.is_comm:
        return _fs_comm(node)

    fn = _FORMULA_DISPATCH.get(op)
    if fn is not None:
        return fn(node)

    ew = _EW_DISPATCH.get(op)
    if ew is not None:
        k, ks = ew
        return _fs_elementwise(node, k, ks)

    for prefix in _SHAPE_OP_PREFIXES:
        if op.startswith(prefix):
            return _fs_shape(node)

    return _fs_default(node)
