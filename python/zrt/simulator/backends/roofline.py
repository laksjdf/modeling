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
│ 卷积 (Convolution)                                                                              │
│   conv2d       │ aten.convolution / aten._convolution   │ 2·N·Cout·Hout·Wout·Cin·Kh·Kw         │
│   conv3d       │ aten.conv2d / conv3d                   │ R=input+weight+bias  W=output        │
│                │ weight=(Cout, Cin/groups, Kh, Kw)      │ groups时 Cin/groups 替代 Cin         │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 注意力 (Attention)                                                                              │
│   sdpa         │ aten.scaled_dot_product_attention       │ 4·N·H·Sq·Sk·D + 5·N·H·Sq·Sk          │
│   flash_attn   │ aten._scaled_dot_product_flash_attn    │   (QK matmul + AV matmul + softmax)   │
│   paged_attn   │ PageAttentionFP16 / paged_attention    │ 同上 + block_tables索引开销          │
│   sparse_attn  │ SparseFlashAttention / local_attention │ (同上) * sparsity_ratio              │
│                │                                        │ R=(Q + ratio*K + ratio*V)*b  W=output│
│   mla_attn     │ flash_attn / sdpa / mla_attn           │ ratio从annotations或attrs获取        │
│   sdpa_backward│ attn_grad                              │ 同上 (backward 代入 grad shape)        │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 归一化 (Norm)                                                                                   │
│   rms_norm     │ rms_norm                               │ 4·N  (sq+mean+rsqrt+scale)            │
│                │                                        │ R=(N+|weight|)·b  W=N·b               │
│   gemma_norm   │ GemmaRMSNorm / gemma_rms_norm          │ 4·N (与 rms_norm 相同)                │
│                │                                        │ R=(N+|weight|)·b  W=N·b               │
│   rms_gated    │ RMSNormGated / rms_norm_gated          │ 9·N  (rms_norm 4N + sigmoid 4N + mul) │
│                │                                        │ R=(N+|weight|+|gate|)·b  W=N·b        │
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
│ 排序 (Sort / Topk) — O(N log N)                                                                 │
│   sort         │ aten.sort / sort.values / sort.indices  │ 2·N·log₂(N)  (compare + swap)         │
│   topk         │ aten.topk / argsort                     │ R=N·b  W=(values+indices)·b           │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 逐元素 — 1 op/elem                                                                              │
│                │ add / sub / rsub / mul / div / neg /   │ 1·N                                   │
│                │ abs / relu / tanh / exp / log /        │ R=sum(|inputs|)·b                     │
│                │ sqrt / rsqrt / pow / masked_fill        │ W=|output|·b                          │
│                │ mean / sum / amax / amin (reduction)    │                                       │
│                │ cumsum / cumprod (scan/prefix sum)      │ 1·N (scan操作)                        │
│                │ copy_                                   │ 0 FLOPs, R=input·b, W=output·b        │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 逐元素 — 2 ops/elem                                                                             │
│                │ reciprocal / clamp / clamp_min/max     │ 2·N                                   │
│                │ var (reduce: sq+mean+sub+sq+mean ≈ 3N) │ 3·N                                   │
│                │ rope (cos*x + sin*x_rot)               │ 2·N                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 激活 — 4 ops/elem                                                                               │
│                │ silu / SiLUActivation (x·σ(x), σ≈4)   │ 4·N                                   │
│                │ gelu  (~x·Φ(x), ≈4 ops)               │ 4·N                                   │
│                │ sigmoid (1/(1+e^-x))                   │ 4·N                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 超越函数 — 10 ops/elem (CORDIC / polynomial)                                                   │
│                │ sin / cos / atan2  (用于 RoPE)          │ 10·N                                  │
│                │                                        │ R=input·b  W=output·b                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ MLP / 专家层 (MLP / MoE expert)                                                                 │
│   swiglu       │ swiglu                                 │ 4·batch·H·I+2·batch·I·H+5·batch·I     │
│                │ (gate+up proj, act, down proj)         │ R=hidden+3·weights·b  W=output·b      │
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
    """Return the dominant dtype for compute-throughput lookup.

    For compute nodes, activation dtype (inputs[0]) drives tensor-core selection.
    Falls back to output dtype for non-compute or no-input nodes.
    Also respects quant_act annotation for hypothetical-quant analysis.
    """
    if node.category != "compute" or not node.inputs:
        if node.outputs:
            return node.outputs[0].dtype
        return DType.BF16
    # Annotation path: QuantizationPass wrote quant_act on this node
    quant_act = node.annotations.get("quant_act")
    if quant_act and quant_act not in ("bf16", "fp16", "fp32"):
        # Normalize fp8 alias to fp8_e4m3 (default FP8 format for training)
        normalized = "fp8_e4m3" if quant_act == "fp8" else quant_act
        try:
            return DType(normalized)
        except ValueError:
            pass
    # Captured dtype path: use activation tensor (inputs[0]) dtype
    return node.inputs[0].dtype


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


def _convolution(node: "OpNode") -> FMR:
    """aten.convolution.default / aten._convolution.default / conv2d / conv3d.

    2D Convolution: input=(N,Cin,H,W), weight=(Cout,Cin/groups,Kh,Kw)
    Output shape: (N, Cout, Hout, Wout)
    Hout = (H + 2*padding - Kh) / stride + 1
    Wout = (W + 2*padding - Kw) / stride + 1

    FLOPs = 2 * N * Cout * Hout * Wout * (Cin/groups * Kh * Kw) * groups
          = 2 * N * Cout * Hout * Wout * Cin * Kh * Kw (当 groups=1)

    3D Convolution: input=(N,Cin,D,H,W), weight=(Cout,Cin/groups,Kd,Kh,Kw)
    FLOPs = 2 * N * Cout * Dout * Hout * Wout * Cin * Kd * Kh * Kw

    Depthwise Conv (groups=Cin): FLOPs = 2 * N * Cin * Hout * Wout * Kh * Kw

    R = |input| + |weight| [+ |bias|]
    W = |output|
    """
    if len(node.inputs) < 2:
        return _default(node)

    inp = node.inputs[0]
    weight = node.inputs[1]

    if len(inp.shape) < 3 or len(weight.shape) < 4:
        return _default(node)

    # Determine conv dimension from weight shape
    # weight shape: (Cout, Cin/groups, Kh, Kw) for 2D
    #              (Cout, Cin/groups, Kd, Kh, Kw) for 3D
    weight_ndim = len(weight.shape) - 2  # spatial dims

    # Batch size and input channels
    N = inp.shape[0]
    Cin = inp.shape[1]

    # Output channels and kernel size
    Cout = weight.shape[0]
    Cin_per_group = weight.shape[1]

    # Groups (from weight shape: Cin/groups == weight.shape[1])
    groups = Cin // Cin_per_group if Cin_per_group > 0 else 1

    # Kernel spatial dimensions
    kernel_dims = weight.shape[2:]  # (Kh, Kw) or (Kd, Kh, Kw)
    kernel_size = 1
    for k in kernel_dims:
        kernel_size *= k

    # Output spatial dimensions (from output tensor if available)
    if node.outputs:
        out = node.outputs[0]
        spatial_dims = out.shape[2:]  # (Hout, Wout) or (Dout, Hout, Wout)
        output_spatial = 1
        for s in spatial_dims:
            output_spatial *= s
    else:
        # Estimate from input and kernel (assuming stride=1, padding=0)
        input_spatial_dims = inp.shape[2:]
        output_spatial = 1
        for i, k in zip(input_spatial_dims, kernel_dims):
            output_spatial *= max(1, i - k + 1)

    # FLOPs calculation
    # For each output element: multiply Cin/groups * kernel_size weights and add
    # groups * Cout output channels, N batch
    flops = 2.0 * N * Cout * output_spatial * (Cin_per_group * kernel_size)

    # Memory bandwidth
    it = inp.dtype.itemsize
    read = float(inp.mem_bytes + weight.mem_bytes)
    if len(node.inputs) >= 3:  # bias
        bias = node.inputs[2]
        flops += N * Cout * output_spatial  # bias add
        read += float(bias.mem_bytes)
    write = float(node.outputs[0].mem_bytes if node.outputs else N * Cout * output_spatial * it)

    return flops, read, write


def _scaled_dot_product_attention(node: "OpNode") -> FMR:
    """aten._scaled_dot_product_flash_attention / scaled_dot_product_attention.

    Input layout assumed: Q=(N,H,Sq,D), K=(N,H,Sk,D), V=(N,H,Sk,Dv)
    FLOPs = 4·N·H·Sq·Sk·D        (QK + AV matmuls)
          + 4·N·H·Sq·Sk           (softmax: max+sub+exp+sum+div)
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
    # Softmax ops ~ 4*N*H*Sq*Sk (sub-dominant, included for completeness)
    flops += 4.0 * N * H * Sq * Sk
    read  = (N*H*Sq*D + N*H*Sk*D + N*H*Sk*D) * it   # Q + K + V
    write = (N*H*Sq*D) * it                           # output
    return flops, read, write


def _paged_attention(node: "OpNode") -> FMR:
    """PageAttentionFP16 / paged_attention (vLLM PagedAttention kernel).

    PagedAttention 允许 KV cache 以非连续内存块存储，减少内存碎片。
    计算上与 FlashAttention 相同，但内存访问通过 block tables 索引。

    Input layout: Q=(num_tokens,H,D), K/V cache 在 page blocks 中
    - block_tables: (num_tokens, max_blocks_per_seq) 紧张映射
    - KV cache: (max_num_blocks, H, block_size, D)

    FLOPs = 4*N*H*Sq*Sk*D + 5*N*H*Sq*Sk  (与 FlashAttention 相同)
    R = Q*b + K*b + V*b + block_tables*sizeof(int)  (额外索引开销)
    W = output*b

    内存带宽估计略高于 FlashAttention，因为 KV 非连续访问有额外开销。
    """
    if not node.inputs:
        return _default(node)

    q = node.inputs[0]
    if len(q.shape) < 3:
        return _default(node)

    # Q shape: (num_tokens, H, D) or (batch, seq_len, H, D)
    if len(q.shape) == 3:
        N, H, D = q.shape[0], q.shape[1], q.shape[2]
        Sq = 1  # decode 阶段通常是 1
    elif len(q.shape) == 4:
        N, H, Sq, D = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
    else:
        return _default(node)

    it = q.dtype.itemsize

    # KV cache length 需从 block_tables 或 node.annotations 估算
    # 若无信息，假设 Sk = Sq * 4 (典型解码场景)
    Sk = node.annotations.get("kv_cache_len", Sq * 4) if node.annotations else Sq * 4

    # FLOPs 与 FlashAttention 相同
    flops = 4.0 * N * H * Sq * Sk * D
    flops += 5.0 * N * H * Sq * Sk

    # 内存带宽：Q + K + V + block_tables 索引
    q_bytes = N * H * Sq * D * it
    kv_bytes = N * H * Sk * D * it * 2  # K + V
    block_table_bytes = N * Sk * 8  # int64 indices, 每个 KV block 一个索引
    read = q_bytes + kv_bytes + block_table_bytes
    write = N * H * Sq * D * it

    return flops, read, write


def _sparse_flash_attention(node: "OpNode") -> FMR:
    """SparseFlashAttention / sparse_attention / local_attention.

    SparseFlashAttention 只计算部分位置的注意力，用于长序列场景。
    稀疏模式包括：local window、global attention、random attention、block sparse 等。

    Input layout: Q=(N,H,Sq,D), K=(N,H,Sk,D), V=(N,H,Sk,D)
    - sparsity_ratio: 实际计算的注意力位置比例 (0 < ratio <= 1)
    - 可从 node.annotations["sparsity_ratio"] 或 attrs["window_size"]/Sk 估算

    FLOPs = (4*N*H*Sq*Sk*D + 5*N*H*Sq*Sk) * sparsity_ratio
    R = (Q + sparsity_ratio*K + sparsity_ratio*V)*b  (只访问稀疏位置的 KV)
    W = output*b

    典型稀疏比例：
    - local window (window_size=256, Sk=4096): ratio ≈ 256/4096 = 0.0625
    - block sparse (block_size=64, dense_blocks=10%): ratio ≈ 0.1
    - Longformer (local + global): ratio ≈ 0.1-0.2
    """
    if len(node.inputs) < 3:
        return _default(node)

    q, k, v = node.inputs[0], node.inputs[1], node.inputs[2]
    if len(q.shape) < 4 or len(k.shape) < 4:
        return _default(node)

    N, H, Sq, D = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
    Sk = k.shape[2]
    it = q.dtype.itemsize

    # 获取稀疏比例，优先级：annotations > attrs > 默认值
    sparsity_ratio = 1.0
    if node.annotations and "sparsity_ratio" in node.annotations:
        sparsity_ratio = float(node.annotations["sparsity_ratio"])
    elif node.attrs:
        # 从 window_size 估算
        if "window_size" in node.attrs:
            window_size = int(node.attrs["window_size"])
            sparsity_ratio = min(1.0, window_size / Sk)
        # 从 block_sparse 配置估算
        elif "block_size" in node.attrs and "num_dense_blocks" in node.attrs:
            block_size = int(node.attrs["block_size"])
            num_dense_blocks = int(node.attrs["num_dense_blocks"])
            total_blocks = Sk // block_size
            sparsity_ratio = num_dense_blocks / total_blocks if total_blocks > 0 else 1.0

    # 限制 sparsity_ratio 在合理范围
    sparsity_ratio = max(0.01, min(1.0, sparsity_ratio))

    # FLOPs 乘以稀疏比例
    flops = (4.0 * N * H * Sq * Sk * D + 5.0 * N * H * Sq * Sk) * sparsity_ratio

    # 内存带宽：Q 全部访问，KV 只访问稀疏位置
    q_bytes = N * H * Sq * D * it
    kv_bytes = N * H * int(Sk * sparsity_ratio) * D * it * 2  # K + V (稀疏部分)
    read = q_bytes + kv_bytes
    write = N * H * Sq * D * it

    return flops, read, write


def _layer_norm(node: "OpNode") -> FMR:
    """aten.layer_norm.default / aten.native_layer_norm.default
    FLOPs ≈ 7·N  (mean + var_sub + var_sq + mean_var + rsqrt + normalize + scale + shift)
    R=(N + 2·|weight|)·b    W=N·b
    """
    if not node.inputs:
        return _default(node)
    inp = node.inputs[0]
    n = _numel(inp.shape)
    it = inp.dtype.itemsize
    # mean(N) + var_sub(N) + var_sq(N) + mean_var(N) + rsqrt(N) + normalize(N) + scale(N) + shift(N) ≈ 7N flops
    flops = 7.0 * n
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


def _rms_norm_gated(node: "OpNode") -> FMR:
    """RMSNormGated / rms_norm_gated: RMSNorm with gating mechanism.
    
    RMSNormGated(x) = RMSNorm(x, weight) ⊗ sigmoid(gate)
    
    输入布局：
    - inputs[0]: hidden tensor (B, S, H) 或 (N,)
    - inputs[1]: norm weight (H,)
    - inputs[2]: gate tensor (可选，与 hidden 同形状) 或 gate weight (H,)
    
    计算步骤：
    1. RMSNorm: 4·N FLOPs (pow+mean+rsqrt+mul+scale)
    2. Sigmoid(gate): 4·N FLOPs (sigmoid 约4 ops/elem)
    3. Multiply: 1·N FLOPs
    总计: 4·N + 4·N + 1·N = 9·N FLOPs
    
    若 gate 通过 matmul 计算（有 gate weight 而非 gate tensor）：
    - gate = x @ W_gate: 2·batch·H·gate_dim FLOPs
    - 此时总计: 4·N + 2·batch·H·gate_dim + 5·N
    
    R = |hidden| + |weight| + |gate| (或 |gate_weight|)
    W = |output|
    """
    if len(node.inputs) < 2:
        return _default(node)
    
    inp = node.inputs[0]
    norm_weight = node.inputs[1]
    
    n = _numel(inp.shape)
    it = inp.dtype.itemsize
    
    # RMSNorm FLOPs
    flops = 4.0 * n
    
    # Gate FLOPs: sigmoid(4N) + multiply(1N) = 5N
    # 如果有第三个输入，判断是 gate tensor 还是 gate weight
    if len(node.inputs) >= 3:
        gate = node.inputs[2]
        gate_n = _numel(gate.shape)
        
        # 如果 gate 形状与 inp 相同，是 gate tensor
        if gate_n == n:
            flops += 5.0 * n  # sigmoid + multiply
            weight_size = inp.shape[-1] if inp.shape else 1
            gate_weight_size = gate.shape[-1] if gate.shape else 1
            read = (n + weight_size + gate_n) * it
        # 如果 gate 是 weight 形状 (gate_dim, hidden_dim)，是 gate weight matmul
        elif len(gate.shape) >= 2:
            # gate matmul: x_norm @ W_gate
            batch = _numel(inp.shape[:-1]) if len(inp.shape) > 1 else 1
            H = inp.shape[-1] if inp.shape else 1
            gate_out = gate.shape[0] if gate.shape[1] == H else gate.shape[1]
            flops += 2.0 * batch * H * gate_out + 5.0 * batch * gate_out
            weight_size = H
            read = (n + weight_size) * it + gate.mem_bytes
        else:
            # 默认：假设 gate tensor
            flops += 5.0 * n
            weight_size = inp.shape[-1] if inp.shape else 1
            read = (n + weight_size + gate_n) * it
    else:
        # 无 gate 输入，假设自 gating (对 norm output 做 sigmoid)
        flops += 5.0 * n
        weight_size = inp.shape[-1] if inp.shape else 1
        read = (n + weight_size) * it
    
    write = n * it
    return flops, read, write


def _softmax(node: "OpNode") -> FMR:
    """aten._softmax.default / aten.softmax.int
    FLOPs ≈ 4·N  (max + sub + exp + sum + div)
    R=N·b    W=N·b
    """
    if not node.inputs:
        return _default(node)
    inp = node.inputs[0]
    n = _numel(inp.shape)
    it = inp.dtype.itemsize
    # max(N) + sub(N) + exp(N) + sum(N) + div(N) ≈ 4N
    flops = 4.0 * n
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


def _sort(node: "OpNode") -> FMR:
    """aten.sort.default / aten.sort.values / topk / argsort
    
    Sort 算子复杂度：O(N log₂ N) 比较操作
    - GPU 并行排序常用 radix sort (k·N) 或 bitonic sort (N·log²N)
    - 保守估计用 2·N·log₂N FLOPs（每次比较约 2 ops: compare + swap condition）
    
    FLOPs ≈ 2·N·log₂(N)   (N 为待排序元素数)
    R = N·b   W = 2·N·b  (输出 values + indices)
    
    对于 topk: FLOPs ≈ N·k·log₂(N/k) 或更小，这里仍用保守估计
    """
    if not node.inputs:
        return _default(node)
    
    inp = node.inputs[0]
    n = _numel(inp.shape)
    it = inp.dtype.itemsize
    
    # log2(N) with minimum 1 to avoid 0 for small tensors
    log_n = max(1.0, math.log2(max(n, 1)))
    
    # 2 ops per comparison (compare + conditional swap)
    flops = 2.0 * n * log_n
    
    # Read input, write sorted values + indices
    read = float(node.total_input_bytes())
    write = float(node.total_output_bytes())
    
    return flops, read, write


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


def _swiglu(node: "OpNode") -> FMR:
    """SwiGLU activation: gate + up projections, sigmoid activation, down projection.
    
    SwiGLU(x) = Swish(x·W_gate) ⊗ (x·W_up) · W_down
              = (sigmoid(x·W_gate) ⊗ x·W_gate ⊗ x·W_up) · W_down
    
    Typical input layout: [hidden=(B,S,H), gate_w=(I,H), up_w=(I,H), down_w=(H,I)]
    
    FLOPs = 2·batch·H·I (gate_proj) + 2·batch·H·I (up_proj)
          + 5·batch·I  (sigmoid 4ops + mul 1op)
          + 2·batch·I·H (down_proj)
          ≈ 4·batch·H·I + 2·batch·I·H + 5·batch·I
    
    R = hidden·b + |gate_w|·b + |up_w|·b + |down_w|·b
    W = output·b (same shape as hidden)
    """
    if len(node.inputs) < 4:
        return _default(node)
    
    hidden = node.inputs[0]
    gate_w, up_w, down_w = node.inputs[1], node.inputs[2], node.inputs[3]
    
    if len(hidden.shape) < 1 or len(gate_w.shape) < 2:
        return _default(node)
    
    batch = _numel(hidden.shape[:-1])
    H = hidden.shape[-1]
    
    I_gate = gate_w.shape[0] if gate_w.shape[1] == H else gate_w.shape[1]
    I_up   = up_w.shape[0] if up_w.shape[1] == H else up_w.shape[1]
    I = max(I_gate, I_up)
    
    it = hidden.dtype.itemsize
    
    flops = 4.0 * batch * H * I + 2.0 * batch * I * H + 5.0 * batch * I
    
    read  = (batch * H + I * H + I * H + H * I) * it
    write = batch * H * it
    
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


def _repo_kernel(node: "OpNode") -> FMR:
    """Heuristic for RepoKernel: try to detect a matmul-like pattern (two 2-D inputs)
    and fall back to default. """
    if len(node.inputs) >= 2:
        a, b = node.inputs[0], node.inputs[1]
        if len(a.shape) >= 2 and len(b.shape) >= 2:
            # reuse mm logic
            return _mm(node)
    return _default(node)


def _moe_infer(node: "OpNode") -> FMR:
    """MoE inference fused block: approximate with fused_mlp cost."""
    return _fused_mlp(node)


def _repo_interleave(node: "OpNode") -> FMR:
    """Interleave / de-interleave used in repo routing — treat as gather/scatter (0 flops)."""
    return _gather(node)


def _repo_complex(node: "OpNode") -> FMR:
    """Complex-number specialized ops: conservative elementwise estimate (2 ops/elem)."""
    return _elementwise(node, 2.0)


def _lightning_indexer(node: "OpNode") -> FMR:
    """Indexer-like ops: routing / indexing — zero-flop gather semantics."""
    return _gather(node)


def _indexer_prolog(node: "OpNode") -> FMR:
    """Prolog for indexer pipelines: treat as gather."""
    return _gather(node)


def _grouped_mm(node: "OpNode") -> FMR:
    """Grouped matmul: first dim is group count G: (G,M,K) @ (G,K,N) -> (G,M,N).
    Falls back to mm/bmm heuristics when shapes differ.
    """
    if len(node.inputs) < 2:
        return _default(node)
    a, b = node.inputs[0], node.inputs[1]
    if len(a.shape) >= 3 and len(b.shape) >= 3:
        G, M, K = a.shape[0], a.shape[1], a.shape[2]
        N = b.shape[-1]
        it = a.dtype.itemsize
        flops = 2.0 * G * M * N * K
        read = (G * M * K + G * K * N) * it
        write = G * M * N * it
        return flops, read, write
    # fallback
    return _mm(node)


def _concat_fp16(node: "OpNode") -> FMR:
    """Concat + cast to fp16: no compute, copy inputs to output (read inputs, write output)."""
    # read: sum of input bytes, write: output bytes
    read = float(node.total_input_bytes())
    write = float(node.total_output_bytes())
    return 0.0, read, write



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
    # ── convolution ────────────────────────────────────────────────────────────
    "aten.convolution.default":         _convolution,
    "aten._convolution.default":        _convolution,
    "aten.conv2d.default":              _convolution,
    "aten.conv3d.default":              _convolution,
    "conv2d":                           _convolution,
    "conv3d":                           _convolution,
    "Conv":                             _convolution,
    # ── attention ─────────────────────────────────────────────────────────────
    "aten._scaled_dot_product_flash_attention.default":    _scaled_dot_product_attention,
    "aten.scaled_dot_product_attention.default":           _scaled_dot_product_attention,
    "aten._scaled_dot_product_efficient_attention.default":_scaled_dot_product_attention,
    # ── paged attention (vLLM) ──────────────────────────────────────────────
    "PageAttentionFP16":                _paged_attention,
    "paged_attention":                  _paged_attention,
    "paged_attention_fp16":             _paged_attention,
    # ── sparse attention (Longformer, BigBird, local window) ──────────────
    "SparseFlashAttention":             _sparse_flash_attention,
    "sparse_attention":                 _sparse_flash_attention,
    "sparse_flash_attention":           _sparse_flash_attention,
    "local_attention":                  _sparse_flash_attention,
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
    "AddInplace":                       _elementwise,
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
    "SiLUActivation":                   lambda n: _elementwise(n, 4.0),
    "silu":                             lambda n: _elementwise(n, 4.0),
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
    # ── scan / prefix sum (cumsum, cumprod) ────────────────────────────────────
    "aten.cumsum.default":              lambda n: _elementwise(n, 1.0),
    "aten.cumsum.dim":                  lambda n: _elementwise(n, 1.0),
    "cumsum":                           lambda n: _elementwise(n, 1.0),
    "aten.cumprod.default":             lambda n: _elementwise(n, 1.0),
    "aten.cumprod.dim":                 lambda n: _elementwise(n, 1.0),
    "cumprod":                          lambda n: _elementwise(n, 1.0),
    # ── sort / topk (O(N log N) complexity) ─────────────────────────────────────
    "aten.sort.default":                _sort,
    "aten.sort.values":                 _sort,
    "aten.sort.indices":                _sort,
    "aten.topk.default":                _sort,
    "aten.argsort.default":             _sort,
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
    "GemmaRMSNorm":                     _rms_norm,
    "gemma_rms_norm":                   _rms_norm,
    "RMSNormGated":                     _rms_norm_gated,
    "rms_norm_gated":                   _rms_norm_gated,
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
    "paged_attn":                       _paged_attention,
    # MLP / SwiGLU
    "swiglu":                           _swiglu,
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
    # Repository / project-specific ops
    "RepoKernel":                       _repo_kernel,
    "MoEInfer":                         _moe_infer,
    "RepoInterleave":                   _repo_interleave,
    "RepoComplex":                      _repo_complex,
    "LightningIndexer":                 _lightning_indexer,
    "IndexerProlog":                    _indexer_prolog,
    "GroupedMm":                        _grouped_mm,
    "GroupedMatMul":                    _grouped_mm,
    "ConcatFP16":                       _concat_fp16,
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


def _fs_convolution(node: "OpNode") -> dict:
    if len(node.inputs) < 2: return _fs_default(node)
    inp, weight = node.inputs[0], node.inputs[1]
    if len(inp.shape) < 3 or len(weight.shape) < 4: return _fs_default(node)

    N = inp.shape[0]
    Cin = inp.shape[1]
    Cout = weight.shape[0]
    Cin_per_group = weight.shape[1]
    groups = Cin // Cin_per_group if Cin_per_group > 0 else 1

    kernel_dims = weight.shape[2:]
    kernel_size = 1
    for k in kernel_dims:
        kernel_size *= k

    if node.outputs:
        out = node.outputs[0]
        spatial_dims = out.shape[2:]
        output_spatial = 1
        for s in spatial_dims:
            output_spatial *= s
    else:
        input_spatial_dims = inp.shape[2:]
        output_spatial = 1
        for i, k in zip(input_spatial_dims, kernel_dims):
            output_spatial *= max(1, i - k + 1)

    flops = 2 * N * Cout * output_spatial * (Cin_per_group * kernel_size)
    if len(node.inputs) >= 3:
        flops += N * Cout * output_spatial

    ri = inp.mem_bytes + weight.mem_bytes
    if len(node.inputs) >= 3:
        ri += node.inputs[2].mem_bytes
    wo = node.outputs[0].mem_bytes if node.outputs else N * Cout * output_spatial * _bw(node)

    kernel_str = "*".join(str(k) for k in kernel_dims)
    return _mk(f"2·N·Cout·Hout·Wout·Cin·{kernel_str}",
               f"{flops} (groups={groups})",
               "input+weight+bias·b", str(ri),
               "output·b", str(wo))


def _fs_sdpa(node: "OpNode") -> dict:
    if len(node.inputs) < 3: return _fs_default(node)
    q, k = node.inputs[0], node.inputs[1]
    if len(q.shape) < 4: return _fs_default(node)
    N, H, Sq, D = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
    Sk, bw = (k.shape[2] if len(k.shape) >= 3 else Sq), _bw(node)
    return _mk("4·N·H·Sq·Sk·D+4·N·H·Sq·Sk",
               f"4·{N}·{H}·{Sq}·{Sk}·{D}+4·{N}·{H}·{Sq}·{Sk}",
               "(Q+K+V)·b",
               f"({N}·{H}·{Sq}·{D}+{N}·{H}·{Sk}·{D}+{N}·{H}·{Sk}·{D})·{bw}",
               "N·H·Sq·D·b", f"{N}·{H}·{Sq}·{D}·{bw}")


def _fs_paged_attention(node: "OpNode") -> dict:
    if len(node.inputs) < 1: return _fs_default(node)
    q = node.inputs[0]
    if len(q.shape) < 3: return _fs_default(node)
    if len(q.shape) == 3:
        N, H, D = q.shape[0], q.shape[1], q.shape[2]
        Sq = 1
    elif len(q.shape) == 4:
        N, H, Sq, D = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
    else:
        return _fs_default(node)
    Sk = node.annotations.get("kv_cache_len", Sq * 4) if node.annotations else Sq * 4
    bw = _bw(node)
    q_bytes = N * H * Sq * D * bw
    kv_bytes = N * H * Sk * D * bw * 2
    block_bytes = N * Sk * 8
    return _mk("4·N·H·Sq·Sk·D+5·N·H·Sq·Sk",
               f"4·{N}·{H}·{Sq}·{Sk}·{D}+5·{N}·{H}·{Sq}·{Sk}",
               "(Q+K+V+block_tables)·b",
               f"{q_bytes+kv_bytes+block_bytes}",
               "N·H·Sq·D·b", f"{N}·{H}·{Sq}·{D}·{bw}")


def _fs_sparse_flash_attention(node: "OpNode") -> dict:
    if len(node.inputs) < 3: return _fs_default(node)
    q, k = node.inputs[0], node.inputs[1]
    if len(q.shape) < 4: return _fs_default(node)
    N, H, Sq, D = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
    Sk, bw = (k.shape[2] if len(k.shape) >= 3 else Sq), _bw(node)

    # 获取稀疏比例
    sparsity_ratio = 1.0
    if node.annotations and "sparsity_ratio" in node.annotations:
        sparsity_ratio = float(node.annotations["sparsity_ratio"])
    elif node.attrs and "window_size" in node.attrs:
        window_size = int(node.attrs["window_size"])
        sparsity_ratio = min(1.0, window_size / Sk)

    sparsity_ratio = max(0.01, min(1.0, sparsity_ratio))
    sparse_Sk = int(Sk * sparsity_ratio)

    # FLOPs 公式
    base_flops = 4 * N * H * Sq * Sk * D + 5 * N * H * Sq * Sk
    actual_flops = int(base_flops * sparsity_ratio)

    # 内存带宽
    q_bytes = N * H * Sq * D * bw
    kv_bytes = N * H * sparse_Sk * D * bw * 2

    return _mk(f"(4·N·H·Sq·Sk·D+5·N·H·Sq·Sk)·ratio",
               f"{actual_flops} (ratio={sparsity_ratio:.3f})",
               "(Q+ratio*K+ratio*V)*b",
               f"{q_bytes+kv_bytes}",
               "N·H·Sq·D·b", f"{N}·{H}·{Sq}·{D}·{bw}")


def _fs_rms_norm(node: "OpNode") -> dict:
    if not node.inputs: return _fs_default(node)
    inp = node.inputs[0]
    N, W, bw = _numel(inp.shape), (inp.shape[-1] if inp.shape else 1), _bw(node)
    return _mk("4·N", f"4·{N}", "(N+|W|)·b", f"({N}+{W})·{bw}", "N·b", f"{N}·{bw}")


def _fs_rms_norm_gated(node: "OpNode") -> dict:
    if len(node.inputs) < 2: return _fs_default(node)
    inp = node.inputs[0]
    N, W, bw = _numel(inp.shape), (inp.shape[-1] if inp.shape else 1), _bw(node)
    
    # 计算 gate 相关的 FLOPs
    if len(node.inputs) >= 3:
        gate = node.inputs[2]
        gate_n = _numel(gate.shape)
        # 如果 gate 是 tensor（与 inp 同形状），9N FLOPs
        if gate_n == N:
            flops_str = "9·N"
            flops_num = f"9·{N}"
            read_str = "(N+|W|+|gate|)·b"
            read_num = f"({N}+{W}+{gate_n})·{bw}"
        # 如果 gate 是 weight matrix，需要额外 matmul
        elif len(gate.shape) >= 2:
            batch = _numel(inp.shape[:-1]) if len(inp.shape) > 1 else 1
            H = inp.shape[-1] if inp.shape else 1
            gate_out = gate.shape[0] if gate.shape[1] == H else gate.shape[1]
            extra_flops = 2 * batch * H * gate_out + 5 * batch * gate_out
            flops_str = f"4·N+2·batch·H·gate_out+5·batch·gate_out"
            flops_num = f"4·{N}+{extra_flops}"
            read_str = "(N+|W|+|gate_w|)·b"
            read_num = f"({N}+{W})·{bw}+{gate.mem_bytes}"
        else:
            flops_str = "9·N"
            flops_num = f"9·{N}"
            read_str = "(N+|W|+|gate|)·b"
            read_num = f"({N}+{W}+{gate_n})·{bw}"
    else:
        # 无 gate 输入，自 gating
        flops_str = "9·N"
        flops_num = f"9·{N}"
        read_str = "(N+|W|)·b"
        read_num = f"({N}+{W})*{bw}"
    
    return _mk(flops_str, flops_num, read_str, read_num, "N·b", f"{N}·{bw}")


def _fs_layer_norm(node: "OpNode") -> dict:
    if not node.inputs: return _fs_default(node)
    inp = node.inputs[0]
    N, W, bw = _numel(inp.shape), (inp.shape[-1] if inp.shape else 1), _bw(node)
    return _mk("7·N", f"7·{N}", "(N+2·|W|)·b", f"({N}+2·{W})·{bw}", "N·b", f"{N}·{bw}")


def _fs_add_norm(node: "OpNode") -> dict:
    if not node.inputs: return _fs_default(node)
    inp = node.inputs[0]
    N, W, bw = _numel(inp.shape), (inp.shape[-1] if inp.shape else 1), _bw(node)
    return _mk("6·N", f"6·{N}", "(2·N+|W|)·b", f"(2·{N}+{W})·{bw}", "N·b", f"{N}·{bw}")


def _fs_softmax(node: "OpNode") -> dict:
    if not node.inputs: return _fs_default(node)
    inp = node.inputs[0]
    N, bw = _numel(inp.shape), _bw(node)
    return _mk("4·N", f"4·{N}", "N·b", f"{N}·{bw}", "N·b", f"{N}·{bw}")


def _fs_sort(node: "OpNode") -> dict:
    if not node.inputs: return _fs_default(node)
    inp = node.inputs[0]
    N, bw = _numel(inp.shape), _bw(node)
    log_n = max(1.0, math.log2(max(N, 1)))
    ri, wo = node.total_input_bytes(), node.total_output_bytes()
    return _mk("2·N·log2(N)", f"2·{N}·log2({N})={int(2*N*log_n)}",
               "N·b", str(ri), "values+indices·b", str(wo))


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


def _fs_swiglu(node: "OpNode") -> dict:
    if len(node.inputs) < 4: return _fs_default(node)
    h = node.inputs[0]
    gate_w, up_w, down_w = node.inputs[1], node.inputs[2], node.inputs[3]
    if len(h.shape) < 1 or len(gate_w.shape) < 2: return _fs_default(node)
    batch, H, bw = _numel(h.shape[:-1]), h.shape[-1], _bw(node)
    I = max(gate_w.shape[0] if gate_w.shape[1] == H else gate_w.shape[1],
            up_w.shape[0] if up_w.shape[1] == H else up_w.shape[1])
    ri = h.mem_bytes + gate_w.mem_bytes + up_w.mem_bytes + down_w.mem_bytes
    wo = node.outputs[0].mem_bytes if node.outputs else batch * H * bw
    return _mk("4·batch·H·I+2·batch·I·H+5·batch·I",
               f"4·{batch}·{H}·{I}+2·{batch}·{I}·{H}+5·{batch}·{I}",
               "hidden+3·weights·b", str(ri),
               "|output|·b", str(wo))


def _fs_comm(node: "OpNode") -> dict:
    vol = sum(t.mem_bytes for t in node.outputs)
    return _mk("0 (comm)", "0", "comm_vol·b", str(vol), "comm_vol·b", str(vol))


def _fs_default(node: "OpNode") -> dict:
    n_out = sum(_numel(o.shape) for o in node.outputs) if node.outputs else 1
    ri, wo = node.total_input_bytes(), node.total_output_bytes()
    return _mk("N_out", str(n_out), "Σ|inputs|·b", str(ri), "Σ|outputs|·b", str(wo))


def _fs_repo_kernel(node: "OpNode") -> dict:
    # Try to mirror mm formatting when possible
    if len(node.inputs) >= 2:
        a, b = node.inputs[0], node.inputs[1]
        if len(a.shape) >= 2 and len(b.shape) >= 2:
            return _fs_mm(node)
    return _fs_default(node)


def _fs_moe_infer(node: "OpNode") -> dict:
    return _fs_mlp(node)


def _fs_repo_interleave(node: "OpNode") -> dict:
    return _fs_gather(node)


def _fs_repo_complex(node: "OpNode") -> dict:
    return _fs_elementwise(node, 2.0, "2")


def _fs_lightning_indexer(node: "OpNode") -> dict:
    return _fs_gather(node)


def _fs_indexer_prolog(node: "OpNode") -> dict:
    return _fs_gather(node)


def _fs_grouped_mm(node: "OpNode") -> dict:
    if len(node.inputs) < 2: return _fs_default(node)
    a, b = node.inputs[0], node.inputs[1]
    if len(a.shape) >= 3 and len(b.shape) >= 3:
        G, M, K, N, bw = a.shape[0], a.shape[1], a.shape[2], b.shape[-1], _bw(node)
        return _mk("2·G·M·K·N",
                   f"2·{G}·{M}·{K}·{N}",
                   "(G·M·K+G·K·N)·b",
                   f"({G}·{M}·{K}+{G}·{K}·{N})·{bw}",
                   "G·M·N·b",
                   f"{G}·{M}·{N}·{bw}")
    return _fs_mm(node)


def _fs_concat_fp16(node: "OpNode") -> dict:
    ri = node.total_input_bytes()
    wo = node.total_output_bytes()
    return _mk("0 (concat/cast)", "0", "Σ|inputs|·b", str(ri), "|output|·b", str(wo))


_EW_DISPATCH: dict[str, tuple] = {
    "aten.add.Tensor": (1.0, "1"), "aten.add_.Tensor": (1.0, "1"),
    "aten.add.Scalar": (1.0, "1"), "AddInplace": (1.0, "1"),
    "aten.sub.Tensor": (1.0, "1"),
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
    "aten.cumsum.default": (1.0, "1"), "aten.cumsum.dim": (1.0, "1"),
    "cumsum": (1.0, "1"),
    "aten.cumprod.default": (1.0, "1"), "aten.cumprod.dim": (1.0, "1"),
    "cumprod": (1.0, "1"),
    "aten.reciprocal.default": (2.0, "2"), "aten.clamp.default": (2.0, "2"),
    "aten.clamp.Scalar": (2.0, "2"), "aten.clamp.Tensor": (2.0, "2"),
    "aten.clamp_min.default": (2.0, "2"), "aten.clamp_max.default": (2.0, "2"),
    "aten.var.correction": (3.0, "3"),
    "aten.silu.default": (4.0, "4"), "aten.silu_.default": (4.0, "4"),
    "SiLUActivation": (4.0, "4"), "silu": (4.0, "4"),
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
    "aten.convolution.default": _fs_convolution,
    "aten._convolution.default": _fs_convolution,
    "aten.conv2d.default": _fs_convolution,
    "aten.conv3d.default": _fs_convolution,
    "conv2d": _fs_convolution,
    "conv3d": _fs_convolution,
    "Conv": _fs_convolution,
    "aten._scaled_dot_product_flash_attention.default": _fs_sdpa,
    "aten.scaled_dot_product_attention.default": _fs_sdpa,
    "aten._scaled_dot_product_efficient_attention.default": _fs_sdpa,
    "PageAttentionFP16": _fs_paged_attention,
    "paged_attention": _fs_paged_attention,
    "paged_attention_fp16": _fs_paged_attention,
    "SparseFlashAttention": _fs_sparse_flash_attention,
    "sparse_attention": _fs_sparse_flash_attention,
    "sparse_flash_attention": _fs_sparse_flash_attention,
    "local_attention": _fs_sparse_flash_attention,
    "aten.layer_norm.default": _fs_layer_norm, "aten.layer_norm": _fs_layer_norm,
    "aten.native_layer_norm.default": _fs_layer_norm,
    "aten._softmax.default": _fs_softmax, "aten.softmax.int": _fs_softmax,
    "aten.special_softmax.int": _fs_softmax,
    "aten.sort.default": _fs_sort, "aten.sort.values": _fs_sort,
    "aten.sort.indices": _fs_sort, "aten.topk.default": _fs_sort,
    "aten.argsort.default": _fs_sort,
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
    "rms_norm": _fs_rms_norm, "GemmaRMSNorm": _fs_rms_norm, "gemma_rms_norm": _fs_rms_norm,
    "RMSNormGated": _fs_rms_norm_gated, "rms_norm_gated": _fs_rms_norm_gated,
    "layer_norm": _fs_layer_norm,
    "add_rms_norm": _fs_add_norm, "add_layer_norm": _fs_add_norm,
    "npu_add_rms_norm": _fs_add_norm, "norm_backward": _fs_add_norm,
    "flash_attn": _fs_sdpa, "sdpa": _fs_sdpa, "sdpa_backward": _fs_sdpa,
    "npu_fusion_attention": _fs_sdpa, "attn": _fs_sdpa, "attn_grad": _fs_sdpa,
    "mla_attn": _fs_sdpa,
    "paged_attn": _fs_paged_attention,
    "gated_mlp": _fs_mlp, "gated_mlp_backward": _fs_mlp,
    "swiglu": _fs_swiglu,
    "mlp": _fs_mlp, "mlp_backward": _fs_mlp,
    "moe_block": _fs_mlp, "moe_expert": _fs_mlp, "moe_shared": _fs_mlp,
    "moe_gate": _fs_linear, "moe_gate_topk": _fs_linear,
    "npu_moe_gate": _fs_linear, "npu_moe_gate_topk": _fs_linear,
    "moe_dispatch": _fs_gather, "npu_moe_dispatch": _fs_gather,
    "embedding": _fs_embedding, "embedding_backward": _fs_embedding,
    "rope": lambda n: _fs_elementwise(n, 2.0, "2"),
    "Linear": _fs_linear_proj, "lm_head": _fs_linear_proj,
    # repo / project-specific formula entries
    "RepoKernel": _fs_repo_kernel, "MoEInfer": _fs_moe_infer,
    "RepoInterleave": _fs_repo_interleave, "RepoComplex": _fs_repo_complex,
    "LightningIndexer": _fs_lightning_indexer, "IndexerProlog": _fs_indexer_prolog,
    "GroupedMm": _fs_grouped_mm, "GroupedMatMul": _fs_grouped_mm,
    "ConcatFP16": _fs_concat_fp16,
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
