"""Integration tests for roofline.py operator FLOPs / read / write formulas.

基线用例，校验 roofline.py 中所有算子公式函数的输出正确性。
修改 roofline.py 公式后必须同步更新本文件。

Run:
    pytest tests/IT/test_roofline_formulas.py -v
"""
from __future__ import annotations

import math
import pytest

from python.zrt.ir.node import OpNode
from python.zrt.ir.types import DType, TensorMeta
from python.zrt.simulator.backends.roofline import RooflineSimulator


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tm(tid: str, shape: tuple[int, ...], dtype: DType = DType.BF16) -> TensorMeta:
    return TensorMeta.from_shape_dtype(tid, shape, dtype)


def _node(op_type: str, inputs: list[TensorMeta], outputs: list[TensorMeta],
          attrs: dict | None = None, fused_from: list[str] | None = None) -> OpNode:
    return OpNode(
        id="test", op_type=op_type, inputs=inputs, outputs=outputs,
        attrs=attrs or {}, fused_from=fused_from or [],
    )


def _fmr(op_type: str, inputs: list[TensorMeta], outputs: list[TensorMeta],
         attrs: dict | None = None, fused_from: list[str] | None = None) -> tuple[float, float, float]:
    """Run _fmr and return (flops, read_bytes, write_bytes)."""
    sim = RooflineSimulator()
    node = _node(op_type, inputs, outputs, attrs, fused_from)
    return sim._fmr(node)


def _approx(a: float, b: float, rel: float = 1e-9) -> bool:
    if a == 0 and b == 0:
        return True
    if a == 0 or b == 0:
        return abs(a - b) < 1e-6
    return abs(a - b) / max(abs(a), abs(b)) < rel


# ─────────────────────────────────────────────────────────────────────────────
# 1. Matrix Multiply Family
# ─────────────────────────────────────────────────────────────────────────────

class TestMatMul:
    """aten.mm.default: A=(M,K) @ B=(K,N) → (M,N)
    FLOPs = 2·M·K·N   R=(M·K+K·N)·b   W=M·N·b
    """

    def test_mm_basic(self):
        M, K, N = 64, 128, 256
        b = 2  # BF16
        flops, r, w = _fmr(
            "aten.mm.default",
            [_tm("a", (M, K)), _tm("b", (K, N))],
            [_tm("c", (M, N))],
        )
        assert flops == 2.0 * M * K * N
        assert r == (M * K + K * N) * b
        assert w == M * N * b

    def test_mm_fp8(self):
        M, K, N = 32, 64, 128
        b = 1  # FP8
        flops, r, w = _fmr(
            "aten.mm.default",
            [_tm("a", (M, K), DType.FP8_E4M3), _tm("b", (K, N), DType.FP8_E4M3)],
            [_tm("c", (M, N), DType.FP8_E4M3)],
        )
        assert flops == 2.0 * M * K * N
        assert r == (M * K + K * N) * b
        assert w == M * N * b

    def test_mm_insufficient_inputs(self):
        """Fallback when < 2 inputs."""
        flops, r, w = _fmr("aten.mm.default", [_tm("a", (4, 4))], [])
        assert flops > 0  # fallback path


class TestAddmm:
    """aten.addmm.default: bias + mat1 @ mat2
    FLOPs = 2·M·K·N + M·N   R=(M·K+K·N+|bias|)·b   W=M·N·b
    """

    def test_addmm_basic(self):
        M, K, N = 32, 64, 128
        b = 2
        flops, r, w = _fmr(
            "aten.addmm.default",
            [_tm("bias", (M, N)), _tm("m1", (M, K)), _tm("m2", (K, N))],
            [_tm("out", (M, N))],
        )
        assert flops == 2.0 * M * K * N + M * N
        assert r == (M * K + K * N + M * N) * b
        assert w == M * N * b


class TestBmm:
    """aten.bmm.default: (B,M,K) @ (B,K,N) → (B,M,N)
    FLOPs = 2·B·M·K·N   R=(B·M·K+B·K·N)·b   W=B·M·N·b
    """

    def test_bmm_basic(self):
        B, M, K, N = 8, 32, 64, 128
        b = 2
        flops, r, w = _fmr(
            "aten.bmm.default",
            [_tm("a", (B, M, K)), _tm("b", (B, K, N))],
            [_tm("c", (B, M, N))],
        )
        assert flops == 2.0 * B * M * K * N
        assert r == (B * M * K + B * K * N) * b
        assert w == B * M * N * b


class TestLinear:
    """aten.linear.default: input=(*,I), weight=(O,I), optional bias=(O,)
    FLOPs = 2·batch·I·O [+ batch·O if bias]
    """

    def test_linear_no_bias(self):
        batch, I, O = 32, 256, 512
        b = 2
        flops, r, w = _fmr(
            "aten.linear.default",
            [_tm("inp", (batch, I)), _tm("w", (O, I))],
            [_tm("out", (batch, O))],
        )
        assert flops == 2.0 * batch * I * O
        assert r == (batch * I + O * I) * b
        assert w == batch * O * b

    def test_linear_with_bias(self):
        batch, I, O = 16, 128, 256
        b = 2
        flops, r, w = _fmr(
            "aten.linear.default",
            [_tm("inp", (batch, I)), _tm("w", (O, I)), _tm("bias", (O,))],
            [_tm("out", (batch, O))],
        )
        assert flops == 2.0 * batch * I * O + batch * O
        assert r == (batch * I + O * I + O) * b
        assert w == batch * O * b

    def test_linear_3d_input(self):
        """batch = product of all dims except last."""
        B, S, I, O = 4, 128, 256, 512
        batch = B * S
        b = 2
        flops, r, w = _fmr(
            "aten.linear.default",
            [_tm("inp", (B, S, I)), _tm("w", (O, I))],
            [_tm("out", (B, S, O))],
        )
        assert flops == 2.0 * batch * I * O


# ─────────────────────────────────────────────────────────────────────────────
# 2. Convolution
# ─────────────────────────────────────────────────────────────────────────────

class TestConvolution:
    """2D Conv: FLOPs = 2·N·Cout·Hout·Wout·Cin·Kh·Kw"""

    def test_conv2d_basic(self):
        N, Cin, H, W = 2, 64, 32, 32
        Cout, Kh, Kw = 128, 3, 3
        Hout, Wout = H - Kh + 1, W - Kw + 1  # stride=1, padding=0
        b = 2
        flops, r, w = _fmr(
            "aten.convolution.default",
            [_tm("inp", (N, Cin, H, W)), _tm("w", (Cout, Cin, Kh, Kw))],
            [_tm("out", (N, Cout, Hout, Wout))],
        )
        expected_flops = 2.0 * N * Cout * Hout * Wout * Cin * Kh * Kw
        assert flops == expected_flops
        assert w == N * Cout * Hout * Wout * b

    def test_conv2d_with_bias(self):
        N, Cin, H, W = 1, 32, 16, 16
        Cout, Kh, Kw = 64, 3, 3
        Hout, Wout = H - Kh + 1, W - Kw + 1
        flops, r, w = _fmr(
            "aten.convolution.default",
            [_tm("inp", (N, Cin, H, W)), _tm("w", (Cout, Cin, Kh, Kw)), _tm("bias", (Cout,))],
            [_tm("out", (N, Cout, Hout, Wout))],
        )
        # bias add: N * Cout * Hout * Wout extra FLOPs
        expected_flops = 2.0 * N * Cout * Hout * Wout * Cin * Kh * Kw + N * Cout * Hout * Wout
        assert flops == expected_flops


# ─────────────────────────────────────────────────────────────────────────────
# 3. Attention
# ─────────────────────────────────────────────────────────────────────────────

class TestAttention:
    """sdpa: FLOPs = 4·N·H·Sq·Sk·D + 4·N·H·Sq·Sk"""

    def test_sdpa_basic(self):
        N, H, Sq, Sk, D = 2, 8, 128, 128, 64
        b = 2
        flops, r, w = _fmr(
            "aten._scaled_dot_product_flash_attention.default",
            [_tm("q", (N, H, Sq, D)), _tm("k", (N, H, Sk, D)), _tm("v", (N, H, Sk, D))],
            [_tm("out", (N, H, Sq, D))],
        )
        expected = 4.0 * N * H * Sq * Sk * D + 4.0 * N * H * Sq * Sk
        assert flops == expected
        assert r == (N * H * Sq * D + N * H * Sk * D + N * H * Sk * D) * b
        assert w == N * H * Sq * D * b

    def test_sdpa_kv_different_length(self):
        N, H, Sq, Sk, D = 1, 4, 64, 256, 32
        flops, r, w = _fmr(
            "aten.scaled_dot_product_attention.default",
            [_tm("q", (N, H, Sq, D)), _tm("k", (N, H, Sk, D)), _tm("v", (N, H, Sk, D))],
            [_tm("out", (N, H, Sq, D))],
        )
        expected = 4.0 * N * H * Sq * Sk * D + 4.0 * N * H * Sq * Sk
        assert flops == expected


class TestSparseAttention:
    """SparseFlashAttention: FLOPs = base * sparsity_ratio"""

    def test_sparse_attn_with_ratio(self):
        N, H, Sq, Sk, D = 1, 8, 512, 512, 64
        b = 2
        flops, r, w = _fmr(
            "SparseFlashAttention",
            [_tm("q", (N, H, Sq, D)), _tm("k", (N, H, Sk, D)), _tm("v", (N, H, Sk, D))],
            [_tm("out", (N, H, Sq, D))],
        )
        # No sparsity_ratio annotation → defaults to 1.0
        base = 4.0 * N * H * Sq * Sk * D + 4.0 * N * H * Sq * Sk
        assert flops == base


class TestPagedAttention:
    """PagedAttention: same as FA + block_tables indexing"""

    def test_paged_attn_decode(self):
        N, H, D = 32, 8, 64
        Sq = 1  # decode
        Sk = 4  # kv_cache_len default = Sq*4
        flops, r, w = _fmr(
            "PageAttentionFP16",
            [_tm("q", (N, H, Sq, D))],
            [_tm("out", (N, H, Sq, D))],
        )
        expected = 4.0 * N * H * Sq * Sk * D + 4.0 * N * H * Sq * Sk
        assert flops == expected


# ─────────────────────────────────────────────────────────────────────────────
# 4. Normalization
# ─────────────────────────────────────────────────────────────────────────────

class TestLayerNorm:
    """layer_norm: FLOPs = 7·N"""

    def test_layer_norm(self):
        N = 32 * 128 * 768
        b = 2
        flops, r, w = _fmr(
            "aten.layer_norm.default",
            [_tm("inp", (32, 128, 768))],
            [_tm("out", (32, 128, 768))],
        )
        assert flops == 7.0 * N
        weight_size = 768
        assert r == (N + 2 * weight_size) * b
        assert w == N * b


class TestRmsNorm:
    """rms_norm: FLOPs = 4·N"""

    def test_rms_norm(self):
        N = 16 * 256 * 512
        b = 2
        flops, r, w = _fmr(
            "rms_norm",
            [_tm("inp", (16, 256, 512))],
            [_tm("out", (16, 256, 512))],
        )
        assert flops == 4.0 * N
        weight_size = 512
        assert r == (N + weight_size) * b
        assert w == N * b

    def test_gemma_rms_norm(self):
        N = 8 * 128 * 256
        flops, r, w = _fmr(
            "GemmaRMSNorm",
            [_tm("inp", (8, 128, 256))],
            [_tm("out", (8, 128, 256))],
        )
        assert flops == 4.0 * N


class TestRmsNormGated:
    """RMSNormGated: FLOPs = 9·N (with gate tensor)"""

    def test_rms_norm_gated_with_gate_tensor(self):
        N = 32 * 128 * 512
        b = 2
        flops, r, w = _fmr(
            "RMSNormGated",
            [_tm("inp", (32, 128, 512)), _tm("w", (512,)), _tm("gate", (32, 128, 512))],
            [_tm("out", (32, 128, 512))],
        )
        assert flops == 9.0 * N


class TestAddRmsNorm:
    """add_rms_norm: FLOPs = 6·N (norm 5N + residual add 1N)"""

    def test_add_rms_norm(self):
        N = 16 * 64 * 256
        b = 2
        flops, r, w = _fmr(
            "add_rms_norm",
            [_tm("inp", (16, 64, 256))],
            [_tm("out", (16, 64, 256))],
        )
        assert flops == 6.0 * N


# ─────────────────────────────────────────────────────────────────────────────
# 5. Softmax
# ─────────────────────────────────────────────────────────────────────────────

class TestSoftmax:
    """softmax: FLOPs = 4·N"""

    def test_softmax(self):
        N = 32 * 8 * 128 * 128
        b = 2
        flops, r, w = _fmr(
            "aten._softmax.default",
            [_tm("inp", (32, 8, 128, 128))],
            [_tm("out", (32, 8, 128, 128))],
        )
        assert flops == 4.0 * N
        assert r == N * b
        assert w == N * b


# ─────────────────────────────────────────────────────────────────────────────
# 6. Elementwise
# ─────────────────────────────────────────────────────────────────────────────

class TestElementwise1:
    """1 op/elem: add, sub, mul, div, relu, exp, etc."""

    def test_add(self):
        N = 32 * 128 * 256
        flops, r, w = _fmr(
            "aten.add.Tensor",
            [_tm("a", (32, 128, 256)), _tm("b", (32, 128, 256))],
            [_tm("out", (32, 128, 256))],
        )
        assert flops == 1.0 * N

    def test_mul(self):
        N = 64 * 64
        flops, r, w = _fmr(
            "aten.mul.Tensor",
            [_tm("a", (64, 64)), _tm("b", (64, 64))],
            [_tm("out", (64, 64))],
        )
        assert flops == 1.0 * N

    def test_relu(self):
        N = 128 * 256
        flops, r, w = _fmr(
            "aten.relu.default",
            [_tm("inp", (128, 256))],
            [_tm("out", (128, 256))],
        )
        assert flops == 1.0 * N


class TestElementwise2:
    """2 ops/elem: reciprocal, clamp"""

    def test_reciprocal(self):
        N = 64 * 128
        flops, r, w = _fmr(
            "aten.reciprocal.default",
            [_tm("inp", (64, 128))],
            [_tm("out", (64, 128))],
        )
        assert flops == 2.0 * N

    def test_clamp(self):
        N = 32 * 64
        flops, r, w = _fmr(
            "aten.clamp.default",
            [_tm("inp", (32, 64))],
            [_tm("out", (32, 64))],
        )
        assert flops == 2.0 * N


class TestActivation:
    """4 ops/elem: silu, gelu, sigmoid"""

    def test_silu(self):
        N = 16 * 256
        flops, r, w = _fmr(
            "aten.silu.default",
            [_tm("inp", (16, 256))],
            [_tm("out", (16, 256))],
        )
        assert flops == 4.0 * N

    def test_gelu(self):
        N = 32 * 128
        flops, r, w = _fmr(
            "aten.gelu.default",
            [_tm("inp", (32, 128))],
            [_tm("out", (32, 128))],
        )
        assert flops == 4.0 * N

    def test_sigmoid(self):
        N = 64 * 64
        flops, r, w = _fmr(
            "aten.sigmoid.default",
            [_tm("inp", (64, 64))],
            [_tm("out", (64, 64))],
        )
        assert flops == 4.0 * N


class TestTranscendental:
    """10 ops/elem: sin, cos, atan2"""

    def test_sin(self):
        N = 128 * 64
        flops, r, w = _fmr(
            "aten.sin.default",
            [_tm("inp", (128, 64))],
            [_tm("out", (128, 64))],
        )
        assert flops == 10.0 * N

    def test_cos(self):
        N = 256 * 32
        flops, r, w = _fmr(
            "aten.cos.default",
            [_tm("inp", (256, 32))],
            [_tm("out", (256, 32))],
        )
        assert flops == 10.0 * N


class TestReduction:
    """mean/sum: 1 op/elem (output), var: 3 ops/elem (output)"""

    def test_mean(self):
        N_out = 32  # reduction output size
        flops, r, w = _fmr(
            "aten.mean.dim",
            [_tm("inp", (32, 128))],
            [_tm("out", (32,))],
        )
        # elementwise uses output numel for flops
        assert flops == 1.0 * N_out

    def test_var(self):
        N_out = 16
        flops, r, w = _fmr(
            "aten.var.correction",
            [_tm("inp", (16, 64))],
            [_tm("out", (16,))],
        )
        assert flops == 3.0 * N_out


# ─────────────────────────────────────────────────────────────────────────────
# 7. Embedding / Gather
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbedding:
    """embedding: FLOPs = 0 (table lookup)"""

    def test_embedding(self):
        vocab, dim = 10000, 768
        seq = 128
        b = 2
        flops, r, w = _fmr(
            "aten.embedding.default",
            [_tm("weight", (vocab, dim)), _tm("indices", (seq,))],
            [_tm("out", (seq, dim))],
        )
        assert flops == 0.0
        assert w == seq * dim * b


class TestGather:
    """gather/scatter/index: FLOPs = 0"""

    def test_index(self):
        N = 32 * 128
        b = 2
        flops, r, w = _fmr(
            "aten.index.Tensor",
            [_tm("inp", (32, 128)), _tm("idx", (16,))],
            [_tm("out", (16, 128))],
        )
        assert flops == 0.0

    def test_scatter(self):
        flops, r, w = _fmr(
            "aten.scatter.src",
            [_tm("inp", (64, 64)), _tm("idx", (32,)), _tm("src", (32, 64))],
            [_tm("out", (64, 64))],
        )
        assert flops == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Sort / Topk
# ─────────────────────────────────────────────────────────────────────────────

class TestSort:
    """sort: FLOPs = 2·N·log₂(N)"""

    def test_sort(self):
        N = 1024
        log_n = math.log2(N)
        flops, r, w = _fmr(
            "aten.sort.default",
            [_tm("inp", (N,))],
            [_tm("values", (N,)), _tm("indices", (N,))],
        )
        assert flops == 2.0 * N * log_n

    def test_topk(self):
        N = 2048
        log_n = math.log2(N)
        flops, r, w = _fmr(
            "aten.topk.default",
            [_tm("inp", (N,))],
            [_tm("values", (10,)), _tm("indices", (10,))],
        )
        assert flops == 2.0 * N * log_n


# ─────────────────────────────────────────────────────────────────────────────
# 9. Dtype Cast / Memory Ops
# ─────────────────────────────────────────────────────────────────────────────

class TestDtypeCast:
    """_to_copy: FLOPs = 0 (pure data movement)"""

    def test_cast_bf16_to_fp32(self):
        N = 64 * 128
        flops, r, w = _fmr(
            "aten._to_copy.default",
            [_tm("inp", (64, 128), DType.BF16)],
            [_tm("out", (64, 128), DType.FP32)],
        )
        assert flops == 0.0
        assert r == N * 2  # BF16 = 2 bytes
        assert w == N * 4  # FP32 = 4 bytes


class TestWriteOnly:
    """new_empty/fill_/zero_: FLOPs = 0, R = 0"""

    def test_new_empty(self):
        N = 32 * 64
        b = 2
        flops, r, w = _fmr(
            "aten.new_empty.default",
            [],
            [_tm("out", (32, 64))],
        )
        assert flops == 0.0
        assert r == 0.0
        assert w == N * b

    def test_zero(self):
        N = 128 * 256
        b = 2
        flops, r, w = _fmr(
            "aten.zero_.default",
            [_tm("inp", (128, 256))],
            [_tm("out", (128, 256))],
        )
        assert flops == 0.0
        assert r == 0.0
        assert w == N * b


# ─────────────────────────────────────────────────────────────────────────────
# 10. Fused Semantic Labels
# ─────────────────────────────────────────────────────────────────────────────

class TestFusedLinear:
    """Linear (fused): 2·batch·I·N"""

    def test_linear_proj(self):
        batch, I, N = 32, 256, 512
        b = 2
        flops, r, w = _fmr(
            "Linear",
            [_tm("inp", (batch, I)), _tm("w", (I, N))],
            [_tm("out", (batch, N))],
        )
        assert flops == 2.0 * batch * I * N


class TestSwiGLU:
    """swiglu: 4·batch·H·I + 2·batch·I·H + 5·batch·I"""

    def test_swiglu(self):
        batch, H, I = 32, 768, 1536
        b = 2
        flops, r, w = _fmr(
            "swiglu",
            [
                _tm("hidden", (batch, H)),
                _tm("gate_w", (I, H)),
                _tm("up_w", (I, H)),
                _tm("down_w", (H, I)),
            ],
            [_tm("out", (batch, H))],
        )
        expected = 4.0 * batch * H * I + 2.0 * batch * I * H + 5.0 * batch * I
        assert flops == expected


class TestFusedMlp:
    """gated_mlp: Σ 2·batch·H·Oᵢ + activation"""

    def test_gated_mlp_two_weights(self):
        batch, H, O = 16, 512, 1024
        b = 2
        flops, r, w = _fmr(
            "gated_mlp",
            [
                _tm("hidden", (batch, H)),
                _tm("w1", (O, H)),
                _tm("w2", (H, O)),
            ],
            [_tm("out", (batch, H))],
        )
        # Two matmuls: 2*batch*H*O each
        mm_flops = 2.0 * batch * H * O + 2.0 * batch * H * O
        assert _approx(flops, mm_flops, rel=0.1)  # allow some tolerance for activation cost


class TestMoeGate:
    """moe_gate: linear FLOPs + 5·N_out (softmax)"""

    def test_moe_gate(self):
        batch, H, num_experts = 32, 768, 64
        b = 2
        flops, r, w = _fmr(
            "moe_gate",
            [_tm("hidden", (batch, H)), _tm("gate_w", (num_experts, H))],
            [_tm("scores", (batch, num_experts))],
        )
        linear_flops = 2.0 * batch * H * num_experts
        softmax_flops = 4.0 * batch * num_experts
        assert flops == linear_flops + softmax_flops

    def test_moe_gate_topk(self):
        batch, H, num_experts = 16, 512, 32
        flops, r, w = _fmr(
            "moe_gate_topk",
            [_tm("hidden", (batch, H)), _tm("gate_w", (num_experts, H))],
            [_tm("scores", (batch, num_experts))],
        )
        linear_flops = 2.0 * batch * H * num_experts
        softmax_flops = 4.0 * batch * num_experts
        topk_flops = batch * num_experts
        assert flops == linear_flops + softmax_flops + topk_flops


class TestRope:
    """rope: 2 ops/elem"""

    def test_rope(self):
        N = 32 * 128 * 64
        flops, r, w = _fmr(
            "rope",
            [_tm("inp", (32, 128, 64))],
            [_tm("out", (32, 128, 64))],
        )
        assert flops == 2.0 * N


# ─────────────────────────────────────────────────────────────────────────────
# 11. Shape / Transparent Ops
# ─────────────────────────────────────────────────────────────────────────────

class TestShapeOps:
    """view/reshape/permute/etc: FLOPs ≈ 0"""

    @pytest.mark.parametrize("op_type", [
        "aten.view.default",
        "aten._unsafe_view.default",
        "aten.reshape.default",
        "aten.expand.default",
        "aten.permute.default",
        "aten.transpose.int",
        "aten.contiguous.memory_format",
        "aten.flatten.using_ints",
        "aten.squeeze.dim",
        "aten.unsqueeze.default",
        "aten.select.int",
        "aten.slice.Tensor",
        "aten.clone.default",
        "aten.t.default",
        "aten.chunk.default",
        "aten.split.Tensor",
        "aten.unbind.int",
        "aten.detach.default",
        "aten.alias.default",
        "aten.cat.default",
        "aten.stack.default",
    ])
    def test_shape_op_zero_flops(self, op_type):
        flops, r, w = _fmr(
            op_type,
            [_tm("inp", (32, 64, 128))],
            [_tm("out", (32, 64, 128))],
        )
        assert flops == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 12. Fused Node Decomposition
# ─────────────────────────────────────────────────────────────────────────────

class TestFusedDecompose:
    """Fused nodes: sum sub-op FLOPs from fused_from list."""

    def test_fused_mm_plus_view(self):
        """mm + view should only count mm FLOPs (view is skipped)."""
        M, K, N = 64, 128, 256
        b = 2
        flops, r, w = _fmr(
            "fused.mm_view",
            [_tm("a", (M, K)), _tm("b", (K, N))],
            [_tm("out", (M, N))],
            fused_from=["aten.mm.default", "aten.view.default"],
        )
        # Only mm contributes; view is skipped
        assert flops == 2.0 * M * K * N

    def test_fused_multiple_matmuls(self):
        """Multiple mm sub-ops should sum."""
        M, K, N = 32, 64, 128
        flops, r, w = _fmr(
            "fused.two_mm",
            [_tm("a", (M, K)), _tm("b", (K, N))],
            [_tm("out", (M, N))],
            fused_from=["aten.mm.default", "aten.mm.default"],
        )
        # Two mm ops
        assert flops == 2 * (2.0 * M * K * N)


# ─────────────────────────────────────────────────────────────────────────────
# 13. Fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestFallback:
    """Unknown ops: FLOPs = N_out (1 flop / output element)"""

    def test_unknown_op(self):
        N = 64 * 128
        flops, r, w = _fmr(
            "some.unknown.op",
            [_tm("inp", (64, 128))],
            [_tm("out", (64, 128))],
        )
        assert flops == float(N)

    def test_unknown_op_empty_output(self):
        flops, r, w = _fmr(
            "some.unknown.op",
            [_tm("inp", (4, 4))],
            [],
        )
        assert flops == 1.0  # default n_out = 1


# ─────────────────────────────────────────────────────────────────────────────
# 14. Project-Specific Ops
# ─────────────────────────────────────────────────────────────────────────────

class TestProjectSpecific:
    """RepoKernel, MoEInfer, GroupedMm, etc."""

    def test_grouped_mm(self):
        G, M, K, N = 4, 32, 64, 128
        b = 2
        flops, r, w = _fmr(
            "GroupedMm",
            [_tm("a", (G, M, K)), _tm("b", (G, K, N))],
            [_tm("out", (G, M, N))],
        )
        assert flops == 2.0 * G * M * K * N

    def test_concat_fp16(self):
        N = 64 * 128
        b = 2
        flops, r, w = _fmr(
            "ConcatFP16",
            [_tm("a", (N // 2,)), _tm("b", (N // 2,))],
            [_tm("out", (N,))],
        )
        assert flops == 0.0

    def test_moe_dispatch(self):
        flops, r, w = _fmr(
            "moe_dispatch",
            [_tm("inp", (32, 768)), _tm("idx", (32,))],
            [_tm("out", (16, 768))],
        )
        assert flops == 0.0
