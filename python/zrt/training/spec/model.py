from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from zrt.training.spec.dtype import Dtype


class LayerKind(Enum):
    DENSE = "dense"
    MOE = "moe"
    MTP = "mtp"


@dataclass
class ModelSpec:
    # geometry
    hidden: int
    ffn: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab: int
    seq_len: int

    # layer composition — order matters for PP balance
    layers: list[LayerKind]

    # MoE (ignored when no MOE layers)
    num_experts: int = 0
    moe_ffn: int = 0
    top_k: int = 0
    capacity_factor: float = 1.0
    expert_imbalance: float = 0.0

    # MTP
    mtp_depth: int = 0

    # dtypes
    param_dtype: Dtype = Dtype.BF16
    grad_dtype: Dtype = Dtype.FP32
    master_dtype: Dtype = Dtype.FP32
    act_dtype: Dtype = Dtype.BF16

    @property
    def head_dim_total(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    def params_per_dense_layer(self) -> int:
        """Params in one dense transformer block (pre-norm style)."""
        h = self.hidden
        h_attn = self.num_heads * self.head_dim
        h_kv = self.num_kv_heads * self.head_dim
        ffn = self.ffn

        # QKV projections: h -> h_attn + 2 * h_kv
        qkv = h * (h_attn + 2 * h_kv)
        # O projection: h_attn -> h
        o_proj = h_attn * h
        # FFN: SwiGLU style (up, gate, down) — ffn = 2/3 * intermediate * 2 + h
        # Standard: up(h, ffn), gate(h, ffn), down(ffn, h)
        ffn_params = h * ffn + h * ffn + ffn * h  # up + gate + down
        # 2x LayerNorm (no params for RMSNorm bias, but has weight)
        ln_params = 2 * h

        return qkv + o_proj + ffn_params + ln_params

    def params_per_moe_layer(self) -> int:
        """Params in one MoE transformer block."""
        h = self.hidden
        h_attn = self.num_heads * self.head_dim
        h_kv = self.num_kv_heads * self.head_dim

        # Attention (same as dense)
        qkv = h * (h_attn + 2 * h_kv)
        o_proj = h_attn * h

        # Router: h -> num_experts
        router = h * self.num_experts

        # Shared expert FFN (same structure as dense FFN but with moe_ffn)
        shared_ffn = h * self.moe_ffn + h * self.moe_ffn + self.moe_ffn * h

        # Routed experts: each expert has up+gate+down with moe_ffn
        expert_ffn = (h * self.moe_ffn + h * self.moe_ffn + self.moe_ffn * h) * self.num_experts

        ln_params = 2 * h

        return qkv + o_proj + router + shared_ffn + expert_ffn + ln_params

    def params_per_mtp_layer(self) -> int:
        """Params in one MTP layer (embedding projection + dense block)."""
        # MTP typically has an embedding projection + a dense transformer block
        # Simplified: same as dense + an extra embedding projection
        h = self.hidden
        return self.params_per_dense_layer() + h * h

    def total_params(self) -> int:
        """Derived total parameter count."""
        # Embedding: vocab * hidden (tied with lm_head by default, count once)
        embed = self.vocab * self.hidden

        # Final LayerNorm
        final_ln = self.hidden

        n_dense = sum(1 for lk in self.layers if lk == LayerKind.DENSE)
        n_moe = sum(1 for lk in self.layers if lk == LayerKind.MOE)
        n_mtp = sum(1 for lk in self.layers if lk == LayerKind.MTP)

        layer_params = (
            n_dense * self.params_per_dense_layer()
            + n_moe * self.params_per_moe_layer()
            + n_mtp * self.params_per_mtp_layer()
        )

        # Count lm_head separately unless tied
        lm_head = self.vocab * self.hidden

        return embed + layer_params + final_ln + lm_head
