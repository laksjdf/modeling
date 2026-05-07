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

    # attention
    attn_compression_ratio: float = 1.0
    
    # Compressed-CP (DeepSeek-V4 style): CSA vs HCA layer distribution
    num_csa_layers: int = 0  # number of CSA layers (compression_ratio=4)
    num_hca_layers: int = 0  # number of HCA layers (compression_ratio=128)
    num_swa_only_layers: int = 0  # SWA-only layers (no CP communication)

    # MoE (ignored when no MOE layers)
    num_experts: int = 0
    moe_ffn: int = 0
    top_k: int = 0
    capacity_factor: float = 1.0
    expert_imbalance: float = 0.0
    n_group: int = 0          # expert routing groups (0 = no group routing)
    n_shared_experts: int = 1  # shared experts per MoE layer (not sharded by EP)

    # MTP
    mtp_depth: int = 0

    # Hyper-Connections (DeepSeek-V4): hc_mult > 1 keeps `hc_mult` parallel
    # residual streams, mixed via hc_pre / hc_post around attn and ffn.
    # hc_mult=1 disables HC entirely (legacy single-residual path).
    hc_mult: int = 1
    hc_sinkhorn_iters: int = 20

    # dtypes
    param_dtype: Dtype = Dtype.BF16
    grad_dtype: Dtype = Dtype.FP32
    master_dtype: Dtype = Dtype.FP32
    act_dtype: Dtype = Dtype.BF16

    # Muon optimizer specific fields (optional, from model YAML)
    muon_ns_steps: int | None = None
    model_type: str = "default"

    def __post_init__(self) -> None:
        self.attn_compression_ratio = float(self.attn_compression_ratio)
        if not (0.0 < self.attn_compression_ratio <= 1.0):
            raise ValueError(
                "attn_compression_ratio must be in (0, 1], "
                f"got {self.attn_compression_ratio}"
            )
        
        # Validate compressed-CP layer distribution
        total_compressed_layers = self.num_csa_layers + self.num_hca_layers + self.num_swa_only_layers
        if total_compressed_layers > 0:
            if total_compressed_layers != len(self.layers):
                raise ValueError(
                    f"Compressed-CP layer distribution ({total_compressed_layers}) "
                    f"must match total layers ({len(self.layers)})"
                )
    
    def get_layer_cp_type(self, layer_id: int) -> str:
        """Determine CP type (CSA/HCA/SWA) for a given layer.
        
        Returns: 'csa', 'hca', or 'swa'
        """
        if self.num_csa_layers == 0 and self.num_hca_layers == 0:
            return 'none'
        
        # SWA-only layers come first
        if layer_id < self.num_swa_only_layers:
            return 'swa'
        
        # CSA layers come after SWA
        csa_start = self.num_swa_only_layers
        csa_end = csa_start + self.num_csa_layers
        if csa_start <= layer_id < csa_end:
            return 'csa'
        
        # HCA layers come after CSA
        hca_start = csa_end
        hca_end = hca_start + self.num_hca_layers
        if hca_start <= layer_id < hca_end:
            return 'hca'
        
        return 'none'

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

    def effective_params_for_flops(self) -> int:
        """Effective parameters for FLOPs calculation.

        For MoE layers, only top_k/num_experts fraction of expert params
        are active per token. This is the correct parameter count for
        the 6P FLOPs rule.

        Dense/MTP layers: 100% of params active
        MoE layers: shared + router + (top_k/num_experts) * expert_ffn params
        """
        # Embedding: vocab * hidden (tied with lm_head by default, count once)
        embed = self.vocab * self.hidden

        # Final LayerNorm
        final_ln = self.hidden

        n_dense = sum(1 for lk in self.layers if lk == LayerKind.DENSE)
        n_moe = sum(1 for lk in self.layers if lk == LayerKind.MOE)
        n_mtp = sum(1 for lk in self.layers if lk == LayerKind.MTP)

        # Dense and MTP: all params active
        dense_params = n_dense * self.params_per_dense_layer()
        mtp_params = n_mtp * self.params_per_mtp_layer()

        # MoE: only active fraction of expert params
        # Active experts = (top_k / num_experts) fraction of routed experts
        if n_moe > 0 and self.num_experts > 0 and self.top_k > 0:
            h = self.hidden
            h_attn = self.num_heads * self.head_dim
            h_kv = self.num_kv_heads * self.head_dim

            # Attention params (same as dense, 100% active)
            attn_params = h * (h_attn + 2 * h_kv) + h_attn * h

            # Router: 100% active (all tokens go through router)
            router_params = h * self.num_experts

            # Shared expert FFN: 100% active
            shared_ffn_params = h * self.moe_ffn + h * self.moe_ffn + self.moe_ffn * h

            # Routed experts: only top_k/num_experts fraction active
            expert_ffn_per_expert = h * self.moe_ffn + h * self.moe_ffn + self.moe_ffn * h
            active_expert_fraction = self.top_k / self.num_experts
            expert_ffn_params = expert_ffn_per_expert * self.num_experts * active_expert_fraction

            ln_params = 2 * h

            moe_layer_params = attn_params + router_params + shared_ffn_params + expert_ffn_params + ln_params
        else:
            # No MoE or zero experts: fall back to full params
            moe_layer_params = n_moe * self.params_per_moe_layer()

        layer_params = dense_params + moe_layer_params + mtp_params

        # Count lm_head separately unless tied
        lm_head = self.vocab * self.hidden

        return embed + layer_params + final_ln + lm_head
