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

    # MLA fields (DeepSeek-V3 / V3.2)
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0

    # Lightning Indexer (V3.2 / V4 CSA layers)
    index_n_heads: int = 0
    index_head_dim: int = 0
    index_topk: int = 0

    # V4 attention fields
    o_lora_rank: int = 0        # grouped output low-rank dim (0 = standard o_proj)
    o_groups: int = 0           # number of output projection groups
    compress_ratios: list[int] = field(default_factory=list)
    swa_window: int = 0

    # V4 MoE fields
    n_hash_routed_layers: int = 0
    scoring_func: str = "sigmoid"  # sigmoid / sqrt_softplus
    routed_expert_dtype: str = "bf16"  # bf16 / fp4
    swiglu_clamp: float = 0.0

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

    # === Per-component compute dtype (DeepSeek-V4 mixed quant) ===
    attn_compute_dtype: Dtype = Dtype.BF16
    shared_expert_compute_dtype: Dtype = Dtype.BF16
    routed_expert_compute_dtype: Dtype = Dtype.BF16

    # === Per-component weight dtype ===
    # Default is ``None`` (sentinel) which resolves to ``Dtype.BF16`` in
    # ``__post_init__`` — unless the legacy ``routed_expert_dtype='fp4'``
    # string is supplied, in which case it syncs to ``Dtype.FP4``. An
    # explicit user value (including ``Dtype.BF16``) always wins.
    routed_expert_weight_dtype: Dtype | None = None  # V4 default: FP4
    # v2: weight dtype for attention / shared expert projections. ``None``
    # resolves to ``param_dtype`` in ``__post_init__`` so legacy configs
    # behave identically to v1.
    attn_weight_dtype: Dtype | None = None
    shared_expert_weight_dtype: Dtype | None = None

    # === Per-region activation dtype (None → fallback) ===
    attn_act_dtype: Dtype | None = None
    moe_act_dtype: Dtype | None = None

    # === Per-component grad dtype ===
    routed_expert_grad_dtype: Dtype = Dtype.FP32
    # v2: grad dtype for attention / shared expert. Defaults to ``grad_dtype``
    # via ``__post_init__`` when left as ``None`` so existing anchors keep
    # their numbers.
    attn_grad_dtype: Dtype | None = None
    shared_expert_grad_dtype: Dtype | None = None

    # normalization kind: "rmsnorm" (DeepSeek) or "layernorm" (LLaMA variants)
    norm_kind: str = "rmsnorm"

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

        # Validate compressed-CP layer distribution (count-based only)
        total_compressed_layers = self.num_csa_layers + self.num_hca_layers + self.num_swa_only_layers
        if total_compressed_layers > 0:
            if total_compressed_layers != len(self.layers):
                raise ValueError(
                    f"Compressed-CP layer distribution ({total_compressed_layers}) "
                    f"must match total layers ({len(self.layers)})"
                )

        # Back-compat: legacy ``routed_expert_dtype: str`` syncs into the
        # new ``routed_expert_weight_dtype`` only when the new field is at
        # its default (sentinel ``None``). An explicit user value wins.
        if self.routed_expert_weight_dtype is None:
            if isinstance(self.routed_expert_dtype, str):
                legacy = self.routed_expert_dtype.lower().strip()
                if legacy == "fp4":
                    self.routed_expert_weight_dtype = Dtype.FP4
                else:
                    self.routed_expert_weight_dtype = Dtype.BF16
            else:
                self.routed_expert_weight_dtype = Dtype.BF16

        # v2: per-component weight dtype defaults to param_dtype unless the
        # user set an explicit value. Mirrors the routed_expert behavior so
        # BF16 baselines keep identical numbers.
        if self.attn_weight_dtype is None:
            self.attn_weight_dtype = self.param_dtype
        if self.shared_expert_weight_dtype is None:
            self.shared_expert_weight_dtype = self.param_dtype

        # v2: per-component grad dtype defaults to global grad_dtype.
        if self.attn_grad_dtype is None:
            self.attn_grad_dtype = self.grad_dtype
        if self.shared_expert_grad_dtype is None:
            self.shared_expert_grad_dtype = self.grad_dtype

    @property
    def use_mla(self) -> bool:
        """True when using DeepSeek-V3 style MLA (kv_lora_rank > 0)."""
        return self.kv_lora_rank > 0

    @property
    def use_v4_attn(self) -> bool:
        """True when using V4-style attention (grouped o_proj + single KV head)."""
        return self.o_groups > 0

    def get_layer_cp_type(self, layer_id: int) -> str:
        """Determine CP type (CSA/HCA/SWA) for a given layer.

        When compress_ratios is provided, uses per-layer lookup.
        Otherwise falls back to count-based dispatch.
        """
        if self.compress_ratios:
            if layer_id >= len(self.compress_ratios):
                return 'none'
            ratio = self.compress_ratios[layer_id]
            if ratio == 0:
                return 'swa'
            elif ratio <= 4:
                return 'csa'
            else:
                return 'hca'

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

    def effective_attn_act_dtype(self) -> Dtype:
        """Attention region activation dtype; falls back to act_dtype."""
        return self.attn_act_dtype if self.attn_act_dtype is not None else self.act_dtype

    def effective_moe_act_dtype(self) -> Dtype:
        """MoE region activation dtype; falls back to routed_expert_compute_dtype.

        Rationale: when a user enables FP8 routed compute via quant preset,
        the matching forward activations are also FP8 unless explicitly
        overridden.
        """
        if self.moe_act_dtype is not None:
            return self.moe_act_dtype
        return self.routed_expert_compute_dtype

    def effective_residual_dtype(self) -> Dtype:
        """Residual stream dtype — PP P2P + cross-region cast targets.

        V4 keeps the residual stream in ``act_dtype`` (typically BF16) even
        when MoE/attention internals are FP8/FP4. Exposed as a helper so a
        future ``residual_dtype`` override field can be added without churn.
        """
        return self.act_dtype

    # ── Attention parameter helpers ──────────────────────────────────────

    def _attn_proj_params(self) -> int:
        """Projection weight params per layer (Q/KV/O, no norms)."""
        h = self.hidden

        if self.use_mla:
            # V3 MLA: q_a → q_b, kv_a → kv_b, o_proj
            h_q = self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
            h_kv_out = self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)
            return (
                h * self.q_lora_rank                                    # q_a_proj
                + self.q_lora_rank * h_q                                # q_b_proj
                + h * (self.kv_lora_rank + self.qk_rope_head_dim)      # kv_a_proj_with_mqa
                + self.kv_lora_rank * h_kv_out                          # kv_b_proj
                + self.num_heads * self.v_head_dim * h                  # o_proj
            )

        if self.use_v4_attn:
            # V4: low-rank Q + single-KV MQA + grouped output
            h_per_group = self.num_heads * self.head_dim // self.o_groups
            return (
                h * self.q_lora_rank                                    # wq_a
                + self.q_lora_rank * self.num_heads * self.head_dim     # wq_b
                + h * self.head_dim                                     # wkv (single head)
                + h_per_group * self.o_groups * self.o_lora_rank        # wo_a
                + self.o_groups * self.o_lora_rank * h                  # wo_b
            )

        # Standard MHA
        h_attn = self.num_heads * self.head_dim
        h_kv = self.num_kv_heads * self.head_dim
        return h * (h_attn + 2 * h_kv) + h_attn * h

    def _attn_inner_norm_params(self) -> int:
        """RMSNorm params inside the attention sub-block (not outer pre-norms)."""
        if self.use_mla:
            return self.q_lora_rank + self.kv_lora_rank
        if self.use_v4_attn:
            return self.q_lora_rank + self.head_dim
        return 0

    def _compressor_params(self, cp_type: str) -> int:
        """KV compressor params for CSA/HCA layers (V4 only)."""
        d = self.head_dim
        if cp_type == 'csa':
            coff = 2  # overlapping windows for m=4
            m = 4
        elif cp_type == 'hca':
            coff = 1  # non-overlapping for m=128
            m = 128
        else:
            return 0
        return (
            self.hidden * coff * d   # wkv
            + self.hidden * coff * d  # wgate
            + m * coff * d            # ape (absolute position embedding)
            + d                       # norm
        )

    def _indexer_params(self) -> int:
        """Indexer params (CSA layers only)."""
        if self.index_n_heads <= 0 or self.index_head_dim <= 0:
            return 0
        ih, id_ = self.index_n_heads, self.index_head_dim
        coff = 2  # indexer compressor also uses overlapping for m=4
        m = 4
        return (
            self.q_lora_rank * ih * id_   # wq_b
            + self.hidden * ih              # weights_proj
            + self.hidden * coff * id_      # indexer.compressor.wkv
            + self.hidden * coff * id_      # indexer.compressor.wgate
            + m * coff * id_                # indexer.compressor.ape
            + id_                           # indexer.compressor.norm
        )

    def _hc_params(self) -> int:
        """Hyper-Connection params per layer (hc_mult > 1 only)."""
        if self.hc_mult <= 1:
            return 0
        hc_dim = self.hc_mult * self.hidden
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        return 2 * (mix_hc * hc_dim + mix_hc + 3)

    # ── Per-layer param methods ──────────────────────────────────────────

    def params_per_dense_layer(self) -> int:
        """Params in one dense transformer block."""
        h = self.hidden
        ffn = self.ffn

        attn = self._attn_proj_params()
        inner_norms = self._attn_inner_norm_params()
        ffn_params = 3 * h * ffn  # SwiGLU: up + gate + down
        outer_norms = 2 * h       # pre-attn + pre-ffn RMSNorm
        hc = self._hc_params()

        return attn + inner_norms + ffn_params + outer_norms + hc

    def params_per_moe_layer(self, cp_type: str = 'none') -> int:
        """Params in one MoE transformer block.

        cp_type: 'csa', 'hca', 'swa', or 'none'.
        """
        h = self.hidden

        attn = self._attn_proj_params()
        inner_norms = self._attn_inner_norm_params()
        compressor = self._compressor_params(cp_type)
        indexer = self._indexer_params() if cp_type == 'csa' else 0
        router = h * self.num_experts + self.num_experts
        shared_ffn = 3 * h * self.moe_ffn
        expert_ffn = 3 * h * self.moe_ffn * self.num_experts
        outer_norms = 2 * h
        hc = self._hc_params()

        return (
            attn + inner_norms + compressor + indexer
            + router + shared_ffn + expert_ffn + outer_norms + hc
        )

    def params_per_mtp_layer(self) -> int:
        """Params in one MTP layer (embedding projections + block)."""
        h = self.hidden
        embed_proj = 2 * h * h   # e_proj + h_proj
        embed_norm = 2 * h       # e_norm + h_norm
        block = self.params_per_moe_layer() if any(
            lk == LayerKind.MOE for lk in self.layers
        ) else self.params_per_dense_layer()
        return embed_proj + embed_norm + block

    def total_params(self) -> int:
        """Derived total parameter count."""
        embed = self.vocab * self.hidden
        final_ln = self.hidden

        layer_params = 0
        for i, lk in enumerate(self.layers):
            if lk == LayerKind.DENSE:
                layer_params += self.params_per_dense_layer()
            elif lk == LayerKind.MOE:
                cp_type = self.get_layer_cp_type(i)
                layer_params += self.params_per_moe_layer(cp_type)
            elif lk == LayerKind.MTP:
                layer_params += self.params_per_mtp_layer()

        lm_head = self.vocab * self.hidden
        return embed + layer_params + final_ln + lm_head

    def effective_params_for_flops(self) -> int:
        """Effective parameters for FLOPs calculation.

        For MoE layers, only top_k/num_experts fraction of expert params
        are active per token.
        """
        embed = self.vocab * self.hidden
        final_ln = self.hidden
        h = self.hidden

        layer_params = 0
        for i, lk in enumerate(self.layers):
            if lk == LayerKind.DENSE:
                layer_params += self.params_per_dense_layer()
            elif lk == LayerKind.MOE:
                cp_type = self.get_layer_cp_type(i)

                attn = self._attn_proj_params()
                inner_norms = self._attn_inner_norm_params()
                compressor = self._compressor_params(cp_type)
                indexer = self._indexer_params() if cp_type == 'csa' else 0
                router = h * self.num_experts + self.num_experts
                shared_ffn = 3 * h * self.moe_ffn
                active_expert = 3 * h * self.moe_ffn * self.top_k
                outer_norms = 2 * h
                hc = self._hc_params()

                layer_params += (
                    attn + inner_norms + compressor + indexer
                    + router + shared_ffn + active_expert + outer_norms + hc
                )
            elif lk == LayerKind.MTP:
                # MTP layer: use active (not total) expert params
                mtp_h = self.hidden
                embed_proj = 2 * mtp_h * mtp_h
                embed_norm = 2 * mtp_h
                if any(lk2 == LayerKind.MOE for lk2 in self.layers):
                    mtp_cp_type = self.get_layer_cp_type(i)
                    mtp_attn = self._attn_proj_params()
                    mtp_inner_norms = self._attn_inner_norm_params()
                    mtp_compressor = self._compressor_params(mtp_cp_type)
                    mtp_indexer = self._indexer_params() if mtp_cp_type == 'csa' else 0
                    mtp_router = mtp_h * self.num_experts + self.num_experts
                    mtp_shared = 3 * mtp_h * self.moe_ffn
                    mtp_active_expert = 3 * mtp_h * self.moe_ffn * self.top_k
                    mtp_outer_norms = 2 * mtp_h
                    mtp_hc = self._hc_params()
                    block_params = (
                        mtp_attn + mtp_inner_norms + mtp_compressor + mtp_indexer
                        + mtp_router + mtp_shared + mtp_active_expert
                        + mtp_outer_norms + mtp_hc
                    )
                else:
                    block_params = self.params_per_dense_layer()
                layer_params += embed_proj + embed_norm + block_params

        lm_head = self.vocab * self.hidden
        return embed + layer_params + final_ln + lm_head
