"""DeepSeek-V4 PretrainedConfig.

Field names follow HF convention; ``to_inference_args()`` maps them onto the
``ModelArgs`` dataclass used by ``inference/model.py``.  Defaults match the
DeepSeek-V4-Pro shipping config (see ``config.json``).
"""
from __future__ import annotations

from typing import Optional, Tuple

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DeepseekV4Config(PretrainedConfig):
    """Configuration for ``DeepseekV4ForCausalLM``.

    The geometry fields mirror DeepSeek-V3 (vocab/hidden/attention/experts) and
    the new V4-only fields cover Hyper-Connections (``hc_*``), KV compression
    (``compress_*``), the sparse-attn indexer (``index_*``) and low-rank
    output projection (``o_lora_rank`` / ``o_groups``).
    """

    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # Core geometry
        vocab_size: int = 129280,
        hidden_size: int = 7168,
        num_hidden_layers: int = 61,
        num_attention_heads: int = 128,
        num_key_value_heads: int = 1,
        num_nextn_predict_layers: int = 1,
        num_hash_layers: int = 3,
        # MoE
        moe_intermediate_size: int = 3072,
        n_shared_experts: int = 1,
        n_routed_experts: int = 384,
        num_experts_per_tok: int = 6,
        routed_scaling_factor: float = 2.5,
        scoring_func: str = "sqrtsoftplus",
        norm_topk_prob: bool = True,
        swiglu_limit: float = 10.0,
        expert_dtype: Optional[str] = "fp4",
        # MLA / attention
        head_dim: int = 512,
        qk_rope_head_dim: int = 64,
        q_lora_rank: int = 1536,
        o_lora_rank: int = 1024,
        o_groups: int = 16,
        sliding_window: int = 128,
        # KV compression (per-layer; len == num_hidden_layers + n_mtp when given)
        compress_ratios: Optional[Tuple[int, ...]] = None,
        compress_rope_theta: float = 160000.0,
        # Sparse-attn indexer
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 1024,
        # Hyper-Connections
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        # RoPE / YaRN
        max_position_embeddings: int = 1048576,
        original_max_position_embeddings: int = 65536,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        # Misc HF-required
        rms_norm_eps: float = 1e-6,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        use_cache: bool = True,
        torch_dtype: str = "bfloat16",
        # Tokens
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        # Core
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_hash_layers = num_hash_layers

        # MoE
        self.moe_intermediate_size = moe_intermediate_size
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.swiglu_limit = swiglu_limit
        self.expert_dtype = expert_dtype

        # Attention
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.sliding_window = sliding_window

        # Compression — default mirrors V4-Pro inference/config.json
        # (length 62 = num_hidden_layers + num_nextn_predict_layers).
        if compress_ratios is None:
            compress_ratios = tuple(
                ([128, 128, 4] + [128, 4] * 29 + [0])[: num_hidden_layers + num_nextn_predict_layers]
            )
        self.compress_ratios = tuple(compress_ratios)
        self.compress_rope_theta = compress_rope_theta

        # Indexer
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk

        # HC
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        # RoPE
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        # Misc
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache
        self.torch_dtype = torch_dtype

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def to_inference_args(self, max_batch_size: int = 4):
        """Return a ``ModelArgs`` instance compatible with ``inference/model.py``.

        Resolves YaRN scaling fields from ``rope_scaling`` when present.
        """
        from .modeling_deepseek import _import_inference_module
        ModelArgs = _import_inference_module().ModelArgs

        rs = self.rope_scaling or {}
        rope_factor = float(rs.get("factor", 16.0))
        beta_fast = int(rs.get("beta_fast", 32))
        beta_slow = int(rs.get("beta_slow", 1))
        original_seq_len = int(
            rs.get("original_max_position_embeddings", self.original_max_position_embeddings)
        )

        # Quant dtype: read from quantization_config when present; falls back
        # to the torch_dtype field for non-quantized configs.
        qcfg = getattr(self, "quantization_config", None) or {}
        if qcfg.get("quant_method") == "fp8":
            dtype = "fp8"
        else:
            dtype = "bf16"
        scale_fmt = qcfg.get("scale_fmt", "ue8m0")
        scale_dtype = "fp8" if scale_fmt == "ue8m0" else "fp32"

        return ModelArgs(
            max_batch_size=max_batch_size,
            max_seq_len=self.max_position_embeddings,
            dtype=dtype,
            scale_fmt=scale_fmt,
            scale_dtype=scale_dtype,
            expert_dtype=self.expert_dtype,
            vocab_size=self.vocab_size,
            dim=self.hidden_size,
            moe_inter_dim=self.moe_intermediate_size,
            n_layers=self.num_hidden_layers,
            n_hash_layers=self.num_hash_layers,
            n_mtp_layers=self.num_nextn_predict_layers,
            n_heads=self.num_attention_heads,
            n_routed_experts=self.n_routed_experts,
            n_shared_experts=self.n_shared_experts,
            n_activated_experts=self.num_experts_per_tok,
            score_func=self.scoring_func,
            route_scale=self.routed_scaling_factor,
            swiglu_limit=self.swiglu_limit,
            q_lora_rank=self.q_lora_rank,
            head_dim=self.head_dim,
            rope_head_dim=self.qk_rope_head_dim,
            norm_eps=self.rms_norm_eps,
            o_groups=self.o_groups,
            o_lora_rank=self.o_lora_rank,
            window_size=self.sliding_window,
            compress_ratios=tuple(self.compress_ratios),
            compress_rope_theta=self.compress_rope_theta,
            original_seq_len=original_seq_len,
            rope_theta=self.rope_theta,
            rope_factor=rope_factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            index_n_heads=self.index_n_heads,
            index_head_dim=self.index_head_dim,
            index_topk=self.index_topk,
            hc_mult=self.hc_mult,
            hc_sinkhorn_iters=self.hc_sinkhorn_iters,
            hc_eps=self.hc_eps,
        )
