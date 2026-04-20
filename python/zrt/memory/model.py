"""Formula-based inference memory estimator."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from python.zrt.hardware.spec import HardwareSpec
from python.zrt.ir.graph import OpGraph
from python.zrt.transform.context import ParallelConfig, QuantConfig

from .activation import analyze_activation
from .budget import MemoryBudget

_MB = 1024.0 ** 2


@dataclass(frozen=True)
class _ProfileView:
    total_params: float | None
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    tie_word_embeddings: bool
    # MLA (Multi-head Latent Attention) 架构：DeepSeek-V2, Qwen-2.5 等
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    # MoE (Mixture of Experts) 架构：DeepSeek-V3, Mixtral 等
    num_experts: int | None = None
    num_shared_experts: int | None = None
    moe_topk: int | None = None


class MemoryModel:
    """Conservative per-device memory estimator for inference."""

    def __init__(self, overhead_ratio: float = 0.05, activation_slack: float = 1.1):
        self.overhead_ratio = max(0.0, overhead_ratio)
        self.activation_slack = max(1.0, activation_slack)

    def estimate(
        self,
        profile: Any,
        hw_spec: HardwareSpec,
        parallel: ParallelConfig,
        quant: QuantConfig | None = None,
        batch_size: int = 1,
        seq_len: int = 4096,
    ) -> MemoryBudget:
        p = self._coerce_profile(profile)
        weights = self._weights(p, parallel, quant)
        kv_cache = self._kv_cache(p, parallel, quant, batch_size, seq_len)

        if isinstance(profile, OpGraph) and profile.nodes:
            activation_analysis = analyze_activation(profile)
            activation = activation_analysis.peak_mb * self.activation_slack
        else:
            activation = self._activation_peak(p, parallel, quant, batch_size, seq_len)

        comm = self._comm_buffer(p, parallel, quant, batch_size, seq_len)
        overhead = (weights + kv_cache + activation + comm) * self.overhead_ratio
        total = weights + kv_cache + activation + comm + overhead
        capacity = hw_spec.memory.capacity_gb * 1024.0
        return MemoryBudget(
            weights_mb=weights,
            kv_cache_mb=kv_cache,
            activation_peak_mb=activation,
            comm_buffer_mb=comm,
            framework_overhead_mb=overhead,
            total_mb=total,
            capacity_mb=capacity,
            is_feasible=total <= capacity,
        )

    def _weights(
        self,
        profile: _ProfileView,
        parallel: ParallelConfig,
        quant: QuantConfig | None,
    ) -> float:
        total_params = profile.total_params or self._estimate_total_params(profile)
        bytes_per_param = quant.weight_bytes if quant else 2.0
        # TP + PP + EP 分片：每个设备只存储总参数量的 1/(tp*pp*ep)
        shard_factor = max(1, parallel.tp) * max(1, parallel.pp) * max(1, parallel.ep)
        return total_params * bytes_per_param / shard_factor / _MB

    def _kv_cache(
        self,
        profile: _ProfileView,
        parallel: ParallelConfig,
        quant: QuantConfig | None,
        batch_size: int,
        seq_len: int,
    ) -> float:
        kv_bytes = self._dtype_bytes((quant.kv_cache if quant else "bf16"))
        local_layers = max(1, math.ceil(profile.num_layers / max(1, parallel.pp)))

        # MLA (Multi-head Latent Attention): kv_dim = kv_lora_rank + qk_rope_head_dim
        if hasattr(profile, 'kv_lora_rank') and profile.kv_lora_rank:
            kv_dim = profile.kv_lora_rank + getattr(profile, 'qk_rope_head_dim', 0)
        else:
            # Standard GQA: kv_dim = kv_heads * head_dim
            local_kv_heads = max(1, math.ceil(profile.num_key_value_heads / max(1, parallel.tp)))
            kv_dim = local_kv_heads * profile.head_dim

        total_bytes = 2 * batch_size * seq_len * local_layers * kv_dim * kv_bytes
        return total_bytes / _MB

    def _activation_peak(
        self,
        profile: _ProfileView,
        parallel: ParallelConfig,
        quant: QuantConfig | None,
        batch_size: int,
        seq_len: int,
    ) -> float:
        act_bytes = self._dtype_bytes((quant.activation if quant else "bf16"))
        local_seq = max(1, math.ceil(seq_len / max(1, parallel.tp))) if parallel.sp else seq_len
        tokens_local = batch_size * local_seq
        hidden = profile.hidden_size
        mlp = max(profile.intermediate_size, hidden * 4)
        attn_scores = tokens_local * max(1, math.ceil(profile.num_attention_heads / max(1, parallel.tp)))
        elements = tokens_local * (4 * hidden + mlp) + attn_scores * max(1, local_seq)
        return elements * act_bytes * self.activation_slack / _MB

    def _comm_buffer(
        self,
        profile: _ProfileView,
        parallel: ParallelConfig,
        quant: QuantConfig | None,
        batch_size: int,
        seq_len: int,
    ) -> float:
        act_bytes = self._dtype_bytes((quant.activation if quant else "bf16"))
        seq_local = max(1, math.ceil(seq_len / max(1, parallel.tp))) if parallel.sp else seq_len
        shard_hidden = max(1, math.ceil(profile.hidden_size / max(1, parallel.tp)))
        tp_buffer = 0.0
        if parallel.tp > 1:
            tp_buffer = batch_size * seq_local * shard_hidden * act_bytes
        ep_buffer = 0.0
        if parallel.ep > 1:
            expert_width = max(profile.intermediate_size, profile.hidden_size)
            ep_buffer = batch_size * seq_local * expert_width * act_bytes
        return max(tp_buffer, ep_buffer) / _MB

    def _estimate_total_params(self, profile: _ProfileView) -> float:
        hidden = profile.hidden_size
        inter = max(profile.intermediate_size, hidden * 4)
        vocab = profile.vocab_size
        layers = profile.num_layers
        embeddings = hidden * vocab
        lm_head = 0 if profile.tie_word_embeddings else hidden * vocab
        attn = layers * 4 * hidden * hidden
        mlp = layers * 3 * hidden * inter
        norms = layers * 4 * hidden
        return float(embeddings + lm_head + attn + mlp + norms)

    def _coerce_profile(self, profile: Any) -> _ProfileView:
        source: Any = profile.metadata if isinstance(profile, OpGraph) else profile
        total_params = self._lookup_optional(
            source,
            "total_params",
            "param_count",
            "num_parameters",
            "parameter_count",
        )
        hidden_size = self._lookup_required(source, "hidden_size")
        intermediate_size = self._lookup_optional(source, "intermediate_size", default=hidden_size * 4)
        num_layers = self._lookup_required(
            source,
            "num_layers",
            "num_hidden_layers",
            "num_hidden_layers (traced)",
            "num_hidden_layers (full)",
        )
        num_attention_heads = self._lookup_required(source, "num_attention_heads")
        num_key_value_heads = self._lookup_optional(
            source,
            "num_key_value_heads",
            default=num_attention_heads,
        )
        head_dim = self._lookup_optional(
            source,
            "head_dim",
            "v_head_dim",
            default=max(1, hidden_size // max(1, num_attention_heads)),
        )
        vocab_size = self._lookup_optional(source, "vocab_size", default=hidden_size * 8)
        tie_word_embeddings = bool(self._lookup_optional(source, "tie_word_embeddings", default=True))
        # MLA 架构字段：DeepSeek-V2, Qwen-2.5 等
        kv_lora_rank = self._lookup_optional(source, "kv_lora_rank")
        qk_rope_head_dim = self._lookup_optional(source, "qk_rope_head_dim")
        # MoE 架构字段：DeepSeek-V3, Mixtral 等
        num_experts = self._lookup_optional(source, "num_experts")
        num_shared_experts = self._lookup_optional(source, "num_shared_experts")
        moe_topk = self._lookup_optional(source, "moe_topk")
        return _ProfileView(
            total_params=float(total_params) if total_params is not None else None,
            hidden_size=int(hidden_size),
            intermediate_size=int(intermediate_size),
            num_layers=int(num_layers),
            num_attention_heads=int(num_attention_heads),
            num_key_value_heads=int(num_key_value_heads),
            head_dim=int(head_dim),
            vocab_size=int(vocab_size),
            tie_word_embeddings=tie_word_embeddings,
            kv_lora_rank=int(kv_lora_rank) if kv_lora_rank is not None else None,
            qk_rope_head_dim=int(qk_rope_head_dim) if qk_rope_head_dim is not None else None,
            num_experts=int(num_experts) if num_experts is not None else None,
            num_shared_experts=int(num_shared_experts) if num_shared_experts is not None else None,
            moe_topk=int(moe_topk) if moe_topk is not None else None,
        )

    def _lookup_required(self, source: Any, *keys: str) -> Any:
        value = self._lookup_optional(source, *keys, default=None)
        if value is None:
            joined = ", ".join(keys)
            raise ValueError(f"Missing required memory profile field: {joined}")
        return value

    def _lookup_optional(self, source: Any, *keys: str, default: Any = None) -> Any:
        for key in keys:
            if isinstance(source, Mapping) and key in source:
                return source[key]
            if hasattr(source, key):
                return getattr(source, key)
            method = getattr(source, key, None)
            if callable(method):
                return method()
        if hasattr(source, "config"):
            return self._lookup_optional(source.config, *keys, default=default)
        return default

    def _dtype_bytes(self, dtype: str) -> float:
        name = dtype.lower()
        if name in {"int4"}:
            return 0.5
        if name in {"int8", "fp8", "fp8_e4m3", "fp8_e5m2"}:
            return 1.0
        if name in {"fp32"}:
            return 4.0
        return 2.0
