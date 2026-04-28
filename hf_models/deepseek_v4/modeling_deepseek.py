"""DeepSeek-V4 HF wrapper around the official ``inference/model.py``.

This module is intentionally a thin shim:

- ``inference/model.py`` (kept verbatim from the official release) defines the
  real architecture as a single ``Transformer`` class.
- This file adapts it to the HuggingFace ``PreTrainedModel`` protocol so the
  ZRT-Sim graph capture pipeline (which goes through
  ``AutoModelForCausalLM.from_config(..., trust_remote_code=True)``) can
  instantiate the model.

Two pieces of glue are needed:

1. ``inference/model.py`` does ``from kernel import act_quant, ...``.  The
   official ``kernel.py`` is a TileLang JIT module that cannot run under
   ``FakeTensorMode``.  We pre-populate ``sys.modules["kernel"]`` with stub
   implementations *before* importing the inference module.  Step 3 replaces
   these stubs with shape/dtype-correct torch-only fakes.

2. ``inference/model.py`` lives in a sub-directory and is not picked up by
   transformers' dynamic-module loader.  We import it by absolute path with
   ``importlib.util.spec_from_file_location``.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_deepseek import DeepseekV4Config

logger = logging.getLogger(__name__)

_INFERENCE_MODULE_NAME = "_v4_inference_model"
_KERNEL_STUB_NAME = "kernel"
_HADAMARD_STUB_NAME = "fast_hadamard_transform"


# ─── Inference module loading ────────────────────────────────────────────────

def _candidate_inference_paths() -> list[Path]:
    """List candidate locations for ``inference/model.py``.

    Tries the file's sibling first (works in source layout), then walks up the
    parents of both ``__file__`` and ``cwd`` looking for
    ``hf_models/deepseek_v4/inference/model.py``.  This makes the loader
    resilient to transformers' dynamic-module copy step, which only copies
    top-level ``.py`` files into ``HF_MODULES_CACHE``.
    """
    here = Path(__file__).resolve()
    cwd = Path.cwd()

    cands: list[Path] = [here.parent / "inference" / "model.py"]

    for base in [here, cwd]:
        for i in range(min(8, len(base.parents))):
            cands.append(base.parents[i] / "hf_models" / "deepseek_v4" / "inference" / "model.py")

    cands.append(cwd / "hf_models" / "deepseek_v4" / "inference" / "model.py")

    seen: set[Path] = set()
    deduped: list[Path] = []
    for p in cands:
        rp = p.resolve(strict=False)
        if rp not in seen:
            seen.add(rp)
            deduped.append(rp)
    return deduped


def _install_kernel_stubs() -> None:
    """Install minimal ``sys.modules['kernel']`` and ``fast_hadamard_transform``.

    Step 3 replaces these with proper shape/dtype-correct fake kernels.  The
    stubs here only need to exist so ``from kernel import ...`` succeeds.
    """
    if _KERNEL_STUB_NAME not in sys.modules:
        kernel = types.ModuleType(_KERNEL_STUB_NAME)

        def act_quant(x, block_size=128, scale_fmt=None,
                      scale_dtype=torch.float32, inplace=False):
            N = x.size(-1)
            if inplace:
                return x
            y = x.new_empty(*x.shape, dtype=torch.float8_e4m3fn)
            s = x.new_empty(*x.shape[:-1], N // block_size, dtype=scale_dtype)
            return y, s

        def fp4_act_quant(x, block_size=32, inplace=False):
            N = x.size(-1)
            if inplace:
                return x
            y = x.new_empty(*x.shape[:-1], N // 2, dtype=torch.float4_e2m1fn_x2)
            s = x.new_empty(*x.shape[:-1], N // block_size, dtype=torch.float8_e8m0fnu)
            return y, s

        def _gemm(x, sx, w, sw, scale_dtype):
            out_shape = (*x.shape[:-1], w.shape[0])
            return x.new_empty(out_shape, dtype=torch.bfloat16)

        def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
            return torch.empty_like(q)

        def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4,
                              sinkhorn_iters=20, eps=1e-6):
            b, s, _ = mixes.shape
            pre = mixes.new_empty(b, s, hc_mult)
            post = mixes.new_empty(b, s, hc_mult)
            comb = mixes.new_empty(b, s, hc_mult, hc_mult)
            return pre, post, comb

        kernel.act_quant = act_quant
        kernel.fp4_act_quant = fp4_act_quant
        kernel.fp8_gemm = _gemm
        kernel.fp4_gemm = _gemm
        kernel.sparse_attn = sparse_attn
        kernel.hc_split_sinkhorn = hc_split_sinkhorn
        kernel._zrt_stub = True
        sys.modules[_KERNEL_STUB_NAME] = kernel

    # Upgrade stubs to the dtype/shape-correct torch fakes when available.
    # This is best-effort: in environments where python.zrt is not on the path
    # (e.g. HF_MODULES_CACHE shadow imports during config validation), the
    # placeholder stubs above remain in place and the inference module still
    # imports cleanly.
    try:
        from python.zrt.graph.v4_fake_kernels import install as _v4_install
        _v4_install()
    except Exception as exc:  # pragma: no cover — fallback path
        logger.debug("v4_fake_kernels not installed (%s); using basic stubs", exc)

    if _HADAMARD_STUB_NAME not in sys.modules:
        fht = types.ModuleType(_HADAMARD_STUB_NAME)

        def hadamard_transform(x, scale=1.0):
            return x * scale

        fht.hadamard_transform = hadamard_transform
        fht._zrt_stub = True
        sys.modules[_HADAMARD_STUB_NAME] = fht


_inference_module_cache = None


def _import_inference_module():
    """Import ``inference/model.py`` once and cache it.

    Stubs ``kernel`` / ``fast_hadamard_transform`` first so the source-level
    ``from kernel import ...`` line resolves without invoking TileLang.
    """
    global _inference_module_cache
    if _inference_module_cache is not None:
        return _inference_module_cache

    _install_kernel_stubs()

    candidates = _candidate_inference_paths()
    for path in candidates:
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location(_INFERENCE_MODULE_NAME, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[_INFERENCE_MODULE_NAME] = module
        spec.loader.exec_module(module)
        _inference_module_cache = module
        logger.info("Loaded DeepSeek-V4 inference module from %s", path)
        return module

    raise RuntimeError(
        "Could not locate DeepSeek-V4 inference/model.py.  Tried:\n  - "
        + "\n  - ".join(str(c) for c in candidates)
    )


# ─── HF PreTrainedModel wrappers ─────────────────────────────────────────────


class DeepseekV4PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV4Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block", "MTPBlock"]
    _skip_keys_device_placement = "past_key_values"


class DeepseekV4Model(DeepseekV4PreTrainedModel):
    """``AutoModel`` target — exposes the inference Transformer directly.

    The capture pipeline does not need a separate "trunk" module; HF only
    requires this class to exist so the ``AutoModel`` mapping resolves.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        inference = _import_inference_module()
        args = config.to_inference_args()
        self.transformer = inference.Transformer(args)

    def get_input_embeddings(self):
        return self.transformer.embed

    def set_input_embeddings(self, value):
        self.transformer.embed = value

    def forward(self, input_ids: torch.LongTensor, **_):
        return self.transformer(input_ids)


class DeepseekV4ForCausalLM(DeepseekV4PreTrainedModel):
    """Wraps ``inference.Transformer`` as a HF causal LM.

    Forward returns a ``CausalLMOutputWithPast`` with ``logits`` populated.
    KV cache management is delegated to the inference Transformer's internal
    buffers (``Attention.kv_cache``, etc.); HF-style ``past_key_values`` is
    ignored.
    """

    _tied_weights_keys: list[str] = []

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        inference = _import_inference_module()
        args = config.to_inference_args()
        self.transformer = inference.Transformer(args)
        self.config = config

    def get_input_embeddings(self):
        return self.transformer.embed

    def set_input_embeddings(self, value):
        self.transformer.embed = value

    def get_output_embeddings(self):
        return self.transformer.head

    def set_output_embeddings(self, new_embeddings):
        self.transformer.head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds=None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if input_ids is None:
            raise ValueError("DeepseekV4ForCausalLM requires input_ids")
        logits = self.transformer(input_ids)
        # inference.Transformer.head returns logits for the last position only
        # ([B, V]); expand to per-token shape so capture sees a (B, S, V) tensor.
        if logits.ndim == 2:
            B, V = logits.shape
            S = input_ids.shape[1]
            logits = logits.unsqueeze(1).expand(B, S, V)
        return CausalLMOutputWithPast(loss=None, logits=logits, past_key_values=None)


__all__ = [
    "DeepseekV4Config",
    "DeepseekV4PreTrainedModel",
    "DeepseekV4Model",
    "DeepseekV4ForCausalLM",
]
