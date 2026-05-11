"""Transformers version compatibility shims and local model registry.

Problem
-------
Model files in ``hf_models/`` are authored for the transformers version at
download time.  When transformers removes or renames an API (e.g., the
``is_flash_attn_greater_or_equal_2_10`` function removed in 5.x), the model
files fail to import even though we never call the removed API at runtime
(we trace on fake tensors, no real attention).

Approach (inspired by vLLM / SGLang)
-------------------------------------
Instead of pinning the transformers version or patching the model files
(which must remain original HF downloads), we:

1. **Inject missing symbols** into ``transformers.*`` sub-modules before any
   model file is imported.  Each shim is a lightweight stub — it only needs
   to satisfy the ``from transformers.xxx import yyy`` at module load time;
   the function body is never actually executed during FakeTensor tracing.

2. **Local model registry** maps ``model_type`` strings and HF Hub IDs to
   local ``hf_models/`` directories.  When a hub model ID fails with
   "architecture not recognized" (transformers doesn't know the model type),
   ``model_loader`` transparently reloads from the local directory which has
   a proper ``auto_map`` in ``config.json``.

Shim inventory
--------------
``is_flash_attn_greater_or_equal_2_10``
    Removed in transformers 5.x; replaced by
    ``is_flash_attn_greater_or_equal("2.1.0")``.
    Required by: DeepSeek-V3, DeepSeek-V3.2 (any model written for 4.x).

``is_torch_fx_available``  (transformers.utils, legacy location)
    Removed in some 5.x versions.  The ``import_utils`` location is already
    handled by ``patches.apply_compat_patches``; this covers the top-level
    ``transformers.utils`` import path used by a few model files.

Local registry
--------------
``_LOCAL_REGISTRY`` maps both ``model_type`` strings (from ``config.json``)
and canonical HF Hub IDs to paths relative to the project root.  To add a
new custom architecture, append an entry here — no other file needs changing.
"""
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Project root  ─────────────────────────────────────────────────────────────
# __file__ = python/zrt/graph/compat.py  →  parent×4 = repo root
_REPO_ROOT: Path = Path(__file__).resolve().parents[3]

# ── Local model registry ──────────────────────────────────────────────────────
# Maps model_type (from config.json) OR canonical HF Hub ID → local directory
# relative to _REPO_ROOT.
#
# The two-key design (by model_type + by hub ID) means:
#   - config loading failures (hub ID not found) use the hub-ID key
#   - model instantiation failures (model_type not registered) use the
#     model_type key
#
# When a new custom architecture is added to hf_models/, add entries for both
# its model_type and any known hub IDs.
_LOCAL_REGISTRY: Dict[str, str] = {
    # DeepSeek-V3 (deepseek_v3 model type)
    "deepseek_v3":                   "hf_models/deepseek_v3",
    "deepseek-ai/DeepSeek-V3":       "hf_models/deepseek_v3",
    "deepseek-ai/DeepSeek-V3-0324":  "hf_models/deepseek_v3",   # same arch as V3

    # DeepSeek-V3.2 (deepseek_v32 model type — not in transformers registry)
    "deepseek_v32":                  "hf_models/deepseek_v3_2",
    "deepseek-ai/DeepSeek-V3.2":     "hf_models/deepseek_v3_2",

    # DeepSeek-V4 (deepseek_v4 model type — not in transformers registry)
    "deepseek_v4":                   "hf_models/deepseek_v4",
    "deepseek-ai/DeepSeek-V4":       "hf_models/deepseek_v4",
    "deepseek-ai/DeepSeek-V4-Pro":   "hf_models/deepseek_v4",
}


def find_local_fallback(model_id_or_type: str) -> Optional[Path]:
    """Return the local model directory for *model_id_or_type*, or ``None``.

    Looks up both the full string and a normalised version (strips org prefix
    for hub IDs).

    Parameters
    ----------
    model_id_or_type:
        A HF Hub model ID (``"deepseek-ai/DeepSeek-V3.2"``) or a
        ``model_type`` string (``"deepseek_v32"``).

    Returns
    -------
    Absolute ``Path`` if a registered local directory exists; ``None`` otherwise.
    """
    keys = [model_id_or_type]
    # Also try the bare name without the org prefix
    if "/" in model_id_or_type:
        keys.append(model_id_or_type.split("/", 1)[1])

    for key in keys:
        rel = _LOCAL_REGISTRY.get(key)
        if rel:
            path = _REPO_ROOT / rel
            if path.exists():
                return path
            logger.warning(
                "Registry entry '%s' -> '%s' points to a non-existent directory.",
                key, path,
            )
    return None


# ── Version shims ─────────────────────────────────────────────────────────────

def apply_version_shims() -> None:
    """Inject symbols removed/renamed between transformers versions.

    Safe to call multiple times (idempotent — skips already-present attrs).
    Must be called **before** any model file is imported so that the
    ``from transformers.xxx import yyy`` at module top-level succeeds.
    """
    # transformers 5.x removed is_flash_attn_greater_or_equal_2_10.
    # The function was used as a bool guard inside flash-attention branches that
    # are never taken during FakeTensor tracing, so a stub suffices.
    _inject_if_missing(
        module="transformers.utils",
        attr="is_flash_attn_greater_or_equal_2_10",
        make_value=_make_flash_attn_2_10,
        reason=(
            "removed in transformers 5.x; "
            "replaced by is_flash_attn_greater_or_equal('2.1.0')"
        ),
    )

    # transformers.utils.is_torch_fx_available was removed from the top-level
    # utils namespace in some 5.x versions (still present in import_utils).
    _inject_if_missing(
        module="transformers.utils",
        attr="is_torch_fx_available",
        make_value=lambda: (lambda: True),
        reason="removed from transformers.utils top-level in 5.x",
    )

    # transformers 5.x removed DynamicCache.from_legacy_cache / to_legacy_cache.
    # These are called by model files written for 4.x when past_key_values is a
    # legacy tuple-of-tuples.  During FakeTensor tracing we need these methods
    # so that the decode pass can receive the KV cache produced by the prefill.
    _shim_dynamic_cache_legacy_api()


def _shim_dynamic_cache_legacy_api() -> None:
    """Restore ``from_legacy_cache`` / ``to_legacy_cache`` on ``DynamicCache``.

    These class / instance methods were available in transformers 4.x but
    removed in 5.x.  Custom model files (e.g. DeepSeek-V3 downloaded for 4.x)
    call them at runtime when ``use_cache=True``.  The shims are lightweight:
    they only marshal between the old tuple-of-tuples format and the new
    ``DynamicCache`` instance.
    """
    try:
        from transformers import DynamicCache
    except ImportError:
        return

    _injected: list[str] = []

    # ── from_legacy_cache / to_legacy_cache ───────────────────────────────
    if not hasattr(DynamicCache, "from_legacy_cache"):
        @classmethod  # type: ignore[misc]
        def from_legacy_cache(cls, past_key_values=None):
            """Construct a ``DynamicCache`` from a legacy ``tuple[tuple[Tensor]]``."""
            cache = cls()
            if past_key_values is None:
                return cache
            # Use update() so the 5.x internal layer storage is populated.
            for layer_idx, layer_past in enumerate(past_key_values):
                cache.update(layer_past[0], layer_past[1], layer_idx)
            return cache

        def to_legacy_cache(self):
            """Convert this cache to legacy ``tuple[tuple[Tensor]]`` format."""
            # 5.x stores layers in self.layers (list of DynamicLayer objects)
            # each with .keys / .values attributes.
            return tuple(
                (layer.keys, layer.values)
                for layer in self.layers
            )

        DynamicCache.from_legacy_cache = from_legacy_cache  # type: ignore[attr-defined]
        DynamicCache.to_legacy_cache = to_legacy_cache       # type: ignore[attr-defined]
        _injected.extend(["from_legacy_cache", "to_legacy_cache"])

    # ── get_usable_length ─────────────────────────────────────────────────
    # 4.x: get_usable_length(new_seq_length, layer_idx=0) → cached seq length.
    # 5.x renamed to get_seq_length(layer_idx=0).
    if not hasattr(DynamicCache, "get_usable_length"):
        def get_usable_length(self, new_seq_length=0, layer_idx=0):  # type: ignore[misc]
            """Compatibility shim: delegate to get_seq_length."""
            return self.get_seq_length(layer_idx)

        DynamicCache.get_usable_length = get_usable_length  # type: ignore[attr-defined]
        _injected.append("get_usable_length")

    # ── seen_tokens property ──────────────────────────────────────────────
    # 4.x: instance attr updated by update(); 5.x: use get_seq_length().
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(  # type: ignore[attr-defined]
            lambda self: self.get_seq_length()
        )
        _injected.append("seen_tokens")

    # ── get_max_length ────────────────────────────────────────────────────
    # 4.x: returns max allowed length (None = unbounded).
    # 5.x: renamed to get_max_cache_shape() with different semantics.
    if not hasattr(DynamicCache, "get_max_length"):
        def get_max_length(self):  # type: ignore[misc]
            """Compatibility shim: DynamicCache is unbounded → return None."""
            return None

        DynamicCache.get_max_length = get_max_length  # type: ignore[attr-defined]
        _injected.append("get_max_length")

    if _injected:
        logger.debug(
            "Injected DynamicCache methods: %s  "
            "[removed in transformers 5.x; required by 4.x model files]",
            ", ".join(_injected),
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _inject_if_missing(
    module: str,
    attr: str,
    make_value,
    reason: str = "",
) -> None:
    """Set *attr* on *module* if not already present.

    Parameters
    ----------
    module:     Dotted module path (``"transformers.utils"``).
    attr:       Attribute name to inject.
    make_value: Zero-argument callable whose return value is injected.
                It is called lazily, only when the attr is actually missing.
    reason:     Human-readable reason logged at DEBUG level.
    """
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return
    if hasattr(mod, attr):
        return
    value = make_value()
    setattr(mod, attr, value)
    logger.debug("Injected %s.%s  [%s]", module, attr, reason)


def _make_flash_attn_2_10():
    """Return a stub for ``is_flash_attn_greater_or_equal_2_10``.

    Prefers the new ``is_flash_attn_greater_or_equal("2.1.0")`` API when
    available; otherwise returns a constant-``False`` lambda.
    """
    import transformers.utils as tu
    if hasattr(tu, "is_flash_attn_greater_or_equal"):
        def _stub() -> bool:
            return tu.is_flash_attn_greater_or_equal("2.1.0")
        return _stub
    return lambda: False
