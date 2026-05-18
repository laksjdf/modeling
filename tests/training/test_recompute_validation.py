"""Validation of recompute YAML inputs."""
from __future__ import annotations

import warnings

import pytest

from zrt.training.io.config_loader import load_specs


_BASE_YAML = """\
model: deepseek_v3_2
system:
  hw: nvidia_h100_sxm
  nodes: 1
  gpus_per_node: 8
strategy:
  tp: 8
  cp: 1
  pp: 1
  ep: 1
  dp: 1
  micro_batch: 1
  global_batch: 8
  recompute:
    per_layer:
{policy}
"""


def _write_tmp(tmp_path, policy: str) -> str:
    yaml_text = _BASE_YAML.format(policy=policy)
    f = tmp_path / "cfg.yaml"
    f.write_text(yaml_text)
    return str(f)


def test_unknown_category_raises(tmp_path):
    cfg = _write_tmp(tmp_path, '      moe: ["atn"]')
    with pytest.raises(ValueError, match="atn"):
        load_specs(cfg)


def test_dropped_attn_upscale_raises(tmp_path):
    cfg = _write_tmp(tmp_path, '      moe: ["attn_upscale"]')
    with pytest.raises(ValueError, match="attn_upscale"):
        load_specs(cfg)


def test_legacy_attn_alias_warns_but_loads(tmp_path):
    cfg = _write_tmp(tmp_path, '      moe: ["attn"]')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_specs(cfg)
    deprecated = [w for w in caught if "deprecated" in str(w.message).lower()]
    assert deprecated, (
        f"Expected DeprecationWarning for legacy 'attn' alias. "
        f"Got: {[str(w.message) for w in caught]}"
    )


def test_canonical_categories_load_silently(tmp_path):
    cfg = _write_tmp(
        tmp_path,
        '      moe: ["attn_core", "ffn_swiglu"]\n      dense: ["ln"]',
    )
    # Should not raise.
    load_specs(cfg)
