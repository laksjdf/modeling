"""Tests for QuantPolicy on Strategy + YAML parsing (Stage A2)."""
from __future__ import annotations

import textwrap

import pytest
import yaml

from zrt.training.io.config_loader import _parse_strategy
from zrt.training.spec.strategy import QuantPolicy, Strategy


def test_default_quant_policy_matches_v1_behaviour():
    """No ``quant`` block in YAML → all casts fused → 0 cost (v1 baseline)."""
    pol = QuantPolicy()
    assert pol.assume_all_casts_fused is True
    assert pol.fuse_ln_epilog is True
    assert pol.fuse_gemm_epilog is True
    assert pol.fuse_attn_internal is True
    assert pol.ln_softmax_promote_fp32 is True
    # is_fused_at honors the master switch
    for site in ("ln_epilog", "gemm_epilog", "attn_internal", "other"):
        assert pol.is_fused_at(site) is True


def test_master_switch_off_honors_finegrained_flags():
    pol = QuantPolicy(
        assume_all_casts_fused=False,
        fuse_ln_epilog=True,
        fuse_gemm_epilog=False,
        fuse_attn_internal=True,
    )
    assert pol.is_fused_at("ln_epilog") is True
    assert pol.is_fused_at("gemm_epilog") is False
    assert pol.is_fused_at("attn_internal") is True
    # unrecognized sites are unfused
    assert pol.is_fused_at("other") is False
    assert pol.is_fused_at("residual_add") is False


def test_strategy_default_has_quant():
    s = Strategy()
    assert isinstance(s.quant, QuantPolicy)
    assert s.quant.assume_all_casts_fused is True


def test_yaml_no_quant_block_uses_defaults():
    cfg = yaml.safe_load(textwrap.dedent("""
        tp: 1
        pp: 1
        dp: 1
    """))
    s = _parse_strategy(cfg)
    assert s.quant == QuantPolicy()


def test_yaml_quant_block_parsed():
    cfg = yaml.safe_load(textwrap.dedent("""
        tp: 1
        pp: 1
        dp: 1
        quant:
          assume_all_casts_fused: false
          fuse_gemm_epilog: false
    """))
    s = _parse_strategy(cfg)
    assert s.quant.assume_all_casts_fused is False
    assert s.quant.fuse_gemm_epilog is False
    # un-specified flags keep their defaults
    assert s.quant.fuse_ln_epilog is True
    assert s.quant.fuse_attn_internal is True
    assert s.quant.ln_softmax_promote_fp32 is True
