"""Test EP rank-product policy — phase-3 adaptation point."""
from __future__ import annotations

import pytest

from zrt.training.spec.strategy import Strategy, rank_product
from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.spec.system import SystemSpec, GPU
from zrt.training.spec.model import ModelSpec, LayerKind


def _make_model():
    return ModelSpec(
        hidden=4096,
        ffn=11008,
        layers=[LayerKind.MOE for _ in range(4)],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        num_experts=8,
        moe_ffn=11008,
        top_k=1,
        seq_len=4096,
        vocab=32000,
    )


def _make_system(world_size=64):
    return SystemSpec(
        gpu=GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80, hbm_bw_gbps=3350),
        host_mem_gb=512,
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0, topology="all_to_all", num_devices=8),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=400, latency_us=5.0, topology="fat_tree"),
        ),
        nodes=world_size // 8,
        gpus_per_node=8,
    )


def test_rank_product_excludes_ep_current_policy():
    """Current policy: rank_product = tp * cp * pp * dp (EP excluded).

    This tests the current behavior where EP is treated as "inside" the
    rank product, matching SearchSpace.strategies() where dp is derived
    from world_size / (tp * cp * pp * ep).

    TODO: Phase-3 decision — if EP consumes distinct ranks, update this test
    to expect tp * cp * pp * ep * dp.
    """
    # 64 GPUs: TP=8, CP=1, PP=2, EP=4, DP=1
    # Current: 8*1*2*1 = 16 != 64 → should fail
    # Phase-3 (if EP inside): 8*1*2*4*1 = 64 → would pass
    model = _make_model()
    system = _make_system(64)
    strategy = Strategy(tp=8, cp=1, pp=2, ep=4, dp=1)

    # Current policy: EP not in rank product, so validation fails
    with pytest.raises(ValueError, match="TP\\(8\\)\\*CP\\(1\\)\\*PP\\(2\\)\\*DP\\(1\\)=16 != world_size\\(64\\)"):
        strategy.validate(model, system)


def test_rank_product_dp_derived_from_world_size():
    """SearchSpace derives dp = world_size / (tp * cp * pp * ep).

    This tests the symmetry between SearchSpace.strategies() and
    Strategy.validate() under the current policy.

    For world_size=64 with TP=8, CP=1, PP=2, EP=4:
    - SearchSpace computes: dp = 64 / (8*1*2*4) = 1
    - rank_product returns: 8*1*2*1 = 16 ≠ 64 → validation fails

    TODO: Phase-3 — when EP integration lands, decide if EP stays outside
    the product (current) or moves inside (would require updating both).
    """
    from zrt.training.search.space import SearchSpace

    model = _make_model()
    system = _make_system(64)
    space = SearchSpace(
        tp_values=[8],
        cp_values=[1],
        pp_values=[2],
        ep_values=[4],
        dp_values=[1],
        zero_stages=[0],  # Fixed to avoid combinatorial explosion
        pp_schedules=[Strategy.pp_schedule],  # Fixed
        recompute_policies=["none"],  # Fixed
        vpp_chunks_values=[1],  # Fixed
    )

    strategies = space.strategies(64)
    # With current policy, this returns strategies with dp=1
    # but validation will fail because rank_product excludes EP
    assert len(strategies) >= 1
    s = strategies[0]
    assert s.tp == 8 and s.cp == 1 and s.pp == 2 and s.ep == 4 and s.dp == 1

    # Validation fails because rank_product doesn't include EP
    with pytest.raises(ValueError):
        s.validate(model, system)


def test_valid_strategy_ep_not_affecting_rank_product():
    """Valid configs must work with current policy where EP is outside rank product.

    For world_size=64 with TP=8, CP=1, PP=2, EP=4:
    - To be valid, we need dp=4: rank_product = 8*1*2*4 = 64 ✓
    """
    model = _make_model()
    system = _make_system(64)
    strategy = Strategy(tp=8, cp=1, pp=2, ep=4, dp=4)

    # Should pass validation
    strategy.validate(model, system)  # No exception


def test_ep_rank_product_helper_function():
    """Test rank_product helper directly."""
    # Current policy: EP is not included
    assert rank_product(tp=8, cp=1, pp=2, ep=4, dp=1) == 16  # 8*1*2*1
    assert rank_product(tp=8, cp=1, pp=2, ep=4, dp=4) == 64  # 8*1*2*4

    # Verify EP value is ignored by the helper (current policy)
    assert rank_product(tp=2, cp=1, pp=2, ep=1, dp=8) == 32  # 2*1*2*8
    assert rank_product(tp=2, cp=1, pp=2, ep=8, dp=8) == 32  # 2*1*2*8 (EP ignored)
