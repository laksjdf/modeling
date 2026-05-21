"""Unit tests for `zrt.training.topology.process_groups`.

Verifies:
  - rank-set output matches Megatron-Core's RankGenerator for canonical
    strategies (TP/DP/PP groups).
  - EP groups partition the DP group correctly when ``ep < dp``.
  - Tier assignment on N-tier interconnects (2, 3, 4 tiers) picks the
    smallest tier that contains every group instance.
"""
from __future__ import annotations

import pytest

from zrt.hardware.spec import (
    InterconnectSpec,
    LinkSpec,
    TopologyTier,
)
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import GPU, SystemSpec
from zrt.training.topology import build_process_groups
from zrt.training.topology.process_groups import (
    DEFAULT_PARALLEL_ORDER,
    _build_groups_for_axis,
)


# ─────────────────────────────────────────────────────────────────────
# Rank-set parity with Megatron's RankGenerator (default order)
# ─────────────────────────────────────────────────────────────────────

def test_ws16_tp2_dp4_pp2_matches_megatron():
    """Canonical Megatron example (docstring of generate_masked_...).

    world_size=16, (tp=2, cp=1, ep=1, dp=4, pp=2), order=tp-cp-ep-dp-pp:
      TP: [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]]
      DP: [[0,2,4,6],[1,3,5,7],[8,10,12,14],[9,11,13,15]]
      PP: [[0,8],[1,9],[2,10],[3,11],[4,12],[5,13],[6,14],[7,15]]
    """
    s = Strategy(tp=2, cp=1, ep=1, dp=4, pp=2)
    g = build_process_groups(16, s)

    assert g.tp_groups == [
        [0, 1], [2, 3], [4, 5], [6, 7],
        [8, 9], [10, 11], [12, 13], [14, 15],
    ]
    assert g.dp_groups == [
        [0, 2, 4, 6], [1, 3, 5, 7],
        [8, 10, 12, 14], [9, 11, 13, 15],
    ]
    assert g.pp_groups == [
        [0, 8], [1, 9], [2, 10], [3, 11],
        [4, 12], [5, 13], [6, 14], [7, 15],
    ]


def test_tp_groups_span_consecutive_global_ranks():
    """TP innermost → ranks in one TP group are contiguous in rank-space."""
    s = Strategy(tp=8, cp=1, ep=1, dp=4, pp=2)
    g = build_process_groups(64, s)
    for grp in g.tp_groups:
        # contiguous, stride 1
        assert all(grp[i + 1] - grp[i] == 1 for i in range(len(grp) - 1))
    # number of TP groups = world_size / tp_size
    assert len(g.tp_groups) == 64 // 8


def test_pp_groups_span_far_apart_ranks():
    """PP outermost → ranks in one PP group are tp*cp*dp apart."""
    s = Strategy(tp=4, cp=2, ep=1, dp=2, pp=4)
    g = build_process_groups(64, s)
    # stride = tp*cp*ep*dp = 4*2*1*2 = 16
    for grp in g.pp_groups:
        assert all(grp[i + 1] - grp[i] == 16 for i in range(len(grp) - 1))
    assert len(g.pp_groups) == 16  # 64 / 4


def test_ep_partitions_dp_when_ep_less_than_dp():
    """ep=2, dp=4 → each DP group splits into 2 EP groups of size 2."""
    s = Strategy(tp=2, cp=1, ep=2, dp=4, pp=2)
    g = build_process_groups(16, s)

    # DP groups stay full size 4
    assert all(len(grp) == 4 for grp in g.dp_groups)
    # EP groups: each is half of a DP group
    assert all(len(grp) == 2 for grp in g.ep_groups)
    # Each EP group is a strict subset of some DP group
    for ep_grp in g.ep_groups:
        assert any(set(ep_grp).issubset(set(dp_grp)) for dp_grp in g.dp_groups)


def test_ws16_tp2_ep2_dp4_pp2_groups_match_megatron_core_layout():
    s = Strategy(tp=2, cp=1, ep=2, dp=4, pp=2)
    g = build_process_groups(16, s)

    assert g.dp_groups == [
        [0, 2, 4, 6], [1, 3, 5, 7],
        [8, 10, 12, 14], [9, 11, 13, 15],
    ]
    assert g.ep_groups == [
        [0, 2], [4, 6], [1, 3], [5, 7],
        [8, 10], [12, 14], [9, 11], [13, 15],
    ]
    assert g.expert_dp_groups == [
        [0, 4], [2, 6], [1, 5], [3, 7],
        [8, 12], [10, 14], [9, 13], [11, 15],
    ]


def test_ep_equals_dp_yields_one_ep_per_dp():
    """ep == dp → EP groups coincide with DP groups."""
    s = Strategy(tp=2, cp=1, ep=4, dp=4, pp=2)
    g = build_process_groups(16, s)
    # As sorted tuples, EP and DP groups are the same set
    assert {tuple(grp) for grp in g.ep_groups} == {tuple(grp) for grp in g.dp_groups}


def test_ep_degenerate_when_ep_one():
    """ep=1 → every rank in its own EP group of one."""
    s = Strategy(tp=2, cp=1, ep=1, dp=4, pp=2)
    g = build_process_groups(16, s)
    assert all(len(grp) == 1 for grp in g.ep_groups)


def test_rank_product_mismatch_rejected():
    """world_size must equal tp*cp*pp*dp."""
    s = Strategy(tp=2, cp=1, ep=1, dp=4, pp=2)  # product = 16
    with pytest.raises(ValueError, match="world_size"):
        build_process_groups(15, s)


def test_ranks_for_group_returns_group_instance():
    s = Strategy(tp=2, cp=1, ep=1, dp=4, pp=2)
    g = build_process_groups(16, s)
    # Rank 5 belongs to TP group [4,5], DP group [1,3,5,7], PP group [5,13]
    assert g.ranks_for("TP", 5) == [4, 5]
    assert g.ranks_for("DP", 5) == [1, 3, 5, 7]
    assert g.ranks_for("PP", 5) == [5, 13]


def test_default_order_is_megatron_default():
    assert DEFAULT_PARALLEL_ORDER == ("tp", "cp", "ep", "dp", "pp")


# ─────────────────────────────────────────────────────────────────────
# Tier assignment
# ─────────────────────────────────────────────────────────────────────

def _make_tiers(*specs: tuple[str, int, float]) -> InterconnectSpec:
    """Build an InterconnectSpec from (name, num_devices, bw) tuples."""
    tiers = []
    for name, n, bw in specs:
        tiers.append(TopologyTier(
            name=name,
            link=LinkSpec(
                type=name, bandwidth_gbps=bw, latency_us=1.0,
                topology="all_to_all" if n > 0 else "fat_tree",
                num_devices=n,
            ),
        ))
    return InterconnectSpec(tiers=tiers)


def _make_system(interconnect: InterconnectSpec, world: int) -> SystemSpec:
    return SystemSpec(
        gpu=GPU(name="test", flops_bf16=100, flops_fp8=200, hbm_gb=80, hbm_bw_gbps=3000),
        host_mem_gb=128,
        interconnect=interconnect,
        nodes=max(1, world // 8), gpus_per_node=8,
        world_size_override=world,
    )


def test_tier_assignment_2_tier_back_compat():
    """Old 2-tier YAML: TP fits intra-node, DP spans inter-node."""
    ic = _make_tiers(("intra", 8, 900), ("inter", 0, 400))
    s = Strategy(tp=8, cp=1, ep=1, dp=4, pp=1)
    g = build_process_groups(32, s, _make_system(ic, 32))
    assert g.tier["TP"].primary_tier == 0  # intra
    assert g.tier["DP"].primary_tier == 1  # inter (spans 4 nodes)


def test_tier_assignment_3_tier_picks_innermost_fit():
    """3-tier: TP→t0, CP→t1, DP→t2.

    Tier-1 size (rack) is chosen as a multiple of ``tp*cp`` so every CP
    group instance falls cleanly inside one rack. Otherwise the
    pessimistic check correctly bumps CP to the next tier (see
    test_tier_assignment_pessimistic_when_some_instances_straddle).
    """
    ic = _make_tiers(
        ("nvlink", 8, 1800),
        ("rack", 64, 900),   # 64 % (tp*cp=32) == 0 → every CP fits
        ("spine", 0, 400),
    )
    s = Strategy(tp=8, cp=4, ep=1, dp=8, pp=2)
    g = build_process_groups(512, s, _make_system(ic, 512))
    # TP=8 contiguous → fits 8-GPU NVLink
    assert g.tier["TP"].primary_tier == 0
    # CP=4 at stride 8 → 4 ranks per group span 25, all instances fit
    # within a 64-GPU rack (because 64 % (tp*cp)=64%32=0).
    assert g.tier["CP"].primary_tier == 1
    # DP=8 at stride 32 → span 225 > 64 → spine
    assert g.tier["DP"].primary_tier == 2


def test_tier_assignment_pessimistic_when_some_instances_straddle():
    """A CP=8 at stride 4 has some instances straddling rack boundaries
    even though span=29 ≤ 72; the algorithm must bump it to the next tier."""
    ic = _make_tiers(
        ("tray", 4, 1800),
        ("rack", 72, 900),
        ("rail", 576, 800),
        ("spine", 0, 400),
    )
    s = Strategy(tp=4, cp=8, ep=1, dp=16, pp=2)
    g = build_process_groups(1024, s, _make_system(ic, 1024))
    assert g.tier["TP"].primary_tier == 0  # tray (4)
    # CP=8 at stride 4: some instances at seed=64 straddle 64/72 boundary
    # → bump to next tier (rail) since rack does not contain every instance
    assert g.tier["CP"].primary_tier >= 2


def test_tier_assignment_falls_back_to_outermost():
    """A group too big for any bounded tier lands on the unbounded outermost."""
    ic = _make_tiers(("nvlink", 8, 1800), ("spine", 0, 400))
    s = Strategy(tp=1, cp=1, ep=1, dp=64, pp=1)
    g = build_process_groups(64, s, _make_system(ic, 64))
    # DP=64, stride 1: span 64 > 8 → spine
    assert g.tier["DP"].primary_tier == 1


def test_inner_tier_ignores_tiers_with_only_one_rank_per_instance():
    """A strided group cannot start decomposition at a tier with no local shard."""
    ic = _make_tiers(("tray", 8, 1800), ("spine", 0, 400))
    s = Strategy(tp=4, cp=4, ep=1, dp=8, pp=1)
    g = build_process_groups(128, s, _make_system(ic, 128))
    assert g.dp_groups[0] == [0, 16, 32, 48, 64, 80, 96, 112]
    assert g.tier["DP"].primary_tier == 1
    assert g.tier["DP"].inner_tier == 1


def test_inner_tier_can_start_at_middle_tier_when_inner_has_no_local_shard():
    """A group can decompose from rack tier even when tray tier has no fanout."""
    ic = _make_tiers(("tray", 4, 1800), ("rack", 64, 900), ("spine", 0, 400))
    s = Strategy(tp=4, cp=4, ep=1, dp=8, pp=1)
    g = build_process_groups(128, s, _make_system(ic, 128))
    assert g.dp_groups[0] == [0, 16, 32, 48, 64, 80, 96, 112]
    assert g.tier["DP"].primary_tier == 2
    assert g.tier["DP"].inner_tier == 1


def test_degree_one_axis_yields_singletons():
    """tp=1 → every TP "group" has size 1."""
    s = Strategy(tp=1, cp=1, ep=1, dp=4, pp=2)
    g = build_process_groups(8, s)
    assert all(len(grp) == 1 for grp in g.tp_groups)
    assert g.group_size("TP") == 1


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────

def test_build_groups_for_axis_strided_layout():
    """Direct check of stride computation for a 4D grid."""
    degrees = {"tp": 2, "cp": 1, "ep": 1, "dp": 2, "pp": 2}
    cp_groups = _build_groups_for_axis(
        target_axis="cp", order=DEFAULT_PARALLEL_ORDER,
        degrees=degrees, world_size=8,
    )
    # cp=1 → degenerate, every rank is its own group
    assert all(len(g) == 1 for g in cp_groups)

    degrees = {"tp": 2, "cp": 2, "ep": 1, "dp": 2, "pp": 1}
    cp_groups = _build_groups_for_axis(
        target_axis="cp", order=DEFAULT_PARALLEL_ORDER,
        degrees=degrees, world_size=8,
    )
    # CP stride = tp = 2, CP size = 2 → groups: [0,2], [1,3], [4,6], [5,7]
    assert cp_groups == [[0, 2], [1, 3], [4, 6], [5, 7]]
