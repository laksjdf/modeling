"""Test communication model — alpha-beta collective costs."""

import pytest
from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.ir.training_graph import Collective
from zrt.training.models.comm import collective_time, tier_for_group, total_comm_time
from zrt.training.spec.system import SystemSpec, GPU


def _intra_link():
    return LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0, topology="all_to_all", num_devices=8)


def _inter_link():
    return LinkSpec(type="IB", bandwidth_gbps=400, latency_us=5.0, topology="fat_tree")


def _make_system():
    return SystemSpec(
        gpu=GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80, hbm_bw_gbps=3350),
        host_mem_gb=256,
        interconnect=InterconnectSpec(intra_node=_intra_link(), inter_node=_inter_link()),
        nodes=4, gpus_per_node=8,
    )


def test_p2p_time():
    """P2P: alpha + S * beta."""
    link = _intra_link()
    c = Collective("test_p2p", "P2P", "PP", 1024 * 1024, "op1")  # 1 MB
    t = collective_time(c, 2, link)
    assert t > 0
    # alpha = 1us; beta uses the unified effective bandwidth (peak ×
    # kb_efficiency). S = 1MB = 1048576 bytes.
    expected = 1e-6 + 1048576 / link.effective_bw_bps(2)
    assert t == pytest.approx(expected, rel=0.01)


def test_ag_time():
    """AG full-mesh (NCCL bus-bw convention): alpha + S·(N-1)/N · beta.

    The previous form `alpha + (S/N)·beta` dropped the (N-1) factor on the
    bandwidth term, under-counting intra-node AG/RS/AR/A2A by ~N×.
    """
    link = _intra_link()
    N = 8
    S = 100 * 1024 * 1024  # 100 MB
    c = Collective("test_ag", "AG", "TP", S, "op1")
    t = collective_time(c, N, link)

    alpha = 1e-6  # all_to_all → switched_full → single-step latency
    beta = 1.0 / link.effective_bw_bps(N)
    expected = alpha + S * (N - 1) / N * beta
    assert t == pytest.approx(expected, rel=0.01)


def test_ar_time_is_2x_ag():
    """AllReduce = 2 * AG time (ring algorithm)."""
    link = _intra_link()
    N = 8
    S = 50 * 1024 * 1024  # 50 MB
    c_ag = Collective("ag", "AG", "TP", S, "op1")
    c_ar = Collective("ar", "AR", "DP", S, "op1")

    t_ag = collective_time(c_ag, N, link)
    t_ar = collective_time(c_ar, N, link)

    assert t_ar == pytest.approx(2 * t_ag, rel=0.01)


def test_a2a_time():
    """A2A full-mesh: alpha + S·(N-1)/N · beta (NCCL bus-bw)."""
    link = _intra_link()
    N = 4
    S = 20 * 1024 * 1024
    c = Collective("test_a2a", "A2A", "EP", S, "op1")
    t = collective_time(c, N, link)

    alpha = 1e-6
    beta = 1.0 / link.effective_bw_bps(N)
    expected = alpha + S * (N - 1) / N * beta
    assert t == pytest.approx(expected, rel=0.01)


def test_tier_selection_intra():
    """TP within a node should use intra_node link."""
    system = _make_system()
    link = tier_for_group("TP", 8, system)
    assert link.type == "NVLink"


def test_tier_selection_inter():
    """DP across nodes should use inter_node link."""
    system = _make_system()
    link = tier_for_group("DP", 32, system)
    assert link.type == "IB"


def test_dp_grad_reduce_zero3_uses_full_per_rank_volume():
    """ZeRO-3 RS payload S in alpha-beta should be the per-rank pre-shard volume,
    not post-shard owner slice. The formula (N-1)*(α + S/N*β) already divides by N.

    Regression for: comm.py:275-277 incorrectly divided total by dp for ZeRO>=2,
    producing ~N× underestimate (1.62 ms vs textbook 154 ms for DSV4-pro DP=512).
    """
    from zrt.training.spec.model import ModelSpec, LayerKind
    from zrt.training.spec.strategy import Strategy
    from zrt.training.models.comm import total_comm_time
    from zrt.training.ir.builders import build_graph

    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 8,
    )
    system = _make_system()
    strategy_z1 = Strategy(tp=1, pp=1, dp=8, micro_batch=1, global_batch=8, zero_stage=1)
    strategy_z3 = Strategy(tp=1, pp=1, dp=8, micro_batch=1, global_batch=8, zero_stage=3)

    g1 = build_graph(model, strategy_z1)
    g3 = build_graph(model, strategy_z3)
    t_z1 = total_comm_time(g1, model, system, strategy_z1)["dp_grad_reduce"]
    t_z3 = total_comm_time(g3, model, system, strategy_z3)["dp_grad_reduce"]

    # ZeRO-3 RS bytes ≈ ZeRO-1 AR bytes / 2 (RS is half of AR, both formulas use same S).
    # Before fix: t_z3 was ~N× smaller than t_z1 due to double-divide.
    # After fix: t_z3 should be roughly t_z1 / 2 (kind difference only).
    assert t_z3 > t_z1 / 4, (
        f"ZeRO-3 dp_grad_reduce={t_z3*1000:.3f}ms looks bug-divided by dp; "
        f"expected ≈ ZeRO-1 AR / 2 ≈ {t_z1*1000/2:.3f}ms"
    )


def test_params_on_rank_for_dp_independent_of_zero_stage():
    """_params_on_rank_for_dp should return the same per-rank gradient quantity
    regardless of zero_stage (1/2/3). ZeRO sharding affects how that volume is
    split during the collective, not the input payload to the collective."""
    from zrt.training.spec.model import ModelSpec, LayerKind
    from zrt.training.spec.strategy import Strategy
    from zrt.training.models.comm import _params_on_rank_for_dp

    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 8,
    )
    s_z1 = Strategy(tp=1, pp=1, dp=8, micro_batch=1, global_batch=8, zero_stage=1)
    s_z3 = Strategy(tp=1, pp=1, dp=8, micro_batch=1, global_batch=8, zero_stage=3)

    assert _params_on_rank_for_dp(model, s_z1) == _params_on_rank_for_dp(model, s_z3), (
        "ZeRO stage should not change the per-rank pre-RS gradient volume"
    )


def test_a2a_large_n_uses_log2_latency():
    """A2A on fat-tree with N > 16 should use Bruck-style log2(N) latency,
    not pairwise-exchange (N-1)·α. NCCL switches to Bruck at scale.

    For N=384, α=2µs: pairwise=766µs, Bruck=9·2µs=18µs."""
    from zrt.hardware.spec import LinkSpec
    from zrt.training.ir.training_graph import Collective
    link = LinkSpec(type="IB", bandwidth_gbps=400, latency_us=2.0, topology="fat_tree")
    N = 384
    S = 1024 * 1024  # 1 MB
    c = Collective("test_a2a", "A2A", "EP", S, "op1")
    t = collective_time(c, N, link)

    # Bandwidth term: (N-1)/N * S * β ≈ S * β for large N
    beta = 1.0 / link.effective_bw_bps(N)
    bandwidth = (N - 1) / N * S * beta
    # Latency: log2(384) ≈ 9 rounds × 2µs = 18 µs
    expected_max = 12 * 2e-6 + bandwidth  # allow ceil + small margin
    assert t < expected_max, (
        f"A2A on N=384 should use log2 latency, got {t*1e6:.1f}µs "
        f"(expected < {expected_max*1e6:.1f}µs)"
    )
    # Sanity: still > pure bandwidth
    assert t > bandwidth, "should have non-zero latency"


def test_hierarchical_ag_beats_flat_for_multi_node():
    """For a DP group that spans nodes, 2-level hierarchical AG should be
    faster than (or equal to) a flat inter-node ring — latency is paid
    only L=N/D times instead of N-1.
    """
    from zrt.training.models.comm import collective_time_hierarchical
    system = _make_system()  # 4 nodes × 8 gpus/node, inter=IB fat_tree
    N = 32  # multi-node group
    S = 100 * 1024 * 1024
    c = Collective("test_hier", "AG", "DP", S, "op1")
    t_hier = collective_time_hierarchical(c, N, system)
    t_flat = collective_time(c, N, system.interconnect.inter_node)
    assert t_hier <= t_flat, (
        f"hierarchical AG ({t_hier*1e6:.1f}µs) should be ≤ flat "
        f"inter-node ({t_flat*1e6:.1f}µs) for multi-node groups"
    )


def test_hierarchical_ag_falls_back_to_intra_within_node():
    """Group fitting in one node: hierarchical should match flat intra-node."""
    from zrt.training.models.comm import collective_time_hierarchical
    system = _make_system()
    N = 8  # fits in one node
    S = 100 * 1024 * 1024
    c = Collective("test_hier", "AG", "TP", S, "op1")
    t_hier = collective_time_hierarchical(c, N, system)
    t_intra = collective_time(c, N, system.interconnect.intra_node)
    assert t_hier == pytest.approx(t_intra, rel=1e-9)


def test_a2a_small_n_keeps_legacy_formula():
    """Small N (≤16) keeps pairwise-exchange (N-1)·(α+S/N·β) for backward-compat
    with intra-node A2A tests."""
    from zrt.hardware.spec import LinkSpec
    from zrt.training.ir.training_graph import Collective
    link = LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0,
                    topology="fat_tree", num_devices=8)  # not full_connectivity
    N = 8
    S = 1024 * 1024
    c = Collective("test_a2a", "A2A", "TP", S, "op1")
    t = collective_time(c, N, link)
    alpha = 1e-6
    beta = 1.0 / link.effective_bw_bps(N)
    expected_legacy = (N - 1) * alpha + S * (N - 1) / N * beta
    assert t == pytest.approx(expected_legacy, rel=1e-3), (
        f"Small N should keep legacy formula; got {t} vs {expected_legacy}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# P1/P2: topology-class dispatch, kb_efficiency, oversubscription
# ─────────────────────────────────────────────────────────────────────────────

def test_switched_tree_uses_log2_latency_vs_ring():
    """fat_tree/clos (switched_tree) AG latency = ceil(log2 N)·α, far below
    ring's (N-1)·α for large N."""
    import math
    S = 64 * 1024 * 1024
    N = 64
    tree = LinkSpec(type="IB", bandwidth_gbps=400, latency_us=5.0, topology="fat_tree")
    ring = LinkSpec(type="x", bandwidth_gbps=400, latency_us=5.0, topology="ring")
    c = Collective("c", "AG", "DP", S, "op1")
    t_tree = collective_time(c, N, tree)
    t_ring = collective_time(c, N, ring)

    alpha = 5e-6
    # Same bandwidth term (both kb_efficiency=0.7, no oversub) → latency only.
    lat_tree = t_tree - S * (N - 1) / N / tree.effective_bw_bps(N)
    lat_ring = t_ring - S * (N - 1) / N / ring.effective_bw_bps(N)
    assert lat_tree == pytest.approx(math.ceil(math.log2(N)) * alpha, rel=1e-6)
    assert lat_ring == pytest.approx((N - 1) * alpha, rel=1e-6)
    assert t_tree < t_ring


def test_unknown_topology_falls_back_to_ring():
    """Unknown topology must behave exactly like ring ((N-1)-step latency)."""
    S = 8 * 1024 * 1024
    N = 16
    unk = LinkSpec(type="x", bandwidth_gbps=200, latency_us=3.0, topology="weird_fabric")
    ring = LinkSpec(type="x", bandwidth_gbps=200, latency_us=3.0, topology="ring")
    c = Collective("c", "AR", "DP", S, "op1")
    assert unk.topology_class == "ring"
    assert collective_time(c, N, unk) == pytest.approx(
        collective_time(c, N, ring), rel=1e-12
    )


def test_kb_efficiency_scales_bandwidth():
    """effective_bw_bps = peak/8 × kb_efficiency; halving kb halves bw."""
    full = LinkSpec(type="x", bandwidth_gbps=800, latency_us=1.0,
                    topology="full_mesh", kb_efficiency=0.7)
    half = LinkSpec(type="x", bandwidth_gbps=800, latency_us=1.0,
                    topology="full_mesh", kb_efficiency=0.35)
    assert full.effective_bw_bps(8) == pytest.approx(800e9 / 8 * 0.7)
    assert full.effective_bw_bps(8) == pytest.approx(2 * half.effective_bw_bps(8))


def test_oversubscription_continuous_scale_derate():
    """switched_tree with oversubscription>1: no derate within the
    non-blocking radix, then a *continuous* monotone derate
    s = 1 + (os-1)*(1 - R/N) rising toward os as the domain grows."""
    link = LinkSpec(type="IB", bandwidth_gbps=400, latency_us=5.0,
                    topology="fat_tree", num_devices=8, oversubscription=4.0)
    base = 400e9 / 8 * 0.7
    assert link.effective_bw_bps(8) == pytest.approx(base)          # at radix → s=1
    # N=64: s = 1 + 3*(1 - 8/64) = 3.625 (continuous, not the flat /4 step)
    assert link.effective_bw_bps(64) == pytest.approx(base / 3.625)
    # monotone decreasing in N, asymptote → base/os
    assert link.effective_bw_bps(16) > link.effective_bw_bps(64) > link.effective_bw_bps(4096)
    assert link.effective_bw_bps(10**9) == pytest.approx(base / 4.0, rel=1e-3)


def test_oversubscription_applies_on_unbounded_inter_link():
    """Inter-node fat_tree convention uses num_devices=0 (cluster-scale, no
    non-blocking bound): the whole link is the over-subscribed spine, so the
    derate must apply for ANY group size (regression for the radix=group_size
    fallback that made oversubscription a no-op on exactly these links)."""
    link = LinkSpec(type="RoCE", bandwidth_gbps=200, latency_us=5.0,
                    topology="fat_tree", num_devices=0, oversubscription=4.0)
    base = 200e9 / 8 * 0.7
    assert link.effective_bw_bps(2) == pytest.approx(base / 4.0)
    assert link.effective_bw_bps(512) == pytest.approx(base / 4.0)


def test_clos_is_own_class_never_derates_keeps_log2_latency():
    """clos is its own class: full per-card bandwidth at any scale (NEVER
    derated, even if oversubscription is mis-set), but log2(N) latency like
    a switched tree."""
    import math
    clos = LinkSpec(type="x", bandwidth_gbps=400, latency_us=5.0,
                    topology="clos", num_devices=8, oversubscription=4.0)
    assert clos.topology_class == "clos"
    # Bandwidth independent of domain size despite oversubscription=4.0.
    base = 400e9 / 8 * 0.7
    assert clos.effective_bw_bps(8) == pytest.approx(base)
    assert clos.effective_bw_bps(4096) == pytest.approx(base)

    # Latency: clos uses the switched-tree log2(N) law, not ring (N-1).
    S, N = 64 * 1024 * 1024, 64
    c = Collective("c", "AG", "DP", S, "op1")
    t = collective_time(c, N, clos)
    lat = t - S * (N - 1) / N / clos.effective_bw_bps(N)
    assert lat == pytest.approx(math.ceil(math.log2(N)) * 5e-6, rel=1e-6)


def test_clos_beats_fat_tree_at_scale_when_oversubscribed():
    """The intent of option (i): with an over-subscribed fat-tree, clos keeps
    full bandwidth while fat-tree degrades with domain size, so clos is
    strictly faster for a large cross-spine collective."""
    S, N = 64 * 1024 * 1024, 256
    clos = LinkSpec(type="x", bandwidth_gbps=400, latency_us=5.0,
                    topology="clos", num_devices=0)
    ftree = LinkSpec(type="x", bandwidth_gbps=400, latency_us=5.0,
                     topology="fat_tree", num_devices=0, oversubscription=4.0)
    c = Collective("c", "AR", "DP", S, "op1")
    assert collective_time(c, N, clos) < collective_time(c, N, ftree)


def test_effective_flops_override_precedence():
    """GPU.compute_efficiency overrides the size-bucket heuristic; None keeps it."""
    from zrt.training.io.perf_tables import effective_flops, achieved_flops_efficiency
    from zrt.training.spec.dtype import Dtype

    flops = 5e10
    g_heur = GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80, hbm_bw_gbps=3350)
    g_over = GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80,
                 hbm_bw_gbps=3350, compute_efficiency=0.9)
    peak = 989e12
    assert effective_flops(g_heur, Dtype.BF16, flops) == pytest.approx(
        peak * achieved_flops_efficiency("h100", Dtype.BF16, flops)
    )
    assert effective_flops(g_over, Dtype.BF16, flops) == pytest.approx(peak * 0.9)


def test_effective_hbm_bw_override_precedence():
    from zrt.training.io.perf_tables import (
        effective_hbm_bw_bps, achieved_bandwidth_efficiency,
    )
    bytes_ = 5e7
    g_heur = GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80, hbm_bw_gbps=3350)
    g_over = GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80,
                 hbm_bw_gbps=3350, mem_bw_efficiency=0.55)
    bw = 3350e9
    assert effective_hbm_bw_bps(g_heur, bytes_) == pytest.approx(
        bw * achieved_bandwidth_efficiency("h100", bytes_)
    )
    assert effective_hbm_bw_bps(g_over, bytes_) == pytest.approx(bw * 0.55)


# ─────────────────────────────────────────────────────────────────────
# N-tier multi-level decomposition
# ─────────────────────────────────────────────────────────────────────

def _make_3tier_system(world_size: int = 512) -> SystemSpec:
    """3-tier system: tray(4) → rack(64) → spine(unbounded)."""
    from zrt.hardware.spec import TopologyTier
    tray = LinkSpec(type="NVLink5", bandwidth_gbps=1800, latency_us=0.5,
                    topology="all_to_all", num_devices=4, kb_efficiency=0.85)
    rack = LinkSpec(type="NVSwitch5", bandwidth_gbps=900, latency_us=1.0,
                    topology="all_to_all", num_devices=64, kb_efficiency=0.75)
    spine = LinkSpec(type="IB_XDR", bandwidth_gbps=400, latency_us=5.0,
                     topology="fat_tree", num_devices=0, kb_efficiency=0.6)
    interconnect = InterconnectSpec(tiers=[
        TopologyTier(name="tray", link=tray),
        TopologyTier(name="rack", link=rack),
        TopologyTier(name="spine", link=spine),
    ])
    return SystemSpec(
        gpu=GPU(name="b300", flops_bf16=5000, flops_fp8=10000, hbm_gb=288, hbm_bw_gbps=8000),
        host_mem_gb=512, interconnect=interconnect,
        nodes=world_size // 4, gpus_per_node=4,
        world_size_override=world_size,
    )


def test_multi_tier_ag_decomposes_across_three_tiers():
    """A group spanning tray + rack + spine pays α-β at every tier it crosses."""
    from zrt.training.models.comm import (
        _tier_breakdown, collective_time_multi_tier,
    )

    sys = _make_3tier_system(world_size=512)
    # Group of 64 ranks at stride 1 → all in one rack (1 rack instance, 16 trays)
    ranks_in_rack = list(range(64))
    breakdown = _tier_breakdown(ranks_in_rack, sys.interconnect.tiers)
    # Decomposes into tray (4) + rack (16) only; no spine traffic
    assert [d for d, _ in breakdown] == [4, 16]

    # Group of 128 ranks → spans 2 racks (1 spine instance), 32 trays
    ranks_cross_rack = list(range(128))
    bd2 = _tier_breakdown(ranks_cross_rack, sys.interconnect.tiers)
    assert [d for d, _ in bd2] == [4, 16, 2]

    # AG cost: full cross-rack > rack-only > tray-only
    S = 64 * 1024 * 1024
    c = Collective("ag_test", "AG", "DP", S, "op")
    t_64 = collective_time_multi_tier(c, ranks_in_rack, sys)
    t_128 = collective_time_multi_tier(c, ranks_cross_rack, sys)
    assert t_128 > t_64, (
        f"crossing the spine should cost more (got {t_128*1e6:.1f}µs vs "
        f"{t_64*1e6:.1f}µs single-rack)"
    )


def test_multi_tier_2tier_backcompat_matches_legacy_hierarchical():
    """For 2-tier system + consecutive ranks aligned to D, multi-tier
    AG cost must equal the legacy 2-level formula exactly."""
    from zrt.training.models.comm import (
        collective_time_hierarchical, collective_time_multi_tier,
    )
    sys = _make_system()  # 2 tiers: intra (D=8), inter (unbounded)
    N = 32  # 4 nodes × 8
    S = 100 * 1024 * 1024
    c = Collective("ag", "AG", "DP", S, "op")
    t_legacy = collective_time_hierarchical(c, N, sys)
    t_multi = collective_time_multi_tier(c, list(range(N)), sys)
    assert t_multi == pytest.approx(t_legacy, rel=1e-12), (
        f"multi-tier ({t_multi*1e9:.1f}ns) must match legacy 2-tier "
        f"({t_legacy*1e9:.1f}ns) bit-for-bit"
    )


def test_multi_tier_ar_is_double_ag():
    """AR over multi-tier = RS_multi + AG_multi = 2× AG."""
    from zrt.training.models.comm import collective_time_multi_tier
    sys = _make_3tier_system(world_size=512)
    ranks = list(range(128))
    S = 32 * 1024 * 1024
    ag = Collective("ag", "AG", "DP", S, "op")
    ar = Collective("ar", "AR", "DP", S, "op")
    t_ag = collective_time_multi_tier(ag, ranks, sys)
    t_ar = collective_time_multi_tier(ar, ranks, sys)
    # AR = RS + AG and RS has same cost as AG → factor 2
    assert t_ar == pytest.approx(2 * t_ag, rel=1e-12)


def test_multi_tier_picks_outermost_link_for_p2p():
    """P2P doesn't hierarchically decompose — uses the outermost tier
    the group spans (the worst-case link)."""
    from zrt.training.models.comm import collective_time_multi_tier
    sys = _make_3tier_system(world_size=512)
    # Two ranks far apart → cross-rack → spine
    ranks = [0, 128]
    S = 1024 * 1024
    c = Collective("p2p", "P2P", "PP", S, "op")
    t = collective_time_multi_tier(c, ranks, sys)
    # Should match flat P2P over spine link
    spine_link = sys.interconnect.tiers[-1].link
    expected = spine_link.latency_us * 1e-6 + S / spine_link.effective_bw_bps(2)
    assert t == pytest.approx(expected, rel=0.01)


def test_total_comm_time_uses_multi_tier_when_3plus_tiers():
    """End-to-end: total_comm_time picks the multi-tier path automatically
    when the system has 3+ interconnect tiers."""
    from zrt.training.ir.training_graph import Graph
    from zrt.training.models.comm import total_comm_time
    from zrt.training.spec.strategy import Strategy

    sys = _make_3tier_system(world_size=512)
    # Strategy that produces a non-trivial DP group spanning the rack tier
    strategy = Strategy(tp=4, cp=4, ep=1, dp=8, pp=4, micro_batch=1, global_batch=8)
    # Tiny model — we only check the DP grad reduce gets priced > 0.
    from zrt.training.spec.model import ModelSpec, LayerKind
    model = ModelSpec(
        hidden=512, ffn=2048, num_heads=8, num_kv_heads=8,
        head_dim=64, vocab=1024, seq_len=128,
        layers=[LayerKind.DENSE] * 4,
    )
    graph = Graph(ops=[], collectives=[])
    result = total_comm_time(graph, model, sys, strategy)
    # DP grad reduce should be priced > 0 (8 ranks at stride tp*cp*pp = 64 →
    # group spans across racks, hits spine).
    assert "dp_grad_reduce" in result
    assert result["dp_grad_reduce"] > 0
