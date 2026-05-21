"""Tests for the unified :class:`CommDomain` resolver.

Verifies:
  - Single source of truth: every comm pricing call site returns the
    same answer when given the same (system, strategy, collective).
  - Tier picks: each parallel kind lands on the expected interconnect
    tier on a 3-tier NVL576-like topology.
  - Search hot-path: the resolver is built once per ``estimate()`` and
    threaded through stage/total/optimizer paths.
  - 2-tier back-compat: bit-exact with the legacy size-only formula.
"""
from __future__ import annotations

import pytest

from zrt.hardware.spec import InterconnectSpec, LinkSpec, TopologyTier
from zrt.training.ir.training_graph import Collective, Graph
from zrt.training.models.comm import (
    collective_time,
    collective_time_hierarchical,
    collective_time_multi_tier,
    optimizer_comm_time,
    pp_p2p_time,
    total_comm_time,
)
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import OptKind, Strategy
from zrt.training.spec.system import GPU, SystemSpec
from zrt.training.topology import CommDomain, build_comm_domain, comm_domain_report


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

def _three_tier_system(world_size: int = 512) -> SystemSpec:
    tray = LinkSpec(type="NVLink5", bandwidth_gbps=1800, latency_us=0.5,
                    topology="all_to_all", num_devices=4, kb_efficiency=0.85)
    rack = LinkSpec(type="NVSwitch5", bandwidth_gbps=900, latency_us=1.0,
                    topology="all_to_all", num_devices=64, kb_efficiency=0.75)
    spine = LinkSpec(type="IB_XDR", bandwidth_gbps=400, latency_us=5.0,
                     topology="fat_tree", num_devices=0, kb_efficiency=0.6)
    ic = InterconnectSpec(tiers=[
        TopologyTier(name="tray", link=tray),
        TopologyTier(name="rack", link=rack),
        TopologyTier(name="spine", link=spine),
    ])
    return SystemSpec(
        gpu=GPU(name="b300", flops_bf16=5000, flops_fp8=10000,
                hbm_gb=288, hbm_bw_gbps=8000),
        host_mem_gb=512, interconnect=ic,
        nodes=world_size // 4, gpus_per_node=4,
        world_size_override=world_size,
    )


def _two_tier_system(world_size: int = 64) -> SystemSpec:
    intra = LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0,
                     topology="all_to_all", num_devices=8)
    inter = LinkSpec(type="IB", bandwidth_gbps=400, latency_us=5.0,
                     topology="fat_tree")
    return SystemSpec(
        gpu=GPU(name="h100", flops_bf16=989, flops_fp8=1979,
                hbm_gb=80, hbm_bw_gbps=3350),
        host_mem_gb=256,
        interconnect=InterconnectSpec(intra_node=intra, inter_node=inter),
        nodes=world_size // 8, gpus_per_node=8,
    )


# ─────────────────────────────────────────────────────────────────────
# Resolver basics
# ─────────────────────────────────────────────────────────────────────

def test_comm_domain_picks_innermost_tier_for_tp_fitting_one_tray():
    """tp=4 on a 4-GPU tray topology rides tier 0 (nvlink_tray)."""
    sys = _three_tier_system(world_size=512)
    s = Strategy(tp=4, cp=1, ep=1, dp=16, pp=8)
    d = CommDomain(system=sys, strategy=s)
    assert d.group_size("TP") == 4
    assert d.tier("TP") == 0
    assert d.link("TP").type == "NVLink5"


def test_comm_domain_report_rejects_empty_interconnect_tiers():
    sys = SystemSpec(
        gpu=GPU(name="test", flops_bf16=1, flops_fp8=2, hbm_gb=1, hbm_bw_gbps=1),
        host_mem_gb=1,
        interconnect=InterconnectSpec(tiers=[]),
        nodes=1,
        gpus_per_node=1,
    )

    with pytest.raises(ValueError, match="interconnect.*tiers"):
        comm_domain_report(sys, Strategy())


def test_comm_domain_bumps_tp_to_rack_when_crossing_trays():
    """tp=8 spans 2 trays → must use tier 1 (nvswitch_rack)."""
    sys = _three_tier_system(world_size=512)
    s = Strategy(tp=8, cp=1, ep=1, dp=8, pp=8)
    d = CommDomain(system=sys, strategy=s)
    assert d.tier("TP") == 1
    assert d.link("TP").type == "NVSwitch5"


def test_comm_domain_dp_lands_on_spine_when_spanning_racks():
    """Strided DP groups spanning many racks land on spine (tier 2)."""
    sys = _three_tier_system(world_size=512)
    # tp=8, cp=1, pp=2, dp=32 → DP groups at stride tp*cp=8, size 32
    # → span (32-1)*8+1 = 249 ranks → must straddle rack boundary
    s = Strategy(tp=8, cp=1, ep=1, dp=32, pp=2)
    d = CommDomain(system=sys, strategy=s)
    assert d.tier("DP") == 2  # spine


def test_comm_domain_time_matches_collective_time_multi_tier_directly():
    """domain.time(c) and collective_time_multi_tier(c, ranks, system)
    must return identical values for the canonical instance."""
    sys = _three_tier_system(world_size=512)
    s = Strategy(tp=4, cp=2, ep=1, dp=16, pp=4)
    d = CommDomain(system=sys, strategy=s)
    c = Collective(name="dp_ar", kind="AR", group="DP",
                   bytes_=64 * 1024 * 1024, inserted_after="end")
    t_domain = d.time(c)
    t_direct = collective_time_multi_tier(c, d.ranks("DP"), sys)
    assert t_domain == pytest.approx(t_direct, rel=1e-12)


def test_comm_domain_2tier_falls_back_to_legacy_size_only_path():
    """On a 2-tier system, ``domain.time(c)`` must equal the legacy
    ``collective_time(c, size, link)`` so anchor numbers don't shift."""
    sys = _two_tier_system(world_size=64)
    s = Strategy(tp=8, cp=1, ep=1, dp=8, pp=1)
    d = CommDomain(system=sys, strategy=s)
    c = Collective(name="tp_ag", kind="AG", group="TP",
                   bytes_=4 * 1024 * 1024, inserted_after="end")
    legacy = collective_time(c, 8, sys.interconnect.intra_node)
    assert d.time(c) == pytest.approx(legacy, rel=1e-12)


def test_comm_domain_zero_bytes_yields_zero_time():
    sys = _three_tier_system(512)
    d = CommDomain(system=sys, strategy=Strategy(tp=4, cp=1, ep=1, dp=16, pp=8))
    c = Collective(name="zero", kind="AG", group="DP", bytes_=0, inserted_after="end")
    assert d.time(c) == 0.0


def test_comm_domain_degenerate_kind_yields_zero_time():
    """For a parallel kind with size 1, all collectives have zero cost."""
    sys = _three_tier_system(64)
    s = Strategy(tp=1, cp=1, ep=1, dp=64, pp=1)
    d = CommDomain(system=sys, strategy=s)
    c = Collective(name="tp_noop", kind="AR", group="TP",
                   bytes_=1024 * 1024, inserted_after="end")
    assert d.time(c) == 0.0


def test_comm_domain_logs_when_multi_tier_falls_back_for_degenerate_groups(caplog):
    sys = _three_tier_system(64)
    # Product mismatch leaves ParallelGroups empty, but DP degree remains non-trivial.
    s = Strategy(tp=2, cp=1, ep=1, dp=8, pp=8)
    d = CommDomain(system=sys, strategy=s)
    c = Collective(name="dp_ag", kind="AG", group="DP", bytes_=1024, inserted_after="end")

    caplog.set_level("DEBUG", logger="zrt.training.topology.comm_domain")

    assert d.time(c) == pytest.approx(collective_time_hierarchical(c, 8, sys))
    assert any(
        "multi-tier skipped for DP (degenerate groups), using legacy" in r.message
        for r in caplog.records
    )


# ─────────────────────────────────────────────────────────────────────
# Unification: every call site agrees through the resolver
# ─────────────────────────────────────────────────────────────────────

def _tiny_model() -> ModelSpec:
    return ModelSpec(
        hidden=512, ffn=2048, num_heads=8, num_kv_heads=8,
        head_dim=64, vocab=1024, seq_len=128,
        layers=[LayerKind.DENSE] * 4,
    )


def test_total_comm_time_shares_domain_with_caller():
    """Passing a pre-built domain to total_comm_time must yield the
    same prices as letting it build its own."""
    sys = _three_tier_system(world_size=512)
    s = Strategy(tp=4, cp=2, ep=1, dp=16, pp=4, micro_batch=1, global_batch=64)
    model = _tiny_model()
    graph = Graph(ops=[], collectives=[])

    d = CommDomain(system=sys, strategy=s)
    res_with = total_comm_time(graph, model, sys, s, domain=d)
    res_without = total_comm_time(graph, model, sys, s)
    assert res_with == res_without


def test_optimizer_comm_time_accepts_shared_domain():
    """optimizer_comm_time must accept a shared CommDomain."""
    sys = _three_tier_system(world_size=512)
    s = Strategy(
        tp=4, cp=1, ep=1, dp=32, pp=4,
        zero_stage=1, optimizer=OptKind.MUON,
        micro_batch=1, global_batch=64,
    )
    model = _tiny_model()
    d = CommDomain(system=sys, strategy=s)
    res_with = optimizer_comm_time(model, sys, s, domain=d)
    res_without = optimizer_comm_time(model, sys, s)
    assert res_with == res_without
    assert res_with["muon_ag"] > 0  # ZeRO-1 Muon → real cost


def test_expert_dp_group_size_rejects_non_divisible_dp_ep():
    sys = _three_tier_system(world_size=512)
    s = Strategy(tp=1, cp=1, ep=384, dp=512, pp=1)
    d = CommDomain(system=sys, strategy=s)

    with pytest.raises(ValueError, match="dp must be divisible by ep"):
        d.group_size("EXPERT_DP")


def test_n_tier_time_uses_max_group_instance_not_rank0_only():
    """Some EXPERT_DP instances cross a slower tier even when rank0 does not."""
    tray = LinkSpec(type="NVLink5", bandwidth_gbps=1800, latency_us=0.5,
                    topology="all_to_all", num_devices=4, kb_efficiency=0.85)
    rack = LinkSpec(type="NVSwitch5", bandwidth_gbps=900, latency_us=1.0,
                    topology="all_to_all", num_devices=72, kb_efficiency=0.75)
    rail = LinkSpec(type="IB_XDR_rail", bandwidth_gbps=800, latency_us=2.0,
                    topology="fat_tree", num_devices=576, kb_efficiency=0.7)
    spine = LinkSpec(type="IB_XDR_spine", bandwidth_gbps=400, latency_us=5.0,
                     topology="fat_tree", num_devices=0, kb_efficiency=0.6)
    sys = SystemSpec(
        gpu=GPU(name="b300", flops_bf16=5000, flops_fp8=10000,
                hbm_gb=288, hbm_bw_gbps=8000),
        host_mem_gb=512,
        interconnect=InterconnectSpec(tiers=[
            TopologyTier(name="tray", link=tray),
            TopologyTier(name="rack", link=rack),
            TopologyTier(name="rail", link=rail),
            TopologyTier(name="spine", link=spine),
        ]),
        nodes=32,
        gpus_per_node=4,
        world_size_override=128,
    )
    s = Strategy(tp=4, cp=2, ep=2, dp=8, pp=2)
    d = CommDomain(system=sys, strategy=s)
    c = Collective("x", "AG", "EXPERT_DP", 100 * 1024 * 1024, "op")

    rank0_time = collective_time_multi_tier(c, d.ranks("EXPERT_DP"), sys)
    expected_max = max(
        collective_time_multi_tier(c, ranks, sys)
        for ranks in d.groups.expert_dp_groups
    )

    assert expected_max > rank0_time
    assert d.time(c) == pytest.approx(expected_max, rel=1e-12)


def test_pp_p2p_time_accepts_shared_domain():
    sys = _three_tier_system(world_size=512)
    s = Strategy(tp=4, cp=2, ep=1, dp=16, pp=4, micro_batch=1)
    model = _tiny_model()
    d = CommDomain(system=sys, strategy=s)
    t_with = pp_p2p_time(model, sys, s, domain=d)
    t_without = pp_p2p_time(model, sys, s)
    assert t_with == pytest.approx(t_without, rel=1e-12)
    assert t_with > 0


# ─────────────────────────────────────────────────────────────────────
# Search hot-path
# ─────────────────────────────────────────────────────────────────────

def test_pipeline_step_time_builds_one_domain_per_call(monkeypatch):
    """In the estimate path, CommDomain construction must be O(1) per
    estimate() call — not per stage or per collective."""
    from zrt.training.compose import schedules

    sys = _three_tier_system(world_size=512)
    s = Strategy(tp=4, cp=2, ep=1, dp=16, pp=4, micro_batch=1, global_batch=64)
    model = _tiny_model()
    from zrt.training.ir.builders import build_graph
    graph = build_graph(model, s)

    constructions: list[int] = []
    real_init = CommDomain.__init__

    def counting_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        constructions.append(1)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(CommDomain, "__init__", counting_init)
    schedules.pipeline_step_time(graph, model, sys, s)
    # Strict: exactly one domain constructed for the whole estimate() call.
    # If this assertion fails after a future refactor, every per-stage /
    # per-collective rebuild would silently inflate search latency.
    assert len(constructions) == 1, (
        f"expected exactly 1 CommDomain per estimate() call, got "
        f"{len(constructions)} — someone is rebuilding the resolver "
        f"inside the per-stage loop"
    )


def test_search_path_tier_picks_visible_through_summary():
    """Tier picks are inspectable for debugging via domain.summary()."""
    sys = _three_tier_system(world_size=512)
    s = Strategy(tp=4, cp=2, ep=1, dp=16, pp=4)
    d = build_comm_domain(sys, s)
    summary = d.summary()
    assert summary["TP"]["tier_name"] == "tray"
    assert summary["TP"]["size"] == 4
    assert summary["DP"]["tier_name"] in ("rack", "spine")
    assert summary["PP"]["size"] == 2


# ─────────────────────────────────────────────────────────────────────
# Bit-exact 2-tier back-compat
# ─────────────────────────────────────────────────────────────────────

def test_2tier_total_comm_time_unchanged_with_or_without_domain():
    """On a 2-tier system, total_comm_time must be byte-identical
    whether or not a CommDomain is passed — this is the anchor-safety
    contract for the migration."""
    sys = _two_tier_system(world_size=64)
    s = Strategy(tp=8, cp=1, ep=1, dp=8, pp=1, micro_batch=1, global_batch=8)
    model = _tiny_model()
    graph = Graph(ops=[], collectives=[])
    res_a = total_comm_time(graph, model, sys, s)
    d = CommDomain(system=sys, strategy=s)
    res_b = total_comm_time(graph, model, sys, s, domain=d)
    assert res_a == res_b


def test_routed_muon_uses_expert_dp_group_not_ep():
    """Regression guard: routed-Muon AG/RS must price over the
    expert-DP group (size dp/ep, ranks orthogonal to EP within a DP
    block), NOT over the EP group (size ep). This is the Megatron-Core
    distributed-optimizer convention; getting it wrong silently scales
    routed-Muon comm cost by ep/(dp/ep) which can be 2-4× off on
    typical DeepSeek configurations.
    """
    from zrt.training.spec.strategy import MuonConfig

    sys = _two_tier_system(world_size=64)
    # ep=2, dp=8 → EP group size 2, expert-DP group size 4
    s = Strategy(
        tp=1, cp=1, ep=2, dp=8, pp=8,
        zero_stage=1, optimizer=OptKind.MUON,
        muon_config=MuonConfig(rotation=True),
        micro_batch=1, global_batch=64,
    )
    d = CommDomain(system=sys, strategy=s)
    # EP group: 2 consecutive ranks within a DP block
    assert d.group_size("EP") == 2
    # Expert-DP group: 4 ranks, ORTHOGONAL partition (every 2nd rank
    # of one DP block — sharing the same intra-EP-block position)
    assert d.group_size("EXPERT_DP") == 4
    edp_ranks = d.ranks("EXPERT_DP")
    assert len(edp_ranks) == 4
    # The expert-DP group must NOT be a subset of any EP group
    # (orthogonality contract).
    ep_ranks = d.ranks("EP")
    assert not set(edp_ranks).issubset(set(ep_ranks))

    # Cost comparison: AG over expert-DP > AG over EP (more ranks, more
    # bandwidth term). If routed-Muon mistakenly used EP it would be
    # under-priced.
    from zrt.training.ir.training_graph import Collective
    S = 100 * 1024 * 1024
    t_edp = d.time(Collective("x", "AG", "EXPERT_DP", S, "op"))
    t_ep = d.time(Collective("y", "AG", "EP", S, "op"))
    assert t_edp > t_ep, (
        f"AG over expert-DP ({t_edp*1e6:.1f}µs) must be > AG over EP "
        f"({t_ep*1e6:.1f}µs) — routed-Muon under-priced if reversed"
    )


def test_expert_dp_group_is_orthogonal_partition_of_dp_by_ep():
    """For ep=2, dp=4: EP partitions DP into 2 blocks of 2; expert-DP
    is the orthogonal partition into 2 groups of 2 (one rank from each
    EP block)."""
    sys = _three_tier_system(64)
    s = Strategy(tp=2, cp=1, ep=2, dp=4, pp=8)
    d = CommDomain(system=sys, strategy=s)
    # Verify orthogonality on every DP group, not just the first.
    for dp_group in d.groups.dp_groups:
        # Each DP group splits into dp/ep = 2 EP blocks of 2 ranks.
        ep_blocks_in_dp = [g for g in d.groups.ep_groups
                           if set(g).issubset(set(dp_group))]
        assert len(ep_blocks_in_dp) == 2
        # And dp/ep = 2 expert-DP groups, each with one rank from each EP block.
        edp_in_dp = [g for g in d.groups.expert_dp_groups
                     if set(g).issubset(set(dp_group))]
        assert len(edp_in_dp) == 2
        for edp in edp_in_dp:
            # Every expert-DP group has exactly one rank in each EP block.
            for ep_blk in ep_blocks_in_dp:
                assert sum(1 for r in edp if r in ep_blk) == 1


def test_2tier_pp_p2p_unchanged():
    """PP P2P on a 2-tier YAML must hit the same intra-vs-inter branch
    as the legacy heuristic — confirms the resolver's pp_p2p_link()
    fall-back path is wired correctly."""
    sys = _two_tier_system(world_size=64)
    model = _tiny_model()
    # Strategy where adjacent PP stages span exactly the node boundary.
    s = Strategy(tp=4, cp=1, ep=1, dp=8, pp=2, micro_batch=1)
    t = pp_p2p_time(model, sys, s)
    # Legacy: tp*cp*2 = 8 ≤ intra_domain (8) → intra-node link
    intra = sys.interconnect.intra_node
    expected_bw = intra.effective_bw_bps(2)
    act_bytes = 1 * 128 * 512 * model.act_dtype.bytes // 4  # mb*seq*hidden*dtype/tp
    expected = intra.latency_us * 1e-6 + act_bytes / expected_bw
    assert t == pytest.approx(expected, rel=1e-9)
