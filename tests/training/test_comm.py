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
    # alpha = 1us, beta = 1/(900e9/8) = 1/(112.5e9) ≈ 8.89e-12 s/byte
    # S = 1MB = 1048576 bytes
    expected = 1e-6 + 1048576 / (900e9 / 8)
    assert t == pytest.approx(expected, rel=0.01)


def test_ag_time():
    """AG with all_to_all topology: single-step (alpha + S/N * beta)."""
    link = _intra_link()
    N = 8
    S = 100 * 1024 * 1024  # 100 MB
    c = Collective("test_ag", "AG", "TP", S, "op1")
    t = collective_time(c, N, link)

    alpha = 1e-6
    beta = 1.0 / (900e9 / 8)
    # nvswitch uses single-step formula
    expected = alpha + (S / N) * beta
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
    """A2A on NVSwitch (full_connectivity): single-step (alpha + S/N * beta)."""
    link = _intra_link()
    N = 4
    S = 20 * 1024 * 1024
    c = Collective("test_a2a", "A2A", "EP", S, "op1")
    t = collective_time(c, N, link)

    alpha = 1e-6
    beta = 1.0 / (900e9 / 8)
    # full_connectivity topology uses single-step A2A
    expected = alpha + (S / N) * beta
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
    beta = 1.0 / (400e9 / 8)
    bandwidth = (N - 1) / N * S * beta
    # Latency: log2(384) ≈ 9 rounds × 2µs = 18 µs
    expected_max = 12 * 2e-6 + bandwidth  # allow ceil + small margin
    assert t < expected_max, (
        f"A2A on N=384 should use log2 latency, got {t*1e6:.1f}µs "
        f"(expected < {expected_max*1e6:.1f}µs)"
    )
    # Sanity: still > pure bandwidth
    assert t > bandwidth, "should have non-zero latency"


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
    beta = 1.0 / (900e9 / 8)
    expected_legacy = (N - 1) * (alpha + (S / N) * beta)
    assert t == pytest.approx(expected_legacy, rel=1e-3), (
        f"Small N should keep legacy formula; got {t} vs {expected_legacy}"
    )
