"""Test communication model — alpha-beta collective costs."""

import pytest
from zrt.training.ir.graph import Collective
from zrt.training.models.comm import collective_time, tier_for_group, total_comm_time
from zrt.training.spec.system import NetTier, SystemSpec, GPU


def _intra_tier():
    return NetTier("intra_node", 900, 1.0, "nvswitch")


def _inter_tier():
    return NetTier("inter_node", 400, 5.0, "fattree")


def _make_system():
    return SystemSpec(
        gpu=GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80, hbm_bw_gbps=3350),
        host_mem_gb=256,
        nets=[_intra_tier(), _inter_tier()],
        nodes=4, gpus_per_node=8,
    )


def test_p2p_time():
    """P2P: alpha + S * beta."""
    tier = _intra_tier()
    c = Collective("test_p2p", "P2P", "PP", 1024 * 1024, "op1")  # 1 MB
    t = collective_time(c, 2, tier)
    assert t > 0
    # alpha = 1us, beta = 1/(900e9/8) = 1/(112.5e9) ≈ 8.89e-12 s/byte
    # S = 1MB = 1048576 bytes
    expected = 1e-6 + 1048576 / (900e9 / 8)
    assert t == pytest.approx(expected, rel=0.01)


def test_ag_time():
    """AG: (N-1) * (alpha + S/N * beta)."""
    tier = _intra_tier()
    N = 8
    S = 100 * 1024 * 1024  # 100 MB
    c = Collective("test_ag", "AG", "TP", S, "op1")
    t = collective_time(c, N, tier)

    alpha = 1e-6
    beta = 1.0 / (900e9 / 8)
    expected = (N - 1) * (alpha + (S / N) * beta)
    assert t == pytest.approx(expected, rel=0.01)


def test_ar_time_is_2x_ag():
    """AllReduce = 2 * AG time (ring algorithm)."""
    tier = _intra_tier()
    N = 8
    S = 50 * 1024 * 1024  # 50 MB
    c_ag = Collective("ag", "AG", "TP", S, "op1")
    c_ar = Collective("ar", "AR", "DP", S, "op1")

    t_ag = collective_time(c_ag, N, tier)
    t_ar = collective_time(c_ar, N, tier)

    assert t_ar == pytest.approx(2 * t_ag, rel=0.01)


def test_a2a_time():
    """A2A: (N-1) * (alpha + S/N * beta), same as AG."""
    tier = _intra_tier()
    N = 4
    S = 20 * 1024 * 1024
    c = Collective("test_a2a", "A2A", "EP", S, "op1")
    t = collective_time(c, N, tier)

    alpha = 1e-6
    beta = 1.0 / (900e9 / 8)
    expected = (N - 1) * (alpha + (S / N) * beta)
    assert t == pytest.approx(expected, rel=0.01)


def test_tier_selection_intra():
    """TP within a node should use intra_node tier."""
    system = _make_system()
    tier = tier_for_group("TP", 8, system)
    assert tier.scope == "intra_node"


def test_tier_selection_inter():
    """DP across nodes should use inter_node tier."""
    system = _make_system()
    tier = tier_for_group("DP", 32, system)
    assert tier.scope == "inter_node"
