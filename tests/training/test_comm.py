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
    """A2A: (N-1) * (alpha + S/N * beta), same as AG."""
    link = _intra_link()
    N = 4
    S = 20 * 1024 * 1024
    c = Collective("test_a2a", "A2A", "EP", S, "op1")
    t = collective_time(c, N, link)

    alpha = 1e-6
    beta = 1.0 / (900e9 / 8)
    expected = (N - 1) * (alpha + (S / N) * beta)
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
