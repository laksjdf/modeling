"""Tests for mixed-quant communication volume."""
import pytest

from zrt.training.models.comm import total_comm_time
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import OptKind, Strategy
from zrt.training.spec.system import GPU, SystemSpec
from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.ir.training_graph import Graph


def _make_system(dp_bw_gbps=900):
    gpu = GPU(name="h100", flops_bf16=989, flops_fp8=3958, flops_fp4=0,
              hbm_gb=80, hbm_bw_gbps=3350)
    link = LinkSpec(type="NVLink", bandwidth_gbps=dp_bw_gbps, latency_us=1,
                    topology="all_to_all", num_devices=16)
    return SystemSpec(gpu=gpu, host_mem_gb=2048,
                      interconnect=InterconnectSpec(intra_node=link, inter_node=link),
                      nodes=2, gpus_per_node=8)


def _moe_model():
    return ModelSpec(
        hidden=512, ffn=2048, num_heads=8, num_kv_heads=8, head_dim=64,
        vocab=4096, seq_len=128,
        layers=[LayerKind.MOE, LayerKind.MOE],
        num_experts=8, moe_ffn=1024, top_k=2,
    )


def test_dp_grad_reduce_smaller_under_ep_than_under_no_ep():
    """When EP > 1, routed expert grads are EP-local and excluded from DP AR."""
    g, sys_ = Graph(), _make_system()
    m = _moe_model()
    st_no_ep = Strategy(dp=2, ep=1, optimizer=OptKind.ADAM)
    st_ep    = Strategy(dp=2, ep=4, optimizer=OptKind.ADAM)
    t_no_ep = total_comm_time(g, m, sys_, st_no_ep)["dp_grad_reduce"]
    t_ep    = total_comm_time(g, m, sys_, st_ep)["dp_grad_reduce"]
    assert t_ep < t_no_ep, (
        f"DP AR with EP=4 should exclude expert grads -> smaller volume "
        f"({t_ep:.6f}s vs no-EP {t_no_ep:.6f}s)"
    )


def test_dp_grad_reduce_volume_unchanged_for_dense_model():
    g, sys_ = Graph(), _make_system()
    m_dense = ModelSpec(
        hidden=512, ffn=2048, num_heads=8, num_kv_heads=8, head_dim=64,
        vocab=4096, seq_len=128, layers=[LayerKind.DENSE, LayerKind.DENSE],
    )
    st_no_ep = Strategy(dp=2, ep=1, optimizer=OptKind.ADAM)
    st_ep_unused = Strategy(dp=2, ep=4, optimizer=OptKind.ADAM)  # ep ignored: no experts
    t1 = total_comm_time(g, m_dense, sys_, st_no_ep)["dp_grad_reduce"]
    t2 = total_comm_time(g, m_dense, sys_, st_ep_unused)["dp_grad_reduce"]
    assert t1 == pytest.approx(t2)
