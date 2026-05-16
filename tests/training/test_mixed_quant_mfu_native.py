"""Tests for mfu_native metric."""
import pytest

from zrt.training.compose.schedules import compute_mfu_native, compute_mfu
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import OptKind, Strategy
from zrt.training.spec.system import GPU, SystemSpec
from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.ir.training_graph import Graph, Op, Tensor


def _make_system():
    gpu = GPU(name="h100", flops_bf16=989, flops_fp8=3958, flops_fp4=0,
              hbm_gb=80, hbm_bw_gbps=3350)
    link = LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1,
                    topology="all_to_all", num_devices=8)
    return SystemSpec(gpu=gpu, host_mem_gb=2048,
                      interconnect=InterconnectSpec(intra_node=link, inter_node=link),
                      nodes=1, gpus_per_node=8)


def _t(name, n):
    return Tensor(name=name, shape_logical=(n,), shape_local=(n,),
                  dtype=Dtype.BF16, is_activation=True)


def test_compute_mfu_native_returns_zero_for_empty_graph():
    g = Graph()
    sys_ = _make_system()
    m = ModelSpec(hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
                  vocab=1000, seq_len=64, layers=[LayerKind.DENSE])
    st = Strategy(optimizer=OptKind.ADAM)
    out = compute_mfu_native(m, st, sys_, step_time=0.1, graph=g)
    assert out == 0.0


def test_compute_mfu_native_returns_zero_for_zero_step_time():
    g = Graph()
    sys_ = _make_system()
    m = ModelSpec(hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
                  vocab=1000, seq_len=64, layers=[LayerKind.DENSE])
    st = Strategy(optimizer=OptKind.ADAM)
    out = compute_mfu_native(m, st, sys_, step_time=0.0, graph=g)
    assert out == 0.0


def test_step_result_has_mfu_native_field():
    """StepResult dataclass has the new mfu_native field with default 0.0."""
    from zrt.training.compose.schedules import StepResult
    sr = StepResult()
    assert sr.mfu_native == 0.0
