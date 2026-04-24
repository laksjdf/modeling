"""Anchor validation tests — compare estimated MFU against published benchmarks.

References:
- GPT-3 175B: Megatron-LM (Narayanan et al. 2021)
- Llama-3 70B: Meta training blog post
- DeepSeek-V3: Technical Report §5.4

CI asserts MFU error within 15% of published values.
"""

import pytest
from zrt.training.search.estimator import estimate
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import PPSched, Strategy
from zrt.training.spec.system import GPU, NetTier, SystemSpec


def _make_h100_system(nodes: int = 1, gpus_per_node: int = 8) -> SystemSpec:
    return SystemSpec(
        gpu=GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80, hbm_bw_gbps=3350),
        host_mem_gb=256,
        nets=[NetTier("intra_node", 900, 1.0, "nvswitch")],
        nodes=nodes,
        gpus_per_node=gpus_per_node,
    )


@pytest.fixture
def gpt3_175b() -> ModelSpec:
    """GPT-3 175B: 96 layers, 12288 hidden, 96 heads, 12288 head_dim."""
    return ModelSpec(
        hidden=12288,
        ffn=49152,
        num_heads=96,
        num_kv_heads=96,
        head_dim=128,
        vocab=50257,
        seq_len=2048,
        layers=[LayerKind.DENSE] * 96,
        param_dtype=Dtype.BF16,
    )


@pytest.fixture
def llama3_70b() -> ModelSpec:
    """Llama-3 70B: 80 layers, 8192 hidden, 64 heads, GQA (8 kv heads)."""
    return ModelSpec(
        hidden=8192,
        ffn=28672,
        num_heads=64,
        num_kv_heads=8,
        head_dim=128,
        vocab=128256,
        seq_len=8192,
        layers=[LayerKind.DENSE] * 80,
        param_dtype=Dtype.BF16,
    )


@pytest.fixture
def deepseek_v3() -> ModelSpec:
    """DeepSeek-V3: 61 layers (first 3 dense, 58 MoE), MoE architecture."""
    layers = [LayerKind.DENSE] * 3 + [LayerKind.MOE] * 58
    return ModelSpec(
        hidden=7168,
        ffn=18432,
        num_heads=128,
        num_kv_heads=128,
        head_dim=128,
        vocab=102400,
        seq_len=4096,
        layers=layers,
        num_experts=256,
        moe_ffn=2048,
        top_k=8,
        expert_imbalance=0.1,
        param_dtype=Dtype.BF16,
    )


ANCHOR_MFU_TARGETS = {
    "gpt3_175b_h100_8x8": {"target_mfu": 0.42, "tolerance": 0.15},
    "llama3_70b_h100_8x8": {"target_mfu": 0.45, "tolerance": 0.15},
    "deepseek_v3_h100_8x8": {"target_mfu": 0.55, "tolerance": 0.15},
}


@pytest.mark.anchor
def test_anchor_gpt3_175b_mfu(gpt3_175b):
    """GPT-3 175B on 64 H100s: MFU ~42% per Megatron-LM paper."""
    system = _make_h100_system(nodes=8, gpus_per_node=8)
    strategy = Strategy(
        tp=8,
        pp=8,
        dp=1,
        micro_batch=4,
        global_batch=512,
        pp_schedule=PPSched.ONE_F_ONE_B,
    )

    report = estimate(gpt3_175b, system, strategy)

    anchor = ANCHOR_MFU_TARGETS["gpt3_175b_h100_8x8"]
    target = anchor["target_mfu"]
    tol = anchor["tolerance"]

    lower = target * (1 - tol)
    upper = target * (1 + tol)

    assert lower <= report.mfu <= upper, (
        f"MFU {report.mfu:.2%} outside [{lower:.2%}, {upper:.2%}] "
        f"(target {target:.2%}, tolerance {tol:.0%})"
    )


@pytest.mark.anchor
def test_anchor_llama3_70b_mfu(llama3_70b):
    """Llama-3 70B on 64 H100s: MFU ~45% per Meta training logs."""
    system = _make_h100_system(nodes=8, gpus_per_node=8)
    strategy = Strategy(
        tp=8,
        pp=4,
        dp=2,
        micro_batch=2,
        global_batch=256,
        pp_schedule=PPSched.ONE_F_ONE_B,
    )

    report = estimate(llama3_70b, system, strategy)

    anchor = ANCHOR_MFU_TARGETS["llama3_70b_h100_8x8"]
    target = anchor["target_mfu"]
    tol = anchor["tolerance"]

    lower = target * (1 - tol)
    upper = target * (1 + tol)

    assert lower <= report.mfu <= upper, (
        f"MFU {report.mfu:.2%} outside [{lower:.2%}, {upper:.2%}] "
        f"(target {target:.2%}, tolerance {tol:.0%})"
    )


@pytest.mark.anchor
def test_anchor_deepseek_v3_mfu(deepseek_v3):
    """DeepSeek-V3 on 64 H100s: MFU ~55% per technical report."""
    system = _make_h100_system(nodes=8, gpus_per_node=8)
    strategy = Strategy(
        tp=8,
        pp=4,
        ep=8,
        dp=1,
        micro_batch=1,
        global_batch=256,
        pp_schedule=PPSched.DUALPIPE,
        dualbatch=True,
    )

    report = estimate(deepseek_v3, system, strategy)

    anchor = ANCHOR_MFU_TARGETS["deepseek_v3_h100_8x8"]
    target = anchor["target_mfu"]
    tol = anchor["tolerance"]

    lower = target * (1 - tol)
    upper = target * (1 + tol)

    assert lower <= report.mfu <= upper, (
        f"MFU {report.mfu:.2%} outside [{lower:.2%}, {upper:.2%}] "
        f"(target {target:.2%}, tolerance {tol:.0%})"
    )


@pytest.mark.anchor
def test_anchor_vpp_reduces_bubble(gpt3_175b):
    """VPP/Interleaved 1F1B should reduce bubble fraction."""
    system = _make_h100_system(nodes=8, gpus_per_node=8)

    s_1f1b = Strategy(
        tp=8, pp=8, dp=1, micro_batch=4, global_batch=512,
        pp_schedule=PPSched.ONE_F_ONE_B,
    )
    s_vpp = Strategy(
        tp=8, pp=8, dp=1, micro_batch=4, global_batch=512,
        pp_schedule=PPSched.INTERLEAVED,
        vpp_chunks=4,
    )

    r_1f1b = estimate(gpt3_175b, system, s_1f1b)
    r_vpp = estimate(gpt3_175b, system, s_vpp)

    assert r_vpp.per_stage[0].bubble_fraction < r_1f1b.per_stage[0].bubble_fraction, (
        f"VPP bubble {r_vpp.per_stage[0].bubble_fraction:.2%} should be < "
        f"1F1B bubble {r_1f1b.per_stage[0].bubble_fraction:.2%}"
    )


@pytest.mark.anchor
def test_anchor_dualpipe_bubble(deepseek_v3):
    """DualPipe bubble should be ~half of 1F1B."""
    system = _make_h100_system(nodes=8, gpus_per_node=8)

    s_1f1b = Strategy(
        tp=8, pp=4, ep=8, dp=1, micro_batch=1, global_batch=256,
        pp_schedule=PPSched.ONE_F_ONE_B,
    )
    s_dp = Strategy(
        tp=8, pp=4, ep=8, dp=1, micro_batch=1, global_batch=256,
        pp_schedule=PPSched.DUALPIPE,
        dualbatch=True,
    )

    r_1f1b = estimate(deepseek_v3, system, s_1f1b)
    r_dp = estimate(deepseek_v3, system, s_dp)

    assert r_dp.per_stage[0].bubble_fraction <= r_1f1b.per_stage[0].bubble_fraction * 0.6, (
        f"DualPipe bubble {r_dp.per_stage[0].bubble_fraction:.2%} should be "
        f"~half of 1F1B {r_1f1b.per_stage[0].bubble_fraction:.2%}"
    )


@pytest.mark.anchor
def test_anchor_ep_imbalance_effect(deepseek_v3):
    """EP imbalance should increase step time."""
    system = _make_h100_system(nodes=8, gpus_per_node=8)

    balanced = ModelSpec(
        hidden=7168,
        ffn=18432,
        num_heads=128,
        num_kv_heads=128,
        head_dim=128,
        vocab=102400,
        seq_len=4096,
        layers=[LayerKind.DENSE] * 3 + [LayerKind.MOE] * 58,
        num_experts=256,
        moe_ffn=2048,
        top_k=8,
        expert_imbalance=0.0,
        param_dtype=Dtype.BF16,
    )

    strategy = Strategy(
        tp=8, pp=4, ep=8, dp=1, micro_batch=1, global_batch=256,
        pp_schedule=PPSched.DUALPIPE,
    )

    r_balanced = estimate(balanced, system, strategy)
    r_imbalanced = estimate(deepseek_v3, system, strategy)

    assert r_imbalanced.step_time_ms > r_balanced.step_time_ms, (
        f"Imbalanced step {r_imbalanced.step_time_ms:.1f}ms should be > "
        f"balanced {r_balanced.step_time_ms:.1f}ms"
    )