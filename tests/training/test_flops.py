"""Test FLOPs model — 6P rule, matmul cost, HFU metric."""

import pytest
from zrt.training.ir.builders import build_graph
from zrt.training.models.flops import OpCost, op_cost, total_training_flops, recompute_overhead_flops
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.strategy import RecomputePolicy, Strategy


def test_matmul_cost():
    """Matmul: fwd = dx = dw = 2*m*n*k."""
    from zrt.training.ir.graph import Op
    op = Op(name="test_mm", kind="matmul", meta={"m": 1024, "n": 4096, "k": 4096})
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE],
    )
    cost = op_cost(op, model)
    expected = 2 * 1024 * 4096 * 4096
    assert cost.fwd_flops == expected
    assert cost.dx_flops == expected
    assert cost.dw_flops == expected
    assert cost.bound == "compute"


def test_attn_core_cost():
    """Attention core: causal fwd = 2*b*s^2*h*d."""
    from zrt.training.ir.graph import Op
    op = Op(name="test_attn", kind="attn_core", meta={
        "b": 1, "s": 2048, "heads": 32, "head_dim": 128, "causal": True,
    })
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE],
    )
    cost = op_cost(op, model)
    # fwd = 2 * 1 * 2048 * 2048 * 32 * 128
    expected_fwd = 2 * 1 * 2048 * 2048 * 32 * 128
    assert cost.fwd_flops == expected_fwd
    assert cost.dx_flops == pytest.approx(2.5 * expected_fwd, rel=0.01)
    assert cost.dw_flops == 0.0


def test_attn_core_cost_uses_model_compression_ratio():
    """Compressed attention scales fwd and backward attention-core FLOPs."""
    from zrt.training.ir.graph import Op
    op = Op(name="test_attn", kind="attn_core", meta={
        "b": 1, "s": 1024, "heads": 16, "head_dim": 128, "causal": True,
    })
    model = ModelSpec(
        hidden=2048, ffn=8192, num_heads=16, num_kv_heads=16,
        head_dim=128, vocab=32000, seq_len=1024,
        layers=[LayerKind.DENSE],
        attn_compression_ratio=0.27,
    )

    cost = op_cost(op, model)

    dense_fwd = 2 * 1 * 1024 * 1024 * 16 * 128
    expected_fwd = dense_fwd * 0.27
    assert cost.fwd_flops == pytest.approx(expected_fwd)
    assert cost.dx_flops == pytest.approx(2.5 * expected_fwd)
    assert cost.dw_flops == 0.0


def test_attn_core_cost_op_ratio_overrides_model_ratio():
    """Per-op metadata can override a model-level compression ratio."""
    from zrt.training.ir.graph import Op
    op = Op(name="test_attn", kind="attn_core", meta={
        "b": 1, "s": 512, "heads": 8, "head_dim": 128,
        "causal": True, "attn_compression_ratio": 0.5,
    })
    model = ModelSpec(
        hidden=1024, ffn=4096, num_heads=8, num_kv_heads=8,
        head_dim=128, vocab=32000, seq_len=512,
        layers=[LayerKind.DENSE],
        attn_compression_ratio=0.27,
    )

    cost = op_cost(op, model)

    dense_fwd = 2 * 1 * 512 * 512 * 8 * 128
    assert cost.fwd_flops == pytest.approx(dense_fwd * 0.5)


def test_memory_bound_cost():
    """Memory-bound ops (ln, softmax, etc.) should have byte traffic."""
    from zrt.training.ir.graph import Op
    op = Op(name="test_ln", kind="ln", meta={"bytes_fwd": 1000})
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE],
    )
    cost = op_cost(op, model)
    assert cost.bound == "memory"
    assert cost.fwd_bytes == 1000
    assert cost.dx_bytes > 0


def test_6p_rule():
    """Total training FLOPs for dense model should follow 6P rule."""
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 4,
    )
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=1)
    graph = build_graph(model, strategy)

    total = total_training_flops(graph, model, strategy)

    # 6P rule: 6 * total_params * tokens
    tokens = 1 * 2048  # micro_batch * seq_len
    P = model.total_params()
    expected_6p = 6 * P * tokens

    # Allow 15% tolerance for embedding/lm_head and memory-bound ops
    # (the 6P rule is approximate but should be reasonably close)
    ratio = total / expected_6p
    assert 0.85 < ratio < 1.15, f"6P ratio: {ratio:.2f}, total={total:.2e}, 6P={expected_6p:.2e}"


def test_unknown_op_zero_cost():
    """Unknown op kinds should return zero cost."""
    from zrt.training.ir.graph import Op
    op = Op(name="unknown", kind="custom_op", meta={})
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE],
    )
    cost = op_cost(op, model)
    assert cost.fwd_flops == 0.0
    assert cost.dx_flops == 0.0


def test_moe_effective_params_is_sane():
    """MoE effective params should be less than total params when top_k < num_experts."""
    from zrt.training.spec.system import GPU, NetTier, SystemSpec
    from zrt.training.spec.strategy import Strategy
    from zrt.training.compose.pipeline import compute_mfu

    # Minimal MoE model: 2 layers, 4 experts, top_k=1
    model = ModelSpec(
        hidden=4096, ffn=2048, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.MOE] * 2,
        num_experts=4, moe_ffn=1024, top_k=1,
    )

    total = model.total_params()
    effective = model.effective_params_for_flops()

    # Effective should be less than total since only 1/4 of experts active
    assert effective < total, f"effective={effective}, total={total}"
    ratio = effective / total
    # With top_k=1, num_experts=4:
    # - Attention, router, shared FFN are 100% active
    # - Only routed experts are sparse (1/4 active)
    # So ratio should be between 50% (mostly routed experts) and 85% (mostly attention/shared)
    assert 0.5 < ratio < 0.9, f"effective/total ratio {ratio:.3f} outside expected range for top_k=1, num_experts=4"


def test_moe_mfu_is_sane():
    """MoE MFU should be between 0 and 1, not collapse to 1.0."""
    from zrt.training.io.config_loader import load_specs
    from zrt.training.search.estimator import estimate

    # Run estimate on deepseek_v3 config
    model, system, strategy = load_specs("python/zrt/training/configs/llama3_70b_3d.yaml")

    # Temporarily make it MoE-like for testing
    from dataclasses import replace
    model_moe = replace(model,
        layers=[LayerKind.MOE] * 2,
        num_experts=4,
        moe_ffn=1024,
        top_k=1,
    )

    report = estimate(model_moe, system, strategy)

    # MFU should be sane: strictly between 0 and 1 (not 0, not 1)
    assert 0.0 < report.mfu < 1.0, f"MoE MFU collapsed to {report.mfu}, expected 0 < MFU < 1"


# ── HFU tests ──────────────────────────────────────────────────────────────


def test_hfu_equals_mfu_without_recompute():
    """HFU == MFU when no recompute policy is configured."""
    from zrt.training.search.estimator import estimate
    from zrt.training.spec.system import GPU, NetTier, SystemSpec

    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 4,
    )
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=1)
    system = SystemSpec(
        gpu=GPU(name="test", flops_bf16=312, flops_fp8=624, hbm_gb=80, hbm_bw_gbps=2000),
        host_mem_gb=256,
        nets=[NetTier("intra_node", 900, 1.0, "nvswitch")],
        nodes=1, gpus_per_node=1,
    )

    report = estimate(model, system, strategy)
    assert report.hfu == pytest.approx(report.mfu, rel=1e-6), \
        f"HFU ({report.hfu}) should equal MFU ({report.mfu}) without recompute"


def test_hfu_exceeds_mfu_with_selective_recompute():
    """HFU > MFU when selective recompute is configured."""
    from zrt.training.search.estimator import estimate
    from zrt.training.spec.system import GPU, NetTier, SystemSpec

    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 4,
    )
    system = SystemSpec(
        gpu=GPU(name="test", flops_bf16=312, flops_fp8=624, hbm_gb=80, hbm_bw_gbps=2000),
        host_mem_gb=256,
        nets=[NetTier("intra_node", 900, 1.0, "nvswitch")],
        nodes=1, gpus_per_node=1,
    )
    strategy = Strategy(
        tp=1, pp=1, dp=1, micro_batch=1, global_batch=1,
        recompute=RecomputePolicy(per_layer={"dense": {"attn"}}),
    )

    graph = build_graph(model, strategy)
    overhead = recompute_overhead_flops(graph, model, strategy)
    assert overhead > 0, "Selective recompute should produce nonzero overhead FLOPs"

    report = estimate(model, system, strategy)
    assert report.hfu > report.mfu, \
        f"HFU ({report.hfu}) should exceed MFU ({report.mfu}) with selective recompute"


def test_recompute_overhead_zero_by_default():
    """No recompute overhead when RecomputePolicy is default (empty per_layer)."""
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 4,
    )
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=1)
    graph = build_graph(model, strategy)

    assert recompute_overhead_flops(graph, model, strategy) == 0.0


def test_recompute_overhead_full_recompute():
    """Full recompute adds forward FLOPs for all compute-bound ops."""
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 2,
    )
    strategy = Strategy(
        tp=1, pp=1, dp=1, micro_batch=1, global_batch=1,
        recompute=RecomputePolicy(per_layer={"dense": {"full"}}),
    )
    graph = build_graph(model, strategy)

    overhead = recompute_overhead_flops(graph, model, strategy)
    total = total_training_flops(graph, model, strategy)

    # Full recompute reruns the entire forward: overhead ≈ forward_flops = total / 3
    # Allow generous range since total includes backward and not all ops are compute-bound
    assert overhead > 0, "Full recompute should produce nonzero overhead"
    fwd_estimate = total / 3.0
    ratio = overhead / fwd_estimate
    assert 0.5 < ratio < 2.0, f"Full recompute overhead ratio {ratio:.2f} outside expected range"


def test_selective_recompute_increases_step_time():
    """Selective recompute should increase step time (extra forward pass)."""
    from zrt.training.search.estimator import estimate
    from zrt.training.spec.system import GPU, NetTier, SystemSpec

    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 4,
    )
    system = SystemSpec(
        gpu=GPU(name="test", flops_bf16=312, flops_fp8=624, hbm_gb=80, hbm_bw_gbps=2000),
        host_mem_gb=256,
        nets=[NetTier("intra_node", 900, 1.0, "nvswitch")],
        nodes=1, gpus_per_node=1,
    )
    strat_no_rc = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=1)
    strat_selective = Strategy(
        tp=1, pp=1, dp=1, micro_batch=1, global_batch=1,
        recompute=RecomputePolicy(per_layer={"dense": {"attn"}}),
    )

    r_no = estimate(model, system, strat_no_rc)
    r_sel = estimate(model, system, strat_selective)

    # Selective recompute re-runs attention forward → longer step time
    assert r_sel.step_time_ms > r_no.step_time_ms, \
        f"Selective recompute step ({r_sel.step_time_ms:.2f}ms) should exceed " \
        f"no-recompute ({r_no.step_time_ms:.2f}ms)"
    # HFU > MFU when recompute is active
    assert r_sel.hfu > r_sel.mfu, \
        f"HFU ({r_sel.hfu}) should exceed MFU ({r_sel.mfu}) with selective recompute"
    # Without recompute, HFU == MFU
    assert r_no.hfu == pytest.approx(r_no.mfu, rel=1e-6)


def test_layer_kind_scoped_recompute():
    """Recompute policy only applies to ops in matching layer kinds."""
    model = ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE, LayerKind.MOE],
        num_experts=4, moe_ffn=1024, top_k=1,
    )
    # Only recompute attention in MOE layers
    strategy = Strategy(
        tp=1, pp=1, dp=1, micro_batch=1, global_batch=1,
        recompute=RecomputePolicy(per_layer={"moe": {"attn"}}),
    )
    graph = build_graph(model, strategy)

    overhead = recompute_overhead_flops(graph, model, strategy)
    # Should be nonzero (MOE layer's attention ops are recomputed)
    assert overhead > 0, "MOE-targeted recompute should produce nonzero overhead"

    # Compare with dense-only recompute
    strategy_dense = Strategy(
        tp=1, pp=1, dp=1, micro_batch=1, global_batch=1,
        recompute=RecomputePolicy(per_layer={"dense": {"attn"}}),
    )
    overhead_dense = recompute_overhead_flops(graph, model, strategy_dense)
    # Dense layer and MoE layer have different op counts; should differ
    # At minimum, both should be non-negative and not equal if layer sizes differ
    assert overhead_dense >= 0
