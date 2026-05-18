"""Tests for EP routed-expert FFN compute scaling under uniform routing."""

from zrt.training.compose.stage import _ep_gemm_time
from zrt.training.ir.builders import build_graph
from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.strategy import Strategy
from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.spec.system import SystemSpec, GPU


def _make_system():
    return SystemSpec(
        gpu=GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80, hbm_bw_gbps=3350),
        host_mem_gb=256,
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0,
                                topology="all_to_all", num_devices=8),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=400, latency_us=2.0,
                                topology="fat_tree"),
        ),
        nodes=48, gpus_per_node=8,
    )


def _moe_model(num_experts: int = 384, top_k: int = 6, n_moe_layers: int = 4):
    return ModelSpec(
        hidden=7168, ffn=3072, num_heads=128, num_kv_heads=1,
        head_dim=512, vocab=129280, seq_len=4096,
        layers=[LayerKind.MOE] * n_moe_layers,
        num_experts=num_experts, moe_ffn=3072, top_k=top_k,
        n_shared_experts=1,
    )


class TestRoutedExpertComputeInvariantToEP:
    """Under uniform routing, per-rank routed_expert GEMM time should be
    roughly constant as EP scales (more local experts ↔ fewer ranks routing in,
    or vice versa — the product is invariant)."""

    def test_gemm_time_constant_across_ep(self):
        """ep=1 (no EP) vs ep=num_experts: per-rank routed_expert work equal."""
        model = _moe_model(num_experts=8, top_k=2, n_moe_layers=2)
        system = _make_system()
        # Build two strategies with same effective parallelism but different EP
        s_no_ep = Strategy(tp=1, pp=1, dp=1, ep=1, micro_batch=1, global_batch=1)
        s_full_ep = Strategy(tp=1, pp=1, dp=8, ep=8, micro_batch=1, global_batch=8)

        g_no = build_graph(model, s_no_ep)
        g_full = build_graph(model, s_full_ep)

        # Take all moe ops from layer 0
        ops_no = [op for op in g_no.ops if op.layer_id == 0]
        ops_full = [op for op in g_full.ops if op.layer_id == 0]

        t_no = _ep_gemm_time(ops_no, model, system, s_no_ep, "h100")
        t_full = _ep_gemm_time(ops_full, model, system, s_full_ep, "h100")

        # Should be within 10% — minor differences from rounding only.
        assert t_no > 0 and t_full > 0
        assert 0.9 < (t_full / t_no) < 1.1, (
            f"Per-rank routed_expert GEMM should be invariant to EP under uniform "
            f"routing; got ep=1 t={t_no*1e3:.2f}ms vs ep=8 t={t_full*1e3:.2f}ms "
            f"(ratio={t_full/t_no:.3f})"
        )

    def test_large_ep_gemm_not_collapsed(self):
        """Regression for DSV4-pro/EP=384 bug: routed_expert GEMM was 0.72ms per
        stage, should be O(100ms). Use scaled-down model to keep the test fast."""
        model = _moe_model(num_experts=64, top_k=6, n_moe_layers=8)
        system = _make_system()
        s = Strategy(tp=1, pp=1, dp=64, ep=64, micro_batch=1, global_batch=64)
        g = build_graph(model, s)
        ops = [op for op in g.ops if op.layer_id == 0]
        t = _ep_gemm_time(ops, model, system, s, "h100")
        # With top_k=6, seq=4096, hidden=7168, moe_ffn=3072 → per-token expert
        # work ≈ 6*7168*3072*6 = 8e8 FLOPs/token × 4096 tokens = 3.3e12 FLOPs.
        # At ~300 TFLOPS effective: ~11 ms per layer. Allow 1 ms lower bound to
        # detect collapse without over-fitting to FLOPs efficiency tables.
        assert t > 1e-3, (
            f"EP={s.ep} routed_expert GEMM collapsed to {t*1e3:.3f}ms; "
            f"expected O(10ms) per layer under uniform routing"
        )


class TestImbalanceStillApplied:
    """Load imbalance (ep_imbalance_factor) should still scale routed-expert
    compute in stage_time(). This asserts the imbalance path remains independent
    of the _apply_ep_sharding change.

    Note: in THIS codebase, stage.py's imbalance path applies
    `ep_imbalance_factor(num_experts, ep, topk)` (a stochastic balls-into-bins
    formula). The model-level `model.expert_imbalance` field is currently only
    surfaced in Excel export, not wired into the compose path. We therefore
    test that EP-driven imbalance still amplifies stage time by comparing
    ep=1 (no imbalance) vs ep>1 (with imbalance), after factoring out the
    raw EP scaling fix landed in this task.
    """

    def test_imbalance_factor_applied_when_ep_active(self):
        from zrt.training.compose.stage import stage_time, ep_imbalance_factor

        model = _moe_model(num_experts=8, top_k=2, n_moe_layers=2)
        system = _make_system()
        s_no_ep = Strategy(tp=1, pp=1, dp=1, ep=1, micro_batch=1, global_batch=1)
        s_ep = Strategy(tp=1, pp=1, dp=8, ep=8, micro_batch=1, global_batch=8)

        g_no = build_graph(model, s_no_ep)
        g_ep = build_graph(model, s_ep)

        st_no = stage_time(g_no.ops, g_no.collectives, model, system, s_no_ep)
        st_ep = stage_time(g_ep.ops, g_ep.collectives, model, system, s_ep)

        # ep=1 path: no imbalance applied. ep=8 path: imbalance amplifies the
        # EP-parallel fraction. After this task's fix the routed_expert raw
        # compute is invariant to EP, so any remaining difference between
        # st_ep.fwd and st_no.fwd should come from the imbalance multiplier.
        imb = ep_imbalance_factor(model.num_experts, s_ep.ep, model.top_k)
        assert imb > 1.0, f"sanity: imbalance factor should be > 1 for ep>1, got {imb}"
        assert st_ep.fwd > st_no.fwd, (
            f"EP imbalance should increase stage fwd time when ep>1; "
            f"got ep=1 fwd={st_no.fwd*1e3:.2f}ms vs ep=8 fwd={st_ep.fwd*1e3:.2f}ms "
            f"(imbalance factor={imb:.3f})"
        )
