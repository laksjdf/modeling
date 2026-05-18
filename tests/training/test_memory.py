"""Test memory model — ZeRO stages, activation scaling."""

import pytest
from zrt.training.ir.builders import build_graph
from zrt.training.models.memory import memory_breakdown
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.strategy import Strategy
from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.spec.system import SystemSpec, GPU


def _make_system():
    return SystemSpec(
        gpu=GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80, hbm_bw_gbps=3350),
        host_mem_gb=256,
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0, topology="all_to_all", num_devices=8),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=400, latency_us=5.0, topology="fat_tree"),
        ),
        nodes=1, gpus_per_node=8,
    )


def _make_model():
    return ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.DENSE] * 4,
    )


def test_zero_0_no_sharding():
    """ZeRO-0: all optimizer state on each rank."""
    model = _make_model()
    system = _make_system()
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, zero_stage=0)
    graph = build_graph(model, strategy)
    mem = memory_breakdown(graph, model, system, strategy)

    assert mem.opt_state > 0
    assert mem.weights > 0
    assert mem.grads > 0


def test_zero_1_divides_opt_state():
    """ZeRO-1: optimizer state divided by DP."""
    model = _make_model()
    system = _make_system()
    strategy_z0 = Strategy(tp=1, pp=1, dp=4, micro_batch=1, zero_stage=0)
    strategy_z1 = Strategy(tp=1, pp=1, dp=4, micro_batch=1, zero_stage=1)

    graph = build_graph(model, strategy_z0)
    mem_z0 = memory_breakdown(graph, model, system, strategy_z0)
    graph1 = build_graph(model, strategy_z1)
    mem_z1 = memory_breakdown(graph1, model, system, strategy_z1)

    # ZeRO-1 should reduce opt_state by ~4x
    ratio = mem_z1.opt_state / mem_z0.opt_state if mem_z0.opt_state > 0 else 0
    assert ratio == pytest.approx(0.25, abs=0.01), f"ZeRO-1 opt_state ratio: {ratio}"


def test_zero_2_divides_grads_and_opt():
    """ZeRO-2: optimizer state + grads divided by DP."""
    model = _make_model()
    system = _make_system()
    strategy_z0 = Strategy(tp=1, pp=1, dp=4, micro_batch=1, zero_stage=0)
    strategy_z2 = Strategy(tp=1, pp=1, dp=4, micro_batch=1, zero_stage=2)

    graph_z0 = build_graph(model, strategy_z0)
    mem_z0 = memory_breakdown(graph_z0, model, system, strategy_z0)
    graph_z2 = build_graph(model, strategy_z2)
    mem_z2 = memory_breakdown(graph_z2, model, system, strategy_z2)

    # ZeRO-2 should reduce opt_state and grads by ~4x
    opt_ratio = mem_z2.opt_state / mem_z0.opt_state if mem_z0.opt_state > 0 else 0
    grad_ratio = mem_z2.grads / mem_z0.grads if mem_z0.grads > 0 else 0
    assert opt_ratio == pytest.approx(0.25, abs=0.01), f"ZeRO-2 opt_state ratio: {opt_ratio}"
    assert grad_ratio == pytest.approx(0.25, abs=0.01), f"ZeRO-2 grads ratio: {grad_ratio}"


def test_tp_reduces_weights():
    """TP should shard weights across ranks."""
    model = _make_model()
    system = _make_system()

    strategy_1 = Strategy(tp=1, pp=1, dp=1, micro_batch=1)
    strategy_2 = Strategy(tp=2, pp=1, dp=1, micro_batch=1)

    graph1 = build_graph(model, strategy_1)
    graph2 = build_graph(model, strategy_2)
    mem1 = memory_breakdown(graph1, model, system, strategy_1)
    mem2 = memory_breakdown(graph2, model, system, strategy_2)

    # TP=2 should halve weights
    ratio = mem2.weights / mem1.weights if mem1.weights > 0 else 0
    assert ratio == pytest.approx(0.5, abs=0.01)


def test_memory_total_positive():
    """Total memory should be positive and reasonable."""
    model = _make_model()
    system = _make_system()
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1)
    graph = build_graph(model, strategy)
    mem = memory_breakdown(graph, model, system, strategy)

    assert mem.total > 0
    gb = mem.to_gb()
    assert gb["total_gb"] > 0
    assert gb["total_gb"] < system.gpu.hbm_gb  # should fit in HBM for small model


def test_activation_memory_scales_with_microbatch():
    """Activation memory should scale linearly with micro_batch."""
    model = _make_model()
    system = _make_system()

    s1 = Strategy(tp=1, pp=1, dp=1, micro_batch=1)
    s2 = Strategy(tp=1, pp=1, dp=1, micro_batch=2)

    g1 = build_graph(model, s1)
    g2 = build_graph(model, s2)
    m1 = memory_breakdown(g1, model, system, s1)
    m2 = memory_breakdown(g2, model, system, s2)

    assert m2.activations == pytest.approx(m1.activations * 2, rel=0.01)


# ============================================================================
# Recompute / Korthikanti-formula corrections
# ============================================================================

from zrt.training.spec.strategy import RecomputePolicy


def test_activation_memory_includes_attention_scores_term():
    """Korthikanti: long-seq activation must include 5·a·s²·bytes term.

    Without the scores term, a 2048×4096 dense model with bf16 has
    activation ≈ s·h·bytes·coeff = 2048*4096*2*10 ≈ 168 MB/layer.
    With scores: + 5·32·2048²·2 = 1.34 GB/layer.
    """
    model = _make_model()  # hidden=4096, num_heads=32, seq_len=2048, 4 dense layers
    system = _make_system()
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1,
                        recompute=RecomputePolicy())

    graph = build_graph(model, strategy)
    mem = memory_breakdown(graph, model, system, strategy)

    # Per-layer scores term: 5·a·s²·bytes = 5·32·2048²·2 ≈ 1.34 GB
    per_layer_scores = 5 * model.num_heads * model.seq_len * model.seq_len * model.act_dtype.bytes
    n_dense_layers = sum(1 for lk in model.layers if lk == LayerKind.DENSE)
    expected_min = per_layer_scores * n_dense_layers
    assert mem.activations >= expected_min, (
        f"activation memory missing s²·a term: got {mem.activations}, "
        f"expected ≥ {expected_min}"
    )


def test_attn_recompute_eliminates_scores_term():
    """When attn recompute is set, the s²·a term must drop."""
    model = _make_model()
    system = _make_system()
    strat_off = Strategy(tp=1, pp=1, dp=1, micro_batch=1,
                         recompute=RecomputePolicy())
    strat_on = Strategy(tp=1, pp=1, dp=1, micro_batch=1,
                        recompute=RecomputePolicy(per_layer={"dense": {"attn_core"}}))

    g_off = build_graph(model, strat_off)
    g_on = build_graph(model, strat_on)
    m_off = memory_breakdown(g_off, model, system, strat_off)
    m_on = memory_breakdown(g_on, model, system, strat_on)

    per_layer_scores = 5 * model.num_heads * model.seq_len * model.seq_len * model.act_dtype.bytes
    n_dense_layers = sum(1 for lk in model.layers if lk == LayerKind.DENSE)
    expected_scores_total = per_layer_scores * n_dense_layers

    saved = m_off.activations - m_on.activations
    assert saved >= expected_scores_total * 0.9, (
        f"attn_core recompute should drop ≥ Σ s²·a (={expected_scores_total}), "
        f"got saved={saved}"
    )


# ============================================================================
# PP in-flight schedule-aware sizing
# ============================================================================

from zrt.training.spec.strategy import PPSched


def _act_only(model, strategy):
    """Run memory_breakdown and return just the activations field."""
    graph = build_graph(model, strategy)
    return memory_breakdown(graph, model, _make_system(), strategy).activations


def _strat_pp(pp, sched, vpp=1):
    return Strategy(
        tp=1, cp=1, pp=pp, ep=1, dp=1,
        micro_batch=1, global_batch=pp * 4,
        pp_schedule=sched, vpp_chunks=vpp,
        recompute=RecomputePolicy(),
    )


def test_1f1b_in_flight_at_least_pp_minus_one():
    """1F1B worst-rank activation ≈ pp microbatches in flight, not pp/2.
    For pp=4, in-flight should be 4 (rank-0 warmup peak), not 2.
    """
    model = _make_model()
    a_pp1 = _act_only(model, _strat_pp(1, PPSched.ONE_F_ONE_B))   # baseline 1 mb
    a_pp4 = _act_only(model, _strat_pp(4, PPSched.ONE_F_ONE_B))
    # Old formula: pp//2 = 2 ⇒ ratio 2.0. New formula: pp = 4 ⇒ ratio ~4.0
    # (Note: per-rank graph still covers all layers; a_pp4 / a_pp1 isolates
    # the in-flight multiplier.)
    assert a_pp4 >= a_pp1 * 3.5, (
        f"1F1B with pp=4 should multiply activations by ≥ pp-1 (=3); "
        f"got ratio = {a_pp4 / a_pp1:.2f}"
    )


def test_interleaved_in_flight_scales_with_vpp():
    """Interleaved (VPP) in-flight ≈ pp · (vpp+1)/2 — vpp=2 must use more
    memory than vpp=1.
    """
    model = _make_model()
    a_v1 = _act_only(model, _strat_pp(4, PPSched.INTERLEAVED, vpp=1))
    a_v2 = _act_only(model, _strat_pp(4, PPSched.INTERLEAVED, vpp=2))
    # vpp=1 → factor pp·2/2 = 4 ; vpp=2 → factor pp·3/2 = 6. Ratio ≈ 1.5.
    assert a_v2 >= a_v1 * 1.3, (
        f"Interleaved vpp=2 should bump activations by ~1.5×; "
        f"got ratio = {a_v2 / a_v1:.2f}"
    )


def test_dualpipev_in_flight_at_least_1f1b():
    """DualPipeV holds two-direction chunks per rank, so its activation
    footprint is at least the 1F1B worst-rank value (no smaller).
    """
    model = _make_model()
    a_1f1b = _act_only(model, _strat_pp(4, PPSched.ONE_F_ONE_B))
    a_dpv = _act_only(model, _strat_pp(4, PPSched.DUALPIPE_V))
    assert a_dpv >= a_1f1b * 0.95, (
        f"DualPipeV in-flight ≥ 1F1B worst-rank; got "
        f"DPV={a_dpv}, 1F1B={a_1f1b}"
    )


# ============================================================================
# Phase-aware peak memory (sum != peak; opt_state and activations
# do not coexist).
# ============================================================================


def test_phase_peak_is_max_not_sum():
    """MemBreakdown must expose phase peaks where:
      peak_forward   = weights + activations + comm_buffers
      peak_backward  = weights + activations + grads + comm_buffers
      peak_optimizer = weights + grads + opt_state
      peak_overall   = max of the three.

    For ZeRO=0 with non-trivial grads/opt_state/activations, peak_overall
    must be strictly less than the algebraic sum (.total).
    """
    model = _make_model()
    system = _make_system()
    strategy = Strategy(
        tp=1, pp=1, dp=1, micro_batch=1, global_batch=1,
        zero_stage=0, recompute=RecomputePolicy(),
    )
    graph = build_graph(model, strategy)
    mb = memory_breakdown(graph, model, system, strategy)

    # Phase peaks must equal the documented compositions.
    assert mb.peak_forward == mb.weights + mb.activations + mb.comm_buffers
    assert mb.peak_backward == (
        mb.weights + mb.activations + mb.grads + mb.comm_buffers
    )
    assert mb.peak_optimizer == mb.weights + mb.grads + mb.opt_state

    # peak_overall = max(...)
    expected_peak = max(mb.peak_forward, mb.peak_backward, mb.peak_optimizer)
    assert mb.peak_overall == expected_peak

    # And: peak < component sum (activations and opt_state do not coexist).
    assert mb.grads > 0 and mb.opt_state > 0 and mb.activations > 0
    assert mb.peak_overall < mb.total, (
        f"peak ({mb.peak_overall}) must be < component sum ({mb.total}) — "
        f"activations and opt_state should not coexist in any phase"
    )


def test_to_gb_includes_peak_keys():
    """to_gb() must expose peak_gb and the three phase peaks so the
    Excel/HTML/search-CSV consumers can render them without reaching into
    private fields."""
    model = _make_model()
    system = _make_system()
    strategy = Strategy(
        tp=1, pp=1, dp=1, micro_batch=1, global_batch=1,
        recompute=RecomputePolicy(),
    )
    graph = build_graph(model, strategy)
    gb = memory_breakdown(graph, model, system, strategy).to_gb()
    for key in ("peak_gb", "peak_forward_gb", "peak_backward_gb",
                "peak_optimizer_gb"):
        assert key in gb, f"to_gb() missing {key!r}; got keys {sorted(gb)}"
    # Sanity: peak_gb < total_gb.
    assert gb["peak_gb"] < gb["total_gb"]
