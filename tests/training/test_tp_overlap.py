"""Tests for GEMM-bound TP overlap in compose/stage.py.

Mirrors the EP wave-overlap model: AG/RS comm is split into K=4 waves, each
overlapped with the corresponding matmul wave. CoC -> K=4 wave-overlap;
MC2 -> assume comm fully covered by GEMM (bounded by gemm time, not flat 0).
"""
from __future__ import annotations

import pytest

from zrt.training.compose.stage import _tp_gemm_time, _wave_overlap_saved
from zrt.training.ir.training_graph import Op, Tensor
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import PPSched, Strategy, TPOverlap
from zrt.training.spec.system import GPU, SystemSpec
from zrt.hardware.spec import InterconnectSpec, LinkSpec


def _system():
    return SystemSpec(
        gpu=GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80, hbm_bw_gbps=3350),
        host_mem_gb=256,
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0,
                               topology="all_to_all", num_devices=8),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=400, latency_us=5.0,
                               topology="fat_tree"),
        ),
        nodes=1, gpus_per_node=8,
    )


def _system_with(nodes: int = 1, gpus_per_node: int = 8):
    return SystemSpec(
        gpu=GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80, hbm_bw_gbps=3350),
        host_mem_gb=256,
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0,
                               topology="all_to_all", num_devices=gpus_per_node),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=400, latency_us=5.0,
                               topology="fat_tree"),
        ),
        nodes=nodes, gpus_per_node=gpus_per_node,
    )


def _model(num_layers: int = 4):
    return ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=4096,
        layers=[LayerKind.DENSE] * num_layers,
    )


def _t(name: str, shape: tuple[int, ...], dtype: Dtype = Dtype.BF16) -> Tensor:
    """Build a Tensor with matching logical/local shapes."""
    return Tensor(name=name, shape_logical=shape, shape_local=shape,
                  dtype=dtype, is_activation=True, is_param=False)


def _matmul_op(name: str, m: int = 4096, n: int = 4096, k: int = 4096,
               layer_id: int = 0) -> Op:
    """Build a matmul Op consistent with the IR builders.

    Op cost is derived by `op_cost()` from meta {m, n, k} and tensor shapes,
    not from precomputed fwd_flops/bytes. We supply both so _matmul_cost
    finds an unambiguous shape.
    """
    return Op(
        name=name, kind="matmul",
        inputs=[_t("a", (m, k))],
        outputs=[_t("c", (m, n))],
        meta={"m": m, "n": n, "k": k},
        layer_id=layer_id, layer_kind=LayerKind.DENSE,
    )


class TestTPGemmTime:
    def test_overlappable_matmul_counted(self):
        """qkv/o_proj/up_proj/down_proj/gate_proj matmuls all count."""
        ops = [_matmul_op(name) for name in
               ["layers.0.attn.qkv", "layers.0.attn.o_proj",
                "layers.0.mlp.up_proj", "layers.0.mlp.gate_proj",
                "layers.0.mlp.down_proj"]]
        t = _tp_gemm_time(ops, _model(), _system(), gpu_name="h100", phase="fwd")
        assert t > 0, "Should count all five overlappable matmuls"

    def test_non_overlappable_matmul_ignored(self):
        """lm_head matmul has no TP AG/RS bracketing it — must be skipped."""
        ops = [_matmul_op("lm_head"), _matmul_op("layers.0.embedding")]
        t = _tp_gemm_time(ops, _model(), _system(), gpu_name="h100", phase="fwd")
        assert t == 0.0, "Non-bracketed matmuls should contribute 0"

    def test_non_matmul_ignored(self):
        """RMSNorm / Softmax / etc. are not matmul → don't count."""
        op = Op(
            name="layers.0.attn.softmax", kind="softmax",
            inputs=[_t("x", (4096, 32, 4096))],
            outputs=[_t("y", (4096, 32, 4096))],
            meta={"bytes_fwd": 4096 * 32 * 4096 * 2},
            layer_id=0, layer_kind=LayerKind.DENSE,
        )
        t = _tp_gemm_time([op], _model(), _system(), gpu_name="h100", phase="fwd")
        assert t == 0.0

    def test_bwd_phase_sums_dx_plus_dw(self):
        """Backward time = dx + dw for each overlappable matmul."""
        ops = [_matmul_op("layers.0.attn.qkv")]
        t_fwd = _tp_gemm_time(ops, _model(), _system(), gpu_name="h100", phase="fwd")
        t_bwd = _tp_gemm_time(ops, _model(), _system(), gpu_name="h100", phase="bwd")
        # dx and dw together are ~2× fwd time for symmetric matmul costs
        assert t_bwd > t_fwd, f"bwd ({t_bwd}) should exceed fwd ({t_fwd})"


class TestStageTimeTPOverlap:
    """Stage-level: verify tp_hidden is GEMM-bound, not a flat 90% of comm."""

    def _stage_with_tp(self, tp_overlap: TPOverlap):
        """Build a 1-layer stage with TP=4. Returns StageTime."""
        from zrt.training.compose.stage import stage_time
        from zrt.training.ir.builders import build_graph
        model = _model()
        system = _system()
        strategy = Strategy(
            tp=4, pp=1, dp=2, ep=1, cp=1,
            micro_batch=1, global_batch=1,
            tp_overlap=tp_overlap,
        )
        graph = build_graph(model, strategy)
        st = stage_time(graph.ops, graph.collectives, model, system, strategy)
        return st

    def test_coc_hides_less_than_full_90pct_when_gemm_bound(self):
        """CoC must reduce comm strictly but not to zero."""
        st_coc = self._stage_with_tp(TPOverlap.COC)
        st_none = self._stage_with_tp(TPOverlap.NONE)
        assert st_coc.comm_fwd > 0, "CoC should still expose some comm when GEMM-bound"
        assert st_coc.comm_fwd < st_none.comm_fwd, \
            "CoC should expose less than NONE"

    def test_mc2_zero_exposed_when_gemm_dominates(self):
        """MC2 fully hides comm only when GEMM ≥ comm. For dense H100 matmul
        with tp=4 the FFN matmul is much larger than the TP AG/RS, so
        MC2 → comm_fwd ≈ 0 from the TP side."""
        st = self._stage_with_tp(TPOverlap.MC2)
        # Allow tiny float residue (comm_fwd is in seconds; 1ms = 0.001)
        assert st.comm_fwd < 0.001, \
            f"MC2 should fully hide TP comm when GEMM-bound; got {st.comm_fwd}"

    def test_tp_hidden_reported_in_stage_time(self):
        """StageTime must expose tp_hidden so schedules.py can use it."""
        st = self._stage_with_tp(TPOverlap.COC)
        assert hasattr(st, "tp_hidden"), "StageTime must have tp_hidden field"
        assert st.tp_hidden > 0, "CoC should hide some TP comm"

    def test_none_overlap_tp_hidden_zero(self):
        """NONE overlap → no hiding."""
        st = self._stage_with_tp(TPOverlap.NONE)
        assert st.tp_hidden == 0.0


class TestSchedulesTPHidden:
    """schedules.py: tp_hidden must come from StageTime.tp_hidden scaled to
    critical path — the same code path used by ep_hidden.

    Before the fix: tp_hidden was derived through two inconsistent branches
    (one for CoC, one for MC2), producing different scaling than the EP path.
    """

    def _run_estimate(self, tp_overlap: TPOverlap):
        from zrt.training.search.estimator import estimate
        model = _model()
        system = _system()
        strategy = Strategy(
            tp=4, pp=1, dp=2, ep=1, cp=1,
            micro_batch=1, global_batch=2,
            tp_overlap=tp_overlap,
        )
        return estimate(model, system, strategy)

    def test_coc_tp_hidden_nonzero(self):
        """CoC produces nonzero tp_hidden in the final step report."""
        r = self._run_estimate(TPOverlap.COC)
        assert r.tp_hidden_ms > 0, f"CoC should hide TP comm; got {r.tp_hidden_ms}"

    def test_mc2_tp_hidden_geq_coc(self):
        """MC2 hides at least as much as CoC (typically more)."""
        r_coc = self._run_estimate(TPOverlap.COC)
        r_mc2 = self._run_estimate(TPOverlap.MC2)
        assert r_mc2.tp_hidden_ms >= r_coc.tp_hidden_ms, \
            f"MC2 ({r_mc2.tp_hidden_ms}) should hide >= CoC ({r_coc.tp_hidden_ms})"

    def test_none_tp_hidden_zero(self):
        """No overlap → no hiding."""
        r = self._run_estimate(TPOverlap.NONE)
        assert r.tp_hidden_ms == 0.0

    def test_tp_exposed_plus_hidden_consistent(self):
        """Either tp_exposed_ms or tp_hidden_ms must be positive when TP overlap is active."""
        r = self._run_estimate(TPOverlap.COC)
        assert r.tp_exposed_ms + r.tp_hidden_ms > 0, \
            f"Expected nonzero TP comm budget; got exposed={r.tp_exposed_ms}, hidden={r.tp_hidden_ms}"

    def _run_estimate_pp(self, tp_overlap: TPOverlap, pp: int, dp: int):
        """Variant that lets caller set pp/dp (for conservation testing)."""
        from zrt.training.search.estimator import estimate
        model = _model(num_layers=8)  # enough layers for pp=2/4
        system = _system_with(nodes=1, gpus_per_node=4 * pp * dp)
        strategy = Strategy(
            tp=4, pp=pp, dp=dp, ep=1, cp=1,
            micro_batch=1, global_batch=pp * dp,
            tp_overlap=tp_overlap,
            pp_schedule=PPSched.ONE_F_ONE_B,
        )
        return estimate(model, system, strategy)

    @pytest.mark.parametrize("overlap", [TPOverlap.COC, TPOverlap.MC2])
    def test_conservation_invariant_holds_with_pp(self, overlap):
        """Regression: exposed_comm_ms must equal Σ per-group _exposed_ms.

        Bug commit 2f6d328 broke this for pp>1 with TP overlap because
        tp_exposed_volume was computed from a different basis than the
        exposed_comm budget. Pin the invariant here so it cannot regress.
        """
        r = self._run_estimate_pp(overlap, pp=2, dp=1)
        total_parts = (r.tp_exposed_ms + r.cp_exposed_ms + r.ep_exposed_ms
                       + r.pp_exposed_ms + r.dp_exposed_ms)
        assert abs(r.exposed_comm_ms - total_parts) < 1e-6, (
            f"Conservation violated for tp_overlap={overlap.value}: "
            f"exposed_comm_ms={r.exposed_comm_ms} vs Σparts={total_parts} "
            f"(tp={r.tp_exposed_ms}, cp={r.cp_exposed_ms}, ep={r.ep_exposed_ms}, "
            f"pp={r.pp_exposed_ms}, dp={r.dp_exposed_ms})"
        )

    def test_pp_p2p_does_not_drop_tp_hidden(self):
        """Regression: schedules.py rebuilds StageTime when pp_p2p>0 (lines
        504-515). Before the fix this rebuild dropped tp_hidden / ep_hidden,
        causing the report to show TP hidden=0 even when stage-level overlap
        actually fired.

        Symptom: CoC and MC2 produce identical step.tp_hidden_ms == 0 when
        pp > 1, while step_time still differs (savings leak into ep_exposed).
        """
        r_none = self._run_estimate_pp(TPOverlap.NONE, pp=2, dp=1)
        r_coc = self._run_estimate_pp(TPOverlap.COC, pp=2, dp=1)
        r_mc2 = self._run_estimate_pp(TPOverlap.MC2, pp=2, dp=1)
        # With pp>1 and TP overlap, tp_hidden_ms must surface in the report
        assert r_coc.tp_hidden_ms > 0, (
            f"CoC at pp=2 must show nonzero tp_hidden; got {r_coc.tp_hidden_ms}"
        )
        assert r_mc2.tp_hidden_ms >= r_coc.tp_hidden_ms, (
            f"MC2 ({r_mc2.tp_hidden_ms}) must hide >= CoC ({r_coc.tp_hidden_ms})"
        )
        assert r_none.tp_hidden_ms == 0.0


class TestTPCollectivePhaseSplit:
    """Regression: pin the 50/50 fwd/bwd split for phase='both' TP collectives.

    All TP collectives inserted by ir/shard.py:_insert_tp_collectives use
    phase='both' (line 126/136/146/156). compose/stage.py allocates these
    50/50 to fwd/bwd. This is approximately correct for Megatron-SP but is
    an undocumented modeling assumption. A future change to fwd/bwd ratios
    must update this test together with the code.
    """

    def test_all_tp_collectives_are_phase_both(self):
        """The IR generates only phase='both' for TP collectives. If this
        invariant changes, stage.py's 50/50 split logic must be revisited
        together with this test."""
        from zrt.training.ir.builders import build_graph
        model = _model()
        strategy = Strategy(tp=4, pp=1, dp=1, ep=1, cp=1,
                            micro_batch=1, global_batch=1)
        graph = build_graph(model, strategy)
        tp_collectives = [c for c in graph.collectives if c.group == "TP"]
        assert len(tp_collectives) > 0, "Expected TP collectives when tp>1"
        for c in tp_collectives:
            assert c.phase == "both", (
                f"TP collective {c.name} has phase={c.phase!r}; if intentional, "
                f"update compose/stage.py (the 50/50 split) accordingly"
            )

    def test_stage_time_splits_tp_comm_50_50(self):
        """With phase='both' TP collectives, stage.py allocates 50% to fwd
        and 50% to bwd. If this changes (e.g., to account for dgrad asymmetry),
        update compose/stage.py AND this test together.

        With cp=1/ep=1/pp=1 and tp_overlap=NONE, all non-TP per-stage comm is
        zero, so any (comm_fwd - comm_bwd) delta must come from a non-50/50
        TP split. We assert the delta is within 1% of the larger value to
        allow for any matmul-tiered second-order effects.
        """
        from zrt.training.compose.stage import stage_time
        from zrt.training.ir.builders import build_graph
        from zrt.training.models.comm import collective_time, tier_for_group
        model = _model()
        system = _system()
        strategy = Strategy(tp=4, pp=1, dp=1, ep=1, cp=1,
                            micro_batch=1, global_batch=1,
                            tp_overlap=TPOverlap.NONE)
        graph = build_graph(model, strategy)
        st = stage_time(graph.ops, graph.collectives, model, system, strategy)
        raw_tp = sum(
            collective_time(c, strategy.tp,
                            tier_for_group(c.group, strategy.tp, system))
            for c in graph.collectives if c.group == "TP"
        )
        assert raw_tp > 0, "Need nonzero TP comm to test the split"
        max_side = max(st.comm_fwd, st.comm_bwd)
        if max_side > 0:
            assert abs(st.comm_fwd - st.comm_bwd) <= 0.01 * max_side, (
                f"fwd ({st.comm_fwd}) and bwd ({st.comm_bwd}) TP shares "
                f"should match within 1%"
            )
