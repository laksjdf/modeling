"""EP tests — graph capture path, UT + E2E combined."""
from __future__ import annotations

import pytest

from python.zrt.ir.edge import Edge
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.ir.types import DType, TensorMeta

pytestmark = pytest.mark.ep

# ═══════════════════════════════════════════════════════════════════════════════
# constants
# ═══════════════════════════════════════════════════════════════════════════════

_EP, _TP = 8, 8
_NUM_EXPERTS, _MOE_ACTIVE = 384, 6
_HIDDEN, _SEQ_LEN, _BATCH = 7168, 128, 1

DT = DType.BF16


def _t(tid: str, shape: tuple[int, ...]) -> TensorMeta:
    return TensorMeta.from_shape_dtype(tid, shape, DT)


def _ut_ctx(ep: int = 1, tp: int = 1, num_experts: int = 0, moe_active: int = 1):
    try: from types import SimpleNamespace
    except ImportError: from argparse import Namespace as SimpleNamespace
    from python.zrt.transform import ParallelConfig, StreamConfig, TransformContext
    import python.zrt.hardware.registry as hw_registry
    hw = hw_registry.load("nvidia_h100_sxm")
    profile = SimpleNamespace(num_experts=num_experts, moe_active=moe_active) if num_experts > 0 else None
    return TransformContext(hw_spec=hw, parallel=ParallelConfig(tp=tp, ep=ep),
                           stream_config=StreamConfig(), profile=profile)


def _make_moe_graph() -> OpGraph:
    """2-layer MoE: router → expert → combine per layer."""
    nodes = {
        "r0": OpNode(id="r0", op_type="aten.mm.default", inputs=[_t("ri0",(1,128,4096))], outputs=[_t("ro0",(1,128,64))],
                     scope="model.layers.0.moe.gate", category="compute", layer="0"),
        "e0": OpNode(id="e0", op_type="aten.mm.default", inputs=[_t("ei0",(1,128,4096))], outputs=[_t("eo0",(1,128,4096))],
                     scope="model.layers.0.moe.experts.0.linear", category="compute", layer="0"),
        "c0": OpNode(id="c0", op_type="aten.add.default", inputs=[_t("ca0",(1,128,4096)),_t("cb0",(1,128,4096))],
                     outputs=[_t("co0",(1,128,4096))], scope="model.layers.0.moe.shared_experts", category="compute", layer="0"),
        "r1": OpNode(id="r1", op_type="aten.mm.default", inputs=[_t("ri1",(1,128,4096))], outputs=[_t("ro1",(1,128,64))],
                     scope="model.layers.1.moe.gate", category="compute", layer="1"),
        "e1": OpNode(id="e1", op_type="aten.mm.default", inputs=[_t("ei1",(1,128,4096))], outputs=[_t("eo1",(1,128,4096))],
                     scope="model.layers.1.moe.experts.3.linear", category="compute", layer="1"),
        "c1": OpNode(id="c1", op_type="aten.add.default", inputs=[_t("ca1",(1,128,4096)),_t("cb1",(1,128,4096))],
                     outputs=[_t("co1",(1,128,4096))], scope="model.layers.1.moe.shared_experts", category="compute", layer="1"),
    }
    edges = [Edge("r0",0,"e0",0,_t("re0",(1,128,4096))), Edge("e0",0,"c0",0,_t("ec0",(1,128,4096))),
             Edge("r1",0,"e1",0,_t("re1",(1,128,4096))), Edge("e1",0,"c1",0,_t("ec1",(1,128,4096)))]
    return OpGraph(name="moe_test", phase="prefill", nodes=nodes, edges=edges,
                   metadata={"seq_len":128, "hidden":4096, "num_experts":64})


# ═══════════════════════════════════════════════════════════════════════════════
#   E2E fixtures (use captured_model from conftest)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def ep8_estimate(captured_model):
    from python.zrt.transform.analysis import estimate_training_from_graphs
    import python.zrt.hardware.registry as hw_registry
    hw = hw_registry.load("nvidia_h100_sxm")
    return estimate_training_from_graphs(
        forward_graph=captured_model.graphs["train_forward"],
        backward_graph=captured_model.graphs["train_backward"],
        hw_spec=hw, tp=_TP, ep=_EP, hidden=_HIDDEN, num_layers=4, seq_len=_SEQ_LEN,
        batch_size=_BATCH, moe_total_experts=_NUM_EXPERTS, moe_active_experts=_MOE_ACTIVE,
        return_transformed=True,
    )

@pytest.fixture(scope="session")
def ep1_estimate(captured_model):
    from python.zrt.transform.analysis import estimate_training_from_graphs
    import python.zrt.hardware.registry as hw_registry
    hw = hw_registry.load("nvidia_h100_sxm")
    return estimate_training_from_graphs(
        forward_graph=captured_model.graphs["train_forward"],
        backward_graph=captured_model.graphs["train_backward"],
        hw_spec=hw, tp=_TP, ep=1, hidden=_HIDDEN, num_layers=4, seq_len=_SEQ_LEN,
        batch_size=_BATCH, moe_total_experts=_NUM_EXPERTS, moe_active_experts=_MOE_ACTIVE,
        return_transformed=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#   UT: _is_expert_scope
# ═══════════════════════════════════════════════════════════════════════════════

from python.zrt.transform.parallel.expert_parallel import _is_expert_scope

@pytest.mark.parametrize("scope", [
    "model.layers.0.moe.experts.0.linear", "EXPERT_0_FFN", ".expertS[0].", "moe_ffn.down_proj",
])
def test_is_expert_scope_matches(scope: str):
    assert _is_expert_scope(scope) is True

@pytest.mark.parametrize("scope", [
    "model.layers.0.self_attn.q_proj", "lm_head", "",
])
def test_is_expert_scope_rejects(scope: str):
    assert _is_expert_scope(scope) is False


# ═══════════════════════════════════════════════════════════════════════════════
#   UT: ExpertParallelPass
# ═══════════════════════════════════════════════════════════════════════════════

from python.zrt.transform import ExpertParallelPass

class TestExpertParallelPassUT:

    def test_noop_when_no_profile(self):
        g = _make_moe_graph()
        assert ExpertParallelPass().run(g, _ut_ctx(ep=8, num_experts=0)) is g

    def test_experts_per_rank_min_1(self):
        out = ExpertParallelPass().run(_make_moe_graph(), _ut_ctx(ep=128, num_experts=64))
        assert out.nodes["e0"].annotations["ep_experts_local"] == 1

    def test_does_not_mutate_input(self):
        g = _make_moe_graph()
        original = dict(g.nodes["e0"].annotations)
        ExpertParallelPass().run(g, _ut_ctx(ep=8, num_experts=64))
        assert g.nodes["e0"].annotations == original


# ═══════════════════════════════════════════════════════════════════════════════
#   UT: CommInserterPass
# ═══════════════════════════════════════════════════════════════════════════════

from python.zrt.transform import CommInserterPass

class TestCommInserterUT:

    @staticmethod
    def _run(graph, ctx):
        g1 = ExpertParallelPass().run(graph, ctx)
        return CommInserterPass().run(g1, ctx)

    def test_dispatch_before_expert(self):
        result = self._run(_make_moe_graph(), _ut_ctx(ep=8, num_experts=64))
        assert any("dispatch" in p for p in result.predecessors("e0"))

    def test_combine_after_expert(self):
        result = self._run(_make_moe_graph(), _ut_ctx(ep=8, num_experts=64))
        assert any("combine" in s for s in result.successors("e0"))

    def test_no_double_insertion(self):
        ctx = _ut_ctx(ep=8, num_experts=64)
        g2 = self._run(_make_moe_graph(), ctx)
        count1 = len([n for n in g2.nodes.values() if n.op_type == "comm.all_to_all"])
        count2 = len([n for n in CommInserterPass().run(g2, ctx).nodes.values()
                      if n.op_type == "comm.all_to_all"])
        assert count2 == count1


# ═══════════════════════════════════════════════════════════════════════════════
#   UT: EP + TP coexistence
# ═══════════════════════════════════════════════════════════════════════════════

def test_ep_with_tp_inserts_both_comm_types():
    from python.zrt.transform import TensorParallelPass
    g = OpGraph(
        name="tp_ep", phase="prefill",
        nodes={
            "q": OpNode(id="q", op_type="aten.mm.default", inputs=[_t("qi",(128,4096))], outputs=[_t("qo",(128,4096))],
                        scope="model.layers.0.self_attn.q_proj", category="compute", layer="0"),
            "e": OpNode(id="e", op_type="aten.mm.default", inputs=[_t("ei",(128,4096))], outputs=[_t("eo",(128,4096))],
                        scope="model.layers.0.moe.experts.0.down_proj", category="compute", layer="0"),
        },
        edges=[Edge("q",0,"e",0,_t("qe",(128,4096)))],
        metadata={"seq_len":128,"hidden":4096,"num_experts":64},
    )
    ctx = _ut_ctx(ep=8, tp=4, num_experts=64, moe_active=8)
    g1 = TensorParallelPass().run(g, ctx)
    g2 = ExpertParallelPass().run(g1, ctx)
    g3 = CommInserterPass().run(g2, ctx)
    comm_ops = {n.op_type for n in g3.comm_nodes()}
    assert "comm.all_reduce" in comm_ops
    assert "comm.all_to_all" in comm_ops


# ═══════════════════════════════════════════════════════════════════════════════
#   E2E: full CLI path on DSv4
# ═══════════════════════════════════════════════════════════════════════════════

class TestEPE2E:

    # ── sanity ────────────────────────────────────────────────────────────

    def test_capture_succeeded(self, captured_model):
        assert captured_model.graphs["train_forward"].num_nodes() > 0
        assert captured_model.graphs["train_backward"].num_nodes() > 0

    def test_unified_graph_returned(self, ep8_estimate):
        _, _, t = ep8_estimate
        u = t["unified"]; assert u is not None and u.num_nodes() > 0

    # ── A2A placement ─────────────────────────────────────────────────────

    def test_a2a_roles(self, ep8_estimate):
        _, _, t = ep8_estimate
        a2a = [n for n in t["unified"].nodes.values() if n.op_type == "comm.all_to_all"]
        assert len(a2a) > 0 and {n.attrs.get("role") for n in a2a} == {"dispatch","combine"}

    def test_a2a_tensor_ids_semantic(self, ep8_estimate):
        _, _, t = ep8_estimate
        for n in t["unified"].nodes.values():
            if n.op_type != "comm.all_to_all": continue
            kw = "dispatch" if n.attrs.get("role") == "dispatch" else "combine"
            assert any(kw in x.id.lower() for x in n.inputs + n.outputs)

    def test_a2a_msg_bytes_formula(self, ep8_estimate):
        _, _, t = ep8_estimate
        u = t["unified"]
        s, h = u.metadata.get("seq_len",_SEQ_LEN), u.metadata.get("hidden",_HIDDEN)
        expected = _BATCH * s * h * _MOE_ACTIVE * 2 // _EP
        for n in u.nodes.values():
            if n.op_type == "comm.all_to_all": assert n.attrs["msg_bytes"] == expected

    def test_shared_expert_not_epoch_annotated(self, ep8_estimate):
        _, _, t = ep8_estimate
        for n in t["unified"].nodes.values():
            if "shared_expert" in n.scope.lower():
                assert "ep_needs_a2a" not in n.annotations

    def test_router_not_epoch_annotated(self, ep8_estimate):
        _, _, t = ep8_estimate
        for n in t["unified"].nodes.values():
            if "gate" in n.scope.lower() and "moe" in n.scope.lower():
                assert "ep_needs_a2a" not in n.annotations

    # ── A2A future (XFAIL) ────────────────────────────────────────────────

    @pytest.mark.xfail(reason="A2A missing phase='both'", strict=True)
    def test_a2a_phase_both(self, ep8_estimate):
        _, _, t = ep8_estimate
        for n in t["unified"].nodes.values():
            if n.op_type == "comm.all_to_all": assert n.annotations.get("phase") == "both"

    @pytest.mark.xfail(reason="A2A missing overlap_target", strict=True)
    def test_a2a_overlap_target(self, ep8_estimate):
        _, _, t = ep8_estimate
        for n in t["unified"].nodes.values():
            if n.op_type == "comm.all_to_all" and n.attrs.get("role") == "dispatch":
                assert "overlap_target" in n.annotations

    @pytest.mark.xfail(reason="A2A msg_bytes should use hidden/TP", strict=True)
    def test_a2a_msg_bytes_accounts_for_tp(self, ep8_estimate):
        _, _, t = ep8_estimate
        u = t["unified"]
        s, h = u.metadata.get("seq_len",_SEQ_LEN), u.metadata.get("hidden",_HIDDEN) // _TP
        expected = _BATCH * s * h * _MOE_ACTIVE * 2 // _EP
        for n in u.nodes.values():
            if n.op_type == "comm.all_to_all": assert n.attrs["msg_bytes"] == expected

    # ── GroupedMM fusion (XFAIL) ──────────────────────────────────────────

    def _mm_assert_grouped(self, g):
        grp = [n for n in g.nodes.values() if n.op_type == "GroupedMatMul"]
        if not grp: pytest.xfail("GroupedMM fusion not yet implemented")
        return grp

    @pytest.mark.xfail(reason="GroupedMM fusion NYI", strict=True)
    def test_grouped_mm_exists(self, ep8_estimate):
        _, _, t = ep8_estimate
        assert len(self._mm_assert_grouped(t["unified"])) >= 2

    @pytest.mark.xfail(reason="GroupedMM fusion NYI", strict=True)
    def test_grouped_mm_per_moe_layer(self, ep8_estimate):
        _, _, t = ep8_estimate
        g = self._mm_assert_grouped(t["unified"])
        moe_layers = len([n for n in t["unified"].nodes.values()
                          if n.op_type == "comm.all_to_all" and n.attrs.get("role") == "dispatch"])
        assert len(g) == moe_layers * 2

    @pytest.mark.xfail(reason="GroupedMM fusion NYI", strict=True)
    def test_grouped_mm_replaces_all_routed_expert_ops(self, ep8_estimate):
        _, _, t = ep8_estimate
        self._mm_assert_grouped(t["unified"])
        for n in t["unified"].nodes.values():
            if "shared_expert" in n.scope.lower(): continue
            if "expert" in n.scope.lower() or "experts." in n.scope.lower():
                assert n.op_type == "GroupedMatMul"

    @pytest.mark.xfail(reason="GroupedMM fusion NYI", strict=True)
    def test_grouped_mm_group_count(self, ep8_estimate):
        _, _, t = ep8_estimate
        for n in self._mm_assert_grouped(t["unified"]):
            assert n.inputs[0].shape[0] == _NUM_EXPERTS // _EP

    @pytest.mark.xfail(reason="GroupedMM fusion NYI", strict=True)
    def test_grouped_mm_token_count(self, ep8_estimate):
        _, _, t = ep8_estimate
        for n in self._mm_assert_grouped(t["unified"]):
            assert n.inputs[0].shape[1] == _BATCH * _SEQ_LEN * _MOE_ACTIVE // _EP

    @pytest.mark.xfail(reason="GroupedMM fusion NYI", strict=True)
    def test_shared_expert_not_grouped(self, ep8_estimate):
        _, _, t = ep8_estimate
        self._mm_assert_grouped(t["unified"])
        for n in t["unified"].nodes.values():
            if "shared_expert" in n.scope.lower(): assert "Grouped" not in n.op_type

    @pytest.mark.xfail(reason="GroupedMM fusion NYI", strict=True)
    def test_swiglu_between_grouped_mm(self, ep8_estimate):
        _, _, t = ep8_estimate
        g = self._mm_assert_grouped(t["unified"])
        u = t["unified"]
        for gm in g:
            if any("silu" in u.nodes[s].op_type.lower() for s in u.successors(gm.id)): return
        pytest.fail("No SwiGLU after any GroupedMM")

    # ── shape ─────────────────────────────────────────────────────────────

    def test_hidden_dim_unchanged(self, ep8_estimate, ep1_estimate):
        _, _, t8 = ep8_estimate; _, _, t1 = ep1_estimate
        u8, u1 = t8["unified"], t1["unified"]
        i8 = {n.id for n in u8.nodes.values() if not n.is_comm}
        i1 = {n.id for n in u1.nodes.values() if not n.is_comm}
        for nid in i8 & i1:
            n8, n1 = u8.nodes[nid], u1.nodes[nid]
            for i in range(min(len(n8.outputs), len(n1.outputs))):
                assert n8.outputs[i].shape == n1.outputs[i].shape

    def test_dispatch_input_shape(self, ep8_estimate):
        _, _, t = ep8_estimate
        for n in t["unified"].nodes.values():
            if n.op_type == "comm.all_to_all" and n.attrs.get("role") == "dispatch":
                assert n.inputs[0].shape == (_BATCH, _SEQ_LEN, _HIDDEN)

    def test_combine_output_shape(self, ep8_estimate):
        _, _, t = ep8_estimate
        for n in t["unified"].nodes.values():
            if n.op_type == "comm.all_to_all" and n.attrs.get("role") == "combine":
                assert n.outputs[0].shape == (_BATCH, _SEQ_LEN, _HIDDEN)

    @pytest.mark.xfail(reason="dispatch token count not reduced by EP", strict=True)
    def test_dispatch_token_count_reduced(self, ep8_estimate):
        _, _, t = ep8_estimate
        expected = _BATCH * _SEQ_LEN * _MOE_ACTIVE // _EP
        for n in t["unified"].nodes.values():
            if n.op_type == "comm.all_to_all" and n.attrs.get("role") == "dispatch":
                assert n.outputs[0].shape[1] == expected

    @pytest.mark.xfail(reason="router output dim not reduced", strict=True)
    def test_router_output_reduced(self, ep8_estimate):
        _, _, t = ep8_estimate
        expected = _NUM_EXPERTS // _EP
        for n in t["unified"].nodes.values():
            for o in n.outputs:
                if o.shape[-1] == _NUM_EXPERTS or o.shape[-1] == expected:
                    assert o.shape[-1] == expected

    # ── edge cases ────────────────────────────────────────────────────────

    def test_ep1_no_ep(self, ep1_estimate):
        _, _, t = ep1_estimate
        for n in t["unified"].nodes.values(): assert "ep_needs_a2a" not in n.annotations
        assert len([n for n in t["unified"].nodes.values() if n.op_type == "comm.all_to_all"]) == 0

    def test_compute_nodes_same(self, ep8_estimate, ep1_estimate):
        _, _, t8 = ep8_estimate; _, _, t1 = ep1_estimate
        assert len(t8["unified"].compute_nodes()) == len(t1["unified"].compute_nodes())

    def test_node_count_diff_is_a2a_count(self, ep8_estimate, ep1_estimate):
        _, _, t8 = ep8_estimate; _, _, t1 = ep1_estimate
        diff = t8["unified"].num_nodes() - t1["unified"].num_nodes()
        a2a = len([n for n in t8["unified"].nodes.values() if n.op_type == "comm.all_to_all"])
        assert diff == a2a and diff > 0

    def test_a2a_symmetry(self, ep8_estimate):
        _, _, t = ep8_estimate
        d = [n for n in t["unified"].nodes.values()
             if n.op_type == "comm.all_to_all" and n.attrs.get("role") == "dispatch"]
        c = [n for n in t["unified"].nodes.values()
             if n.op_type == "comm.all_to_all" and n.attrs.get("role") == "combine"]
        assert len(d) == len(c)
        for dp, cp in zip(d, c): assert dp.attrs["msg_bytes"] == cp.attrs["msg_bytes"]

    def test_all_nodes_have_flops_and_stream(self, ep8_estimate):
        _, _, t = ep8_estimate
        for n in t["unified"].nodes.values():
            assert "flops" in n.annotations and n.annotations["flops"] >= 0
            assert "stream_id" in n.annotations

    def test_dag(self, ep8_estimate):
        _, _, t = ep8_estimate; u = t["unified"]
        assert len(u.topo_sort()) == u.num_nodes()

    # ── TrainingReport ────────────────────────────────────────────────────

    def test_report_metrics(self, ep8_estimate):
        r, _, _ = ep8_estimate
        assert r.step_time_ms > 0 and 0 < r.mfu <= 1.0 and r.training_flops > 0

    def test_step_time_differs(self, ep8_estimate, ep1_estimate):
        assert ep8_estimate[0].step_time_ms != ep1_estimate[0].step_time_ms

    def test_context(self, ep8_estimate):
        _, ctx, _ = ep8_estimate
        assert ctx.parallel.ep == _EP and ctx.parallel.tp == _TP
        assert ctx.profile.num_experts == _NUM_EXPERTS
