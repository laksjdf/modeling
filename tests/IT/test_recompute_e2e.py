"""Recompute E2E tests — full CLI path on DSv4 model.

Validates the final pipeline output: annotations, FLOPs, timing,
memory, and report metrics across none/full/selective policies.

Matches::

    python -m python.zrt --model-id hf_models/deepseek_v4 --layers 4
        --train --hw nvidia_h100_sxm --tp 1
        --recompute-policy none|full|selective
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.recompute

_HW, _TP, _EP = "nvidia_h100_sxm", 1, 1
_HIDDEN, _SEQ, _LAYERS, _BATCH = 7168, 128, 4, 1


# ═══════════════════════════════════════════════════════════════════════════════
# fixtures
# ═══════════════════════════════════════════════════════════════════════════════

def _run_estimate(policy, captured_model):
    from python.zrt.transform.analysis import estimate_training_from_graphs
    import python.zrt.hardware.registry as hw_registry
    hw = hw_registry.load(_HW)
    return estimate_training_from_graphs(
        forward_graph=captured_model.graphs["train_forward"],
        backward_graph=captured_model.graphs["train_backward"],
        hw_spec=hw, tp=_TP, ep=_EP, hidden=_HIDDEN, num_layers=_LAYERS,
        seq_len=_SEQ, batch_size=_BATCH,
        recompute_policy=policy, return_transformed=True,
    )


@pytest.fixture(scope="module")
def none_all(captured_model):
    return _run_estimate("none", captured_model)


@pytest.fixture(scope="module")
def full_all(captured_model):
    return _run_estimate("full", captured_model)


@pytest.fixture(scope="module")
def sel_all(captured_model):
    return _run_estimate("selective", captured_model)


# ═══════════════════════════════════════════════════════════════════════════════
# tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecomputeE2E:

    # ── annotation ─────────────────────────────────────────────────────────

    def test_none_no_annotations(self, none_all):
        _, _, t = none_all
        rc = [n for n in t["unified"].nodes.values() if n.annotations.get("recompute")]
        assert len(rc) == 0

    def test_full_all_fwd_annotated(self, full_all):
        _, _, t = full_all
        for n in t["unified"].nodes.values():
            phase = n.annotations.get("phase", "fwd")
            if phase not in ("bwd", "backward", "train_backward"):
                assert n.annotations.get("recompute") is True, \
                    f"{n.id} not annotated under full"

    def test_selective_annotates_subset(self, full_all, sel_all):
        _, _, tf = full_all; _, _, ts = sel_all
        cf = sum(1 for n in tf["unified"].nodes.values() if n.annotations.get("recompute"))
        cs = sum(1 for n in ts["unified"].nodes.values() if n.annotations.get("recompute"))
        assert 0 < cs < cf, f"sel={cs} should be between 0 and full={cf}"

    def test_bwd_nodes_excluded(self, full_all):
        _, _, t = full_all
        for n in t["unified"].nodes.values():
            phase = n.annotations.get("phase", "fwd")
            if phase in ("bwd", "backward", "train_backward"):
                assert not n.annotations.get("recompute"), \
                    f"{n.id} bwd node annotated"

    # ── shape ─────────────────────────────────────────────────────────────

    def test_shapes_unchanged(self, none_all, full_all, sel_all):
        ref = {n.id: [t.shape for t in n.outputs] for n in none_all[2]["unified"].nodes.values()}
        for tag, (_, _, t) in ("full", full_all), ("sel", sel_all):
            u = t["unified"]
            for nid, shapes in ref.items():
                if nid in u.nodes:
                    for i, s in enumerate(shapes):
                        if i < len(u.nodes[nid].outputs):
                            assert u.nodes[nid].outputs[i].shape == s, \
                                f"policy={tag} {nid}[{i}] changed"

    # ── FLOPs ─────────────────────────────────────────────────────────────

    def test_recompute_nodes_2x(self, full_all):
        _, _, t = full_all
        for n in t["unified"].nodes.values():
            flops = n.annotations.get("flops", 0)
            ff = n.annotations.get("flops_fwd", 0)
            if n.annotations.get("recompute") and flops > 0:
                assert ff == flops * 2, f"{n.id}: flops_fwd={ff} != 2x{flops}"

    def test_non_recompute_nodes_1x(self, none_all):
        _, _, t = none_all
        for n in t["unified"].nodes.values():
            flops = n.annotations.get("flops", 0)
            ff = n.annotations.get("flops_fwd", 0)
            if flops > 0:
                assert ff == flops, f"{n.id}: flops_fwd={ff} != flops={flops}"

    def test_metadata_has_recompute_flops(self, full_all, none_all):
        u_full = full_all[2]["unified"]
        u_none = none_all[2]["unified"]
        assert u_full.metadata.get("recompute_flops", 0) > 0
        assert u_none.metadata.get("recompute_flops", 0) == 0

    # ── timing ────────────────────────────────────────────────────────────

    def _phase_latency_sum(self, g, phase):
        return sum(n.annotations.get("latency_us", 0)
                   for n in g.nodes.values()
                   if n.annotations.get("phase") == phase)

    def test_fwd_latency_unchanged(self, none_all, full_all, sel_all):
        u_none = none_all[2]["unified"]
        u_full = full_all[2]["unified"]
        u_sel  = sel_all[2]["unified"]
        f_none = self._phase_latency_sum(u_none, "fwd")
        f_full = self._phase_latency_sum(u_full, "fwd")
        f_sel  = self._phase_latency_sum(u_sel, "fwd")
        # Forward phase should be identical — recompute doesn't change forward pass
        assert f_none == f_full == f_sel, \
            f"fwd latency differs: none={f_none} full={f_full} sel={f_sel}"

    def test_bwd_latency_full_greater_than_none(self, none_all, full_all):
        b_none = self._phase_latency_sum(none_all[2]["unified"], "bwd")
        b_full = self._phase_latency_sum(full_all[2]["unified"], "bwd")
        assert b_full > b_none, \
            f"bwd latency not increased: full={b_full} < none={b_none}"

    def test_bwd_latency_sel_between(self, none_all, full_all, sel_all):
        b_none = self._phase_latency_sum(none_all[2]["unified"], "bwd")
        b_full = self._phase_latency_sum(full_all[2]["unified"], "bwd")
        b_sel  = self._phase_latency_sum(sel_all[2]["unified"], "bwd")
        assert b_none <= b_sel <= b_full, \
            f"bwd: none={b_none} sel={b_sel} full={b_full}"

    def test_step_time_full_greater_than_none(self, none_all, full_all):
        r_none, _, _ = none_all
        r_full, _, _ = full_all
        assert r_full.step_time_ms > r_none.step_time_ms, \
            f"step: full={r_full.step_time_ms}ms <= none={r_none.step_time_ms}ms"

    def test_step_time_sel_between(self, none_all, full_all, sel_all):
        r_none, _, _ = none_all
        r_full, _, _ = full_all
        r_sel,  _, _ = sel_all
        assert r_none.step_time_ms <= r_sel.step_time_ms <= r_full.step_time_ms, \
            f"step: none={r_none.step_time_ms} sel={r_sel.step_time_ms} full={r_full.step_time_ms}"

    # ── efficiency ────────────────────────────────────────────────────────

    def test_mfu_full_lower_than_none(self, none_all, full_all):
        r_none, _, _ = none_all
        r_full, _, _ = full_all
        assert r_full.mfu < r_none.mfu, \
            f"MFU: full={r_full.mfu} >= none={r_none.mfu}"

    def test_hfu_exceeds_mfu(self, full_all):
        _, _, t = full_all
        pm = t["unified"].metadata.get("pipeline_metrics")
        if pm is not None:
            assert pm.hfu > pm.mfu, f"HFU={pm.hfu} <= MFU={pm.mfu}"

    # ── memory ────────────────────────────────────────────────────────────

    def test_activation_memory_reduced(self, none_all, full_all):
        mb_none = none_all[2]["unified"].metadata.get("memory_breakdown")
        mb_full = full_all[2]["unified"].metadata.get("memory_breakdown")
        assert mb_none is not None and mb_full is not None
        assert mb_full.activations < mb_none.activations, \
            f"activations: full={mb_full.activations} >= none={mb_none.activations}"

    # ── per-node memory access ────────────────────────────────────────────

    def test_read_write_bytes_unchanged(self, none_all, full_all):
        u_none = none_all[2]["unified"]
        u_full = full_all[2]["unified"]
        for nid, n_none in u_none.nodes.items():
            if nid not in u_full.nodes:
                continue
            n_full = u_full.nodes[nid]
            assert n_none.annotations.get("read_bytes") == n_full.annotations.get("read_bytes"), \
                f"{nid} read_bytes differs"
            assert n_none.annotations.get("write_bytes") == n_full.annotations.get("write_bytes"), \
                f"{nid} write_bytes differs"
