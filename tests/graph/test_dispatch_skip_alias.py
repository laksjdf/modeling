"""Regression test for SKIP-op tensor-id aliasing in ``RecordingDispatch``.

Before the fix, ops in ``SKIP_OPS`` (view/_unsafe_view/t/_to_copy/detach...)
produced fresh tracker IDs for their outputs.  Because those ops emit no
record, the next consumer's input tensor id had no producer record, and
``records_to_opgraph`` silently dropped the edge.  In DSV4 this broke the
chain ``aten.mm → aten._unsafe_view → aten.mul`` that ``F.linear * rsqrt``
expands to.

The fix aliases SKIP-op outputs to the first input ID inside the dispatch
hook so the dataflow chain stays intact.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from python.zrt.graph.dispatch import RecordingDispatch, TensorTracker
from python.zrt.graph.tensor_utils import SHAPE_OPS, SKIP_OPS


def _run_chain(use_fake: bool = True):
    """Run mm → _unsafe_view → mul under RecordingDispatch and return the
    records + tracker so the test can inspect tensor-id continuity."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    tracker = TensorTracker()
    recorder = RecordingDispatch(tracker, module_tracker=None, skip_reshapes=True)
    fake_mode = FakeTensorMode() if use_fake else None

    if fake_mode is not None:
        fake_mode.__enter__()
    try:
        with recorder:
            x = torch.randn(1, 64, 7168)
            w = torch.randn(28672, 7168)
            x2 = x.flatten(0, 1)                # view (skipped)
            y2 = torch.nn.functional.linear(x2, w)
            # F.linear emits mm + _unsafe_view internally.
            y = y2.view(1, 64, 28672)            # view (skipped)
            rsq = torch.randn(1, 64, 1)
            out = y * rsq                        # mul.Tensor
    finally:
        if fake_mode is not None:
            fake_mode.__exit__(None, None, None)

    return recorder.records, tracker


def test_view_after_mm_chains_to_mul():
    """The mul's input tensor ID must match an mm output's tensor ID,
    proving the SKIP-op alias kept the chain intact."""
    records, _ = _run_chain()
    mm_recs = [r for r in records if r["aten_op"] == "aten.mm.default"]
    mul_recs = [r for r in records if r["aten_op"] == "aten.mul.Tensor"]
    assert mm_recs, "no mm op captured"
    assert mul_recs, "no mul op captured"

    mm_out_ids = {tid for r in mm_recs for tid in r["_output_ids"]}
    mul_in_ids = {tid for r in mul_recs for tid in r["_input_ids"]}
    overlap = mm_out_ids & mul_in_ids
    assert overlap, (
        f"mm outputs {mm_out_ids} share no tensor id with mul inputs "
        f"{mul_in_ids} — SKIP-op alias regressed"
    )


def test_skip_ops_do_not_emit_records():
    """SKIP ops still produce zero records (the alias fix preserves IDs
    without recording the op itself)."""
    records, _ = _run_chain()
    recorded_ops = {r["aten_op"] for r in records}
    for op in SHAPE_OPS:
        assert op not in recorded_ops, (
            f"SHAPE_OP {op} leaked into records; alias fix should be silent"
        )


def test_records_to_opgraph_wires_mm_to_mul():
    """End-to-end: ``records_to_opgraph`` builds an edge from mm to mul
    through the skipped view."""
    from python.zrt.ir.adapter import records_to_opgraph

    records, _ = _run_chain()
    g = records_to_opgraph(records, name="t", phase="fwd")

    # Find the mm op_id and the mul op_id, assert an edge between them
    mm_ids = [f"op_{r['node_id']}" for r in records if r["aten_op"] == "aten.mm.default"]
    mul_ids = [f"op_{r['node_id']}" for r in records if r["aten_op"] == "aten.mul.Tensor"]
    edges = {(e.src, e.dst) for e in g.edges}
    has_path = any((mm, mul) in edges for mm in mm_ids for mul in mul_ids)
    assert has_path, f"no edge mm→mul in graph; edges={edges!r}"
