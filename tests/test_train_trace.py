"""Tests for training phase tracing (train_forward / train_backward)."""
import pytest
from python.zrt.pipeline import run_trace_phases

# Uses local config only — no network required
# MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_ID = "deepseek-ai/DeepSeek-V3"
_COMMON = dict(model_id=MODEL_ID, num_layers=2, batch_size=1, seq_len=16)


def test_train_forward_produces_records():
    _, phase_records = run_trace_phases(**_COMMON, phases=("train_forward",))
    records = phase_records["train_forward"]
    assert len(records) > 0


def test_train_alias_equals_train_forward():
    """'train' is an alias for 'train_forward'."""
    _, pr_alias = run_trace_phases(**_COMMON, phases=("train",))
    _, pr_full = run_trace_phases(**_COMMON, phases=("train_forward",))
    ops_alias = [r["aten_op"] for r in pr_alias["train_forward"]]
    ops_full = [r["aten_op"] for r in pr_full["train_forward"]]
    assert ops_alias == ops_full


def test_train_backward_has_ops():
    """Backward phase captures gradient ops (only backward ops, not forward ops)."""
    _, phase_records = run_trace_phases(
        **_COMMON, phases=("train_forward", "train_backward")
    )
    bwd = phase_records["train_backward"]
    assert len(bwd) > 0, "train_backward phase produced no records"


def test_train_and_inference_phases_coexist():
    """Training phases and inference phases can be mixed in one call."""
    _, phase_records = run_trace_phases(
        **_COMMON, phases=("prefill", "train_forward")
    )
    assert "prefill" in phase_records
    assert "train_forward" in phase_records
    assert len(phase_records["prefill"]) > 0
    assert len(phase_records["train_forward"]) > 0


# ── End-to-end tests (forward + backward together) ────────────────────────────

def test_train_e2e_both_phases_records_present(tmp_path):
    """E2E: train_forward + train_backward both produce non-empty op records."""
    result = run_trace_phases(
        **_COMMON, phases=("train_forward", "train_backward"), output_dir=tmp_path
    )
    _, phase_records = result

    for phase in ("train_forward", "train_backward"):
        assert len(phase_records[phase]) > 0, f"{phase} produced no records"


def test_train_e2e_opgraph_has_nodes(tmp_path):
    """OpGraph IR built from training phases must have nodes and edges."""
    result = run_trace_phases(
        **_COMMON, phases=("train_forward", "train_backward"), output_dir=tmp_path
    )
    for phase in ("train_forward", "train_backward"):
        raw_g = result.graphs[phase]
        assert raw_g.num_nodes() > 0, f"{phase} raw OpGraph is empty"


def test_train_backward_has_backward_only_ops():
    """Backward phase must contain aten ops absent from the forward-only phase.

    These are gradient ops (e.g. transposed mm, threshold_backward, etc.)
    that PyTorch autograd emits only during the backward pass.
    """
    _, phase_records = run_trace_phases(
        **_COMMON, phases=("train_forward", "train_backward")
    )
    fwd_ops = {r["aten_op"] for r in phase_records["train_forward"]}
    bwd_ops = {r["aten_op"] for r in phase_records["train_backward"]}

    bwd_only = bwd_ops - fwd_ops
    assert len(bwd_only) > 0, (
        "Expected backward-only gradient ops but found none. "
        f"forward ops: {len(fwd_ops)}, backward ops: {len(bwd_ops)}"
    )
