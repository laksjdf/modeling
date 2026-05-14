from __future__ import annotations

from types import SimpleNamespace

import pytest


def _check_torch_available():
    """Check if torch is available."""
    try:
        import torch
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _check_torch_available(), reason="torch not installed")
class _Report:
    def summary(self) -> str:
        return "graph-native report"


def test_train_hw_cli_delegates_to_graph_native_modeller(monkeypatch, capsys):
    if not _check_torch_available():
        pytest.skip("torch not installed")
    
    from python.zrt import cli

    calls = []

    def fake_estimate_training_from_graphs(**kwargs):
        calls.append(kwargs)
        # cli.py unpacks (report, ctx, transformed); return matching 3-tuple
        return _Report(), None, {}

    monkeypatch.setattr(
        "python.zrt.transform.analysis.estimate_training_from_graphs",
        fake_estimate_training_from_graphs,
    )

    args = SimpleNamespace(
        hw="test_gpu",
        layers=4,
        batch_size=2,
        seq_len=128,
        tp=2,
        pp=3,
        ep=1,
        dp=4,
        cp=5,
        cp_kind="ring",
        zero_stage=2,
        optimizer="adamw",
        muon_rotation=True,
        muon_ns_steps=None,
        micro_batch=1,
        global_batch=16,
        total_params=123e9,
        hidden=4096,
        num_layers_full=32,
        quant=None,
    )
    fwd_graph = object()
    bwd_graph = object()
    result = SimpleNamespace(
        graphs={
            "train_forward": fwd_graph,
            "train_backward": bwd_graph,
        },
        output_dir=None,
    )
    hw = object()

    cli._run_training_modelling(args, "hf_models/llama3_8b", hw, result)

    assert len(calls) == 1
    assert calls[0]["forward_graph"] is fwd_graph
    assert calls[0]["backward_graph"] is bwd_graph
    assert calls[0]["hw_spec"] is hw
    assert calls[0]["tp"] == 2
    assert calls[0]["pp"] == 3
    assert calls[0]["dp"] == 4
    assert calls[0]["cp"] == 5
    assert calls[0]["cp_kind"] == "ring"
    assert calls[0]["zero_stage"] == 2
    assert "graph-native report" in capsys.readouterr().out
