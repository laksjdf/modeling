"""PyTest infrastructure — marker registration + shared model capture.

Test files are self-contained.  This module only provides the
session-scoped ``captured_model`` fixture so that multiple E2E
test files can share a single ``run_trace_phases`` call.

Do NOT place test-specific logic or fixtures here.
"""
from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "ep: expert parallelism test")
    config.addinivalue_line("markers", "recompute: recompute / activation checkpointing test")


@pytest.fixture(scope="session")
def captured_model():
    """Capture DSv4 4-layer training graphs once per session."""
    pytest.importorskip("torch")
    pytest.importorskip("tqdm")
    pytest.importorskip("transformers")
    from python.zrt.pipeline import run_trace_phases
    return run_trace_phases(
        model_id="hf_models/deepseek_v4", num_layers=4,
        batch_size=1, seq_len=128,
        phases=("train_forward", "train_backward"),
    )
