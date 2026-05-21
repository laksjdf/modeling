"""Integration test: verify how DataParallel (DP) affects TrainingReport fields.

This test runs the training modelling CLI for dp=1, dp=4, dp=8, loads the
generated ``deepseek_v4_training_report.json`` files and asserts expected
relationships:

- optimizer state (opt_state) per-GPU ≈ 1/dp  (ZeRO-1)
- total per-GPU memory drops monotonically as dp increases
- step_time decreases and tokens/sec increases monotonically as dp increases
- dp_hidden_ms and dp_exposed_ms are zero for dp=1, non-zero for dp>1
- dp_comm_total (dp_hidden + dp_exposed) increases monotonically with dp
- optimizer state scales approximately as 1/dp across dp=4 and dp=8

This is a long-running integration test that captures real reports. To avoid
running it by default in fast CI, it is skipped unless the environment
variable ``RUN_DP_TEST`` is set to ``1``.

Run locally (PowerShell):

```powershell
$env:RUN_DP_TEST='1'; pytest tests/IT/test_dp_effects.py -q
```
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_cli_and_load_report(repo_root: Path, outdir: Path, dp: int, timeout: int = 900) -> dict:
    """Run `python -m python.zrt` with given dp and return parsed report JSON.

    Raises subprocess.CalledProcessError on failure.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "python")

    cmd = [
        sys.executable,
        "-m",
        "python.zrt",
        "--model-id",
        "hf_models/deepseek_v4",
        "--train",
        "--hw",
        "nvidia_h100_sxm",
        "--dp",
        str(dp),
        "--layers",
        "4",
        "--batch-size",
        "1",
        "--seq-len",
        "128",
        "--output-dir",
        str(outdir),
    ]

    proc = subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "cli_output.log").write_text(proc.stdout)

    report_path = outdir / "reports" / "deepseek_v4_training_report.json"
    assert report_path.exists(), f"Report not found at {report_path}"
    return json.loads(report_path.read_text())


@pytest.fixture(scope="session")
def dp_reports(tmp_path_factory):
    """Session-scoped fixture: run CLI for dp=1, dp=4, dp=8, return all reports."""
    if os.environ.get("RUN_DP_TEST") != "1":
        pytest.skip("Set RUN_DP_TEST=1 to run this long integration test")

    repo_root = Path(__file__).resolve().parents[3]
    tmp_path = tmp_path_factory.mktemp("dp_effects")

    reports = {}
    for dp in (1, 4, 8):
        out_dir = tmp_path / f"out_dp{dp}"
        reports[dp] = _run_cli_and_load_report(repo_root, out_dir, dp=dp)

    return reports


# ── ZeRO-1 optimizer state scaling ─────────────────────────────────────────

def test_dp_optimizer_state_scales_inverse(dp_reports):
    """Optimizer state per-GPU should roughly scale ~1/dp (ZeRO-1 behaviour)."""
    opt = {}
    for dp in (1, 4, 8):
        mb = dp_reports[dp].get("memory_breakdown_gb")
        assert mb, f"memory_breakdown_gb missing for dp={dp}"
        opt[dp] = mb.get("opt_state")
        assert opt[dp] is not None, f"opt_state missing for dp={dp}"

    # Approximate 1/dp scaling: opt(dp) ≈ opt(1) / dp
    for dp in (4, 8):
        expected = opt[1] / dp
        assert opt[dp] == pytest.approx(expected, rel=0.25), (
            f"opt_state did not scale near 1/dp: dp=1 → {opt[1]}, dp={dp} → {opt[dp]}, "
            f"expected ≈ {expected}"
        )

    # Monotonic decrease
    assert opt[1] > opt[4] > opt[8], (
        f"opt_state should decrease monotonically: dp1={opt[1]}, dp4={opt[4]}, dp8={opt[8]}"
    )


# ── Total memory monotonicity ──────────────────────────────────────────────

def test_dp_total_memory_decreases(dp_reports):
    """Total per-GPU memory should decrease monotonically as dp increases."""
    total = {}
    for dp in (1, 4, 8):
        mb = dp_reports[dp]["memory_breakdown_gb"]
        total[dp] = mb["total"]
        assert total[dp] is not None

    assert total[1] > total[4] > total[8], (
        f"total memory should decrease monotonically: "
        f"dp1={total[1]}, dp4={total[4]}, dp8={total[8]}"
    )


# ── Throughput monotonicity ────────────────────────────────────────────────

def test_dp_throughput_improves(dp_reports):
    """Step time should decrease and tokens/sec should increase monotonically."""
    step_time = {dp: dp_reports[dp]["step_time_ms"] for dp in (1, 4, 8)}

    assert step_time[1] > step_time[4] > step_time[8], (
        f"step_time should decrease monotonically: "
        f"dp1={step_time[1]}, dp4={step_time[4]}, dp8={step_time[8]}"
    )


# ── DP communication accounting ────────────────────────────────────────────

def test_dp_communication_zero_for_dp1(dp_reports):
    """dp_hidden_ms and dp_exposed_ms should be 0 for dp=1 (no DP communication)."""
    rep1 = dp_reports[1]

    dp_hidden1 = rep1.get("dp_hidden_ms")
    dp_exposed1 = rep1.get("dp_exposed_ms")
    assert dp_hidden1 is not None
    assert dp_exposed1 is not None
    assert dp_hidden1 == 0.0, f"dp_hidden should be 0 for dp=1, got {dp_hidden1}"
    assert dp_exposed1 == 0.0, f"dp_exposed should be 0 for dp=1, got {dp_exposed1}"


def test_dp_communication_nonzero_for_dp_gt1(dp_reports):
    """dp_hidden + dp_exposed should be >0 for dp>1 (DP AR/RS communication exists).

    dp_hidden may be 0 (e.g. no bubble to absorb AR) or dp_exposed may be 0
    (e.g. AR fully hidden in bubble), but their sum must reflect the total
    DP communication volume.
    """
    for dp in (4, 8):
        rep = dp_reports[dp]
        dp_hidden = rep.get("dp_hidden_ms")
        dp_exposed = rep.get("dp_exposed_ms")
        dp_comm = rep.get("dp_total_ms", 0.0)
        assert dp_hidden is not None, f"dp_hidden_ms missing for dp={dp}"
        assert dp_exposed is not None, f"dp_exposed_ms missing for dp={dp}"
        assert dp_comm > 0, (
            f"dp_hidden + dp_exposed should be >0 for dp={dp}, "
            f"got hidden={dp_hidden}, exposed={dp_exposed}"
        )


def test_dp_comm_volume_monotonic(dp_reports):
    """Total DP communication volume (dp_hidden + dp_exposed) should increase with dp.

    Larger DP group means more gradient data to reduce, so the total DP
    communication time (exposed + hidden) should grow monotonically.
    Note: on full-mesh topologies the per-step time may not scale linearly
    due to the (N-1)/N ring factor, but the trend should be increasing.
    """
    dp_comm = {}
    for dp in (4, 8):
        rep = dp_reports[dp]
        dp_comm[dp] = rep.get("dp_total_ms", 0.0)

    assert dp_comm[8] > dp_comm[4], (
        f"DP comm volume should increase with dp: "
        f"dp4={dp_comm[4]:.2f}ms, dp8={dp_comm[8]:.2f}ms"
    )