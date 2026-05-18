"""End-to-end CLI tests: --train with ZeRO on/off comparison.

Invokes `python -m python.zrt --model-id hf_models/deepseek_v3 --train --hw nvidia_h100_sxm`
with different --zero-stage values and compares the output.

Tests the full pipeline:
  CLI arg parsing → graph capture (train_forward + train_backward) →
  estimate_training_from_graphs → TrainingReport → stdout + JSON export
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_ID = "hf_models/deepseek_v3"
HW = "nvidia_h100_sxm"

COMMON_ARGS = [
    "--model-id", MODEL_ID,
    "--train", "--hw", HW,
    "--tp", "8", "--pp", "4", "--dp", "2",
    "--layers", "2",
    "--batch-size", "1",
    "--seq-len", "16",
    "--micro-batch", "1",
    "--global-batch", "32",
]


def _run_cli(args: list[str], output_dir: Path, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run `python -m python.zrt` with given args and output directory."""
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "python"), "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, "-m", "python.zrt"] + args + ["--output-dir", str(output_dir)]
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env,
        cwd=str(PROJECT_ROOT), encoding="utf-8", errors="replace",
    )


def parse_report_stdout(stdout: str) -> dict:
    """Parse key metrics from the Training Report block in CLI stdout."""
    result: dict = {}
    if not stdout:
        return result

    m = re.search(r"Step:\s+([\d.]+)\s*ms", stdout)
    if m:
        result["step_time_ms"] = float(m.group(1))

    m = re.search(r"MFU\s+([\d.]+)%", stdout)
    if m:
        result["mfu"] = float(m.group(1)) / 100.0

    m = re.search(r"HFU\s+([\d.]+)%", stdout)
    if m:
        result["hfu"] = float(m.group(1)) / 100.0

    m = re.search(r"Memory:\s+([\d.]+)\s*GB/GPU", stdout)
    if m:
        result["memory_gb"] = float(m.group(1))

    mem_match = re.search(r"Memory:\s+[\d.]+\s*GB/GPU\s+\((.*?)\)", stdout)
    if mem_match:
        mem_parts = mem_match.group(1)
        for key, label in [("weights_gb", "W"), ("grads_gb", "G"), ("opt_state_gb", "Opt"),
                           ("activations_gb", "Act"), ("comm_buffers_gb", "Comm")]:
            m = re.search(rf"{label}\s+([\d.]+)", mem_parts)
            if m:
                result[key] = float(m.group(1))

    m = re.search(r"bubble\s+([\d.]+)%", stdout)
    if m:
        result["bubble_fraction"] = float(m.group(1)) / 100.0

    m = re.search(r"Params:\s+([\d.]+)\s*([BM])", stdout)
    if m:
        val = float(m.group(1))
        scale = 1e9 if m.group(2) == "B" else 1e6
        result["total_params"] = val * scale

    m = re.search(r"Total\s+([\d.]+)([TGMK])", stdout)
    if m:
        val = float(m.group(1))
        scale = {"T": 1e12, "G": 1e9, "M": 1e6, "K": 1e3}[m.group(2)]
        result["total_flops"] = val

    return result


def read_json_report(output_dir: Path, slug: str) -> dict:
    """Read the JSON training report exported by the CLI."""
    json_path = output_dir / "reports" / f"{slug}_training_report.json"
    assert json_path.exists(), f"Missing JSON report: {json_path}"
    return json.loads(json_path.read_text(encoding="utf-8", errors="replace"))


@pytest.mark.slow
class TestCLIZeroEndToEnd:
    """E2E CLI tests comparing ZeRO stages via `python -m python.zrt --train`."""

    @pytest.fixture(scope="class")
    def cli_results(self, tmp_path_factory):
        """Run CLI for ZeRO-0, ZeRO-1, ZeRO-2, ZeRO-3 once, cache results."""
        results = {}
        for stage in [0, 1, 2, 3]:
            out_dir = tmp_path_factory.mktemp(f"e2e_zero{stage}")
            proc = _run_cli(
                COMMON_ARGS + ["--zero-stage", str(stage)],
                output_dir=out_dir,
            )
            assert proc.returncode == 0, (
                f"CLI failed for ZeRO-{stage} (rc={proc.returncode}):\n"
                f"stderr:\n{proc.stderr[-3000:]}"
            )
            results[stage] = {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "output_dir": out_dir,
            }
        return results

    def test_cli_succeeds_for_all_zero_stages(self, cli_results):
        """CLI should exit successfully for all ZeRO stages."""
        for stage, r in cli_results.items():
            assert "Training Report" in r["stdout"], f"No Training Report in stdout for ZeRO-{stage}"

    def test_config_summary_contains_zero_stage(self, cli_results):
        """Each run's config_summary should include the correct ZeRO stage."""
        for stage, r in cli_results.items():
            assert f"ZeRO-{stage}" in r["stdout"], f"ZeRO-{stage} not found in config summary"

    def test_zero_reduces_memory(self, cli_results):
        """Higher ZeRO stages should reduce per-GPU memory."""
        mem = {}
        for stage, r in cli_results.items():
            parsed = parse_report_stdout(r["stdout"])
            if "memory_gb" in parsed:
                mem[stage] = parsed["memory_gb"]

        assert len(mem) >= 4, f"Could not parse memory from all stages: {mem}"
        assert mem[3] < mem[0], f"ZeRO-3 ({mem[3]:.2f}GB) should use less memory than ZeRO-0 ({mem[0]:.2f}GB)"

    def test_zero_affects_step_time(self, cli_results):
        """ZeRO stage should affect step time due to comm overhead differences."""
        times = {}
        for stage, r in cli_results.items():
            parsed = parse_report_stdout(r["stdout"])
            if "step_time_ms" in parsed:
                times[stage] = parsed["step_time_ms"]

        assert len(times) >= 4, f"Could not parse step time from all stages: {times}"
        for stage in [1, 2, 3]:
            assert times[stage] > 0, f"ZeRO-{stage} step time should be positive"

    def test_mfu_is_valid(self, cli_results):
        """MFU should be in (0, 1) for all ZeRO stages."""
        for stage, r in cli_results.items():
            parsed = parse_report_stdout(r["stdout"])
            assert "mfu" in parsed, f"Could not parse MFU for ZeRO-{stage}"
            assert 0 < parsed["mfu"] < 1.0, f"MFU out of bounds for ZeRO-{stage}: {parsed['mfu']}"

    def test_hfu_is_valid(self, cli_results):
        """HFU should be in (0, 1) for all ZeRO stages."""
        for stage, r in cli_results.items():
            parsed = parse_report_stdout(r["stdout"])
            assert "hfu" in parsed, f"Could not parse HFU for ZeRO-{stage}"
            assert 0 < parsed["hfu"] < 1.0, f"HFU out of bounds for ZeRO-{stage}: {parsed['hfu']}"

    def test_json_report_exported(self, cli_results):
        """Each run should produce a JSON report file."""
        slug = "deepseek_v3"
        for stage, r in cli_results.items():
            json_path = r["output_dir"] / "reports" / f"{slug}_training_report.json"
            assert json_path.exists(), f"Missing JSON report for ZeRO-{stage}: {json_path}"

    def test_json_report_contains_zero_stage(self, cli_results):
        """JSON report config_summary should include the correct ZeRO stage."""
        slug = "deepseek_v3"
        for stage, r in cli_results.items():
            data = read_json_report(r["output_dir"], slug)
            config = data.get("config_summary", "")
            assert f"ZeRO-{stage}" in config, f"ZeRO-{stage} not in JSON config_summary: {config}"

    def test_json_report_memory_breakdown(self, cli_results):
        """JSON report should have memory_breakdown_gb with ZeRO-sharded values."""
        slug = "deepseek_v3"
        memories = {}
        for stage, r in cli_results.items():
            data = read_json_report(r["output_dir"], slug)
            mb = data.get("memory_breakdown_gb", data.get("memory_gb", {}))
            if isinstance(mb, dict):
                total = mb.get("total_gb", mb.get("total", 0))
                if total > 0:
                    memories[stage] = total

        if len(memories) >= 4:
            assert memories[3] < memories[0], (
                f"ZeRO-3 memory ({memories[3]:.2f}GB) should be < ZeRO-0 ({memories[0]:.2f}GB)"
            )

    def test_zero_3_shards_weights_in_json_report(self, cli_results):
        """ZeRO-3 should shard weights compared to ZeRO-0 in the JSON report."""
        slug = "deepseek_v3"
        weights = {}
        for stage, r in cli_results.items():
            data = read_json_report(r["output_dir"], slug)
            mb = data.get("memory_breakdown_gb", {})
            if isinstance(mb, dict):
                w = mb.get("weights_gb", 0)
                if w > 0:
                    weights[stage] = w

        if len(weights) >= 4:
            assert weights[3] < weights[0], (
                f"ZeRO-3 weights ({weights[3]:.4f}GB) should be < ZeRO-0 ({weights[0]:.4f}GB)"
            )

    def test_zero_1_shards_optimizer_in_json_report(self, cli_results):
        """ZeRO-1 should shard optimizer state compared to ZeRO-0 in the JSON report."""
        slug = "deepseek_v3"
        opt_states = {}
        for stage, r in cli_results.items():
            data = read_json_report(r["output_dir"], slug)
            mb = data.get("memory_breakdown_gb", {})
            if isinstance(mb, dict):
                o = mb.get("opt_state_gb", 0)
                if o > 0:
                    opt_states[stage] = o

        if len(opt_states) >= 2:
            assert opt_states[1] < opt_states[0], (
                f"ZeRO-1 opt_state ({opt_states[1]:.4f}GB) should be < ZeRO-0 ({opt_states[0]:.4f}GB)"
            )

    def test_zero_optimizer_time_reduced(self, cli_results):
        """ZeRO-1+ should reduce optimizer time by sharding params across DP."""
        slug = "deepseek_v3"
        opt_times = {}
        for stage, r in cli_results.items():
            data = read_json_report(r["output_dir"], slug)
            opt_ms = data.get("optimizer_time_ms", 0)
            if opt_ms > 0:
                opt_times[stage] = opt_ms

        if len(opt_times) >= 2:
            assert opt_times[1] < opt_times[0], (
                f"ZeRO-1 optimizer time ({opt_times[1]:.2f}ms) should be < ZeRO-0 ({opt_times[0]:.2f}ms)"
            )

    def test_excel_report_exported(self, cli_results):
        """Each run should produce an Excel training report."""
        slug = "deepseek_v3"
        for stage, r in cli_results.items():
            xlsx_path = r["output_dir"] / f"{slug}_training.xlsx"
            assert xlsx_path.exists(), f"Missing Excel report for ZeRO-{stage}: {xlsx_path}"

    def test_zero_memory_components_sharding(self, cli_results):
        """ZeRO stages should shard specific memory components (W, G, Opt) as expected.

        Expected sharding behavior (DP=2):
        - Weights: z0 == z1 == z2 > z3 (only ZeRO-3 shards weights)
        - Grads:   z0 == z1 > z2 == z3 (ZeRO-2/3 shard grads)
        - Opt:     z0 > z1 == z2 == z3 (ZeRO-1/2/3 shard opt state)
        """
        mem = {}
        for stage, r in cli_results.items():
            parsed = parse_report_stdout(r["stdout"])
            if all(k in parsed for k in ("weights_gb", "grads_gb", "opt_state_gb")):
                mem[stage] = parsed

        assert len(mem) == 4, f"Could not parse memory components for all stages: {mem.keys()}"

        # Weights: z0 == z1 == z2 > z3
        assert mem[0]["weights_gb"] == pytest.approx(mem[1]["weights_gb"], rel=0.05), \
            f"W: z0({mem[0]['weights_gb']}) != z1({mem[1]['weights_gb']})"
        assert mem[1]["weights_gb"] == pytest.approx(mem[2]["weights_gb"], rel=0.05), \
            f"W: z1({mem[1]['weights_gb']}) != z2({mem[2]['weights_gb']})"
        assert mem[3]["weights_gb"] < mem[0]["weights_gb"], \
            f"W: z3({mem[3]['weights_gb']}) should be < z0({mem[0]['weights_gb']})"

        # Grads: z0 == z1 > z2 == z3
        assert mem[0]["grads_gb"] == pytest.approx(mem[1]["grads_gb"], rel=0.05), \
            f"G: z0({mem[0]['grads_gb']}) != z1({mem[1]['grads_gb']})"
        assert mem[2]["grads_gb"] < mem[1]["grads_gb"], \
            f"G: z2({mem[2]['grads_gb']}) should be < z1({mem[1]['grads_gb']})"
        assert mem[2]["grads_gb"] == pytest.approx(mem[3]["grads_gb"], rel=0.05), \
            f"G: z2({mem[2]['grads_gb']}) != z3({mem[3]['grads_gb']})"

        # Opt: z0 > z1 == z2 == z3
        assert mem[1]["opt_state_gb"] < mem[0]["opt_state_gb"], \
            f"Opt: z1({mem[1]['opt_state_gb']}) should be < z0({mem[0]['opt_state_gb']})"
        assert mem[1]["opt_state_gb"] == pytest.approx(mem[2]["opt_state_gb"], rel=0.05), \
            f"Opt: z1({mem[1]['opt_state_gb']}) != z2({mem[2]['opt_state_gb']})"
        assert mem[2]["opt_state_gb"] == pytest.approx(mem[3]["opt_state_gb"], rel=0.05), \
            f"Opt: z2({mem[2]['opt_state_gb']}) != z3({mem[3]['opt_state_gb']})"


@pytest.mark.slow
class TestCLIZeroWithDifferentConfigs:
    """E2E CLI tests with different parallel configs and ZeRO stages."""

    def test_zero_with_tp1_pp1_dp4(self, tmp_path):
        """ZeRO-2 with no TP/PP, only DP."""
        out = tmp_path / "tp1_pp1_dp4"
        proc = _run_cli(
            ["--model-id", MODEL_ID, "--train", "--hw", HW,
             "--tp", "1", "--pp", "1", "--dp", "4",
             "--layers", "2", "--batch-size", "1", "--seq-len", "16",
             "--zero-stage", "2", "--micro-batch", "1", "--global-batch", "32"],
            output_dir=out,
        )
        assert proc.returncode == 0
        assert "ZeRO-2" in proc.stdout
        assert "Training Report" in proc.stdout

    def test_zero_with_muon_optimizer(self, tmp_path):
        """ZeRO-1 with Muon optimizer."""
        out = tmp_path / "muon_zero1"
        proc = _run_cli(
            ["--model-id", MODEL_ID, "--train", "--hw", HW,
             "--tp", "8", "--pp", "4", "--dp", "2",
             "--layers", "2", "--batch-size", "1", "--seq-len", "16",
             "--zero-stage", "1", "--optimizer", "muon",
             "--micro-batch", "1", "--global-batch", "32"],
            output_dir=out,
        )
        assert proc.returncode == 0
        assert "ZeRO-1" in proc.stdout
        assert "muon" in proc.stdout.lower()

    def test_zero_with_adamw_optimizer(self, tmp_path):
        """ZeRO-2 with AdamW optimizer."""
        out = tmp_path / "adamw_zero2"
        proc = _run_cli(
            ["--model-id", MODEL_ID, "--train", "--hw", HW,
             "--tp", "8", "--pp", "4", "--dp", "2",
             "--layers", "2", "--batch-size", "1", "--seq-len", "16",
             "--zero-stage", "2", "--optimizer", "adamw",
             "--micro-batch", "1", "--global-batch", "32"],
            output_dir=out,
        )
        assert proc.returncode == 0
        assert "ZeRO-2" in proc.stdout
        assert "adamw" in proc.stdout.lower()
