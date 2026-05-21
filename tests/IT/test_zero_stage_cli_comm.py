from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

# Number of traced layers (matches --layers 4 in _BASE_CMD)
_NUM_LAYERS = 4

# Base CLI command
_BASE_CMD = [
    sys.executable, "-m", "python.zrt",
    "--model-id", "hf_models/deepseek_v3",
    "--train",
    "--hw", "nvidia_h100_sxm",
    "--tp", "8",
    "--pp", "4",
    "--dp", "2",
    "--global-batch", "102",
    "--layers", str(_NUM_LAYERS),
]


def _run_cli_with_zero_stage(zero_stage: int, output_dir: Path) -> Path:
    """Run CLI with specified zero-stage and return the trace file path."""
    cmd = _BASE_CMD + [
        "--zero-stage", str(zero_stage),
        "--output-dir", str(output_dir),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        pytest.fail(
            f"CLI failed with returncode {result.returncode}\n"
            f"stderr: {result.stderr[-2000:]}"
        )

    slug = "deepseek_v3"
    trace_path = output_dir / slug / "reports" / f"{slug}_train_trace.json"
    if not trace_path.exists():
        trace_path_alt = output_dir / "reports" / f"{slug}_train_trace.json"
        if trace_path_alt.exists():
            return trace_path_alt
        pytest.fail(f"Trace file not found at {trace_path}")
    return trace_path


def _extract_comm_events(trace_path: Path) -> list[dict]:
    """Load trace JSON and return only communication events."""
    data = json.loads(trace_path.read_text())
    return [e for e in data["traceEvents"] if e.get("cat") == "communication"]


def _extract_comm_by_op_type(comm_events: list[dict]) -> dict[str, list[dict]]:
    """Group communication events by op_type."""
    groups: dict[str, list[dict]] = {}
    for e in comm_events:
        op_type = e.get("args", {}).get("op_type", "unknown")
        groups.setdefault(op_type, []).append(e)
    return groups


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


class TestZeroStageCLICommComparison:
    """Integration test comparing ZeRO-0 vs ZeRO-3 communication from actual CLI runs."""

    @pytest.fixture(scope="class")
    def zero0_trace(self, tmp_path_factory):
        """Run CLI with --zero-stage 0 and return the trace file path."""
        output_dir = tmp_path_factory.mktemp("zero0_output")
        return _run_cli_with_zero_stage(0, output_dir)

    @pytest.fixture(scope="class")
    def zero3_trace(self, tmp_path_factory):
        """Run CLI with --zero-stage 3 and return the trace file path."""
        output_dir = tmp_path_factory.mktemp("zero3_output")
        return _run_cli_with_zero_stage(3, output_dir)

    def test_zero3_has_per_layer_all_gather(self, zero3_trace: Path):
        """ZeRO-3 should have exactly one all_gather per traced layer."""
        comm_events = _extract_comm_events(zero3_trace)
        ag_events = [e for e in comm_events
                     if e.get("args", {}).get("op_type") == "comm.all_gather"]
        assert len(ag_events) == _NUM_LAYERS, (
            f"ZeRO-3 expected {_NUM_LAYERS} all_gather events (one per traced layer), got {len(ag_events)}"
        )

    def test_zero3_has_per_layer_reduce_scatter(self, zero3_trace: Path):
        """ZeRO-3 should have exactly one reduce_scatter per traced layer."""
        comm_events = _extract_comm_events(zero3_trace)
        rs_events = [e for e in comm_events
                     if e.get("args", {}).get("op_type") == "comm.reduce_scatter"
                     and e.get("args", {}).get("node_id").startswith("comm_fsdp")]
        assert len(rs_events) == _NUM_LAYERS, (
            f"ZeRO-3 expected {_NUM_LAYERS} reduce_scatter events (one per traced layer), got {len(rs_events)}"
        )


    # ── Comparison tests ──────────────────────────────────────────────────

    def test_zero3_has_more_comm_events_than_zero0(self, zero0_trace: Path, zero3_trace: Path):
        """ZeRO-3 should have more communication events than ZeRO-0 due to per-layer FSDP comm."""
        comm_z0 = _extract_comm_events(zero0_trace)
        comm_z3 = _extract_comm_events(zero3_trace)

        assert len(comm_z3) > len(comm_z0), (
            f"ZeRO-3 ({len(comm_z3)} events) should have more comm events than ZeRO-0 ({len(comm_z0)})"
        )

    def test_zero3_has_more_comm_op_types(self, zero0_trace: Path, zero3_trace: Path):
        """ZeRO-3 should have more diverse communication op types (AG + RS in addition to AR + SR)."""
        comm_z0 = _extract_comm_events(zero0_trace)
        comm_z3 = _extract_comm_events(zero3_trace)

        z0_types = {e["args"]["op_type"] for e in comm_z0}
        z3_types = {e["args"]["op_type"] for e in comm_z3}

        # Both should have TP all_reduce and PP send_recv
        assert "comm.all_reduce" in z0_types
        assert "comm.send_recv" in z0_types
        assert "comm.all_reduce" in z3_types
        assert "comm.send_recv" in z3_types

        # ZeRO-3 should additionally have AG and RS
        assert "comm.all_gather" in z3_types
        assert "comm.reduce_scatter" in z3_types

        # ZeRO-3 should have more op types
        assert len(z3_types) > len(z0_types), (
            f"ZeRO-3 op types ({z3_types}) should be superset of ZeRO-0 ({z0_types})"
        )

    def test_zero0_vs_zero3_total_comm_latency(self, zero0_trace: Path, zero3_trace: Path):
        """Compare total communication latency between ZeRO-0 and ZeRO-3."""
        comm_z0 = _extract_comm_events(zero0_trace)
        comm_z3 = _extract_comm_events(zero3_trace)

        total_z0_us = sum(e.get("dur", 0) for e in comm_z0)
        total_z3_us = sum(e.get("dur", 0) for e in comm_z3)

        # Both should have non-zero comm latency
        assert total_z0_us > 0, "ZeRO-0 should have non-zero comm latency"
        assert total_z3_us > 0, "ZeRO-3 should have non-zero comm latency"

        # ZeRO-3 total comm latency should be greater than ZeRO-0
        # because ZeRO-3 adds per-layer FSDP comm (all_gather + reduce_scatter)
        # on top of the TP all_reduce and PP send_recv that both share.
        assert total_z3_us > total_z0_us, (
            f"ZeRO-3 total comm latency ({total_z3_us:.2f}µs) should be greater than "
            f"ZeRO-0 ({total_z0_us:.2f}µs) due to per-layer FSDP communication"
        )


class TestZeroStageCLIMemorySharding:
    """Test ZeRO memory component sharding across all stages (0-3)."""

    @pytest.fixture(scope="class")
    def cli_results(self, tmp_path_factory):
        """Run CLI for ZeRO-0, ZeRO-1, ZeRO-2, ZeRO-3 once, cache results."""
        results = {}
        for stage in [0, 1, 2, 3]:
            out_dir = tmp_path_factory.mktemp(f"zero{stage}_mem")
            trace_path = _run_cli_with_zero_stage(stage, out_dir)
            results[stage] = {
                "output_dir": out_dir,
                "trace_path": trace_path,
            }
        return results

    def test_zero_memory_components_sharding(self, cli_results):
        """ZeRO stages should shard specific memory components (W, G, Opt) as expected."""
        mem = {}
        for stage, r in cli_results.items():
            out_dir = r["output_dir"]
            slug = "deepseek_v3"
            json_report = read_json_report(out_dir, slug)
            mem_breakdown = json_report.get("memory_breakdown_gb", {})
            if all(k in mem_breakdown for k in ("weights", "grads", "opt_state")):
                mem[stage] = mem_breakdown

        assert len(mem) == 4, f"Could not parse memory components for all stages: {mem.keys()}"

        # Weights: z0 == z1 == z2 > z3
        assert mem[0]["weights"] == pytest.approx(mem[1]["weights"], rel=0.05)
        assert mem[1]["weights"] == pytest.approx(mem[2]["weights"], rel=0.05)
        assert mem[3]["weights"] < mem[0]["weights"]

        # Grads: z0 == z1 > z2 == z3
        assert mem[0]["grads"] == pytest.approx(mem[1]["grads"], rel=0.05)
        assert mem[2]["grads"] < mem[1]["grads"]
        assert mem[2]["grads"] == pytest.approx(mem[3]["grads"], rel=0.05)

        # Opt: z0 > z1 == z2 == z3
        assert mem[1]["opt_state"] < mem[0]["opt_state"]
        assert mem[1]["opt_state"] == pytest.approx(mem[2]["opt_state"], rel=0.05)
        assert mem[2]["opt_state"] == pytest.approx(mem[3]["opt_state"], rel=0.05)
