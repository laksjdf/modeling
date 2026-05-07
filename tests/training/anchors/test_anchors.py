"""Test anchor YAML data files.

Phase 4.5: Structural validation + MFU calibration output
P3: Strict MFU tolerance gating for calibrated anchors
"""
from __future__ import annotations

import yaml
import pytest
from pathlib import Path

from zrt.training.anchor.validate import Anchor, validate_anchor
from zrt.training.search.estimator import Report


ANCHOR_DIR = Path(__file__).parent


def _load_anchor(yaml_path: Path) -> dict:
    return yaml.safe_load(yaml_path.read_text(encoding='utf-8'))


@pytest.mark.parametrize("yaml_file", sorted(ANCHOR_DIR.glob("*.yaml")))
def test_anchor_yaml_is_valid(yaml_file):
    data = _load_anchor(yaml_file)
    assert "name" in data
    assert "targets" in data
    anchor = Anchor(name=data["name"], **data["targets"])
    assert anchor.name


@pytest.mark.parametrize("yaml_file", sorted(ANCHOR_DIR.glob("*.yaml")))
def test_anchor_yaml_has_config(yaml_file):
    data = _load_anchor(yaml_file)
    assert "config" in data
    config = data["config"]
    assert "tp" in config
    assert "dp" in config


@pytest.mark.parametrize("yaml_file", sorted(ANCHOR_DIR.glob("*.yaml")))
def test_anchor_config_is_internally_consistent(yaml_file):
    """Verify anchor configs are internally consistent (no conflicting products)."""
    data = _load_anchor(yaml_file)
    config = data["config"]
    name = data["name"]

    tp = config.get("tp", 1)
    pp = config.get("pp", 1)
    dp = config.get("dp", 1)

    # Basic sanity: world_size should be consistent with tp * pp * dp
    # (EP excluded per current policy — see test_ep_rank_product.py)
    world_size = config.get("world_size", tp * pp * dp)
    rank_product = tp * pp * dp

    assert rank_product == world_size, (
        f"Anchor '{name}': TP*PP*DP={rank_product} != world_size={world_size}. "
        f"Internal consistency check failed."
    )


def test_anchor_validate_with_report():
    report = Report(step_time_ms=100.0, mfu=0.50, total_flops=1e12)
    anchor = Anchor(name="test", mfu=0.50, tolerance=0.15, strict_mfu_check=True)
    warnings = validate_anchor(report, anchor)
    assert len(warnings) == 0


def test_anchor_validate_fails_with_bad_report_strict():
    """Strict MFU check should fail when deviation exceeds tolerance."""
    report = Report(step_time_ms=200.0, mfu=0.20, total_flops=1e12)
    anchor = Anchor(name="test", mfu=0.50, tolerance=0.15, strict_mfu_check=True)
    warnings = validate_anchor(report, anchor)
    assert len(warnings) > 0
    assert "[STRICT]" in warnings[0]


def test_anchor_validate_calibration_mode_no_failure():
    """Calibration mode records MFU deviation but doesn't fail."""
    report = Report(step_time_ms=200.0, mfu=0.20, total_flops=1e12)
    anchor = Anchor(name="test", mfu=0.50, tolerance=0.15, strict_mfu_check=False)
    warnings = validate_anchor(report, anchor)
    # Should have warning but marked as [CALIBRATION], not [STRICT]
    assert len(warnings) > 0
    assert "[CALIBRATION]" in warnings[0]
    assert "[STRICT]" not in warnings[0]


def test_anchor_estimate_integration():
    """Run estimate() for each anchor and enforce calibrated strict checks."""
    from zrt.training.io.config_loader import load_anchor_config
    from zrt.training.search.estimator import estimate

    calibration_results = []

    for yaml_file in sorted(ANCHOR_DIR.glob("*.yaml")):
        model, system, strategy = load_anchor_config(yaml_file)
        anchor_data = _load_anchor(yaml_file)
        anchor = Anchor(name=anchor_data["name"], **anchor_data["targets"])

        # Validate strategy consistency first: no internally inconsistent device product.
        strategy.validate(model, system)

        report = estimate(model, system, strategy)
        warnings = validate_anchor(report, anchor)

        mfu_error = abs(report.mfu - anchor.mfu) / anchor.mfu if anchor.mfu and anchor.mfu > 0 else 0
        calibration_results.append({
            "name": anchor.name,
            "estimated_mfu": report.mfu,
            "reference_mfu": anchor.mfu,
            "mfu_error_pct": mfu_error * 100,
            "within_tolerance": mfu_error <= anchor.tolerance,
            "strict_mfu_check": anchor.strict_mfu_check,
            "warnings": warnings,
        })

        print(f"\n{anchor.name}:")
        print(f"  Estimated MFU: {report.mfu:.4f}")
        print(f"  Reference MFU:  {anchor.mfu:.4f}")
        print(f"  Error: {mfu_error*100:.2f}% (tolerance: {anchor.tolerance*100:.0f}%)")
        if warnings:
            for w in warnings:
                print(f"  WARNING: {w}")

        if anchor.strict_mfu_check:
            strict_warnings = [w for w in warnings if w.startswith("[STRICT]")]
            assert not strict_warnings, (
                f"Anchor {anchor.name} failed strict validation: {strict_warnings}"
            )

    # Print summary
    print("\n" + "=" * 60)
    print("Anchor Calibration Summary")
    print("=" * 60)
    for r in calibration_results:
        status = "PASS" if r["within_tolerance"] else "CALIBRATION NEEDED"
        print(f"  {r['name']}: {status} (MFU error: {r['mfu_error_pct']:.2f}%)")
    print("=" * 60)
