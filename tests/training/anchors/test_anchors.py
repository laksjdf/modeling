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


def test_anchor_mfu_strict():
    """Hard MFU gate: estimated MFU must be within ±tolerance of anchor target.

    Skips anchors with `strict_mfu_check: false` (calibration mode — anchor MFU
    is sourced from published reality, simulator not yet calibrated to match).
    Anchors without the field default to strict enforcement.
    """
    from zrt.training.io.config_loader import load_anchor_config
    from zrt.training.search.estimator import estimate

    failures = []

    for yaml_file in sorted(ANCHOR_DIR.glob("*.yaml")):
        model, system, strategy = load_anchor_config(yaml_file)
        anchor_data = _load_anchor(yaml_file)
        targets = anchor_data["targets"]
        anchor_mfu = targets.get("mfu", 0)
        tolerance = targets.get("tolerance", 0.15)

        if anchor_mfu <= 0:
            continue
        if targets.get("strict_mfu_check") is False:
            print(f"{anchor_data['name']}: SKIPPED (calibration mode)")
            continue

        report = estimate(model, system, strategy)
        diff = abs(report.mfu - anchor_mfu)

        print(f"{anchor_data['name']}: MFU={report.mfu:.4f} "
              f"(target={anchor_mfu:.2f}, tol=±{tolerance:.2f}, diff={diff:.4f})")

        if diff > tolerance:
            failures.append(
                f"{anchor_data['name']}: MFU={report.mfu:.4f}, "
                f"target={anchor_mfu:.2f}±{tolerance:.2f}, "
                f"diff={diff:.4f} > {tolerance:.4f}"
            )

    assert not failures, (
        f"{len(failures)} anchor(s) failed strict MFU check:\n" +
        "\n".join(failures)
    )


def test_anchor_step_time_strict():
    """Hard step_time gate: estimated step_time_ms must be within ±tolerance (relative).

    Catches 6PN FLOPs-formula and active-param drift independent of MFU.
    Skips anchors without `step_time_ms` in targets and anchors marked
    `strict_mfu_check: false` (calibration mode).
    """
    from zrt.training.io.config_loader import load_anchor_config
    from zrt.training.search.estimator import estimate

    failures = []

    for yaml_file in sorted(ANCHOR_DIR.glob("*.yaml")):
        anchor_data = _load_anchor(yaml_file)
        targets = anchor_data["targets"]
        anchor_step_time = targets.get("step_time_ms")
        tolerance = targets.get("tolerance", 0.15)

        if not anchor_step_time or anchor_step_time <= 0:
            continue
        if targets.get("strict_mfu_check") is False:
            print(f"{anchor_data['name']}: step_time SKIPPED (calibration mode)")
            continue

        model, system, strategy = load_anchor_config(yaml_file)
        report = estimate(model, system, strategy)
        rel_diff = abs(report.step_time_ms - anchor_step_time) / anchor_step_time

        print(f"{anchor_data['name']}: step_time={report.step_time_ms:.1f}ms "
              f"(target={anchor_step_time:.1f}ms, tol=±{tolerance:.0%}, "
              f"rel_diff={rel_diff:.1%})")

        if rel_diff > tolerance:
            failures.append(
                f"{anchor_data['name']}: step_time={report.step_time_ms:.1f}ms, "
                f"target={anchor_step_time:.1f}ms±{tolerance:.0%}, "
                f"rel_diff={rel_diff:.1%} > {tolerance:.0%}"
            )

    assert not failures, (
        f"{len(failures)} anchor(s) failed strict step_time check:\n" +
        "\n".join(failures)
    )
