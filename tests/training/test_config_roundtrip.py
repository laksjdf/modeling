"""Test config loading and roundtrip."""

import pytest
import tempfile
import os
from pathlib import Path

from zrt.training.io.config_loader import load_specs, _parse_layers
from zrt.training.search.estimator import estimate
from zrt.training.search.report import report_to_dict, report_to_json, report_summary
from zrt.training.spec.model import LayerKind


CONFIGS_DIR = Path(__file__).parent.parent.parent / "python" / "zrt" / "training" / "configs"


def test_parse_layers_list():
    layers = _parse_layers(["dense", "moe", "mtp"])
    assert layers == [LayerKind.DENSE, LayerKind.MOE, LayerKind.MTP]


def test_parse_layers_star_bracket():
    layers = _parse_layers("[dense]*3+[moe]*2+[mtp]")
    assert layers == [LayerKind.DENSE]*3 + [LayerKind.MOE]*2 + [LayerKind.MTP]


def test_parse_layers_bracket_star():
    layers = _parse_layers("3*[dense]+2*[moe]")
    assert layers == [LayerKind.DENSE]*3 + [LayerKind.MOE]*2


def test_config_loading():
    """Test loading the combined config file."""
    config_path = CONFIGS_DIR / "llama3_70b_3d.yaml"
    if not config_path.exists():
        pytest.skip("Config file not found")

    model, system, strategy = load_specs(config_path)

    assert model.hidden == 8192
    assert model.num_heads == 64
    assert len(model.layers) == 80
    assert system.gpu.name == "NVIDIA H100 SXM"
    assert system.world_size == 64
    assert strategy.tp == 8
    assert strategy.pp == 2


def test_estimate_and_report():
    """Full roundtrip: load config → estimate → report."""
    config_path = CONFIGS_DIR / "llama3_70b_3d.yaml"
    if not config_path.exists():
        pytest.skip("Config file not found")

    model, system, strategy = load_specs(config_path)
    report = estimate(model, system, strategy)

    assert report.step_time_ms > 0
    assert 0 < report.mfu <= 1.0
    assert report.memory is not None
    assert report.memory.total > 0

    # Test report formats
    d = report_to_dict(report)
    assert "step_time_ms" in d
    assert "mfu" in d
    assert "memory" in d

    summary = report_summary(report)
    assert "Training Estimation Report" in summary
    assert "Step time" in summary

    # Test JSON output
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        report_to_json(report, f.name)
        assert os.path.exists(f.name)
        assert os.path.getsize(f.name) > 0
        os.unlink(f.name)
