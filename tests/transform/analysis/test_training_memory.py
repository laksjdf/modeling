"""Tests for TrainingMemoryPass layer scaling."""
import pytest
from unittest.mock import MagicMock, patch

from python.zrt.transform.analysis.training import TrainingMemoryPass, TrainingMemoryBreakdown


def _make_graph_and_ctx(num_layers=61, num_layers_traced=4):
    """Create a minimal mock graph + context for testing."""
    g = MagicMock()
    g.metadata = {
        "num_layers": num_layers,
        "num_layers_traced": num_layers_traced,
    }
    g.clone.return_value = g
    g.nodes = {}
    g.edges = []
    # No zero metadata → use fallback path

    ctx = MagicMock()
    ctx.parallel.dp = 1
    ctx.parallel.tp = 1
    ctx.parallel.cp = 1
    ctx.parallel.pp = 1
    ctx.training = None
    ctx.hw_spec = None
    ctx.is_training = True
    return g, ctx


def test_memory_pass_layer_scale():
    """TrainingMemoryPass should scale total_params by layer_scale.

    With num_layers=61 and num_layers_traced=4, layer_scale = 61/4 = 15.25.
    """
    g, ctx = _make_graph_and_ctx(num_layers=61, num_layers_traced=4)

    with patch("python.zrt.transform.analysis.training.count_params") as mock_count:
        # Simulate: traced 4 layers = 100M params
        mock_count.return_value = 100_000_000

        result = TrainingMemoryPass().run(g, ctx)

        # layer_scale = 61/4 = 15.25
        # total_params = 100M * 15.25 = 1,525,000,000
        expected_total = int(100_000_000 * 61 / 4)

        # weights_bytes = total_params * 2 / (pp * tp) = 1.525B * 2 / 1
        expected_weights = expected_total * 2  # BF16 = 2 bytes

        breakdown = result.metadata["memory_breakdown"]
        assert breakdown.weights == pytest.approx(expected_weights, rel=0.01)


def test_memory_pass_no_scale_when_full_model():
    """When num_layers == num_layers_traced, no scaling should occur."""
    g, ctx = _make_graph_and_ctx(num_layers=4, num_layers_traced=4)

    with patch("python.zrt.transform.analysis.training.count_params") as mock_count:
        mock_count.return_value = 100_000_000

        result = TrainingMemoryPass().run(g, ctx)

        # No scaling: total_params = 100M
        expected_weights = 100_000_000 * 2
        assert result.metadata["memory_breakdown"].weights == pytest.approx(
            expected_weights, rel=0.01
        )


def test_memory_pass_uses_metadata_dtype():
    """param_dtype should come from metadata when available."""
    g, ctx = _make_graph_and_ctx(num_layers=4, num_layers_traced=4)
    g.metadata["param_dtype_bytes"] = 1  # FP8

    with patch("python.zrt.transform.analysis.training.count_params") as mock_count:
        mock_count.return_value = 100_000_000

        result = TrainingMemoryPass().run(g, ctx)

        # weights = params * 1 (FP8) instead of params * 2 (BF16)
        assert result.metadata["memory_breakdown"].weights == pytest.approx(
            100_000_000, rel=0.01
        )


def test_memory_pass_grad_scaling():
    """Gradient memory should also be scaled by layer_scale."""
    g, ctx = _make_graph_and_ctx(num_layers=61, num_layers_traced=4)

    with patch("python.zrt.transform.analysis.training.count_params") as mock_count:
        mock_count.return_value = 100_000_000

        result = TrainingMemoryPass().run(g, ctx)

        expected_total = int(100_000_000 * 61 / 4)
        expected_grads = expected_total * 2  # BF16 = 2 bytes

        breakdown = result.metadata["memory_breakdown"]
        assert breakdown.grads == pytest.approx(expected_grads, rel=0.01)
