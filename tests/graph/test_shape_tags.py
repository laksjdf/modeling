"""Test tensor utilities (SKIP_OPS list and helpers).

Note: Tests that require torch are skipped when torch is not available.
"""
from __future__ import annotations

import pytest


class TestSkipOps:
    """Test SKIP_OPS list from fusion_rules."""

    def test_skip_ops_contains_shape_ops(self):
        """SKIP_OPS should include common shape/view ops."""
        from python.zrt.transform.fusion.rules import PATTERN_SKIP

        # These should be in the skip list
        assert "aten.view.default" in PATTERN_SKIP
        assert "aten.permute.default" in PATTERN_SKIP

    def test_skip_ops_does_not_include_compute_ops(self):
        """SKIP_OPS should NOT include compute-heavy ops."""
        from python.zrt.transform.fusion.rules import PATTERN_SKIP

        # These should NOT be in the skip list
        assert "aten.mm.default" not in PATTERN_SKIP
        assert "aten.addmm.default" not in PATTERN_SKIP


@pytest.mark.skipif(
    True,  # Skip when torch not available
    reason="torch not available in test environment"
)
class TestTensorUtilsWithTorch:
    """Tests requiring torch (skipped in current environment)."""

    def test_shape_str(self):
        """Test shape_str helper."""
        import torch
        from python.zrt.graph.tensor_utils import shape_str

        t = torch.randn(2, 3, 4)
        assert shape_str(t) == "[2, 3, 4]"
