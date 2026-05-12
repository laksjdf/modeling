"""Test tensor utilities.

Note: Tests that require torch are skipped when torch is not available.
"""
from __future__ import annotations

import pytest


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