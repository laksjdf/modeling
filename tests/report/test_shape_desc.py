"""Tests for python.zrt.report.shape_desc — AC-3: Shape description generation."""

import pytest
from unittest.mock import MagicMock, PropertyMock
from python.zrt.ir.types import TensorMeta, DType
from python.zrt.report.shape_desc import (
    describe_shapes,
    describe_shapes_from_tensors,
    _desc_mm, _desc_bmm, _desc_norm, _desc_scalar,
    _desc_attention, _desc_comm, _desc_fallback,
)


def _tm(shape: tuple[int, ...], dtype: DType = DType.BF16) -> TensorMeta:
    """Helper: create a TensorMeta with given shape."""
    return TensorMeta.from_shape_dtype("t0", shape, dtype)


def _node(op_type: str, inputs=None, outputs=None, attrs=None, category="compute") -> MagicMock:
    """Helper: create a mock OpNode."""
    node = MagicMock()
    node.op_type = op_type
    node.inputs = inputs or []
    node.outputs = outputs or []
    node.attrs = attrs or {}
    node.category = category
    return node


# ═══════════════════════════════════════════════════════════════════════════════
# AC-3: Shape descriptions
# ═══════════════════════════════════════════════════════════════════════════════

class TestAC3ShapeDesc:
    """AC-3: Shape description generation for all op categories."""

    def test_mm_shape(self):
        """mm → M=128, K=7168, N=2048"""
        node = _node("aten.mm.default",
                     inputs=[_tm((128, 7168)), _tm((7168, 2048))],
                     outputs=[_tm((128, 2048))])
        desc = describe_shapes(node)
        assert desc == "M=128, K=7168, N=2048"

    def test_matmul_shape(self):
        """matmul uses same descriptor as mm."""
        node = _node("aten.matmul",
                     inputs=[_tm((128, 7168)), _tm((7168, 2048))],
                     outputs=[_tm((128, 2048))])
        desc = describe_shapes(node)
        assert desc == "M=128, K=7168, N=2048"

    def test_addmm_shape(self):
        """addmm → same mm descriptor."""
        node = _node("aten.addmm.default",
                     inputs=[_tm((128, 7168)), _tm((7168, 2048))],
                     outputs=[_tm((128, 2048))])
        desc = describe_shapes(node)
        assert "M=" in desc and "K=" in desc and "N=" in desc

    def test_bmm_shape(self):
        """bmm → B=32, M=128, K=7168, N=2048"""
        node = _node("aten.bmm.default",
                     inputs=[_tm((32, 128, 7168)), _tm((32, 7168, 2048))],
                     outputs=[_tm((32, 128, 2048))])
        desc = describe_shapes(node)
        assert desc == "B=32, M=128, K=7168, N=2048"

    def test_norm_shape(self):
        """rms_norm → N=7168"""
        node = _node("aten.rms_norm.default",
                     inputs=[_tm((128, 7168))],
                     outputs=[_tm((128, 7168))])
        desc = describe_shapes(node)
        assert desc == "N=7168"

    def test_layer_norm_shape(self):
        """layer_norm → N=7168"""
        node = _node("aten.layer_norm",
                     inputs=[_tm((128, 7168))],
                     outputs=[_tm((128, 7168))])
        desc = describe_shapes(node)
        assert desc == "N=7168"

    def test_add_rms_norm_shape(self):
        """add_rms_norm → N=7168"""
        node = _node("add_rms_norm",
                     inputs=[_tm((128, 7168))],
                     outputs=[_tm((128, 7168))])
        desc = describe_shapes(node)
        assert desc == "N=7168"

    def test_softmax_shape(self):
        """softmax → N=128"""
        node = _node("aten._softmax",
                     inputs=[_tm((1, 24, 128, 128))],
                     outputs=[_tm((1, 24, 128, 128))])
        desc = describe_shapes(node)
        assert desc == "N=128"

    def test_silu_shape(self):
        """silu → N=128"""
        node = _node("aten.silu",
                     inputs=[_tm((1, 128))],
                     outputs=[_tm((1, 128))])
        desc = describe_shapes(node)
        assert desc == "N=128"

    def test_gelu_shape(self):
        """gelu → N=128"""
        node = _node("aten.gelu",
                     inputs=[_tm((1, 128))],
                     outputs=[_tm((1, 128))])
        desc = describe_shapes(node)
        assert desc == "N=128"

    def test_attention_shape(self):
        """SDPA → B=1, H=24, Sq=128, D=128"""
        node = _node("scaled_dot_product_attention",
                     inputs=[_tm((1, 24, 128, 128))],
                     outputs=[_tm((1, 24, 128, 128))])
        desc = describe_shapes(node)
        assert "B=1" in desc
        assert "H=24" in desc
        assert "Sq=128" in desc
        assert "D=128" in desc

    def test_flash_attn_shape(self):
        node = _node("flash_attn",
                     inputs=[_tm((1, 32, 4096, 128))],
                     outputs=[_tm((1, 32, 4096, 128))])
        desc = describe_shapes(node)
        assert "B=1" in desc
        assert "H=32" in desc
        assert "Sq=4096" in desc

    def test_comm_all_reduce_shape(self):
        """comm.all_reduce → data=X MB, group=N"""
        node = _node("comm.all_reduce",
                     inputs=[_tm((128, 7168))],
                     outputs=[_tm((128, 7168))],
                     attrs={"group_size": 4},
                     category="communication")
        desc = describe_shapes(node)
        assert "data=" in desc
        assert "group=4" in desc

    def test_embedding_shape(self):
        """embedding → tokens=128, hidden=7168"""
        node = _node("aten.embedding",
                     inputs=[_tm((128,))],
                     outputs=[_tm((128, 7168))])
        desc = describe_shapes(node)
        assert desc == "tokens=128, hidden=7168"

    def test_fallback_shape(self):
        """Unknown op → in=[...] out=[...]"""
        node = _node("custom.op",
                     inputs=[_tm((128, 7168))],
                     outputs=[_tm((128, 2048))])
        desc = describe_shapes(node)
        assert "in=[" in desc
        assert "out=[" in desc

    def test_describe_shapes_from_tensors(self):
        """describe_from_tensors for standalone use."""
        desc = describe_shapes_from_tensors(
            inputs=[_tm((128, 7168)), _tm((7168, 2048))],
            outputs=[_tm((128, 2048))],
        )
        assert "in=[" in desc
        assert "out=[" in desc

    def test_describe_shapes_from_tensors_empty(self):
        """Empty inputs/outputs returns empty string."""
        desc = describe_shapes_from_tensors([], [])
        assert desc == ""


class TestCommSizeFormatting:
    """Communication ops format data sizes correctly."""

    def test_mb_size(self):
        node = _node("comm.all_reduce",
                     inputs=[_tm((1024, 1024, 1024))],
                     outputs=[_tm((1024, 1024, 1024))],
                     category="communication")
        desc = describe_shapes(node)
        assert "MB" in desc

    def test_kb_size(self):
        node = _node("comm.all_reduce",
                     inputs=[_tm((1024,))],
                     outputs=[_tm((1024,))],
                     category="communication")
        desc = describe_shapes(node)
        # small data should show KB or B
        assert ("KB" in desc or "B," in desc)

    def test_rope_shape(self):
        node = _node("rotary_embedding",
                     inputs=[_tm((1, 24, 128, 64))],
                     outputs=[_tm((1, 24, 128, 64))])
        desc = describe_shapes(node)
        assert "N=64" in desc or "64" in desc
