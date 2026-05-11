"""Tests for python.zrt.report.formula_registry — AC-2: FormulaRegistry coverage."""

import pytest
from python.zrt.report.formula_registry import FormulaRegistry, FormulaEntry


@pytest.fixture
def reg() -> FormulaRegistry:
    return FormulaRegistry()


class TestFormulaRegistryAC2:
    """AC-2: FormulaRegistry coverage for key op_types."""

    def test_lookup_mm(self, reg):
        entry = reg.lookup("aten.mm.default")
        assert entry is not None
        assert entry.display_name == "Matrix Multiply"
        assert "2·M·K·N" in entry.flops_formula
        assert entry.category == "compute"

    def test_lookup_addmm(self, reg):
        entry = reg.lookup("aten.addmm.default")
        assert entry is not None
        assert entry.display_name == "AddMM"
        assert "2·M·K·N" in entry.flops_formula
        assert "bias" in entry.notes

    def test_lookup_bmm(self, reg):
        entry = reg.lookup("aten.bmm.default")
        assert entry is not None
        assert entry.display_name == "Batch Matrix Multiply"
        assert "B·M·K·N" in entry.flops_formula

    def test_lookup_linear(self, reg):
        entry = reg.lookup("aten.linear")
        assert entry is not None
        assert entry.display_name == "Linear"
        assert "batch·I·O" in entry.flops_formula

    def test_lookup_rms_norm(self, reg):
        entry = reg.lookup("aten.rms_norm.default")
        assert entry is not None
        assert entry.display_name == "RMS Norm"
        assert entry.flops_formula == "4·N"
        assert entry.category == "compute"

    def test_lookup_layer_norm(self, reg):
        entry = reg.lookup("aten.layer_norm")
        assert entry is not None
        assert entry.display_name == "Layer Norm"
        assert entry.flops_formula == "5·N"

    def test_lookup_softmax(self, reg):
        entry = reg.lookup("aten._softmax")
        assert entry is not None
        assert entry.display_name == "Softmax"
        assert entry.flops_formula == "5·N"

    def test_lookup_silu(self, reg):
        entry = reg.lookup("aten.silu")
        assert entry is not None
        assert entry.display_name == "SiLU"
        assert entry.flops_formula == "4·N"

    def test_lookup_gelu(self, reg):
        entry = reg.lookup("aten.gelu")
        assert entry is not None
        assert entry.display_name == "GELU"
        assert entry.flops_formula == "4·N"

    def test_lookup_all_reduce(self, reg):
        entry = reg.lookup("comm.all_reduce")
        assert entry is not None
        assert entry.display_name == "AllReduce"
        assert entry.category == "communication"
        assert entry.flops_formula == "—"
        assert "(n-1)/n" in entry.io_formula

    def test_lookup_sdpa(self, reg):
        entry = reg.lookup("scaled_dot_product_attention")
        assert entry is not None
        assert "Attention" in entry.display_name

    def test_lookup_flash_attn(self, reg):
        entry = reg.lookup("flash_attn")
        assert entry is not None
        assert entry.display_name == "Flash Attention"

    def test_lookup_topk(self, reg):
        entry = reg.lookup("aten.topk")
        assert entry is not None
        assert entry.display_name == "TopK"
        assert "log₂" in entry.flops_formula

    def test_lookup_embedding(self, reg):
        entry = reg.lookup("aten.embedding")
        assert entry is not None
        assert entry.display_name == "Embedding Lookup"
        assert entry.category == "memory"
        assert entry.flops_formula == "0"

    def test_lookup_elementwise(self, reg):
        entry = reg.lookup("aten.add.Tensor")
        assert entry is not None
        assert "Element-wise" in entry.display_name
        assert entry.flops_formula == "1·N"

    def test_lookup_moe_gate(self, reg):
        entry = reg.lookup("moe_gate_topk")
        assert entry is not None
        assert entry.display_name == "MoE Gate + TopK"
        assert entry.category == "compute"

    def test_lookup_moe_dispatch(self, reg):
        entry = reg.lookup("moe_dispatch")
        assert entry is not None
        assert entry.display_name == "MoE Dispatch"
        assert entry.category == "communication"

    def test_lookup_comm_ops(self, reg):
        """All communication ops are discoverable."""
        comm_ops = ["comm.all_gather", "comm.reduce_scatter",
                    "comm.all_to_all", "comm.send_recv", "comm.broadcast"]
        for op in comm_ops:
            entry = reg.lookup(op)
            assert entry is not None, f"Missing entry for {op}"
            assert entry.category == "communication"

    def test_lookup_unknown_returns_none(self, reg):
        """AC-2: Unknown op_type returns None."""
        entry = reg.lookup("nonexistent.op")
        assert entry is None

    def test_lookup_unknown_aten_op_returns_catchall(self, reg):
        """Unknown aten op returns catch-all entry (not None)."""
        entry = reg.lookup("aten.nonexistent_op")
        assert entry is not None
        assert entry.flops_formula == "?"
        assert entry.io_formula == "?"

    def test_display_info_unknown(self, reg):
        """display_info on unknown op returns sensible defaults."""
        info = reg.display_info("custom.op")
        assert info["display_name"] == "op"
        assert info["category"] == "compute"
        assert info["flops_formula"] == "?"

    def test_display_info_known(self, reg):
        info = reg.display_info("aten.mm.default")
        assert info["display_name"] == "Matrix Multiply"
        assert "2·M·K·N" in info["flops_formula"]

    def test_case_insensitive_match(self, reg):
        """Regex matches should be case-insensitive."""
        entry = reg.lookup("ATEN.MM.DEFAULT")
        assert entry is not None
        assert entry.display_name == "Matrix Multiply"

    def test_gated_mlp(self, reg):
        entry = reg.lookup("gated_mlp")
        assert entry is not None
        assert entry.display_name == "Gated MLP (SwiGLU)"
        assert "I·O" in entry.flops_formula

    def test_rope(self, reg):
        entry = reg.lookup("rotary_embedding")
        assert entry is not None
        assert entry.display_name == "RoPE"
        assert entry.flops_formula == "2·N"

    def test_lm_head(self, reg):
        entry = reg.lookup("lm_head")
        assert entry is not None
        assert entry.display_name == "LM Head"


class TestFormulaEntry:
    def test_fields_match_spec(self):
        entry = FormulaEntry(
            op_pattern="test",
            display_name="Test Op",
            category="compute",
            flops_formula="2·N",
            io_formula="N",
            notes="test notes",
        )
        assert entry.op_pattern == "test"
        assert entry.display_name == "Test Op"
        assert entry.category == "compute"
        assert entry.notes == "test notes"
