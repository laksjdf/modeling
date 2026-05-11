"""Tests for python.zrt.report.report_types — all Phase 1 data classes."""
import pytest
from dataclasses import is_dataclass, fields
from python.zrt.report.report_types import (
    ReportContext, BlockDetail, SubStructureDetail,
    OpFamilyDetail, OpDetail,
)


# ═══════════════════════════════════════════════════════════════════════════════
# OpDetail — single operator atom
# ═══════════════════════════════════════════════════════════════════════════════

class TestOpDetail:
    """AC-1: OpDetail dataclass integrity."""

    def test_is_dataclass(self):
        assert is_dataclass(OpDetail)

    def test_default_construction(self):
        od = OpDetail(op_node_id="n1", op_type="aten.mm.default", scope="test.scope")
        assert od.op_node_id == "n1"
        assert od.op_type == "aten.mm.default"
        assert od.scope == "test.scope"
        assert od.layer == ""
        assert od.input_shapes == []
        assert od.output_shapes == []
        assert od.shape_desc == ""
        assert od.flops == 0
        assert od.read_bytes == 0
        assert od.write_bytes == 0
        assert od.compute_us == 0.0
        assert od.memory_us == 0.0
        assert od.latency_us == 0.0
        assert od.bound == ""
        assert od.confidence == 0.0

    def test_full_construction(self):
        od = OpDetail(
            op_node_id="n42",
            op_type="aten.bmm.default",
            scope="model.layers.0.self_attn",
            layer="0",
            input_shapes=["[32,128,512]", "[32,512,256]"],
            output_shapes=["[32,128,256]"],
            shape_desc="B=32, M=128, K=512, N=256",
            flops=4194304,
            read_bytes=65536,
            write_bytes=32768,
            compute_us=12.5,
            memory_us=3.2,
            latency_us=15.7,
            bound="compute",
            confidence=0.85,
        )
        assert od.op_node_id == "n42"
        assert od.flops == 4194304
        assert od.shape_desc == "B=32, M=128, K=512, N=256"
        assert od.latency_us == 15.7
        assert od.confidence == 0.85

    def test_field_count(self):
        """OpDetail should have all required fields."""
        field_names = {f.name for f in fields(OpDetail)}
        expected = {
            "op_node_id", "op_type", "scope", "layer",
            "input_shapes", "output_shapes", "shape_desc",
            "flops", "read_bytes", "write_bytes",
            "compute_us", "memory_us", "latency_us",
            "bound", "confidence",
        }
        assert expected.issubset(field_names)


# ═══════════════════════════════════════════════════════════════════════════════
# OpFamilyDetail — 12-column aggregated operator family
# ═══════════════════════════════════════════════════════════════════════════════

class TestOpFamilyDetail:
    """AC-1: OpFamilyDetail 12-column integrity."""

    def test_is_dataclass(self):
        assert is_dataclass(OpFamilyDetail)

    def test_default_construction(self):
        ofd = OpFamilyDetail(op_type="aten.mm.default")
        assert ofd.op_type == "aten.mm.default"
        assert ofd.display_name == ""
        assert ofd.category == "compute"
        assert ofd.count == 0
        assert ofd.repeat == 1
        assert ofd.first_scope == ""
        assert ofd.shape_desc == ""
        assert ofd.formula == ""
        assert ofd.io_formula == ""
        assert ofd.tflops == 0.0
        assert ofd.hbm_bytes == 0
        assert ofd.comm_bytes == 0
        assert ofd.compute_ms == 0.0
        assert ofd.memory_ms == 0.0
        assert ofd.comm_ms == 0.0
        assert ofd.total_ms == 0.0
        assert ofd.bound == ""
        assert ofd.confidence == 0.0
        assert ofd.pct_of_substructure == 0.0
        assert ofd.children == []

    def test_12_column_fields_present(self):
        """OpFamilyDetail must contain all 12 columns from the target report table."""
        ofd = OpFamilyDetail(
            op_type="aten.mm.default",
            display_name="Matrix Multiply",
            category="compute",
            count=4,
            repeat=61,
            first_scope="model.layers.0.self_attn",
            shape_desc="M=128, K=7168, N=2048",
            formula="2·M·K·N",
            io_formula="R=(M·K+K·N)·dtype  W=M·N·dtype",
            tflops=0.123,
            hbm_bytes=1048576,
            comm_bytes=0,
            compute_ms=0.5,
            memory_ms=0.2,
            comm_ms=0.0,
            total_ms=0.7,
            bound="compute",
            confidence=0.9,
            pct_of_substructure=45.0,
            children=[],
        )
        # The 12 columns: count, type, shape, formula, tflops, hbm, comm,
        # comp_ms, mem_ms, comm_ms, total_ms, bound
        assert ofd.count == 4
        assert ofd.display_name == "Matrix Multiply"
        assert ofd.shape_desc == "M=128, K=7168, N=2048"
        assert ofd.formula == "2·M·K·N"
        assert ofd.tflops == 0.123
        assert ofd.hbm_bytes == 1048576
        assert ofd.comm_bytes == 0
        assert ofd.compute_ms == 0.5
        assert ofd.memory_ms == 0.2
        assert ofd.comm_ms == 0.0
        assert ofd.total_ms == 0.7
        assert ofd.bound == "compute"

    def test_children_list_mutable_default(self):
        """Each instance gets its own children list (no shared mutable default)."""
        a = OpFamilyDetail(op_type="a")
        b = OpFamilyDetail(op_type="b")
        a.children.append(OpDetail(op_node_id="x", op_type="t", scope="s"))
        assert len(a.children) == 1
        assert len(b.children) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# SubStructureDetail — functional module within a Block
# ═══════════════════════════════════════════════════════════════════════════════

class TestSubStructureDetail:
    """AC-1: SubStructureDetail integrity."""

    def test_is_dataclass(self):
        assert is_dataclass(SubStructureDetail)

    def test_default_construction(self):
        ss = SubStructureDetail(name="Attention")
        assert ss.name == "Attention"
        assert ss.scope_group == ""
        assert ss.component_type == ""
        assert ss.total_ms == 0.0
        assert ss.pct_of_block == 0.0
        assert ss.op_families == []

    def test_with_op_families(self):
        fam = OpFamilyDetail(op_type="aten.mm.default")
        ss = SubStructureDetail(
            name="Attention",
            scope_group="self_attn",
            component_type="attn",
            total_ms=12.5,
            pct_of_block=60.0,
            op_families=[fam],
        )
        assert len(ss.op_families) == 1
        assert ss.op_families[0].op_type == "aten.mm.default"
        assert ss.total_ms == 12.5
        assert ss.pct_of_block == 60.0

    def test_op_families_mutable_default(self):
        a = SubStructureDetail(name="A")
        b = SubStructureDetail(name="B")
        a.op_families.append(OpFamilyDetail(op_type="test"))
        assert len(a.op_families) == 1
        assert len(b.op_families) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# BlockDetail — top-level model block
# ═══════════════════════════════════════════════════════════════════════════════

class TestBlockDetail:
    """AC-1: BlockDetail integrity."""

    def test_is_dataclass(self):
        assert is_dataclass(BlockDetail)

    def test_default_construction(self):
        bd = BlockDetail(name="MoEBlock")
        assert bd.name == "MoEBlock"
        assert bd.scope == ""
        assert bd.phase == ""
        assert bd.repeat == 1
        assert bd.total_ms == 0.0
        assert bd.pct_of_total == 0.0
        assert bd.dominant_bound == ""
        assert bd.sub_structures == []
        assert bd.model_params == {}

    def test_with_sub_structures(self):
        ss = SubStructureDetail(name="Attention", total_ms=10.0)
        bd = BlockDetail(
            name="MoEBlock",
            scope="model.layers.0",
            phase="decode",
            repeat=61,
            total_ms=500.0,
            pct_of_total=80.0,
            dominant_bound="compute",
            sub_structures=[ss],
            model_params={"num_experts": 384, "active_per_token": 6},
        )
        assert len(bd.sub_structures) == 1
        assert bd.repeat == 61
        assert bd.total_ms == 500.0
        assert bd.model_params["num_experts"] == 384

    def test_sub_structures_mutable_default(self):
        a = BlockDetail(name="A")
        b = BlockDetail(name="B")
        a.sub_structures.append(SubStructureDetail(name="test"))
        assert len(a.sub_structures) == 1
        assert len(b.sub_structures) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# ReportContext — top-level report container
# ═══════════════════════════════════════════════════════════════════════════════

class TestReportContext:
    """AC-1: ReportContext integrity."""

    def test_is_dataclass(self):
        assert is_dataclass(ReportContext)

    def test_default_construction(self):
        rc = ReportContext()
        assert rc.model == ""
        assert rc.hardware == ""
        assert rc.phase == ""
        assert rc.parallel_desc == ""
        assert rc.topology_desc == ""
        assert rc.batch_size == 1
        assert rc.seq_len == 8192
        assert rc.active_params == 0
        assert rc.total_params == 0
        assert rc.prefill_ms is None
        assert rc.tpot_ms is None
        assert rc.mtp_adjusted_tpot_ms is None
        assert rc.tokens_per_sec == 0.0
        assert rc.memory_per_gpu_gb == 0.0
        assert rc.model_blocks == 0
        assert rc.compute_pct == 0.0
        assert rc.memory_pct == 0.0
        assert rc.communication_pct == 0.0
        assert rc.blocks == []
        assert rc.calibration == []
        assert rc.warnings == []

    def test_full_construction(self):
        rc = ReportContext(
            model="DeepSeek-V3",
            hardware="nvidia_h100_sxm",
            phase="decode",
            parallel_desc="TP8-EP8-PP4",
            topology_desc="2Node-16GPU",
            batch_size=128,
            seq_len=4096,
            active_params=49_000_000_000,
            total_params=1_600_000_000_000,
            prefill_ms=None,
            tpot_ms=17.5,
            mtp_adjusted_tpot_ms=11.2,
            tokens_per_sec=7262.3,
            memory_per_gpu_gb=52.0,
            model_blocks=61,
            compute_pct=65.0,
            memory_pct=25.0,
            communication_pct=10.0,
            mtp_depth=3,
            mtp_acceptance_rate=0.65,
            mtp_effective_tokens=2.3,
            blocks=[
                BlockDetail(name="Embedding", total_ms=0.003),
                BlockDetail(name="MoEBlock", repeat=61, total_ms=50.74),
                BlockDetail(name="Output", total_ms=0.05),
            ],
            warnings=["MFU anomaly detected"],
        )
        assert rc.model == "DeepSeek-V3"
        assert rc.tpot_ms == 17.5
        assert rc.mtp_adjusted_tpot_ms == 11.2
        assert rc.mtp_depth == 3
        assert rc.mtp_acceptance_rate == 0.65
        assert rc.compute_pct == 65.0
        assert len(rc.blocks) == 3
        assert rc.blocks[1].name == "MoEBlock"
        assert rc.blocks[1].repeat == 61

    def test_blocks_mutable_default(self):
        a = ReportContext()
        b = ReportContext()
        a.blocks.append(BlockDetail(name="test"))
        assert len(a.blocks) == 1
        assert len(b.blocks) == 0

    def test_mtp_fields_defaults(self):
        rc = ReportContext()
        assert rc.mtp_depth == 1
        assert rc.mtp_acceptance_rate == 0.0
        assert rc.mtp_effective_tokens == 1.0

    def test_field_count(self):
        """ReportContext should cover metadata, KPI, bound, hierarchy, and refs."""
        field_names = {f.name for f in fields(ReportContext)}
        required = {
            "model", "hardware", "phase", "parallel_desc", "topology_desc",
            "batch_size", "seq_len", "active_params", "total_params",
            "prefill_ms", "tpot_ms", "mtp_adjusted_tpot_ms",
            "tokens_per_sec", "memory_per_gpu_gb", "model_blocks",
            "mtp_depth", "mtp_acceptance_rate", "mtp_effective_tokens",
            "compute_pct", "memory_pct", "communication_pct",
            "blocks", "calibration", "warnings",
        }
        assert required.issubset(field_names)
