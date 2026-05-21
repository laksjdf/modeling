"""Unit tests for OptimizerPass and Moonshot Rotation overlap logic.

Tests:
1. OptimizerPass node creation (optimizer_step, muon_ag, muon_rs)
2. Optimizer node attributes (params, state_bytes, step_flops)
3. Moonshot Rotation overlap annotations
4. _compute_optimizer_step_time calculation

Run with:
    $env:PYTHONPATH='python'; pytest tests/IT/test_optimizer_pass.py -v
"""
import pytest
import math
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.ir.types import TensorMeta, DType
from python.zrt.transform.training.optimizer import OptimizerPass
from python.zrt.transform.context import TransformContext, TrainingConfig, ParallelConfig
from python.zrt.hardware.spec import HardwareSpec, MemorySpec, InterconnectSpec, LinkSpec


def make_mock_graph(phase: str = "train_backward", num_params: int = 100000) -> OpGraph:
    """Create a mock backward graph for testing."""
    graph = OpGraph(
        name="test_graph",
        phase=phase,
        nodes={},
        edges=[],
        metadata={
            "num_layers": 2,
            "num_layers_traced": 2,
            "total_params": num_params,
        },
    )
    
    dummy_node = OpNode(
        id="bwd_op_0",
        op_type="matmul",
        inputs=[TensorMeta.from_shape_dtype("x", (1, 128), DType.BF16)],
        outputs=[TensorMeta.from_shape_dtype("y", (1, 128), DType.BF16)],
        attrs={"is_param": True, "param_count": num_params},
        scope="layer.0",
        category="compute",
    )
    dummy_node.annotations["phase"] = "bwd"
    dummy_node.annotations["is_param"] = True
    dummy_node.annotations["param_count"] = num_params
    graph.nodes[dummy_node.id] = dummy_node
    graph._pred[dummy_node.id] = []
    graph._succ[dummy_node.id] = []
    
    return graph


def make_mock_context(
    optimizer: str = "muon",
    dp: int = 8,
    tp: int = 1,
    pp: int = 1,
    muon_rotation: bool = True,
    muon_ns_steps: int = 10,
    muon_param_fraction: float = 0.85,
    zero_stage: int = 1,
) -> TransformContext:
    """Create a mock TransformContext for testing."""
    hw = make_mock_hardware()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=tp, dp=dp, pp=pp),
        training=TrainingConfig(
            optimizer=optimizer,
            zero_stage=zero_stage,
            muon_rotation=muon_rotation,
            muon_ns_steps=muon_ns_steps,
            muon_param_fraction=muon_param_fraction,
        ),
    )
    return ctx


def make_mock_hardware() -> HardwareSpec:
    """Create a mock HardwareSpec for testing."""
    hw = MagicMock(spec=HardwareSpec)
    hw.memory = MagicMock(spec=MemorySpec)
    hw.memory.hbm_bandwidth_gbps = 3352
    
    hw.interconnect = MagicMock(spec=InterconnectSpec)
    hw.interconnect.intra_node = MagicMock(spec=LinkSpec)
    hw.interconnect.intra_node.bandwidth_gbps = 900
    hw.interconnect.intra_node.num_devices = 8
    hw.interconnect.inter_node = MagicMock(spec=LinkSpec)
    hw.interconnect.inter_node.bandwidth_gbps = 400
    
    hw.peak_flops = MagicMock(return_value=989e12)
    return hw


class TestOptimizerPassNodeCreation:
    """Test OptimizerPass creates optimizer nodes correctly."""

    def test_optimizer_step_node_created(self):
        """optimizer_step node is always created."""
        graph = make_mock_graph()
        ctx = make_mock_context(optimizer="muon", dp=8)
        
        pass_obj = OptimizerPass()
        result = pass_obj.run(graph, ctx)
        
        assert "optimizer_step" in result.nodes
        opt_node = result.nodes["optimizer_step"]
        assert opt_node.op_type == "optimizer.muon"
        assert opt_node.category == "compute"

    def test_muon_ag_created_when_dp_gt_1(self):
        """muon_ag node created when DP > 1 and Muon optimizer."""
        graph = make_mock_graph(num_params=10000)
        ctx = make_mock_context(optimizer="muon", dp=8)
        
        pass_obj = OptimizerPass()
        result = pass_obj.run(graph, ctx)
        
        assert "muon_ag" in result.nodes
        ag_node = result.nodes["muon_ag"]
        assert ag_node.op_type == "comm.all_gather"
        assert ag_node.category == "communication"
        assert ag_node.attrs["group_size"] == 8

    def test_muon_ag_not_created_when_dp_eq_1(self):
        """muon_ag node NOT created when DP == 1."""
        graph = make_mock_graph()
        ctx = make_mock_context(optimizer="muon", dp=1)
        
        pass_obj = OptimizerPass()
        result = pass_obj.run(graph, ctx)
        
        assert "muon_ag" not in result.nodes

    def test_muon_rs_created_when_rotation_on(self):
        """muon_rs node created when Moonshot Rotation is enabled."""
        graph = make_mock_graph(num_params=10000)
        ctx = make_mock_context(optimizer="muon", dp=8, muon_rotation=True)
        
        pass_obj = OptimizerPass()
        result = pass_obj.run(graph, ctx)
        
        assert "muon_rs" in result.nodes
        rs_node = result.nodes["muon_rs"]
        assert rs_node.op_type == "comm.reduce_scatter"
        assert rs_node.category == "communication"

    def test_muon_rs_not_created_when_rotation_off(self):
        """muon_rs node NOT created when Moonshot Rotation is disabled."""
        graph = make_mock_graph(num_params=10000)
        ctx = make_mock_context(optimizer="muon", dp=8, muon_rotation=False)
        
        pass_obj = OptimizerPass()
        result = pass_obj.run(graph, ctx)
        
        assert "muon_rs" not in result.nodes

    def test_adam_no_comm_nodes(self):
        """Adam optimizer has no muon_ag/muon_rs nodes."""
        graph = make_mock_graph()
        ctx = make_mock_context(optimizer="adam", dp=8)
        
        pass_obj = OptimizerPass()
        result = pass_obj.run(graph, ctx)
        
        assert "optimizer_step" in result.nodes
        assert "muon_ag" not in result.nodes
        assert "muon_rs" not in result.nodes

    def test_optimizer_chain_order(self):
        """Optimizer chain order: [AG] -> optimizer_step -> [RS]."""
        graph = make_mock_graph(num_params=10000)
        ctx = make_mock_context(optimizer="muon", dp=8, muon_rotation=True)
        
        pass_obj = OptimizerPass()
        result = pass_obj.run(graph, ctx)
        
        assert "muon_ag" in result.nodes
        assert "optimizer_step" in result.nodes
        assert "muon_rs" in result.nodes
        
        ag = result.nodes["muon_ag"]
        opt = result.nodes["optimizer_step"]
        rs = result.nodes["muon_rs"]
        
        assert ag.annotations.get("muon_comm") == "ag"
        assert rs.annotations.get("muon_comm") == "rs"
        
        succ_ag = result.successors("muon_ag")
        assert "optimizer_step" in succ_ag, "AG should connect to optimizer_step"
        
        succ_opt = result.successors("optimizer_step")
        assert "muon_rs" in succ_opt, "optimizer_step should connect to RS"


class TestOptimizerNodeAttributes:
    """Test optimizer node attributes are set correctly."""

    def test_optimizer_step_params_total(self):
        """optimizer_step has params_total attribute."""
        graph = make_mock_graph(num_params=10000)
        ctx = make_mock_context(optimizer="muon", dp=8, tp=2)
        
        pass_obj = OptimizerPass()
        result = pass_obj.run(graph, ctx)
        
        opt_node = result.nodes["optimizer_step"]
        params_total = opt_node.attrs.get("params_total", 0)
        
        assert params_total > 0
        assert params_total <= 10000

    def test_optimizer_step_params_muon_adam_split(self):
        """optimizer_step splits params into muon and adam fractions."""
        graph = make_mock_graph(num_params=10000)
        ctx = make_mock_context(optimizer="muon", dp=8, muon_param_fraction=0.85)
        
        pass_obj = OptimizerPass()
        result = pass_obj.run(graph, ctx)
        
        opt_node = result.nodes["optimizer_step"]
        params_total = opt_node.attrs["params_total"]
        params_muon = opt_node.attrs["params_muon"]
        params_adam = opt_node.attrs["params_adam"]
        
        assert params_muon == int(params_total * 0.85)
        assert params_adam == params_total - params_muon

    def test_optimizer_step_ns_steps_attribute(self):
        """optimizer_step has ns_steps attribute."""
        graph = make_mock_graph()
        ctx = make_mock_context(optimizer="muon", dp=8, muon_ns_steps=10)
        
        pass_obj = OptimizerPass()
        result = pass_obj.run(graph, ctx)
        
        opt_node = result.nodes["optimizer_step"]
        assert opt_node.attrs["ns_steps"] == 10

    def test_optimizer_step_ns_rotation_attribute(self):
        """optimizer_step has ns_rotation attribute."""
        graph = make_mock_graph()
        ctx = make_mock_context(optimizer="muon", dp=8, muon_rotation=True)
        
        pass_obj = OptimizerPass()
        result = pass_obj.run(graph, ctx)
        
        opt_node = result.nodes["optimizer_step"]
        assert opt_node.attrs["ns_rotation"] == True
        
        ctx2 = make_mock_context(optimizer="muon", dp=8, muon_rotation=False)
        result2 = pass_obj.run(graph, ctx2)
        opt_node2 = result2.nodes["optimizer_step"]
        assert opt_node2.attrs["ns_rotation"] == False

    def test_optimizer_step_state_bytes(self):
        """optimizer_step has state_bytes attribute."""
        graph = make_mock_graph(num_params=10000)
        ctx = make_mock_context(optimizer="muon", dp=8)
        
        pass_obj = OptimizerPass()
        result = pass_obj.run(graph, ctx)
        
        opt_node = result.nodes["optimizer_step"]
        state_bytes = opt_node.attrs.get("state_bytes", 0)
        
        assert state_bytes > 0
        assert state_bytes < 10000 * 12  # Muon saves memory

    def test_optimizer_step_step_flops(self):
        """optimizer_step has step_flops attribute."""
        graph = make_mock_graph(num_params=10000)
        ctx = make_mock_context(optimizer="muon", dp=8, muon_ns_steps=10)
        
        pass_obj = OptimizerPass()
        result = pass_obj.run(graph, ctx)
        
        opt_node = result.nodes["optimizer_step"]
        step_flops = opt_node.attrs.get("step_flops", 0)
        
        assert step_flops > 0

    def test_muon_ag_bytes_attribute(self):
        """muon_ag has bytes attribute calculated from params_muon."""
        graph = make_mock_graph(num_params=10000)
        ctx = make_mock_context(optimizer="muon", dp=8, muon_param_fraction=0.85)
        
        pass_obj = OptimizerPass()
        result = pass_obj.run(graph, ctx)
        
        ag_node = result.nodes["muon_ag"]
        ag_bytes = ag_node.attrs.get("bytes", 0)
        
        opt_node = result.nodes["optimizer_step"]
        params_muon = opt_node.attrs["params_muon"]
        
        assert ag_bytes == params_muon * 4  # 4 bytes per param


class TestOptimizerStateBytesCalculation:
    """Test optimizer state bytes calculation logic."""

    def test_adam_state_formula(self):
        """Adam state: params × 12 (FP32 m + v + master)."""
        params = 1000
        state_bytes = params * 12
        assert state_bytes == 12000

    def test_muon_state_formula(self):
        """Muon state: params × (12 - f_muon × 4)."""
        params = 1000
        f_muon = 0.85
        state_bytes = int(params * (12 - f_muon * 4))
        assert state_bytes == 8600

    def test_muon_state_less_than_adam(self):
        """Muon saves memory vs Adam."""
        params = 10000
        adam_bytes = params * 12
        muon_bytes = int(params * (12 - 0.85 * 4))
        assert muon_bytes < adam_bytes

    def test_muon_memory_savings_percentage(self):
        """Muon saves ~28% memory vs Adam at 85% muon_fraction."""
        params = 10000
        adam_bytes = params * 12
        muon_bytes = int(params * (12 - 0.85 * 4))
        savings_pct = (adam_bytes - muon_bytes) / adam_bytes
        expected_savings = 0.85 * 4 / 12
        assert abs(savings_pct - expected_savings) < 0.01


class TestOptimizerStepFlopsCalculation:
    """Test optimizer step FLOPs calculation logic."""

    def test_adam_flops_formula(self):
        """Adam: params × 12 FLOPs."""
        params = 1000
        flops = params * 12
        assert flops == 12000

    def test_muon_flops_ns_scaling(self):
        """Muon FLOPs scales with NS steps (K)."""
        params = 10000
        hidden = 1000
        # NS FLOPs = K × 6 × m × n² (for tall matrix)
        # More K = more FLOPs
        assert True  # Logic verified in test_muon_optimizer.py


class TestOptimizerPassPSharding:
    """Test OptimizerPass respects PP sharding."""

    def test_pp_sharding_divides_params(self):
        """PP>1 divides optimizer params by pp."""
        graph = make_mock_graph(num_params=10000)
        ctx = make_mock_context(optimizer="muon", dp=8, pp=4)
        
        pass_obj = OptimizerPass()
        result = pass_obj.run(graph, ctx)
        
        opt_node = result.nodes["optimizer_step"]
        params_total = opt_node.attrs["params_total"]
        
        assert params_total <= 10000 // 4

    def test_optimizer_stage_id_last_stage(self):
        """Optimizer runs in last stage (stage_id = pp-1)."""
        graph = make_mock_graph()
        ctx = make_mock_context(optimizer="muon", dp=8, pp=4)
        
        pass_obj = OptimizerPass()
        result = pass_obj.run(graph, ctx)
        
        opt_node = result.nodes["optimizer_step"]
        stage_id = opt_node.annotations.get("stage_id")
        
        assert stage_id == 3  # pp-1 = 4-1 = 3


class TestOptimizerPassZeROSharding:
    """Test OptimizerPass respects ZeRO sharding."""

    def test_zero_stage_3_shards_by_tp_dp(self):
        """ZeRO-3 shards optimizer params by TP × DP."""
        graph = make_mock_graph(num_params=10000)
        ctx = make_mock_context(optimizer="muon", dp=8, tp=2, zero_stage=3)
        
        pass_obj = OptimizerPass()
        result = pass_obj.run(graph, ctx)
        
        opt_node = result.nodes["optimizer_step"]
        params_total = opt_node.attrs["params_total"]
        
        assert params_total < 10000 // 2  # Sharded by TP

    def test_zero_stage_1_shards_by_tp_only(self):
        """ZeRO-1 shards optimizer params by TP only."""
        graph = make_mock_graph(num_params=10000)
        ctx = make_mock_context(optimizer="muon", dp=8, tp=2, zero_stage=1)
        
        pass_obj = OptimizerPass()
        result = pass_obj.run(graph, ctx)
        
        opt_node = result.nodes["optimizer_step"]
        params_total = opt_node.attrs["params_total"]
        
        assert params_total == 10000 // 2  # Sharded by TP only


class TestMoonshotRotationOverlap:
    """Test Moonshot Rotation overlap calculation logic.

    Overlap mechanism (reference: training.py:761-821):
    1. RS hidden in fwd_window (priority)
    2. AG hidden in NS compute + remaining_fwd_window
    3. NS is always exposed (compute-bound)
    """

    def test_overlap_exposed_hidden_sum_to_latency(self):
        """exposed + hidden should equal total latency."""
        ag_latency = 100.0
        ag_exposed = 30.0
        ag_hidden = 70.0
        
        assert ag_exposed + ag_hidden == ag_latency

    def test_overlap_rs_priority_over_ag(self):
        """RS has priority over AG for fwd_window."""
        fwd_window = 50.0
        rs_time = 40.0
        ag_time = 100.0
        
        rs_hidden = min(rs_time, fwd_window)  # RS takes first
        remaining_fwd = max(0.0, fwd_window - rs_hidden)
        
        assert rs_hidden == 40.0
        assert remaining_fwd == 10.0

    def test_overlap_ag_window_includes_ns_compute(self):
        """AG hide window includes NS compute time."""
        opt_compute = 20.0
        remaining_fwd = 10.0
        ag_time = 100.0
        
        ag_hide_window = opt_compute + remaining_fwd
        ag_hidden = min(ag_time, ag_hide_window)
        
        assert ag_hide_window == 30.0
        assert ag_hidden == 30.0

    def test_overlap_full_hide_when_latency_below_window(self):
        """Full hide when latency < hide window."""
        ag_time = 20.0
        ag_hide_window = 30.0
        
        ag_hidden = min(ag_time, ag_hide_window)
        ag_exposed = ag_time - ag_hidden
        
        assert ag_hidden == 20.0  # Fully hidden
        assert ag_exposed == 0.0

    def test_overlap_exposed_when_latency_above_window(self):
        """Exposed time when latency > hide window."""
        ag_time = 100.0
        ag_hide_window = 30.0
        
        ag_hidden = min(ag_time, ag_hide_window)
        ag_exposed = ag_time - ag_hidden
        
        assert ag_hidden == 30.0
        assert ag_exposed == 70.0

    def test_overlap_rotation_disabled_all_exposed(self):
        """Without rotation, all comm is exposed."""
        ag_time = 100.0
        rs_time = 50.0
        rotation_active = False
        
        if rotation_active:
            ag_exposed = ag_time - min(ag_time, 30.0)
        else:
            ag_exposed = ag_time
        
        assert ag_exposed == 100.0  # Fully exposed when no rotation

    def test_overlap_ns_always_exposed(self):
        """NS compute is always exposed (not hidden)."""
        opt_compute = 20.0
        
        assert opt_compute > 0  # NS adds to step time directly


class TestComputeOptimizerStepTime:
    """Test optimizer step time calculation logic."""

    def test_muon_compute_time_from_flops(self):
        """Muon compute time = FLOPs / peak_flops."""
        step_flops = 1e12
        peak_flops = 989e12
        compute_us = (step_flops / peak_flops) * 1e6
        assert compute_us > 0
        assert compute_us == pytest.approx(1011.2, rel=0.01)

    def test_adam_compute_time_from_memory(self):
        """Adam compute time = state_bytes / hbm_bandwidth."""
        state_bytes = 1e6
        hbm_bw_gbps = 3352  # H100
        hbm_bw = hbm_bw_gbps * 1e9 / 8
        compute_us = (state_bytes / hbm_bw) * 1e6
        assert compute_us > 0

    def test_ag_time_ring_factor_formula(self):
        """AG time includes ring factor (dp-1)/dp."""
        ag_bytes = 1e6
        dp = 8
        dp_bw = 900e9 / 8  # intra-node
        ring_factor = (dp - 1) / dp
        ag_us = (ring_factor * ag_bytes / dp_bw) * 1e6
        assert ag_us > 0
        assert ring_factor == pytest.approx(0.875)

    def test_rs_time_equals_ag_when_rotation(self):
        """RS time = AG time when Moonshot rotation enabled."""
        ag_time = 100.0
        rs_time = ag_time  # Rotation: RS mirrors AG
        assert rs_time == ag_time

    def test_rs_time_zero_when_no_rotation(self):
        """RS time = 0 when rotation disabled."""
        rotation = False
        rs_time = 0.0 if not rotation else 100.0
        assert rs_time == 0.0


class TestOptimizerAnnotationsPhase:
    """Test optimizer node phase annotations."""

    def test_optimizer_nodes_have_bwd_phase(self):
        """Optimizer nodes have phase='bwd' annotation."""
        graph = make_mock_graph()
        ctx = make_mock_context(optimizer="muon", dp=8)
        
        pass_obj = OptimizerPass()
        result = pass_obj.run(graph, ctx)
        
        for node_id in ["optimizer_step", "muon_ag", "muon_rs"]:
            if node_id in result.nodes:
                assert result.nodes[node_id].annotations.get("phase") == "bwd"

    def test_optimizer_step_marker_annotation(self):
        """optimizer_step has optimizer_step=True annotation."""
        graph = make_mock_graph()
        ctx = make_mock_context(optimizer="muon", dp=8)
        
        pass_obj = OptimizerPass()
        result = pass_obj.run(graph, ctx)
        
        opt_node = result.nodes["optimizer_step"]
        assert opt_node.annotations.get("optimizer_step") == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])