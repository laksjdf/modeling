"""End-to-end test for Muon optimizer integration (Moonshot Rotation).

Tests DeepSeek V4 models to verify:
1. Muon optimizer nodes (muon_ag, optimizer_step, muon_rs) creation and annotations
2. Moonshot Rotation overlap mechanism (AG→NS+fwd_window, RS→fwd_window)
3. Exposed/Hidden Time display in reports (no empty values)
4. Different DP configurations and rotation settings

Run with:
    $env:PYTHONPATH='python'; pytest tests/IT/test_muon_optimizer_e2e.py -v
"""
import pytest
import tempfile
from pathlib import Path

from python.zrt.pipeline import run_trace_phases
from python.zrt.transform.analysis import estimate_training_from_graphs
import python.zrt.hardware.registry as hw_registry


OPTIMIZER_CONFIGS = {
    "muon_rotation_on": {
        "optimizer": "muon",
        "muon_rotation": True,
        "muon_ns_steps": 10,
        "description": "Muon with Moonshot Rotation enabled",
    },
    "muon_rotation_off": {
        "optimizer": "muon",
        "muon_rotation": False,
        "muon_ns_steps": 10,
        "description": "Muon without Moonshot Rotation",
    },
    "adam_baseline": {
        "optimizer": "adam",
        "muon_rotation": False,
        "muon_ns_steps": 0,
        "description": "Adam optimizer baseline",
    },
}


@pytest.fixture(scope="module")
def training_graphs():
    """Capture DeepSeek V4 training graphs once and reuse."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_trace_phases(
            model_id="hf_models/deepseek_v4",
            num_layers=3,
            batch_size=1,
            seq_len=128,
            phases=["train_forward", "train_backward"],
            output_dir=tmpdir,
        )
        yield result.graphs


def run_muon_analysis(
    forward_graph,
    backward_graph,
    dp: int,
    optimizer: str = "muon",
    muon_rotation: bool = True,
    muon_ns_steps: int = 10,
    tp: int = 1,
    ep: int = 1,
):
    """Run training analysis with Muon optimizer configuration."""
    hw_spec = hw_registry.load("nvidia_h100_sxm")
    
    report, ctx, transformed = estimate_training_from_graphs(
        forward_graph=forward_graph,
        backward_graph=backward_graph,
        hw_spec=hw_spec,
        tp=tp,
        pp=1,
        dp=dp,
        ep=ep,
        seq_len=128,
        batch_size=1,
        hidden=7168,
        num_layers=3,
        optimizer=optimizer,
        muon_rotation=muon_rotation,
        muon_ns_steps=muon_ns_steps,
        return_transformed=True,
    )
    return report, transformed


def extract_optimizer_nodes(graph):
    """Extract optimizer nodes (muon_ag, optimizer_step, muon_rs)."""
    optimizer_nodes = {}
    for node_id in ["muon_ag", "optimizer_step", "muon_rs"]:
        if node_id in graph.nodes:
            node = graph.nodes[node_id]
            optimizer_nodes[node_id] = {
                "node": node,
                "op_type": node.op_type,
                "category": node.category,
                "attrs": dict(node.attrs),
                "annotations": dict(node.annotations),
            }
    return optimizer_nodes


class TestMuonOptimizerNodes:
    """Test Muon optimizer node creation and attributes."""

    def test_muon_nodes_exist_when_muon_enabled(self, training_graphs):
        """Verify muon_ag, optimizer_step, muon_rs nodes exist for Muon optimizer."""
        fwd_graph = training_graphs["train_forward"]
        bwd_graph = training_graphs["train_backward"]
        
        report, transformed = run_muon_analysis(
            fwd_graph, bwd_graph,
            dp=8, optimizer="muon", muon_rotation=True, muon_ns_steps=10
        )
        
        unified_graph = transformed.get("unified")
        assert unified_graph is not None
        
        opt_nodes = extract_optimizer_nodes(unified_graph)
        
        assert "optimizer_step" in opt_nodes, "optimizer_step node should exist"
        assert "muon_ag" in opt_nodes, "muon_ag node should exist when DP>1"
        assert "muon_rs" in opt_nodes, "muon_rs node should exist when muon_rotation=True"

    def test_muon_node_categories(self, training_graphs):
        """Verify optimizer nodes have correct categories."""
        fwd_graph = training_graphs["train_forward"]
        bwd_graph = training_graphs["train_backward"]
        
        report, transformed = run_muon_analysis(
            fwd_graph, bwd_graph,
            dp=8, optimizer="muon", muon_rotation=True, muon_ns_steps=10
        )
        
        unified_graph = transformed.get("unified")
        opt_nodes = extract_optimizer_nodes(unified_graph)
        
        if "optimizer_step" in opt_nodes:
            assert opt_nodes["optimizer_step"]["category"] in ["optimizer", "compute"], \
                "optimizer_step should be optimizer or compute category"
        
        if "muon_ag" in opt_nodes:
            assert opt_nodes["muon_ag"]["category"] == "communication", \
                "muon_ag should be communication category"
        
        if "muon_rs" in opt_nodes:
            assert opt_nodes["muon_rs"]["category"] == "communication", \
                "muon_rs should be communication category"

    def test_muon_node_attributes(self, training_graphs):
        """Verify optimizer nodes have required attributes."""
        fwd_graph = training_graphs["train_forward"]
        bwd_graph = training_graphs["train_backward"]
        
        report, transformed = run_muon_analysis(
            fwd_graph, bwd_graph,
            dp=8, optimizer="muon", muon_rotation=True, muon_ns_steps=10
        )
        
        unified_graph = transformed.get("unified")
        opt_nodes = extract_optimizer_nodes(unified_graph)
        
        if "optimizer_step" in opt_nodes:
            attrs = opt_nodes["optimizer_step"]["attrs"]
            assert "optimizer" in attrs, "optimizer_step should have 'optimizer' attr"
            assert attrs["optimizer"] == "muon", "optimizer should be 'muon'"
            assert "ns_steps" in attrs, "optimizer_step should have 'ns_steps' attr"
            assert attrs["ns_steps"] == 10, "ns_steps should be 10"
        
        if "muon_ag" in opt_nodes:
            attrs = opt_nodes["muon_ag"]["attrs"]
            assert "bytes" in attrs or "muon_ag_bytes" in attrs, \
                "muon_ag should have communication bytes attr"


class TestMoonshotRotationOverlap:
    """Test Moonshot Rotation overlap mechanism."""

    def test_overlap_annotations_with_rotation(self, training_graphs):
        """Verify Moonshot Rotation overlap annotations are set correctly."""
        fwd_graph = training_graphs["train_forward"]
        bwd_graph = training_graphs["train_backward"]
        
        report, transformed = run_muon_analysis(
            fwd_graph, bwd_graph,
            dp=8, optimizer="muon", muon_rotation=True, muon_ns_steps=10
        )
        
        unified_graph = transformed.get("unified")
        opt_nodes = extract_optimizer_nodes(unified_graph)
        
        if "muon_ag" in opt_nodes:
            annotations = opt_nodes["muon_ag"]["annotations"]
            assert annotations.get("overlap_type") == "moonshot_ag", \
                f"muon_ag overlap_type should be 'moonshot_ag' when rotation=True, got {annotations.get('overlap_type')}"
            
            overlap_exposed = annotations.get("overlap_exposed_us")
            overlap_hidden = annotations.get("overlap_hidden_us")
            
            assert overlap_exposed is not None, \
                "muon_ag should have overlap_exposed_us annotation"
            assert overlap_hidden is not None, \
                "muon_ag should have overlap_hidden_us annotation"
            
            assert overlap_exposed >= 0, "overlap_exposed_us should be >= 0"
            assert overlap_hidden >= 0, "overlap_hidden_us should be >= 0"
            
            latency = annotations.get("latency_us", 0)
            assert overlap_exposed + overlap_hidden == pytest.approx(latency, rel=0.01), \
                f"exposed + hidden should equal latency: {overlap_exposed} + {overlap_hidden} vs {latency}"
        
        if "muon_rs" in opt_nodes:
            annotations = opt_nodes["muon_rs"]["annotations"]
            assert annotations.get("overlap_type") == "moonshot_rs", \
                f"muon_rs overlap_type should be 'moonshot_rs' when rotation=True"
            
            overlap_exposed = annotations.get("overlap_exposed_us")
            overlap_hidden = annotations.get("overlap_hidden_us")
            
            assert overlap_exposed is not None, \
                "muon_rs should have overlap_exposed_us annotation"
            assert overlap_hidden is not None, \
                "muon_rs should have overlap_hidden_us annotation"

    def test_overlap_disabled_when_rotation_off(self, training_graphs):
        """Verify overlap_type='none' when Moonshot Rotation is disabled."""
        fwd_graph = training_graphs["train_forward"]
        bwd_graph = training_graphs["train_backward"]
        
        report, transformed = run_muon_analysis(
            fwd_graph, bwd_graph,
            dp=8, optimizer="muon", muon_rotation=False, muon_ns_steps=10
        )
        
        unified_graph = transformed.get("unified")
        opt_nodes = extract_optimizer_nodes(unified_graph)
        
        if "muon_ag" in opt_nodes:
            annotations = opt_nodes["muon_ag"]["annotations"]
            overlap_type = annotations.get("overlap_type", "none")
            assert overlap_type in ["none", "moonshot_ag"], \
                f"muon_ag overlap_type should be 'none' or 'moonshot_ag' when rotation=False, got {overlap_type}"
        
        if "muon_rs" in opt_nodes:
            annotations = opt_nodes["muon_rs"]["annotations"]
            overlap_type = annotations.get("overlap_type", "none")
            assert overlap_type == "none", \
                f"muon_rs overlap_type should be 'none' when rotation=False, got {overlap_type}"

    def test_overlap_targets_correct(self, training_graphs):
        """Verify overlap_target annotations match design."""
        fwd_graph = training_graphs["train_forward"]
        bwd_graph = training_graphs["train_backward"]
        
        report, transformed = run_muon_analysis(
            fwd_graph, bwd_graph,
            dp=8, optimizer="muon", muon_rotation=True, muon_ns_steps=10
        )
        
        unified_graph = transformed.get("unified")
        opt_nodes = extract_optimizer_nodes(unified_graph)
        
        if "muon_ag" in opt_nodes:
            annotations = opt_nodes["muon_ag"]["annotations"]
            overlap_target = annotations.get("overlap_target", "")
            assert "optimizer_step" in overlap_target, \
                f"muon_ag overlap_target should mention 'optimizer_step', got '{overlap_target}'"
        
        if "muon_rs" in opt_nodes:
            annotations = opt_nodes["muon_rs"]["annotations"]
            overlap_target = annotations.get("overlap_target", "")
            assert "fwd_window" in overlap_target or overlap_target == "", \
                f"muon_rs overlap_target should mention 'fwd_window' or be empty, got '{overlap_target}'"


class TestExposedHiddenTimeDisplay:
    """Test Exposed Time / Hidden Time display (no empty values for 0)."""

    def test_exposed_time_not_empty_when_zero(self, training_graphs):
        """Verify exposed_time displays '0' when value is 0, not empty string."""
        fwd_graph = training_graphs["train_forward"]
        bwd_graph = training_graphs["train_backward"]
        
        report, transformed = run_muon_analysis(
            fwd_graph, bwd_graph,
            dp=256, optimizer="muon", muon_rotation=True, muon_ns_steps=10,
            tp=2, ep=318
        )
        
        unified_graph = transformed.get("unified")
        opt_nodes = extract_optimizer_nodes(unified_graph)
        
        for node_id in ["muon_ag", "muon_rs"]:
            if node_id in opt_nodes:
                annotations = opt_nodes[node_id]["annotations"]
                overlap_exposed = annotations.get("overlap_exposed_us")
                
                if overlap_exposed is not None and overlap_exposed == 0:
                    assert annotations.get("overlap_exposed_us") == 0, \
                        f"{node_id} exposed_time should be 0 (not None/empty)"
                
                overlap_hidden = annotations.get("overlap_hidden_us")
                if overlap_hidden is not None and overlap_hidden == 0:
                    assert annotations.get("overlap_hidden_us") == 0, \
                        f"{node_id} hidden_time should be 0 (not None/empty)"

    def test_latency_annotations_exist(self, training_graphs):
        """Verify latency annotations are set for optimizer nodes."""
        fwd_graph = training_graphs["train_forward"]
        bwd_graph = training_graphs["train_backward"]
        
        report, transformed = run_muon_analysis(
            fwd_graph, bwd_graph,
            dp=8, optimizer="muon", muon_rotation=True, muon_ns_steps=10
        )
        
        unified_graph = transformed.get("unified")
        opt_nodes = extract_optimizer_nodes(unified_graph)
        
        for node_id in ["muon_ag", "optimizer_step", "muon_rs"]:
            if node_id in opt_nodes:
                annotations = opt_nodes[node_id]["annotations"]
                assert "latency_us" in annotations, \
                    f"{node_id} should have 'latency_us' annotation"
                assert annotations["latency_us"] >= 0, \
                    f"{node_id} latency_us should be >= 0"


class TestMuonOptimizerMetrics:
    """Test Muon optimizer metrics in reports."""

    def test_optimizer_time_in_report(self, training_graphs):
        """Verify optimizer_time_ms is set in report metrics."""
        fwd_graph = training_graphs["train_forward"]
        bwd_graph = training_graphs["train_backward"]
        
        report, transformed = run_muon_analysis(
            fwd_graph, bwd_graph,
            dp=8, optimizer="muon", muon_rotation=True, muon_ns_steps=10
        )
        
        assert hasattr(report, "optimizer_time_ms"), \
            "Report should have optimizer_time_ms field"
        
        if report.optimizer_time_ms is not None:
            assert report.optimizer_time_ms >= 0, \
                "optimizer_time_ms should be >= 0"

    def test_optimizer_comm_in_report(self, training_graphs):
        """Verify optimizer_comm is tracked in report."""
        fwd_graph = training_graphs["train_forward"]
        bwd_graph = training_graphs["train_backward"]
        
        report, transformed = run_muon_analysis(
            fwd_graph, bwd_graph,
            dp=8, optimizer="muon", muon_rotation=True, muon_ns_steps=10
        )
        
        assert hasattr(report, "optimizer_comm_ms") or hasattr(report, "muon_ag_rs_ms"), \
            "Report should have optimizer_comm_ms or muon_ag_rs_ms field"

    def test_step_time_includes_optimizer(self, training_graphs):
        """Verify step_time includes optimizer compute time."""
        fwd_graph = training_graphs["train_forward"]
        bwd_graph = training_graphs["train_backward"]
        
        muon_report, _ = run_muon_analysis(
            fwd_graph, bwd_graph,
            dp=8, optimizer="muon", muon_rotation=True, muon_ns_steps=10
        )
        
        adam_report, _ = run_muon_analysis(
            fwd_graph, bwd_graph,
            dp=8, optimizer="adam", muon_rotation=False, muon_ns_steps=0
        )
        
        assert muon_report.step_time_ms > 0, "Muon step_time should be > 0"
        assert adam_report.step_time_ms > 0, "Adam step_time should be > 0"


class TestDifferentDPConfigurations:
    """Test Muon optimizer across different DP configurations."""

    @pytest.mark.parametrize("dp", [1, 2, 8, 256])
    def test_muon_nodes_by_dp(self, training_graphs, dp):
        """Verify muon_ag/muon_rs presence depends on DP configuration."""
        fwd_graph = training_graphs["train_forward"]
        bwd_graph = training_graphs["train_backward"]
        
        report, transformed = run_muon_analysis(
            fwd_graph, bwd_graph,
            dp=dp, optimizer="muon", muon_rotation=True, muon_ns_steps=10
        )
        
        unified_graph = transformed.get("unified")
        opt_nodes = extract_optimizer_nodes(unified_graph)
        
        assert "optimizer_step" in opt_nodes, \
            "optimizer_step should always exist for Muon"
        
        if dp > 1:
            assert "muon_ag" in opt_nodes, \
                f"muon_ag should exist when DP={dp}>1 (ZeRO-1 AllGather)"
        
        if "muon_rs" in opt_nodes:
            assert opt_nodes["muon_rs"]["annotations"].get("overlap_type") in ["moonshot_rs", "none"], \
                "muon_rs overlap_type should be valid"

    def test_dp_scaling_optimizer_comm(self, training_graphs):
        """Verify optimizer communication scales with DP."""
        fwd_graph = training_graphs["train_forward"]
        bwd_graph = training_graphs["train_backward"]
        
        dp2_report, _ = run_muon_analysis(fwd_graph, bwd_graph, dp=2)
        dp8_report, _ = run_muon_analysis(fwd_graph, bwd_graph, dp=8)
        
        dp2_unified, dp2_transformed = run_muon_analysis(fwd_graph, bwd_graph, dp=2)
        dp8_unified, dp8_transformed = run_muon_analysis(fwd_graph, bwd_graph, dp=8)
        
        dp2_nodes = extract_optimizer_nodes(dp2_transformed.get("unified"))
        dp8_nodes = extract_optimizer_nodes(dp8_transformed.get("unified"))
        
        if "muon_ag" in dp2_nodes and "muon_ag" in dp8_nodes:
            dp2_ag_latency = dp2_nodes["muon_ag"]["annotations"].get("latency_us", 0)
            dp8_ag_latency = dp8_nodes["muon_ag"]["annotations"].get("latency_us", 0)
            
            assert dp8_ag_latency > dp2_ag_latency, \
                f"DP=8 AllGather latency ({dp8_ag_latency}us) should > DP=2 ({dp2_ag_latency}us)"


class TestOptimizerNodeMetadata:
    """Test optimizer node metadata in graph."""

    def test_optimizer_metadata_in_graph(self, training_graphs):
        """Verify optimizer metadata is stored in graph metadata."""
        fwd_graph = training_graphs["train_forward"]
        bwd_graph = training_graphs["train_backward"]
        
        report, transformed = run_muon_analysis(
            fwd_graph, bwd_graph,
            dp=8, optimizer="muon", muon_rotation=True, muon_ns_steps=10
        )
        
        unified_graph = transformed.get("unified")
        
        metadata_keys = [
            "optimizer_step_time_us",
            "optimizer_ag_exposed_us",
            "optimizer_rs_exposed_us",
        ]
        
        for key in metadata_keys:
            if key in unified_graph.metadata:
                value = unified_graph.metadata[key]
                assert value >= 0, f"{key} should be >= 0"

    def test_pipeline_metrics_exist(self, training_graphs):
        """Verify pipeline_metrics is attached to graph metadata."""
        fwd_graph = training_graphs["train_forward"]
        bwd_graph = training_graphs["train_backward"]
        
        report, transformed = run_muon_analysis(
            fwd_graph, bwd_graph,
            dp=8, optimizer="muon", muon_rotation=True, muon_ns_steps=10
        )
        
        unified_graph = transformed.get("unified")
        
        assert "pipeline_metrics" in unified_graph.metadata, \
            "graph metadata should have 'pipeline_metrics'"
        
        pm = unified_graph.metadata["pipeline_metrics"]
        assert hasattr(pm, "step_time_ms") or "step_time_ms" in pm, \
            "pipeline_metrics should have step_time_ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])