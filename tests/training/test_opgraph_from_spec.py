"""Test OpGraph.from_model_spec() factory method."""

import pytest

from zrt.ir.graph import OpGraph
from zrt.training.ir.builders import build_graph
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import Strategy


def test_opgraph_from_spec_matches_training_graph():
    """OpGraph.from_model_spec() should produce nodes matching build_graph()."""
    model = ModelSpec(
        hidden=4096,
        ffn=16384,
        num_heads=32,
        num_kv_heads=32,
        head_dim=128,
        vocab=32000,
        seq_len=2048,
        layers=[LayerKind.DENSE] * 4,
    )
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=1)

    # Old way: build training IR
    from zrt.training.ir.training_graph import Graph as TrainingGraph
    training_g: TrainingGraph = build_graph(model, strategy)

    # New way: use OpGraph factory
    opgraph = OpGraph.from_model_spec(model, strategy)

    # Verify node count matches
    assert len(opgraph.nodes) == len(training_g.ops)

    # Verify op types match
    for op_node, training_op in zip(opgraph.nodes.values(), training_g.ops):
        assert op_node.op_type == training_op.kind


def test_opgraph_from_spec_metadata():
    """OpGraph.from_model_spec() should set source metadata."""
    model = ModelSpec(
        hidden=4096,
        ffn=16384,
        num_heads=32,
        num_kv_heads=32,
        head_dim=128,
        vocab=32000,
        seq_len=2048,
        layers=[LayerKind.DENSE] * 2,
    )
    strategy = Strategy(tp=2, pp=1, dp=1, micro_batch=1, global_batch=1)

    opgraph = OpGraph.from_model_spec(model, strategy)

    # Verify metadata
    assert opgraph.metadata["source"] == "model_spec"
    assert opgraph.metadata["hidden"] == model.hidden
    assert opgraph.metadata["layers"] == len(model.layers)
    assert "strategy" in opgraph.metadata
    assert "collectives" in opgraph.metadata


def test_opgraph_from_spec_layer_annotations():
    """Each OpNode should have layer_id and layer_kind annotations."""
    model = ModelSpec(
        hidden=2048,
        ffn=8192,
        num_heads=16,
        num_kv_heads=16,
        head_dim=128,
        vocab=32000,
        seq_len=1024,
        layers=[LayerKind.DENSE, LayerKind.MOE],
        num_experts=4,
        moe_ffn=1024,
        top_k=1,
    )
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=1)

    opgraph = OpGraph.from_model_spec(model, strategy)

    # All nodes should have layer annotations
    for node in opgraph.nodes.values():
        assert "layer_id" in node.annotations
        assert "layer_kind" in node.annotations
        assert isinstance(node.annotations["layer_id"], int)
        assert isinstance(node.annotations["layer_kind"], str)


def test_opgraph_from_spec_edges_within_layers():
    """Edges should only connect ops within the same layer."""
    model = ModelSpec(
        hidden=2048,
        ffn=8192,
        num_heads=16,
        num_kv_heads=16,
        head_dim=128,
        vocab=32000,
        seq_len=1024,
        layers=[LayerKind.DENSE] * 3,
    )
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=1)

    opgraph = OpGraph.from_model_spec(model, strategy)

    # Verify edges only connect nodes with same layer_id
    for edge in opgraph.edges:
        src_layer = opgraph.nodes[edge.src].annotations.get("layer_id")
        dst_layer = opgraph.nodes[edge.dst].annotations.get("layer_id")
        assert src_layer == dst_layer


def test_opgraph_from_spec_phase_parameter():
    """Custom phase parameter should be reflected in the graph."""
    model = ModelSpec(
        hidden=2048,
        ffn=8192,
        num_heads=16,
        num_kv_heads=16,
        head_dim=128,
        vocab=32000,
        seq_len=1024,
        layers=[LayerKind.DENSE],
    )
    strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=1)

    opgraph = OpGraph.from_model_spec(model, strategy, phase="inference")

    assert opgraph.phase == "inference"
    assert "inference" in opgraph.name
