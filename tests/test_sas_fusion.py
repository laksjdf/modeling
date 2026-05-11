"""Tests for SparseAttnSharedKV (npu_sas) fusion rules and annotation."""
import pytest
from python.zrt.ir.node import OpNode
from python.zrt.ir.edge import Edge
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.types import TensorMeta, DType
from python.zrt.transform.fusion.pass_ import FusionPass
from python.zrt.transform.context import TransformContext, ParallelConfig, StreamConfig
import python.zrt.hardware.registry as hw_registry


# ── shared helpers ────────────────────────────────────────────────────────────

def _t(tid, shape=(1, 128), dtype=DType.BF16):
    return TensorMeta.from_shape_dtype(tid, shape, dtype)


def _node(nid, op_type, scope="", layer="0", module_class=""):
    return OpNode(
        id=nid, op_type=op_type,
        inputs=[_t(f"{nid}_in")], outputs=[_t(f"{nid}_out")],
        scope=scope, layer=layer, module_class=module_class, category="compute",
    )


def _edge(src, dst):
    return Edge(src=src, src_idx=0, dst=dst, dst_idx=0, tensor=_t("e"))


def _graph(nodes, edges, name="test", metadata=None):
    return OpGraph(name=name, phase="prefill",
                   nodes={n.id: n for n in nodes},
                   edges=edges, metadata=metadata)


def _ctx(hw_name="ascend_910b"):
    hw = hw_registry.load(hw_name)
    return TransformContext(hw_spec=hw, parallel=ParallelConfig(tp=1),
                            stream_config=StreamConfig())


# ── npu_sas inference pattern ─────────────────────────────────────────────────

def test_inference_gather_pattern_ascend():
    """gather → bmm → softmax → bmm in Attention scope on ascend → 'npu_sas'."""
    scope = "transformer.layers.0.attn"
    mc = "DeepseekV4Attention"
    ops = [
        ("gather", "aten.gather.default"),
        ("bmm1",   "aten.bmm.default"),
        ("sfmx",   "aten.softmax.int"),
        ("bmm2",   "aten.bmm.default"),
    ]
    nodes = [_node(nid, op, scope=scope, layer="0", module_class=mc)
             for nid, op in ops]
    edges = [_edge(ops[i][0], ops[i+1][0]) for i in range(len(ops)-1)]
    g = _graph(nodes, edges)

    out = FusionPass().run(g, _ctx("ascend_910b"))
    fused = next(iter(out.nodes.values()))
    assert fused.op_type == "npu_sas", f"Expected 'npu_sas', got '{fused.op_type}'"


# ── npu_sas training pattern (Compressor gated-pooling) ──────────────────────

def test_training_compressor_class_ascend():
    """Any Compressor-class ops on ascend → 'npu_sas' (class-only match).

    The V4 Compressor performs gated KV pooling mapped to npu_sparse_attn_sharedkv.
    Its ops are split across multiple topo-sort groups due to interleaving with
    the norm sub-module, so class-name matching (empty op_seq) is used.
    """
    scope = "transformer.layers.0.attn.compressor"
    mc = "Compressor"
    ops = [
        ("add", "aten.add.Tensor"),
        ("sfmx", "aten._softmax.default"),
        ("mul",  "aten.mul.Tensor"),
    ]
    nodes = [_node(nid, op, scope=scope, layer="0", module_class=mc)
             for nid, op in ops]
    edges = [_edge(ops[i][0], ops[i+1][0]) for i in range(len(ops)-1)]
    g = _graph(nodes, edges)

    out = FusionPass().run(g, _ctx("ascend_910b"))
    fused = next(iter(out.nodes.values()))
    assert fused.op_type == "npu_sas", f"Expected 'npu_sas', got '{fused.op_type}'"


# ── npu_sas not triggered on CUDA ─────────────────────────────────────────────

def test_sas_not_triggered_on_cuda():
    """gather→bmm→softmax→bmm on H100 → 'v4_sparse_attn', not 'npu_sas'."""
    scope = "transformer.layers.0.attn"
    mc = "DeepseekV4Attention"
    ops = [
        ("gather", "aten.gather.default"),
        ("bmm1",   "aten.bmm.default"),
        ("sfmx",   "aten.softmax.int"),
        ("bmm2",   "aten.bmm.default"),
    ]
    nodes = [_node(nid, op, scope=scope, layer="0", module_class=mc)
             for nid, op in ops]
    edges = [_edge(ops[i][0], ops[i+1][0]) for i in range(len(ops)-1)]
    g = _graph(nodes, edges)

    out = FusionPass().run(g, _ctx("nvidia_h100_sxm"))
    fused = next(iter(out.nodes.values()))
    assert fused.op_type == "v4_sparse_attn", (
        f"Expected 'v4_sparse_attn' on CUDA, got '{fused.op_type}'"
    )


# ── FusionPass annotation ─────────────────────────────────────────────────────

def _compressor_group(layer_idx: int) -> tuple[list, list]:
    """Return (nodes, edges) for a 3-op Compressor group at the given layer."""
    scope = f"transformer.layers.{layer_idx}.attn.compressor"
    ops = [
        (f"add_{layer_idx}",  "aten.add.Tensor"),
        (f"sfmx_{layer_idx}", "aten._softmax.default"),
        (f"mul_{layer_idx}",  "aten.mul.Tensor"),
    ]
    nodes = [_node(nid, op, scope=scope, layer=str(layer_idx),
                   module_class="Compressor") for nid, op in ops]
    edges = [_edge(ops[i][0], ops[i + 1][0]) for i in range(len(ops) - 1)]
    return nodes, edges


def test_fusion_pass_annotates_npu_sas():
    """FusionPass sets attn_type/compress_ratio on npu_sas nodes when compress_ratios in metadata."""
    compress_ratios = [128, 4, 0]
    expected_types = ["HCA", "CSA", "SWA"]
    ctx = _ctx("ascend_910b")

    for layer_idx, (exp_type, exp_cr) in enumerate(zip(expected_types, compress_ratios)):
        nodes, edges = _compressor_group(layer_idx)
        g = _graph(nodes, edges, metadata={"compress_ratios": compress_ratios})
        out = FusionPass().run(g, ctx)
        npu_sas = next((n for n in out.topo_sort() if n.op_type == "npu_sas"), None)
        assert npu_sas is not None, f"layer {layer_idx}: expected npu_sas node"
        assert npu_sas.annotations.get("attn_type") == exp_type, (
            f"layer {layer_idx}: expected {exp_type!r}, "
            f"got {npu_sas.annotations.get('attn_type')!r}"
        )
        assert npu_sas.annotations.get("compress_ratio") == exp_cr


def test_fusion_pass_no_annotation_without_compress_ratios():
    """FusionPass skips attn_type annotation when compress_ratios not in metadata."""
    nodes, edges = _compressor_group(0)
    g = _graph(nodes, edges)  # no compress_ratios in metadata
    ctx = _ctx("ascend_910b")
    out = FusionPass().run(g, ctx)
    npu_sas = next((n for n in out.topo_sort() if n.op_type == "npu_sas"), None)
    assert npu_sas is not None
    assert "attn_type" not in npu_sas.annotations
