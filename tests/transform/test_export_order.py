"""Tests for execution-order display in the Fused Operators sheet.

Regression coverage for two bugs that caused rows 4-37 to appear out of
order in ``Fused Operators (fwd)``:

1. ``_layer_sort_key`` lumped every node with empty ``layer`` into the
   pre-layer bucket, so post-layer head/norm and inserted P2P comms
   floated to the top.
2. ``pipeline_parallel`` constructed ``comm.send_recv`` nodes without
   propagating ``layer`` / ``scope``, leaving them in the same bucket.
"""
from __future__ import annotations

from python.zrt.ir.graph import Edge, OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.ir.types import TensorMeta, DType
from python.zrt.transform.exporter import layer_stable_sort


def _t(name: str, shape=(1,)) -> TensorMeta:
    return TensorMeta.from_shape_dtype(name, shape, DType.BF16)


def _mk_node(nid: str, layer: str = "", op_type: str = "aten.add.Tensor",
             scope: str = "") -> OpNode:
    return OpNode(
        id=nid,
        op_type=op_type,
        inputs=[_t(f"{nid}_in")],
        outputs=[_t(f"{nid}_out")],
        scope=scope,
        layer=layer,
    )


def _chain_graph(nodes: list[OpNode]) -> OpGraph:
    g = OpGraph(name="t", phase="prefill")
    for n in nodes:
        g.nodes[n.id] = n
    for a, b in zip(nodes, nodes[1:]):
        g.edges.append(Edge(src=a.id, src_idx=0, dst=b.id, dst_idx=0,
                            tensor=a.outputs[0]))
    g._rebuild_adjacency()
    return g


def test_pre_layer_op_stays_at_top():
    embed = _mk_node("embed", layer="", scope="transformer.embed")
    l0 = _mk_node("l0", layer="0")
    l1 = _mk_node("l1", layer="1")
    g = _chain_graph([embed, l0, l1])
    out = layer_stable_sort(g.topo_sort(), graph=g)
    assert [n.id for n in out] == ["embed", "l0", "l1"]


def test_post_layer_head_sorts_after_last_layer():
    """Head/norm ops have empty layer but predecessors are in layer N — they
    should land *after* layer N, not before layer 0."""
    embed = _mk_node("embed", layer="", scope="transformer.embed")
    l0 = _mk_node("l0", layer="0")
    l1 = _mk_node("l1", layer="1")
    norm = _mk_node("norm", layer="", scope="transformer.norm")
    head = _mk_node("head", layer="", scope="transformer.head")
    g = _chain_graph([embed, l0, l1, norm, head])
    out = layer_stable_sort(g.topo_sort(), graph=g)
    assert [n.id for n in out] == ["embed", "l0", "l1", "norm", "head"]


def test_inserted_p2p_inherits_layer_via_topology():
    """A send_recv inserted between layer-0 and layer-1 should sort with
    layer 0 (the producer side), not float to the top."""
    embed = _mk_node("embed", layer="", scope="transformer.embed")
    l0 = _mk_node("l0", layer="0")
    p2p = _mk_node("p2p_0_1", layer="", op_type="comm.send_recv",
                   scope="pipeline.p2p.fwd.stage0_to_1")
    l1 = _mk_node("l1", layer="1")
    g = _chain_graph([embed, l0, p2p, l1])
    out = layer_stable_sort(g.topo_sort(), graph=g)
    assert [n.id for n in out] == ["embed", "l0", "p2p_0_1", "l1"]


def test_p2p_with_explicit_layer_sorts_with_neighbors():
    """When the PP pass now stamps layer="0" on the inserted comm, it
    sorts immediately with the layer-0 block."""
    l0a = _mk_node("l0a", layer="0")
    p2p = _mk_node("p2p", layer="0", op_type="comm.send_recv")
    l0b = _mk_node("l0b", layer="0")
    l1 = _mk_node("l1", layer="1")
    g = _chain_graph([l0a, p2p, l0b, l1])
    out = layer_stable_sort(g.topo_sort(), graph=g)
    assert [n.id for n in out] == ["l0a", "p2p", "l0b", "l1"]


def test_head_with_only_cross_phase_edges_uses_scope_fallback():
    """A fwd head op connected only to bwd ops must not be pulled into the
    bwd subgraph's layer range — its scope (transformer.head) should pin it
    to the post-layer bucket."""
    embed = _mk_node("embed", layer="", scope="transformer.embed")
    embed.annotations = {"phase": "fwd"}
    l0 = _mk_node("l0", layer="0", scope="transformer.layers.0.attn")
    l0.annotations = {"phase": "fwd"}
    l1 = _mk_node("l1", layer="1", scope="transformer.layers.1.attn")
    l1.annotations = {"phase": "fwd"}
    head = _mk_node("head", layer="", scope="transformer.head",
                    op_type="aten.mm.default")
    head.annotations = {"phase": "fwd"}
    bwd0 = _mk_node("bwd_head", layer="", scope="transformer.head")
    bwd0.annotations = {"phase": "bwd"}
    bwd1 = _mk_node("bwd_l1", layer="1", scope="transformer.layers.1.attn")
    bwd1.annotations = {"phase": "bwd"}
    bwd0_to_bwd1 = bwd1  # alias

    g = OpGraph(name="t", phase="train")
    for n in [embed, l0, l1, head, bwd0, bwd1]:
        g.nodes[n.id] = n
    # fwd chain
    g.edges.append(Edge(src="embed", src_idx=0, dst="l0", dst_idx=0,
                        tensor=embed.outputs[0]))
    g.edges.append(Edge(src="l0", src_idx=0, dst="l1", dst_idx=0,
                        tensor=l0.outputs[0]))
    # cross-phase: head fwd -> head bwd, head bwd -> layer-1 bwd
    g.edges.append(Edge(src="head", src_idx=0, dst="bwd_head", dst_idx=0,
                        tensor=head.outputs[0]))
    g.edges.append(Edge(src="bwd_head", src_idx=0, dst="bwd_l1", dst_idx=0,
                        tensor=bwd0.outputs[0]))
    g._rebuild_adjacency()

    out = layer_stable_sort(g.topo_sort(), graph=g)
    fwd_ids = [n.id for n in out if n.annotations.get("phase") == "fwd"]
    # head must come AFTER l1, not before l0 / between l0-l1
    assert fwd_ids == ["embed", "l0", "l1", "head"], fwd_ids


def test_legacy_no_graph_falls_back_to_layer_field_only():
    """When called without ``graph=``, the legacy numeric-only path runs."""
    embed = _mk_node("embed", layer="")
    l0 = _mk_node("l0", layer="0")
    l1 = _mk_node("l1", layer="1")
    g = _chain_graph([embed, l0, l1])
    # No graph= → empty-layer ops end up in the tail bucket (legacy
    # behaviour).  We only assert that all numbered layers stay ordered.
    out = layer_stable_sort(g.topo_sort())
    numeric_ids = [n.id for n in out if n.layer]
    assert numeric_ids == ["l0", "l1"]
