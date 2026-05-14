"""Regression coverage for graph construction in ``records_to_opgraph``.

These tests use *synthetic* dispatch records (no torch needed) so they run
fast and exercise the edge-construction logic directly.

Two regressions are pinned here:

1. **SKIP-op alias propagation** (the ``F.linear(x, w) * rsqrt`` case).
   ``aten._unsafe_view``/``aten.t``/``aten.view`` etc. are silently skipped
   by ``RecordingDispatch``; their outputs must alias to the input tensor
   ID so downstream consumers can still resolve a producer.  Before the
   fix, half of the consumed tensor IDs had no producer and edges were
   silently dropped.

2. **In-place / tensor-id recycling**.  Ops like ``aten.copy_`` reuse an
   input tensor ID as the output.  Consumers after the mutation must
   depend on the mutator, not the *first* writer.  The single-pass walk
   in ``records_to_opgraph`` maintains a running ``tensor_producer`` map
   so the most recent writer is the one selected for the edge.
"""
from __future__ import annotations

from python.zrt.ir.adapter import records_to_opgraph


def _rec(node_id, op, input_ids, output_ids, **extras):
    base = {
        "node_id": node_id,
        "aten_op": op,
        "op_short": op.split(".")[1] if "." in op else op,
        "module_path": "",
        "module_class": "",
        "leaf_attr": "",
        "layer": "",
        "component": "",
        "src_file": "",
        "src_line": 0,
        "src_code": "",
        "src_func": "",
        "extra_args": "",
        "input_shapes": "",
        "input_dtypes": "",
        "output_shapes": "",
        "output_dtypes": "",
        "num_inputs": len(input_ids),
        "num_outputs": len(output_ids),
        "_input_ids": list(input_ids),
        "_output_ids": list(output_ids),
        "call_id": 0,
    }
    base.update(extras)
    return base


def test_chain_mm_then_mul_keeps_edge_when_tensor_ids_match():
    """Sanity: when mm's output id == mul's input id, the edge exists."""
    records = [
        _rec(0, "aten.mm.default",   input_ids=[1, 2], output_ids=[3]),
        _rec(1, "aten.mul.Tensor",   input_ids=[3, 4], output_ids=[5]),
    ]
    g = records_to_opgraph(records, name="t", phase="fwd")
    assert ("op_0", "op_1") in [(e.src, e.dst) for e in g.edges]


def test_inplace_op_routes_subsequent_reads_through_mutator():
    """``copy_`` reuses its output tensor id from input.  A later consumer
    of that tensor id must depend on the in-place mutator, not the
    original producer of the same id."""
    records = [
        # op 0 produces tensor 10 (e.g. allocates a KV cache slot)
        _rec(0, "aten.empty.default",  input_ids=[],   output_ids=[10]),
        # op 1 produces tensor 20 (the new data to copy in)
        _rec(1, "aten.add.Tensor",     input_ids=[1],  output_ids=[20]),
        # op 2 copies 20 → 10 (output id reused = 10, in-place mutator)
        _rec(2, "aten.copy_.default",  input_ids=[10, 20], output_ids=[10]),
        # op 3 reads tensor 10 — should depend on op 2 (the mutator), not op 0
        _rec(3, "aten.mm.default",     input_ids=[10, 30], output_ids=[40]),
    ]
    g = records_to_opgraph(records, name="t", phase="fwd")
    edges = [(e.src, e.dst) for e in g.edges]
    # op_2 must be the producer for op_3's tensor-10 input
    assert ("op_2", "op_3") in edges, edges
    # op_0 must NOT be a producer for op_3
    assert ("op_0", "op_3") not in edges, edges


def test_skip_op_alias_preserves_edge_through_view():
    """Simulate the dispatch behaviour after the SKIP-alias fix: a view
    output gets the *same* tracker id as the view input.  ``records_to_opgraph``
    sees mm.out=X, mul.in=X — edge must exist.  Even though the view is
    not recorded, the chain is preserved because the IDs match."""
    records = [
        # mm produces tensor 5
        _rec(0, "aten.mm.default", input_ids=[1, 2], output_ids=[5]),
        # view skipped — no record.  Its output is aliased to input id 5
        # by the dispatch-layer fix, so mul reads tensor 5.
        _rec(1, "aten.mul.Tensor", input_ids=[5, 6], output_ids=[7]),
    ]
    g = records_to_opgraph(records, name="t", phase="fwd")
    assert ("op_0", "op_1") in [(e.src, e.dst) for e in g.edges]


def test_no_orphan_consumers_when_producers_present():
    """Strong invariant: every input tensor whose id was produced by some
    earlier record must have at least one inbound edge in the graph."""
    records = [
        _rec(0, "aten.mm.default", input_ids=[1, 2], output_ids=[3]),
        _rec(1, "aten.add.Tensor", input_ids=[3, 4], output_ids=[5]),
        _rec(2, "aten.mul.Tensor", input_ids=[5, 3], output_ids=[6]),
    ]
    g = records_to_opgraph(records, name="t", phase="fwd")

    produced = set()
    for r in records:
        produced.update(r["_output_ids"])

    orphans: list[tuple[str, int]] = []
    for r in records:
        consumer_id = f"op_{r['node_id']}"
        in_edges_tids = {e.tensor_id for e in g.edges if e.dst == consumer_id}
        for tid in r["_input_ids"]:
            if tid in produced and tid not in in_edges_tids:
                orphans.append((consumer_id, tid))
    assert not orphans, orphans


def test_multiple_consumers_of_same_tensor_all_wired():
    """One producer, many consumers — each consumer gets its own edge."""
    records = [
        _rec(0, "aten.mm.default",   input_ids=[1, 2], output_ids=[3]),
        _rec(1, "aten.relu.default", input_ids=[3],    output_ids=[4]),
        _rec(2, "aten.tanh.default", input_ids=[3],    output_ids=[5]),
        _rec(3, "aten.add.Tensor",   input_ids=[3, 4], output_ids=[6]),
    ]
    g = records_to_opgraph(records, name="t", phase="fwd")
    incoming_from_0 = [e.dst for e in g.edges if e.src == "op_0"]
    assert sorted(incoming_from_0) == ["op_1", "op_2", "op_3"]


def test_external_weight_inputs_have_no_edge():
    """Tensor IDs only consumed (never produced) are graph-level externals
    (weights, input_ids) — they correctly get no inbound edges."""
    records = [
        # tensor 100 is an external weight (only consumed)
        _rec(0, "aten.mm.default", input_ids=[100, 1], output_ids=[2]),
    ]
    g = records_to_opgraph(records, name="t", phase="fwd")
    assert len(g.edges) == 0  # external inputs don't generate edges
    assert len(g.nodes) == 1
