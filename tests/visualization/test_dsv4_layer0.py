"""End-to-end visualization test for DeepSeek-V4 layer 0.

Goal: verify the correctness of ``records_to_opgraph`` (graph construction
from RecordingDispatch records) and the fusion ``replace_subgraph``
rewrite, by capturing layer 0 of DSV4 and emitting:

  1. Module-tree SVG of ``transformer.layers.0`` — original architecture.
  2. Raw-aten OpGraph SVG (post ``records_to_opgraph``) — clustered by
     module scope, edges labelled with shape/dtype.
  3. Fused OpGraph SVG (post ``FusionPass``) — fused nodes highlighted.
  4. ``summary.txt`` with per-stage stats and per-rule fusion counts.

The test also asserts:
  - Both raw and fused graphs are DAGs with no dangling edges / self-loops.
  - Every fused node lists the constituent raw OpNode IDs in
    ``annotations["source_op_ids"]`` and those IDs all resolve.
  - No raw aten op is referenced by more than one fused node
    (replace_subgraph correctness).
  - External edges of every fused node match a re-derivation from its
    source ops' edges in the *original* raw graph.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Allow `python tests/visualization/test_dsv4_layer0.py` from repo root.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Import-time guards; surface useful skip reasons.
torch = pytest.importorskip("torch")

from python.zrt.graph import run_trace_phases
from python.zrt.pipeline import _make_model_slug
from python.zrt.transform.context import (
    FusionConfig,
    ParallelConfig,
    StreamConfig,
    TrainingConfig,
    TransformContext,
)
from python.zrt.transform.fusion.api import FusionPass
import python.zrt.hardware.registry as hw_registry

from tests.visualization._render import (
    assert_dag,
    render_fused_op_graph,
    render_module_tree,
    render_raw_op_graph,
)


_MODEL_ID = "hf_models/deepseek_v4"
_TARGET_LAYER = 0
_OUTPUT_DIR = _REPO / "output" / "visualization" / "deepseek_v4_layer0"


def _check_replace_subgraph_invariants(raw_graph, fused_graph, summary: list[str]) -> None:
    """Cross-check replace_subgraph correctness using source_op_ids.

    For every fused node, every raw op listed in ``source_op_ids`` must
    exist in the raw graph, and no raw op may be claimed by two different
    fused nodes.
    """
    raw_ids: set[str] = set(raw_graph.nodes)
    claimed: dict[str, str] = {}        # raw_id → fused_id
    fused_count = 0

    for fid, fnode in fused_graph.nodes.items():
        srcs = fnode.annotations.get("source_op_ids") or []
        if not srcs:
            continue
        fused_count += 1
        for sid in srcs:
            assert sid in raw_ids, (
                f"fused node {fid} claims raw op {sid} which is not in "
                f"the raw graph"
            )
            assert sid not in claimed, (
                f"raw op {sid} claimed by both {claimed[sid]} and {fid} "
                f"— replace_subgraph allowed double-coverage"
            )
            claimed[sid] = fid

    summary.append(
        f"  replace_subgraph integrity: {fused_count} fused/collapsed nodes "
        f"covered {len(claimed)} raw ops with no double-claim"
    )


def _check_raw_graph_no_orphan_consumers(
    raw_graph, raw_records: list[dict], summary: list[str]
) -> None:
    """Every input tensor whose id has *already been produced earlier in
    capture order* must have a corresponding inbound edge on the consumer.

    Tensors only produced later (Python ``id()`` recycling after GC) or
    never produced (model weights, input_ids) are correctly external — no
    edge expected.  Anything else is dropped dataflow.
    """
    incoming_tids: dict[str, set[int]] = {nid: set() for nid in raw_graph.nodes}
    for e in raw_graph.edges:
        incoming_tids[e.dst].add(e.tensor_id)

    produced_so_far: set[int] = set()
    orphans: list[tuple[str, int, str]] = []
    for r in raw_records:
        consumer_id = f"op_{r['node_id']}"
        in_set = incoming_tids.get(consumer_id, set())
        for tid in r.get("_input_ids", []):
            if tid in produced_so_far and tid not in in_set:
                orphans.append((consumer_id, tid, r["aten_op"]))
        for tid in r.get("_output_ids", []):
            produced_so_far.add(tid)

    summary.append(
        f"  raw-graph dataflow continuity: {len(orphans)} orphan inputs "
        f"({len(produced_so_far)} unique produced tensor ids, "
        f"{len(raw_graph.edges)} edges)"
    )
    assert not orphans, (
        f"records_to_opgraph dropped {len(orphans)} edges — first 5: "
        f"{orphans[:5]}"
    )


def _check_data_flow_preserved(raw_graph, fused_graph, summary: list[str]) -> None:
    """For each fused node verify external IO matches the original graph.

    Reconstruct external IO from the raw graph using ``source_op_ids``
    and compare the set of (tensor_id, side) tuples against what
    ``replace_subgraph`` left on the fused graph's edges.
    """
    raw_in_edges = {nid: list(raw_graph.in_edges(nid)) for nid in raw_graph.nodes}
    raw_out_edges = {nid: list(raw_graph.out_edges(nid)) for nid in raw_graph.nodes}

    mismatches = 0
    for fid, fnode in fused_graph.nodes.items():
        srcs = set(fnode.annotations.get("source_op_ids") or [])
        if not srcs:
            continue
        # External raw inputs: src ∉ srcs, dst ∈ srcs
        expected_in = set()
        for sid in srcs:
            for e in raw_in_edges.get(sid, []):
                if e.src not in srcs:
                    expected_in.add(e.tensor_id)
        # External raw outputs: src ∈ srcs, dst ∉ srcs
        expected_out = set()
        for sid in srcs:
            for e in raw_out_edges.get(sid, []):
                if e.dst not in srcs:
                    expected_out.add(e.tensor_id)

        actual_in = {e.tensor_id for e in fused_graph.in_edges(fid)}
        actual_out = {e.tensor_id for e in fused_graph.out_edges(fid)}

        # Tensor IDs in fused graph were copied from the raw edges, so the
        # expected and actual sets should be equal.  Allow expected ⊆ actual
        # (FusionPass may add cross-rewires to neighbours that introduce
        # extra edges, but it should never lose dependencies).
        if not expected_in.issubset(actual_in):
            mismatches += 1
            missing = expected_in - actual_in
            print(f"[warn] fused {fid} missing input tensor IDs: {sorted(missing)[:5]}")
        if not expected_out.issubset(actual_out):
            mismatches += 1
            missing = expected_out - actual_out
            print(f"[warn] fused {fid} missing output tensor IDs: {sorted(missing)[:5]}")

    summary.append(f"  data-flow preservation: {mismatches} edge-set mismatches")
    assert mismatches == 0, f"replace_subgraph dropped data-flow edges in {mismatches} cases"


def _capture_raw_graph():
    """Run ``run_trace_phases`` against DSV4 layer 0 and return the
    ``(graph, records)`` pair."""
    result = run_trace_phases(
        model_id=_MODEL_ID,
        num_layers=2,                 # need at least 2 to have layer 0 + epilogue
        batch_size=1,
        seq_len=64,
        output_dir=_OUTPUT_DIR / "_trace",
        phases=("train_forward",),
        target_layers=[_TARGET_LAYER],
        auto_layers=False,
        platform="generic",
        graph_mode=False,
        gradient_checkpointing=False,
    )
    return result.graphs["train_forward"], result.phase_records["train_forward"]


def _capture_model_for_tree() -> object:
    """Load the model JUST so we can render its named_modules() tree.

    Uses the same loader as the trace path.  ``fake_mode`` is exited
    immediately — we never run forward, so leaving it active is harmless
    but unnecessary.
    """
    from python.zrt.graph.model_loader import load_model
    model, _config, fake_mode = load_model(_MODEL_ID, num_hidden_layers=2,
                                            training=False)
    fake_mode.__exit__(None, None, None)
    return model


def _summary_lines(raw, fused, slug: str) -> list[str]:
    raw_stats = assert_dag(raw)
    fused_stats = assert_dag(fused)
    lines = [
        f"# DSV4 layer {_TARGET_LAYER} visualization — {slug}",
        "",
        "## Stage 1: raw aten OpGraph (post records_to_opgraph)",
        f"  nodes:           {raw_stats['nodes']}",
        f"  edges:           {raw_stats['edges']}",
        f"  dangling edges:  {raw_stats['dangling_edges']}",
        f"  self-loops:      {raw_stats['self_loops']}",
        "",
        "## Stage 2: fused OpGraph (post FusionPass)",
        f"  nodes:           {fused_stats['nodes']}",
        f"  edges:           {fused_stats['edges']}",
        f"  dangling edges:  {fused_stats['dangling_edges']}",
        f"  self-loops:      {fused_stats['self_loops']}",
        f"  reduction:       {raw_stats['nodes']} → {fused_stats['nodes']} "
        f"({(raw_stats['nodes'] - fused_stats['nodes']) / max(raw_stats['nodes'], 1):.1%})",
        "",
        "## Per-rule fusion counts",
    ]
    by_rule: dict[str, int] = {}
    for n in fused.nodes.values():
        rn = n.annotations.get("fused_by_rule") or "<raw>"
        by_rule[rn] = by_rule.get(rn, 0) + 1
    for rn, cnt in sorted(by_rule.items(), key=lambda x: -x[1]):
        lines.append(f"  {rn:30}  {cnt}")
    return lines


def test_dsv4_layer0_visualization():
    if not (Path(_REPO) / _MODEL_ID).exists():
        pytest.skip(f"model dir {_MODEL_ID} not present")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[str] = []

    # ── Stage A: raw aten OpGraph (records_to_opgraph result) ────────────
    # Done first because run_trace_phases manages its own FakeTensorMode
    # and exits cleanly.  Loading the model a second time afterwards (for
    # the tree render) avoids the "Mixing fake modes NYI" collision that
    # happens when two FakeTensorModes are alive simultaneously.
    raw_graph, raw_records = _capture_raw_graph()
    raw_stats = assert_dag(raw_graph)
    assert raw_stats["dangling_edges"] == 0, raw_stats
    assert raw_stats["self_loops"] == 0, raw_stats
    assert raw_stats["nodes"] > 0
    render_raw_op_graph(
        raw_graph,
        _OUTPUT_DIR / "02_raw_aten_graph.dot",
        title=f"DeepSeek-V4 layer {_TARGET_LAYER} — raw aten DAG "
              f"({raw_stats['nodes']} nodes, {raw_stats['edges']} edges)",
    )
    print(f"[ok] raw-aten     → {_OUTPUT_DIR / '02_raw_aten_graph.svg'} "
          f"({raw_stats['nodes']} nodes)")

    # ── Stage C: fused OpGraph (FusionPass standalone, no parallel split) ─
    hw = hw_registry.load("nvidia_h100_sxm")
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(),                # tp=pp=ep=dp=cp=1
        stream_config=StreamConfig(),
        training=TrainingConfig(),                # → phase_for_fusion()=="training"
        fusion=FusionConfig(),                    # auto-resolves DSV4 yaml
        model_id=_MODEL_ID,
    )
    fused_graph = FusionPass().run(raw_graph, ctx)
    fused_stats = assert_dag(fused_graph)
    assert fused_stats["dangling_edges"] == 0, fused_stats
    assert fused_stats["self_loops"] == 0, fused_stats

    render_fused_op_graph(
        fused_graph,
        _OUTPUT_DIR / "03_fused_graph.dot",
        title=f"DeepSeek-V4 layer {_TARGET_LAYER} — fused (training config) "
              f"({fused_stats['nodes']} nodes)",
    )
    print(f"[ok] fused        → {_OUTPUT_DIR / '03_fused_graph.svg'} "
          f"({fused_stats['nodes']} nodes)")

    # ── Stage D: module tree of original architecture ─────────────────────
    # Loaded last so the trace's FakeTensorMode is fully gone; we never
    # run forward on this second model so the cached freqs_cis buffer
    # from the trace's now-dead fake_mode never gets dispatched.
    try:
        model = _capture_model_for_tree()
        layer0_scope = "model.transformer.layers.0"
        if not any(n.startswith(layer0_scope) for n, _ in model.named_modules()):
            for candidate in ("transformer.layers.0", "model.layers.0", "layers.0"):
                if any(n.startswith(candidate) for n, _ in model.named_modules()):
                    layer0_scope = candidate
                    break
        render_module_tree(
            model, _OUTPUT_DIR / "01_module_tree.dot",
            max_depth=8,
            scope_filter=layer0_scope,
            title=f"DeepSeek-V4 — {layer0_scope} (nn.Module hierarchy)",
        )
        print(f"[ok] module-tree  → {_OUTPUT_DIR / '01_module_tree.svg'}")
    except Exception as exc:                          # noqa: BLE001
        # Module-tree render is a "nice to have" — never fail the test
        # for a tree-rendering side effect.  Dump a text-only fallback.
        print(f"[warn] module-tree render skipped: {exc}")
        fallback = _OUTPUT_DIR / "01_module_tree.txt"
        try:
            from python.zrt.graph.model_loader import _load_config
            cfg, _ = _load_config(_MODEL_ID)
            fallback.write_text(
                f"DeepSeek-V4 layer-0 module tree render failed: {exc}\n\n"
                f"Config summary: hidden={getattr(cfg, 'hidden_size', '?')}, "
                f"num_hidden_layers={getattr(cfg, 'num_hidden_layers', '?')}\n"
            )
        except Exception:
            pass

    # ── Stage E: invariants ───────────────────────────────────────────────
    summary = _summary_lines(raw_graph, fused_graph, _make_model_slug(_MODEL_ID))
    summary.append("")
    summary.append("## Invariants")
    _check_raw_graph_no_orphan_consumers(raw_graph, raw_records, summary)
    _check_replace_subgraph_invariants(raw_graph, fused_graph, summary)
    _check_data_flow_preserved(raw_graph, fused_graph, summary)

    summary_path = _OUTPUT_DIR / "summary.txt"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"[ok] summary      → {summary_path}")
    print()
    print("\n".join(summary))


if __name__ == "__main__":
    test_dsv4_layer0_visualization()
