"""Runtime aten-sequence tracer.

Wraps ``zrt.graph.run_trace_phases`` and groups consecutive aten ops by
``module_class`` so the joiner can compare static AST class names with
the dynamic op sequences observed during a fake-tensor forward pass.

We intentionally keep the API minimal: input is a model id, output is
``{module_class: [seq, ...]}`` where each ``seq`` is a deduplicated tuple
of full aten op names with shape/dtype-only ops filtered out (they are
already in ``DEFAULT_SKIP_OPS`` so downstream matching ignores them
anyway).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# Mirrors ``DEFAULT_SKIP_OPS`` from rule.py.  We duplicate it here so the
# tracer module can be imported without importing the transform package.
_SKIP_OPS: frozenset[str] = frozenset({
    "aten.view.default",
    "aten.reshape.default",
    "aten.permute.default",
    "aten.transpose.int",
    "aten.expand.default",
    "aten.squeeze.dim",
    "aten.unsqueeze.default",
    "aten.flatten.using_ints",
    "aten.unflatten.int",
    "aten.contiguous.memory_format",
    "aten._to_copy.default",
    "aten.to.dtype",
    "aten.to.dtype_layout",
    "aten.clone.default",
    "aten.detach.default",
    "aten.alias.default",
})


def run_runtime_trace(
    model_id: str,
    *,
    num_layers: int = 4,
    phase: str = "prefill",
) -> Dict[str, List[Tuple[str, ...]]]:
    """Run a fake-tensor forward pass and bucket aten ops by module_class.

    Parameters
    ----------
    model_id:
        HuggingFace id (or local path) understood by ``run_trace_phases``.
    num_layers:
        Number of transformer layers to construct.  4 is enough for
        DeepSeek-V3/V4 to surface both dense and MoE blocks.
    phase:
        Either ``"prefill"`` (default) or ``"train_forward"``.  Use
        ``train_forward`` when the model relies on training-only kernel
        stubs (DeepSeek-V4).

    Returns
    -------
    Dict mapping each ``module_class`` to a deduplicated list of observed
    aten op sequences.  Sequences are sorted (longest first) so the
    joiner can pick the dominant pattern deterministically.
    """
    # Imported lazily so unit tests that mock-trace can avoid pulling in
    # the full graph stack.
    from python.zrt.graph import run_trace_phases  # type: ignore[import-not-found]

    output_dir, phase_records = run_trace_phases(
        model_id=model_id,
        num_layers=num_layers,
        phases=(phase,),
        output_dir=Path("/tmp/zrt_fusion_discover"),
    )
    records = phase_records.get(phase, [])
    return _bucket_by_module_class(records)


# ─── Helpers (also used by tests) ─────────────────────────────────────────────

def _bucket_by_module_class(
    records: List[Dict[str, str]],
) -> Dict[str, List[Tuple[str, ...]]]:
    """Group consecutive op records by ``module_class`` and dedupe.

    A "run" is a maximal contiguous slice of records sharing the same
    ``module_class``.  Skip-listed ops are filtered out **inside** each
    run so the resulting sequences are already noise-free.
    """
    runs: list[tuple[str, tuple[str, ...]]] = []
    cur_mc: str | None = None
    cur_seq: list[str] = []

    for rec in records:
        mc = rec.get("module_class") or ""
        ot = rec.get("op_type") or ""
        if mc != cur_mc:
            if cur_mc and cur_seq:
                runs.append((cur_mc, tuple(cur_seq)))
            cur_mc = mc
            cur_seq = []
        if ot and ot not in _SKIP_OPS:
            cur_seq.append(ot)
    if cur_mc and cur_seq:
        runs.append((cur_mc, tuple(cur_seq)))

    by_class: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for mc, seq in runs:
        if seq:
            by_class[mc].add(seq)

    return {
        mc: sorted(seqs, key=lambda s: (-len(s), s))
        for mc, seqs in sorted(by_class.items())
    }
