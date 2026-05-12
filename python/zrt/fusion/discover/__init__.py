"""Fusion-rule discovery (static AST + runtime trace + templates).

High-level API:

    from zrt.fusion.discover import discover_fusion_rules
    rules, notes = discover_fusion_rules(
        "hf_models/deepseek_v4/inference/model.py",
        hf_id="deepseek-ai/DeepSeek-V4",
        train=True,
    )

Lower-level building blocks (each is independently unit-testable):

  * ``scan_model_file`` — pure AST pass over a source file.
  * ``run_runtime_trace`` — wraps ``run_trace_phases``; returns
    ``{module_class: [aten_op_sequence, ...]}``.
  * ``join_rules`` — merges AST + runtime + templates into a list of
    YAML-ready dicts plus human review notes.
  * ``TEMPLATES`` / ``get_template`` — built-in rule templates per class.
"""
from __future__ import annotations

from typing import List, Tuple

from .ast_scanner import AstClassInfo, AstScanResult, scan_model_file
from .joiner import join_rules
from .runtime_tracer import run_runtime_trace
from .templates import TEMPLATES, get_template


def discover_fusion_rules(
    model_file: str,
    hf_id: str | None = None,
    num_layers: int = 4,
    train: bool = False,
    skip_runtime: bool = False,
) -> Tuple[List[dict], List[str]]:
    """Static-AST-plus-optional-runtime discovery of fusion rules.

    Parameters
    ----------
    model_file:
        Path to the source ``model.py`` to scan with AST.
    hf_id:
        HuggingFace id (or local path) for the runtime trace.  Required
        unless ``skip_runtime=True``.
    num_layers:
        Number of transformer layers to trace.
    train:
        ``True`` → run ``train_forward`` (needed for DeepSeek-V4 etc.);
        ``False`` → run ``prefill``.
    skip_runtime:
        Bypass the runtime trace entirely (CI / static-only mode).

    Returns
    -------
    ``(rules, review_notes)``.  ``rules`` is yaml-dumpable; each entry is
    consumable by ``zrt.transform.fusion.yaml_loader.load_yaml_rules``.
    """
    ast_result = scan_model_file(model_file)
    runtime: dict = {}
    if not skip_runtime:
        if not hf_id:
            raise ValueError("hf_id is required when skip_runtime is False")
        phase = "train_forward" if train else "prefill"
        runtime = run_runtime_trace(hf_id, num_layers=num_layers, phase=phase)
    return join_rules(ast_result, runtime)


__all__ = [
    "discover_fusion_rules",
    "scan_model_file",
    "run_runtime_trace",
    "join_rules",
    "TEMPLATES",
    "get_template",
    "AstScanResult",
    "AstClassInfo",
]
