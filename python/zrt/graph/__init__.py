"""Graph-level package: dispatch-level tracing infrastructure.

Public API::

    from python.zrt.graph import run_trace, load_model, build_config_summary
    from python.zrt.graph import build_op_graph, build_fused_op_graph, build_fused_nx_from_opgraph, export_graphs

Imports from submodules that require torch (main, model_loader, patches, classifier,
graph_builder, graph_exporter, compat) are lazy — they only load when accessed.
"""

from python.zrt.ir.graph import OpGraph
from python.zrt.ir.adapter import records_to_opgraph, fused_records_to_opgraph

_MODULE_MAP = {
    # ── Graph builder / tracer (require torch) ──────────────────────────────
    "main": "python.zrt.graph.main",
    "run_trace": "python.zrt.graph.main",
    "run_trace_phases": "python.zrt.graph.main",
    "TraceResult": "python.zrt.graph.main",
    "TracePhaseResult": "python.zrt.graph.main",
    "build_config_summary": "python.zrt.graph.main",
    "infer_layer_types": "python.zrt.graph.main",
    "auto_target_layers": "python.zrt.graph.main",
    "load_model": "python.zrt.graph.model_loader",
    "_is_moe_module": "python.zrt.graph.patches",
    "_patch_moe_for_meta": "python.zrt.graph.patches",
    "_classify_component": "python.zrt.graph.classifier",
    "_extract_layer_idx": "python.zrt.graph.classifier",
    "build_op_graph": "python.zrt.graph.graph_builder",
    "build_fused_op_graph": "python.zrt.graph.graph_builder",
    "build_fused_nx_from_opgraph": "python.zrt.graph.graph_builder",
    "export_graphs": "python.zrt.report.onnx_exporter",
    "find_local_fallback": "python.zrt.graph.compat",
    "_LOCAL_REGISTRY": "python.zrt.graph.compat",
}

_CACHE: dict = {}


def __getattr__(name: str):
    if name in _MODULE_MAP:
        mod_path = _MODULE_MAP[name]
        if mod_path not in _CACHE:
            import importlib
            _CACHE[mod_path] = importlib.import_module(mod_path)
        return getattr(_CACHE[mod_path], name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "load_model",
    "build_op_graph",
    "build_fused_op_graph",
    "build_fused_nx_from_opgraph",
    "export_graphs",
    "_classify_component",
    "_extract_layer_idx",
    "_is_moe_module",
    "_patch_moe_for_meta",
    "find_local_fallback",
    "_LOCAL_REGISTRY",
]
