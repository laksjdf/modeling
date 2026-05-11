"""Graph-level package: dispatch-level tracing infrastructure.

Public API::

    from python.zrt.graph.model_loader import load_model
    from python.zrt.graph.graph_builder import build_op_graph, build_fused_op_graph

Pipeline orchestration lives in ``python.zrt.pipeline``.
Fusion rules live in ``python.zrt.transform.fusion.rules``.
"""

from python.zrt.graph.model_loader import load_model
from python.zrt.graph.graph_builder import build_op_graph, build_fused_op_graph

__all__ = [
    "load_model",
    "build_op_graph",
    "build_fused_op_graph",
]
