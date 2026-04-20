"""Graph-based tensor liveness analysis for activation peak memory."""
from __future__ import annotations

from dataclasses import dataclass

from python.zrt.ir.graph import OpGraph

_MB = 1024.0 ** 2


@dataclass(frozen=True)
class ActivationAnalysis:
    """Result of tensor liveness analysis on an OpGraph."""

    peak_bytes: int
    peak_mb: float
    peak_node_id: str
    per_node_live_mb: dict[str, float]


def analyze_activation(graph: OpGraph) -> ActivationAnalysis:
    """Analyze activation memory by tracking live tensor set through topo order.

    Parameters
    ----------
    graph : OpGraph
        Computation graph with nodes and edges carrying TensorMeta.

    Returns
    -------
    ActivationAnalysis
        Peak activation in bytes/MB, the node after which peak occurs,
        and per-node live memory breakdown.
    """
    if not graph.nodes:
        return ActivationAnalysis(
            peak_bytes=0,
            peak_mb=0.0,
            peak_node_id="",
            per_node_live_mb={},
        )

    topo_nodes = graph.topo_sort()
    topo_idx = {node.id: i for i, node in enumerate(topo_nodes)}

    last_use_idx: dict[str, int] = {}
    for edge in graph.edges:
        if edge.tensor is None or edge.tensor.mem_bytes <= 0:
            continue
        tensor_id = edge.tensor.id
        dst_idx = topo_idx.get(edge.dst, -1)
        if tensor_id not in last_use_idx:
            last_use_idx[tensor_id] = dst_idx
        else:
            last_use_idx[tensor_id] = max(last_use_idx[tensor_id], dst_idx)

    live_set: dict[str, int] = {}
    peak_bytes = 0
    peak_node_id = ""
    per_node_live_mb: dict[str, float] = {}

    for idx, node in enumerate(topo_nodes):
        for output_tensor in node.outputs:
            if output_tensor.mem_bytes > 0:
                live_set[output_tensor.id] = output_tensor.mem_bytes

        live_bytes = sum(live_set.values())
        per_node_live_mb[node.id] = live_bytes / _MB

        if live_bytes > peak_bytes:
            peak_bytes = live_bytes
            peak_node_id = node.id

        to_remove = [tid for tid in live_set if last_use_idx.get(tid, -1) == idx]
        for tid in to_remove:
            del live_set[tid]

    return ActivationAnalysis(
        peak_bytes=int(peak_bytes),
        peak_mb=peak_bytes / _MB,
        peak_node_id=peak_node_id,
        per_node_live_mb=per_node_live_mb,
    )
