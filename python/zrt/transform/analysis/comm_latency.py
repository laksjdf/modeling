"""CommLatencyPass: estimate communication latency using interconnect bandwidth.

Replaces the Roofline estimate for communication nodes with proper collective
communication formulas that account for intra-node vs inter-node bandwidth.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext


def _estimate_comm_latency(
    collective: str,
    group_size: int,
    data_bytes: int,
    bandwidth_bps: float,
    link_latency_us: float,
) -> float:
    """Estimate latency for a collective communication operation.

    Parameters
    ----------
    collective : str
        Type of collective: "all_reduce", "all_gather", "reduce_scatter",
        "all_to_all", "send_recv", "broadcast"
    group_size : int
        Number of participating devices
    data_bytes : int
        Data size in bytes
    bandwidth_bps : float
        Link bandwidth in bytes/second
    link_latency_us : float
        Link latency in microseconds

    Returns
    -------
    float
        Estimated latency in microseconds
    """
    if group_size <= 1 or data_bytes == 0:
        return 0.0

    n = group_size
    latency_term = link_latency_us

    if collective == "all_reduce":
        # Ring AllReduce: 2(n-1)/n * D / BW + 2(n-1) * lat
        volume = 2 * (n - 1) / n * data_bytes
        latency_us = volume / bandwidth_bps * 1e6 + 2 * (n - 1) * latency_term
    elif collective == "all_gather":
        # AllGather: (n-1)/n * D / BW + (n-1) * lat
        volume = (n - 1) / n * data_bytes
        latency_us = volume / bandwidth_bps * 1e6 + (n - 1) * latency_term
    elif collective == "reduce_scatter":
        # ReduceScatter: (n-1)/n * D / BW + (n-1) * lat
        volume = (n - 1) / n * data_bytes
        latency_us = volume / bandwidth_bps * 1e6 + (n - 1) * latency_term
    elif collective == "all_to_all":
        # AllToAll: (n-1)/n * D / BW + lat (each device exchanges with n-1 others)
        volume = (n - 1) / n * data_bytes
        latency_us = volume / bandwidth_bps * 1e6 + latency_term
    elif collective in ("send_recv", "p2p"):
        # Point-to-point: D / BW + lat
        latency_us = data_bytes / bandwidth_bps * 1e6 + latency_term
    elif collective == "broadcast":
        # Broadcast (tree): D / BW + log(n) * lat (simplified as (n-1)*lat for conservatism)
        latency_us = data_bytes / bandwidth_bps * 1e6 + (n - 1) * latency_term
    else:
        # Unknown collective: conservative estimate
        latency_us = data_bytes / bandwidth_bps * 1e6 + (n - 1) * latency_term

    return max(0.0, latency_us)


class CommLatencyPass(GraphPass):
    """Estimate communication latency using interconnect bandwidth.

    This pass runs after RooflinePass and overwrites the latency_us annotation
    for communication nodes using proper collective communication formulas that
    account for:
    - Intra-node bandwidth (NVLink, HCCS) vs Inter-node bandwidth (RoCE, IB)
    - Collective algorithm (ring, tree, etc.)
    - Message size-dependent alpha-beta model

    Detection of intra vs inter-node:
    - Intra-node if group_size <= hw.interconnect.intra_node.num_devices
    - Inter-node otherwise
    """

    name = "comm_latency"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        g = graph.clone()
        hw_spec = ctx.hw_spec
        if hw_spec is None:
            return g

        for node in g.nodes.values():
            if node.category != "communication":
                continue

            collective = node.attrs.get("collective", "all_reduce")
            group_size = node.attrs.get("group_size", 1)

            # Compute total data bytes: prefer explicit msg_bytes attr, else tensor sizes
            data_bytes = node.attrs.get("msg_bytes", 0)
            if data_bytes == 0:
                data_bytes = sum(t.mem_bytes for t in node.inputs)
            if data_bytes == 0:
                data_bytes = sum(t.mem_bytes for t in node.outputs)
            if data_bytes == 0:
                data_bytes = 1  # Conservative: at least 1 byte

            # Determine if cross-node
            intra_node_devices = hw_spec.interconnect.intra_node.num_devices
            cross_node = group_size > intra_node_devices

            # Select appropriate link
            if cross_node:
                link = hw_spec.interconnect.inter_node
            else:
                link = hw_spec.interconnect.intra_node

            # bandwidth_gbps is aggregate GB/s; divide by 8 to convert to per-GPU
            # B/s (matching ring-algorithm effective per-GPU bandwidth)
            bandwidth_bps = link.bandwidth_gbps * 1e9 / 8.0

            # Estimate latency
            latency_us = _estimate_comm_latency(
                collective, group_size, data_bytes, bandwidth_bps, link.latency_us
            )

            # Annotate results
            node.annotations["latency_us"] = latency_us
            node.annotations["cross_node"] = cross_node
            node.annotations["comm_algorithm"] = "ring" if collective == "all_reduce" else collective

        return g
