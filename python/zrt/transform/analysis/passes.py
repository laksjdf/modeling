"""Analysis passes (Stage 4): FLOPs annotation, Roofline, and Stream assignment."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext


# ── FlopsPass ─────────────────────────────────────────────────────────────────

class FlopsPass(GraphPass):
    """Annotate every node with theoretical FLOPs, read_bytes, write_bytes.

    Reuses the Roofline simulator's per-op formula dispatch (hardware-agnostic).
    Adds:
      node.annotations["flops"]       : int
      node.annotations["read_bytes"]  : int
      node.annotations["write_bytes"] : int
    """

    name = "flops"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        from python.zrt.simulator.backends.roofline import RooflineSimulator
        sim = RooflineSimulator()
        g = graph.clone()
        for node in g.nodes.values():
            flops, read_b, write_b = sim._fmr(node)
            node.annotations["flops"]       = int(flops)
            node.annotations["read_bytes"]  = int(read_b)
            node.annotations["write_bytes"] = int(write_b)
        return g


# ── RooflinePass ──────────────────────────────────────────────────────────────

class RooflinePass(GraphPass):
    """Annotate nodes with Roofline-model timing estimates and bound classification.

    Requires hw_spec in ctx.  Adds:
      node.annotations["compute_us"]           : float
      node.annotations["memory_us"]            : float
      node.annotations["latency_us"]           : float
      node.annotations["arithmetic_intensity"] : float
      node.annotations["bound"]                : "compute" | "memory" | "latency"
    """

    name = "roofline"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        from python.zrt.simulator.backends.roofline import RooflineSimulator
        from python.zrt.ir.types import DType
        sim = RooflineSimulator()
        hw  = ctx.hw_spec
        g   = graph.clone()

        for node in g.nodes.values():
            flops, read_b, write_b = sim._fmr(node)
            total_b = read_b + write_b

            # Use primary dtype for peak throughput lookup
            dtype = node.outputs[0].dtype if node.outputs else DType.BF16
            peak  = hw.peak_flops(dtype)   # ops/s
            bw    = hw.hbm_bandwidth()     # bytes/s

            compute_us = (flops / peak  * 1e6) if peak > 0 else 0.0
            memory_us  = (total_b / bw  * 1e6) if bw   > 0 else 0.0

            ai = flops / total_b if total_b > 0 else math.inf

            if compute_us > 0 or memory_us > 0:
                bound = "compute" if compute_us >= memory_us else "memory"
            else:
                bound = "latency"

            node.annotations["compute_us"]           = compute_us
            node.annotations["memory_us"]            = memory_us
            node.annotations["arithmetic_intensity"] = ai
            node.annotations["bound"]                = bound
            # Respect pre-existing latency_us (e.g. from profiling or test injection)
            if "latency_us" not in node.annotations:
                node.annotations["latency_us"] = max(compute_us, memory_us, 1e-3)

        return g


# ── StreamAssignPass ──────────────────────────────────────────────────────────

class StreamAssignPass(GraphPass):
    """Assign each node to a compute or communication stream.

    Stream layout (IDs):
      0 .. num_compute_streams-1  → compute streams
      num_compute_streams ..      → comm streams

    Assignment policy:
      - category == "communication" → comm streams (round-robin)
      - category == "compute" / "memory" → compute streams (round-robin)

    Overlap detection (based on existing annotations/attrs):
      - Ring-CP: overlap_target starts with "fa_tile:" → overlap_type="ring_cp"
      - MC2: attrs["fused_ag_matmul"] → overlap_type="mc2"
      - CoC: attrs["coc_tile_k"] → overlap_type="coc"
      - Otherwise → overlap_type="none"

    Adds to every node:
      node.annotations["stream_id"]    : int
      node.annotations["stream_type"]  : "compute" | "comm"
      node.annotations["overlap_type"] : "coc" | "mc2" | "ring_cp" | "none"  (comm nodes only)
    """

    name = "stream_assign"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        sc = ctx.stream_config
        g  = graph.clone()

        compute_idx = 0
        comm_idx    = 0

        for node in g.topo_sort():
            if node.category == "communication":
                sid  = sc.comm_stream_id(comm_idx)
                stype = "comm"
                comm_idx += 1

                # Overlap type detection from existing annotations/attrs
                overlap = self._detect_overlap_type(node)
                node.annotations["overlap_type"] = overlap
            else:
                sid  = sc.compute_stream_id(compute_idx)
                stype = "compute"
                compute_idx += 1

            node.annotations["stream_id"]   = sid
            node.annotations["stream_type"] = stype

        return g

    @staticmethod
    def _detect_overlap_type(node) -> str:
        """Determine overlap type from node annotations and attrs."""
        # Ring-CP: P2P nodes with fa_tile overlap_target
        overlap_target = node.annotations.get("overlap_target", "")
        if overlap_target.startswith("fa_tile:"):
            return "ring_cp"
        # MC2: fused all_gather + matmul
        if node.attrs.get("fused_ag_matmul"):
            return "mc2"
        # CoC: communication-over-compute with tile factor
        if node.attrs.get("coc_tile_k"):
            return "coc"
        return "none"
