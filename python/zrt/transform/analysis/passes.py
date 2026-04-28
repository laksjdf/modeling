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

    In training mode (ctx.training is set), also annotates gradient FLOPs
    (flops_dx, flops_dw) and applies the recompute 2x multiplier.

    Always writes:
      node.annotations["flops"]       : int  (raw forward FLOPs)
      node.annotations["read_bytes"]  : int
      node.annotations["write_bytes"] : int

    Training-only additions:
      node.annotations["flops_fwd"]   : int  (flops * recompute_multiplier)
      node.annotations["flops_dx"]    : int
      node.annotations["flops_dw"]    : int
    """

    name = "flops"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        from python.zrt.simulator.backends.roofline import RooflineSimulator
        sim = RooflineSimulator()
        g = graph.clone()

        is_train = ctx.training is not None

        for node in g.nodes.values():
            flops, read_b, write_b = sim._fmr(node)
            node.annotations["flops"]       = int(flops)
            node.annotations["read_bytes"]  = int(read_b)
            node.annotations["write_bytes"] = int(write_b)

            if is_train:
                phase = node.annotations.get("phase", "fwd")
                is_bwd = phase in {"bwd", "backward", "train_backward"}

                # For attention ops, apply compression ratio to FLOPs accounting
                train_flops = flops
                if _is_attention_op(node.op_type):
                    ratio = _attn_compression_ratio(node, g)
                    train_flops = flops * ratio

                if is_bwd:
                    dx_flops, dw_flops = 0.0, 0.0
                else:
                    dx_flops, dw_flops = self._calculate_grad_flops(node, train_flops)

                rec_mult = 2.0 if node.annotations.get("recompute") and not is_bwd else 1.0
                node.annotations["flops_fwd"] = int(train_flops * rec_mult)
                node.annotations["flops_dx"]  = int(dx_flops)
                node.annotations["flops_dw"]  = int(dw_flops)

        return g

    @staticmethod
    def _calculate_grad_flops(node, fwd_flops: float) -> tuple[float, float]:
        op_type = node.op_type
        dx_flops = 0.0
        dw_flops = 0.0

        if op_type.startswith("aten.mm") or op_type.startswith("aten.linear") or op_type.startswith("aten.addmm"):
            dx_flops = fwd_flops
            dw_flops = fwd_flops
        elif _is_attention_op(op_type):
            dx_flops = 2.5 * fwd_flops
        elif "layer_norm" in op_type.lower() or "ln" in op_type.lower():
            dx_flops = fwd_flops
        elif "softmax" in op_type.lower():
            dx_flops = fwd_flops
        elif "swiglu" in op_type.lower() or "gelu" in op_type.lower():
            dx_flops = fwd_flops

        return dx_flops, dw_flops


def _is_attention_op(op_type: str) -> bool:
    return "attention" in op_type.lower() or "attn" in op_type.lower()


def _attn_compression_ratio(node, graph) -> float:
    value = node.annotations.get("attn_compression_ratio")
    if value is None:
        value = node.attrs.get("attn_compression_ratio")
    if value is None:
        value = graph.metadata.get("attn_compression_ratio", 1.0)
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not (0.0 < ratio <= 1.0):
        return 1.0
    return ratio


# ── RooflinePass ──────────────────────────────────────────────────────────────

def _effective_compute_dtype(node: "OpNode") -> "DType":
    """Return the compute dtype for throughput lookup.

    For compute nodes, activation dtype (inputs[0]) drives tensor-core selection.
    Falls back to output dtype for non-compute or no-input nodes.
    Also respects quant_act annotation for hypothetical-quant analysis.
    """
    from python.zrt.ir.types import DType

    if node.category != "compute" or not node.inputs:
        return node.outputs[0].dtype if node.outputs else DType.BF16
    # Annotation path: QuantizationPass wrote quant_act on this node
    quant_act = node.annotations.get("quant_act")
    if quant_act and quant_act not in ("bf16", "fp16", "fp32"):
        # Normalize fp8 alias to fp8_e4m3 (default FP8 format for training)
        normalized = "fp8_e4m3" if quant_act == "fp8" else quant_act
        try:
            return DType(normalized)
        except ValueError:
            pass
    # Captured dtype path: use activation tensor (inputs[0]) dtype
    return node.inputs[0].dtype


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
            flops  = node.annotations.get("flops", 0)
            read_b = node.annotations.get("read_bytes", 0)
            write_b = node.annotations.get("write_bytes", 0)
            if flops == 0 and read_b == 0:   # FlopsPass didn't run — fall back
                flops, read_b, write_b = sim._fmr(node)
            total_b = read_b + write_b

            # Use activation input dtype for compute throughput (INT8/FP8 vs BF16)
            dtype = _effective_compute_dtype(node)
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
