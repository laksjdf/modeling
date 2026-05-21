"""Analysis passes (Stage 4): FLOPs annotation, Roofline, and Stream assignment."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from python.zrt.transform.base import GraphPass
from python.zrt.transform.training.recompute import is_external_recompute_node

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

    # Scope substrings that identify individual expert computation.
    _EXPERT_KEYWORDS = ("experts.", "expert_", ".experts[")

    @staticmethod
    def _is_expert_scope(scope: str) -> bool:
        s = scope.lower()
        if "shared_expert" in s:
            return False
        return any(k in s for k in FlopsPass._EXPERT_KEYWORDS)

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        from python.zrt.simulator.backends.roofline import RooflineSimulator
        sim = RooflineSimulator()
        g = graph.clone()

        is_train = ctx.training is not None
        moe_scale = graph.metadata.get("moe_active_experts", 1)
        num_experts_total = graph.metadata.get("moe_total_experts", 0)

        for node in g.nodes.values():
            # Priority: use sem_flops (from fusion semantics) if available
            # sem_flops already includes CP shape split adjustment
            sem_flops = node.annotations.get("sem_flops")
            if sem_flops is not None:
                # Use sem_flops directly (already scaled by CP if cp_split exists)
                flops = float(sem_flops)
                sem_io = node.annotations.get("sem_io", {})
                read_b = sum(v.get("bytes", 0) for v in sem_io.values() if v.get("role") != "output")
                write_b = sum(v.get("bytes", 0) for v in sem_io.values() if v.get("role") == "output")
                # If sem_io has explicit roles
                if "activation" in sem_io:
                    read_b = sem_io["activation"].get("bytes", 0)
                if "output" in sem_io:
                    write_b = sem_io["output"].get("bytes", 0)
            else:
                # Fallback to roofline formula
                flops, read_b, write_b = sim._fmr(node)

            # MoE expert scaling: unfused captures usually contain one routed
            # expert's ops, while ExpertGroupedMMPass materializes the local
            # expert group in the GroupedMatMul shape already.
            if moe_scale > 1 and node.scope and self._is_expert_scope(node.scope):
                if node.annotations.get("fused_by") == "expert_grouped_mm":
                    scale = 1.0
                elif (ep_local := node.annotations.get("ep_experts_local", 0)) > 0 and num_experts_total > 0:
                    ep_frac = ep_local / num_experts_total
                    scale = moe_scale * ep_frac
                elif getattr(ctx.parallel, "ep", 1) > 1:
                    # With EP enabled, unannotated residual expert-scope helper
                    # ops are already local to this rank after EP/grouped-MM
                    # lowering. Active-expert scaling would charge them again.
                    scale = 1.0
                else:
                    scale = moe_scale
                flops = int(flops * scale)
                read_b = int(read_b * scale)
                write_b = int(write_b * scale)

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

                # flops_fwd includes external checkpoint replay overhead (2x).
                # FA/SDPA attention cores already include internal recompute in
                # their backward formula and must not be doubled here.
                rec_mult = 2.0 if is_external_recompute_node(node) and not is_bwd else 1.0
                node.annotations["flops_fwd"]  = int(train_flops * rec_mult)
                node.annotations["flops_dx"]   = int(dx_flops)
                node.annotations["flops_dw"]   = int(dw_flops)

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

            base_flops = flops
            base_read_b = read_b
            base_write_b = write_b
            recompute_flops = 0
            recompute_read_b = 0
            recompute_write_b = 0

            # Add external checkpoint replay overhead from fwd predecessors.
            # FA/SDPA attention cores already pay internal recompute inside
            # their backward kernel, so they are excluded from this replay.
            phase = node.annotations.get("phase", "fwd")
            is_bwd = phase in ("bwd", "backward", "train_backward")
            if is_bwd and ctx.training:
                for e in g.in_edges(node.id):
                    src_id = e.src
                    if src_id in g.nodes and is_external_recompute_node(g.nodes[src_id]):
                        src = g.nodes[src_id]
                        recompute_flops += src.annotations.get("flops", 0)
                        recompute_read_b += src.annotations.get("read_bytes", 0)
                        recompute_write_b += src.annotations.get("write_bytes", 0)

            # Use activation input dtype for compute throughput (INT8/FP8 vs BF16)
            dtype = _effective_compute_dtype(node)
            peak  = hw.peak_flops(dtype)   # ops/s
            bw    = hw.hbm_bandwidth()     # bytes/s

            base_total_b = base_read_b + base_write_b
            base_compute_us = (base_flops / peak * 1e6) if peak > 0 else 0.0
            base_memory_us = (base_total_b / bw * 1e6) if bw > 0 else 0.0
            if base_compute_us > 0 or base_memory_us > 0:
                base_latency_us = max(base_compute_us, base_memory_us, 1e-3)
            else:
                base_latency_us = 0.0
            recompute_total_b = recompute_read_b + recompute_write_b
            recompute_compute_us = (recompute_flops / peak * 1e6) if peak > 0 else 0.0
            recompute_memory_us = (recompute_total_b / bw * 1e6) if bw > 0 else 0.0
            if recompute_compute_us > 0 or recompute_memory_us > 0:
                recompute_latency_us = max(recompute_compute_us, recompute_memory_us)
            else:
                recompute_latency_us = 0.0

            saved_activation_b = 0
            if ctx.training and not is_bwd and not node.annotations.get("recompute"):
                saved_activation_b = sum(t.mem_bytes for t in node.outputs)
            activation_memory_us = (saved_activation_b / bw * 1e6) if bw > 0 else 0.0

            compute_us = base_compute_us
            memory_us = base_memory_us + activation_memory_us
            if is_bwd:
                final_latency_us = base_latency_us + recompute_latency_us
            elif base_compute_us > 0 or base_memory_us > 0 or activation_memory_us > 0:
                final_latency_us = base_latency_us + activation_memory_us
            else:
                final_latency_us = 0.0

            total_b = base_total_b + saved_activation_b
            ai = base_flops / total_b if total_b > 0 else math.inf

            if compute_us > 0 or memory_us > 0:
                bound = "compute" if compute_us >= memory_us else "memory"
            else:
                bound = "latency"

            node.annotations["compute_us"]           = compute_us
            node.annotations["memory_us"]            = memory_us
            node.annotations["base_compute_us"]      = base_compute_us
            node.annotations["base_memory_us"]       = base_memory_us
            node.annotations["base_latency_us"]      = base_latency_us
            node.annotations["saved_activation_bytes"] = saved_activation_b
            node.annotations["activation_memory_us"] = activation_memory_us
            node.annotations["checkpoint_activation_bytes"] = 0
            node.annotations["checkpoint_memory_us"] = 0.0
            node.annotations["recompute_flops"]      = recompute_flops
            node.annotations["recompute_read_bytes"] = recompute_read_b
            node.annotations["recompute_write_bytes"] = recompute_write_b
            node.annotations["recompute_compute_us"] = recompute_compute_us
            node.annotations["recompute_memory_us"]  = recompute_memory_us
            node.annotations["recompute_latency_us"] = recompute_latency_us
            node.annotations["arithmetic_intensity"] = ai
            node.annotations["bound"]                = bound
            # Respect pre-existing latency_us (e.g. from profiling or test injection)
            if "latency_us" not in node.annotations:
                node.annotations["latency_us"] = final_latency_us

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
