"""HTML exporter for spec-based training estimation (--estimate-config).

Produces a self-contained HTML page with:
1. Summary: model / hardware / topology / strategy / performance / memory / bound.
2. Logical cluster topology with PP / DP / TP / EP / CP rank labels.
3. Model-level timing overview in Transformer-Explainer-like blocks.
4. Expandable model -> layer/block -> op timing/formula/bound analysis.
5. Calibration table with public/official reference points.

Security notes:
- JSON payload is injected with JSON.parse(<safe JS string literal>), not as a raw
  object literal.
- HTML title is escaped.
- User/model/op strings are escaped again in browser-side rendering.
"""

from __future__ import annotations

import html as html_lib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zrt.training.ir.training_graph import Graph, Op
    from zrt.training.models.flops import OpCost
    from zrt.training.spec.model import ModelSpec
    from zrt.training.spec.report import TrainingReport
    from zrt.training.spec.strategy import Strategy
    from zrt.training.spec.system import SystemSpec


def _fmt_shape(shape: tuple[int, ...]) -> str:
    """Format shape tuple as string."""
    if not shape:
        return "()"
    return "(" + ", ".join(str(d) for d in shape) + ")"


def _fmt_tensor(t) -> str:
    """Format a Tensor as 'shape dtype'."""
    dtype = getattr(getattr(t, "dtype", None), "value", getattr(t, "dtype", ""))
    return f"{_fmt_shape(t.shape_local)} {dtype}"


def _tensor_list_info(tensors: list, label: str = "") -> str:
    """Format input/output tensor list."""
    if not tensors:
        return "-"

    parts = []
    for t in tensors:
        flag = "W" if getattr(t, "is_param", False) else "A"
        name = getattr(t, "name", "")
        parts.append(f"[{flag}] {name}: {_fmt_tensor(t)}")
    return " | ".join(parts)


def _bpe_from_op(op) -> int:
    """Bytes per element from op tensors."""
    if getattr(op, "inputs", None):
        return op.inputs[0].dtype.bytes
    if getattr(op, "outputs", None):
        return op.outputs[0].dtype.bytes
    return 2


def _fmt_e(v: float) -> str:
    """Format a number in compact scientific-like units."""
    try:
        v = float(v)
    except Exception:
        return "-"

    if v <= 0:
        return "-"
    if v >= 1e15:
        return f"{v / 1e15:.2f}P"
    if v >= 1e12:
        return f"{v / 1e12:.2f}T"
    if v >= 1e9:
        return f"{v / 1e9:.2f}G"
    if v >= 1e6:
        return f"{v / 1e6:.2f}M"
    if v >= 1e3:
        return f"{v / 1e3:.2f}K"
    return f"{v:.0f}"


def _fwd_flops(cost) -> float:
    return cost.fwd_cube_flops + cost.fwd_vector_flops


def _dx_flops(cost) -> float:
    return cost.dx_cube_flops + cost.dx_vector_flops


def _dw_flops(cost) -> float:
    return cost.dw_cube_flops + cost.dw_vector_flops


def _op_formula(op, cost):
    """Return formula strings for one op.

    Returns:
        tuple:
            fwd_formula,
            bwd_formula,
            fwd_bytes_formula,
            bwd_bytes_formula
    """
    m = getattr(op, "meta", {}) or {}

    ff = _fwd_flops(cost)
    df = _dx_flops(cost)
    wf = _dw_flops(cost)

    if op.kind == "matmul" or op.kind == "lm_head":
        mm = m.get("m", 0)
        nn = m.get("n_local", m.get("n", 0))
        kk = m.get("k_local", m.get("k", 0))
        mult = m.get("fwd_multiplier", 1.0)
        bpe = _bpe_from_op(op)

        fwd_val = ff
        bytes_val = cost.fwd_bytes

        if mult != 1.0:
            fwd_str = (
                f"2×m×n×k×{mult} = "
                f"2×{mm}×{nn}×{kk}×{mult} = {_fmt_e(fwd_val)}"
            )
        else:
            fwd_str = (
                f"2×m×n×k = "
                f"2×{mm}×{nn}×{kk} = {_fmt_e(fwd_val)}"
            )

        bwd_str = f"dx+dw = 2×fwd = {_fmt_e(df + wf)}"
        bytes_str = (
            f"(m×k+k×n+m×n)×bpe = "
            f"({mm}×{kk}+{kk}×{nn}+{mm}×{nn})×{bpe} = {_fmt_e(bytes_val)}"
        )
        return fwd_str, bwd_str, bytes_str, bytes_str

    if op.kind in ("attn_core", "sparse_attn", "hca_attn", "swa_attn"):
        b = m.get("b", 1)
        s = m.get("s", 0)
        h = m.get("heads", 0)
        d = m.get("head_dim", 0)
        topk = m.get("sparse_topk", 0)
        cr = m.get("compress_ratio", 0)
        swa = m.get("swa_window", 0)

        if topk > 0:
            eff = topk + swa
            fwd_str = (
                f"2×b×s×(topk+swa)×h×d = "
                f"2×{b}×{s}×{eff}×{h}×{d} = {_fmt_e(ff)}"
            )
        elif cr > 0:
            c_len = max(1, s // cr)
            eff = c_len + swa
            fwd_str = (
                f"2×b×s×(s/r+swa)×h×d = "
                f"2×{b}×{s}×{eff}×{h}×{d} = {_fmt_e(ff)}"
            )
        elif swa > 0:
            fwd_str = (
                f"2×b×s×swa×h×d = "
                f"2×{b}×{s}×{swa}×{h}×{d} = {_fmt_e(ff)}"
            )
        else:
            fwd_str = (
                f"2×b×s²×h×d = "
                f"2×{b}×{s}²×{h}×{d} = {_fmt_e(ff)}"
            )

        bwd_str = f"2.5×fwd, including attention backward/recompute approximation = {_fmt_e(df)}"

        kv_len = (
            (topk + swa)
            if topk > 0
            else (max(1, s // cr) + swa if cr > 0 else (swa if swa > 0 else s))
        )

        bytes_str = (
            f"(2×b×h×s×d + 2×b×h×kv_len×d)×bpe = "
            f"(2×{b}×{h}×{s}×{d} + 2×{b}×{h}×{kv_len}×{d})×bpe "
            f"= {_fmt_e(cost.fwd_bytes)}"
        )
        bwd_bytes_str = (
            f"(3×b×h×s×d + 4×b×h×kv_len×d)×bpe "
            f"= {_fmt_e(cost.dx_bytes)}"
        )
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind in ("rmsnorm", "ln"):
        n = sum(t.num_elements() for t in op.outputs) if op.outputs else 0
        fpe = 5 if op.kind == "rmsnorm" else 7
        fwd_str = f"{fpe}×N = {fpe}×{n} = {_fmt_e(ff)}"
        bwd_str = f"2.5×fwd = {_fmt_e(df)}"
        bytes_str = f"fwd_bytes = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"dx_bytes = {_fmt_e(cost.dx_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "rope":
        n = sum(t.num_elements() for t in op.outputs) if op.outputs else 0
        fwd_str = f"2×N, sin/cos mul = 2×{n} = {_fmt_e(ff)}"
        bwd_str = f"2.5×fwd = {_fmt_e(df)}"
        bytes_str = f"tensor I/O = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"dx_bytes = {_fmt_e(cost.dx_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "swiglu":
        n = sum(t.num_elements() for t in op.outputs) if op.outputs else 0
        fwd_str = f"5×N, sigmoid + mul = 5×{n} = {_fmt_e(ff)}"
        bwd_str = f"2.5×fwd = {_fmt_e(df)}"
        bytes_str = f"tensor I/O = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"dx_bytes = {_fmt_e(cost.dx_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "add":
        n = sum(t.num_elements() for t in op.outputs) if op.outputs else 0
        fwd_str = f"1×N, add = {n} = {_fmt_e(ff)}"
        bwd_str = f"2.5×fwd = {_fmt_e(df)}"
        bytes_str = f"tensor I/O = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"dx_bytes = {_fmt_e(cost.dx_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "softmax":
        n = sum(t.num_elements() for t in op.outputs) if op.outputs else 0
        fwd_str = f"4×N, max/sub/exp/div = 4×{n} = {_fmt_e(ff)}"
        bwd_str = f"2.5×fwd = {_fmt_e(df)}"
        bytes_str = f"tensor I/O = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"dx_bytes = {_fmt_e(cost.dx_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "indexer_topk":
        s = m.get("s", 0)
        kv = m.get("kv_len", s)
        ih = m.get("ih_local", m.get("ih", 0))
        id_ = m.get("id", 0)
        fwd_str = (
            f"2×s×kv_len×ih×id = "
            f"2×{s}×{kv}×{ih}×{id_} = {_fmt_e(ff)}"
        )
        bwd_str = f"2×fwd, indexer query gradient approximation = {_fmt_e(df)}"
        bytes_str = f"fwd_bytes = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = "dx_bytes = - or folded into matmul backward"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "compressor_pool":
        s = m.get("s", 0)
        mm_ = m.get("m", 4)
        co = m.get("coff", 1)
        dd = m.get("d_local", m.get("d", 0))
        fwd_str = (
            f"4×(s/m)×coff×m×d = "
            f"4×({s}//{mm_})×{co}×{mm_}×{dd} = {_fmt_e(ff)}"
        )
        bwd_str = f"= fwd = {_fmt_e(df)}"
        bytes_str = f"bytes = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"dx_bytes = {_fmt_e(cost.dx_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "embed":
        s = m.get("m", 0)
        h = m.get("n", 0)
        fwd_str = "0, gather-style embedding lookup has no modeled FLOPs"
        bwd_str = "0, scatter-style embedding gradient has no modeled FLOPs"
        bytes_str = f"s×h×bpe = {s}×{h}×bpe = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"same as fwd = {_fmt_e(cost.dx_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind in ("mhc_pre", "mhc_post", "mhc_head"):
        fwd_str = f"mHC fused op fwd = {_fmt_e(ff)}"
        bwd_str = f"2.5×fwd = {_fmt_e(df + wf)}"
        bytes_str = f"= {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"= {_fmt_e(cost.dx_bytes + cost.dw_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "hash_route":
        return "negligible", "negligible", "negligible", "negligible"

    return (
        f"fwd = {_fmt_e(ff)}",
        f"bwd = {_fmt_e(df + wf)}",
        f"bytes = {_fmt_e(cost.fwd_bytes)}",
        f"dx+dw bytes = {_fmt_e(cost.dx_bytes + cost.dw_bytes)}",
    )


def _op_detail(op, cost):
    """Return a dict with full op info for HTML."""
    inputs_info = _tensor_list_info(op.inputs, "input")
    outputs_info = _tensor_list_info(op.outputs, "output")
    fwd_formula, bwd_formula, fwd_bytes_formula, bwd_bytes_formula = _op_formula(op, cost)

    return {
        "inputs": inputs_info,
        "outputs": outputs_info,
        "fwd_formula": fwd_formula,
        "bwd_formula": bwd_formula,
        "fwd_bytes_formula": fwd_bytes_formula,
        "bwd_bytes_formula": bwd_bytes_formula,
    }


def _classify_ops_in_layer(ops: list["Op"], layer_kind: str) -> list[dict]:
    """Split a layer's ops into logical blocks.

    This function is intentionally heuristic. It groups common DeepSeek-like
    layer op names into user-facing blocks for the expandable report.
    """
    layer_kind = str(layer_kind).lower()

    blocks = {
        "HyperConnection": [],
        "Attention + Indexer": [],
        "Router + Shared Expert": [],
        "Routed Expert": [],
        "MTP Embed": [],
        "FFN": [],
        "Other Ops": [],
    }

    for op in ops:
        name = op.name.lower()
        kind = op.kind.lower()
        comp = str(getattr(op, "component", "") or "").lower()

        if kind in ("mhc_pre", "mhc_post", "mhc_head") or "mhc" in name or "hc_" in name:
            blocks["HyperConnection"].append(op)
            continue

        if layer_kind == "mtp" and ("mtp" in name or "embed_proj" in name):
            blocks["MTP Embed"].append(op)
            continue

        if (
            "attn" in name
            or "attention" in comp
            or "ln1" in name
            or "wq" in name
            or "wkv" in name
            or "q_norm" in name
            or "kv_norm" in name
            or "rope" in name
            or "compress" in name
            or "indexer" in name
            or "idx_" in name
            or "wo_" in name
            or kind in ("attn_core", "sparse_attn", "hca_attn", "swa_attn")
        ):
            blocks["Attention + Indexer"].append(op)
            continue

        if (
            "router" in name
            or "topk" in name
            or "hash_route" in name
            or "shared_" in name
            or "shared" in comp
        ):
            blocks["Router + Shared Expert"].append(op)
            continue

        if (
            "routed" in name
            or "expert_agg" in name
            or "expert" in comp
        ):
            blocks["Routed Expert"].append(op)
            continue

        if (
            "ln2" in name
            or "ffn" in name
            or "up_proj" in name
            or "down_proj" in name
            or "gate_proj" in name
            or "swiglu" in name
        ):
            blocks["FFN"].append(op)
            continue

        blocks["Other Ops"].append(op)

    ordered_names = [
        "MTP Embed",
        "HyperConnection",
        "Attention + Indexer",
        "Router + Shared Expert",
        "Routed Expert",
        "FFN",
        "Other Ops",
    ]

    return [
        {"name": name, "ops": blocks[name]}
        for name in ordered_names
        if blocks[name]
    ]


def _op_to_dict(op: "Op", cost: "OpCost", system) -> dict:
    """Convert an op + cost to a dict for the HTML template."""
    from zrt.training.compose.stage import _cost_phase_time, has_heterogeneous_compute

    gpu_name = system.gpu.name
    overlap = system.gpu.overlap_ratio.get(op.kind, 0.0) if has_heterogeneous_compute(system) else 0.0

    fwd_t = _cost_phase_time(cost, "fwd", system, gpu_name, overlap)
    dx_t = _cost_phase_time(cost, "dx", system, gpu_name, overlap)
    dw_t = _cost_phase_time(cost, "dw", system, gpu_name, overlap)

    detail = _op_detail(op, cost)

    component = str(getattr(op, "component", "") or "")

    return {
        "name": op.name,
        "kind": op.kind,
        "component": component,
        "layer_id": getattr(op, "layer_id", None),
        "layer_kind": str(getattr(op, "layer_kind", "") or ""),
        "fwd_flops": _fwd_flops(cost),
        "dx_flops": _dx_flops(cost),
        "dw_flops": _dw_flops(cost),
        "fwd_bytes": cost.fwd_bytes,
        "dx_bytes": cost.dx_bytes,
        "dw_bytes": cost.dw_bytes,
        "bound": cost.bound,
        "fwd_ms": fwd_t * 1000,
        "dx_ms": dx_t * 1000,
        "dw_ms": dw_t * 1000,
        "total_ms": (fwd_t + dx_t + dw_t) * 1000,
        "inputs": detail["inputs"],
        "outputs": detail["outputs"],
        "fwd_formula": detail["fwd_formula"],
        "bwd_formula": detail["bwd_formula"],
        "fwd_bytes_formula": detail["fwd_bytes_formula"],
        "bwd_bytes_formula": detail["bwd_bytes_formula"],
        "meta": getattr(op, "meta", {}) or {},
    }


def _enum_value(x) -> str:
    """Return enum.value when present, otherwise str(x)."""
    if x is None:
        return ""
    return str(getattr(x, "value", x))


def _safe_get(obj, name: str, default=None):
    """getattr with callable-property tolerance."""
    v = getattr(obj, name, default)
    try:
        if callable(v):
            return v()
    except TypeError:
        return v
    return v


def _component_of_op(op) -> str:
    """Map IR op to high-level model block for model-level summary."""
    name = op.name.lower()
    kind = op.kind.lower()
    comp = str(getattr(op, "component", "") or "").lower()

    if "input" in name:
        return "input"

    if "embed" in name:
        return "embedding"

    if (
        "attn" in name
        or "attention" in comp
        or "wq" in name
        or "wkv" in name
        or "q_norm" in name
        or "kv_norm" in name
        or "rope" in name
        or "compress" in name
        or "indexer" in name
        or "idx_" in name
        or "wo_" in name
        or kind in ("attn_core", "sparse_attn", "hca_attn", "swa_attn")
    ):
        return "attention"

    if (
        "moe" in comp
        or "expert" in name
        or "router" in name
        or "topk" in name
        or "swiglu" in name
        or "shared_" in name
        or "routed_" in name
        or "hash_route" in name
    ):
        return "moe"

    if "lm_head" in name or "final" in name or "logit" in name or "output" in name:
        return "output"

    return "other"


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _build_summary(
    report: "TrainingReport",
    graph: "Graph",
    model: "ModelSpec",
    system: "SystemSpec",
    strategy: "Strategy",
    op_dicts: list[dict],
) -> dict:
    gpu = getattr(system, "gpu", None)

    nodes = _safe_get(system, "nodes", 1) or 1
    gpus_per_node = _safe_get(system, "gpus_per_node", 1) or 1
    world_size = _safe_get(system, "world_size", None)
    if world_size is None:
        world_size = int(nodes) * int(gpus_per_node)

    layer_kinds = [_enum_value(x).lower() for x in getattr(model, "layers", [])]
    layer_count = len(layer_kinds)

    memory = {}
    if getattr(report, "memory", None) is not None and hasattr(report.memory, "to_gb"):
        memory = report.memory.to_gb()
    elif getattr(report, "memory_breakdown", None):
        memory = dict(report.memory_breakdown)

    compute_bound_ms = 0.0
    memory_bound_ms = 0.0

    for od in op_dicts:
        bound = str(od.get("bound", "")).lower()
        total_ms = _to_float(od.get("total_ms", 0.0))
        if "memory" in bound or "hbm" in bound:
            memory_bound_ms += total_ms
        elif "comm" in bound:
            pass
        else:
            compute_bound_ms += total_ms

    comm_ms = (
        _to_float(getattr(report, "exposed_comm_ms", 0.0))
        + _to_float(getattr(report, "optimizer_comm_ms", 0.0))
    )
    bound_total = max(compute_bound_ms + memory_bound_ms + comm_ms, 1e-12)

    return {
        "model": {
            "name": getattr(model, "model_type", "DeepSeek-like"),
            "hidden": getattr(model, "hidden", None),
            "layers": layer_count,
            "moe_layers": layer_kinds.count("moe"),
            "mtp_layers": layer_kinds.count("mtp"),
            "dense_layers": layer_kinds.count("dense"),
            "seq_len": getattr(model, "seq_len", None),
            "num_heads": getattr(model, "num_heads", None),
            "head_dim": getattr(model, "head_dim", None),
            "num_experts": getattr(model, "num_experts", None),
            "top_k": getattr(model, "top_k", None),
            "moe_ffn": getattr(model, "moe_ffn", None),
            "hc_mult": getattr(model, "hc_mult", None),
            "mtp_depth": getattr(model, "mtp_depth", None),
            "param_dtype": _enum_value(getattr(model, "param_dtype", "")),
            "act_dtype": _enum_value(getattr(model, "act_dtype", "")),
            "total_params": getattr(report, "total_params", 0),
            "effective_params": getattr(report, "effective_params", 0),
        },
        "hardware": {
            "name": getattr(gpu, "name", "unknown"),
            "hbm_gb": getattr(gpu, "hbm_gb", None),
            "hbm_bw_gbps": getattr(gpu, "hbm_bw_gbps", None),
            "tflops_bf16": getattr(gpu, "tflops_bf16", None),
            "tflops_fp16": getattr(gpu, "tflops_fp16", None),
            "tops_fp8": getattr(gpu, "tops_fp8", None),
            "cube_tflops": getattr(gpu, "cube_tflops", None),
            "vector_tflops": getattr(gpu, "vector_tflops", None),
        },
        "topology": {
            "nodes": nodes,
            "gpus_per_node": gpus_per_node,
            "world_size": world_size,
            "host_mem_gb": getattr(system, "host_mem_gb", None),
        },
        "strategy": {
            "tp": getattr(strategy, "tp", 1),
            "cp": getattr(strategy, "cp", 1),
            "pp": getattr(strategy, "pp", 1),
            "ep": getattr(strategy, "ep", 1),
            "dp": getattr(strategy, "dp", 1),
            "micro_batch": getattr(strategy, "micro_batch", None),
            "global_batch": getattr(strategy, "global_batch", None),
            "num_microbatches": (
                strategy.num_microbatches()
                if hasattr(strategy, "num_microbatches")
                else None
            ),
            "pp_schedule": _enum_value(getattr(strategy, "pp_schedule", "")),
            "zero_stage": getattr(strategy, "zero_stage", None),
            "optimizer": _enum_value(getattr(strategy, "optimizer", "")),
        },
        "performance": {
            "step_time_ms": getattr(report, "step_time_ms", 0.0),
            "pipeline_time_ms": getattr(report, "pipeline_time_ms", 0.0),
            "compute_time_ms": getattr(report, "compute_time_ms", 0.0),
            "exposed_comm_ms": getattr(report, "exposed_comm_ms", 0.0),
            "hidden_comm_ms": getattr(report, "hidden_comm_ms", 0.0),
            "recompute_time_ms": getattr(report, "recompute_time_ms", 0.0),
            "recompute_time_raw_ms": getattr(report, "recompute_time_raw_ms", 0.0),
            "optimizer_time_ms": getattr(report, "optimizer_time_ms", 0.0),
            "optimizer_comm_ms": getattr(report, "optimizer_comm_ms", 0.0),
            "tokens_per_sec": getattr(report, "tokens_per_sec", 0.0),
            "mfu": getattr(report, "mfu", 0.0),
            "hfu": getattr(report, "hfu", 0.0),
            "mfu_native": getattr(report, "mfu_native", 0.0),
            "bubble_fraction": getattr(report, "bubble_fraction", 0.0),
            "bubble_time_ms": getattr(report, "bubble_time_ms", 0.0),
            "flops_per_token": getattr(report, "flops_per_token", 0.0),
            "total_flops": getattr(report, "total_flops", 0.0),
            "recompute_time_ms": getattr(report, "recompute_time_ms", 0.0),
            "recompute_time_raw_ms": getattr(report, "recompute_time_raw_ms", 0.0),
            "bubble_time_ms": getattr(report, "bubble_time_ms", 0.0),
        },
        "memory": memory,
        "bound": {
            "compute_ms": compute_bound_ms,
            "memory_ms": memory_bound_ms,
            "communication_ms": comm_ms,
            "compute_pct": compute_bound_ms / bound_total,
            "memory_pct": memory_bound_ms / bound_total,
            "communication_pct": comm_ms / bound_total,
        },
        "comm_breakdown": {
            "tp_exposed_ms": getattr(report, "tp_exposed_ms", 0.0),
            "cp_exposed_ms": getattr(report, "cp_exposed_ms", 0.0),
            "ep_exposed_ms": getattr(report, "ep_exposed_ms", 0.0),
            "pp_exposed_ms": getattr(report, "pp_exposed_ms", 0.0),
            "dp_exposed_ms": getattr(report, "dp_exposed_ms", 0.0),
            "tp_hidden_ms": getattr(report, "tp_hidden_ms", 0.0),
            "ep_hidden_ms": getattr(report, "ep_hidden_ms", 0.0),
            "dp_hidden_ms": getattr(report, "dp_hidden_ms", 0.0),
            "total_comm_volume_ms": getattr(report, "total_comm_volume_ms", 0.0),
        },
    }


def _get_rank_placement_from_config(system, strategy):
    """Best-effort optional hook for real runtime rank placement.

    Supported shapes:
    1. strategy.rank_placement / system.rank_placement as list[dict]:
       [
         {"rank": 0, "node": 0, "local_rank": 0,
          "tp": 0, "pp": 0, "dp": 0, "ep": 0, "cp": 0},
         ...
       ]

    2. dict with key "ranks":
       {"ranks": [ ... ]}

    Existing configs continue to work. If no explicit placement is found, the
    report will fall back to an inferred visualization-only layout.
    """
    candidates = [
        getattr(strategy, "rank_placement", None),
        getattr(strategy, "rank_map", None),
        getattr(system, "rank_placement", None),
        getattr(system, "rank_map", None),
    ]

    for candidate in candidates:
        if not candidate:
            continue

        if isinstance(candidate, dict) and "ranks" in candidate:
            candidate = candidate["ranks"]

        if isinstance(candidate, list):
            normalized = []
            for item in candidate:
                if not isinstance(item, dict):
                    continue
                if "rank" not in item:
                    continue

                rank = _to_int(item.get("rank", 0))
                normalized.append(
                    {
                        "rank": rank,
                        "node": _to_int(item.get("node", 0)),
                        "local_rank": _to_int(item.get("local_rank", rank)),
                        "tp": _to_int(item.get("tp", 0)),
                        "pp": _to_int(item.get("pp", 0)),
                        "dp": _to_int(item.get("dp", 0)),
                        "ep": _to_int(item.get("ep", 0)),
                        "cp": _to_int(item.get("cp", 0)),
                    }
                )

            if normalized:
                return normalized

    return None


def _build_topology_data(system, strategy) -> dict:
    nodes = int(_safe_get(system, "nodes", 1) or 1)
    gpus_per_node = int(_safe_get(system, "gpus_per_node", 1) or 1)

    world_size = _safe_get(system, "world_size", None)
    if world_size is None:
        world_size = nodes * gpus_per_node
    world_size = int(world_size)

    tp = max(1, int(getattr(strategy, "tp", 1) or 1))
    pp = max(1, int(getattr(strategy, "pp", 1) or 1))
    ep = max(1, int(getattr(strategy, "ep", 1) or 1))
    dp = max(1, int(getattr(strategy, "dp", 1) or 1))
    cp = max(1, int(getattr(strategy, "cp", 1) or 1))

    real_placement = _get_rank_placement_from_config(system, strategy)

    if real_placement:
        by_node: dict[int, list[dict]] = {}
        for r in real_placement:
            by_node.setdefault(int(r["node"]), []).append(r)

        rank_nodes = []
        for n in sorted(by_node):
            ranks = sorted(by_node[n], key=lambda x: x["local_rank"])
            rank_nodes.append({"node": n, "ranks": ranks})

        return {
            "nodes": rank_nodes,
            "legend": {
                "mapping_source": "configured_runtime_rank_placement",
                "is_inferred": False,
                "note": (
                    "Rank placement is loaded from configured runtime rank placement. "
                    "It should match the launcher or distributed runtime mapping."
                ),
                "tp": tp,
                "pp": pp,
                "ep": ep,
                "dp": dp,
                "cp": cp,
            },
        }

    rank_nodes = []
    for n in range(nodes):
        ranks = []
        for local in range(gpus_per_node):
            r = n * gpus_per_node + local
            if r >= world_size:
                continue

            # Visualization-only heuristic.
            # Do not treat this as authoritative runtime rank placement.
            tp_idx = r % tp
            pp_idx = (r // tp) % pp
            dp_idx = (r // max(1, tp * pp)) % dp
            ep_idx = r % ep
            cp_idx = r % cp

            ranks.append(
                {
                    "rank": r,
                    "local_rank": local,
                    "node": n,
                    "tp": tp_idx,
                    "pp": pp_idx,
                    "dp": dp_idx,
                    "ep": ep_idx,
                    "cp": cp_idx,
                }
            )

        rank_nodes.append({"node": n, "ranks": ranks})

    return {
        "nodes": rank_nodes,
        "legend": {
            "mapping_source": "inferred_visualization_only",
            "is_inferred": True,
            "note": (
                "This topology is an inferred logical visualization only. "
                "It may not match the real runtime rank placement used by torchrun, "
                "MindSpore, HCCL, Megatron, or a custom launcher. For accurate topology "
                "validation, provide explicit rank_placement data."
            ),
            "tp": tp,
            "pp": pp,
            "ep": ep,
            "dp": dp,
            "cp": cp,
        },
    }


def _build_model_overview(op_dicts: list[dict]) -> list[dict]:
    buckets = {
        "input": {
            "name": "Input",
            "desc": "tokens / batch / sequence",
            "total_ms": 0.0,
            "ops": 0,
        },
        "embedding": {
            "name": "Embedding",
            "desc": "token embedding / hc_expand",
            "total_ms": 0.0,
            "ops": 0,
        },
        "attention": {
            "name": "Attention",
            "desc": "Q/KV, RoPE, CSA/HCA/SWA, output projection",
            "total_ms": 0.0,
            "ops": 0,
        },
        "moe": {
            "name": "MoE",
            "desc": "router, shared expert, routed experts",
            "total_ms": 0.0,
            "ops": 0,
        },
        "output": {
            "name": "Output",
            "desc": "final norm / lm_head",
            "total_ms": 0.0,
            "ops": 0,
        },
        "other": {
            "name": "Other",
            "desc": "residual / misc ops",
            "total_ms": 0.0,
            "ops": 0,
        },
    }

    for od in op_dicts:
        comp = od.get("component_group", "other")
        if comp not in buckets:
            comp = "other"
        buckets[comp]["total_ms"] += _to_float(od.get("total_ms", 0.0))
        buckets[comp]["ops"] += 1

    ordered = [
        buckets[k]
        for k in ("input", "embedding", "attention", "moe", "output", "other")
    ]

    total = max(sum(x["total_ms"] for x in ordered), 1e-12)
    for x in ordered:
        x["pct"] = x["total_ms"] / total

    return ordered


def _build_layer_tree(
    graph: "Graph",
    model: "ModelSpec",
    system: "SystemSpec",
    op_costs: dict[str, "OpCost"],
) -> tuple[dict, list[dict]]:
    """Build hierarchical tree: model -> layer -> block -> op."""
    layers = getattr(model, "layers", [])

    tree = {
        "model_name": f"{len(layers)} layers, hidden={getattr(model, 'hidden', '-')}",
        "total_ops": len(graph.ops),
        "layers": [],
        "global_ops": [],
    }

    all_op_dicts: list[dict] = []

    from zrt.training.models.flops import op_cost

    # Global ops: embedding / final head etc.
    for op in graph.ops:
        if getattr(op, "layer_id", -1) < 0:
            cost = op_costs.get(op.name)
            if cost is None:
                cost = op_cost(op, model, system)

            od = _op_to_dict(op, cost, system)
            od["component_group"] = _component_of_op(op)

            tree["global_ops"].append(od)
            all_op_dicts.append(od)

    # Per-layer data.
    for lid in range(len(layers)):
        lk = _enum_value(layers[lid]).lower()
        layer_ops = graph.ops_for_layer(lid)
        blocks = _classify_ops_in_layer(layer_ops, lk)

        layer_data = {
            "id": lid,
            "kind": lk,
            "op_count": len(layer_ops),
            "total_ms": 0.0,
            "blocks": [],
        }

        captured = set()

        for blk in blocks:
            blk_data = {
                "name": blk["name"],
                "ops": [],
                "total_ms": 0.0,
                "bound": {
                    "compute_ms": 0.0,
                    "memory_ms": 0.0,
                    "other_ms": 0.0,
                },
            }

            for op in blk["ops"]:
                captured.add(op.name)

                cost = op_costs.get(op.name)
                if cost is None:
                    cost = op_cost(op, model, system)

                od = _op_to_dict(op, cost, system)
                od["component_group"] = _component_of_op(op)

                blk_data["ops"].append(od)
                all_op_dicts.append(od)

                ms = _to_float(od.get("total_ms", 0.0))
                blk_data["total_ms"] += ms

                b = str(od.get("bound", "")).lower()
                if "memory" in b or "hbm" in b:
                    blk_data["bound"]["memory_ms"] += ms
                elif "compute" in b or "cube" in b or "vector" in b:
                    blk_data["bound"]["compute_ms"] += ms
                else:
                    blk_data["bound"]["other_ms"] += ms

            layer_data["blocks"].append(blk_data)
            layer_data["total_ms"] += blk_data["total_ms"]

        missed = [op for op in layer_ops if op.name not in captured]
        if missed:
            blk_data = {
                "name": "Other Ops",
                "ops": [],
                "total_ms": 0.0,
                "bound": {
                    "compute_ms": 0.0,
                    "memory_ms": 0.0,
                    "other_ms": 0.0,
                },
            }

            for op in missed:
                cost = op_costs.get(op.name)
                if cost is None:
                    cost = op_cost(op, model, system)

                od = _op_to_dict(op, cost, system)
                od["component_group"] = _component_of_op(op)

                blk_data["ops"].append(od)
                all_op_dicts.append(od)

                ms = _to_float(od.get("total_ms", 0.0))
                blk_data["total_ms"] += ms

                b = str(od.get("bound", "")).lower()
                if "memory" in b or "hbm" in b:
                    blk_data["bound"]["memory_ms"] += ms
                elif "compute" in b or "cube" in b or "vector" in b:
                    blk_data["bound"]["compute_ms"] += ms
                else:
                    blk_data["bound"]["other_ms"] += ms

            layer_data["blocks"].append(blk_data)
            layer_data["total_ms"] += blk_data["total_ms"]

        tree["layers"].append(layer_data)

    return tree, all_op_dicts


def _build_calibration_data(
    report: "TrainingReport",
    graph: "Graph",
    model: "ModelSpec",
    strategy: "Strategy",
    all_op_dicts: list[dict],
) -> list[dict]:
    """Build reference calibration rows.

    Public references are used as sanity checks, not as automatic correction
    unless the current scenario is known to be comparable.
    """
    rows = []

    hidden = getattr(model, "hidden", None)
    num_experts = getattr(model, "num_experts", None)
    top_k = getattr(model, "top_k", None)
    moe_ffn = getattr(model, "moe_ffn", None)

    is_dsv4_pro_like = (
        hidden == 7168
        and num_experts == 384
        and top_k == 6
        and moe_ffn == 3072
    )

    rows.append(
        {
            "name": "DeepSeek-V4-Pro architecture",
            "official": (
                "DeepSeek-V4-Pro public model card/technical material: "
                "1.6T total params, 49B activated params, 1M context, "
                "FP4+FP8 mixed precision."
            ),
            "modeled": (
                f"hidden={hidden}, experts={num_experts}, top_k={top_k}, "
                f"moe_ffn={moe_ffn}, layers={len(getattr(model, 'layers', []))}"
            ),
            "status": "pass" if is_dsv4_pro_like else "check",
            "delta": "structure check",
            "source": "DeepSeek-V4-Pro model card / technical report",
            "url": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro",
            "note": (
                "Use this row to confirm the YAML/model spec matches the public "
                "DeepSeek-V4-Pro geometry."
            ),
        }
    )

    routed_ops = [
        od
        for od in all_op_dicts
        if "routed_expert_ffn" in od.get("name", "")
        or "routed" in od.get("component", "").lower()
    ]

    modeled_us = None
    if routed_ops:
        modeled_us = (
            sum(_to_float(x.get("fwd_ms", 0.0)) for x in routed_ops)
            / len(routed_ops)
            * 1000.0
        )

    ep = getattr(strategy, "ep", 1)
    comparable = is_dsv4_pro_like and int(ep) == 8

    official_us = 369.6
    if modeled_us is not None and comparable:
        err_pct = abs(modeled_us - official_us) / official_us * 100.0
        delta = f"{err_pct:.1f}%"
        status = "pass" if err_pct <= 20.0 else "warn"
        modeled = f"{modeled_us:.1f} us avg routed_expert_fwd"
    elif modeled_us is not None:
        delta = "not apples-to-apples"
        status = "info"
        modeled = f"{modeled_us:.1f} us avg routed_expert_fwd"
    else:
        delta = "no routed expert op found"
        status = "check"
        modeled = "-"

    rows.append(
        {
            "name": "DeepGEMM Mega MoE benchmark",
            "official": (
                "DeepSeek-V4-Pro EP8, 512 tokens/rank: 369.6 us; "
                "1098 TFLOPS; 4619 GB/s global memory; 182 GB/s interconnect."
            ),
            "modeled": modeled,
            "status": status,
            "delta": delta,
            "source": "DeepGEMM PR #316",
            "url": "https://github.com/deepseek-ai/DeepGEMM/pull/316",
            "note": (
                "This is a serving MegaMoE kernel reference. For this training "
                "report, treat it as a routed expert kernel sanity check unless "
                "EP=8, token shape and dtype match the PR scenario."
            ),
        }
    )

    rows.append(
        {
            "name": "Step-level calibration placeholder",
            "official": (
                "No public official full-step Ascend 910C DeepSeek-V4-Pro "
                "training reference is embedded in this exporter."
            ),
            "modeled": (
                f"step_time={getattr(report, 'step_time_ms', 0.0):.3f} ms, "
                f"MFU={getattr(report, 'mfu', 0.0) * 100:.2f}%"
            ),
            "status": "check",
            "delta": "requires local cluster sampling",
            "source": "Local benchmark hook",
            "url": "",
            "note": (
                "Recommended: add one measured step-time sample from the target "
                "cluster and store it in report metadata; then compute "
                "correction_factor = measured_step_ms / modeled_step_ms."
            ),
        }
    )

    return rows


def _build_html_data(
    graph: "Graph",
    model: "ModelSpec",
    system: "SystemSpec",
    strategy: "Strategy",
    report: "TrainingReport",
    op_costs: dict[str, "OpCost"],
) -> dict:
    tree, all_op_dicts = _build_layer_tree(graph, model, system, op_costs)
    summary = _build_summary(report, graph, model, system, strategy, all_op_dicts)

    return {
        "summary": summary,
        "topology": _build_topology_data(system, strategy),
        "model_overview": _build_model_overview(all_op_dicts),
        "tree": tree,
        "calibration": _build_calibration_data(
            report,
            graph,
            model,
            strategy,
            all_op_dicts,
        ),
        "warnings": getattr(report, "warnings", []),
    }


def _json_parse_literal_for_script(data) -> str:
    """Return a safe JavaScript string literal for JSON.parse(...).

    This avoids raw object literal injection inside <script>. The data becomes a
    JS string literal argument to JSON.parse, so substrings such as ${...} or
    backticks inside model/op names are data, not executable template syntax.
    """
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    # Prevent data from closing the script element.
    payload = payload.replace("</", "<\\/")

    # Defensive escaping for JavaScript string literal compatibility.
    payload = payload.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")

    # Return a quoted JS string literal. ensure_ascii=True makes the outer JS
    # literal ASCII-safe even when the report contains Chinese text.
    return json.dumps(payload, ensure_ascii=True)


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>__TITLE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
:root {
  --bg: #f5f7fb;
  --panel: #ffffff;
  --panel2: #f8fafc;
  --text: #152033;
  --muted: #637083;
  --line: #d9e1ec;
  --blue: #2563eb;
  --green: #16a34a;
  --amber: #d97706;
  --red: #dc2626;
  --purple: #7c3aed;
  --chip: #eef2ff;
  --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--text);
}
header {
  padding: 28px 36px;
  background: linear-gradient(135deg, #0f172a, #1e3a8a);
  color: white;
}
header h1 { margin: 0 0 8px; font-size: 28px; }
header p { margin: 0; color: #dbeafe; }
main { padding: 24px 36px 64px; }
section {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 22px;
  margin-bottom: 22px;
  box-shadow: var(--shadow);
}
h2 { margin: 0 0 14px; font-size: 22px; }
h3 { margin: 16px 0 10px; font-size: 16px; }
small, .muted { color: var(--muted); }
.grid { display: grid; gap: 14px; }
.grid-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.grid-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.grid-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.card {
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 14px;
  background: var(--panel2);
}
.metric {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.metric .label { font-size: 12px; color: var(--muted); }
.metric .value { font-size: 20px; font-weight: 760; }
.kv {
  display: grid;
  grid-template-columns: 190px 1fr;
  gap: 8px 14px;
  font-size: 13px;
}
.kv div:nth-child(odd) { color: var(--muted); }
.bound-bar {
  display: flex;
  width: 100%;
  height: 18px;
  overflow: hidden;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: #e5e7eb;
}
.bound-bar span { display: block; height: 100%; }
.bound-compute { background: #2563eb; }
.bound-memory { background: #d97706; }
.bound-comm { background: #7c3aed; }
.legend {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 10px;
  font-size: 12px;
}
.dot {
  width: 10px;
  height: 10px;
  display: inline-block;
  border-radius: 999px;
  margin-right: 5px;
}
.topology {
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
}
.node {
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 14px;
  background: #fbfdff;
}
.node-title {
  font-weight: 760;
  margin-bottom: 10px;
}
.ranks {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}
.rank {
  border: 1px solid #cbd5e1;
  border-radius: 12px;
  padding: 8px;
  background: white;
  font-size: 12px;
}
.rank strong { display: block; margin-bottom: 4px; }
.chips {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}
.chip {
  border-radius: 999px;
  padding: 2px 6px;
  background: var(--chip);
  color: #3730a3;
  font-size: 10px;
  font-weight: 650;
}
.chip.pp { background: #dbeafe; color: #1d4ed8; }
.chip.tp { background: #dcfce7; color: #166534; }
.chip.ep { background: #fef3c7; color: #92400e; }
.chip.dp { background: #f3e8ff; color: #6b21a8; }
.chip.cp { background: #fee2e2; color: #991b1b; }

.model-flow {
  display: grid;
  grid-template-columns: repeat(6, minmax(120px, 1fr));
  gap: 14px;
  align-items: stretch;
}
.flow-block {
  position: relative;
  border: 2px solid #dbeafe;
  border-radius: 18px;
  padding: 14px;
  background: linear-gradient(180deg, #ffffff, #eff6ff);
  min-height: 132px;
}
.flow-block:not(:last-child)::after {
  content: "→";
  position: absolute;
  right: -18px;
  top: 42%;
  color: #64748b;
  font-size: 22px;
}
.flow-title { font-weight: 800; margin-bottom: 8px; }
.flow-time { font-size: 20px; font-weight: 800; color: #1d4ed8; }
.flow-desc { font-size: 12px; color: var(--muted); min-height: 30px; }
.mini-bar {
  height: 8px;
  border-radius: 999px;
  background: #e5e7eb;
  overflow: hidden;
  margin-top: 10px;
}
.mini-bar span {
  display: block;
  height: 100%;
  background: #2563eb;
}

.toolbar {
  display: flex;
  gap: 10px;
  margin: 10px 0 16px;
}
button, input {
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 8px 10px;
  background: white;
}
button {
  cursor: pointer;
  font-weight: 650;
}
input { min-width: 260px; }
details {
  border: 1px solid var(--line);
  border-radius: 12px;
  margin: 8px 0;
  background: white;
}
details > summary {
  list-style: none;
  cursor: pointer;
  padding: 12px 14px;
  font-weight: 760;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
details > summary::-webkit-details-marker { display: none; }
details > summary::before {
  content: "+";
  width: 22px;
  height: 22px;
  border-radius: 6px;
  margin-right: 8px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: #e0ecff;
  color: #1d4ed8;
  font-weight: 800;
}
details[open] > summary::before { content: "−"; }
.summary-left {
  display: flex;
  align-items: center;
  gap: 8px;
}
.summary-right {
  color: var(--muted);
  font-weight: 600;
  font-size: 12px;
}
.op-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  margin: 0 0 10px;
}
.op-table th, .op-table td {
  border-top: 1px solid var(--line);
  padding: 8px;
  vertical-align: top;
  text-align: left;
}
.op-table th {
  background: #f8fafc;
  color: #475569;
}
.formula {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 11px;
  white-space: pre-wrap;
  color: #334155;
}
.bound {
  display: inline-block;
  border-radius: 999px;
  padding: 3px 8px;
  font-size: 11px;
  font-weight: 760;
}
.bound.compute { background: #dbeafe; color: #1e40af; }
.bound.memory { background: #fef3c7; color: #92400e; }
.bound.communication { background: #ede9fe; color: #6d28d9; }
.bound.other { background: #f1f5f9; color: #475569; }
.calib-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
.calib-table th, .calib-table td {
  border-top: 1px solid var(--line);
  padding: 10px;
  text-align: left;
  vertical-align: top;
}
.status {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 999px;
  font-weight: 800;
  font-size: 11px;
}
.status.pass { background: #dcfce7; color: #166534; }
.status.warn { background: #fef3c7; color: #92400e; }
.status.check { background: #fee2e2; color: #991b1b; }
.status.info { background: #dbeafe; color: #1e40af; }
.warning {
  border-left: 4px solid var(--amber);
  background: #fffbeb;
  padding: 10px 12px;
  border-radius: 10px;
  margin: 8px 0;
}
@media (max-width: 1100px) {
  .grid-2, .grid-3, .grid-4, .model-flow { grid-template-columns: 1fr; }
  .flow-block:not(:last-child)::after { display: none; }
}
</style>
</head>
<body>
<header>
  <h1>ZRT Training Estimate Report</h1>
  <p>Spec-driven IR graph report: summary, topology, model-level timing, expandable op analysis, and calibration.</p>
</header>

<main>
  <section id="summary"></section>
  <section id="topology"></section>
  <section id="model-overview"></section>
  <section id="hierarchy"></section>
  <section id="calibration"></section>
</main>

<script>
const DATA = JSON.parse(__DATA_JSON__);

const fmt = {
  ms(v) {
    v = Number(v || 0);
    if (Math.abs(v) >= 1000) return (v / 1000).toFixed(2) + "s";
    if (Math.abs(v) >= 1) return v.toFixed(2) + "ms";
    return (v * 1000).toFixed(2) + "us";
  },
  pct(v) { return (Number(v || 0) * 100).toFixed(2) + "%"; },
  num(v) {
    v = Number(v || 0);
    if (Math.abs(v) >= 1e15) return (v / 1e15).toFixed(2) + "P";
    if (Math.abs(v) >= 1e12) return (v / 1e12).toFixed(2) + "T";
    if (Math.abs(v) >= 1e9) return (v / 1e9).toFixed(2) + "G";
    if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(2) + "M";
    if (Math.abs(v) >= 1e3) return (v / 1e3).toFixed(2) + "K";
    return String(v.toFixed ? v.toFixed(2) : v);
  },
  gb(v) { return Number(v || 0).toFixed(2) + " GB"; }
};

function esc(x) {
  return String(x ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[ch]));
}

function boundClass(b) {
  b = String(b || "").toLowerCase();
  if (b.includes("memory") || b.includes("hbm")) return "memory";
  if (b.includes("comm")) return "communication";
  if (b.includes("compute") || b.includes("cube") || b.includes("vector")) return "compute";
  return "other";
}

function renderSummary() {
  const s = DATA.summary;
  const p = s.performance;
  const b = s.bound;
  const m = s.memory || {};

  document.getElementById("summary").innerHTML = `
    <h2>1. 报告 Summary</h2>

    <div class="grid grid-4">
      <div class="card metric"><div class="label">Step Time</div><div class="value">${fmt.ms(p.step_time_ms)}</div></div>
      <div class="card metric"><div class="label">Tokens / sec</div><div class="value">${fmt.num(p.tokens_per_sec)}</div></div>
      <div class="card metric"><div class="label">MFU</div><div class="value">${fmt.pct(p.mfu)}</div></div>
      <div class="card metric"><div class="label">HFU</div><div class="value">${fmt.pct(p.hfu)}</div></div>
    </div>

    <h3>模型 / 硬件 / 拓扑 / 并行策略</h3>
    <div class="grid grid-2">
      <div class="card">
        <h3>Model</h3>
        <div class="kv">
          <div>name</div><div>${esc(s.model.name)}</div>
          <div>hidden / layers</div><div>${esc(s.model.hidden)} / ${esc(s.model.layers)}</div>
          <div>MoE / MTP / Dense</div><div>${esc(s.model.moe_layers)} / ${esc(s.model.mtp_layers)} / ${esc(s.model.dense_layers)}</div>
          <div>experts / top_k / moe_ffn</div><div>${esc(s.model.num_experts)} / ${esc(s.model.top_k)} / ${esc(s.model.moe_ffn)}</div>
          <div>heads / head_dim</div><div>${esc(s.model.num_heads)} / ${esc(s.model.head_dim)}</div>
          <div>seq_len</div><div>${esc(s.model.seq_len)}</div>
          <div>dtype</div><div>param=${esc(s.model.param_dtype)}, act=${esc(s.model.act_dtype)}</div>
        </div>
      </div>

      <div class="card">
        <h3>Hardware / Topology</h3>
        <div class="kv">
          <div>accelerator</div><div>${esc(s.hardware.name)}</div>
          <div>HBM</div><div>${esc(s.hardware.hbm_gb)} GB, ${esc(s.hardware.hbm_bw_gbps)} GB/s</div>
          <div>compute</div><div>BF16=${esc(s.hardware.tflops_bf16)} TFLOPS, FP8=${esc(s.hardware.tops_fp8)} TOPS</div>
          <div>nodes × ranks/node</div><div>${esc(s.topology.nodes)} × ${esc(s.topology.gpus_per_node)}</div>
          <div>world size</div><div>${esc(s.topology.world_size)}</div>
          <div>parallel</div><div>TP=${esc(s.strategy.tp)}, CP=${esc(s.strategy.cp)}, PP=${esc(s.strategy.pp)}, EP=${esc(s.strategy.ep)}, DP=${esc(s.strategy.dp)}</div>
          <div>schedule / optimizer</div><div>${esc(s.strategy.pp_schedule)} / ${esc(s.strategy.optimizer)}</div>
        </div>
      </div>
    </div>

    <h3>整体性能评估结果</h3>
    <div class="grid grid-3">
      <div class="card kv">
        <div>pipeline time</div><div>${fmt.ms(p.pipeline_time_ms)}</div>
        <div>compute time</div><div>${fmt.ms(p.compute_time_ms)}</div>
        <div>exposed comm</div><div>${fmt.ms(p.exposed_comm_ms)}</div>
        <div>pipeline bubble</div><div>${fmt.ms(p.bubble_time_ms)}</div>
      </div>

      <div class="card kv">
        <div>optimizer time</div><div>${fmt.ms(p.optimizer_time_ms)}</div>
        <div>optimizer comm</div><div>${fmt.ms(p.optimizer_comm_ms)}</div>
        <div>bubble fraction</div><div>${fmt.pct(p.bubble_fraction)}</div>
        <div>FLOPs/token</div><div>${fmt.num(p.flops_per_token)}</div>
      </div>

      <div class="card kv">
        <div>TP exposed</div><div>${fmt.ms(s.comm_breakdown.tp_exposed_ms)}</div>
        <div>EP exposed</div><div>${fmt.ms(s.comm_breakdown.ep_exposed_ms)}</div>
        <div>DP exposed</div><div>${fmt.ms(s.comm_breakdown.dp_exposed_ms)}</div>
        <div>comm volume</div><div>${fmt.ms(s.comm_breakdown.total_comm_volume_ms)}</div>
      </div>
    </div>

    <h3>step time breakdown</h3>
    <div class="card kv">
      <div>Useful Compute</div><div>${fmt.ms(p.compute_time_ms)}</div>
      <div>Recompute (crit. path)</div><div>${fmt.ms(p.recompute_time_ms || 0)}</div>
      <div>Communication (exposed)</div><div>${fmt.ms(p.exposed_comm_ms)}</div>
      <div>Pipeline Bubble</div><div>${fmt.ms(p.bubble_time_ms)}</div>
      <div>Optimizer (compute)</div><div>${fmt.ms(p.optimizer_time_ms)}</div>
      <div>Optimizer (comm)</div><div>${fmt.ms(p.optimizer_comm_ms)}</div>
    </div>

    <h3>内存消耗结果</h3>
    <div class="grid grid-4">
      ${Object.entries(m).map(([k, v]) => `
        <div class="card metric">
          <div class="label">${esc(k)}</div>
          <div class="value">${fmt.gb(v)}</div>
        </div>
      `).join("")}
    </div>

    <h3>计算过程 Bound 占比</h3>
    <div class="card">
      <div class="bound-bar">
        <span class="bound-compute" style="width:${b.compute_pct * 100}%"></span>
        <span class="bound-memory" style="width:${b.memory_pct * 100}%"></span>
        <span class="bound-comm" style="width:${b.communication_pct * 100}%"></span>
      </div>
      <div class="legend">
        <span><i class="dot bound-compute"></i>Compute ${fmt.pct(b.compute_pct)} / ${fmt.ms(b.compute_ms)}</span>
        <span><i class="dot bound-memory"></i>Memory ${fmt.pct(b.memory_pct)} / ${fmt.ms(b.memory_ms)}</span>
        <span><i class="dot bound-comm"></i>Communication ${fmt.pct(b.communication_pct)} / ${fmt.ms(b.communication_ms)}</span>
      </div>
    </div>

    ${(DATA.warnings || []).map(w => `<div class="warning">${esc(w)}</div>`).join("")}
  `;
}

function renderTopology() {
  const t = DATA.topology;

  document.getElementById("topology").innerHTML = `
    <h2>2. 集群逻辑拓扑</h2>

    ${t.legend.is_inferred ? `
      <div class="warning">
        <strong>注意：</strong>当前 Rank 拓扑是 inferred visualization only，不代表真实 runtime rank placement。
        如果 launcher 使用自定义 rank order，请在配置中提供 rank_placement 后再用于拓扑校验。
      </div>
    ` : `
      <div class="card">
        <strong>Rank placement source:</strong> ${esc(t.legend.mapping_source)}
      </div>
    `}

    <p class="muted">${esc(t.legend.note)}</p>

    <div class="topology">
      ${t.nodes.map(node => `
        <div class="node">
          <div class="node-title">Node ${node.node}</div>
          <div class="ranks">
            ${node.ranks.map(r => `
              <div class="rank">
                <strong>Rank ${r.rank}</strong>
                <div class="muted">local ${r.local_rank}</div>
                <div class="chips">
                  <span class="chip pp">PP${r.pp}</span>
                  <span class="chip tp">TP${r.tp}</span>
                  <span class="chip ep">EP${r.ep}</span>
                  <span class="chip dp">DP${r.dp}</span>
                  <span class="chip cp">CP${r.cp}</span>
                </div>
              </div>
            `).join("")}
          </div>
        </div>
      `).join("")}
    </div>
  `;
}

function renderModelOverview() {
  const blocks = DATA.model_overview || [];

  document.getElementById("model-overview").innerHTML = `
    <h2>3. 模型级计算耗时</h2>
    <p class="muted">Transformer Explainer 风格概览：Input → Embedding → Attention → MoE → Output。耗时来自 IR op 的 fwd + dx + dw 估算。</p>

    <div class="model-flow">
      ${blocks.map(b => `
        <div class="flow-block">
          <div class="flow-title">${esc(b.name)}</div>
          <div class="flow-desc">${esc(b.desc)}</div>
          <div class="flow-time">${fmt.ms(b.total_ms)}</div>
          <div class="muted">${esc(b.ops)} ops · ${fmt.pct(b.pct)}</div>
          <div class="mini-bar"><span style="width:${Math.max(1, b.pct * 100)}%"></span></div>
        </div>
      `).join("")}
    </div>
  `;
}

function renderOpTable(ops) {
  return `
    <table class="op-table">
      <thead>
        <tr>
          <th>Op</th>
          <th>Kind</th>
          <th>Time</th>
          <th>Bound</th>
          <th>Formula / 原理</th>
          <th>Tensor I/O</th>
        </tr>
      </thead>
      <tbody>
        ${ops.map(op => {
          const bc = boundClass(op.bound);
          return `
            <tr class="op-row" data-search="${esc((op.name + " " + op.kind + " " + op.bound).toLowerCase())}">
              <td><strong>${esc(op.name)}</strong></td>
              <td>${esc(op.kind)}</td>
              <td>
                total=${fmt.ms(op.total_ms)}<br/>
                fwd=${fmt.ms(op.fwd_ms)}<br/>
                dx=${fmt.ms(op.dx_ms)}<br/>
                dw=${fmt.ms(op.dw_ms)}
              </td>
              <td><span class="bound ${bc}">${esc(op.bound || bc)}</span></td>
              <td class="formula">
                <b>FWD FLOPs</b>: ${esc(op.fwd_formula)}<br/>
                <b>BWD FLOPs</b>: ${esc(op.bwd_formula)}<br/>
                <b>FWD Bytes</b>: ${esc(op.fwd_bytes_formula)}<br/>
                <b>BWD Bytes</b>: ${esc(op.bwd_bytes_formula)}<br/>
                <b>原理</b>: roofline = max(compute_time, memory_time); bound 表示该 op 当前瓶颈来源。
              </td>
              <td class="formula">
                <b>Inputs</b>: ${esc(op.inputs)}<br/>
                <b>Outputs</b>: ${esc(op.outputs)}
              </td>
            </tr>
          `;
        }).join("")}
      </tbody>
    </table>
  `;
}

function renderHierarchy() {
  const tree = DATA.tree;

  document.getElementById("hierarchy").innerHTML = `
    <h2>4. 分层展示计算耗时</h2>
    <p class="muted">模型 → Layer / Block → 算子。每一层可以展开和收缩，算子层包含耗时、公式、公式原理和 bound 类型。</p>

    <div class="toolbar">
      <button onclick="setAllDetails(true)">expand all</button>
      <button onclick="setAllDetails(false)">collapse all</button>
      <input id="opSearch" placeholder="search op / kind / bound..." oninput="filterOps(this.value)" />
    </div>

    <details open>
      <summary>
        <span class="summary-left">Model: ${esc(tree.model_name)}</span>
        <span class="summary-right">${esc(tree.total_ops)} ops</span>
      </summary>

      ${tree.global_ops && tree.global_ops.length ? `
        <details>
          <summary>
            <span class="summary-left">Global Ops</span>
            <span class="summary-right">${tree.global_ops.length} ops · ${fmt.ms(tree.global_ops.reduce((a,b)=>a+Number(b.total_ms||0),0))}</span>
          </summary>
          ${renderOpTable(tree.global_ops)}
        </details>
      ` : ""}

      ${tree.layers.map(layer => `
        <details>
          <summary>
            <span class="summary-left">Layer ${layer.id} · ${esc(layer.kind)}</span>
            <span class="summary-right">${layer.op_count} ops · ${fmt.ms(layer.total_ms)}</span>
          </summary>

          ${layer.blocks.map(block => `
            <details>
              <summary>
                <span class="summary-left">${esc(block.name)}</span>
                <span class="summary-right">${block.ops.length} ops · ${fmt.ms(block.total_ms)}</span>
              </summary>
              ${renderOpTable(block.ops)}
            </details>
          `).join("")}
        </details>
      `).join("")}
    </details>
  `;
}

function setAllDetails(open) {
  document.querySelectorAll("details").forEach(d => d.open = open);
}

function filterOps(q) {
  q = String(q || "").trim().toLowerCase();
  document.querySelectorAll(".op-row").forEach(row => {
    if (!q) {
      row.style.display = "";
    } else {
      row.style.display = row.dataset.search.includes(q) ? "" : "none";
    }
  });
}

function renderCalibration() {
  const rows = DATA.calibration || [];

  document.getElementById("calibration").innerHTML = `
    <h2>5. 报告校准</h2>
    <p class="muted">校准分为结构校验、公开 kernel benchmark 对齐、以及目标集群本地 step-time 采样。公开 benchmark 不一定与当前训练配置完全同构，需要看 status 和 note。</p>

    <table class="calib-table">
      <thead>
        <tr>
          <th>校准项</th>
          <th>官方数据</th>
          <th>本报告数据</th>
          <th>偏差 / 状态</th>
          <th>引用出处</th>
          <th>说明</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map(r => `
          <tr>
            <td><strong>${esc(r.name)}</strong></td>
            <td>${esc(r.official)}</td>
            <td>${esc(r.modeled)}</td>
            <td><span class="status ${esc(r.status)}">${esc(r.status)}</span><br/>${esc(r.delta)}</td>
            <td>${r.url ? `<a href="${esc(r.url)}" target="_blank" rel="noopener noreferrer">${esc(r.source)}</a>` : esc(r.source)}</td>
            <td>${esc(r.note)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

renderSummary();
renderTopology();
renderModelOverview();
renderHierarchy();
renderCalibration();
</script>
</body>
</html>
"""


def export_estimate_html(
    *,
    report: "TrainingReport",
    graph: "Graph",
    model: "ModelSpec",
    system: "SystemSpec",
    strategy: "Strategy",
    op_costs: dict[str, "OpCost"],
    output_path: str | Path,
) -> Path:
    """Write a self-contained HTML report.

    Layout:
    1. Summary: model/hardware/topology/strategy/perf/memory/bound.
    2. Logical cluster topology with rank-level PP/DP/TP/EP/CP labels.
    3. Model-level timing overview in Transformer-Explainer-like blocks.
    4. Expandable model -> layer/block -> op timing/formula/bound analysis.
    5. Calibration table with public/official reference points.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = _build_html_data(graph, model, system, strategy, report, op_costs)

    title = (
        f"ZRT Training Estimate — "
        f"{getattr(model, 'hidden', '-')}d_{len(getattr(model, 'layers', []))}L"
    )
    safe_title = html_lib.escape(title, quote=True)

    json_data = _json_parse_literal_for_script(data)

    html_text = (
        _HTML_TEMPLATE
        .replace("__TITLE__", safe_title)
        .replace("__DATA_JSON__", json_data)
    )

    output_path.write_text(html_text, encoding="utf-8")
    return output_path