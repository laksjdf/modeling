"""HTML exporter for spec-based training estimation (--estimate-config).

Produces a self-contained HTML page showing the hierarchical model structure
(model -> layer -> block -> op) as a computational graph with timing info.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zrt.training.ir.training_graph import Graph, Op
    from zrt.training.models.flops import OpCost
    from zrt.training.spec.model import ModelSpec
    from zrt.training.spec.strategy import Strategy
    from zrt.training.spec.system import SystemSpec
    from zrt.training.spec.report import TrainingReport


def _fmt_shape(shape: tuple[int, ...]) -> str:
    """Format shape tuple as string."""
    if not shape:
        return "()"
    return "(" + ", ".join(str(d) for d in shape) + ")"


def _fmt_tensor(t) -> str:
    """Format a Tensor as 'name: shape dtype'."""
    return f"{_fmt_shape(t.shape_local)} {t.dtype.value}"


def _tensor_list_info(tensors: list, label: str = "") -> str:
    """Format input/output tensor list."""
    if not tensors:
        return "-"
    parts = []
    for t in tensors:
        flag = "W" if t.is_param else "A"  # Weight vs Activation
        parts.append(f"[{flag}] {_fmt_tensor(t)}")
    return " | ".join(parts)


def _op_formula(op, cost):
    """Return (fwd_formula, bwd_formula, fwd_bytes_formula, bwd_bytes_formula) for an op."""
    m = op.meta
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
            fwd_str = f"2×m×n×k×{mult} = 2×{mm}×{nn}×{kk}×{mult} = {_fmt_e(fwd_val)}"
        else:
            fwd_str = f"2×m×n×k = 2×{mm}×{nn}×{kk} = {_fmt_e(fwd_val)}"
        bwd_str = f"dx+dw = 2×fwd = {_fmt_e(df + wf)}"
        bytes_str = f"(m×k+k×n+m×n)×bpe = ({mm}×{kk}+{kk}×{nn}+{mm}×{nn})×{bpe} = {_fmt_e(bytes_val)}"
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
            fwd_str = f"2×b×s×(topk+swa)×h×d = 2×{b}×{s}×{eff}×{h}×{d} = {_fmt_e(ff)}"
        elif cr > 0:
            c_len = max(1, s // cr)
            eff = c_len + swa
            fwd_str = f"2×b×s×(s/r+swa)×h×d = 2×{b}×{s}×{eff}×{h}×{d} = {_fmt_e(ff)}"
        elif swa > 0:
            fwd_str = f"2×b×s×swa×h×d = 2×{b}×{s}×{swa}×{h}×{d} = {_fmt_e(ff)}"
        else:
            fwd_str = f"2×b×s²×h×d = 2×{b}×{s}²×{h}×{d} = {_fmt_e(ff)}"
        bwd_str = f"2.5×fwd (FlashAttn internal recompute) = {_fmt_e(df)}"
        kv_len = (topk + swa) if topk > 0 else (max(1, s // cr) + swa if cr > 0 else (swa if swa > 0 else s))
        bytes_str = f"(2×b×h×s×d + 2×b×h×kv_len×d)×bpe = (2×{b}×{h}×{s}×{d} + 2×{b}×{h}×{kv_len}×{d})×bpe = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"(3×b×h×s×d + 4×b×h×kv_len×d)×bpe = {_fmt_e(cost.dx_bytes)}"
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
        fwd_str = f"2×N (sin/cos mul) = 2×{n} = {_fmt_e(ff)}"
        bwd_str = f"2.5×fwd = {_fmt_e(df)}"
        bytes_str = f"tensor I/O = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"dx_bytes = {_fmt_e(cost.dx_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "swiglu":
        n = sum(t.num_elements() for t in op.outputs) if op.outputs else 0
        fwd_str = f"5×N (sig+mul) = 5×{n} = {_fmt_e(ff)}"
        bwd_str = f"2.5×fwd = {_fmt_e(df)}"
        bytes_str = f"tensor I/O = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"dx_bytes = {_fmt_e(cost.dx_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "add":
        n = sum(t.num_elements() for t in op.outputs) if op.outputs else 0
        fwd_str = f"1×N (add) = {n} = {_fmt_e(ff)}"
        bwd_str = f"2.5×fwd = {_fmt_e(df)}"
        bytes_str = f"tensor I/O = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"dx_bytes = {_fmt_e(cost.dx_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "softmax":
        n = sum(t.num_elements() for t in op.outputs) if op.outputs else 0
        fwd_str = f"4×N (max/sub/exp/div) = 4×{n} = {_fmt_e(ff)}"
        bwd_str = f"2.5×fwd = {_fmt_e(df)}"
        bytes_str = f"tensor I/O = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"dx_bytes = {_fmt_e(cost.dx_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "indexer_topk":
        s = m.get("s", 0)
        kv = m.get("kv_len", s)
        ih = m.get("ih_local", m.get("ih", 0))
        id_ = m.get("id", 0)
        fwd_str = f"2×s×kv_len×ih×id_ = 2×{s}×{kv}×{ih}×{id_} = {_fmt_e(ff)}"
        bwd_str = f"2×fwd (idx_q grad only) = {_fmt_e(df)}"
        bytes_str = f"fwd_bytes = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"dx_bytes = - (grad via matmul bwd)"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "compressor_pool":
        s = m.get("s", 0)
        mm_ = m.get("m", 4)
        co = m.get("coff", 1)
        dd = m.get("d_local", m.get("d", 0))
        fwd_str = f"4×(s/m)×coff×m×d = 4×({s}//{mm_})×{co}×{mm_}×{dd} = {_fmt_e(ff)}"
        bwd_str = f"= fwd = {_fmt_e(df)}"
        bytes_str = f"bytes = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"dx_bytes = {_fmt_e(cost.dx_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "embed":
        s = m.get("m", 0)
        h = m.get("n", 0)
        fwd_str = f"0 (gather, no FLOPs)"
        bwd_str = f"0 (scatter, no FLOPs)"
        bytes_str = f"s×h×bpe = {s}×{h}×bpe = {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"same as fwd = {_fmt_e(cost.dx_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind in ("mhc_pre", "mhc_post", "mhc_head"):
        fwd_str = f"fwd = {_fmt_e(ff)}"
        bwd_str = f"2.5×fwd = {_fmt_e(df + wf)}"
        bytes_str = f"= {_fmt_e(cost.fwd_bytes)}"
        bwd_bytes_str = f"= {_fmt_e(cost.dx_bytes + cost.dw_bytes)}"
        return fwd_str, bwd_str, bytes_str, bwd_bytes_str

    if op.kind == "hash_route":
        return "negligible", "negligible", "negligible", "negligible"

    # Fallback
    return f"fwd = {_fmt_e(ff)}", f"bwd = {_fmt_e(df + wf)}", \
           f"bytes = {_fmt_e(cost.fwd_bytes)}", f"dx_bytes = {_fmt_e(cost.dx_bytes + cost.dw_bytes)}"


def _bpe_from_op(op) -> int:
    """Bytes per element from op tensors."""
    if op.inputs:
        return op.inputs[0].dtype.bytes
    if op.outputs:
        return op.outputs[0].dtype.bytes
    return 2


def _fmt_e(v: float) -> str:
    """Format a number in scientific notation."""
    if v <= 0:
        return "-"
    if v >= 1e15:
        return f"{v/1e15:.2f}P"
    if v >= 1e12:
        return f"{v/1e12:.2f}T"
    if v >= 1e9:
        return f"{v/1e9:.2f}G"
    if v >= 1e6:
        return f"{v/1e6:.2f}M"
    if v >= 1e3:
        return f"{v/1e3:.2f}K"
    return f"{v:.0f}"


def _fwd_flops(cost) -> float:
    return cost.fwd_cube_flops + cost.fwd_vector_flops

def _dx_flops(cost) -> float:
    return cost.dx_cube_flops + cost.dx_vector_flops

def _dw_flops(cost) -> float:
    return cost.dw_cube_flops + cost.dw_vector_flops

def _op_detail(op, cost):
    """Return a dict with full op info for Excel/HTML."""
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


def _classify_ops_in_layer(ops: list[Op], layer_kind: str) -> list[dict]:
    """Split a layer's ops into logical blocks using explicit name markers."""
    if layer_kind == "moe":
        boundaries = [
            ("Attention + Indexer", "ln1", "residual1"),
            ("Router + Shared Expert", "ln2", "shared_down_proj"),
            ("Routed Expert", "routed_expert_ffn", "residual2"),
        ]
    elif layer_kind == "mtp":
        boundaries = [
            ("MTP Embed", None, "ln1"),
            ("Attention + Indexer", "ln1", "residual1"),
            ("Router + Shared Expert", "ln2", "residual2"),
        ]
    else:
        boundaries = [
            ("Attention", "ln1", "residual1"),
            ("FFN", "ln2", "residual2"),
        ]

    blocks = []
    for name, start_marker, end_marker in boundaries:
        block_ops = []
        capturing = start_marker is None
        for op in ops:
            if start_marker and start_marker in op.name:
                capturing = True
            if capturing:
                block_ops.append(op)
                if end_marker and end_marker in op.name:
                    # Remove end marker op from this block if it's the start of next
                    if start_marker is None or (start_marker and start_marker in op.name):
                        block_ops.pop()
                    break
        if block_ops:
            blocks.append({"name": name, "ops": block_ops})

    return blocks


def _op_to_dict(op: Op, cost: OpCost, system) -> dict:
    """Convert an op + its cost to a dict for the HTML template."""
    from zrt.training.compose.stage import _cost_phase_time, has_heterogeneous_compute
    gpu_name = system.gpu.name
    overlap = system.gpu.overlap_ratio.get(op.kind, 0.0) if has_heterogeneous_compute(system) else 0.0
    fwd_t = _cost_phase_time(cost, "fwd", system, gpu_name, overlap)
    dx_t = _cost_phase_time(cost, "dx", system, gpu_name, overlap)
    dw_t = _cost_phase_time(cost, "dw", system, gpu_name, overlap)

    detail = _op_detail(op, cost)

    return {
        "name": op.name,
        "kind": op.kind,
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
    }


def _build_tree_data(graph, model, system, op_costs):
    """Build hierarchical tree: model -> layer -> block -> op."""
    layers = model.layers
    tree = {
        "model_name": f"{len(layers)} layers, hidden={model.hidden}",
        "total_ops": len(graph.ops),
        "layers": [],
        "global_ops": [],
    }

    # Per-layer data
    for lid in range(len(layers)):
        lk = layers[lid].value
        layer_ops = graph.ops_for_layer(lid)
        blocks = _classify_ops_in_layer(layer_ops, lk)

        layer_data = {
            "id": lid,
            "kind": lk,
            "op_count": len(layer_ops),
            "blocks": [],
        }

        for blk in blocks:
            blk_data = {
                "name": blk["name"],
                "ops": [],
                "total_ms": 0.0,
            }
            for op in blk["ops"]:
                cost = op_costs.get(op.name)
                if cost is None:
                    from zrt.training.models.flops import op_cost
                    cost = op_cost(op, model)
                op_dict = _op_to_dict(op, cost, system)
                blk_data["ops"].append(op_dict)
                blk_data["total_ms"] += op_dict["total_ms"]
            layer_data["blocks"].append(blk_data)

        tree["layers"].append(layer_data)

    # Global ops
    for op in graph.ops:
        if op.layer_id < 0:
            cost = op_costs.get(op.name)
            if cost is None:
                from zrt.training.models.flops import op_cost
                cost = op_cost(op, model)
            tree["global_ops"].append(_op_to_dict(op, cost, system))

    return tree


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ZRT — {title}</title>
<style>
  :root {{
    --bg: #0d0d0d;
    --surface: #141414;
    --surface-hi: #1c1c1c;
    --border: #262626;
    --border-hi: #363636;
    --text: #ddd8cc;
    --text-dim: #696560;
    --text-muted: #3e3c3a;
    --amber: #d48c0c;
    --amber-bright: #f0a828;
    --amber-dim: #7a5008;
    --teal: #00b890;
    --coral: #e04848;
    --dense-tag-bg: #112a1c; --dense-tag-fg: #4ab870; --dense-tag-bd: #1a4428;
    --moe-tag-bg: #280e38;   --moe-tag-fg: #9060d0; --moe-tag-bd: #3c1450;
    --mtp-tag-bg: #0a2038;   --mtp-tag-fg: #3888d0; --mtp-tag-bd: #0e2e52;
    --attn-stripe: #131820;
    --ffn-stripe: #131a13;
    --router-stripe: #1c1a10;
    --r: 3px;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    background: var(--bg);
    background-image: radial-gradient(var(--border) 1px, transparent 1px);
    background-size: 22px 22px;
    color: var(--text);
    font-family: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', 'SF Mono', ui-monospace, monospace;
    font-size: 12px;
    line-height: 1.6;
  }}

  /* ── HEADER ── */
  .hdr {{
    position: sticky; top: 0; z-index: 100;
    background: rgba(13,13,13,0.94);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--border);
    padding: 12px 26px;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  }}
  .hdr-logo {{
    width: 26px; height: 26px;
    border: 1.5px solid var(--amber); border-radius: 3px;
    display: flex; align-items: center; justify-content: center;
    color: var(--amber); font-size: 10px; font-weight: 700; letter-spacing: 0.5px;
    flex-shrink: 0;
  }}
  .hdr-title {{ font-size: 12px; color: var(--text); letter-spacing: 0.3px; }}
  .hdr-title em {{ color: var(--amber); font-style: normal; }}
  .hdr-meta {{ display: flex; gap: 18px; margin-left: auto; flex-wrap: wrap; }}
  .hdr-stat {{ color: var(--text-dim); font-size: 10px; }}
  .hdr-stat b {{ color: var(--amber-bright); font-weight: 600; }}

  /* ── TOOLBAR ── */
  .toolbar {{
    position: sticky; top: 51px; z-index: 99;
    background: rgba(13,13,13,0.90);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--border);
    padding: 7px 26px;
    display: flex; gap: 4px; align-items: center;
  }}
  .tb-btn {{
    background: transparent;
    color: var(--text-dim);
    border: 1px solid var(--border);
    padding: 3px 11px; border-radius: var(--r);
    cursor: pointer; font: inherit; font-size: 11px; letter-spacing: 0.2px;
    transition: color 0.1s, border-color 0.1s, background 0.1s;
  }}
  .tb-btn:hover, .tb-btn.on {{
    color: var(--amber); border-color: var(--amber-dim);
    background: rgba(212,140,12,0.07);
  }}
  .tb-sep {{ width: 1px; height: 14px; background: var(--border); margin: 0 5px; }}
  .tb-search {{
    background: var(--surface); color: var(--text);
    border: 1px solid var(--border); padding: 3px 9px;
    border-radius: var(--r); font: inherit; font-size: 11px; width: 190px;
    transition: border-color 0.12s;
  }}
  .tb-search:focus {{ outline: none; border-color: var(--amber-dim); }}
  .tb-search::placeholder {{ color: var(--text-muted); }}
  .tb-hint {{ color: var(--text-muted); font-size: 10px; margin-left: 4px; }}

  /* ── CONTAINER ── */
  .wrap {{ padding: 18px 26px; max-width: 1440px; margin: 0 auto; }}

  /* ── METRICS ── */
  .metrics {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 1px; background: var(--border);
    border: 1px solid var(--border); border-radius: var(--r);
    overflow: hidden; margin-bottom: 18px;
  }}
  .metric {{
    background: var(--surface); padding: 12px 14px; position: relative;
  }}
  .metric::before {{
    content: ''; position: absolute;
    top: 0; left: 0; width: 2px; height: 100%;
    background: var(--amber-dim);
  }}
  .metric-lbl {{ color: var(--text-dim); font-size: 9px; letter-spacing: 0.9px; text-transform: uppercase; margin-bottom: 5px; }}
  .metric-val {{ font-size: 20px; font-weight: 700; color: var(--amber-bright); line-height: 1; letter-spacing: -0.5px; }}

  /* ── GLOBAL OPS ── */
  .global-bar {{
    display: flex; gap: 6px; margin-bottom: 14px;
    padding: 7px 12px; background: var(--surface);
    border: 1px solid var(--border); border-radius: var(--r); flex-wrap: wrap;
    align-items: center;
  }}
  .global-lbl {{ color: var(--text-muted); font-size: 9px; letter-spacing: 0.6px; text-transform: uppercase; margin-right: 4px; }}
  .op-chip {{
    background: var(--bg); border: 1px solid var(--border);
    padding: 2px 7px; border-radius: 2px;
    font-size: 10px; color: var(--text-dim); transition: border-color 0.1s;
  }}
  .op-chip:hover {{ border-color: var(--amber-dim); color: var(--text); }}
  .chip-time {{ color: var(--amber); margin-left: 4px; }}

  /* ── LAYER ── */
  .layer {{ margin-bottom: 3px; border: 1px solid var(--border); border-radius: var(--r); overflow: hidden; transition: border-color 0.1s; }}
  .layer:hover {{ border-color: var(--border-hi); }}

  .layer-hdr {{
    display: grid;
    grid-template-columns: 14px 48px 52px 1fr 72px 100px;
    align-items: center; gap: 10px;
    padding: 6px 14px; cursor: pointer; user-select: none;
    background: var(--surface); transition: background 0.1s;
  }}
  .layer-hdr:hover {{ background: var(--surface-hi); }}
  .layer-arrow {{ color: var(--text-muted); font-size: 8px; transition: transform 0.18s cubic-bezier(.4,0,.2,1); }}
  .layer-hdr.open .layer-arrow {{ transform: rotate(90deg); color: var(--amber); }}
  .layer-idx {{ color: var(--text-dim); font-size: 10px; }}
  .layer-kind {{
    font-size: 9px; padding: 2px 5px; border-radius: 2px;
    font-weight: 700; letter-spacing: 0.7px; text-transform: uppercase; text-align: center;
  }}
  .layer-kind.dense {{ background: var(--dense-tag-bg); color: var(--dense-tag-fg); border: 1px solid var(--dense-tag-bd); }}
  .layer-kind.moe   {{ background: var(--moe-tag-bg);   color: var(--moe-tag-fg);   border: 1px solid var(--moe-tag-bd); }}
  .layer-kind.mtp   {{ background: var(--mtp-tag-bg);   color: var(--mtp-tag-fg);   border: 1px solid var(--mtp-tag-bd); }}
  .layer-timebar {{ height: 2px; background: var(--border); border-radius: 1px; overflow: hidden; }}
  .layer-timebar-fill {{ height: 100%; background: var(--amber); opacity: 0.5; border-radius: 1px; }}
  .layer-opcount {{ color: var(--text-muted); font-size: 10px; text-align: right; white-space: nowrap; }}
  .layer-ms {{ color: var(--amber); font-size: 10px; text-align: right; white-space: nowrap; }}
  .layer-ms span {{ color: var(--text-muted); }}

  .layer-body {{ max-height: 0; overflow: hidden; transition: max-height 0.22s cubic-bezier(.4,0,.2,1); background: var(--bg); }}
  .layer-body.open {{ max-height: 9999px; transition: max-height 0.45s cubic-bezier(.4,0,.2,1); }}

  /* ── BLOCK ── */
  .block {{ margin: 3px 8px; border-radius: 2px; overflow: hidden; border-left: 2px solid var(--border); }}
  .block.attn    {{ border-left-color: #204060; }}
  .block.ffn     {{ border-left-color: #1e4020; }}
  .block.router  {{ border-left-color: #504018; }}
  .block.indexer {{ border-left-color: #402e10; }}

  .block-hdr {{
    display: flex; align-items: center; gap: 8px;
    padding: 4px 10px; cursor: pointer; user-select: none;
    background: var(--surface); font-size: 11px; transition: background 0.1s;
  }}
  .block-hdr:hover {{ background: var(--surface-hi); }}
  .block.attn   .block-hdr {{ background: var(--attn-stripe); }}
  .block.ffn    .block-hdr {{ background: var(--ffn-stripe); }}
  .block.router .block-hdr {{ background: var(--router-stripe); }}
  .block-arrow {{ color: var(--text-muted); font-size: 8px; transition: transform 0.14s; }}
  .block-hdr.open .block-arrow {{ transform: rotate(90deg); }}
  .block-name {{ color: var(--text-dim); }}
  .block-time {{ margin-left: auto; color: var(--amber); font-size: 10px; }}

  .block-body {{ max-height: 0; overflow: hidden; transition: max-height 0.18s cubic-bezier(.4,0,.2,1); }}
  .block-body.open {{ max-height: 5000px; transition: max-height 0.35s cubic-bezier(.4,0,.2,1); padding: 3px 4px 4px; }}

  /* ── OP NODE ── */
  .op-node {{
    display: grid;
    grid-template-columns: 88px 158px 86px 1fr 58px 34px;
    align-items: center; gap: 8px;
    padding: 3px 8px; margin: 1px 0;
    background: var(--surface); border-radius: 2px;
    transition: background 0.08s;
  }}
  .op-node:hover {{ background: var(--surface-hi); }}
  .op-node.hidden {{ display: none; }}

  .kind-tag {{
    font-size: 9px; padding: 1px 5px; border-radius: 2px;
    text-align: center; font-weight: 600; letter-spacing: 0.2px;
    border: 1px solid transparent;
  }}
  .kind-tag.matmul         {{ background:#0c2040; color:#4898e0; border-color:#1a3660; }}
  .kind-tag.lm_head        {{ background:#0c2040; color:#4898e0; border-color:#1a3660; }}
  .kind-tag.attn_core      {{ background:#280c30; color:#c060e8; border-color:#481450; }}
  .kind-tag.sparse_attn    {{ background:#280c30; color:#c060e8; border-color:#481450; }}
  .kind-tag.hca_attn       {{ background:#280c30; color:#c060e8; border-color:#481450; }}
  .kind-tag.swa_attn       {{ background:#280c30; color:#c060e8; border-color:#481450; }}
  .kind-tag.rmsnorm        {{ background:#1c1c1c; color:#686460; border-color:#2c2c2c; }}
  .kind-tag.ln             {{ background:#1c1c1c; color:#686460; border-color:#2c2c2c; }}
  .kind-tag.swiglu         {{ background:#0c2416; color:#40c870; border-color:#184428; }}
  .kind-tag.add            {{ background:#161616; color:#505050; border-color:#242424; }}
  .kind-tag.rope           {{ background:#180e2c; color:#7060c8; border-color:#2c1e48; }}
  .kind-tag.indexer_topk   {{ background:#281600; color:#c08020; border-color:#442c00; }}
  .kind-tag.compressor_pool{{ background:#281600; color:#c08020; border-color:#442c00; }}
  .kind-tag.softmax        {{ background:#181800; color:#a0a020; border-color:#2c2c08; }}
  .kind-tag.embed          {{ background:#001620; color:#00a8c0; border-color:#002c40; }}
  .kind-tag.hash_route     {{ background:#161616; color:#383838; border-color:#242424; }}
  .kind-tag.mhc_pre        {{ background:#1c0e1e; color:#8050a8; border-color:#341c3c; }}
  .kind-tag.mhc_post       {{ background:#1c0e1e; color:#8050a8; border-color:#341c3c; }}
  .kind-tag.mhc_head       {{ background:#1c0e1e; color:#8050a8; border-color:#341c3c; }}

  .op-name {{ color: var(--text); font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .op-flops {{ color: var(--text-dim); font-size: 10px; text-align: right; }}

  .op-segbar {{ height: 5px; background: var(--border); border-radius: 1px; overflow: hidden; display: flex; }}
  .op-segbar .seg {{ height: 100%; }}
  .op-segbar .seg.fwd {{ background: var(--teal); }}
  .op-segbar .seg.dx  {{ background: var(--amber); }}
  .op-segbar .seg.dw  {{ background: var(--coral); }}

  .op-ms  {{ color: var(--amber); font-size: 10px; text-align: right; }}
  .op-pct {{ color: var(--text-muted); font-size: 9px; text-align: right; }}

  /* ── CHART VIEW ── */
  .chart-section {{ margin-bottom: 22px; }}
  .chart-title {{
    font-size: 9px; letter-spacing: 0.9px; text-transform: uppercase;
    color: var(--text-dim); margin-bottom: 8px;
    padding-bottom: 5px; border-bottom: 1px solid var(--border);
  }}
  .chart-row {{
    display: grid; grid-template-columns: 155px 1fr 80px;
    align-items: center; gap: 10px; margin: 2px 0; font-size: 10px;
  }}
  .chart-lbl {{ color: var(--text-dim); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .chart-wrap {{ background: var(--surface); height: 9px; border-radius: 1px; overflow: hidden; }}
  .chart-bar {{ height: 100%; border-radius: 1px; }}
  .chart-bar.compute {{ background: var(--teal); opacity: 0.75; }}
  .chart-bar.memory  {{ background: var(--amber); opacity: 0.75; }}
  .chart-val {{ color: var(--amber); text-align: right; }}

  /* ── TOOLTIP ── */
  #tooltip {{
    display: none; position: fixed;
    background: #111; border: 1px solid var(--border-hi);
    border-radius: var(--r); padding: 10px 14px;
    font-size: 11px; z-index: 1000; max-width: 360px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.85), 0 0 0 1px rgba(212,140,12,0.08);
    pointer-events: none;
  }}
  #tooltip.show {{ display: block; animation: tip-in 0.1s ease; }}
  @keyframes tip-in {{ from {{ opacity:0; transform:translateY(3px); }} to {{ opacity:1; transform:none; }} }}
  .tip-name {{ color: var(--amber-bright); font-weight: 600; margin-bottom: 7px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }}
  .tip-row {{ display: grid; grid-template-columns: 86px 1fr; gap: 6px; margin: 2px 0; align-items: start; }}
  .tip-k {{ color: var(--text-dim); }}
  .tip-v {{ color: var(--text); word-break: break-all; }}
  .tip-timing {{
    display: grid; grid-template-columns: repeat(3,1fr); gap: 6px;
    margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border);
  }}
  .tip-tc {{ text-align: center; }}
  .tip-tc .tv {{ font-size: 13px; font-weight: 600; line-height: 1; }}
  .tip-tc .tl {{ font-size: 9px; color: var(--text-dim); letter-spacing: 0.5px; text-transform: uppercase; margin-top: 3px; }}
  .tip-tc.fwd .tv {{ color: var(--teal); }}
  .tip-tc.dx  .tv {{ color: var(--amber); }}
  .tip-tc.dw  .tv {{ color: var(--coral); }}
</style>
</head>
<body>

<div class="hdr">
  <div class="hdr-logo">ZRT</div>
  <div class="hdr-title">Training Estimate <em>/ {title}</em></div>
  <div class="hdr-meta" id="hdr-meta"></div>
</div>

<div class="toolbar">
  <button class="tb-btn" onclick="expandAll()">expand all</button>
  <button class="tb-btn" onclick="collapseAll()">collapse all</button>
  <div class="tb-sep"></div>
  <button class="tb-btn on" id="btn-tree" onclick="showTree()">tree</button>
  <button class="tb-btn" id="btn-chart" onclick="showChart()">chart</button>
  <div class="tb-sep"></div>
  <input type="text" class="tb-search" id="search" placeholder="/ search ops…" oninput="filterOps()">
  <span class="tb-hint">esc clears</span>
</div>

<div class="wrap" id="main"></div>
<div id="tooltip"></div>

<script>
const DATA = {json_data};
const STEP_MS = {step_time_ms};

function fmtF(v) {{
  if (v <= 0) return '—';
  if (v >= 1e15) return (v/1e15).toFixed(2)+'P';
  if (v >= 1e12) return (v/1e12).toFixed(2)+'T';
  if (v >= 1e9)  return (v/1e9).toFixed(2)+'G';
  if (v >= 1e6)  return (v/1e6).toFixed(2)+'M';
  return v.toFixed(0);
}}

function fmtMs(v) {{
  if (v <= 0) return '—';
  if (v < 0.01) return (v*1000).toFixed(1)+'µs';
  if (v < 1)    return v.toFixed(3)+'ms';
  if (v < 1000) return v.toFixed(2)+'ms';
  return (v/1000).toFixed(3)+'s';
}}

function blkCls(name) {{
  const n = name.toLowerCase();
  if (n.includes('router')||n.includes('routed')||n.includes('shared')) return 'router';
  if (n.includes('indexer')) return 'indexer';
  if (n.includes('attn')||n.includes('attention')) return 'attn';
  return 'ffn';
}}

function kindCls(k) {{ return k.replace(/[^a-z0-9_]/gi,'_'); }}

function buildMetrics() {{
  const nD = DATA.layers.filter(l=>l.kind==='dense').length;
  const nM = DATA.layers.filter(l=>l.kind==='moe').length;
  document.getElementById('hdr-meta').innerHTML =
    `<div class="hdr-stat">step <b>${{fmtMs(STEP_MS)}}</b></div>` +
    `<div class="hdr-stat">layers <b>${{DATA.layers.length}}</b></div>` +
    `<div class="hdr-stat">ops <b>${{DATA.total_ops}}</b></div>`;
  return `<div class="metrics">
    <div class="metric"><div class="metric-lbl">Step Time</div><div class="metric-val">${{fmtMs(STEP_MS)}}</div></div>
    <div class="metric"><div class="metric-lbl">Layers</div><div class="metric-val">${{DATA.layers.length}}</div></div>
    <div class="metric"><div class="metric-lbl">Dense</div><div class="metric-val">${{nD}}</div></div>
    <div class="metric"><div class="metric-lbl">MoE</div><div class="metric-val">${{nM}}</div></div>
    <div class="metric"><div class="metric-lbl">Total Ops</div><div class="metric-val">${{DATA.total_ops}}</div></div>
  </div>`;
}}

function buildGlobal() {{
  if (!DATA.global_ops.length) return '';
  let h = '<div class="global-bar"><span class="global-lbl">global</span>';
  for (const op of DATA.global_ops) {{
    const pct = STEP_MS > 0 ? (op.total_ms/STEP_MS*100).toFixed(1) : '0.0';
    const td = JSON.stringify(op).replace(/"/g,'&quot;');
    h += `<div class="op-chip" onmouseenter="showTip(event,${{td}})" onmouseleave="hideTip()">${{op.name}}<span class="chip-time">${{fmtMs(op.total_ms)}} ${{pct}}%</span></div>`;
  }}
  return h+'</div>';
}}

function buildOp(op, blkTotal) {{
  const tot = op.fwd_ms + op.dx_ms + op.dw_ms;
  const fw = tot>0 ? (op.fwd_ms/tot*100).toFixed(1) : 0;
  const dxw= tot>0 ? (op.dx_ms/tot*100).toFixed(1) : 0;
  const dww= tot>0 ? (op.dw_ms/tot*100).toFixed(1) : 0;
  const pct= blkTotal>0 ? (op.total_ms/blkTotal*100).toFixed(0) : 0;
  const td = JSON.stringify(op).replace(/"/g,'&quot;');
  return `<div class="op-node" data-name="${{op.name.toLowerCase()}}"
    onmouseenter="showTip(event,${{td}})" onmouseleave="hideTip()">
    <span class="kind-tag ${{kindCls(op.kind)}}">${{op.kind}}</span>
    <span class="op-name" title="${{op.name}}">${{op.name}}</span>
    <span class="op-flops">${{fmtF(op.fwd_flops)}}</span>
    <div class="op-segbar">
      <div class="seg fwd" style="width:${{fw}}%"></div>
      <div class="seg dx"  style="width:${{dxw}}%"></div>
      <div class="seg dw"  style="width:${{dww}}%"></div>
    </div>
    <span class="op-ms">${{fmtMs(op.total_ms)}}</span>
    <span class="op-pct">${{pct}}%</span>
  </div>`;
}}

let modelMs = 0;

function buildLayer(layer) {{
  const lt = layer.blocks.reduce((s,b)=>s+b.total_ms,0);
  const pct = modelMs>0 ? (lt/modelMs*100).toFixed(1) : 0;
  const bw  = modelMs>0 ? (lt/modelMs*100).toFixed(1) : 0;
  let h = `<div class="layer" data-kind="${{layer.kind}}">
    <div class="layer-hdr" onclick="toggleLayer(this)">
      <span class="layer-arrow">▶</span>
      <span class="layer-idx">L${{layer.id}}</span>
      <span class="layer-kind ${{layer.kind}}">${{layer.kind}}</span>
      <div class="layer-timebar"><div class="layer-timebar-fill" style="width:${{bw}}%"></div></div>
      <span class="layer-opcount">${{layer.op_count}} ops</span>
      <span class="layer-ms">${{fmtMs(lt)}} <span>(${{pct}}%)</span></span>
    </div>
    <div class="layer-body">`;
  for (const blk of layer.blocks) {{
    const bc = blkCls(blk.name);
    const bp = lt>0 ? (blk.total_ms/lt*100).toFixed(1) : 0;
    h += `<div class="block ${{bc}}">
      <div class="block-hdr" onclick="toggleBlock(this)">
        <span class="block-arrow">▶</span>
        <span class="block-name">${{blk.name}}</span>
        <span class="block-time">${{fmtMs(blk.total_ms)}} (${{bp}}%)</span>
      </div>
      <div class="block-body">`;
    for (const op of blk.ops) h += buildOp(op, blk.total_ms);
    h += `</div></div>`;
  }}
  h += `</div></div>`;
  return h;
}}

function renderTree() {{
  document.getElementById('btn-tree').classList.add('on');
  document.getElementById('btn-chart').classList.remove('on');
  modelMs = DATA.layers.reduce((s,l)=>s+l.blocks.reduce((ss,b)=>ss+b.total_ms,0),0);
  let h = buildMetrics() + buildGlobal();
  for (const layer of DATA.layers) h += buildLayer(layer);
  document.getElementById('main').innerHTML = h;
}}

function renderChart() {{
  document.getElementById('btn-chart').classList.add('on');
  document.getElementById('btn-tree').classList.remove('on');
  const km = {{}};
  for (const layer of DATA.layers)
    for (const blk of layer.blocks)
      for (const op of blk.ops) {{
        if (!km[op.kind]) km[op.kind]={{kind:op.kind,total:0,count:0}};
        km[op.kind].total+=op.total_ms; km[op.kind].count++;
      }}
  for (const op of DATA.global_ops) {{
    if (!km[op.kind]) km[op.kind]={{kind:op.kind,total:0,count:0}};
    km[op.kind].total+=op.total_ms; km[op.kind].count++;
  }}
  const sk = Object.values(km).sort((a,b)=>b.total-a.total);
  const mk = sk.length>0 ? sk[0].total : 1;

  let maxL=0;
  for (const l of DATA.layers) {{ const lt=l.blocks.reduce((s,b)=>s+b.total_ms,0); if(lt>maxL)maxL=lt; }}

  let h = buildMetrics();
  h += `<div class="chart-section"><div class="chart-title">time by op kind</div>`;
  for (const k of sk) {{
    const w=(k.total/mk*100).toFixed(1);
    const pct=STEP_MS>0?(k.total/STEP_MS*100).toFixed(1):'—';
    h+=`<div class="chart-row"><span class="chart-lbl">${{k.kind}} <span style="color:var(--text-muted)">×${{k.count}}</span></span><div class="chart-wrap"><div class="chart-bar compute" style="width:${{w}}%"></div></div><span class="chart-val">${{fmtMs(k.total)}}</span></div>`;
  }}
  h+=`</div><div class="chart-section"><div class="chart-title">time by layer</div>`;
  for (const layer of DATA.layers) {{
    const lt=layer.blocks.reduce((s,b)=>s+b.total_ms,0);
    const w=maxL>0?(lt/maxL*100).toFixed(1):0;
    h+=`<div class="chart-row"><span class="chart-lbl">L${{layer.id}} <span style="color:var(--text-muted)">(${{layer.kind}})</span></span><div class="chart-wrap"><div class="chart-bar memory" style="width:${{w}}%"></div></div><span class="chart-val">${{fmtMs(lt)}}</span></div>`;
  }}
  h+=`</div>`;
  document.getElementById('main').innerHTML = h;
}}

function toggleLayer(el) {{ el.classList.toggle('open'); el.nextElementSibling.classList.toggle('open'); }}
function toggleBlock(el) {{ el.classList.toggle('open'); el.nextElementSibling.classList.toggle('open'); }}
function expandAll()  {{ document.querySelectorAll('.layer-hdr,.block-hdr').forEach(el=>{{ el.classList.add('open'); el.nextElementSibling.classList.add('open'); }}); }}
function collapseAll(){{ document.querySelectorAll('.layer-hdr,.block-hdr').forEach(el=>{{ el.classList.remove('open'); el.nextElementSibling.classList.remove('open'); }}); }}
function showTree()   {{ renderTree(); }}
function showChart()  {{ renderChart(); }}

function filterOps() {{
  const q=document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('.op-node').forEach(el=>el.classList.toggle('hidden',!!q&&!el.dataset.name.includes(q)));
}}

function showTip(e, op) {{
  const t=document.getElementById('tooltip');
  let fr='';
  if (op.fwd_formula&&op.fwd_formula!=='negligible') {{
    fr=`<div class="tip-row"><span class="tip-k">fwd FLOPs</span><span class="tip-v">${{op.fwd_formula}}</span></div>
    <div class="tip-row"><span class="tip-k">bwd FLOPs</span><span class="tip-v">${{op.bwd_formula}}</span></div>
    <div class="tip-row"><span class="tip-k">fwd bytes</span><span class="tip-v">${{op.fwd_bytes_formula}}</span></div>
    <div class="tip-row"><span class="tip-k">bwd bytes</span><span class="tip-v">${{op.bwd_bytes_formula}}</span></div>`;
  }}
  t.innerHTML=`<div class="tip-name">${{op.name}}</div>
    <div class="tip-row"><span class="tip-k">kind</span><span class="tip-v">${{op.kind}}</span></div>
    <div class="tip-row"><span class="tip-k">bound</span><span class="tip-v">${{op.bound}}</span></div>
    <div class="tip-row"><span class="tip-k">input</span><span class="tip-v">${{op.inputs}}</span></div>
    <div class="tip-row"><span class="tip-k">output</span><span class="tip-v">${{op.outputs}}</span></div>
    ${{fr}}
    <div class="tip-timing">
      <div class="tip-tc fwd"><div class="tv">${{fmtMs(op.fwd_ms)}}</div><div class="tl">fwd</div></div>
      <div class="tip-tc dx" ><div class="tv">${{fmtMs(op.dx_ms)}}</div><div class="tl">dx</div></div>
      <div class="tip-tc dw" ><div class="tv">${{fmtMs(op.dw_ms)}}</div><div class="tl">dw</div></div>
    </div>`;
  t.classList.add('show');
  t.style.left=Math.min(e.clientX+14,window.innerWidth-380)+'px';
  t.style.top=Math.min(e.clientY+14,window.innerHeight-220)+'px';
}}

function hideTip() {{ document.getElementById('tooltip').classList.remove('show'); }}

document.addEventListener('keydown',e=>{{
  if (e.key==='/'&&document.activeElement.tagName!=='INPUT') {{ e.preventDefault(); document.getElementById('search').focus(); }}
  if (e.key==='Escape') {{ document.getElementById('search').value=''; filterOps(); document.getElementById('search').blur(); }}
}});

renderTree();
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
    """Write the estimation report to a self-contained HTML file with hierarchical graph view."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tree_data = _build_tree_data(graph, model, system, op_costs)
    json_data = json.dumps(tree_data, ensure_ascii=False)

    html = _HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    html = html.replace("{title}", f"{model.hidden}d {len(model.layers)}L")
    html = html.replace("{json_data}", json_data)
    html = html.replace("{step_time_ms}", str(report.step_time_ms))

    output_path.write_text(html, encoding="utf-8")
    return output_path
