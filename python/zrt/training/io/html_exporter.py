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

    return {
        "name": op.name,
        "kind": op.kind,
        "fwd_flops": cost.fwd_flops,
        "dx_flops": cost.dx_flops,
        "dw_flops": cost.dw_flops,
        "fwd_bytes": cost.fwd_bytes,
        "dx_bytes": cost.dx_bytes,
        "dw_bytes": cost.dw_bytes,
        "bound": cost.bound,
        "fwd_ms": fwd_t * 1000,
        "dx_ms": dx_t * 1000,
        "dw_ms": dw_t * 1000,
        "total_ms": (fwd_t + dx_t + dw_t) * 1000,
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
<title>ZRT Estimate — {title}</title>
<style>
  :root {{
    --bg: #1a1a2e;
    --surface: #16213e;
    --surface2: #0f3460;
    --text: #e0e0e0;
    --text-dim: #8899aa;
    --accent: #e94560;
    --accent2: #533483;
    --green: #00b894;
    --orange: #fdcb6e;
    --border: #2a3a5e;
    --dense-color: #0984e3;
    --moe-color: #6c5ce7;
    --mtp-color: #00b894;
    --attn-bg: #1a2744;
    --ffn-bg: #1a2a1a;
    --indexer-bg: #2a1a3a;
    --router-bg: #2a2a1a;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    font-size: 13px;
    line-height: 1.5;
  }}

  .header {{
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
    position: sticky;
    top: 0;
    z-index: 100;
  }}
  .header h1 {{ font-size: 18px; color: #fff; }}
  .header .stats {{ display: flex; gap: 24px; margin-top: 8px; flex-wrap: wrap; }}
  .header .stat {{ color: var(--text-dim); }}
  .header .stat span {{ color: #fff; font-weight: bold; }}

  .toolbar {{
    padding: 12px 24px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
  }}
  .toolbar button {{
    background: var(--surface2);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 6px 14px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
  }}
  .toolbar button:hover {{ background: var(--accent); border-color: var(--accent); }}
  .toolbar .filter {{
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 12px;
  }}

  .container {{ padding: 16px 24px; }}

  /* Global ops bar */
  .global-ops {{
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }}
  .global-ops .op-chip {{
    background: var(--surface2);
    border: 1px solid var(--border);
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 11px;
  }}
  .global-ops .op-chip .time {{ color: var(--orange); }}

  /* Layer */
  .layer {{
    margin-bottom: 8px;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }}
  .layer-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 14px;
    cursor: pointer;
    user-select: none;
    background: var(--surface);
    transition: background 0.15s;
  }}
  .layer-header:hover {{ background: var(--surface2); }}
  .layer-header .kind {{
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 3px;
    font-weight: bold;
    text-transform: uppercase;
  }}
  .layer-header .kind.dense {{ background: var(--dense-color); color: #fff; }}
  .layer-header .kind.moe {{ background: var(--moe-color); color: #fff; }}
  .layer-header .kind.mtp {{ background: var(--mtp-color); color: #1a1a2e; }}
  .layer-header .idx {{ color: var(--text-dim); min-width: 50px; }}
  .layer-header .ops-count {{ color: var(--text-dim); font-size: 11px; }}
  .layer-header .total-ms {{ color: var(--orange); font-size: 11px; }}
  .layer-header .arrow {{ color: var(--text-dim); transition: transform 0.2s; }}
  .layer-header.open .arrow {{ transform: rotate(90deg); }}

  .layer-body {{ display: none; background: var(--bg); }}
  .layer-body.open {{ display: block; }}

  /* Block */
  .block {{
    margin: 4px 8px;
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }}
  .block-header {{
    padding: 6px 12px;
    cursor: pointer;
    user-select: none;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: background 0.15s;
  }}
  .block-header:hover {{ filter: brightness(1.2); }}
  .block-header.attn {{ background: var(--attn-bg); }}
  .block-header.ffn {{ background: var(--ffn-bg); }}
  .block-header.indexer {{ background: var(--indexer-bg); }}
  .block-header.router {{ background: var(--router-bg); }}
  .block-header .arrow {{ color: var(--text-dim); font-size: 10px; transition: transform 0.2s; }}
  .block-header.open .arrow {{ transform: rotate(90deg); }}
  .block-header .name {{ font-size: 12px; color: var(--text-dim); }}
  .block-header .time {{ color: var(--orange); font-size: 11px; margin-left: auto; }}

  .block-body {{ display: none; padding: 4px; }}
  .block-body.open {{ display: block; }}

  /* Op node */
  .op-node {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 10px;
    margin: 2px 0;
    background: var(--surface);
    border-radius: 3px;
    font-size: 12px;
    position: relative;
  }}
  .op-node::before {{
    content: '';
    position: absolute;
    left: 16px;
    top: -4px;
    bottom: -4px;
    width: 1px;
    background: var(--border);
  }}
  .op-node:first-child::before {{ display: none; }}
  .op-node .kind-tag {{
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 2px;
    min-width: 80px;
    text-align: center;
  }}
  .op-node .kind-tag.matmul {{ background: #2d5a8e; color: #fff; }}
  .op-node .kind-tag.rmsnorm {{ background: #4a4a6a; color: #ddd; }}
  .op-node .kind-tag.attn_core {{ background: #8e2d5a; color: #fff; }}
  .op-node .kind-tag.sparse_attn {{ background: #8e2d5a; color: #fff; }}
  .op-node .kind-tag.swiglu {{ background: #4a8e2d; color: #fff; }}
  .op-node .kind-tag.add {{ background: #4a6a4a; color: #ddd; }}
  .op-node .kind-tag.rope {{ background: #6a4a8e; color: #fff; }}
  .op-node .kind-tag.indexer_topk {{ background: #8e5a2d; color: #fff; }}
  .op-node .kind-tag.compressor_pool {{ background: #8e5a2d; color: #ddd; }}
  .op-node .kind-tag.softmax {{ background: #6a6a2a; color: #fff; }}
  .op-node .kind-tag.embed {{ background: #2d8e5a; color: #fff; }}
  .op-node .kind-tag.lm_head {{ background: #2d8e5a; color: #fff; }}
  .op-node .kind-tag.hash_route {{ background: #3a3a3a; color: #888; }}
  .op-node .name {{ color: var(--text); min-width: 180px; }}
  .op-node .flops {{ color: var(--text-dim); min-width: 100px; font-size: 11px; }}
  .op-node .time-bar {{
    flex: 1;
    height: 8px;
    background: var(--bg);
    border-radius: 2px;
    overflow: hidden;
    min-width: 80px;
  }}
  .op-node .time-bar .fill {{
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s;
  }}
  .op-node .time-bar .fill.compute {{ background: var(--accent); }}
  .op-node .time-bar .fill.memory {{ background: var(--accent2); }}
  .op-node .ms {{ color: var(--orange); min-width: 60px; text-align: right; font-size: 11px; }}
  .op-node .pct {{ color: var(--text-dim); min-width: 40px; text-align: right; font-size: 10px; }}

  /* Bar chart view */
  .bar-chart {{
    margin: 8px 0;
    padding: 0 12px;
  }}
  .bar-chart .bar-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 2px 0;
    font-size: 11px;
  }}
  .bar-chart .bar-label {{ min-width: 120px; color: var(--text-dim); }}
  .bar-chart .bar {{
    height: 14px;
    border-radius: 2px;
    min-width: 2px;
    transition: width 0.3s;
  }}
  .bar-chart .bar.compute {{ background: var(--accent); }}
  .bar-chart .bar.memory {{ background: var(--accent2); }}
  .bar-chart .bar-val {{ min-width: 60px; color: var(--orange); }}

  /* Summary cards */
  .summary-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
  }}
  .card .label {{ color: var(--text-dim); font-size: 11px; }}
  .card .value {{ color: #fff; font-size: 20px; font-weight: bold; margin-top: 4px; }}
  .card .unit {{ color: var(--text-dim); font-size: 12px; }}

  /* Tooltip */
  .tooltip {{
    display: none;
    position: fixed;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
    font-size: 11px;
    z-index: 200;
    max-width: 320px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
  }}
  .tooltip.show {{ display: block; }}
  .tooltip .row {{ display: flex; justify-content: space-between; gap: 16px; margin: 2px 0; }}
  .tooltip .row .k {{ color: var(--text-dim); }}
  .tooltip .row .v {{ color: #fff; }}

  .search-hint {{ color: var(--text-dim); font-size: 11px; }}
</style>
</head>
<body>

<div class="header">
  <h1>ZRT Training Estimate</h1>
  <div class="stats" id="header-stats"></div>
</div>

<div class="toolbar">
  <button onclick="expandAll()">Expand All</button>
  <button onclick="collapseAll()">Collapse All</button>
  <button onclick="showChart()">Bar Chart</button>
  <button onclick="showTree()">Tree View</button>
  <input type="text" class="filter" id="search" placeholder="Search op name..." oninput="filterOps()">
  <span class="search-hint">Press / to focus search</span>
</div>

<div class="container" id="main"></div>

<div class="tooltip" id="tooltip"></div>

<script>
const DATA = {json_data};
const STEP_TIME_MS = {step_time_ms};

function fmtFlops(v) {{
  if (v <= 0) return '-';
  if (v >= 1e15) return (v/1e15).toFixed(2) + ' PF';
  if (v >= 1e12) return (v/1e12).toFixed(2) + ' TF';
  if (v >= 1e9) return (v/1e9).toFixed(2) + ' GF';
  if (v >= 1e6) return (v/1e6).toFixed(2) + ' MF';
  return v.toFixed(0);
}}

function fmtMs(v) {{
  if (v < 0.01) return (v * 1000).toFixed(1) + ' us';
  if (v < 1) return v.toFixed(2) + ' ms';
  if (v < 1000) return v.toFixed(2) + ' ms';
  return (v / 1000).toFixed(2) + ' s';
}}

  function getBlockClass(name) {
    const n = name.toLowerCase();
    if (n.includes('router')) return 'router';
    if (n.includes('routed expert')) return 'router';
    if (n.includes('shared')) return 'router';
    if (n.includes('indexer')) return 'indexer';
    if (n.includes('attention') || n.includes('attn')) return 'attn';
    if (n.includes('ffn')) return 'ffn';
    return 'attn';
  }

function buildGlobalOps() {{
  if (!DATA.global_ops.length) return '';
  let html = '<div class="global-ops"><b style="color:var(--text-dim)">Global ops:</b>';
  for (const op of DATA.global_ops) {{
    const pct = STEP_TIME_MS > 0 ? (op.total_ms / STEP_TIME_MS * 100) : 0;
    html += `<div class="op-chip">${{op.name}} <span class="time">(${{fmtMs(op.total_ms)}}, ${{pct.toFixed(1)}}%)</span></div>`;
  }}
  html += '</div>';
  return html;
}}

function buildOpNode(op, blockTotalMs) {{
  const pct = blockTotalMs > 0 ? (op.total_ms / blockTotalMs * 100) : 0;
  const cls = op.kind.replace(/[^a-z]/g, '_');
  return `
    <div class="op-node" data-name="${{op.name.toLowerCase()}}"
         onmouseenter="showTooltip(event, ${JSON.stringify(op).replace(/"/g, '&quot;')})"
         onmouseleave="hideTooltip()">
      <span class="kind-tag ${{cls}}">${{op.kind}}</span>
      <span class="name">${{op.name}}</span>
      <span class="flops">${{fmtFlops(op.fwd_flops)}}</span>
      <div class="time-bar"><div class="fill ${{op.bound}}" style="width:${{pct}}%"></div></div>
      <span class="ms">${{fmtMs(op.total_ms)}}</span>
      <span class="pct">${{pct.toFixed(1)}}%</span>
    </div>`;
}}

function buildLayer(layer, modelTotalMs) {{
  const layerTotal = layer.blocks.reduce((s, b) => s + b.total_ms, 0);
  const pct = modelTotalMs > 0 ? (layerTotal / modelTotalMs * 100) : 0;

  let html = `<div class="layer" data-kind="${{layer.kind}}">`;
  html += `<div class="layer-header" onclick="toggleLayer(this)">`;
  html += `<span class="arrow">▶</span>`;
  html += `<span class="idx">L${{layer.id}}</span>`;
  html += `<span class="kind ${{layer.kind}}">${{layer.kind}}</span>`;
  html += `<span class="ops-count">${{layer.op_count}} ops</span>`;
  html += `<span class="total-ms">${{fmtMs(layerTotal)}} (${{pct.toFixed(1)}}%)</span>`;
  html += `</div>`;
  html += `<div class="layer-body">`;

  for (const blk of layer.blocks) {{
    const bcls = getBlockClass(blk.name);
    const blkPct = layerTotal > 0 ? (blk.total_ms / layerTotal * 100) : 0;
    html += `<div class="block">`;
    html += `<div class="block-header ${{bcls}}" onclick="toggleBlock(this)">`;
    html += `<span class="arrow">▶</span>`;
    html += `<span class="name">${{blk.name}}</span>`;
    html += `<span class="time">${{fmtMs(blk.total_ms)}} (${{blkPct.toFixed(1)}}%)</span>`;
    html += `</div>`;
    html += `<div class="block-body">`;
    for (const op of blk.ops) {{
      html += buildOpNode(op, blk.total_ms);
    }}
    html += `</div></div>`;
  }}

  html += `</div></div>`;
  return html;
}}

function renderTree() {{
  let html = '';

  // Summary cards
  const stepT = STEP_TIME_MS;
  html += `<div class="summary-cards">`;
  html += `<div class="card"><div class="label">Step Time</div><div class="value">${{fmtMs(stepT)}}</div></div>`;
  html += `<div class="card"><div class="label">Total Ops</div><div class="value">${{DATA.total_ops}}</div></div>`;
  html += `<div class="card"><div class="label">Layers</div><div class="value">${{DATA.layers.length}}</div></div>`;
  html += `<div class="card"><div class="label">Dense</div><div class="value">${{DATA.layers.filter(l=>l.kind==='dense').length}}</div></div>`;
  html += `<div class="card"><div class="label">MoE</div><div class="value">${{DATA.layers.filter(l=>l.kind==='moe').length}}</div></div>`;
  html += `</div>`;

  // Global ops
  html += buildGlobalOps();

  // Calculate total model time (sum of all layers)
  let modelTotalMs = 0;
  for (const layer of DATA.layers) {{
    modelTotalMs += layer.blocks.reduce((s, b) => s + b.total_ms, 0);
  }}

  // Layers
  for (const layer of DATA.layers) {{
    html += buildLayer(layer, modelTotalMs);
  }}

  document.getElementById('main').innerHTML = html;
}}

function renderChart() {{
  // Aggregate time by op kind across all ops
  const kindMap = {{}};
  for (const layer of DATA.layers) {{
    for (const blk of layer.blocks) {{
      for (const op of blk.ops) {{
        if (!kindMap[op.kind]) kindMap[op.kind] = {{ kind: op.kind, total: 0, count: 0 }};
        kindMap[op.kind].total += op.total_ms;
        kindMap[op.kind].count++;
      }}
    }}
  }}
  for (const op of DATA.global_ops) {{
    if (!kindMap[op.kind]) kindMap[op.kind] = {{ kind: op.kind, total: 0, count: 0 }};
    kindMap[op.kind].total += op.total_ms;
    kindMap[op.kind].count++;
  }}

  const sorted = Object.values(kindMap).sort((a, b) => b.total - a.total);
  const maxTotal = sorted.length > 0 ? sorted[0].total : 1;

  let html = '<div class="summary-cards">';
  html += `<div class="card"><div class="label">Step Time</div><div class="value">${{fmtMs(STEP_TIME_MS)}}</div></div>`;
  html += `</div>`;
  html += '<h3 style="margin:12px 0 8px;color:var(--text-dim)">Time by Op Kind</h3>';
  html += '<div class="bar-chart">';

  for (const k of sorted) {{
    const pct = (k.total / maxTotal * 100).toFixed(1);
    const w = (k.total / maxTotal * 100).toFixed(1);
    html += `<div class="bar-row">
      <span class="bar-label">${{k.kind}} (${{k.count}})</span>
      <div class="bar compute" style="width:${{w}}%"></div>
      <span class="bar-val">${{fmtMs(k.total)}} (${{pct}}%)</span>
    </div>`;
  }}

  html += '</div>';

  // Per-layer breakdown
  html += '<h3 style="margin:16px 0 8px;color:var(--text-dim)">Time by Layer</h3>';
  html += '<div class="bar-chart">';

  let maxL = 0;
  for (const layer of DATA.layers) {{
    const lt = layer.blocks.reduce((s, b) => s + b.total_ms, 0);
    if (lt > maxL) maxL = lt;
  }}

  for (const layer of DATA.layers) {{
    const lt = layer.blocks.reduce((s, b) => s + b.total_ms, 0);
    const w = maxL > 0 ? (lt / maxL * 100).toFixed(1) : 0;
    html += `<div class="bar-row">
      <span class="bar-label">L${{layer.id}} (${{layer.kind}})</span>
      <div class="bar memory" style="width:${{w}}%"></div>
      <span class="bar-val">${{fmtMs(lt)}}</span>
    </div>`;
  }}

  html += '</div>';

  document.getElementById('main').innerHTML = html;
}}

function toggleLayer(el) {{
  el.classList.toggle('open');
  const body = el.nextElementSibling;
  body.classList.toggle('open');
}}

function toggleBlock(el) {{
  el.classList.toggle('open');
  const body = el.nextElementSibling;
  body.classList.toggle('open');
}}

function expandAll() {{
  document.querySelectorAll('.layer-header, .block-header').forEach(el => {{
    el.classList.add('open');
    el.nextElementSibling.classList.add('open');
  }});
}}

function collapseAll() {{
  document.querySelectorAll('.layer-header, .block-header').forEach(el => {{
    el.classList.remove('open');
    el.nextElementSibling.classList.remove('open');
  }});
}}

function showChart() {{ renderChart(); }}
function showTree() {{ renderTree(); }}

function filterOps() {{
  const q = document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('.op-node').forEach(el => {{
    el.style.display = el.dataset.name.includes(q) ? '' : 'none';
  }});
}}

function showTooltip(e, op) {{
  const tip = document.getElementById('tooltip');
  tip.innerHTML = `
    <div style="font-weight:bold;margin-bottom:6px;color:#fff">${{op.name}}</div>
    <div class="row"><span class="k">Kind</span><span class="v">${{op.kind}}</span></div>
    <div class="row"><span class="k">Bound</span><span class="v">${{op.bound}}</span></div>
    <div class="row"><span class="k">FWD FLOPs</span><span class="v">${{fmtFlops(op.fwd_flops)}}</span></div>
    <div class="row"><span class="k">BWD FLOPs</span><span class="v">${{fmtFlops(op.dx_flops + op.dw_flops)}}</span></div>
    <div class="row"><span class="k">FWD Bytes</span><span class="v">${{fmtFlops(op.fwd_bytes)}}</span></div>
    <div class="row"><span class="k">FWD Time</span><span class="v">${{fmtMs(op.fwd_ms)}}</span></div>
    <div class="row"><span class="k">DX Time</span><span class="v">${{fmtMs(op.dx_ms)}}</span></div>
    <div class="row"><span class="k">DW Time</span><span class="v">${{fmtMs(op.dw_ms)}}</span></div>
    <div class="row"><span class="k" style="color:var(--orange)">Total</span><span class="v" style="color:var(--orange)">${{fmtMs(op.total_ms)}}</span></div>
  `;
  tip.classList.add('show');
  tip.style.left = Math.min(e.clientX + 12, window.innerWidth - 340) + 'px';
  tip.style.top = Math.min(e.clientY + 12, window.innerHeight - 200) + 'px';
}}

function hideTooltip() {{
  document.getElementById('tooltip').classList.remove('show');
}}

document.addEventListener('keydown', e => {{
  if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {{
    e.preventDefault();
    document.getElementById('search').focus();
  }}
  if (e.key === 'Escape') {{
    document.getElementById('search').value = '';
    filterOps();
    document.getElementById('search').blur();
  }}
}});

// Init
renderTree();

// Populate header
document.getElementById('header-stats').innerHTML = `
  <span class="stat">Model: <span>${{DATA.model_name}}</span></span>
  <span class="stat">Step Time: <span>${{fmtMs(STEP_TIME_MS)}}</span></span>
  <span class="stat">Layers: <span>${{DATA.layers.length}}</span></span>
`;
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
