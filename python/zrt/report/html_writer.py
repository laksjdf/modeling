"""Export performance reports to interactive HTML (zero external dependencies)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.report.summary import E2ESummary, TrainingSummary
    from python.zrt.report.report_types import ReportContext

logger = logging.getLogger(__name__)

# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #f5f5f5; color: #333; padding: 24px; }
h1 { font-size: 22px; margin-bottom: 16px; color: #1a237e; }
h2 { font-size: 16px; margin: 20px 0 10px; color: #37474f; border-bottom: 2px solid #1a237e; padding-bottom: 4px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }
.card { background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }
.card .label { font-size: 12px; color: #78909c; text-transform: uppercase; }
.card .value { font-size: 24px; font-weight: 700; color: #1a237e; margin-top: 4px; }
.card .unit { font-size: 12px; color: #78909c; }
.chart-container { background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); margin-bottom: 16px; }
canvas { width: 100%; height: 280px; }
table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }
th { background: #1a237e; color: #fff; padding: 10px 12px; text-align: left; font-size: 13px; cursor: pointer; user-select: none; }
th:hover { background: #283593; }
td { padding: 8px 12px; border-bottom: 1px solid #e0e0e0; font-size: 13px; }
tr:hover td { background: #f5f5f5; }
.heatmap { display: grid; grid-template-columns: repeat(auto-fill, minmax(60px, 1fr)); gap: 4px; margin: 12px 0; }
.heat-cell { padding: 8px 4px; text-align: center; border-radius: 4px; font-size: 11px; color: #fff; font-weight: 600; }
.meta { font-size: 13px; color: #78909c; margin-bottom: 16px; }
.meta span { margin-right: 16px; }

/* ── Timeline wrapper + interactive tooltip ── */
.tl-wrap { position: relative; }
.tl-tooltip {
    display: none;
    position: absolute;
    z-index: 100;
    pointer-events: none;
    background: rgba(15,23,42,0.93);
    color: #f1f5f9;
    border-radius: 10px;
    padding: 10px 13px;
    min-width: 200px;
    max-width: 320px;
    font-size: 12px;
    line-height: 1.55;
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.08);
}
.tl-head { display:flex; align-items:center; gap:6px; margin-bottom:6px; flex-wrap:wrap; }
.tl-opname { font-size:13px; font-weight:700; color:#f8fafc; }
.tl-scope {
    font-size: 10.5px; color: #94a3b8;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    word-break: break-all; margin-bottom: 5px;
}
.tl-kv { display:flex; justify-content:space-between; gap:12px; color:#cbd5e1; }
.tl-kv span { color:#64748b; }
.tl-kv b { color:#e2e8f0; font-weight:600; }
/* phase badges */
.tl-badge { display:inline-block; padding:1px 7px; border-radius:999px;
    font-size:10px; font-weight:700; text-transform:uppercase; }
.tl-ph-fwd  { background:#1d4ed8; color:#bfdbfe; }
.tl-ph-bwd  { background:#7e22ce; color:#e9d5ff; }
/* bound badges */
.tl-bd-compute      { background:#854d0e; color:#fef3c7; }
.tl-bd-memory       { background:#14532d; color:#bbf7d0; }
.tl-bd-communication{ background:#1e1b4b; color:#e0e7ff; }
"""

# ── JS ────────────────────────────────────────────────────────────────────────

_JS = """
<script>
// ── Timeline chart (interactive) ──────────────────────────────────────────────
function drawTimeline(canvasId, ops) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !ops.length) return;

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const cRect = canvas.getBoundingClientRect();
    canvas.width  = cRect.width  * dpr;
    canvas.height = cRect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = cRect.width, H = cRect.height;
    const pad = { top: 24, right: 20, bottom: 32, left: 72 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top  - pad.bottom;

    const minT    = Math.min(...ops.map(o => o.start));
    const maxT    = Math.max(...ops.map(o => o.end));
    const range   = maxT - minT || 1;
    const streams = [...new Set(ops.map(o => o.stream))].sort((a, b) => a - b);
    const nStream = streams.length;
    const streamH = plotH / nStream;

    // Base colors by category / stream_type
    const BASE_COLOR = { compute: "#22c55e", comm: "#ef4444", memory: "#f59e0b" };
    function opColor(op) { return BASE_COLOR[op.type] || "#90a4ae"; }
    function hexAlpha(hex, a) {
        const r = parseInt(hex.slice(1,3),16),
              g = parseInt(hex.slice(3,5),16),
              b = parseInt(hex.slice(5,7),16);
        return `rgba(${r},${g},${b},${a})`;
    }

    // Pre-compute hit areas once (CSS-pixel coordinates)
    const hitAreas = ops.map(op => {
        const x  = pad.left + ((op.start - minT) / range) * plotW;
        const w  = Math.max(1.5, ((op.end - op.start) / range) * plotW);
        const si = streams.indexOf(op.stream);
        const y  = pad.top + si * streamH + 3;
        const h  = streamH - 6;
        return { x, y, w, h, op };
    });

    // ── Draw helpers ──────────────────────────────────────────────────────────
    function drawAxes() {
        // Y-axis: stream labels
        ctx.font = "11px system-ui,sans-serif";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        streams.forEach((s, i) => {
            const ly = pad.top + i * streamH + streamH / 2;
            ctx.fillStyle = "#475569";
            ctx.fillText("Stream " + s, pad.left - 8, ly);
            // Horizontal guide line
            ctx.strokeStyle = "#e2e8f0";
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(pad.left, ly); ctx.lineTo(pad.left + plotW, ly);
            ctx.stroke();
        });
        // X-axis: time labels
        ctx.fillStyle  = "#64748b";
        ctx.textAlign  = "center";
        ctx.textBaseline = "alphabetic";
        ctx.font = "10px system-ui,sans-serif";
        const ticks = 6;
        for (let i = 0; i <= ticks; i++) {
            const t = minT + (range * i / ticks);
            const x = pad.left + plotW * i / ticks;
            ctx.fillText((t / 1000).toFixed(2) + " ms", x, H - 6);
            ctx.strokeStyle = "#e2e8f0"; ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH);
            ctx.stroke();
        }
    }

    function renderBars(highlightOp) {
        ctx.clearRect(0, 0, W, H);
        drawAxes();
        hitAreas.forEach(({ x, y, w, h, op }) => {
            const isHover = op === highlightOp;
            ctx.fillStyle = highlightOp
                ? (isHover ? opColor(op) : hexAlpha(opColor(op), 0.22))
                : opColor(op);
            ctx.beginPath();
            if (ctx.roundRect) ctx.roundRect(x, y, w, h, 2);
            else ctx.rect(x, y, w, h);
            ctx.fill();
            if (isHover) {
                ctx.strokeStyle = "#fff";
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
        });
        // Phase separators (fwd / bwd boundary)
        let prevPhase = null;
        hitAreas.forEach(({ x, op }) => {
            if (prevPhase && op.phase && op.phase !== prevPhase) {
                ctx.strokeStyle = "#94a3b8";
                ctx.lineWidth = 1;
                ctx.setLineDash([4, 3]);
                ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH);
                ctx.stroke();
                ctx.setLineDash([]);
            }
            if (op.phase) prevPhase = op.phase;
        });
    }

    renderBars(null);

    // ── Tooltip ───────────────────────────────────────────────────────────────
    const tooltip = document.getElementById(canvasId + "-tooltip");
    if (!tooltip) return;

    let hovered = null;

    function showTooltip(op, mx, my) {
        const latMs  = (op.latency_us / 1000).toFixed(3);
        const latUs  = op.latency_us.toFixed(1);
        const opName = (op.op_type || "").split(".").filter(Boolean).pop() || op.op_type || "?";
        const scope  = op.scope  ? `<div class="tl-scope">${op.scope}</div>` : "";
        const layer  = op.layer  ? `<div class="tl-kv"><span>Layer</span><b>${op.layer}</b></div>` : "";
        const tflops = op.tflops ? `<div class="tl-kv"><span>TFLOPs</span><b>${op.tflops}</b></div>` : "";
        const phase  = op.phase  ? `<span class="tl-badge tl-ph-${op.phase}">${op.phase.toUpperCase()}</span>` : "";
        const bound  = op.bound  ? `<span class="tl-badge tl-bd-${op.bound}">${op.bound}</span>` : "";

        tooltip.innerHTML =
            `<div class="tl-head"><span class="tl-opname">${opName}</span>${phase}${bound}</div>` +
            scope + layer +
            `<div class="tl-kv"><span>Stream</span><b>${op.stream}</b></div>` +
            `<div class="tl-kv"><span>Duration</span><b>${latMs} ms&nbsp;(${latUs} µs)</b></div>` +
            tflops;

        tooltip.style.display = "block";
        const tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
        const wrapW = canvas.getBoundingClientRect().width;
        let tx = mx + 14, ty = my - th / 2;
        if (tx + tw + 4 > wrapW) tx = mx - tw - 14;
        if (ty < 4) ty = 4;
        tooltip.style.left = tx + "px";
        tooltip.style.top  = ty + "px";
    }

    canvas.addEventListener("mousemove", function(e) {
        const r  = canvas.getBoundingClientRect();
        const mx = e.clientX - r.left;
        const my = e.clientY - r.top;

        let found = null;
        for (let i = hitAreas.length - 1; i >= 0; i--) {
            const { x, y, w, h, op } = hitAreas[i];
            if (mx >= x && mx <= x + w && my >= y && my <= y + h) {
                found = op; break;
            }
        }

        if (found !== hovered) {
            hovered = found;
            renderBars(hovered);
        }
        if (hovered) showTooltip(hovered, mx, my);
        else          tooltip.style.display = "none";
    });

    canvas.addEventListener("mouseleave", function() {
        hovered = null;
        renderBars(null);
        tooltip.style.display = "none";
    });
}

// ── Pie chart ──
function drawPie(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !data.length) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;
    const cx = W * 0.35, cy = H / 2, r = Math.min(cx, cy) - 20;
    const total = data.reduce((s, d) => s + d.value, 0) || 1;
    const palette = ["#1a237e","#283593","#3949ab","#5c6bc0","#7986cb","#9fa8da","#c5cae9","#e8eaf6","#37474f","#546e7a"];
    let angle = -Math.PI / 2;
    data.forEach((d, i) => {
        const slice = (d.value / total) * 2 * Math.PI;
        ctx.beginPath(); ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, r, angle, angle + slice);
        ctx.fillStyle = palette[i % palette.length];
        ctx.fill();
        angle += slice;
    });
    // Inner circle (donut)
    ctx.beginPath(); ctx.arc(cx, cy, r * 0.5, 0, Math.PI * 2);
    ctx.fillStyle = "#fff"; ctx.fill();
    // Legend
    const lx = W * 0.65; let ly = 30;
    ctx.font = "12px sans-serif"; ctx.textAlign = "left";
    data.forEach((d, i) => {
        ctx.fillStyle = palette[i % palette.length];
        ctx.fillRect(lx, ly - 8, 12, 12);
        ctx.fillStyle = "#333";
        ctx.fillText(`${d.label}  ${d.value.toFixed(1)}%`, lx + 18, ly + 2);
        ly += 22;
    });
}

// ── Sortable table ──
function sortTable(tableId, colIdx) {
    const table = document.getElementById(tableId);
    const rows = Array.from(table.querySelectorAll("tbody tr"));
    const asc = table.dataset.sortCol == colIdx ? table.dataset.sortDir === "asc" : true;
    rows.sort((a, b) => {
        const va = a.cells[colIdx].textContent.trim();
        const vb = b.cells[colIdx].textContent.trim();
        const na = parseFloat(va), nb = parseFloat(vb);
        if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
        return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    });
    const tbody = table.querySelector("tbody");
    rows.forEach(r => tbody.appendChild(r));
    table.dataset.sortCol = colIdx;
    table.dataset.sortDir = asc ? "desc" : "asc";
}
</script>
"""


def export_html_report(
    summary: "E2ESummary | TrainingSummary",
    output_path: Path,
    timeline_data: list[dict] | None = None,
) -> Path:
    """Export an interactive HTML performance report (flat E2ESummary).

    Parameters
    ----------
    summary : E2ESummary | TrainingSummary
        Performance summary from ``build_summary()`` or ``build_training_summary()``.
    output_path : Path
        Output HTML file path.
    timeline_data : list[dict] | None
        Optional timeline ops for the Gantt chart. Each dict:
        ``{"start": float, "end": float, "stream": int, "type": "compute"|"comm"}``.
        If None, the chart section is omitted.

    Returns
    -------
    Path
        The output HTML file path.
    """
    from python.zrt.report.summary import E2ESummary, TrainingSummary

    is_training = isinstance(summary, TrainingSummary)

    # ── Build page title and metadata ──
    if is_training:
        title = f"Training Report: {summary.model}"
        meta_parts = [
            f"Hardware: {summary.hardware}",
            f"Parallel: {summary.parallel_desc}",
            f"Step: {summary.step_ms:.3f} ms",
            f"MFU: {summary.mfu:.1%}",
        ]
    else:
        title = f"Inference Report: {summary.model} | {summary.phase.upper()}"
        meta_parts = [
            f"Hardware: {summary.hardware}",
            f"Parallel: {summary.parallel_desc}",
            f"Latency: {summary.latency_ms:.3f} ms",
            f"MFU: {summary.mfu:.1%}",
        ]

    # ── Cards ──
    if is_training:
        cards = [
            ("Step Latency", f"{summary.step_ms:.3f}", "ms"),
            ("Forward", f"{summary.forward_ms:.3f}", "ms"),
            ("Backward", f"{summary.backward_ms:.3f}", "ms"),
            ("Tokens/s", f"{summary.tokens_per_sec:.0f}", ""),
            ("MFU", f"{summary.mfu:.1%}", ""),
            ("HBM Util", f"{summary.hbm_bw_util:.1%}", ""),
        ]
    else:
        cards = [
            ("Latency", f"{summary.latency_ms:.3f}", "ms"),
            ("Throughput", f"{summary.tokens_per_sec:.0f}", "tok/s"),
            ("MFU", f"{summary.mfu:.1%}", ""),
            ("HBM Util", f"{summary.hbm_bandwidth_util:.1%}", ""),
        ]
        if summary.ttft_ms is not None:
            cards.insert(1, ("TTFT", f"{summary.ttft_ms:.3f}", "ms"))
        if summary.tpot_ms is not None:
            cards.insert(2, ("TPOT", f"{summary.tpot_ms:.3f}", "ms/token"))

    cards_html = "".join(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{value}<span class="unit"> {unit}</span></div></div>'
        for label, value, unit in cards
    )

    # ── Timeline ──
    # Canvas + tooltip div; init call deferred after _JS.
    timeline_html = ""
    timeline_init = ""
    if timeline_data:
        ops_json = json.dumps(timeline_data)
        timeline_html = """
<h2>Timeline</h2>
<div class="chart-container">
    <div class="tl-wrap">
        <canvas id="timeline" style="display:block;width:100%;height:280px"></canvas>
        <div id="timeline-tooltip" class="tl-tooltip"></div>
    </div>
</div>
"""
        timeline_init = f"drawTimeline('timeline', {ops_json});"

    # ── Component pie chart ──
    pie_html = ""
    if summary.by_component:
        pie_data = [{"label": k, "value": v} for k, v in
                     sorted(summary.by_component.items(), key=lambda x: -x[1])]
        pie_json = json.dumps(pie_data)
        pie_html = f"""
<h2>Component Breakdown</h2>
<div class="chart-container">
    <canvas id="pie" style="display:block;width:100%;height:280px"></canvas>
</div>
"""
        timeline_init += f"\ndrawPie('pie', {pie_json});"

    # ── Layer heatmap ──
    heatmap_html = ""
    if summary.by_layer:
        layers = summary.by_layer
        max_lat = max(layers) if layers else 1
        cells = []
        for i, lat in enumerate(layers):
            intensity = lat / max_lat if max_lat > 0 else 0
            r = int(26 + intensity * 218)
            g = int(35 + (1 - intensity) * 100)
            b = int(126 + (1 - intensity) * 80)
            cells.append(
                f'<div class="heat-cell" style="background:rgb({r},{g},{b})">'
                f"L{i}<br>{lat:.2f}ms</div>"
            )
        heatmap_html = f"""
<h2>Layer Latency Heatmap</h2>
<div class="chart-container">
    <div class="heatmap">{''.join(cells)}</div>
</div>
"""

    # ── Bottleneck table ──
    bottleneck_html = ""
    if summary.top_bottleneck_ops:
        rows = "".join(
            f"<tr><td>{i}</td><td>{desc}</td><td>{lat:.1f}</td></tr>"
            for i, (desc, lat) in enumerate(summary.top_bottleneck_ops, 1)
        )
        bottleneck_html = f"""
<h2>Top Bottleneck Operators</h2>
<table id="bottlenecks" data-sort-col="-1" data-sort-dir="asc">
    <thead><tr><th onclick="sortTable('bottlenecks',0)">#</th>
    <th onclick="sortTable('bottlenecks',1)">Operator</th>
    <th onclick="sortTable('bottlenecks',2)">Latency (us)</th></tr></thead>
    <tbody>{rows}</tbody>
</table>
"""

    # ── Assemble ──
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
{_CSS}
</head>
<body>
<h1>{title}</h1>
<div class="meta">{''.join(f'<span>{p}</span>' for p in meta_parts)}</div>
<div class="cards">{cards_html}</div>
{timeline_html}
{pie_html}
{heatmap_html}
{bottleneck_html}
{_JS}
<script>{timeline_init}</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Exported HTML report to %s", output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical HTML Report (Phase 1+ rendering)
# ─────────────────────────────────────────────────────────────────────────────

_HIER_CSS = """
/* ── Timeline layout ── */
.timeline { position: relative; padding-left: 28px; }
.timeline::before { content: ''; position: absolute; left: 9px; top: 12px; bottom: 12px;
    width: 2px; background: #dbe4ee; }
.timeline-item { position: relative; margin-bottom: 18px; }
.timeline-dot { position: absolute; left: -17px; top: 22px; width: 12px; height: 12px;
    border-radius: 50%; background: #38bdf8; border: 2px solid white;
    box-shadow: 0 0 0 2px #bae6fd; }

/* ── Details / Summary tree ── */
details summary {
    list-style: none; cursor: pointer; display: flex;
    align-items: center; justify-content: space-between; gap: 12px; font-weight: 700;
}
details summary::-webkit-details-marker { display: none; }
details summary::before {
    content: '+'; width: 22px; height: 22px; line-height: 20px;
    text-align: center; border-radius: 50%; border: 1px solid #cbd5e1;
    color: #334155; flex: 0 0 22px; background: #fff; font-weight: 700;
}
details[open] summary::before { content: '−'; }

.summary-right { margin-left: auto; color: #475569; font-weight: 600; font-size: 13px; }
.sum-left { display: flex; align-items: center; gap: 8px; }

/* ── Block card ── */
.block {
    padding: 10px 12px;
    background: linear-gradient(180deg, #ffffff 0%, #fcfdff 100%);
    border: 1px solid #dbe4ee; border-radius: 18px;
    box-shadow: 0 4px 20px rgba(15,23,42,.05);
}
.op-family {
    padding: 10px 12px; margin-top: 10px;
    border: 1px solid #dbe4ee; border-radius: 18px;
    background: #fff; box-shadow: 0 4px 20px rgba(15,23,42,.05);
}

.block-meta { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 8px 34px; }
.chip { display: inline-flex; padding: 5px 10px; border-radius: 999px;
    border: 1px solid #dbe4ee; background: #f8fbff; color: #334155; font-size: 12px; }

/* ── Sub-structure lanes ── */
.subgraph-wrap { margin: 12px 0 10px 34px; padding: 14px;
    border: 1px dashed #cbd5e1; border-radius: 16px; background: #fbfdff; }
.subgraph-lane + .subgraph-lane { margin-top: 16px; }
.lane-title { font-size: 13px; font-weight: 800; color: #0f172a; margin-bottom: 8px; }
.subgraph-flow { display: flex; gap: 10px; overflow-x: auto; align-items: stretch; padding-bottom: 4px; }
.sg-arrow { display: flex; align-items: center; justify-content: center;
    min-width: 22px; color: #94a3b8; font-size: 20px; }

/* ── Comp-box cards ── */
.comp-box { min-width: 160px; border-radius: 14px; border: 1px solid #dbe4ee;
    background: #fff; padding: 10px; box-shadow: 0 2px 10px rgba(15,23,42,.03); }
.comp-box.norm    { background: #f8fafc; }
.comp-box.proj    { background: #eff6ff; }
.comp-box.index   { background: #f5f3ff; }
.comp-box.attn    { background: #ecfeff; }
.comp-box.memory  { background: #ecfdf5; }
.comp-box.comm    { background: #fef2f2; }
.comp-box.router  { background: #fff7ed; }
.comp-box.act     { background: #faf5ff; }
.comp-box.shared  { background: #fdf2f8; }
.comp-box.resid   { background: #f8fafc; }
.cb-title { font-weight: 800; font-size: 13px; color: #334155; }
.cb-ms { font-size: 20px; font-weight: 800; margin-top: 8px; color: #0f172a; }
.cb-meta { color: #64748b; margin-top: 6px; font-size: 11px; line-height: 1.4; }

/* ── Bound bar ── */
.bound-row { position: relative; height: 28px; background: #eff4f8; border-radius: 999px;
    overflow: hidden; margin: 4px 0; border: 1px solid #e2e8f0; }
.bound-row .bound-fill { height: 100%; border-radius: 999px; }
.bound-row.comp .bound-fill { background: #fdba74; }
.bound-row.mem  .bound-fill { background: #86efac; }
.bound-row.comm .bound-fill { background: #a5b4fc; }
.bound-label { position: absolute; left: 12px; top: 50%; transform: translateY(-50%);
    font-size: 12px; font-weight: 600; color: #334155; white-space: nowrap; }

/* ── Bound tag ── */
.bound-tag { display: inline-block; padding: 1px 8px; border-radius: 999px;
    font-size: 10px; font-weight: 700; text-transform: uppercase; margin-left: 8px; }
.bound-tag.compute { background: #fed7aa; color: #9a3412; }
.bound-tag.memory  { background: #bbf7d0; color: #166534; }
.bound-tag.communication { background: #e9d5ff; color: #6b21a8; }

/* ── Compact table (for op-family detail) ── */
.compact { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px;
    background: white; border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; }
.compact th, .compact td { border-bottom: 1px solid #e5e7eb; padding: 6px 8px;
    text-align: right; vertical-align: top; }
.compact th:nth-child(1), .compact td:nth-child(1),
.compact th:nth-child(2), .compact td:nth-child(2),
.compact th:nth-child(3), .compact td:nth-child(3),
.compact th:nth-child(4), .compact td:nth-child(4) { text-align: left; }
.compact th { position: sticky; top: 0; background: #f8fafc; z-index: 1;
    color: #334155; font-size: 11px; font-weight: 700; }
.compact code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    white-space: pre-wrap; word-break: break-word; font-size: 11px; }

.bound-compute { color: #9a3412; font-weight: 700; }
.bound-memory { color: #047857; font-weight: 700; }
.bound-communication { color: #3730a3; font-weight: 700; }

.warning-box { background: #FFF8E1; border-left: 4px solid #FFA000; padding: 12px 16px;
    margin: 12px 0; border-radius: 4px; font-size: 13px; }
.warning-box li { margin-left: 18px; padding: 2px 0; }
"""

_HIER_JS = """
<script>
function switchViz(panelId, btn) {
    document.querySelectorAll('.viz-panel').forEach(p => p.style.display = 'none');
    document.querySelectorAll('.viz-tab').forEach(b => {b.style.background='#f8fafc';b.style.color='#64748b';b.style.borderColor='#e2e8f0'});
    document.getElementById(panelId).style.display = 'block';
    btn.style.background = '#3b82f6';
    btn.style.color = '#fff';
    btn.style.borderColor = '#3b82f6';
}
</script>
"""


# ── Render helpers for hierarchical report ──────────────────────────────────

def _comp_box_css_class(component_type: str) -> str:
    """Map a component_type to a CSS class for comp-box coloring."""
    ct = (component_type or "").lower()
    if "attn" in ct:
        return "attn"
    if "norm" in ct:
        return "norm"
    if "router" in ct or ("gate" in ct and "moe" in ct):
        return "router"
    if "ffn" in ct or "mlp" in ct or "moe" in ct or "expert" in ct:
        return "proj"
    if "comm" in ct or "dispatch" in ct or "combine" in ct:
        return "comm"
    if "shared" in ct:
        return "shared"
    if "add" in ct or "residual" in ct:
        return "resid"
    if "rope" in ct or "rotary" in ct or "misc" in ct:
        return "index"
    if "dsa" in ct or "csa" in ct or "index" in ct:
        return "index"
    if "silu" in ct or "gelu" in ct or "act" in ct:
        return "act"
    if "kv" in ct or "append" in ct or "copy" in ct:
        return "memory"
    return "proj"


def _group_sub_structures(ss_list):
    """Group sub-structures into semantic lanes (Attention, MoE FFN, etc.)."""
    from collections import OrderedDict
    lanes: dict[str, list] = OrderedDict()

    for ss in ss_list:
        ct = (ss.component_type or "").lower()

        # Determine lane
        if "attn" in ct or ("norm" in ct and "attn" in (ss.scope_group or "").lower()):
            lane = "Attention"
        elif "router" in ct or ("gate" in ct and "moe" in ct):
            lane = "Router / Gate"
        elif "dispatch" in ct or "combine" in ct:
            lane = "Communication"
        elif "comm" in ct:
            lane = "Communication"
        elif "ffn" in ct or "mlp" in ct or "moe" in ct or "expert" in ct or "shared" in ct:
            lane = "MoE / FFN"
        elif "embed" in ct or "lm_head" in ct:
            lane = "Embedding / Output"
        elif "residual" in ct or "add" in ct:
            lane = "Residual"
        elif "norm" in ct:
            lane = "Norm"
        else:
            lane = "Other"

        if lane not in lanes:
            lanes[lane] = []
        lanes[lane].append(ss)

    return list(lanes.items())


def _render_comp_box_card(ss) -> str:
    """Render a single comp-box card for a SubStructureDetail."""
    css = _comp_box_css_class(ss.component_type)
    display = ss.name or ss.scope_group or "?"
    # Compute aggregate FLOPs + HBM from op_families
    total_tflops = sum(f.tflops for f in ss.op_families)
    total_hbm = sum(f.hbm_bytes for f in ss.op_families)
    flops_str = f"{total_tflops:.2f} TF" if total_tflops >= 0.01 else f"{total_tflops * 1000:.0f} GF"
    if total_hbm >= 1e9:
        hbm_str = f"{total_hbm / 1e9:.2f} GB"
    elif total_hbm >= 1e6:
        hbm_str = f"{total_hbm / 1e6:.2f} MB"
    else:
        hbm_str = "0 B"

    return (
        f'<div class="comp-box {css}">'
        f'<div class="cb-title">{display}</div>'
        f'<div class="cb-ms">{ss.total_ms:.3f} ms</div>'
        f'<div class="cb-meta">FLOPs {flops_str} · HBM {hbm_str}</div>'
        f'</div>'
    )


def _fmt_bytes(n: int) -> str:
    """Format bytes into human-readable string."""
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.2f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.1f} KB"
    return f"{n} B"


def _render_op_family_details(fam) -> str:
    """Render a single op-family as a <details>/<summary> expandable block."""
    bound_cls = f"bound-{fam.bound}" if fam.bound else ""
    display = fam.display_name or fam.op_type.split(".")[-1]

    summary_right = (
        f'<span class="summary-right">'
        f'{fam.total_ms:.3f} ms · {fam.pct_of_substructure:.1f}% · '
        f'<span class="{bound_cls}">{fam.bound}</span>'
        f'</span>'
    )

    # Build the 12-column detail table row
    children_rows = ""
    for ch in fam.children:
        hbm = ch.read_bytes + ch.write_bytes
        children_rows += (
            f"<tr>"
            f"<td>{fam.count}</td>"
            f"<td>{fam.op_type}</td>"
            f"<td><code>{ch.shape_desc}</code></td>"
            f"<td><code>{fam.formula}</code></td>"
            f"<td>{fam.tflops:.4f}</td>"
            f"<td>{_fmt_bytes(hbm)}</td>"
            f"<td>{_fmt_bytes(fam.comm_bytes)}</td>"
            f"<td>{fam.compute_ms:.4f}</td>"
            f"<td>{fam.memory_ms:.4f}</td>"
            f"<td>{fam.comm_ms:.4f}</td>"
            f"<td><b>{fam.total_ms:.4f}</b></td>"
            f"<td class='{bound_cls}'>{fam.bound}</td>"
            f"</tr>"
        )
        break  # one row per OpFamily (aggregated), not per child

    return (
        f"<details class='op-family'>"
        f"<summary><span>{display}</span>{summary_right}</summary>"
        f"<table class='compact'><thead><tr>"
        f"<th>Count</th><th>Type</th><th>Shape</th><th>Formula</th>"
        f"<th>TFLOPs</th><th>HBM Bytes</th><th>Comm Bytes</th>"
        f"<th>Compute ms</th><th>Memory ms</th><th>Comm ms</th>"
        f"<th>Total ms</th><th>Bound</th>"
        f"</tr></thead><tbody>{children_rows}</tbody></table>"
        f"</details>"
    )


def export_hierarchical_html_report(
    rc: "ReportContext",
    output_path: Path,
    timeline_data: list[dict] | None = None,
    *,
    hw_spec: "Any | None" = None,
    parallel: "Any | None" = None,
) -> Path:
    """Export a 4-level hierarchical HTML performance report from ReportContext.

    Sections rendered:
      - Hero Card (5 KPI: Prefill/TPOT/TPS/Memory/Blocks)
      - Bound Bar (compute / memory / communication %)
      - Block → SubStructure → OpFamily expandable hierarchy
      - 12-column OpFamily detail table (on click expand)
      - Timeline canvas (optional)
      - Warnings box

    Parameters
    ----------
    rc : ReportContext
        Built by ``build_report_context()`` with full hierarchical data.
    output_path : Path
        Output HTML file path.
    timeline_data : list[dict] | None
        Optional Gantt chart data.

    Returns
    -------
    Path
    """
    from python.zrt.report.report_types import ReportContext

    # ── Title & metadata ──
    phase_label = rc.phase.upper() if rc.phase else ""
    title = f"Performance Report: {rc.model} | {phase_label}"
    meta_parts = [
        f"Hardware: {rc.hardware}",
        f"Topology: {rc.topology_desc}",
        f"Parallel: {rc.parallel_desc}",
        f"Batch: {rc.batch_size}",
        f"SeqLen: {rc.seq_len}",
    ]

    # ── Hero Cards ──
    cards: list[tuple[str, str, str]] = []
    if rc.prefill_ms is not None:
        cards.append(("Prefill Latency", f"{rc.prefill_ms:.3f}", "ms"))
    if rc.tpot_ms is not None:
        cards.append(("TPOT", f"{rc.tpot_ms:.3f}", "ms/token"))
        if rc.mtp_adjusted_tpot_ms is not None:
            cards.append(("MTP-adj TPOT", f"{rc.mtp_adjusted_tpot_ms:.3f}", "ms/token"))
    cards.append(("Tokens/s", f"{rc.tokens_per_sec:.0f}", ""))
    if rc.memory_per_gpu_gb > 0:
        cards.append(("Memory/GPU", f"{rc.memory_per_gpu_gb:.1f}", "GB"))
    if rc.model_blocks > 0:
        cards.append(("Model Blocks", str(rc.model_blocks), ""))

    cards_html = "".join(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{value}<span class="unit"> {unit}</span></div></div>'
        for label, value, unit in cards
    )

    # ── Bound Bar ──
    bound_html = ""
    if rc.compute_pct > 0 or rc.memory_pct > 0 or rc.communication_pct > 0:
        bars = ""
        if rc.communication_pct > 0:
            bars += (
                f'<div class="bound-row comm">'
                f'<div class="bound-fill" style="width:{rc.communication_pct:.1f}%"></div>'
                f'<span class="bound-label">communication: {rc.communication_ms:.2f} ms ({rc.communication_pct:.1f}%)</span>'
                f'</div>'
            )
        if rc.compute_pct > 0:
            bars += (
                f'<div class="bound-row comp">'
                f'<div class="bound-fill" style="width:{rc.compute_pct:.1f}%"></div>'
                f'<span class="bound-label">compute: {rc.compute_ms:.2f} ms ({rc.compute_pct:.1f}%)</span>'
                f'</div>'
            )
        if rc.memory_pct > 0:
            bars += (
                f'<div class="bound-row mem">'
                f'<div class="bound-fill" style="width:{rc.memory_pct:.1f}%"></div>'
                f'<span class="bound-label">memory: {rc.memory_ms:.2f} ms ({rc.memory_pct:.1f}%)</span>'
                f'</div>'
            )
        bound_html = (
            f'<div class="section">'
            f'<h2>总体 Bound</h2>'
            f'{bars}'
            f'<div style="color:#64748b;font-size:12px;margin-top:4px">'
            f'橙色=计算受限，绿色=访存受限，紫色=通信受限。</div>'
            f'</div>'
        )

    # ── Hierarchy (Block → SubStructure → OpFamily) ──
    hierarchy_html = ""
    if rc.blocks:
        hierarchy_parts = ['<div class="timeline">']
        for bi, blk in enumerate(rc.blocks):
            repeat_str = f" × {blk.repeat} layers" if blk.repeat > 1 else ""
            op_family_count = sum(len(ss.op_families) for ss in blk.sub_structures)

            # Block: details/summary with timeline-dot
            block_title = f"<span class='sum-left'>{rc.phase}.{blk.name}{repeat_str}</span>"
            block_right = (
                f"<span class='summary-right'>"
                f"{blk.total_ms:.3f} ms · Share {blk.pct_of_total:.1f}% · {blk.dominant_bound} bound"
                f"</span>"
            )

            hierarchy_parts.append(
                f'<div class="timeline-item">'
                f'<div class="timeline-dot"></div>'
                f"<details class='block' open>"
                f"<summary>{block_title}{block_right}</summary>"
            )

            # Block meta chips
            hierarchy_parts.append(
                f'<div class="block-meta">'
                f'<span class="chip">重复层数: {blk.repeat}</span>'
                f'<span class="chip">主导瓶颈: {blk.dominant_bound}</span>'
                f'<span class="chip">算子族数量: {op_family_count}</span>'
                f'</div>'
            )

            # Sub-structures as card lanes with arrow connectors
            if blk.sub_structures:
                ss_by_lane = _group_sub_structures(blk.sub_structures)
                hierarchy_parts.append('<div class="subgraph-wrap">')
                for lane_name, ss_list in ss_by_lane:
                    hierarchy_parts.append(
                        f'<div class="subgraph-lane">'
                        f'<div class="lane-title">{lane_name}</div>'
                        f'<div class="subgraph-flow">'
                    )
                    for si, ss in enumerate(ss_list):
                        if si > 0:
                            hierarchy_parts.append('<span class="sg-arrow">→</span>')
                        hierarchy_parts.append(_render_comp_box_card(ss))
                    hierarchy_parts.append('</div></div>')  # subgraph-flow / subgraph-lane
                hierarchy_parts.append('</div>')  # subgraph-wrap

            # Op families as nested <details>/<summary>
            for si, ss in enumerate(blk.sub_structures):
                for fi, fam in enumerate(ss.op_families):
                    hierarchy_parts.append(_render_op_family_details(fam))

            hierarchy_parts.append('</details></div>')  # details.block / timeline-item

        hierarchy_parts.append('</div>')  # timeline
        hierarchy_html = "\n".join(hierarchy_parts)

    # ── Timeline ──
    # Canvas + tooltip div; init deferred after _JS so drawTimeline is defined.
    timeline_html = ""
    timeline_init = ""
    if timeline_data:
        ops_json = json.dumps(timeline_data)
        timeline_html = """
<h2>Timeline</h2>
<div class="chart-container">
    <div class="tl-wrap">
        <canvas id="timeline" style="display:block;width:100%;height:280px"></canvas>
        <div id="timeline-tooltip" class="tl-tooltip"></div>
    </div>
</div>
"""
        timeline_init = f"drawTimeline('timeline', {ops_json});"

    # ── Warnings ──
    warning_html = ""
    if rc.warnings:
        items = "".join(f"<li>{w}</li>" for w in rc.warnings)
        warning_html = (
            f'<div class="section">'
            f'<h2>告警与说明</h2>'
            f'<ul style="color:#64748b;line-height:1.8;font-size:13px">{items}</ul>'
            f'</div>'
        )

    # ── Calibration ──
    calib_html = ""
    if rc.calibration:
        # Helper: safely convert calibration value to float, handling "671.0B" etc.
        def _safe_calib_float(v):
            if v is None:
                return 0.0
            s = str(v)
            # Strip trailing non-numeric characters (units like B, M, G, %)
            while s and not s[-1].isdigit():
                s = s[:-1]
            try:
                return float(s) if s else 0.0
            except ValueError:
                return 0.0

        # Compute overall consistency
        total_entries = len(rc.calibration)
        exact_matches = sum(
            1 for c in rc.calibration
            if abs(_safe_calib_float(c.get("official", 0)) - _safe_calib_float(c.get("modeled", 0))) < 0.001
        )
        consistency = exact_matches / total_entries * 100 if total_entries > 0 else 100
        grade = "高" if consistency >= 90 else ("中" if consistency >= 70 else "低")
        grade_color = "high" if consistency >= 90 else ("midhigh" if consistency >= 70 else "low")

        calib_rows = ""
        for c in rc.calibration:
            official = c.get("official", "")
            modeled = c.get("modeled", "")
            unit = c.get("unit", "")
            source_name = c.get("source_name", "")
            source_url = c.get("source_url", "#")
            source_note = c.get("source_note", "")

            # Compute errors
            try:
                off_val = float(official)
                mod_val = float(modeled)
                abs_err = abs(off_val - mod_val)
                rel_err = abs_err / abs(off_val) * 100 if abs(off_val) > 0 else 0.0
            except (ValueError, TypeError):
                abs_err = "-"
                rel_err = "-"

            cite_html = ""
            if source_name:
                cite_html = (
                    f'<a class="cite-link" href="{source_url}" rel="noopener noreferrer" target="_blank">'
                    f'{source_name}</a>'
                )
                if source_note:
                    cite_html += f'<div class="cite-note" style="font-size:11px;color:#94a3b8">{source_note}</div>'

            calib_rows += (
                f"<tr>"
                f"<td>{c.get('metric', '')}</td>"
                f"<td>{official}</td>"
                f"<td>{modeled}</td>"
                f"<td>{unit}</td>"
                f"<td>{abs_err}</td>"
                f"<td>{rel_err if isinstance(rel_err, str) else f'{rel_err:.2f}%'}</td>"
                f"<td>{cite_html}</td>"
                f"</tr>"
            )

        calib_html = (
            f'<div class="section">'
            f'<h2>校准对比（官方数据 vs 建模结果）</h2>'
            f'<div class="subcards" style="display:grid;grid-template-columns:repeat(2,minmax(180px,1fr));gap:12px;margin-bottom:14px">'
            f'<div class="mini-card" style="background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:14px">'
            f'<div style="font-size:12px;color:#64748b">整体校准一致性</div>'
            f'<div class="mini-value {grade_color}" style="font-size:24px;font-weight:800;margin-top:6px">{consistency:.1f}%</div>'
            f'</div>'
            f'<div class="mini-card" style="background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:14px">'
            f'<div style="font-size:12px;color:#64748b">结论</div>'
            f'<div class="mini-value {grade_color}" style="font-size:24px;font-weight:800;margin-top:6px">{grade}</div>'
            f'</div>'
            f'</div>'
            f'<table class="compact" style="font-size:13px">'
            f'<thead><tr>'
            f'<th>指标</th><th>官方值</th><th>建模值</th><th>单位</th>'
            f'<th>绝对误差</th><th>相对误差</th><th>官方引用</th>'
            f'</tr></thead>'
            f'<tbody>{calib_rows}</tbody>'
            f'</table>'
            f'</div>'
        )

    # ── Operator-level calibration ──
    op_calib_html = ""
    if rc.blocks:
        all_fams = []
        for blk in rc.blocks:
            for ss in blk.sub_structures:
                all_fams.extend(ss.op_families)
        if all_fams:
            all_fams.sort(key=lambda f: -f.total_ms)
            total_ms = sum(f.total_ms for f in all_fams) or 1.0

            # Confidence sub-cards
            weighted_conf = sum(f.confidence * f.total_ms for f in all_fams) / total_ms * 100
            high_conf_pct = sum(f.total_ms for f in all_fams if f.confidence >= 0.7) / total_ms * 100
            grade = "高" if weighted_conf >= 80 else ("中" if weighted_conf >= 60 else "低")
            grade_c = "high" if weighted_conf >= 80 else ("midhigh" if weighted_conf >= 60 else "low")

            # Calibration method distribution
            method_counts: dict[str, float] = {}
            for f in all_fams:
                method = "解析公式/经验模板"
                if f.confidence >= 0.9:
                    method = "官方实现/官方资料"
                elif f.confidence >= 0.7:
                    method = "Source-backed"
                elif f.category == "communication":
                    method = "通信模板/NCCL近似"
                method_counts[method] = method_counts.get(method, 0) + f.total_ms
            method_bars = ""
            method_colors = {
                "官方实现/官方资料": "#22c55e",
                "Source-backed": "#3b82f6",
                "解析公式/经验模板": "#f59e0b",
                "通信模板/NCCL近似": "#ef4444",
            }
            for method, ms in sorted(method_counts.items(), key=lambda x: -x[1]):
                pct = ms / total_ms * 100
                color = method_colors.get(method, "#94a3b8")
                method_bars += (
                    f'<div class="bound-row" style="margin:2px 0">'
                    f'<div class="bound-fill" style="width:{pct:.1f}%;background:{color}"></div>'
                    f'<span class="bound-label">{method}: {pct:.1f}%</span>'
                    f'</div>'
                )

            # Per-op rows
            op_rows = ""
            for f in all_fams[:20]:  # top 20 ops
                method = "解析公式/经验模板"
                if f.confidence >= 0.9:
                    method = "官方实现/官方资料"
                elif f.confidence >= 0.7:
                    method = "Source-backed"
                elif f.category == "communication":
                    method = "通信模板/NCCL近似"
                conf_pct = int(f.confidence * 100)
                conf_label = "高" if conf_pct >= 70 else ("中" if conf_pct >= 40 else "低")
                op_rows += (
                    f"<tr>"
                    f"<td><code>{f.op_type.split('.')[-1]}</code></td>"
                    f"<td>{f.total_ms:.3f} ms</td>"
                    f"<td>{f.total_ms / total_ms * 100:.1f}%</td>"
                    f"<td><span class='bound-tag {f.bound}'>{f.bound}</span></td>"
                    f"<td>{method}</td>"
                    f"<td>{conf_pct}% ({conf_label})</td>"
                    f"</tr>"
                )

            op_calib_html = (
                f'<h3 style="margin-top:18px;">算子级校准覆盖与置信度分析</h3>'
                f'<div style="display:grid;grid-template-columns:repeat(4,minmax(140px,1fr));'
                f'gap:12px;margin-bottom:14px">'
                f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:14px">'
                f'<div style="font-size:12px;color:#64748b">时延加权置信度</div>'
                f'<div style="font-size:24px;font-weight:800;margin-top:6px;'
                f'color:#166534">{weighted_conf:.1f}%</div>'
                f'<small style="color:#64748b">{grade}</small></div>'
                f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:14px">'
                f'<div style="font-size:12px;color:#64748b">高置信覆盖率</div>'
                f'<div style="font-size:24px;font-weight:800;margin-top:6px;'
                f'color:#166534">{high_conf_pct:.1f}%</div>'
                f'<small style="color:#64748b">conf ≥ 70%</small></div>'
                f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:14px">'
                f'<div style="font-size:12px;color:#64748b">算子族数量</div>'
                f'<div style="font-size:24px;font-weight:800;margin-top:6px;'
                f'color:#0f172a">{len(all_fams)}</div></div>'
                f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:14px">'
                f'<div style="font-size:12px;color:#64748b">结论</div>'
                f'<div style="font-size:24px;font-weight:800;margin-top:6px" '
                f'class="mini-value {grade_c}">{grade}</div></div>'
                f'</div>'
                f'{method_bars}'
                f'<table style="font-size:13px;margin-top:8px">'
                f'<thead><tr>'
                f'<th>算子类型</th><th>时延 ms</th><th>占比</th><th>瓶颈</th>'
                f'<th>校准方法</th><th>置信度</th>'
                f'</tr></thead>'
                f'<tbody>{op_rows}</tbody>'
                f'</table>'
                f'<div style="color:#64748b;font-size:12px;margin-top:8px">'
                f'说明：算子级校准采用解析公式/经验模板方式；'
                f'报告显式给出覆盖率与时延加权置信度。</div>'
            )

    # ── References ──
    ref_html = ""
    if rc.references:
        ref_items = ""
        for ref in rc.references:
            title = ref.get("title", "")
            url = ref.get("url", "#")
            note = ref.get("note", "")
            ref_items += (
                f'<li>'
                f'<a href="{url}" rel="noopener noreferrer" target="_blank"'
                f' style="color:#2563eb">{title}</a>'
            )
            if note:
                ref_items += f' — <span style="color:#64748b">{note}</span>'
            ref_items += '</li>'
        ref_html = (
            f'<div class="section">'
            f'<h2>官方引用与校准来源</h2>'
            f'<ul style="color:#334155;line-height:1.8;font-size:13px">{ref_items}</ul>'
            f'</div>'
        )

    # ── Topology SVG ──
    topo_html = ""
    try:
        if hw_spec is not None or parallel is not None:
            from python.zrt.report.topology_renderer import render_topology_svg
            topo_svg = render_topology_svg(parallel=parallel, hw_spec=hw_spec)
            topo_html = (
                f'<div class="section">'
                f'<h2>逻辑集群拓扑图</h2>'
                f'{topo_svg}'
                f'<div style="color:#64748b;font-size:12px;margin-top:8px">'
                f'本图中：GPU 背景色表示 TP 并行域，GPU 边框颜色表示 EP 并行域；'
                f'节点内部展示 NVSwitch/NVLink。'
                f'</div>'
                f'</div>'
            )
    except Exception:
        topo_html = ""

    # ── Structure SVG ──
    struct_html = ""
    if rc.blocks:
        try:
            from python.zrt.report.structure_renderer import render_structure_html
            struct_html = render_structure_html(
                rc.blocks,
                phase=rc.phase,
                blocks_bwd=rc.blocks_bwd if rc.blocks_bwd else None,
            )
        except Exception:
            struct_html = ""

    # ── Assemble ──
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
{_CSS}
{_HIER_CSS}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">{''.join(f'<span>{p}</span>' for p in meta_parts)}</div>
<div class="cards">{cards_html}</div>
{bound_html}
<h2>模型 → Block → 子结构 → 算子分层展开</h2>
<div class="chart-container">{hierarchy_html}</div>
{topo_html}
{struct_html}
{timeline_html}
{calib_html}
{op_calib_html}
{ref_html}
{warning_html}
{_JS}
{_HIER_JS}
<script>{timeline_init}</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Exported hierarchical HTML report to %s", output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Unified report export (used by CLI inference + training pipelines)
# ─────────────────────────────────────────────────────────────────────────────


def export_reports(
    *,
    model: str,
    hardware: str,
    phase: str,
    batch_size: int,
    seq_len: int,
    graph: "OpGraph",
    hw_spec: "HardwareSpec",
    ctx: "TransformContext",
    output_dir: "Path",
    slug: str,
    profile: "Any | None" = None,
    memory_budget: "Any | None" = None,
    flat_summary: bool = False,
) -> "tuple[ReportContext, Any | None]":
    """Run schedule+simulate on a transformed graph, then export all report formats.

    Produces:
      - ``{slug}_{phase}_hier.html``  — 4-level hierarchical HTML
      - ``{slug}_{phase}_trace.json`` — Chrome Trace JSON
      - ``{slug}_{phase}_report.html`` — flat HTML (only if flat_summary=True)
      - ``{slug}_{phase}_e2e.json``    — E2E JSON dump (only if flat_summary=True)

    Parameters
    ----------
    model / hardware / phase / batch_size / seq_len
        Metadata for the report.
    graph : OpGraph
        Already-transformed graph (after pipeline.run).
    hw_spec : HardwareSpec
        Hardware specification for simulation.
    ctx : TransformContext
        Transform context with parallel config.
    output_dir : Path
        Directory to write report files into.
    slug : str
        File name prefix, e.g. ``"DeepSeek-V3_decode"``.
    profile : ModelProfile | None
        Optional model profile for param counts.
    memory_budget : MemoryBudget | None
        Optional memory breakdown.
    flat_summary : bool
        If True, also build E2ESummary and export flat HTML + E2E JSON.

    Returns
    -------
    tuple[ReportContext, E2ESummary | None]
        ReportContext (always) and E2ESummary (only if flat_summary=True).
    """
    from pathlib import Path as _Path

    from python.zrt.executor import DAGScheduler
    from python.zrt.simulator import SimulatorHub
    from python.zrt.report.report_builder import build_report_context as _build_rc
    from python.zrt.report.chrome_trace import export_chrome_trace
    from python.zrt.report.summary import E2ESummary, build_summary

    output_dir = _Path(output_dir)
    report_dir = output_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Schedule + Simulate ─────────────────────────────────────────────
    hub = SimulatorHub.default()
    scheduler = DAGScheduler(hw_spec=hw_spec)
    tl = scheduler.schedule(graph)
    sim_results = hub.simulate_graph(graph, hw_spec)

    # ── 2. Build timeline JSON (shared by all HTML exports) ────────────────
    # Enrich each entry with metadata for the interactive tooltip.
    timeline_json = []
    for op in tl.scheduled_ops:
        entry: dict = {
            "start":      op.start_us,
            "end":        op.end_us,
            "stream":     op.stream_id,
            "type":       op.stream_type,
            "op_type":    op.op_type,
            "latency_us": round(op.latency_us, 2),
            "category":   op.category,
            "phase":      op.phase,
        }
        node = graph.nodes.get(op.node_id)
        if node:
            entry["scope"] = node.scope or ""
            entry["layer"] = getattr(node, "layer", "") or ""
        sr = sim_results.get(op.node_id)
        if sr:
            entry["bound"] = sr.bound
            flops = getattr(sr, "flops", 0) or 0
            if flops:
                entry["tflops"] = round(flops / 1e12, 5)
        timeline_json.append(entry)

    # ── 3. Hierarchical ReportContext + HTML ──────────────────────────────
    rc = _build_rc(
        model=model, hardware=hardware, phase=phase,
        batch_size=batch_size, seq_len=seq_len,
        graph=graph, sim_results=sim_results, timeline=tl,
        hw_spec=hw_spec, ctx=ctx,
        profile=profile, memory_budget=memory_budget,
    )
    export_hierarchical_html_report(
        rc,
        report_dir / f"{slug}_{phase}_hier.html",
        timeline_data=timeline_json,
        hw_spec=hw_spec,
        parallel=getattr(ctx, "parallel", None) if ctx else None,
    )

    # ── 4. Chrome Trace ───────────────────────────────────────────────────
    parallel_desc = ctx.parallel.describe() if ctx else "single"
    export_chrome_trace(
        tl,
        report_dir / f"{slug}_{phase}_trace.json",
        name=f"{model} | {phase}",
        metadata={"model": model, "hardware": hardware,
                  "phase": phase, "parallel": parallel_desc},
    )

    # ── 5. Flat E2ESummary (inference path) ───────────────────────────────
    flat: E2ESummary | None = None
    if flat_summary:
        flat = build_summary(
            model=model, hardware=hardware, phase=phase,
            batch_size=batch_size, seq_len=seq_len,
            graph=graph, sim_results=sim_results, timeline=tl,
            hw_spec=hw_spec, parallel_desc=parallel_desc,
            memory_budget=memory_budget,
        )
        export_html_report(
            flat,
            report_dir / f"{slug}_{phase}_report.html",
            timeline_data=timeline_json,
        )

        # E2E JSON dump for debugging / automated processing
        import json as _json
        e2e_json = report_dir / f"{slug}_{phase}_e2e.json"
        e2e_json.write_text(_json.dumps({
            "model": model, "hardware": hardware, "phase": phase,
            "latency_ms": flat.latency_ms,
            "tokens_per_sec": flat.tokens_per_sec,
            "ttft_ms": flat.ttft_ms,
            "tpot_ms": flat.tpot_ms,
            "mfu": flat.mfu,
            "hbm_bw_util": flat.hbm_bandwidth_util,
        }, indent=2))

    return rc, flat
