"""Export performance reports to interactive HTML (zero external dependencies)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.report.summary import E2ESummary, TrainingSummary

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
</style>
"""

# ── JS ────────────────────────────────────────────────────────────────────────

_JS = """
<script>
// ── Timeline chart ──
function drawTimeline(canvasId, ops) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !ops.length) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;
    const pad = {top: 20, right: 20, bottom: 30, left: 60};
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const minT = Math.min(...ops.map(o => o.start));
    const maxT = Math.max(...ops.map(o => o.end));
    const range = maxT - minT || 1;
    const streams = [...new Set(ops.map(o => o.stream))];
    const streamH = plotH / streams.length;

    const colors = {compute: "#4CAF50", comm: "#F44336", memory: "#FF9800"};

    ops.forEach(op => {
        const x = pad.left + ((op.start - minT) / range) * plotW;
        const w = Math.max(1, ((op.end - op.start) / range) * plotW);
        const si = streams.indexOf(op.stream);
        const y = pad.top + si * streamH + 4;
        const h = streamH - 8;
        ctx.fillStyle = colors[op.type] || "#90A4AE";
        ctx.fillRect(x, y, w, h);
    });

    // Y-axis labels
    ctx.fillStyle = "#333"; ctx.font = "11px sans-serif"; ctx.textAlign = "right";
    streams.forEach((s, i) => {
        ctx.fillText("Stream " + s, pad.left - 6, pad.top + i * streamH + streamH / 2 + 4);
    });

    // X-axis labels
    ctx.textAlign = "center";
    for (let i = 0; i <= 5; i++) {
        const t = minT + (range * i / 5);
        const x = pad.left + plotW * i / 5;
        ctx.fillText((t / 1000).toFixed(1) + "ms", x, H - 8);
    }
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
    """Export an interactive HTML performance report.

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
    timeline_html = ""
    if timeline_data:
        ops_json = json.dumps(timeline_data)
        timeline_html = f"""
<h2>Timeline</h2>
<div class="chart-container">
    <canvas id="timeline"></canvas>
</div>
<script>drawTimeline("timeline", {ops_json});</script>
"""

    # ── Component pie chart ──
    pie_html = ""
    if summary.by_component:
        pie_data = [{"label": k, "value": v} for k, v in
                     sorted(summary.by_component.items(), key=lambda x: -x[1])]
        pie_json = json.dumps(pie_data)
        pie_html = f"""
<h2>Component Breakdown</h2>
<div class="chart-container">
    <canvas id="pie"></canvas>
</div>
<script>drawPie("pie", {pie_json});</script>
"""

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
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Exported HTML report to %s", output_path)
    return output_path
