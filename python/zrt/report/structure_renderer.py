"""Model structure SVG renderer (Section 4).

Generates an interactive SVG showing the model's high-level architecture
with module flow, repeat counts, and sub-structure annotations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.report.report_types import ReportContext, BlockDetail


# ── Color palette for module types ────────────────────────────────────────────

_MODULE_COLORS = {
    "input":        ("#f8fafc", "#cbd5e1"),
    "embedding":    ("#eff6ff", "#93c5fd"),
    "block":        ("#fefcff", "#d8b4fe"),
    "output":       ("#eef2ff", "#a5b4fc"),
}

_SUB_COLORS = {
    "norm":      ("#f8fafc", "#cbd5e1"),
    "attention": ("#ecfeff", "#67e8f9"),
    "residual":  ("#f1f5f9", "#cbd5e1"),
    "router":    ("#fff7ed", "#fdba74"),
    "dispatch":  ("#fef2f2", "#fca5a5"),
    "experts":   ("#eef2ff", "#a5b4fc"),
    "combine":   ("#fef2f2", "#fca5a5"),
    "shared":    ("#fdf2f8", "#f9a8d4"),
}


def _block_type(name: str) -> str:
    n = (name or "").lower()
    if "embed" in n or "tok_embeddings" in n:
        return "embedding"
    if "input" in n:
        return "input"
    if "output" in n or "lm_head" in n or "norm" in n:
        return "output"
    if "block" in n or "layer" in n or "moe" in n:
        return "block"
    return "block"


def render_structure_svg(
    blocks: "list[BlockDetail]",
    *,
    phase: str = "decode",
    width: int = 1200,
    height: int = 360,
) -> str:
    """Generate a model structure SVG.

    Renders three zones left-to-right:
        [Embedding]  →  [TransformerBlock × N (with sub-structure pills)]  →  [Output]

    Parameters
    ----------
    blocks : list[BlockDetail]
        Top-level model blocks from ReportContext.
    phase : str
        Phase label ("prefill" | "decode" | "train").
    width, height : int
        Minimum SVG canvas dimensions (auto-expanded for tall content).

    Returns
    -------
    str
        SVG markup ready to embed in HTML.
    """
    lines: list[str] = []

    def add(s: str) -> None:
        lines.append(s)

    # ── Classify blocks by semantic type ──────────────────────────────────────
    embed_blks  = [b for b in blocks if _block_type(b.name) == "embedding"]
    layer_blks  = [b for b in blocks if _block_type(b.name) == "block"]
    output_blks = [b for b in blocks if _block_type(b.name) == "output"]

    # Fallback when classification yields nothing
    if not layer_blks and not embed_blks and not output_blks:
        if len(blocks) >= 3:
            embed_blks  = [blocks[0]]
            layer_blks  = blocks[1:-1]
            output_blks = [blocks[-1]]
        elif len(blocks) == 2:
            layer_blks  = [blocks[0]]
            output_blks = [blocks[-1]]
        else:
            layer_blks = blocks

    # ── Layout constants ──────────────────────────────────────────────────────
    margin_x  = 26
    gap       = 22          # arrow gap between zones
    box_y     = 90          # top of boxes
    box_h_std = 130         # standard box height (embed / output)

    # Sub-structure pills inside the layer block
    sub_h, sub_gap = 24, 4
    sub_cols        = 3
    sub_pad_top     = 38    # room for block title inside box
    sub_pad_bot     = 32    # room for ms label at bottom

    # Arrow center Y (midpoint of the standard box)
    arrow_y = box_y + box_h_std // 2

    # ── X positions (left-to-right, variable-width per layer block) ──────────
    embed_w  = 155 if embed_blks  else 0
    output_w = 200 if output_blks else 0

    # Compute per-layer-block widths based on sub-structure count.
    # Each block gets its own width to accommodate its pill grid.
    _layer_widths: list[int] = []
    for b in layer_blks:
        _sc = len([ss for ss in b.sub_structures if ss.name]) or 1
        _cols = min(sub_cols, _sc)
        _w = max(200, 82 * _cols + 32)
        _layer_widths.append(_w)

    # Compute per-block X coords (stacked left-to-right with gap between).
    x = margin_x
    embed_x = x
    x += embed_w + (gap if embed_blks and (layer_blks or output_blks) else 0)

    _layer_xs: list[int] = []
    for i, _lw in enumerate(_layer_widths):
        _layer_xs.append(x)
        x += _lw + (gap if (i < len(_layer_widths) - 1) or output_blks else 0)

    output_x = x
    # Use first block for arrow_y (midpoint)
    layer_x = _layer_xs[0] if _layer_xs else x

    total_w = output_x + output_w + margin_x if output_blks else (output_x if not _layer_xs else _layer_xs[-1] + _layer_widths[-1] + margin_x)
    svg_w   = max(width,  total_w)
    # Compute max box height across all layer blocks (for SVG canvas sizing).
    _max_layer_h = 0
    for b in layer_blks:
        _sc = max(len([ss for ss in b.sub_structures if ss.name]), 1)
        _pr = (_sc + sub_cols - 1) // sub_cols
        _h = sub_pad_top + _pr * (sub_h + sub_gap) + sub_pad_bot
        _max_layer_h = max(_max_layer_h, _h)
    svg_h   = max(height, box_y + max(_max_layer_h, box_h_std) + 50)

    viewBox = f"0 0 {svg_w} {svg_h}"

    # ── Arrow marker def ──
    add('<defs>')
    add('<marker id="arrowHead" markerHeight="10" markerWidth="10" '
        'orient="auto" refX="7" refY="3">')
    add('<path d="M0,0 L0,6 L8,3 z" fill="#94a3b8"/>')
    add('</marker>')
    add('</defs>')

    # ── Background ──
    add(f'<rect fill="#fbfdff" height="{svg_h - 16}" rx="28" stroke="#dbe4ee" '
        f'width="{svg_w - 16}" x="8" y="8"/>')

    # ── Title ──
    phase_label = phase.upper() if phase else ""
    add(f'<text fill="#0f172a" font-size="18" font-weight="800" x="26" y="32">'
        f'Forward / Main Path — {phase_label}</text>')
    add(f'<text fill="#64748b" font-size="12" x="26" y="52">'
        f'模型主干结构图：Embedding → Transformer Blocks → Output</text>')

    def _draw_arrow(x1: int, x2: int) -> None:
        if x2 <= x1:
            return
        mid = (x1 + x2) // 2
        add(f'<line marker-end="url(#arrowHead)" stroke="#94a3b8" stroke-width="2.5" '
            f'x1="{x1}" x2="{x2 - 4}" y1="{arrow_y}" y2="{arrow_y}"/>')

    # ── Embedding block ───────────────────────────────────────────────────────
    if embed_blks:
        b = embed_blks[0]
        fill, stroke = _MODULE_COLORS["embedding"]
        add(f'<rect fill="{fill}" height="{box_h_std}" rx="20" stroke="{stroke}" '
            f'stroke-width="2" width="{embed_w}" x="{embed_x}" y="{box_y}"/>')
        add(f'<text fill="#0f172a" font-size="16" font-weight="800" '
            f'x="{embed_x + 14}" y="{box_y + 26}">Embedding</text>')
        add(f'<text fill="#475569" font-size="11" '
            f'x="{embed_x + 14}" y="{box_y + 46}">Vocab → Hidden</text>')
        add(f'<text fill="#0f172a" font-size="19" font-weight="700" '
            f'x="{embed_x + 14}" y="{box_y + box_h_std - 18}">{b.total_ms:.3f} ms</text>')
        if layer_blks or output_blks:
            _draw_arrow(embed_x + embed_w, layer_x if layer_blks else output_x)

    # ── Layer block(s) ────────────────────────────────────────────────────────
    for li, b in enumerate(layer_blks):
        lx = _layer_xs[li]
        lw = _layer_widths[li]

        # Per-block box height based on its own sub-structure count
        _b_sub_count = max(
            len([ss for ss in b.sub_structures if ss.name]), 1)
        _b_pills_rows = (_b_sub_count + sub_cols - 1) // sub_cols
        _b_box_h = sub_pad_top + _b_pills_rows * (sub_h + sub_gap) + sub_pad_bot

        fill, stroke = _MODULE_COLORS["block"]
        repeat_label = f" × {b.repeat}" if b.repeat > 1 else ""

        # Block box
        add(f'<rect fill="{fill}" height="{_b_box_h}" rx="20" stroke="{stroke}" '
            f'stroke-width="2" width="{lw}" x="{lx}" y="{box_y}"/>')
        add(f'<text fill="#0f172a" font-size="16" font-weight="800" '
            f'x="{lx + 14}" y="{box_y + 24}">{b.name}{repeat_label}</text>')

        # Model-specific params (MoE experts)
        if b.model_params:
            pm = b.model_params
            parts: list[str] = []
            if pm.get("num_experts"):
                parts.append(f"experts={pm['num_experts']}")
            if pm.get("active_per_token"):
                parts.append(f"top-k={pm['active_per_token']}")
            if parts:
                add(f'<text fill="#475569" font-size="10" '
                    f'x="{lx + 14}" y="{box_y + 38}">{" · ".join(parts)}</text>')

        # Sub-structure pills
        pill_w   = max(78, (lw - 28 - (sub_cols - 1) * 8) // sub_cols)
        pill_x0  = lx + 14
        pill_y0  = box_y + sub_pad_top

        col_i, row_i = 0, 0
        for ss in b.sub_structures:
            if not ss.name:
                continue
            name_lower = ss.name.lower()
            sub_cat = "attention"
            if "norm" in name_lower:
                sub_cat = "norm"
            elif "attn" in name_lower or "attention" in name_lower:
                sub_cat = "attention"
            elif "residual" in name_lower or "add" in name_lower:
                sub_cat = "residual"
            elif "router" in name_lower or "gate" in name_lower:
                sub_cat = "router"
            elif "dispatch" in name_lower:
                sub_cat = "dispatch"
            elif "expert" in name_lower or "ffn" in name_lower or "mlp" in name_lower:
                sub_cat = "experts"
            elif "combine" in name_lower:
                sub_cat = "combine"
            elif "shared" in name_lower:
                sub_cat = "shared"

            sub_fill, sub_stroke = _SUB_COLORS.get(sub_cat, ("#f8fafc", "#cbd5e1"))
            px = pill_x0 + col_i * (pill_w + 8)
            py = pill_y0 + row_i * (sub_h + sub_gap)

            if py + sub_h > box_y + _b_box_h - sub_pad_bot:
                break  # out of vertical space

            add(f'<rect fill="{sub_fill}" height="{sub_h}" rx="11" '
                f'stroke="{sub_stroke}" stroke-width="1.5" width="{pill_w}" '
                f'x="{px}" y="{py}"/>')
            label = ss.name[:11] if len(ss.name) > 11 else ss.name
            add(f'<text fill="#334155" font-size="10.5" font-weight="700" '
                f'x="{px + 8}" y="{py + 15}">{label}</text>')

            col_i += 1
            if col_i >= sub_cols:
                col_i = 0
                row_i += 1

        add(f'<text fill="#0f172a" font-size="19" font-weight="700" '
            f'x="{lx + 14}" y="{box_y + _b_box_h - 12}">'
            f'{b.total_ms:.3f} ms</text>')

        # Arrow: between layer blocks, or from last layer to output
        if li < len(layer_blks) - 1:
            _next_x = _layer_xs[li + 1]
            _draw_arrow(lx + lw, _next_x)
        elif output_blks:
            _draw_arrow(lx + lw, output_x)
            mid_x = (lx + lw + output_x) // 2
            add(f'<text fill="#64748b" font-size="10" text-anchor="middle" '
                f'x="{mid_x}" y="{arrow_y - 8}">residual</text>')

    # ── Output block ──────────────────────────────────────────────────────────
    if output_blks:
        b = output_blks[0]
        fill, stroke = _MODULE_COLORS["output"]
        add(f'<rect fill="{fill}" height="{box_h_std}" rx="20" stroke="{stroke}" '
            f'stroke-width="2" width="{output_w}" x="{output_x}" y="{box_y}"/>')
        add(f'<text fill="#0f172a" font-size="16" font-weight="800" '
            f'x="{output_x + 14}" y="{box_y + 26}">Output</text>')
        add(f'<text fill="#475569" font-size="11" '
            f'x="{output_x + 14}" y="{box_y + 46}">Final Norm + LM Head</text>')
        add(f'<text fill="#0f172a" font-size="19" font-weight="700" '
            f'x="{output_x + 14}" y="{box_y + box_h_std - 18}">{b.total_ms:.3f} ms</text>')

    return (
        f'<svg class="arch-svg" viewBox="{viewBox}" xmlns="http://www.w3.org/2000/svg">\n'
        + "\n".join(lines)
        + "\n</svg>"
    )


def render_structure_html(blocks, phase="decode", blocks_bwd=None) -> str:
    """Render the full structure HTML section with SVG and tab UI.

    Parameters
    ----------
    blocks : list[BlockDetail]
        Top-level forward (or combined) model blocks.
    phase : str
        Phase label.
    blocks_bwd : list[BlockDetail] | None
        Backward-only blocks.  When provided the Backward tab renders a real
        SVG instead of the placeholder.

    Returns
    -------
    str
        Complete HTML for the structure section.
    """
    svg_fwd = render_structure_svg(blocks, phase=phase)

    if blocks_bwd:
        svg_bwd = render_structure_svg(blocks_bwd, phase=f"{phase} backward")
        bwd_panel = f"""
    <div class="viz-panel" id="viz-backward" style="display:none">
        {svg_bwd}
    </div>"""
    else:
        bwd_panel = """
    <div class="viz-panel" id="viz-backward" style="display:none;padding:40px;
         text-align:center;background:#f8fafc;border:1px solid #e2e8f0;
         border-radius:0 8px 8px 8px;color:#64748b;font-size:14px">
        Backward 结构视图将在支持逆向图捕获后可用。<br>
        <small style="color:#94a3b8">需要 train_backward 阶段的 GraphHierarchy 数据</small>
    </div>"""

    return f"""
<div class="section">
    <h2>模型结构（交互式 SVG 视图）</h2>
    <div class="viz-tabs" style="display:flex;gap:0;margin-bottom:0">
        <button class="viz-tab active" onclick="switchViz('viz-forward',this)"
         style="padding:8px 20px;border:1px solid #3b82f6;background:#3b82f6;color:#fff;
         border-radius:8px 8px 0 0;cursor:pointer;font-size:13px;font-weight:600">
         Forward 结构视图</button>
        <button class="viz-tab" onclick="switchViz('viz-backward',this)"
         style="padding:8px 20px;border:1px solid #e2e8f0;background:#f8fafc;color:#64748b;
         border-radius:8px 8px 0 0;cursor:pointer;font-size:13px;font-weight:600;
         border-bottom:none">
         Backward 结构视图</button>
    </div>
    <div class="viz-panel active" id="viz-forward" style="display:block">
        {svg_fwd}
    </div>
{bwd_panel}
    <div style="color:#64748b;font-size:12px;margin-top:8px">
        点击上方标签即可切换结构视图。SVG 图中使用连线显式表示主路径，适合培训、评审和报告演示。
    </div>
</div>"""
