"""Logical cluster topology SVG renderer.

Renders a GPU grid with TP/EP coloring, NVSwitch/NVLink connections,
and inter-node Spine/Leaf fabric — matching the target report's Section 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.transform.context import ParallelConfig
    from python.zrt.hardware.spec import HardwareSpec


# ── Color palettes ────────────────────────────────────────────────────────────

_TP_COLORS = ["#dbeafe", "#ede9fe"]  # TP0 light-blue, TP1 light-violet
_EP_STROKES = [
    "#2563eb", "#7c3aed", "#059669", "#ea580c",
    "#dc2626", "#0891b2", "#db2777", "#65a30d",
]


def _tp_fill(tp_idx: int) -> str:
    return _TP_COLORS[tp_idx % len(_TP_COLORS)]


def _ep_stroke(ep_idx: int) -> str:
    return _EP_STROKES[ep_idx % len(_EP_STROKES)]


# ── Public API ────────────────────────────────────────────────────────────────


def render_topology_svg(
    parallel: "ParallelConfig | None" = None,
    hw_spec: "HardwareSpec | None" = None,
    *,
    nodes: int = 2,
    gpus_per_node: int = 8,
    world_size: int | None = None,
) -> str:
    """Generate a logical cluster topology SVG string.

    Parameters
    ----------
    parallel : ParallelConfig | None
        Parallel config with TP/EP/DP/PP/CP degrees.
        If None, defaults to TP=1, EP=1, DP=1, PP=1, CP=1.
    hw_spec : HardwareSpec | None
        Hardware spec for interconnect bandwidth labels.
    nodes : int
        Number of physical nodes (default 2).
    gpus_per_node : int
        GPUs per node (default 8).
    world_size : int | None
        Total device count. Auto-computed from parallel if None.

    Returns
    -------
    str
        SVG markup ready to embed in HTML.
    """
    tp = getattr(parallel, "tp", 1) if parallel else 1
    ep = getattr(parallel, "ep", 1) if parallel else 1
    dp = getattr(parallel, "dp", 1) if parallel else 1
    pp = getattr(parallel, "pp", 1) if parallel else 1
    cp = getattr(parallel, "cp", 1) if parallel else 1

    if world_size is None:
        world_size = tp * ep * dp * pp * cp

    total_gpus = nodes * gpus_per_node

    # Bandwidth labels
    if hw_spec and hasattr(hw_spec, "interconnect"):
        intra_bw = hw_spec.interconnect.intra_node.bandwidth_gbps
        inter_bw = hw_spec.interconnect.inter_node.bandwidth_gbps
    else:
        intra_bw = 16000
        inter_bw = 16000

    # Layout constants
    gpu_w, gpu_h = 43, 18
    gpu_gap_x, gpu_gap_y = 6, 5
    cols_per_row = 4  # GPUs per row within a node
    node_pad_x = 16
    rows = max(1, gpus_per_node // cols_per_row)

    node_w = node_pad_x * 2 + cols_per_row * gpu_w + (cols_per_row - 1) * gpu_gap_x

    # Vertical layout (top-to-bottom inside a node box):
    #   title_h   — "Node N" + bandwidth label
    #   gpu_area  — all GPU rows
    #   nvs_gap   — spacing before NVSwitch
    #   nvs_h     — NVSwitch bar
    #   bottom    — padding below NVSwitch
    title_h  = 36
    gpu_area = rows * gpu_h + max(0, rows - 1) * gpu_gap_y
    nvs_gap  = 10
    nvs_h    = 16
    bot_pad  = 10
    node_h   = title_h + gpu_area + nvs_gap + nvs_h + bot_pad

    # Y offsets relative to node_y
    gpu_y0   = title_h                        # first GPU row top
    nvs_y_off = title_h + gpu_area + nvs_gap  # NVSwitch top

    rack_w = node_w * nodes + (nodes + 1) * 26
    # Rack contains a 38px top margin (for rack title) + node content + 16px bottom
    rack_h = 38 + node_h + 16

    svg_w = rack_w + 200
    svg_h = 96 + rack_h + 40  # 96px header area + rack + footer
    viewBox = f"0 0 {svg_w} {svg_h}"

    lines: list[str] = []

    def add(s: str) -> None:
        lines.append(s)

    # ── Background ──
    add(f'<rect fill="#f8fbff" height="{svg_h - 8}" rx="20" stroke="#d8e3f0" '
        f'width="{svg_w - 16}" x="8" y="8"/>')

    # ── Title ──
    add(f'<text fill="#0f172a" font-size="19" font-weight="700" x="24" y="34">'
        f'逻辑集群拓扑连接图（TP/EP 并行域着色）</text>')
    add(f'<text fill="#475569" font-size="12" x="24" y="56">'
        f'World={world_size} · Nodes={nodes} · GPUs/Node={gpus_per_node} · '
        f'Intra-node NVLink/NVSwitch≈{intra_bw} GB/s · '
        f'Interconnect≈{inter_bw} Gb/s</text>')

    # ── Legend ──
    lx = 24
    add(f'<text fill="#334155" font-size="12" font-weight="700" x="{lx}" y="76">TP 组着色</text>')
    for ti in range(min(tp, 2)):
        add(f'<rect fill="{_tp_fill(ti)}" height="10" rx="3" stroke="#94a3b8" '
            f'width="18" x="{lx + 74 + ti * 38}" y="66"/>')
        add(f'<text fill="#475569" font-size="10" x="{lx + 96 + ti * 38}" y="76">TP{ti}</text>')
    add(f'<text fill="#334155" font-size="12" font-weight="700" x="{lx + 198}" y="76">EP 组边框</text>')
    for ei in range(min(ep, 8)):
        bx = lx + 278 + ei * 38
        add(f'<rect fill="#ffffff" height="10" rx="3" stroke="{_ep_stroke(ei)}" '
            f'stroke-width="2" width="18" x="{bx}" y="66"/>')
        add(f'<text fill="#475569" font-size="10" x="{bx + 22}" y="76">EP{ei}</text>')

    # ── Rack ──
    rack_x = 24
    rack_y = 96
    add(f'<rect fill="#ffffff" height="{rack_h}" rx="16" stroke="#cbd5e1" '
        f'width="{rack_w}" x="{rack_x}" y="{rack_y}"/>')
    add(f'<text fill="#0f172a" font-size="14" font-weight="700" x="{rack_x + 16}" '
        f'y="{rack_y + 24}">Rack 0 / 机框 0</text>')

    # ── Spine / Leaf switch ──
    spine_x = rack_x + rack_w + 10
    spine_y = rack_y + 24
    add(f'<rect fill="#eef2ff" height="34" rx="10" stroke="#818cf8" width="120" '
        f'x="{spine_x}" y="{spine_y - 12}"/>')
    add(f'<text fill="#3730a3" font-size="12" font-weight="700" x="{spine_x + 12}" '
        f'y="{spine_y + 2}">Spine / Leaf</text>')
    add(f'<text fill="#4f46e5" font-size="10" x="{spine_x + 12}" '
        f'y="{spine_y + 14}">{inter_bw} Gb/s</text>')

    # ── Nodes ──
    for node_i in range(nodes):
        node_x = rack_x + 24 + node_i * (node_w + 26)
        node_y = rack_y + 38  # below rack title

        # Node background
        add(f'<rect fill="#f8fafc" height="{node_h}" rx="12" stroke="#cbd5e1" '
            f'width="{node_w}" x="{node_x}" y="{node_y}"/>')
        add(f'<text fill="#0f172a" font-size="12" font-weight="700" '
            f'x="{node_x + 8}" y="{node_y + 14}">Node {node_i}</text>')
        add(f'<text fill="#475569" font-size="9" '
            f'x="{node_x + 8}" y="{node_y + 26}">'
            f'NVSwitch / NVLink {intra_bw} GB/s</text>')

        # NVSwitch bar — positioned after all GPU rows
        nvs_x = node_x + (node_w - 80) // 2
        nvs_y = node_y + nvs_y_off
        add(f'<rect fill="#dcfce7" height="{nvs_h}" rx="7" stroke="#22c55e" width="80" '
            f'x="{nvs_x}" y="{nvs_y}"/>')
        add(f'<text fill="#166534" font-size="9" x="{nvs_x + 12}" '
            f'y="{nvs_y + 11}">NV Fabric</text>')

        # Inter-node connection line (midpoint of node box)
        add(f'<line stroke="#94a3b8" stroke-dasharray="4 3" '
            f'x1="{node_x + node_w}" x2="{spine_x}" '
            f'y1="{node_y + node_h // 2}" y2="{spine_y}"/>')

        # GPUs
        for gpu_i in range(gpus_per_node):
            global_gpu = node_i * gpus_per_node + gpu_i
            col = gpu_i % cols_per_row
            row = gpu_i // cols_per_row
            gx = node_x + node_pad_x + col * (gpu_w + gpu_gap_x)
            gy = node_y + gpu_y0 + row * (gpu_h + gpu_gap_y)

            # TP → fill, EP → border
            tp_idx = (global_gpu // ep) % tp if tp > 0 else 0
            ep_idx = global_gpu % ep if ep > 0 else 0

            # D/P/T/E/C annotation: D and P always 0 unless DP/PP > 1
            d_idx = global_gpu % dp if dp > 0 else 0
            p_idx = (global_gpu // (tp * ep * cp)) % pp if pp > 0 else 0
            c_idx = (global_gpu // ep) % cp if cp > 0 else 0

            fill = _tp_fill(tp_idx)
            stroke = _ep_stroke(ep_idx)

            add(f'<rect fill="{fill}" height="{gpu_h}" rx="4" '
                f'stroke="{stroke}" stroke-width="2" width="{gpu_w}" '
                f'x="{gx}" y="{gy}"/>')
            add(f'<text fill="#0f172a" font-size="6.5" font-weight="700" '
                f'x="{gx + 3}" y="{gy + 8}">GPU{gpu_i}/R{global_gpu}</text>')
            add(f'<text fill="#334155" font-size="5.8" '
                f'x="{gx + 3}" y="{gy + 15}">'
                f'D{d_idx} P{p_idx} T{tp_idx} E{ep_idx} C{c_idx}</text>')

            # NVLink connection from GPU to NVSwitch
            add(f'<line stroke="#86efac" '
                f'x1="{gx + gpu_w // 2}" x2="{nvs_x + 37}" '
                f'y1="{gy + gpu_h}" y2="{nvs_y}"/>')

    return f'<svg class="topo-svg" viewBox="{viewBox}" xmlns="http://www.w3.org/2000/svg">\n' + \
           "\n".join(lines) + "\n</svg>"
