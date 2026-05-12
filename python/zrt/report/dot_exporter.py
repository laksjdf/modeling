"""Export OpGraph to Graphviz DOT format for visualization."""
from __future__ import annotations

import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.ir.node import OpNode

_CATEGORY_COLOR: dict[str, str] = {
    "compute":       "#4A90D9",
    "communication": "#E05252",
    "memory":        "#E09A52",
}
_DEFAULT_COLOR = "#AAAAAA"


def _escape(s: str) -> str:
    """Escape a string for DOT double-quoted attribute values."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _shape_str(shape: tuple[int, ...]) -> str:
    """Format a shape tuple as ``[dim,dim,...]``."""
    return "[" + ",".join(str(d) for d in shape) + "]"


def _node_label(node: "OpNode") -> str:
    """Build a DOT label: ``Component\\n[in] → [out]``."""
    parts = [_escape(node.component or node.op_type)]
    inp = node.input_shapes()
    out = node.output_shapes()
    if inp and out:
        parts.append(f"{_shape_str(inp[0])} → {_shape_str(out[0])}")
    # \\n in Python source → literal \n in file → newline in DOT renderer
    return "\\n".join(parts)


def export_dot(
    graph: "OpGraph",
    output_path: Path,
    title: str = "",
) -> Path:
    """Write an OpGraph to a Graphviz DOT file.

    Parameters
    ----------
    graph : OpGraph
        Any OpGraph — inference, training-forward, training-backward, or stitched.
    output_path : Path
        Destination ``.dot`` file path.
    title : str
        Optional graph title shown as label.

    Returns
    -------
    Path
        Path to the written ``.dot`` file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("digraph {")
    lines.append(
        '  graph [rankdir=TB, fontname="Helvetica", '
        'splines=ortho, concentrate=true]'
    )
    lines.append(
        '  node  [shape=box, style="filled,rounded", '
        'fontname="Helvetica", fontsize=10]'
    )
    lines.append("  edge  [arrowsize=0.7]")

    if title:
        lines.append(f'  label="{_escape(title)}"')

    # ── cluster nodes by layer ──────────────────────────────────────────────
    layer_nodes: dict[str, list] = defaultdict(list)
    for node in graph.nodes.values():
        key = node.layer if node.layer else "__other__"
        layer_nodes[key].append(node)

    for layer_key in sorted(layer_nodes.keys()):
        nodes = layer_nodes[layer_key]
        if layer_key == "__other__":
            cluster_name = "cluster_other"
            cluster_label = "Other"
        else:
            cluster_name = f"cluster_layer{layer_key}"
            cluster_label = f"Layer {layer_key}"

        lines.append(f"  subgraph {cluster_name} {{")
        lines.append(
            f'    label="{_escape(cluster_label)}"; '
            f'style=dashed; color="#888888";'
        )

        for node in nodes:
            color = _CATEGORY_COLOR.get(node.category, _DEFAULT_COLOR)
            label = _node_label(node)
            fontcolor = "white" if color != _DEFAULT_COLOR else "black"
            nid = _escape(node.id)
            lines.append(
                f'    "{nid}" [label="{label}", '
                f'fillcolor="{color}", fontcolor={fontcolor}];'
            )

        lines.append("  }")

    # ── edges (after clusters to avoid rendering artifacts) ──────────────────
    for edge in graph.edges:
        lines.append(f'  "{_escape(edge.src)}" -> "{_escape(edge.dst)}"')

    lines.append("}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


_RENDER_DOT_TIMEOUT_SECONDS = 30
"""Hard wall-clock cap on a single ``dot`` invocation.

Graphviz's layout (especially with ``splines=ortho``) is super-linear in
node count and can run for hours on a 2k-node graph.  Production
pipelines call ``render_dot`` on stitched fwd+bwd graphs without ever
inspecting the SVG, so we'd rather skip the render than hang.  The cap
is large enough that real diagnostic graphs (≤ a few hundred nodes)
still render successfully on a typical laptop.
"""


def render_dot(dot_path: Path, format: str = "svg") -> Path | None:
    """Render a ``.dot`` file to SVG/PDF via the ``dot`` CLI.

    Returns ``None`` silently when Graphviz is not installed, when the
    rendering exceeds :data:`_RENDER_DOT_TIMEOUT_SECONDS`, or when ``dot``
    exits non-zero.  These failure modes are non-fatal — the caller
    treats the .dot file as the canonical artefact.
    """
    if not shutil.which("dot"):
        return None
    out = dot_path.with_suffix(f".{format}")
    try:
        subprocess.run(
            ["dot", f"-T{format}", str(dot_path), "-o", str(out)],
            check=True,
            capture_output=True,
            timeout=_RENDER_DOT_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return None
    return out
