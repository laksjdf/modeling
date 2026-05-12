"""Rich DOT renderers for the layer-0 visualization test.

Three renderers, all producing ``.dot`` + ``.svg`` files via the system
``dot`` binary:

* :func:`render_module_tree` — model architecture (nn.Module hierarchy).
* :func:`render_raw_op_graph` — post ``records_to_opgraph`` aten DAG with
  scope-based nested subgraphs and shape-labelled edges.
* :func:`render_fused_op_graph` — same layout but highlighting fused
  nodes vs raw aten passthroughs.

Designed to be visually scannable on a single page even with several
hundred aten ops; uses ``concentrate=true`` + ``splines=ortho`` and
collapses scope siblings into clusters.
"""
from __future__ import annotations

import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# Distinct colours per semantic component.
_COMPONENT_COLOR: dict[str, str] = {
    "embedding":       "#9C27B0",
    "final_norm":      "#00838F",
    "attn.score":      "#1565C0",
    "attn.abs":        "#1976D2",
    "attn.amax":       "#1976D2",
    "attn.clamp_min":  "#1976D2",
    "attn.div":        "#1976D2",
    "attn.add":        "#42A5F5",
    "attn.mul":        "#42A5F5",
    "attn.pow":        "#42A5F5",
    "attn.mean":       "#42A5F5",
    "attn.rsqrt":      "#42A5F5",
    "attn.cat":        "#0D47A1",
    "attn.bmm":        "#0D47A1",
    "attn._softmax":   "#0D47A1",
    "attn.arange":     "#90CAF9",
    "attn.sub":        "#90CAF9",
    "attn.clamp":      "#90CAF9",
    "attn.gt":         "#90CAF9",
    "attn.scalar_tensor": "#90CAF9",
    "hc.pre_attn":     "#7B1FA2",
    "hc.post_attn":    "#7B1FA2",
    "hc.pre_ffn":      "#7B1FA2",
    "hc.post_ffn":     "#7B1FA2",
    "moe_gate":        "#EF6C00",
    "moe_expert":      "#F57C00",
    "moe_dispatch":    "#FB8C00",
    "kv_compressor":   "#43A047",
    "sparse_indexer":  "#388E3C",
    "mm":              "#5C6BC0",
}
_FALLBACK = "#9E9E9E"
_COMM_COLOR = "#E53935"
_FUSED_BORDER = "#FFB300"


def _esc(s: Any) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _shape(t: Any) -> str:
    sh = getattr(t, "shape", None)
    if sh is None:
        return ""
    return "[" + ",".join(str(d) for d in sh) + "]"


def _dtype(t: Any) -> str:
    dt = getattr(t, "dtype", None)
    if dt is None:
        return ""
    val = getattr(dt, "value", str(dt))
    return val.replace("torch.", "")


def _component_color(node: Any, default: str = _FALLBACK) -> str:
    if getattr(node, "category", "") == "communication":
        return _COMM_COLOR
    return _COMPONENT_COLOR.get(getattr(node, "component", ""), default)


def _scope_path(scope: str) -> list[str]:
    """Split ``transformer.layers.0.attn.q_norm`` into a list of nested keys."""
    if not scope:
        return ["__module_level__"]
    return scope.split(".")


def _write_dot_and_render(dot_path: Path, dot_text: str,
                          formats: tuple[str, ...] = ("svg", "png")) -> dict[str, Path]:
    dot_path = Path(dot_path)
    dot_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path.write_text(dot_text, encoding="utf-8")
    out = {"dot": dot_path}
    if not shutil.which("dot"):
        return out
    for fmt in formats:
        target = dot_path.with_suffix(f".{fmt}")
        try:
            subprocess.run(
                ["dot", f"-T{fmt}", str(dot_path), "-o", str(target)],
                check=True, capture_output=True, timeout=120,
            )
            out[fmt] = target
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            stderr = getattr(exc, "stderr", b"")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="ignore")
            print(f"[warn] dot -T{fmt} failed for {dot_path.name}: {stderr[:200]}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1) Module tree — original model architecture
# ─────────────────────────────────────────────────────────────────────────────

def render_module_tree(model: Any, output_path: Path,
                       max_depth: int = 6,
                       scope_filter: str | None = None,
                       title: str = "") -> dict[str, Path]:
    """Render ``model.named_modules()`` as a hierarchical tree.

    ``scope_filter`` keeps only modules whose name starts with the prefix
    (e.g. ``"transformer.layers.0"`` to focus on layer 0).
    """
    nodes: list[tuple[str, str]] = []  # (name, class_name)
    edges: list[tuple[str, str]] = []
    seen: set[str] = set()

    for name, mod in model.named_modules():
        if scope_filter and not (name == scope_filter or
                                 name.startswith(scope_filter + ".") or
                                 scope_filter.startswith(name + ".") or
                                 name == ""):
            continue
        if name.count(".") > max_depth:
            continue
        cls = type(mod).__name__
        nodes.append((name, cls))
        seen.add(name)

    name_set = set(n for n, _ in nodes)
    for name, _ in nodes:
        if "." in name:
            parent = name.rsplit(".", 1)[0]
            if parent in name_set:
                edges.append((parent, name))

    lines: list[str] = ["digraph module_tree {"]
    lines.append('  graph [rankdir=LR, fontname="Helvetica", concentrate=true, '
                 'ranksep=0.6, nodesep=0.2]')
    lines.append('  node  [shape=box, style="filled,rounded", '
                 'fontname="Helvetica", fontsize=11]')
    lines.append('  edge  [arrowsize=0.6, color="#888888"]')
    if title:
        lines.append(f'  label="{_esc(title)}"; labelloc=t; fontsize=14;')

    palette = ["#E1F5FE", "#F3E5F5", "#FFF3E0", "#E8F5E9", "#FCE4EC", "#F1F8E9"]

    for name, cls in nodes:
        depth = name.count(".") if name else 0
        color = palette[depth % len(palette)]
        label = f"{name.rsplit('.', 1)[-1] if name else '<root>'}\\n[{cls}]"
        nid = name or "__root__"
        lines.append(f'  "{_esc(nid)}" [label="{label}", fillcolor="{color}"];')

    for parent, child in edges:
        lines.append(f'  "{_esc(parent)}" -> "{_esc(child)}";')
    lines.append("}")
    return _write_dot_and_render(output_path, "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# 2) Raw aten OpGraph (post records_to_opgraph)
# ─────────────────────────────────────────────────────────────────────────────

def _build_scope_clusters(graph: Any) -> tuple[list[str], dict[str, list[str]]]:
    """Return (scope_paths_in_topo_order, scope→[node_id]).

    A "scope path" is the dotted prefix used to nest subgraphs in DOT.
    """
    scope_to_nodes: dict[str, list[str]] = defaultdict(list)
    for node in graph.topo_sort():
        key = node.scope or "__top__"
        scope_to_nodes[key].append(node.id)
    return list(scope_to_nodes.keys()), scope_to_nodes


def render_raw_op_graph(graph: Any, output_path: Path,
                        title: str = "",
                        max_edge_label_chars: int = 24,
                        cluster_by_scope: bool = True) -> dict[str, Path]:
    """Render an OpGraph (typically post-records_to_opgraph) as DOT/SVG.

    Nodes are clustered by ``scope`` to mirror the nn.Module hierarchy.
    Edge labels carry ``shape × dtype`` for the data tensor.
    """
    lines: list[str] = ["digraph raw_aten_graph {"]
    lines.append('  graph [rankdir=TB, fontname="Helvetica", '
                 'concentrate=true, splines=spline, ranksep=0.4, nodesep=0.25]')
    lines.append('  node  [shape=box, style="filled,rounded", '
                 'fontname="Helvetica", fontsize=10]')
    lines.append('  edge  [arrowsize=0.7, fontsize=8, fontname="Helvetica", '
                 'color="#666666", fontcolor="#444444"]')
    if title:
        lines.append(f'  label="{_esc(title)}"; labelloc=t; fontsize=14;')

    if cluster_by_scope:
        _, scope_to_nodes = _build_scope_clusters(graph)
        for scope, nids in scope_to_nodes.items():
            label = scope if scope != "__top__" else "<top-level>"
            cluster_id = f"cluster_{abs(hash(scope)) % 10**9}"
            lines.append(f'  subgraph {cluster_id} {{')
            lines.append(
                f'    label="{_esc(label)}"; style="filled,rounded"; '
                'color="#cccccc"; fillcolor="#fafafa"; fontsize=11;'
            )
            for nid in nids:
                node = graph.nodes[nid]
                color = _component_color(node, "#BBDEFB")
                op_label = node.op_short or (node.op_type.split(".")[-2]
                                              if "." in node.op_type
                                              else node.op_type)
                rec_id = nid.replace("op_", "")
                short_op = node.op_type.replace("aten.", "").replace(".default", "")
                ann = []
                if getattr(node, "call_id", 0):
                    ann.append(f"cid={node.call_id}")
                ann_str = " " + " ".join(ann) if ann else ""
                lbl = f"#{rec_id} {short_op}{ann_str}"
                lines.append(
                    f'    "{_esc(nid)}" [label="{_esc(lbl)}", '
                    f'fillcolor="{color}", fontcolor="white"];'
                )
            lines.append("  }")
    else:
        for nid, node in graph.nodes.items():
            color = _component_color(node, "#BBDEFB")
            short_op = node.op_type.replace("aten.", "").replace(".default", "")
            lines.append(
                f'  "{_esc(nid)}" [label="{_esc(short_op)}", '
                f'fillcolor="{color}", fontcolor="white"];'
            )

    for e in graph.edges:
        if e.src not in graph.nodes or e.dst not in graph.nodes:
            continue
        tensor = e.tensor
        sh = _shape(tensor)
        dt = _dtype(tensor)
        elabel = sh
        if dt:
            elabel = f"{sh} {dt}" if sh else dt
        if len(elabel) > max_edge_label_chars:
            elabel = elabel[: max_edge_label_chars - 1] + "…"
        if elabel:
            lines.append(
                f'  "{_esc(e.src)}" -> "{_esc(e.dst)}" [label="{_esc(elabel)}"];'
            )
        else:
            lines.append(f'  "{_esc(e.src)}" -> "{_esc(e.dst)}";')
    lines.append("}")
    return _write_dot_and_render(output_path, "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# 3) Fused OpGraph
# ─────────────────────────────────────────────────────────────────────────────

def render_fused_op_graph(graph: Any, output_path: Path,
                          title: str = "",
                          max_edge_label_chars: int = 24) -> dict[str, Path]:
    """Render a post-FusionPass OpGraph.

    Fused nodes get a thick orange border; raw aten passthroughs stay
    thin.  Each fused node label shows the rule name and child count.
    """
    lines: list[str] = ["digraph fused_op_graph {"]
    lines.append('  graph [rankdir=TB, fontname="Helvetica", '
                 'concentrate=true, splines=spline, ranksep=0.55, nodesep=0.3]')
    lines.append('  node  [shape=box, style="filled,rounded", '
                 'fontname="Helvetica", fontsize=11, penwidth=1]')
    lines.append('  edge  [arrowsize=0.8, fontsize=8, fontname="Helvetica", '
                 'color="#666666", fontcolor="#444444"]')
    if title:
        lines.append(f'  label="{_esc(title)}"; labelloc=t; fontsize=15;')

    _, scope_to_nodes = _build_scope_clusters(graph)
    for scope, nids in scope_to_nodes.items():
        label = scope if scope != "__top__" else "<top-level>"
        cluster_id = f"cluster_{abs(hash(scope)) % 10**9}"
        lines.append(f'  subgraph {cluster_id} {{')
        lines.append(
            f'    label="{_esc(label)}"; style="filled,rounded"; '
            'color="#bbbbbb"; fillcolor="#fdfdfd"; fontsize=11;'
        )
        for nid in nids:
            node = graph.nodes[nid]
            color = _component_color(node, "#90CAF9")
            is_fused = bool(node.fused_from) or (node.num_sub_ops or 0) > 1
            rule_name = node.annotations.get("fused_by_rule", "")
            display_op = node.op_type.replace("aten.", "").replace(".default", "")
            if is_fused:
                lbl = f"{node.name or display_op}\\n[{display_op}]"
                if rule_name:
                    lbl += f"\\nrule={rule_name}"
                lbl += f"\\nsub={node.num_sub_ops}"
                border = _FUSED_BORDER
                pen = "3"
            else:
                lbl = display_op
                border = "#666666"
                pen = "1"
            lines.append(
                f'    "{_esc(nid)}" [label="{_esc(lbl)}", '
                f'fillcolor="{color}", fontcolor="white", '
                f'color="{border}", penwidth={pen}];'
            )
        lines.append("  }")

    for e in graph.edges:
        if e.src not in graph.nodes or e.dst not in graph.nodes:
            continue
        tensor = e.tensor
        sh = _shape(tensor)
        dt = _dtype(tensor)
        elabel = sh
        if dt:
            elabel = f"{sh} {dt}" if sh else dt
        if len(elabel) > max_edge_label_chars:
            elabel = elabel[: max_edge_label_chars - 1] + "…"
        if elabel:
            lines.append(
                f'  "{_esc(e.src)}" -> "{_esc(e.dst)}" [label="{_esc(elabel)}"];'
            )
        else:
            lines.append(f'  "{_esc(e.src)}" -> "{_esc(e.dst)}";')
    lines.append("}")
    return _write_dot_and_render(output_path, "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — graph correctness checks
# ─────────────────────────────────────────────────────────────────────────────

def assert_dag(graph: Any) -> dict[str, int]:
    """Verify the graph is a DAG and every edge endpoint exists."""
    stats = {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "dangling_edges": 0,
        "self_loops": 0,
    }
    for e in graph.edges:
        if e.src == e.dst:
            stats["self_loops"] += 1
        if e.src not in graph.nodes or e.dst not in graph.nodes:
            stats["dangling_edges"] += 1
    # Topo sort raises if cycle.
    graph.topo_sort()
    return stats
