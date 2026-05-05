"""GraphDumpPass: passthrough transform that snapshots the graph as DOT."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext

from python.zrt.report.dot_exporter import export_dot, render_dot


class GraphDumpPass(GraphPass):
    """Passthrough pass that snapshots the graph as a DOT file for debugging.

    Insert at any point in the transform pipeline to capture intermediate
    graph state — useful for reviewing parallelization, fusion, and
    communication insertion results visually.

    Parameters
    ----------
    label : str
        Short identifier used for the output filename (``{label}.dot``).
    output_dir : Path
        Directory for the DOT output.
    render : bool
        If True, also render to SVG when ``dot`` CLI is available.
    """

    def __init__(self, label: str, output_dir: Path, render: bool = True):
        self._label = label
        self._output_dir = Path(output_dir)
        self._render = render

    @property
    def name(self) -> str:
        return f"dump_{self._label}"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / f"{self._label}.dot"
        export_dot(graph, path)
        if self._render:
            render_dot(path)  # no-op when graphviz absent
        return graph  # passthrough — no mutation
