"""OpGraph: the central IR data structure for the entire simulation system.

Design constraints
------------------
- Pure Python, no NetworkX dependency
- Nodes stored in insertion-order dict for stable iteration
- Adjacency caches rebuilt lazily; mutating operations call _rebuild()
- Transform passes always clone() before mutating (functional style)
- ``hierarchy`` is built on first access and invalidated by structural mutations
"""
from __future__ import annotations

import copy
from collections import deque
from typing import Any, Iterator, Optional

from .edge import Edge
from .node import OpNode
from .types import TensorMeta


class OpGraph:
    """Directed acyclic graph of OpNodes connected by data-flow Edges.

    Parameters
    ----------
    name     : human-readable name, e.g. "DeepSeek-V3_prefill"
    phase    : "prefill" | "decode" | "forward" | "fused_prefill" | ...
    nodes    : initial nodes dict (node_id → OpNode); may be empty
    edges    : initial edge list; may be empty
    metadata : arbitrary key/value metadata (model, batch_size, seq_len, ...)
    """

    def __init__(
        self,
        name: str,
        phase: str,
        nodes: Optional[dict[str, OpNode]] = None,
        edges: Optional[list[Edge]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.name     = name
        self.phase    = phase
        self.nodes:    dict[str, OpNode] = dict(nodes)  if nodes    else {}
        self.edges:    list[Edge]        = list(edges)  if edges    else []
        self.metadata: dict[str, Any]   = dict(metadata) if metadata else {}

        # adjacency caches
        self._succ: dict[str, list[str]] = {}
        self._pred: dict[str, list[str]] = {}

        # lazy hierarchy
        self._hier: Optional["GraphHierarchy"] = None  # noqa: F821

        self._rebuild_adjacency()

    # ── internal ─────────────────────────────────────────────────────────────

    def _rebuild_adjacency(self) -> None:
        """Rebuild _succ/_pred from the current edge list."""
        self._succ = {nid: [] for nid in self.nodes}
        self._pred = {nid: [] for nid in self.nodes}
        for e in self.edges:
            if e.src in self._succ and e.dst in self._pred:
                if e.dst not in self._succ[e.src]:
                    self._succ[e.src].append(e.dst)
                if e.src not in self._pred[e.dst]:
                    self._pred[e.dst].append(e.src)
        self._hier = None  # invalidate hierarchy cache

    # ── read-only graph queries ───────────────────────────────────────────────

    def predecessors(self, node_id: str) -> list[str]:
        """Return IDs of nodes that have an edge pointing to ``node_id``."""
        return list(self._pred.get(node_id, []))

    def successors(self, node_id: str) -> list[str]:
        """Return IDs of nodes that ``node_id`` has an edge pointing to."""
        return list(self._succ.get(node_id, []))

    def in_edges(self, node_id: str) -> list[Edge]:
        """All edges whose dst == node_id."""
        return [e for e in self.edges if e.dst == node_id]

    def out_edges(self, node_id: str) -> list[Edge]:
        """All edges whose src == node_id."""
        return [e for e in self.edges if e.src == node_id]

    def topo_sort(self, debug: bool = False) -> list[OpNode]:
        """Return nodes in topological order (Kahn's algorithm).

        Raises RuntimeError if the graph has a cycle.
        """
        in_deg: dict[str, int] = {
            nid: len(preds) for nid, preds in self._pred.items()
        }
        q: deque[str] = deque(
            nid for nid, deg in in_deg.items() if deg == 0
        )
        result: list[OpNode] = []
        while q:
            nid = q.popleft()
            result.append(self.nodes[nid])
            for s in self._succ.get(nid, []):
                in_deg[s] -= 1
                if in_deg[s] == 0:
                    q.append(s)
        if len(result) != len(self.nodes):
            unreachable = [nid for nid in self.nodes if nid not in {n.id for n in result}]
            if debug:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Cycle detected in {self.name}. Unreachable nodes ({len(unreachable)}):")
                for nid in unreachable[:10]:
                    preds = self._pred.get(nid, [])
                    logger.error(f"  {nid}: {self.nodes[nid].op_type} (preds: {preds})")
            raise RuntimeError(
                f"OpGraph '{self.name}' contains a cycle; "
                f"only {len(result)}/{len(self.nodes)} nodes reached. "
                f"Unreachable: {unreachable[:5]}..."
            )
        return result

    def __iter__(self) -> Iterator[OpNode]:
        """Iterate nodes in insertion order."""
        return iter(self.nodes.values())

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.nodes

    # ── structural mutations (always call _rebuild_adjacency after) ───────────

    def add_node(self, node: OpNode) -> None:
        """Add a node.  Does NOT rebuild adjacency — call _rebuild_adjacency()
        manually or use insert_after() / replace_subgraph() which do so."""
        self.nodes[node.id] = node
        self._succ.setdefault(node.id, [])
        self._pred.setdefault(node.id, [])
        self._hier = None

    def add_edge(self, edge: Edge) -> None:
        """Append an edge and update adjacency caches."""
        self.edges.append(edge)
        if edge.src in self._succ and edge.dst not in self._succ[edge.src]:
            self._succ[edge.src].append(edge.dst)
        if edge.dst in self._pred and edge.src not in self._pred[edge.dst]:
            self._pred[edge.dst].append(edge.src)
        self._hier = None

    def insert_after(self, ref_id: str, new_node: OpNode,
                     new_edges: list[Edge]) -> None:
        """Insert ``new_node`` after ``ref_id``, wiring it with ``new_edges``."""
        self.nodes[new_node.id] = new_node
        self.edges.extend(new_edges)
        self._rebuild_adjacency()

    def replace_subgraph(self, old_ids: set[str],
                         new_node: OpNode) -> None:
        """Replace a set of nodes with a single fused node.

        External edges that enter ``old_ids`` are rewired to ``new_node``'s
        inputs; edges leaving ``old_ids`` are rewired from ``new_node``'s
        outputs.  Internal edges (both endpoints in ``old_ids``) are dropped.
        """
        # collect external edge endpoints before deletion
        in_edges  = [e for e in self.edges
                     if e.dst in old_ids and e.src not in old_ids]
        out_edges = [e for e in self.edges
                     if e.src in old_ids and e.dst not in old_ids]

        # remove old nodes and their edges
        for nid in old_ids:
            self.nodes.pop(nid, None)
        self.edges = [e for e in self.edges
                      if e.src not in old_ids and e.dst not in old_ids]

        # insert new node
        self.nodes[new_node.id] = new_node

        # rewire
        for e in in_edges:
            self.edges.append(Edge(
                src=e.src, src_idx=e.src_idx,
                dst=new_node.id, dst_idx=e.dst_idx,
                tensor=e.tensor, tensor_id=e.tensor_id,
            ))
        for e in out_edges:
            self.edges.append(Edge(
                src=new_node.id, src_idx=e.src_idx,
                dst=e.dst, dst_idx=e.dst_idx,
                tensor=e.tensor, tensor_id=e.tensor_id,
            ))

        self._rebuild_adjacency()

    # ── derived graph constructors ────────────────────────────────────────────

    def subgraph(self, node_ids: set[str]) -> "OpGraph":
        """Return a new OpGraph induced on ``node_ids``."""
        nodes = {nid: self.nodes[nid] for nid in node_ids if nid in self.nodes}
        edges = [e for e in self.edges
                 if e.src in node_ids and e.dst in node_ids]
        return OpGraph(
            name=f"{self.name}[sub]",
            phase=self.phase,
            nodes=nodes,
            edges=edges,
            metadata=dict(self.metadata),
        )

    def clone(self) -> "OpGraph":
        """Deep copy — safe to mutate without affecting the original."""
        return copy.deepcopy(self)

    @classmethod
    def from_model_spec(
        cls,
        model,  # zrt.training.spec.model.ModelSpec
        strategy,  # zrt.training.spec.strategy.Strategy
        phase: str = "training",
    ) -> "OpGraph":
        """Build OpGraph from ModelSpec geometry without tracing.

        This creates a lightweight DAG suitable for analytical estimation,
        using the same OpGraph structure as traced graphs.

        Args:
            model: ModelSpec with layer geometry
            strategy: Strategy with parallel configuration
            phase: Phase name (default: "training")

        Returns:
            OpGraph with nodes corresponding to training.ir.Graph ops
        """
        from zrt.training.ir.builders import build_graph
        from zrt.training.ir.training_graph import Graph as TrainingGraph

        training_g: TrainingGraph = build_graph(model, strategy)

        # Convert Training Ops → OpNodes
        nodes = {}
        for op in training_g.ops:
            nodes[op.name] = OpNode(
                id=op.name,
                op_type=op.kind,
                inputs=[],
                outputs=[],
                attrs={**op.meta},
                annotations={
                    "layer_id": op.layer_id,
                    "layer_kind": str(op.layer_kind),
                },
            )

        # Build sequential edges (op[i] → op[i+1] within layer)
        edges = []
        op_names = list(nodes.keys())
        for i in range(len(op_names) - 1):
            curr, nxt = op_names[i], op_names[i + 1]
            # Skip edge if crossing layer boundary
            if nodes[curr].annotations.get("layer_id") == nodes[nxt].annotations.get("layer_id"):
                edges.append(Edge(src=curr, src_idx=0, dst=nxt, dst_idx=0))

        # Attach collectives as JSON-safe dicts (no nodes - they're insert points)
        from dataclasses import asdict
        collectives = {c.name: asdict(c) for c in training_g.collectives}

        # Generate name from model geometry
        model_desc = f"h{model.hidden}_l{len(model.layers)}"
        return cls(
            name=f"spec_{model_desc}_{phase}",
            phase=phase,
            nodes=nodes,
            edges=edges,
            metadata={
                "source": "model_spec",
                "hidden": model.hidden,
                "layers": len(model.layers),
                "strategy": str(strategy),
                "collectives": collectives,
            },
        )

    # ── hierarchy view ────────────────────────────────────────────────────────

    @property
    def hierarchy(self) -> "GraphHierarchy":  # noqa: F821
        """Lazily-built scope tree.  Invalidated by structural mutations."""
        if self._hier is None:
            from .hierarchy import GraphHierarchy
            self._hier = GraphHierarchy(self)
        return self._hier

    # ── convenience stats ─────────────────────────────────────────────────────

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self) -> int:
        return len(self.edges)

    def compute_nodes(self) -> list[OpNode]:
        return [n for n in self.nodes.values() if n.category == "compute"]

    def comm_nodes(self) -> list[OpNode]:
        return [n for n in self.nodes.values() if n.category == "communication"]

    def memory_nodes(self) -> list[OpNode]:
        return [n for n in self.nodes.values() if n.category == "memory"]

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"OpGraph('{self.name}', phase={self.phase}, "
            f"nodes={len(self.nodes)}, edges={len(self.edges)})"
        )
