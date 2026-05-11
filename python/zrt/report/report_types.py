"""Report data types for multi-level hierarchical performance reports.

Defines the four-level drilldown structure:
    Block → SubStructure → OpFamily → OpDetail

along with the top-level ReportContext that gathers all metadata, KPI cards,
bound breakdown, and hierarchical data for rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Level 4: OpDetail — single operator atom
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpDetail:
    """Single operator atom — one OpNode + its SimResult."""

    op_node_id: str
    op_type: str                    # "aten.mm.default" | "comm.all_reduce"
    scope: str                      # full module path
    layer: str = ""                 # layer index "0" | "1" | ""

    # ── shapes ──
    input_shapes: list[str] = field(default_factory=list)    # ["[128,7168]", ...]
    output_shapes: list[str] = field(default_factory=list)   # ["[128,2048]"]
    shape_desc: str = ""            # "M=128, K=7168, N=2048"

    # ── performance (μs) ──
    flops: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    compute_us: float = 0.0
    memory_us: float = 0.0
    latency_us: float = 0.0
    bound: str = ""                 # "compute" | "memory" | "communication"
    confidence: float = 0.0         # [0, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Level 3: OpFamilyDetail — aggregated group of same-type ops
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpFamilyDetail:
    """Aggregated view of same-type operators within one SubStructure.

    This is the 12-column row in the target report table.
    """

    # ── identity ──
    op_type: str                    # canonical op_type, e.g. "aten.mm.default"
    display_name: str = ""          # human-readable, e.g. "Matrix Multiply"
    category: str = "compute"       # "compute" | "communication" | "memory"

    # ── stats ──
    count: int = 0                  # instance count within this sub-structure
    repeat: int = 1                 # cross-layer repeat multiplier
    first_scope: str = ""           # scope of first instance (for display)

    # ── shape / formula ──
    shape_desc: str = ""            # "tokens=128, hidden=7168"
    formula: str = ""               # "2·M·K·N"
    io_formula: str = ""            # "R=(M·K+K·N)·dtype  W=M·N·dtype"

    # ── aggregated performance (ms) ──
    tflops: float = 0.0
    hbm_bytes: int = 0              # read + write
    comm_bytes: int = 0             # communication volume (only for comm ops)
    compute_ms: float = 0.0
    memory_ms: float = 0.0
    comm_ms: float = 0.0
    total_ms: float = 0.0

    # ── bound ──
    bound: str = ""
    confidence: float = 0.0
    pct_of_substructure: float = 0.0   # % of parent SubStructure's total_ms

    # ── children ──
    children: list[OpDetail] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Level 2: SubStructureDetail — functional module within a Block
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubStructureDetail:
    """Functional module within a Block (e.g. Attention, MLP, Router)."""

    # ── identity ──
    name: str                       # "Attention" | "Router" | "Norm" | "Experts"
    scope_group: str = ""           # scope key: "self_attn" | "mlp" | "gate"
    component_type: str = ""        # from classify_component: "attn" | "ffn" | ...

    # ── stats ──
    total_ms: float = 0.0
    pct_of_block: float = 0.0

    # ── children ──
    op_families: list[OpFamilyDetail] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Level 1: BlockDetail — top-level model block
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BlockDetail:
    """Top-level model structural block (Embedding, TransformerLayer, Output)."""

    # ── identity ──
    name: str                       # "Embedding" | "MoEBlock" | "Output"
    scope: str = ""                 # HierNode scope, e.g. "model.layers.0"
    phase: str = ""                 # "prefill" | "decode"

    # ── stats ──
    repeat: int = 1                 # cross-layer repeat count (61 for 61-layer)
    total_ms: float = 0.0           # single-layer_ms × repeat
    pct_of_total: float = 0.0       # % of total latency
    dominant_bound: str = ""        # "compute" | "memory" | "communication"

    # ── children ──
    sub_structures: list[SubStructureDetail] = field(default_factory=list)

    # ── extra model params ──
    model_params: dict[str, Any] = field(default_factory=dict)
    # e.g. {"routed_experts": 384, "active_per_token": 6, "mtp_depth": 3}


# ─────────────────────────────────────────────────────────────────────────────
# ReportContext — top-level container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReportContext:
    """Complete report context — all data needed to render the HTML report."""

    # ── metadata (Hero Card subtitle) ──
    model: str = ""
    hardware: str = ""
    phase: str = ""                 # "prefill" | "decode"
    parallel_desc: str = ""         # "TP2_EP8_PP1"
    topology_desc: str = ""         # "2Node-16NPU"
    batch_size: int = 1
    seq_len: int = 8192
    active_params: int = 0          # e.g. 49_000_000_000
    total_params: int = 0           # e.g. 1_600_000_000_000

    # ── KPI cards ──
    prefill_ms: float | None = None
    tpot_ms: float | None = None
    mtp_adjusted_tpot_ms: float | None = None
    tokens_per_sec: float = 0.0
    memory_per_gpu_gb: float = 0.0
    model_blocks: int = 0

    # ── MTP metadata ──
    mtp_depth: int = 1
    mtp_acceptance_rate: float = 0.0
    mtp_effective_tokens: float = 1.0

    # ── Bound bar ──
    compute_pct: float = 0.0
    memory_pct: float = 0.0
    communication_pct: float = 0.0
    compute_ms: float = 0.0
    memory_ms: float = 0.0
    communication_ms: float = 0.0

    # ── hierarchical data ──
    blocks: list[BlockDetail] = field(default_factory=list)

    # Phase-split block lists (populated when graph has fwd+bwd stitched nodes).
    # blocks_bwd is used exclusively by the Backward structure SVG tab.
    blocks_bwd: list[BlockDetail] = field(default_factory=list)

    # ── calibration / warnings (Phase 4) ──
    calibration: list[dict] = field(default_factory=list)
    references: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
