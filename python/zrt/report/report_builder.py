"""Report builder: constructs hierarchical ReportContext from raw simulation outputs.

Four-phase construction:
  1. build_metadata()   — Hero Card fields + KPI + bound bar
  2. identify_blocks()  — Group GraphHierarchy nodes into model-level blocks
  3. build_sub_structures() — Within each block, group by component
  4. build_op_families()    — Within each sub-structure, aggregate by op_type
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from python.zrt.report.report_types import (
    BlockDetail,
    OpDetail,
    OpFamilyDetail,
    ReportContext,
    SubStructureDetail,
)
from python.zrt.report.formula_registry import FormulaRegistry
from python.zrt.report.shape_desc import describe_shapes

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.ir.hierarchy import GraphHierarchy, HierNode
    from python.zrt.ir.node import OpNode
    from python.zrt.simulator.result import SimResult
    from python.zrt.executor.scheduler import Timeline
    from python.zrt.hardware.spec import HardwareSpec
    from python.zrt.transform.context import TransformContext

# Global registry instance (lazy-init, cached per process)
_formula_registry: FormulaRegistry | None = None


def _get_formula_registry() -> FormulaRegistry:
    global _formula_registry
    if _formula_registry is None:
        _formula_registry = FormulaRegistry()
    return _formula_registry


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def build_report_context(
    model: str,
    hardware: str,
    phase: str,
    batch_size: int,
    seq_len: int,
    graph: "OpGraph",
    sim_results: "dict[str, SimResult]",
    timeline: "Timeline",
    hw_spec: "HardwareSpec",
    ctx: "TransformContext",
    profile: "Any | None" = None,
    memory_budget: "Any | None" = None,  # MemoryBudget
) -> ReportContext:
    """Build the complete ReportContext from simulation outputs.

    Parameters
    ----------
    model / hardware / phase / batch_size / seq_len
        Descriptive metadata.
    graph : OpGraph
        Transformed graph (after all transform passes).
    sim_results : dict[str, SimResult]
        node_id → simulation result.
    timeline : Timeline
        Scheduled timeline from DAGScheduler.
    hw_spec : HardwareSpec
        Hardware specification.
    ctx : TransformContext
        Transform context with parallel config etc.
    profile : ModelProfile | None
        Model profile with structure info.
    memory_budget : MemoryBudget | None
        Memory breakdown estimate.
    """
    from python.zrt.ir.hierarchy import GraphHierarchy

    rc = ReportContext()
    hier = GraphHierarchy(graph)

    # ── Phase 1: metadata + KPI ──────────────────────────────────────────────
    _build_metadata(rc, model, hardware, phase, batch_size, seq_len,
                    timeline, hw_spec, ctx, profile, memory_budget)

    # ── Phase 2: bound bar ───────────────────────────────────────────────────
    _build_bound(rc, sim_results, graph)

    # ── Phase 3: hierarchical data ───────────────────────────────────────────
    rc.blocks = _build_blocks(hier, graph, sim_results, phase, profile)

    # For stitched fwd+bwd training graphs, also build backward-only blocks so
    # that the Backward structure SVG tab can render real content.
    if graph.metadata.get("fwd_bwd_stitched"):
        rc.blocks_bwd = _build_phase_filtered_blocks(
            graph, sim_results, "bwd", phase, profile)

    # ── KPI correction for partial-trace layer scaling ───────────────────────
    # When only N layers were traced but the full model has M > N layers,
    # _build_blocks scales block.total_ms by real_repeat to represent the full
    # model.  The timeline-derived tpot_ms / prefill_ms still reflects the N-layer
    # traced graph, causing the header KPI to disagree with the block breakdown.
    # Bring them into sync: use full_model_total_ms (sum of scaled blocks) as
    # the canonical latency estimate.
    _layer_scale = graph.metadata.get("layer_scale", 1.0)
    if _layer_scale > 1.0 and rc.blocks:
        _full_model_total_ms = sum(bd.total_ms for bd in rc.blocks)
        if _full_model_total_ms > 0:
            _latency_s = _full_model_total_ms * 1e-3
            if rc.tpot_ms is not None:
                rc.tpot_ms = _full_model_total_ms
                rc.tokens_per_sec = rc.batch_size / _latency_s if _latency_s > 0 else 0.0
            elif rc.prefill_ms is not None:
                rc.prefill_ms = _full_model_total_ms
                rc.tokens_per_sec = (
                    rc.batch_size * (rc.seq_len or 1) / _latency_s
                    if _latency_s > 0 else 0.0
                )

    # ── Phase 4: calibration / references / warnings ──────────────────────────
    _build_calibration(rc, graph, sim_results, profile)
    _build_references(rc, model, hardware, hw_spec)
    _build_warnings(rc, phase, ctx, profile)

    return rc


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: metadata + KPI
# ─────────────────────────────────────────────────────────────────────────────


def _build_metadata(
    rc: ReportContext,
    model: str,
    hardware: str,
    phase: str,
    batch_size: int,
    seq_len: int,
    timeline: "Timeline",
    hw_spec: "HardwareSpec",
    ctx: "TransformContext",
    profile: "Any | None",
    memory_budget: "Any | None",
) -> None:
    rc.model = model
    rc.hardware = hardware
    rc.phase = phase
    rc.batch_size = batch_size
    rc.seq_len = seq_len

    # ── parallel / topology description ────────────────────────────────────
    rc.parallel_desc = ctx.parallel.describe() if ctx else "single"
    nodes = getattr(hw_spec, "nodes", 1)
    gpus_per = getattr(hw_spec, "gpus_per_node", 1)
    rc.topology_desc = f"{nodes}Node-{nodes * gpus_per}GPU" if nodes > 1 else f"{nodes * gpus_per}GPU"

    # ── KPI: latency / throughput ──────────────────────────────────────────
    latency_s = timeline.total_latency_us * 1e-6
    latency_ms = timeline.total_latency_us / 1000.0

    if phase == "prefill":
        rc.prefill_ms = latency_ms
        rc.tpot_ms = None
        rc.tokens_per_sec = (batch_size * seq_len / latency_s) if latency_s > 0 else 0.0
    else:
        rc.prefill_ms = None
        rc.tpot_ms = latency_ms
        rc.tokens_per_sec = (batch_size / latency_s) if latency_s > 0 else 0.0

    # ── MTP-adjusted metrics ───────────────────────────────────────────────
    if ctx and ctx.training:
        rc.mtp_acceptance_rate = getattr(ctx.training, "mtp_acceptance_rate", 0.0)
        rc.mtp_depth = getattr(ctx.training, "mtp_depth", 1)
        if rc.mtp_depth > 1 and rc.mtp_acceptance_rate > 0:
            rc.mtp_effective_tokens = 1.0 + (rc.mtp_depth - 1) * rc.mtp_acceptance_rate
            if rc.tpot_ms is not None:
                rc.mtp_adjusted_tpot_ms = rc.tpot_ms / rc.mtp_effective_tokens

    # ── model params ───────────────────────────────────────────────────────
    if profile:
        rc.active_params = getattr(profile, "active_param_count", 0) or getattr(profile, "param_count", lambda: 0)()
        if callable(rc.active_params):
            rc.active_params = rc.active_params()
        rc.total_params = getattr(profile, "total_param_count", 0) or rc.active_params

        # Adjust for parallelism
        tp = ctx.parallel.tp if ctx else 1
        pp = ctx.parallel.pp if ctx else 1
        if tp > 1:
            rc.active_params = rc.active_params // tp
        if pp > 1:
            rc.active_params = rc.active_params // pp

    # ── memory per GPU ─────────────────────────────────────────────────────
    if memory_budget:
        rc.memory_per_gpu_gb = getattr(memory_budget, "total_mb", 0.0) / 1024.0
    else:
        rc.memory_per_gpu_gb = 0.0

    # ── model blocks count ─────────────────────────────────────────────────
    rc.model_blocks = getattr(profile, "num_layers", 0) if profile else 0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: bound bar
# ─────────────────────────────────────────────────────────────────────────────


def _build_bound(
    rc: ReportContext,
    sim_results: "dict[str, SimResult]",
    graph: "OpGraph",
) -> None:
    """Compute compute/memory/communication latency fractions."""
    total_compute = 0.0
    total_memory = 0.0
    total_comm = 0.0

    for node_id, sr in sim_results.items():
        node = graph.nodes.get(node_id)
        if node and node.category == "communication":
            total_comm += sr.latency_us
        elif sr.bound == "memory":
            total_memory += sr.latency_us
        else:
            total_compute += sr.latency_us

    total = total_compute + total_memory + total_comm
    if total > 0:
        rc.compute_pct = total_compute / total * 100.0
        rc.memory_pct = total_memory / total * 100.0
        rc.communication_pct = total_comm / total * 100.0
        rc.compute_ms = total_compute / 1000.0
        rc.memory_ms = total_memory / 1000.0
        rc.communication_ms = total_comm / 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: hierarchical blocks
# ─────────────────────────────────────────────────────────────────────────────


# ── Component groups (ordered by typical transformer data flow) ──────────────

_COMPONENT_ORDER = [
    "norm", "attn", "hc.pre_attn", "hc.post_attn",
    "residual", "add",
    "moe.gate", "router",
    "moe.shared", "shared",
    "ffn", "mlp",
    "moe.dispatch", "moe.combine",
    "moe.experts", "moe",
    "hc.pre_ffn", "hc.post_ffn",
    "comm",
    "embedding", "lm_head", "final_norm",
]

_COMPONENT_GROUP_NAMES: dict[str, tuple[str, str]] = {
    # ── Norm variants ──
    "attn_norm":    ("Attn Norm", "norm"),
    "ffn_norm":     ("FFN Norm", "norm"),
    "final_norm":   ("Final Norm", "norm"),
    "norm":         ("Norm", "norm"),
    # ── Attention sub-components ──
    "attn.q_a_proj":("Q Proj", "attn"),
    "attn.q_b_proj":("Q_B Proj", "attn"),
    "attn.kv_a_proj":("KV Proj", "attn"),
    "attn.kv_b_proj":("KV_B Proj", "attn"),
    "attn.score":   ("Score", "attn"),
    "attn.softmax": ("Softmax", "attn"),
    "attn.rope":    ("RoPE", "attn"),
    "attn.o_proj":  ("O Proj", "attn"),
    "attn":         ("Attention", "attn"),
    # ── MoE sub-components ──
    "moe.gate":     ("Router", "router"),
    "moe.shared":   ("Shared Expert", "shared"),
    "moe.experts":  ("Routed Experts", "proj"),
    "moe.dispatch": ("Dispatch", "comm"),
    "moe.combine":  ("Combine", "comm"),
    "moe":          ("MoE", "proj"),
    # ── FFN sub-components ──
    "ffn.gate_proj":("Gate Proj", "proj"),
    "ffn.up_proj":  ("Up Proj", "proj"),
    "ffn.down_proj":("Down Proj", "proj"),
    "ffn.silu":     ("SiLU", "act"),
    "ffn.gelu":     ("GELU", "act"),
    "ffn":          ("FFN", "proj"),
    "mlp":          ("MLP", "proj"),
    # ── Communication ──
    "comm":         ("Communication", "comm"),
    # ── Embedding / Output ──
    "embedding":    ("Embedding", "proj"),
    "lm_head":      ("LM Head", "proj"),
    "final_norm":   ("Final Norm", "norm"),
    # ── Residual / HC ──
    "residual":     ("Residual", "resid"),
    "add":          ("Residual Add", "resid"),
    "hc.pre_attn":  ("HC Pre-Attn", "norm"),
    "hc.post_attn": ("HC Post-Attn", "norm"),
    "hc.pre_ffn":   ("HC Pre-FFN", "norm"),
    "hc.post_ffn":  ("HC Post-FFN", "norm"),
    "shared":       ("Shared", "shared"),
    "router":       ("Router", "router"),
}


def _component_group_name(component: str) -> tuple[str, str]:
    """Return (display_name, css_class) for a component string."""
    # Try exact match
    if component in _COMPONENT_GROUP_NAMES:
        return _COMPONENT_GROUP_NAMES[component]
    # Try two-segment dot prefix (e.g. "moe.gate.mm" → "moe.gate")
    parts = component.split(".")
    if len(parts) >= 2:
        two_seg = ".".join(parts[:2])
        if two_seg in _COMPONENT_GROUP_NAMES:
            return _COMPONENT_GROUP_NAMES[two_seg]
    # Try single-segment prefix
    if parts[0] in _COMPONENT_GROUP_NAMES:
        return _COMPONENT_GROUP_NAMES[parts[0]]
    # Fallback: title-cased original name
    display = component.replace(".", " ").replace("_", " ").title()
    return (display, "proj")


def _component_sort_key(component: str) -> int:
    """Return a sort key for ordering components in dataflow order."""
    for i, c in enumerate(_COMPONENT_ORDER):
        if component.startswith(c):
            return i
    return len(_COMPONENT_ORDER)


# ── Block identification ─────────────────────────────────────────────────────

# Scope names injected by parallelism transforms — never real model blocks.
_PARALLELISM_SCOPE_NAMES: frozenset[str] = frozenset({
    "data_parallel", "pipeline", "pipeline_parallel", "tensor_parallel",
    "expert_parallel", "context_parallel", "grad_reduce", "p2p", "nccl",
    "all_reduce", "all_gather", "reduce_scatter", "comm", "optimizer",
})


def _is_parallelism_scope(name: str) -> bool:
    """True if *name* (a single scope segment) is a parallelism-injected container."""
    n = name.lower()
    return n in _PARALLELISM_SCOPE_NAMES or any(n.startswith(kw) for kw in _PARALLELISM_SCOPE_NAMES)


def _find_layer_container(hier: "GraphHierarchy") -> "HierNode | None":
    """Return the HierNode that directly parents transformer layer blocks.

    Picks the shallowest node with the most numeric-named children (e.g. model.layers
    with children "0", "1", …, "N-1").  Parallelism-scope containers are skipped.
    """
    best: "HierNode | None" = None
    best_depth = 999
    best_count = 0

    for hn in hier._scope_map.values():
        if not hn.children:
            continue
        if _is_parallelism_scope(hn.name):
            continue
        numeric_children = [c for c in hn.children if c.name.isdigit()]
        n = len(numeric_children)
        if n >= 2:
            if hn.depth < best_depth or (hn.depth == best_depth and n > best_count):
                best = hn
                best_depth = hn.depth
                best_count = n

    return best


def _build_phase_filtered_blocks(
    graph: "OpGraph",
    sim_results: "dict[str, SimResult]",
    target_phase_ann: str,
    phase: str,
    profile: "Any | None",
) -> "list[BlockDetail]":
    """Build blocks using only nodes annotated with *target_phase_ann*.

    Used to produce separate forward / backward block lists from a stitched
    training graph so that each structure SVG tab shows the correct data.
    """
    from python.zrt.ir.graph import OpGraph
    from python.zrt.ir.hierarchy import GraphHierarchy

    filtered_nodes = {
        nid: node for nid, node in graph.nodes.items()
        if node.annotations.get("phase") == target_phase_ann
    }
    if not filtered_nodes:
        return []

    sub = OpGraph(
        name=f"{graph.name}_{target_phase_ann}",
        phase=phase,
        nodes=filtered_nodes,
    )
    sub_hier = GraphHierarchy(sub)
    sub_sim = {nid: sr for nid, sr in sim_results.items() if nid in filtered_nodes}
    return _build_blocks(sub_hier, sub, sub_sim, phase, profile)


def _build_blocks(
    hier: "GraphHierarchy",
    graph: "OpGraph",
    sim_results: "dict[str, SimResult]",
    phase: str,
    profile: "Any | None",
) -> list[BlockDetail]:
    """Build the list of BlockDetail using architecture-aware scope analysis.

    Algorithm:
      1. Locate the *layers container* — the node whose children are the actual
         transformer block indices (e.g. ``model.layers`` → children "0", "1", …).
      2. Layer blocks = numeric children of that container.
      3. Special blocks = non-numeric siblings of the container (embed_tokens, norm)
         + depth-1 output nodes (lm_head).
      4. Parallelism-injected scopes (data_parallel, pipeline, optimizer, …) are
         excluded at every step, so they never appear as top-level blocks.
    """
    latency_map = {r.op_node_id: r.latency_us for r in sim_results.values()}
    total_latency = sum(latency_map.values()) or 1.0

    layer_hnodes: list["HierNode"] = []
    special_hnodes: list["HierNode"] = []
    seen_scopes: set[str] = set()

    def _try_add_special(hn: "HierNode") -> None:
        if hn.scope not in seen_scopes and hn.all_leaf_ids():
            seen_scopes.add(hn.scope)
            special_hnodes.append(hn)

    # ── Step 1: Locate layers container ───────────────────────────────────
    layers_container = _find_layer_container(hier)

    if layers_container:
        # Numeric children = actual transformer layer blocks
        layer_hnodes = [c for c in layers_container.children if c.name.isdigit()]
        for hn in layer_hnodes:
            seen_scopes.add(hn.scope)

        # Parent of layers container (e.g. "model" for model.layers)
        parent_scope = (
            layers_container.scope.rsplit(".", 1)[0]
            if "." in layers_container.scope else ""
        )
        parent_hn = hier.get(parent_scope)

        # Siblings of the layers container → Embedding, Norm, etc.
        if parent_hn:
            for sibling in parent_hn.children:
                if sibling is layers_container:
                    continue
                if sibling.name.isdigit():
                    continue
                if _is_parallelism_scope(sibling.name):
                    continue
                _try_add_special(sibling)

        # Build set of scopes that ARE ancestors of the layers container
        # (e.g. "model", "model.layers") so we don't re-add them below.
        ancestor_scopes: set[str] = set()
        _s = layers_container.scope
        while _s:
            ancestor_scopes.add(_s)
            _s = _s.rsplit(".", 1)[0] if "." in _s else ""

        # Depth-1 output-style nodes not already covered (e.g. lm_head)
        for hn in hier.at_depth(1):
            if hn.name.isdigit():
                continue
            if _is_parallelism_scope(hn.name):
                continue
            if hn.scope in ancestor_scopes:
                continue
            _try_add_special(hn)

    else:
        # Fallback when no layers container is found (unusual graph structure).
        # Use the depth with the most numeric children, skip parallelism roots.
        best_depth = 3
        best_count = 0
        for depth in range(2, 6):
            numeric = [
                hn for hn in hier.at_depth(depth)
                if hn.name.isdigit()
                and not _is_parallelism_scope(hn.scope.split(".")[0])
            ]
            if len(numeric) > best_count:
                best_count = len(numeric)
                best_depth = depth

        for hn in hier.at_depth(best_depth):
            if _is_parallelism_scope(hn.scope.split(".")[0]):
                continue
            if hn.name.isdigit():
                layer_hnodes.append(hn)
            seen_scopes.add(hn.scope)

        # Special blocks: embed / norm / head at shallower depths
        _special_markers = ("embed", "norm", "head", "final", "lm_head", "wte",
                            "tok_embed", "ln_f", "rms")
        for depth in range(1, best_depth):
            for hn in hier.at_depth(depth):
                if hn.name.isdigit():
                    continue
                if _is_parallelism_scope(hn.name):
                    continue
                if hn.scope in seen_scopes:
                    continue
                scope_lower = hn.scope.lower()
                if any(m in scope_lower for m in _special_markers):
                    _try_add_special(hn)

    # ── Build BlockDetail objects ──────────────────────────────────────────
    # Correct order: prefix (embedding) → layer groups → suffix (norm/head/output).
    # Suffix blocks with the same display name are merged into one to avoid
    # duplicates (e.g. model.norm and lm_head both produce "Output").

    def _scope_is_prefix(hn: "HierNode") -> bool:
        s = hn.scope.lower()
        return any(kw in s for kw in
                   ("embed", "tok_embed", "wte", "wpe", "word_embed", "pos_embed"))

    prefix_hnodes = [hn for hn in special_hnodes if _scope_is_prefix(hn)]
    suffix_hnodes = [hn for hn in special_hnodes if not _scope_is_prefix(hn)]

    blocks: list[BlockDetail] = []

    # 1) Prefix blocks (Embedding, positional encoding, …)
    for hn in prefix_hnodes:
        bd = _build_single_block(hn, graph, sim_results, phase, total_latency,
                                 repeat=1, profile=profile)
        if bd is not None:
            blocks.append(bd)

    # 2) Layer blocks: one block per traced layer (preserves temporal order).
    #    No cross-layer grouping — interleaved patterns (e.g. V4 even/odd) are
    #    shown in their actual sequential order.
    if layer_hnodes:
        total_traced_layers = len(layer_hnodes)
        total_real_layers = (getattr(profile, "num_layers", None) if profile else None)

        # Scale factor: each traced layer represents this many full-model layers.
        _layer_scale = (total_real_layers / total_traced_layers
                        if total_real_layers and total_real_layers > total_traced_layers
                        else 1.0)

        for hn in layer_hnodes:
            layer_idx = int(hn.name) if hn.name.isdigit() else -1
            bd = _build_single_block(
                hn, graph, sim_results, phase, total_latency,
                repeat=1, profile=profile,
            )
            if bd is None:
                continue
            # Scale timing to represent the full model depth.
            bd.total_ms = bd.total_ms * _layer_scale if bd.total_ms > 0 else 0
            bd.repeat = max(1, int(round(_layer_scale)))
            # Label with the actual traced layer index.
            bd.name = f"{bd.name} [layer {layer_idx}]"
            blocks.append(bd)

    # 3) Suffix blocks (Output, Final Norm, LM Head, …)
    #    Merge same-named blocks so norm + lm_head → one "Output" entry.
    suffix_merged: dict[str, BlockDetail] = {}
    for hn in suffix_hnodes:
        bd = _build_single_block(hn, graph, sim_results, phase, total_latency,
                                 repeat=1, profile=profile)
        if bd is None:
            continue
        if bd.name in suffix_merged:
            existing = suffix_merged[bd.name]
            prev_ms = existing.total_ms
            existing.total_ms += bd.total_ms
            if bd.total_ms > prev_ms:
                existing.dominant_bound = bd.dominant_bound
            existing.sub_structures.extend(bd.sub_structures)
            existing.pct_of_total = (
                (existing.total_ms / (total_latency / 1000.0)) * 100
                if total_latency > 0 else 0
            )
        else:
            suffix_merged[bd.name] = bd
    blocks.extend(suffix_merged.values())

    # Recompute pct_of_total against the full-model estimated total.
    # Required when layer repeats were scaled from profile.num_layers, because
    # the traced-graph total_latency no longer matches the scaled block totals.
    full_model_total_ms = sum(bd.total_ms for bd in blocks)
    if full_model_total_ms > 0:
        for bd in blocks:
            bd.pct_of_total = bd.total_ms / full_model_total_ms * 100

    return blocks


def _build_single_block(
    hn: "HierNode",
    graph: "OpGraph",
    sim_results: "dict[str, SimResult]",
    phase: str,
    total_latency_us: float,
    repeat: int = 1,
    profile: "Any | None" = None,
) -> BlockDetail | None:
    """Build a BlockDetail from a single HierNode."""
    node_ids = hn.all_leaf_ids()
    if not node_ids:
        return None

    # ── block-level aggregation ─────────────────────────────────────────────
    block_latency_us = sum(
        sim_results[nid].latency_us
        for nid in node_ids
        if nid in sim_results
    )
    block_ms = block_latency_us / 1000.0
    pct = (block_latency_us / total_latency_us * 100) if total_latency_us > 0 else 0.0

    # Dominant bound
    bounds = defaultdict(float)
    for nid in node_ids:
        if nid in sim_results:
            bounds[sim_results[nid].bound] += sim_results[nid].latency_us
    dominant_bound = max(bounds, key=bounds.get) if bounds else "compute"

    # Block name
    block_name = _block_display_name(hn, graph, profile)

    # ── Build sub-structures ────────────────────────────────────────────────
    sub_structures = _build_sub_structures(hn, graph, sim_results, block_latency_us, repeat)

    # ── Model params ─────────────────────────────────────────────────────────
    model_params: dict = {}
    if profile:
        if getattr(profile, "is_moe", False):
            model_params["num_experts"] = getattr(profile, "num_experts", 0)
            model_params["active_per_token"] = getattr(profile, "moe_topk", 0)

    return BlockDetail(
        name=block_name,
        scope=hn.scope,
        phase=phase,
        repeat=repeat,
        total_ms=block_ms,
        pct_of_total=pct,
        dominant_bound=dominant_bound,
        sub_structures=sub_structures,
        model_params=model_params,
    )


def _block_display_name(
    hn: "HierNode",
    graph: "OpGraph",
    profile: "Any | None",
) -> str:
    """Heuristic: determine block display name from scope + model info."""
    scope = hn.scope.lower()

    # Embedding
    if "embed" in scope or "tok_embeddings" in scope:
        return "Embedding"

    # Layer blocks (numeric)
    if hn.name.isdigit():
        layer_idx = int(hn.name)

        # Architecture-aware: profile's per-layer indices take precedence over
        # all heuristics (prevents first_k_dense layers from being mis-labelled
        # as MoEBlock in mixed-architecture models like DeepSeek-V3).
        if profile:
            _sparse = getattr(profile, "sparse_layer_indices", None)
            _dense = getattr(profile, "dense_layer_indices", None)
            if _sparse is not None or _dense is not None:
                if _sparse and layer_idx in _sparse:
                    return "MoEBlock"
                return "TransformerBlock"

        # Fallback heuristics (used when profile has no per-layer info)
        for child in hn.children:
            if "expert" in child.name.lower() or "moe" in child.name.lower():
                return "MoEBlock"
        if "moe" in scope:
            return "MoEBlock"
        for nid in hn.all_leaf_ids()[:10]:
            node = graph.nodes.get(nid)
            if node and node.module_class:
                mc = node.module_class.lower()
                if any(kw in mc for kw in ("moe", "expert", "sparse", "gate", "router")):
                    return "MoEBlock"
        if profile and getattr(profile, "is_moe", False):
            return "MoEBlock"
        return "TransformerBlock"

    # Output / final
    if "norm" in scope or "final" in scope:
        return "Output"
    if "lm_head" in scope or "head" in scope:
        return "Output"

    # Fallback
    return hn.name.replace("_", " ").title()


# ── Sub-structure building ───────────────────────────────────────────────────


def _build_sub_structures(
    block_hn: "HierNode",
    graph: "OpGraph",
    sim_results: "dict[str, SimResult]",
    block_latency_us: float,
    repeat: int,
) -> list[SubStructureDetail]:
    """Group a block's children into SubStructureDetails by component."""
    from python.zrt.graph.classifier import classify_component

    # Collect all leaf ops — group by component
    leaf_ids = block_hn.all_leaf_ids()
    comp_groups: dict[str, list[str]] = defaultdict(list)

    for nid in leaf_ids:
        if nid not in graph.nodes:
            continue
        node = graph.nodes[nid]
        component = classify_component(node.scope, node.op_type)
        if not component:
            component = node.category  # fallback: "compute" | "communication" | "memory"
        comp_groups[component].append(nid)

    # Build SubStructureDetail per component group
    sub_structures: list[SubStructureDetail] = []
    for component, group_ids in sorted(
        comp_groups.items(), key=lambda x: _component_sort_key(x[0])
    ):
        ss = _build_single_substructure(
            component, group_ids, graph, sim_results, block_latency_us, repeat,
        )
        if ss is not None:
            sub_structures.append(ss)

    return sub_structures


def _build_single_substructure(
    component: str,
    node_ids: list[str],
    graph: "OpGraph",
    sim_results: "dict[str, SimResult]",
    block_latency_us: float,
    repeat: int,
) -> SubStructureDetail | None:
    """Build a SubStructureDetail for a component group."""
    if not node_ids:
        return None

    display_name, css_class = _component_group_name(component)

    # Aggregate latency
    ss_latency_us = sum(
        sim_results[nid].latency_us
        for nid in node_ids
        if nid in sim_results
    )
    ss_ms = ss_latency_us / 1000.0
    pct_of_block = (ss_latency_us / block_latency_us * 100) if block_latency_us > 0 else 0.0

    # Scope group: first scope's immediate parent or last segment
    first_scope = ""
    for nid in node_ids:
        if nid in graph.nodes:
            first_scope = graph.nodes[nid].scope
            break
    scope_group = first_scope.rsplit(".", 1)[-1] if first_scope else component

    # Build op families
    op_families = _build_op_families(node_ids, graph, sim_results, ss_latency_us, repeat)

    return SubStructureDetail(
        name=display_name,
        scope_group=scope_group,
        component_type=component,
        total_ms=ss_ms,
        pct_of_block=pct_of_block,
        op_families=op_families,
    )


# ── Op family building ───────────────────────────────────────────────────────


def _build_op_families(
    node_ids: list[str],
    graph: "OpGraph",
    sim_results: "dict[str, SimResult]",
    ss_latency_us: float,
    repeat: int,
) -> list[OpFamilyDetail]:
    """Aggregate ops within a sub-structure by op_type into OpFamilyDetails."""
    reg = _get_formula_registry()

    # Group by op_type
    type_groups: dict[str, list[str]] = defaultdict(list)
    for nid in node_ids:
        if nid in graph.nodes:
            op_type = graph.nodes[nid].op_type
            type_groups[op_type].append(nid)

    families: list[OpFamilyDetail] = []
    for op_type, group_ids in sorted(type_groups.items()):
        ofd = _build_single_op_family(op_type, group_ids, graph, sim_results, ss_latency_us, repeat, reg)
        if ofd is not None:
            families.append(ofd)

    # Sort by total_ms descending
    families.sort(key=lambda f: -f.total_ms)
    return families


def _build_single_op_family(
    op_type: str,
    node_ids: list[str],
    graph: "OpGraph",
    sim_results: "dict[str, SimResult]",
    ss_latency_us: float,
    repeat: int,
    reg: FormulaRegistry,
) -> OpFamilyDetail | None:
    """Build an OpFamilyDetail for one op_type group."""
    if not node_ids:
        return None

    # Formula lookup
    entry = reg.lookup(op_type)
    display_name = entry.display_name if entry else op_type.split(".")[-1]
    category = entry.category if entry else "compute"
    flops_formula = entry.flops_formula if entry else "?"
    io_formula = entry.io_formula if entry else "?"

    # Aggregate metrics
    count = len(node_ids)
    total_flops = 0
    total_read = 0
    total_write = 0
    total_compute_us = 0.0
    total_memory_us = 0.0
    total_comm_us = 0.0
    total_comm_bytes = 0
    total_latency_us = 0.0
    bounds = defaultdict(float)
    confidences = []

    for nid in node_ids:
        if nid not in sim_results:
            continue
        sr = sim_results[nid]
        total_flops += sr.flops
        total_read += sr.read_bytes
        total_write += sr.write_bytes
        total_compute_us += sr.compute_us
        total_memory_us += sr.memory_us
        total_latency_us += sr.latency_us
        bounds[sr.bound] += sr.latency_us
        confidences.append(sr.confidence)

        if nid in graph.nodes and graph.nodes[nid].category == "communication":
            total_comm_us += sr.latency_us
            total_comm_bytes += sr.read_bytes + sr.write_bytes

    total_ms = (total_latency_us * repeat) / 1000.0
    dominant_bound = max(bounds, key=bounds.get) if bounds else "compute"
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    pct = (total_latency_us / ss_latency_us * 100) if ss_latency_us > 0 else 0.0

    # Shape description from first node
    shape_desc = ""
    first_scope = ""
    for nid in node_ids:
        if nid in graph.nodes:
            shape_desc = describe_shapes(graph.nodes[nid])
            first_scope = graph.nodes[nid].scope
            break

    # Build child OpDetail list (sample first 3 for detail)
    children: list[OpDetail] = []
    for nid in node_ids[:3]:
        if nid in graph.nodes and nid in sim_results:
            node = graph.nodes[nid]
            sr = sim_results[nid]
            children.append(OpDetail(
                op_node_id=nid,
                op_type=node.op_type,
                scope=node.scope,
                layer=node.layer,
                input_shapes=[str(getattr(t, "shape", "?")) for t in node.inputs],
                output_shapes=[str(getattr(t, "shape", "?")) for t in node.outputs],
                shape_desc=describe_shapes(node),
                flops=sr.flops,
                read_bytes=sr.read_bytes,
                write_bytes=sr.write_bytes,
                compute_us=sr.compute_us,
                memory_us=sr.memory_us,
                latency_us=sr.latency_us,
                bound=sr.bound,
                confidence=sr.confidence,
            ))

    return OpFamilyDetail(
        op_type=op_type,
        display_name=display_name,
        category=category,
        count=count,
        repeat=repeat,
        first_scope=first_scope,
        shape_desc=shape_desc,
        formula=flops_formula,
        io_formula=io_formula,
        tflops=total_flops / 1e12 * repeat,
        hbm_bytes=(total_read + total_write) * repeat,
        comm_bytes=total_comm_bytes * repeat,
        compute_ms=total_compute_us / 1000.0 * repeat,
        memory_ms=total_memory_us / 1000.0 * repeat,
        comm_ms=total_comm_us / 1000.0 * repeat,
        total_ms=total_ms,
        bound=dominant_bound,
        confidence=avg_confidence,
        pct_of_substructure=pct,
        children=children,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: calibration / references / warnings
# ─────────────────────────────────────────────────────────────────────────────


def _build_calibration(
    rc: ReportContext,
    graph: "OpGraph",
    sim_results: "dict[str, SimResult]",
    profile: "Any | None",
) -> None:
    """Auto-populate calibration entries from available data."""
    entries: list[dict] = []

    # Layer count from profile
    if profile:
        num_layers = getattr(profile, "num_layers", 0)
        if num_layers > 0:
            entries.append({
                "metric": "层数",
                "official": str(num_layers),
                "modeled": str(num_layers),
                "unit": "layers",
                "source_name": "模型配置",
                "source_url": "#",
                "source_note": "config.json",
            })
        hidden = getattr(profile, "hidden_size", getattr(profile, "hidden", 0))
        if hidden > 0:
            entries.append({
                "metric": "Hidden Size",
                "official": str(hidden),
                "modeled": str(hidden),
                "unit": "dim",
                "source_name": "模型配置",
                "source_url": "#",
                "source_note": "config.json",
            })
        if getattr(profile, "is_moe", False):
            entries.append({
                "metric": "Routed Experts",
                "official": str(getattr(profile, "num_experts", "?")),
                "modeled": str(getattr(profile, "num_experts", "?")),
                "unit": "experts",
                "source_name": "模型配置",
                "source_url": "#",
                "source_note": "config.json",
            })
            entries.append({
                "metric": "Activated Experts/Token",
                "official": str(getattr(profile, "moe_topk", "?")),
                "modeled": str(getattr(profile, "moe_topk", "?")),
                "unit": "experts",
                "source_name": "模型配置",
                "source_url": "#",
                "source_note": "config.json",
            })

    # Total param count
    if profile:
        tp = getattr(profile, "total_param_count", 0)
        if tp > 0:
            entries.append({
                "metric": "Total Parameters",
                "official": f"{tp / 1e9:.1f}B",
                "modeled": f"{tp / 1e9:.1f}B",
                "unit": "params",
                "source_name": "模型配置",
                "source_url": "#",
                "source_note": "from model profile",
            })

    rc.calibration = entries


def _build_references(
    rc: ReportContext,
    model: str,
    hardware: str,
    hw_spec: "Any | None",
) -> None:
    """Auto-populate references from model and hardware info."""
    refs: list[dict] = []

    # Model reference
    if "deepseek" in model.lower():
        model_short = model.split("/")[-1] if "/" in model else model
        refs.append({
            "title": f"{model_short} on Hugging Face",
            "url": f"https://huggingface.co/{model}" if "/" in model
                  else f"https://huggingface.co/deepseek-ai/{model}",
            "note": "Official model card and config.json",
        })

    # Hardware reference
    if hw_spec and hasattr(hw_spec, "name"):
        hw_name = hw_spec.name
        hw_vendor = getattr(hw_spec, "vendor", "")
        if "nvidia" in hw_vendor.lower():
            refs.append({
                "title": f"NVIDIA {hw_name} specifications",
                "url": f"https://www.nvidia.com/en-us/data-center/{hw_name.lower().replace('_', '-')}/",
                "note": f"Official {hw_name} datasheet and whitepaper",
            })

    # System reference
    refs.append({
        "title": "ZRT-Sim Performance Modeling",
        "url": "#",
        "note": "Graph-based operator-level simulation with roofline/regression backends",
    })

    rc.references = refs


def _build_warnings(
    rc: ReportContext,
    phase: str,
    ctx: "Any | None",
    profile: "Any | None",
) -> None:
    """Auto-populate warnings from pipeline context."""
    warnings: list[str] = []

    if phase == "decode":
        warnings.append("Decode-only report: prefill stage intentionally omitted for this modeling iteration.")
    elif phase == "prefill":
        warnings.append("Prefill-only report: decode stage not included in this modeling iteration.")

    # MTP warning
    if ctx and hasattr(ctx, "training") and ctx.training:
        mtp_depth = getattr(ctx.training, "mtp_depth", 1)
        mtp_rate = getattr(ctx.training, "mtp_acceptance_rate", 0.0)
        if mtp_depth > 1:
            effective = 1.0 + (mtp_depth - 1) * mtp_rate
            warnings.append(
                f"MTP inference enabled: depth={mtp_depth}, acceptance_rate={mtp_rate:.2f}, "
                f"effective_tokens_per_decode_iteration={effective:.2f}. "
                f"TPOT/TPS are MTP-adjusted."
            )

    # Training-specific warnings
    if phase == "train":
        if ctx and hasattr(ctx, "training") and ctx.training:
            opt = getattr(ctx.training, "optimizer", "adam")
            zs = getattr(ctx.training, "zero_stage", 1)
            warnings.append(
                f"Training estimation: optimizer={opt}, ZeRO-stage={zs}. "
                f"Optimizer step overhead not included in step time."
            )
        warnings.append(
            "Training model uses graph-native estimation with Roofline backend. "
            "Actual performance depends on kernel implementations and cluster topology."
        )

    # Model size warning
    if profile:
        total_p = getattr(profile, "total_param_count", 0)
        if total_p > 500e9:
            warnings.append(
                f"Large model detected ({total_p / 1e9:.1f}B params). "
                f"Memory estimates should be validated against actual deployment constraints."
            )

    rc.warnings = warnings
