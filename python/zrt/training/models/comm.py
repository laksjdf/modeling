"""Alpha-beta communication model.

Per-collective cost using topology-aware latency and bandwidth.

The model has three layers, in order of generality:

1. :func:`collective_time` — flat α-β cost for one collective on ONE
   link (the leaf primitive; topology-class determines latency steps
   and bandwidth-derate).

2. :func:`collective_time_hierarchical` — legacy 2-tier decomposition
   (intra/inter) used when only group SIZE is known. Preserved
   bit-for-bit so existing anchors stay green.

3. :func:`collective_time_multi_tier` — N-tier hierarchical
   decomposition over an EXPLICIT rank set. Used when the caller
   provides a :class:`ParallelGroups` (built by
   ``zrt.training.topology.build_process_groups``). The collective
   decomposes innermost → outermost across each tier the group spans.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from zrt.hardware.spec import LinkSpec, TopologyTier
from zrt.training.ir.training_graph import Collective, Graph
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import SystemSpec

if TYPE_CHECKING:
    from zrt.training.topology.comm_domain import CommDomain
    from zrt.training.topology.process_groups import ParallelGroups


def collective_time(c: Collective, group_size: int, link: LinkSpec) -> float:
    """Return time in seconds for one collective operation.

    Alpha-beta model with NCCL bus-bandwidth convention:
      alpha = per-link latency (s); beta = 1 / bw_bytes_per_sec
      S = total post-collective bytes per rank (AG output, RS input)
      Bandwidth term for AG/RS = S × (N-1)/N × beta   (bus-bw form)

    Topology-class latency-step selection (``link.topology_class``):
      switched_full ("all_to_all"/"nvswitch"/"full_mesh"): single-step
        AG/RS:  α + S·(N-1)/N · β
        AR:     2α + 2·S·(N-1)/N · β
      switched_tree ("fat_tree"/"clos"): recursive-doubling / tree
        AG/RS:  ⌈log2 N⌉·α + S·(N-1)/N · β
        AR:     2·⌈log2 N⌉·α + 2·S·(N-1)/N · β
      ring/torus/unknown (default):
        AG/RS:  (N-1)·α + S·(N-1)/N · β
        AR:     2(N-1)·α + 2·S·(N-1)/N · β
      A2A: switched_full → α ; else Bruck (N>16, ⌈log2 N⌉) / pairwise (N-1).

    β uses ``link.effective_bw_bps(N)`` — peak × kb_efficiency, plus the
    non-clos switched-fabric over-subscription derate once N exceeds the
    non-blocking radix. P2P: rounds × (α + S·β).
    """
    alpha = link.latency_us * 1e-6
    bw_bytes = link.effective_bw_bps(group_size)
    beta = 1.0 / bw_bytes if bw_bytes > 0 else float("inf")

    N = group_size
    S = c.bytes_
    rounds = c.rounds

    if N <= 1 or S <= 0:
        return 0.0

    cls = link.topology_class
    if cls == "switched_full":
        steps = 1
    elif cls in ("switched_tree", "clos"):
        # clos: non-blocking switched → tree latency, full bandwidth.
        steps = max(1, math.ceil(math.log2(N)))
    else:  # ring / torus / unknown
        steps = N - 1
    bw_term = S * (N - 1) / N * beta  # NCCL bus-bw factor

    if c.kind in ("AG", "RS"):
        return steps * alpha + bw_term

    if c.kind == "AR":
        return 2 * steps * alpha + 2 * bw_term

    if c.kind == "A2A":
        if cls == "switched_full":
            return alpha + bw_term
        # NCCL/HCCL use Bruck (log2 rounds) above ~16 ranks; below that,
        # pairwise-exchange is competitive.
        if N > 16:
            return max(1, math.ceil(math.log2(N))) * alpha + bw_term
        return (N - 1) * alpha + bw_term

    if c.kind == "P2P":
        return rounds * (alpha + S * beta)

    return 0.0


def collective_time_hierarchical(
    c: Collective, group_size: int, system: SystemSpec,
) -> float:
    """AG/RS/AR over a (possibly multi-node) group using 2-level hierarchy.

    Megatron-Core / NCCL pattern when the group spans nodes:
      Stage 1 (intra-node, D ranks): each rank gathers/reduces its
        node's D shards over NVLink/NVSwitch.
      Stage 2 (inter-node, L = N/D nodes): node-leaders gather/reduce
        across nodes over IB.

    Falls back to flat ``collective_time`` for kinds we don't decompose
    (P2P / A2A — A2A hierarchy is non-trivial and not modeled here) or
    when the group fits in one node, or when N is not D-aligned.
    """
    intra = system.interconnect.intra_node
    inter = system.interconnect.inter_node
    D = intra.num_devices if intra.num_devices > 0 else system.gpus_per_node
    N = group_size
    S = c.bytes_

    if N <= D:
        return collective_time(c, N, intra)
    if N % D != 0 or c.kind not in ("AG", "RS", "AR"):
        return collective_time(c, N, inter)

    L = N // D

    if c.kind in ("AG", "RS"):
        # Stage 1: intra AG/RS of D ranks, output bytes per rank = S/L.
        # Stage 2: inter AG/RS of L nodes, output bytes per rank = S.
        intra_c = Collective(name=c.name + "_intra", kind=c.kind,
                             group=c.group, bytes_=S // L)
        inter_c = Collective(name=c.name + "_inter", kind=c.kind,
                             group=c.group, bytes_=S)
        return collective_time(intra_c, D, intra) + collective_time(inter_c, L, inter)

    # AR decomposes into RS + AG, each hierarchical.
    rs_c = Collective(name=c.name + "_rs", kind="RS", group=c.group, bytes_=S)
    ag_c = Collective(name=c.name + "_ag", kind="AG", group=c.group, bytes_=S)
    return (
        collective_time_hierarchical(rs_c, N, system)
        + collective_time_hierarchical(ag_c, N, system)
    )


# ─────────────────────────────────────────────────────────────────────
# N-tier multi-level α-β decomposition (for explicit rank sets)
# ─────────────────────────────────────────────────────────────────────

def _tier_breakdown(
    ranks: list[int], tiers: list[TopologyTier],
) -> list[tuple[int, LinkSpec]]:
    """Decompose a group's ranks into a per-tier branching schedule.

    Returns ``[(d_0, link_0), (d_1, link_1), ...]`` innermost → outermost,
    where ``d_i`` is the fanout the collective handles at tier ``i`` and
    ``link_i`` is that tier's interconnect link. The product of all
    ``d_i`` (skipping the suppressed ``d_i == 1`` steps) equals the
    group size ``N``.

    Algorithm: at each tier ``i`` with per-instance domain
    ``D_i = link.num_devices``, count how many tier-``i`` instances
    the group occupies (= ``|{r // D_i : r ∈ ranks}|``). The branching
    factor at tier ``i`` is ``inst_{i-1} / inst_i`` — i.e. how many
    tier-``i-1`` islands fit into one tier-``i`` island under the
    current group. A tier with ``D_i == 0`` is unbounded and folds
    "everything else" into one instance.

    For a 2-tier system with ``ranks = range(N)`` and ``N % D == 0``,
    this yields ``[(D, intra), (N/D, inter)]`` — matching the legacy
    :func:`collective_time_hierarchical` decomposition bit-for-bit.
    """
    if not ranks or not tiers:
        return []

    N = len(ranks)
    # Per-tier instance counts. The outermost tier is by convention
    # cluster-spanning (one global instance); we treat its
    # ``num_devices == 0`` *or* unset (default ``1`` in LinkSpec when the
    # legacy 2-tier ``inter_node`` is built without an explicit
    # num_devices) as "unbounded → one instance covers everything".
    inst_counts: list[int] = []
    last_idx = len(tiers) - 1
    for i, tier in enumerate(tiers):
        D = tier.link.num_devices
        is_outermost = i == last_idx
        if D == 0 or (is_outermost and D <= 1):
            inst_counts.append(1)  # unbounded → one instance covers everything
        else:
            inst_counts.append(len({r // D for r in ranks}))

    breakdown: list[tuple[int, LinkSpec]] = []
    prev_inst = N
    for tier, inst in zip(tiers, inst_counts):
        if inst <= 0:
            continue
        d_i = prev_inst // inst if inst > 0 else 1
        if d_i > 1:
            breakdown.append((d_i, tier.link))
        prev_inst = inst
    return breakdown


def collective_time_multi_tier(
    c: Collective,
    ranks: list[int],
    system: SystemSpec,
) -> float:
    """N-tier α-β cost for AG/RS/AR/P2P/A2A over an explicit rank set.

    AG/RS decomposes hierarchically across each tier the group spans
    (Megatron-Core / NCCL pattern, generalized to N ≥ 2 tiers): at
    each stage the per-rank output-byte volume is
    ``S × (∏_{j≤i} d_j) / N``. AR runs as RS + AG, each hierarchical.

    P2P / A2A pick the *outermost* tier the group spans and run flat —
    A2A hierarchy is non-trivial and not modeled here. Cost is
    additive across tier steps; this assumes the simulator wants a
    sequential decomposition (worst-case for non-overlapping
    implementations), matching the existing 2-tier formula.

    For a 2-tier system this is bit-for-bit identical to
    :func:`collective_time_hierarchical` when ``ranks = range(N)``
    and ``N`` is a multiple of the intra-node domain size — the
    common case the legacy path was tuned for.
    """
    N = len(ranks)
    S = c.bytes_
    if N <= 1 or S <= 0:
        return 0.0

    tiers = system.interconnect.tiers
    breakdown = _tier_breakdown(ranks, tiers)
    if not breakdown:
        # No useful decomposition (group fits a single trivial domain).
        return collective_time(c, N, system.interconnect.tiers[0].link)

    # P2P / A2A — pick the outermost spanned tier, run flat.
    if c.kind not in ("AG", "RS", "AR"):
        outermost_link = breakdown[-1][1]
        return collective_time(c, N, outermost_link)

    if c.kind == "AR":
        rs_c = Collective(name=c.name + "_rs", kind="RS", group=c.group, bytes_=S)
        ag_c = Collective(name=c.name + "_ag", kind="AG", group=c.group, bytes_=S)
        return (
            collective_time_multi_tier(rs_c, ranks, system)
            + collective_time_multi_tier(ag_c, ranks, system)
        )

    # AG / RS — hierarchical decomposition.
    total = 0.0
    prod_inner = 1
    for d_i, link_i in breakdown:
        prod_inner *= d_i
        out_bytes = S * prod_inner // N if N > 0 else 0
        sub_c = Collective(
            name=f"{c.name}_t{prod_inner}", kind=c.kind, group=c.group,
            bytes_=out_bytes,
        )
        total += collective_time(sub_c, d_i, link_i)
    return total


def tier_for_group(
    group: str, group_size: int, system: SystemSpec,
) -> LinkSpec:
    """Select the appropriate network link for a collective group.

    PP always uses inter_node. Otherwise, use intra_node if the group
    fits within the intra-node interconnect domain (num_devices),
    falling back to gpus_per_node when num_devices is unset or zero.
    """
    if group == "PP":
        return system.interconnect.inter_node
    intra_domain = system.interconnect.intra_node.num_devices
    if intra_domain <= 0:
        intra_domain = system.gpus_per_node
    if group_size <= intra_domain:
        return system.interconnect.intra_node
    return system.interconnect.inter_node


def total_comm_time(
    graph: Graph, model: ModelSpec, system: SystemSpec, strategy: Strategy,
    *,
    domain: "CommDomain | None" = None,
) -> dict[str, float]:
    """Return per-collective time in seconds.

    Pricing routes through :class:`CommDomain` so the dispatch (N-tier
    explicit-ranks vs legacy 2-tier size-only) lives in ONE place. The
    optional ``domain`` parameter lets the caller share a pre-built
    resolver with the rest of the estimate path
    (:func:`pipeline_step_time` does this so ParallelGroups is built
    once per estimate() call, not per function).
    """
    from zrt.training.topology.comm_domain import CommDomain
    if domain is None:
        domain = CommDomain(system=system, strategy=strategy)
    result: dict[str, float] = {}
    groups = domain.groups

    for c in graph.collectives:
        result[c.name] = domain.time(c)

    # DP gradient reduction (at step end)
    if strategy.dp > 1:
        P = _params_on_rank_for_dp(model, strategy)
        # Routed expert grads stay in the EP group (no DP AR) when EP > 1.
        # Subtract their per-rank contribution so the DP volume reflects only
        # non-expert + shared-expert grads.
        if strategy.ep > 1 and model.num_experts > 0:
            from zrt.training.models.memory import _routed_expert_params_on_rank
            P_expert_on_rank = _routed_expert_params_on_rank(model, strategy)
            P_dp = max(0, P - P_expert_on_rank)
        else:
            P_dp = P
        grad_bytes = int(P_dp * model.grad_dtype.bytes)
        dp_c = Collective(
            name="dp_grad_reduce", kind="AR" if strategy.zero_stage == 0 else "RS",
            group="DP", bytes_=grad_bytes, inserted_after="optimizer_step",
        )
        result[dp_c.name] = domain.time(dp_c)

    # Muon optimizer ZeRO-1 AllGather/ReduceScatter
    if strategy.optimizer.value == "muon" and strategy.dp > 1 and strategy.zero_stage >= 1:
        muon_comm = optimizer_comm_time(model, system, strategy, domain=domain)
        result["muon_ag"] = muon_comm.get("muon_ag", 0.0)
        result["muon_rs"] = muon_comm.get("muon_rs", 0.0)

    # PP P2P activation transfer between adjacent stages
    if strategy.pp > 1:
        result["pp_p2p"] = pp_p2p_time(model, system, strategy, domain=domain)

    return result


def optimizer_comm_time(
    model: ModelSpec, system: SystemSpec, strategy: Strategy,
    *,
    domain: "CommDomain | None" = None,
    groups: "ParallelGroups | None" = None,
) -> dict[str, float]:
    """Return Muon optimizer communication time in seconds.

    Whenever ZeRO stage ≥ 1 (1, 2, or FSDP/ZeRO-3), the FP32 master copy
    is sharded across DP, so Muon-rotation must:
      - AllGather full FP32 master params before the NS step
      - ReduceScatter updated params after the NS step (when rotation=True)

    The byte count and group sizes are identical across ZeRO 1/2/3 — they
    all shard the optimizer state by ``strategy.dp``. ZeRO-3 additionally
    shards forward-pass weights, but that AG is per-layer in fwd/bwd and
    is accounted for as graph-level collectives, not here.

    Sharding groups (matches Megatron-Core distributed optimizer):
      - Non-routed params (dense + shared experts + embed/lm_head):
        sharded across the full DP group of size ``strategy.dp``.
      - Routed-expert params: sharded across the **expert-DP** group. With EP
        enabled this is ``dp // ep`` after requiring a regular expert-DP
        layout; when ``dp == ep`` the expert-DP group has a single rank and
        routed AG/RS is free.

    Returns: dict with "muon_ag" and "muon_rs" time in seconds.
    """
    if strategy.optimizer.value != "muon" or strategy.dp <= 1 or strategy.zero_stage < 1:
        return {"muon_ag": 0.0, "muon_rs": 0.0}

    muon_config = strategy.muon_config
    f_muon = (
        muon_config.muon_param_fraction
        if muon_config and muon_config.muon_param_fraction is not None
        else 0.85
    )
    rotation = muon_config.rotation if muon_config else True
    param_bytes = 4  # FP32 master copy

    P_total = _params_on_rank_for_dp(model, strategy)

    if strategy.ep > 1 and model.num_experts > 0:
        from zrt.training.models.memory import _routed_expert_params_on_rank
        P_routed = _routed_expert_params_on_rank(model, strategy)
    else:
        P_routed = 0
    P_non_routed = max(0, P_total - P_routed)

    P_muon_non_routed = int(P_non_routed * f_muon)
    P_muon_routed = int(P_routed * f_muon)

    # Build/reuse the resolver. Callers from pipeline_step_time pass
    # `domain=` already built so groups are shared; legacy callers
    # (groups=…) get a temp resolver to keep back-compat.
    if domain is None:
        from zrt.training.topology.comm_domain import CommDomain
        domain = CommDomain(system=system, strategy=strategy)

    def _ag_rs(bytes_, group_kind: str) -> tuple[float, float]:
        """Cost the AG (+optional RS) over `group_kind` via the resolver."""
        group_size = domain.group_size(group_kind)
        if group_size <= 1 or bytes_ <= 0:
            return 0.0, 0.0
        ag_c = Collective(
            name="muon_ag", kind="AG", group=group_kind,
            bytes_=bytes_, inserted_after="backward_end",
        )
        rs_c = Collective(
            name="muon_rs", kind="RS", group=group_kind,
            bytes_=bytes_, inserted_after="optimizer_step",
        )
        ag = domain.time(ag_c)
        rs = domain.time(rs_c) if rotation else 0.0
        return ag, rs

    # Non-routed params share the full DP group; routed-expert params
    # are sharded across the **expert-DP** group (Megatron-Core
    # distributed optimizer convention) — size dp/ep, ranks orthogonal
    # to EP within a DP block. CommDomain knows both groups so the
    # tier resolution and α-β decomposition are correct for each.
    ag_dense, rs_dense = _ag_rs(P_muon_non_routed * param_bytes, "DP")
    ag_routed, rs_routed = _ag_rs(P_muon_routed * param_bytes, "EXPERT_DP")

    return {
        "muon_ag": ag_dense + ag_routed,
        "muon_rs": rs_dense + rs_routed,
    }


def pp_p2p_time(
    model: ModelSpec, system: SystemSpec, strategy: Strategy,
    *,
    domain: "CommDomain | None" = None,
    groups: "ParallelGroups | None" = None,
) -> float:
    """One-way P2P activation transfer time between adjacent pipeline stages (seconds).

    Activation shape per PP boundary: (micro_batch, seq_len/cp, hidden/tp).
    CP shards the sequence, so each rank only transfers its local token slice.

    Tier selection runs through :meth:`CommDomain.pp_p2p_link`:
    - For 3+ tier systems the PP group's primary tier is the inter-stage link.
    - For 2-tier systems the legacy ``tp*cp*2 ≤ intra_domain`` heuristic applies.

    ``groups`` kept for back-compat but ignored when ``domain`` is set.
    """
    if strategy.pp <= 1:
        return 0.0

    cp = max(strategy.cp, 1)
    act_bytes = (
        strategy.micro_batch * (model.seq_len // cp) * model.hidden
        * model.act_dtype.bytes // max(strategy.tp, 1)
    )

    if domain is None:
        from zrt.training.topology.comm_domain import CommDomain
        domain = CommDomain(system=system, strategy=strategy)
    link = domain.pp_p2p_link()

    bw_bytes = link.effective_bw_bps(2)  # adjacent-stage point-to-point
    if bw_bytes <= 0:
        return 0.0

    latency = link.latency_us * 1e-6
    return latency + act_bytes / bw_bytes


def _group_size_for(group: str, strategy: Strategy) -> int:
    if group == "TP":
        return strategy.tp
    if group == "CP":
        return strategy.cp
    if group == "EP":
        return strategy.ep
    if group == "DP":
        return strategy.dp
    if group == "PP":
        return 2  # P2P between adjacent stages
    return 1


def _params_on_rank_for_dp(model: ModelSpec, strategy: Strategy) -> int:
    """Params per rank for DP gradient reduce (after TP/EP/PP sharding).

    For MoE models, routed expert params are sharded by EP across ranks.
    The DP gradient reduction only needs to reduce gradients for params
    actually held on this rank (dense + shared experts + routed/EP).

    Mirrors the logic in memory._params_on_rank() for consistency.
    """
    from zrt.training.models.memory import (
        _dense_params, _shared_expert_params, _moe_params,
    )

    dense_params = _dense_params(model)
    moe_params = _moe_params(model)
    total = dense_params + moe_params

    # TP: shard all params by TP
    if strategy.tp > 1:
        total //= strategy.tp

    # EP: shard routed expert params by EP (shared experts NOT sharded by EP)
    if strategy.ep > 1 and model.num_experts > 0:
        shared_params = _shared_expert_params(model)
        routed_params = moe_params - shared_params
        if strategy.tp > 1:
            shared_params //= strategy.tp
            routed_params //= strategy.tp
        routed_after_ep = routed_params // strategy.ep
        moe_after_tp_ep = shared_params + routed_after_ep
        moe_after_tp = moe_params // strategy.tp if strategy.tp > 1 else moe_params
        total = (total - moe_after_tp) + moe_after_tp_ep

    # PP: only hold params for layers on this stage
    if strategy.pp > 1:
        n_layers = len(model.layers)
        embed_params = model.vocab * model.hidden * 2  # embed + lm_head
        non_embed = total - embed_params
        non_embed = int(non_embed * (n_layers / strategy.pp) / n_layers)
        total = non_embed + embed_params // strategy.pp

    # NOTE: do NOT divide by dp here. The alpha-beta formula
    #   T_RS = (N-1)·(α + S/N·β)
    # in collective_time() already divides S by N. Passing S = full_per_rank
    # gradient volume is what the textbook expects. ZeRO stage only changes
    # whether the collective is AR (zero=0) or RS (zero>=1); the input volume
    # to the collective is the same per-rank gradient produced by backward.
    return total
