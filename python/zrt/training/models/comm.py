"""Alpha-beta communication model.

Per-collective cost using topology-aware latency and bandwidth.
"""

from __future__ import annotations

from zrt.training.ir.training_graph import Collective, Graph
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import NetTier, SystemSpec


def collective_time(c: Collective, group_size: int, tier: NetTier) -> float:
    """Return time in seconds for one collective operation.

    Uses the alpha-beta model:
      alpha = per-link latency (seconds)
      beta = per-byte transfer time (seconds/byte) = 1 / (bw_bytes_per_sec)
    """
    alpha = tier.latency_us * 1e-6
    bw_bytes = tier.bw_gbps * 1e9 / 8  # convert GB/s to bytes/s
    beta = 1.0 / bw_bytes if bw_bytes > 0 else float("inf")

    N = group_size
    S = c.bytes_

    if c.kind in ("AG", "RS"):
        # Ring algorithm: (N-1) steps of S/N bytes each
        return (N - 1) * (alpha + (S / N) * beta)

    if c.kind == "AR":
        # Ring all-reduce = AG + RS
        return 2 * (N - 1) * (alpha + (S / N) * beta)

    if c.kind == "A2A":
        # Each rank sends S/N to each of N-1 peers
        return (N - 1) * (alpha + (S / N) * beta)

    if c.kind == "P2P":
        return alpha + S * beta

    return 0.0


def tier_for_group(
    group: str, group_size: int, system: SystemSpec,
) -> NetTier:
    """Select the appropriate network tier for a collective group.

    If all ranks in the group are on one node, use intra_node.
    Otherwise use inter_node.
    """
    intra = system.intra_tier()
    inter = system.inter_tier()

    if group in ("TP",) and group_size <= system.gpus_per_node:
        # TP is typically within a node
        return intra if intra else inter

    if group in ("DP",) and group_size <= system.gpus_per_node:
        return intra if intra else inter

    if group in ("PP",):
        # PP P2P: depends on rank placement; default to inter
        return inter if inter else intra

    # Default: if group fits in one node use intra, else inter
    if group_size <= system.gpus_per_node and intra:
        return intra
    return inter if inter else (intra if intra else NetTier("default", 0, 0, "unknown"))


def total_comm_time(
    graph: Graph, model: ModelSpec, system: SystemSpec, strategy: Strategy,
) -> dict[str, float]:
    """Return per-collective time in seconds."""
    result: dict[str, float] = {}

    for c in graph.collectives:
        group_size = _group_size_for(c.group, strategy)
        tier = tier_for_group(c.group, group_size, system)
        t = collective_time(c, group_size, tier)
        result[c.name] = t

    # DP gradient reduction (at step end)
    if strategy.dp > 1:
        P = _params_on_rank_for_dp(model, strategy)
        grad_bytes = P * model.grad_dtype.bytes
        dp_c = Collective(
            name="dp_grad_reduce", kind="AR" if strategy.zero_stage == 0 else "RS",
            group="DP", bytes_=grad_bytes, inserted_after="optimizer_step",
        )
        group_size = strategy.dp
        tier = tier_for_group("DP", group_size, system)
        result[dp_c.name] = collective_time(dp_c, group_size, tier)

    # Muon optimizer ZeRO-1 AllGather/ReduceScatter
    if strategy.optimizer.value == "muon" and strategy.dp > 1 and strategy.zero_stage >= 1:
        muon_comm = optimizer_comm_time(model, system, strategy)
        result["muon_ag"] = muon_comm.get("muon_ag", 0.0)
        result["muon_rs"] = muon_comm.get("muon_rs", 0.0)

    return result


def optimizer_comm_time(
    model: ModelSpec, system: SystemSpec, strategy: Strategy,
) -> dict[str, float]:
    """Return Muon optimizer communication time in seconds.

    Muon with ZeRO-1 requires:
      - AllGather before optimizer step to gather full Muon parameters
      - ReduceScatter after optimizer step to distribute updated gradients

    Communication volume per step:
      - AG: (DP-1)/DP × P_muon × 4B
      - RS: (DP-1)/DP × P_muon × 4B (same volume)

    Args:
        model: ModelSpec with hidden dimension and param count
        system: SystemSpec with network tiers
        strategy: Strategy with optimizer config and muon_config

    Returns:
        Dict with "muon_ag" and "muon_rs" time in seconds
    """
    if strategy.optimizer.value != "muon" or strategy.dp <= 1 or strategy.zero_stage < 1:
        return {"muon_ag": 0.0, "muon_rs": 0.0}

    muon_config = strategy.muon_config
    f_muon = (
        muon_config.muon_param_fraction
        if muon_config and muon_config.muon_param_fraction is not None
        else 0.85
    )

    P = _params_on_rank_for_dp(model, strategy)
    P_muon = int(P * f_muon)
    param_bytes = 4  # FP32 master copy

    # Total bytes to gather = P_muon × 4B
    # Ring algorithm factor applied in collective_time(), not pre-scaled here
    comm_bytes = int(P_muon * param_bytes)

    group_size = strategy.dp
    tier = tier_for_group("DP", group_size, system)

    ag_c = Collective(
        name="muon_ag", kind="AG", group="DP", bytes_=comm_bytes,
        inserted_after="backward_end",
    )
    ag_time = collective_time(ag_c, group_size, tier)

    # ReduceScatter only if rotation=True (Moonshot optimization)
    # When rotation=False, each rank independently computes and slices, no RS needed
    rotation = muon_config.rotation if muon_config else True
    if rotation:
        rs_c = Collective(
            name="muon_rs", kind="RS", group="DP", bytes_=comm_bytes,
            inserted_after="optimizer_step",
        )
        rs_time = collective_time(rs_c, group_size, tier)
    else:
        rs_time = 0.0

    return {
        "muon_ag": ag_time,
        "muon_rs": rs_time,
    }


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
    """Params per rank for DP gradient reduce (after TP/PP sharding)."""
    P = model.total_params()
    if strategy.tp > 1:
        P //= strategy.tp
    if strategy.pp > 1:
        n_layers = len(model.layers)
        embed = model.vocab * model.hidden * 2
        non_embed = P - embed
        non_embed = int(non_embed * (n_layers / strategy.pp) / n_layers)
        P = non_embed + embed // strategy.pp
    if strategy.zero_stage >= 2:
        P //= strategy.dp
    return P
