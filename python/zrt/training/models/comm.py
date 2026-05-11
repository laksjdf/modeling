"""Alpha-beta communication model.

Per-collective cost using topology-aware latency and bandwidth.
"""

from __future__ import annotations

from zrt.hardware.spec import LinkSpec
from zrt.training.ir.training_graph import Collective, Graph
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import SystemSpec


def collective_time(c: Collective, group_size: int, link: LinkSpec) -> float:
    """Return time in seconds for one collective operation.

    Uses the alpha-beta model:
      alpha = per-link latency (seconds)
      beta = per-byte transfer time (seconds/byte) = 1 / (bw_bytes_per_sec)

    For P2P with multiple rounds (Ring CP), the total time is:
      rounds × (alpha + S × beta)

    Topology-aware algorithm selection:
      - "all_to_all" / "nvswitch" / "full_mesh": single-step gather/scatter
      - "ring" / "fat_tree" / default: ring allreduce ((N-1)*(α + S/N*β))

    Note: When P2P overlaps with compute, only exposed portion counts.
    """
    alpha = link.latency_us * 1e-6
    bw_bytes = link.bandwidth_gbps * 1e9 / 8
    beta = 1.0 / bw_bytes if bw_bytes > 0 else float("inf")

    N = group_size
    S = c.bytes_
    rounds = c.rounds

    full_connectivity = link.topology in ("all_to_all", "nvswitch", "full_mesh")

    if c.kind in ("AG", "RS"):
        if full_connectivity:
            return alpha + (S / N) * beta
        return (N - 1) * (alpha + (S / N) * beta)

    if c.kind == "AR":
        if full_connectivity:
            return 2 * (alpha + (S / N) * beta)
        return 2 * (N - 1) * (alpha + (S / N) * beta)

    if c.kind == "A2A":
        return (N - 1) * (alpha + (S / N) * beta)

    if c.kind == "P2P":
        return rounds * (alpha + S * beta)

    return 0.0


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
) -> dict[str, float]:
    """Return per-collective time in seconds."""
    result: dict[str, float] = {}

    for c in graph.collectives:
        group_size = _group_size_for(c.group, strategy)
        link = tier_for_group(c.group, group_size, system)
        t = collective_time(c, group_size, link)
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
        link = tier_for_group("DP", group_size, system)
        result[dp_c.name] = collective_time(dp_c, group_size, link)

    # Muon optimizer ZeRO-1 AllGather/ReduceScatter
    if strategy.optimizer.value == "muon" and strategy.dp > 1 and strategy.zero_stage >= 1:
        muon_comm = optimizer_comm_time(model, system, strategy)
        result["muon_ag"] = muon_comm.get("muon_ag", 0.0)
        result["muon_rs"] = muon_comm.get("muon_rs", 0.0)

    # PP P2P activation transfer between adjacent stages
    if strategy.pp > 1:
        result["pp_p2p"] = pp_p2p_time(model, system, strategy)

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
    link = tier_for_group("DP", group_size, system)

    ag_c = Collective(
        name="muon_ag", kind="AG", group="DP", bytes_=comm_bytes,
        inserted_after="backward_end",
    )
    ag_time = collective_time(ag_c, group_size, link)

    # ReduceScatter only if rotation=True (Moonshot optimization)
    # When rotation=False, each rank independently computes and slices, no RS needed
    rotation = muon_config.rotation if muon_config else True
    if rotation:
        rs_c = Collective(
            name="muon_rs", kind="RS", group="DP", bytes_=comm_bytes,
            inserted_after="optimizer_step",
        )
        rs_time = collective_time(rs_c, group_size, link)
    else:
        rs_time = 0.0

    return {
        "muon_ag": ag_time,
        "muon_rs": rs_time,
    }


def pp_p2p_time(model: ModelSpec, system: SystemSpec, strategy: Strategy) -> float:
    """One-way P2P activation transfer time between adjacent pipeline stages (seconds).

    Activation shape per PP boundary: (micro_batch, seq_len/cp, hidden/tp).
    CP shards the sequence, so each rank only transfers its local token slice.

    Tier selection: in Megatron layout [DP, PP, CP, TP] (outer→inner), adjacent PP
    stages are separated by cp*tp GPUs. They are intra-node only when both stages'
    full CP×TP groups fit within one node: cp*tp*2 <= gpus_per_node.

    Returns the per-microbatch P2P overhead added to each stage's fwd and bwd time.
    """
    if strategy.pp <= 1:
        return 0.0

    cp = max(strategy.cp, 1)
    act_bytes = (
        strategy.micro_batch * (model.seq_len // cp) * model.hidden
        * model.act_dtype.bytes // max(strategy.tp, 1)
    )

    # Adjacent PP stages span cp*tp GPUs; intra-node only if both fit in one node.
    intra_domain = system.interconnect.intra_node.num_devices
    if intra_domain <= 0:
        intra_domain = system.gpus_per_node
    if strategy.tp * cp * 2 <= intra_domain:
        link = system.interconnect.intra_node
    else:
        link = system.interconnect.inter_node

    if link.bandwidth_gbps <= 0:
        return 0.0

    bw_bytes = link.bandwidth_gbps * 1e9 / 8
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

    if strategy.zero_stage >= 2:
        total //= strategy.dp
    return total
