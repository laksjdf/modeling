"""Communication-domain construction for the training simulator.

Given world_size and a Strategy, this package builds the **explicit set of
ranks** that form each parallel-group (TP/CP/EP/DP/PP) and reports which
network tier(s) each group spans.

The construction mirrors Megatron-Core's ``parallel_state.RankGenerator``:
the world is laid out as a multi-dimensional rank grid whose axes are the
parallel degrees, ordered innermost → outermost. A rank's coordinates
determine its membership in every subgroup. The innermost axis maps to
the fastest interconnect tier; the outermost to the slowest.

Public API::

    from zrt.training.topology import build_process_groups, ParallelGroups

    groups = build_process_groups(world_size, strategy, system)
    # groups.tp_groups[i] → list of ranks in TP group i
    # groups.group_tier["TP"].primary → tier index this group rides on
"""
from zrt.training.topology.comm_domain import (
    CommDomain,
    build_comm_domain,
    comm_domain_report,
    format_comm_domain_entry,
)
from zrt.training.topology.process_groups import (
    DEFAULT_PARALLEL_ORDER,
    GroupTierAssignment,
    ParallelGroups,
    build_process_groups,
)

__all__ = [
    "CommDomain",
    "build_comm_domain",
    "comm_domain_report",
    "format_comm_domain_entry",
    "GroupTierAssignment",
    "ParallelGroups",
    "build_process_groups",
    "DEFAULT_PARALLEL_ORDER",
]
