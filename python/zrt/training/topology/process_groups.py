"""Process-group (communication-domain) construction.

Mirrors Megatron-Core's ``parallel_state.RankGenerator`` /
``generate_masked_orthogonal_rank_groups`` algorithm. The world is
treated as a multi-dimensional rank grid; each parallel-degree
(TP/CP/EP/DP/PP) is one axis. A subgroup is the set of ranks that
agree on every axis *except* the one the subgroup is parallelizing
over.

Default order — innermost to outermost — is ``"tp-cp-ep-dp-pp"``,
matching Megatron-Core's default. The innermost axis varies fastest
(consecutive global ranks belong to the same TP island), the outermost
axis varies slowest (PP stages are far apart in rank-space). This is
what lets the implementation give TP to NVLink and PP to inter-node IB
without ever talking to the user about tiers explicitly.

Outputs:
- ``ParallelGroups`` — for each parallelism kind, the list-of-rank-lists.
- ``GroupTierAssignment`` — for each kind, the innermost interconnect
  tier whose per-instance domain fully contains one group instance, plus
  a flag indicating whether the group is "tier-aligned" (its ranks all
  fall within a single instance of that tier).

This module deliberately knows nothing about collective costs — it only
enumerates rank sets and tier mapping. ``zrt.training.models.comm``
consumes the output to price each collective.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from zrt.hardware.spec import InterconnectSpec, TopologyTier
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import SystemSpec


# Megatron-Core default: tp innermost → pp outermost. Each token is a
# parallelism kind. EP shares ranks with DP/CP/TP (it does not consume
# new ranks — see Strategy.rank_product comments) but it does live as
# its own axis in the grid that defines EP groups.
DEFAULT_PARALLEL_ORDER: tuple[str, ...] = ("tp", "cp", "ep", "dp", "pp")

_ALL_GROUPS: tuple[str, ...] = ("TP", "CP", "EP", "DP", "PP")


@dataclass
class GroupTierAssignment:
    """Which interconnect tier(s) one group instance spans.

    ``primary_tier`` — the innermost tier whose per-instance domain
    contains the *whole* group (i.e. one collective on this group stays
    within one tier instance). Used by the cost model as the "outermost"
    tier the collective must reach when running flat.

    ``inner_tier`` — the smallest tier the group *also* spans, i.e. the
    first tier where hierarchical decomposition can do useful local work
    before reaching ``primary_tier``.

    ``tier_aligned`` — True when the group's rank set falls inside a
    single instance of ``primary_tier``. Always True in the construction
    here because the algorithm picks ``primary_tier`` to be the smallest
    tier that fits. Provided as an explicit field so future relaxations
    (non-aligned PP striping etc.) can flag misalignment.
    """
    group: str
    group_size: int
    primary_tier: int       # tier index in system.interconnect.tiers
    inner_tier: int         # smallest tier this group also spans
    tier_aligned: bool = True


@dataclass
class ParallelGroups:
    """Enumerated rank sets + tier assignments for a (world, strategy) pair.

    Each ``*_groups`` field is a list of rank-lists: outer index is the
    group instance index, inner list is the ranks belonging to that
    instance. For TP/CP/EP/DP/PP, ranks in one inner list form one
    collective communication domain.

    Beyond the five primary kinds, two derived groups are populated:

    - ``expert_dp_groups``: the **expert-DP** group used by Megatron-Core's
      distributed optimizer for routed-expert params. Each group has
      size ``dp // ep`` (1 if ``ep >= dp``) and contains the ranks that
      hold the SAME routed-expert shard — i.e. the orthogonal partition
      of a DP group by EP blocks. Used to price Muon AG/RS for routed
      params instead of the EP group.

    Example for ``world_size=16, strategy(tp=2, cp=1, ep=2, dp=4, pp=2)``
    with default order::

        dp_groups = [[0,2,4,6], [1,3,5,7], [8,10,12,14], [9,11,13,15]]
        ep_groups = [[0,2], [4,6], [1,3], [5,7], [8,10], [12,14], [9,11], [13,15]]
        expert_dp_groups = [[0,4], [2,6], [1,5], [3,7], [8,12], [10,14], [9,13], [11,15]]
    """
    tp_groups: list[list[int]] = field(default_factory=list)
    cp_groups: list[list[int]] = field(default_factory=list)
    ep_groups: list[list[int]] = field(default_factory=list)
    dp_groups: list[list[int]] = field(default_factory=list)
    pp_groups: list[list[int]] = field(default_factory=list)
    # Derived: expert-DP / non-routed-DP / ...
    expert_dp_groups: list[list[int]] = field(default_factory=list)

    tier: dict[str, GroupTierAssignment] = field(default_factory=dict)
    world_size: int = 0
    parallel_order: tuple[str, ...] = DEFAULT_PARALLEL_ORDER

    def ranks_for(self, group_kind: str, rank: int) -> list[int]:
        """Return the group instance containing ``rank`` for ``group_kind``.

        ``group_kind ∈ {"TP", "CP", "EP", "DP", "PP", "EXPERT_DP"}``.
        Returns an empty list if the group is degenerate (size 1) or
        the rank is out of bounds.
        """
        kind = group_kind.upper()
        attr = self._kind_to_attr(kind)
        groups = getattr(self, attr, None)
        if groups is None:
            raise ValueError(f"Unknown group kind: {group_kind}")
        for g in groups:
            if rank in g:
                return list(g)
        return []

    def group_size(self, group_kind: str) -> int:
        kind = group_kind.upper()
        attr = self._kind_to_attr(kind)
        groups = getattr(self, attr, None)
        if not groups:
            return 1
        return len(groups[0])

    @staticmethod
    def _kind_to_attr(kind: str) -> str:
        """Map a user-facing kind ("EXPERT_DP" / "TP" / …) to the field name."""
        if kind in ("EXPERT_DP", "EDP"):
            return "expert_dp_groups"
        return f"{kind.lower()}_groups"


def _degree(strategy: Strategy, axis: str) -> int:
    """Map an axis name (tp/cp/ep/dp/pp) to the strategy's degree value."""
    return {
        "tp": strategy.tp,
        "cp": strategy.cp,
        "ep": strategy.ep,
        "dp": strategy.dp,
        "pp": strategy.pp,
    }[axis]


def _build_groups_for_axis(
    *,
    target_axis: str,
    order: Sequence[str],
    degrees: dict[str, int],
    world_size: int,
) -> list[list[int]]:
    """Enumerate the rank-lists for one parallelism axis.

    Implements ``generate_masked_orthogonal_rank_groups`` for a single
    axis. The rank grid layout is column-major in ``order``: the
    innermost-listed axis has stride 1 and the outermost has stride
    ``∏(inner degrees)``. Ranks differing only in ``target_axis`` form
    one group; iterating all the *other* axes yields one group per
    fixed coordinate tuple.

    Important — EP semantics: EP shares physical ranks with the inner
    "tp-cp" block (EP groups are typically built across the DP axis with
    TP held fixed). The current implementation treats EP as its own axis
    of size ``ep`` between ``cp`` and ``dp`` per Megatron's
    ``generate_masked_orthogonal_rank_groups`` — i.e. ``dp_effective =
    dp / ep`` in the placement and ``ep`` re-uses ranks within the
    original dp×cp×tp slab. We keep that convention here. Callers that
    set ``ep > dp`` are rejected upstream (Strategy.validate / search
    util).
    """
    # Strides per axis, given current order (innermost first).
    strides: dict[str, int] = {}
    stride = 1
    for ax in order:
        strides[ax] = stride
        stride *= degrees[ax]
    assert stride == world_size, (
        f"order={order} degrees={degrees} → product={stride} != world={world_size}"
    )

    target_deg = degrees[target_axis]
    if target_deg <= 1:
        # Degenerate axis — every rank is its own group of one.
        return [[r] for r in range(world_size)]

    target_stride = strides[target_axis]
    other_axes = [ax for ax in order if ax != target_axis]
    other_extent = world_size // target_deg

    groups: list[list[int]] = []
    # Iterate over all "other" coordinate tuples by counting from 0 to
    # other_extent-1 in the *compressed* mixed-radix where the target
    # axis is removed. Easier: enumerate every rank, group by everything
    # except target_axis.
    visited: set[int] = set()
    for base in range(world_size):
        if base in visited:
            continue
        # The group containing `base` is the orbit under varying the
        # target_axis: base ± k * target_stride, k ∈ [0, target_deg).
        # But we must keep base anchored at target_axis == 0 to avoid
        # duplicate seeds.
        base_target_coord = (base // target_stride) % target_deg
        seed = base - base_target_coord * target_stride
        if seed in visited:
            continue
        members = [seed + k * target_stride for k in range(target_deg)]
        for m in members:
            visited.add(m)
        groups.append(members)

    return groups


def _tier_for_groups(
    groups: list[list[int]],
    interconnect: InterconnectSpec,
) -> GroupTierAssignment:
    """Pick the innermost tier that contains *every* group instance.

    For each tier with bounded per-instance domain ``num_devices = N``,
    a group instance fits if ``max(ranks) // N == min(ranks) // N``
    (min and max land in the same tier-instance, so all intermediate
    ranks do too). All group instances of a given parallelism kind must
    fit before we accept the tier — otherwise the simulator would
    underestimate cost for the misaligned instances.

    An outermost tier with ``num_devices == 0`` is treated as unbounded
    and always matches.

    Also reports ``inner_tier``: the deepest tier the group can fully
    decompose into. For default Megatron layout this is tier 0 when
    the group's per-instance ranks all land in one innermost domain;
    otherwise it equals ``primary_tier`` (no hierarchical decomposition).
    """
    if not interconnect.tiers:
        raise ValueError("Interconnect has no tiers configured")
    if not groups or not groups[0]:
        return GroupTierAssignment(
            group="", group_size=0,
            primary_tier=0, inner_tier=0, tier_aligned=True,
        )

    size = len(groups[0])
    if size <= 1:
        return GroupTierAssignment(
            group="", group_size=1,
            primary_tier=0, inner_tier=0, tier_aligned=True,
        )

    def _all_fit(n: int) -> bool:
        if n <= 0:
            return True
        return all((max(g) // n) == (min(g) // n) for g in groups)

    primary = len(interconnect.tiers) - 1
    last_idx = len(interconnect.tiers) - 1
    for i, tier in enumerate(interconnect.tiers):
        n = tier.link.num_devices
        # Outermost tier is conventionally cluster-spanning: treat its
        # num_devices ≤ 1 (the LinkSpec dataclass default when the legacy
        # 2-tier inter_node was built without an explicit value) as
        # "unbounded → matches anything".
        if (n == 0) or (i == last_idx and n <= 1) or _all_fit(n):
            primary = i
            break

    # inner_tier: innermost tier that provides a real local fanout for
    # every group instance. If lower tiers only contain one rank per
    # instance, decomposition must start at the primary tier or later.
    inner = primary
    if primary > 0:
        for i, tier in enumerate(interconnect.tiers[:primary]):
            n = tier.link.num_devices
            if all(len({r // n for r in g}) < len(g) for g in groups):
                inner = i
                break

    return GroupTierAssignment(
        group="", group_size=size,
        primary_tier=primary,
        inner_tier=inner,
        tier_aligned=True,
    )


def build_process_groups(
    world_size: int,
    strategy: Strategy,
    system: SystemSpec | None = None,
    order: Sequence[str] = DEFAULT_PARALLEL_ORDER,
) -> ParallelGroups:
    """Construct all communication domains for a (world, strategy) pair.

    Parameters
    ----------
    world_size
        Total number of ranks. Must equal ``tp * cp * pp * dp`` (EP shares
        ranks; see :func:`zrt.training.spec.strategy.rank_product`).
    strategy
        Parallel-degree configuration.
    system
        Interconnect-aware system spec. When provided, fills
        ``ParallelGroups.tier`` with per-group tier assignments. When
        ``None`` the tier field stays empty (useful in unit tests).
    order
        Axis order, innermost → outermost. Default is Megatron's
        ``"tp-cp-ep-dp-pp"``. The user-facing surface stays default-only
        per current scope; this parameter is exposed for tests.

    Returns
    -------
    ParallelGroups
        With every ``*_groups`` populated. Each is a list of rank-lists,
        one inner list per group instance.
    """
    order = tuple(order)
    degrees = {ax: _degree(strategy, ax) for ax in order}

    # EP shares ranks with the dp×cp×tp slab — the Megatron convention
    # is that EP groups span the DP axis with TP/CP held fixed. The
    # placement here treats EP as occupying the slot between CP and DP
    # in the rank grid; degrees[dp] is the effective DP after EP shares
    # ranks with it. We keep it that way and let the caller pre-validate
    # rank_product (Strategy.validate) — i.e. tp*cp*pp*dp == world_size.
    prod = 1
    for ax in order:
        if ax == "ep":
            continue  # EP shares ranks, doesn't multiply world
        prod *= degrees[ax]
    if prod != world_size:
        raise ValueError(
            f"world_size={world_size} but tp*cp*pp*dp={prod} "
            f"(strategy: tp={strategy.tp}, cp={strategy.cp}, "
            f"pp={strategy.pp}, dp={strategy.dp})"
        )

    # For the rank-grid enumeration, treat the EP axis as a degree-1
    # axis (i.e. it does not change the stride layout). EP groups are
    # built separately below from the DP axis (the Megatron convention).
    grid_degrees = dict(degrees)
    grid_degrees["ep"] = 1
    # Build groups for axes that *do* take their own stride.
    tp = _build_groups_for_axis(
        target_axis="tp", order=order, degrees=grid_degrees, world_size=world_size,
    )
    cp = _build_groups_for_axis(
        target_axis="cp", order=order, degrees=grid_degrees, world_size=world_size,
    )
    dp = _build_groups_for_axis(
        target_axis="dp", order=order, degrees=grid_degrees, world_size=world_size,
    )
    pp = _build_groups_for_axis(
        target_axis="pp", order=order, degrees=grid_degrees, world_size=world_size,
    )

    # EP groups: built across the DP axis with TP/CP/PP held fixed; ep
    # consecutive DP ranks form one EP group, ep groups in total per
    # (TP, CP, PP) coordinate. When ep == dp, the EP group is exactly
    # the DP group; when ep < dp, each DP group is partitioned into
    # dp/ep disjoint EP groups; when ep == 1, EP is degenerate.
    #
    # Expert-DP groups (Megatron-Core distributed-optimizer convention)
    # are the **orthogonal** partition: take the i-th rank from each
    # EP block within a DP group to form one expert-DP group of size
    # dp/ep. There are ep such expert-DP groups per DP group, each
    # holding the same routed-expert shard. Used by Muon AG/RS for
    # routed params — different group from EP both in size and ranks.
    if degrees["ep"] <= 1 or degrees["dp"] <= 1:
        ep_groups: list[list[int]] = [[r] for r in range(world_size)]
        expert_dp_groups: list[list[int]] = (
            [list(g) for g in dp] if degrees["dp"] > 1
            else [[r] for r in range(world_size)]
        )
    else:
        ep_groups = []
        expert_dp_groups = []
        ep_deg = degrees["ep"]
        for dp_group in dp:
            # EP groups: consecutive `ep` ranks within the DP group.
            blocks: list[list[int]] = []
            for i in range(0, len(dp_group), ep_deg):
                block = dp_group[i:i + ep_deg]
                ep_groups.append(block)
                blocks.append(block)
            # Expert-DP groups: i-th rank from each EP block →
            # one expert-DP group of size dp/ep = len(blocks).
            for i in range(ep_deg):
                expert_dp_groups.append([blk[i] for blk in blocks if i < len(blk)])

    groups = ParallelGroups(
        tp_groups=tp,
        cp_groups=cp,
        ep_groups=ep_groups,
        dp_groups=dp,
        pp_groups=pp,
        expert_dp_groups=expert_dp_groups,
        world_size=world_size,
        parallel_order=tuple(order),
    )

    if system is not None:
        groups.tier = _assign_tiers(groups, system.interconnect)

    return groups


def _assign_tiers(
    groups: ParallelGroups,
    interconnect: InterconnectSpec,
) -> dict[str, GroupTierAssignment]:
    """Compute the tier each parallel-group instance rides on.

    All instances of a given group kind share the same tier (they have
    the same rank-span by construction). Returns one assignment per
    kind in ``{"TP","CP","EP","DP","PP"}``.
    """
    out: dict[str, GroupTierAssignment] = {}
    for kind, lst in (
        ("TP", groups.tp_groups),
        ("CP", groups.cp_groups),
        ("EP", groups.ep_groups),
        ("DP", groups.dp_groups),
        ("PP", groups.pp_groups),
        ("EXPERT_DP", groups.expert_dp_groups),
    ):
        if not lst:
            continue
        sample = lst[0]
        if len(sample) <= 1:
            # Degenerate — pin to innermost tier; collective cost will
            # be zero anyway.
            out[kind] = GroupTierAssignment(
                group=kind, group_size=1,
                primary_tier=0, inner_tier=0,
                tier_aligned=True,
            )
            continue
        assign = _tier_for_groups(lst, interconnect)
        assign.group = kind
        out[kind] = assign
    return out
