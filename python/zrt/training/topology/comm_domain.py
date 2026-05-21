"""Unified communication-domain interface.

Every comm-cost consumer in the estimate path (per-stage collectives,
EP-A2A totals, DP grad reduce, optimizer AG/RS, PP P2P) routes through
:class:`CommDomain`. Built once per ``(system, strategy)``, it owns
the :class:`ParallelGroups` (Megatron-style explicit rank sets) and
exposes a single ``time(collective)`` entry point so call sites stop
re-deriving "which tier does this group ride".

Design goals
------------
- **No duplication**: tier-selection logic lives in ONE place.
- **Cheap construction**: ``ParallelGroups`` is enumerated lazily,
  once per domain, then cached. Search workers can build many domains
  per second.
- **Back-compat**: for 2-tier interconnects, ``CommDomain.time(c)``
  delegates to the legacy size-only formula (bit-exact). For 3+ tier
  interconnects, it uses the explicit rank set via
  :func:`collective_time_multi_tier`.
- **Graceful degradation**: when the strategy doesn't satisfy
  ``tp*cp*pp*dp == world_size`` (validation skipped) the domain
  silently falls back to legacy SIZE-only pricing.

Public surface
--------------
``domain.time(c)``              — cost a single :class:`Collective`.
``domain.ranks(kind)``          — explicit rank list for one group of ``kind``.
``domain.link(kind)``           — :class:`LinkSpec` of the resolved tier.
``domain.tier(kind)``           — tier index in ``system.interconnect.tiers``.
``domain.pp_p2p_link()``        — adjacent-stage PP P2P link.
``domain.group_size(kind)``     — degree per parallel kind.

All read-only after construction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from zrt.hardware.spec import LinkSpec
from zrt.training.ir.training_graph import Collective
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import SystemSpec

if TYPE_CHECKING:
    from zrt.training.topology.process_groups import ParallelGroups


logger = logging.getLogger(__name__)
_VALID_KINDS = ("TP", "CP", "EP", "DP", "PP", "EXPERT_DP")
_REPORT_KINDS = ("EP", "PP", "DP", "TP", "CP")


@dataclass
class CommDomain:
    """Communication-domain resolver for a ``(system, strategy)`` pair.

    Instances are immutable after construction (``_groups`` is the only
    field that mutates, via lazy init in :meth:`groups`). Safe to share
    across threads as long as :meth:`groups` is called once before
    concurrent reads.
    """

    system: SystemSpec
    strategy: Strategy

    # ─ Lazy state ────────────────────────────────────────────────────
    _groups: "ParallelGroups | None" = None
    _use_multi_tier: bool | None = None

    # ─ Construction ─────────────────────────────────────────────────

    def __post_init__(self) -> None:
        # Decide once whether the explicit-ranks N-tier path applies.
        # Only triggered for 3+ tier interconnects to keep 2-tier anchors
        # bit-exact with prior behavior.
        n_tiers = len(self.system.interconnect.tiers)
        self._use_multi_tier = n_tiers >= 3

    @property
    def groups(self) -> "ParallelGroups":
        """Lazily-built :class:`ParallelGroups` (the explicit rank sets).

        Built on first access. Returns an EMPTY ParallelGroups when the
        strategy is degenerate (world-size mismatch etc.) so the
        ``ranks``/``link``/``tier`` queries silently fall back to the
        size-only legacy path.
        """
        if self._groups is None:
            from zrt.training.topology.process_groups import (
                ParallelGroups,
                build_process_groups,
            )
            try:
                self._groups = build_process_groups(
                    self.system.world_size, self.strategy, self.system,
                )
            except ValueError:
                # Degenerate (world_size mismatch) — return an empty
                # ParallelGroups so the resolver falls back gracefully.
                self._groups = ParallelGroups(world_size=self.system.world_size)
        return self._groups

    # ─ Queries ──────────────────────────────────────────────────────

    def group_size(self, kind: str) -> int:
        """Degree of one parallel dimension; size of one group instance.

        ``EXPERT_DP`` uses Megatron-Core's distributed-optimizer convention
        for routed-expert AG/RS: ``dp`` when EP is disabled, otherwise
        ``dp // ep`` after requiring a regular expert-DP layout.
        """
        kind = kind.upper()
        if kind in ("EXPERT_DP", "EDP"):
            return self._expert_dp_group_size()
        return {
            "TP": self.strategy.tp,
            "CP": self.strategy.cp,
            "EP": self.strategy.ep,
            "DP": self.strategy.dp,
            "PP": 2,  # P2P between adjacent stages
        }.get(kind, 1)

    def _expert_dp_group_size(self) -> int:
        ep = max(self.strategy.ep, 1)
        dp = max(self.strategy.dp, 1)
        if ep <= 1:
            return dp
        if dp < ep:
            raise ValueError(
                f"dp must be >= ep for expert-DP sharding (dp={dp}, ep={ep})"
            )
        if dp % ep != 0:
            raise ValueError(
                f"dp must be divisible by ep for expert-DP sharding "
                f"(dp={dp}, ep={ep})"
            )
        return dp // ep

    def group_instances(self, kind: str) -> list[list[int]]:
        """All non-degenerate rank instances for ``kind``.

        N-tier pricing uses the slowest instance, not just rank 0's
        canonical group, because some instances can cross a higher tier.
        """
        kind = kind.upper()
        if kind not in _VALID_KINDS:
            return []
        if kind in ("EXPERT_DP", "EDP"):
            self._expert_dp_group_size()
        attr = self.groups._kind_to_attr(kind)
        return [list(g) for g in getattr(self.groups, attr, []) if len(g) > 1]

    def ranks(self, kind: str) -> list[int]:
        """Explicit rank set for one instance of ``kind``.

        Returns ``[]`` when the strategy is degenerate (lazy fall-back
        to size-only path) or when the kind has size ≤ 1. Pass rank ``0``
        for the canonical instance — all instances of the same kind have
        the same shape by construction.
        """
        kind = kind.upper()
        if kind not in _VALID_KINDS:
            return []
        return self.groups.ranks_for(kind, 0)

    def tier(self, kind: str) -> int:
        """Tier index in ``system.interconnect.tiers`` for this group.

        Returns ``0`` (innermost) when the domain is in size-only mode
        (e.g. 2-tier system, or degenerate strategy). For the N-tier
        path the value is the smallest tier whose per-instance domain
        contains every group instance — the "tier-aligned primary"
        picked by :func:`zrt.training.topology._tier_for_groups`.
        """
        kind = kind.upper()
        if not self.system.interconnect.tiers:
            raise ValueError("interconnect has no tiers configured")
        if kind in self.groups.tier:
            return self.groups.tier[kind].primary_tier
        # Fall back to a 2-tier heuristic identical to the legacy
        # tier_for_group(): PP → outermost, else innermost-if-fits.
        if kind == "PP":
            return len(self.system.interconnect.tiers) - 1
        intra = self.system.interconnect.tiers[0].link
        intra_domain = intra.num_devices if intra.num_devices > 0 else (
            self.system.gpus_per_node
        )
        size = self.group_size(kind)
        return 0 if size <= intra_domain else len(self.system.interconnect.tiers) - 1

    def link(self, kind: str) -> LinkSpec:
        """:class:`LinkSpec` of the resolved tier for this group."""
        idx = self.tier(kind)
        return self.system.interconnect.tiers[idx].link

    def pp_p2p_link(self) -> LinkSpec:
        """Link adjacent PP stages use for activation P2P.

        For 3+ tier systems this consults ``ParallelGroups`` (the PP
        group's primary tier IS the inter-stage link). For 2-tier, the
        legacy heuristic applies: PP fits intra-node iff
        ``tp*cp*2 <= intra_domain``.
        """
        if self._use_multi_tier and "PP" in self.groups.tier:
            return self.system.interconnect.tiers[
                self.groups.tier["PP"].primary_tier
            ].link
        # Legacy 2-tier heuristic.
        intra = self.system.interconnect.tiers[0].link
        inter = self.system.interconnect.tiers[-1].link
        intra_domain = intra.num_devices if intra.num_devices > 0 else (
            self.system.gpus_per_node
        )
        cp = max(self.strategy.cp, 1)
        if self.strategy.tp * cp * 2 <= intra_domain:
            return intra
        return inter

    # ─ Pricing ──────────────────────────────────────────────────────

    def time(self, c: Collective) -> float:
        """Cost ``c`` using the appropriate tier-aware path.

        Dispatch:

        - 3+ tier systems with a non-degenerate strategy → cost is
          :func:`collective_time_multi_tier` over the group's explicit
          rank set (AG/RS decompose innermost → outermost across every
          tier, AR = RS + AG, P2P/A2A take the outermost spanned tier
          flat).
        - 2-tier systems (and graph-level collectives where TP/CP/EP
          historically used a single LINK) → AG/RS/AR runs through
          :func:`collective_time_hierarchical` (the legacy 2-stage
          intra+inter decomposition), other kinds run flat on the
          size-resolved link. Matches pre-refactor behavior bit-exact
          for every kind that previously went through the hierarchical
          path (DP grad-reduce, Muon AG/RS).
        - Degenerate group (size 1) → 0.
        """
        from zrt.training.models.comm import (
            collective_time,
            collective_time_hierarchical,
            collective_time_multi_tier,
        )
        if c.bytes_ <= 0:
            return 0.0

        if self._use_multi_tier:
            groups = self.group_instances(c.group)
            if groups:
                return max(
                    collective_time_multi_tier(c, ranks, self.system)
                    for ranks in groups
                )
            if not groups and self.group_size(c.group) <= 1:
                return 0.0
            # Otherwise (rare: degenerate ParallelGroups but non-trivial
            # strategy degree) fall through to legacy size-only.
            logger.debug(
                "CommDomain: multi-tier skipped for %s (degenerate groups), using legacy",
                c.group,
            )

        # Legacy 2-tier (or fallback) path.
        size = self.group_size(c.group)
        if size <= 1:
            return 0.0
        # AG/RS/AR previously used the 2-stage intra+inter hierarchical
        # decomposition for ALL multi-node groups. Preserve that path
        # exactly. Other kinds (P2P / A2A) historically ran flat on a
        # single link picked by tier_for_group — keep that behavior.
        if c.kind in ("AG", "RS", "AR"):
            return collective_time_hierarchical(c, size, self.system)
        return collective_time(c, size, self.link(c.group))

    def time_ar(self, c: Collective) -> float:
        """Convenience: AR cost regardless of ``c.kind`` (treat as AR).

        Some call sites pass an AR-volumed collective tagged as RS to
        signal ZeRO≥1. This explicit helper avoids hijacking the kind
        field for that purpose.
        """
        ar_c = Collective(name=c.name, kind="AR", group=c.group, bytes_=c.bytes_)
        return self.time(ar_c)

    # ─ Debug ────────────────────────────────────────────────────────

    def summary(self) -> dict[str, dict[str, object]]:
        """Human-readable per-kind tier picks. Used by tests and reports."""
        out: dict[str, dict[str, object]] = {}
        for kind in _VALID_KINDS:
            size = self.group_size(kind)
            if size <= 1:
                continue
            idx = self.tier(kind)
            tier_name = self.system.interconnect.tiers[idx].name
            out[kind] = {
                "size": size,
                "tier_idx": idx,
                "tier_name": tier_name,
                "rank_sample": self.ranks(kind),
            }
        return out


def build_comm_domain(system: SystemSpec, strategy: Strategy) -> CommDomain:
    """Convenience factory — equivalent to ``CommDomain(system, strategy)``.

    Provided for symmetry with :func:`build_process_groups` and to
    minimize churn at call sites.
    """
    return CommDomain(system=system, strategy=strategy)


def _format_rank_sample(ranks: list[int], limit: int = 8) -> str:
    if not ranks:
        return "[]"
    head = ", ".join(str(r) for r in ranks[:limit])
    if len(ranks) > limit:
        head += f", ... (+{len(ranks) - limit})"
    return f"[{head}]"


def comm_domain_report(
    system: SystemSpec,
    strategy: Strategy,
    kinds: tuple[str, ...] = _REPORT_KINDS,
) -> dict[str, dict[str, object]]:
    """Return display-ready communication-domain metadata by parallel axis."""
    domain = CommDomain(system=system, strategy=strategy)
    out: dict[str, dict[str, object]] = {}
    for kind in kinds:
        kind = kind.upper()
        size = domain.group_size(kind)
        tier_idx = domain.tier(kind)
        tier_name = system.interconnect.tiers[tier_idx].name
        ranks = domain.ranks(kind)
        out[kind] = {
            "group_size": size,
            "tier_idx": tier_idx,
            "tier_name": tier_name,
            "tier": f"{tier_idx}:{tier_name}",
            "rank_sample": _format_rank_sample(ranks),
        }
    return out


def format_comm_domain_entry(entry: dict[str, object]) -> str:
    """Compact single-cell representation for search result tables."""
    return (
        f"size={entry['group_size']}; "
        f"tier={entry['tier']}; "
        f"ranks={entry['rank_sample']}"
    )
