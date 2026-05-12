"""Greedy left-to-right partial-match scanner for ordered_regex rules.

Finds contiguous subsequences within a bucket that match an
``ordered_regex`` pattern.  Used by ``MultiPassFuser`` as a fallback
after whole-bucket matching misses.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

from python.zrt.transform.fusion.core.pattern import MatchPattern
from python.zrt.transform.fusion.core.rule import ModuleFusionRule


def try_match_at(
    operator_types: Sequence[str],
    start: int,
    pattern: MatchPattern,
) -> Optional[int]:
    """Anchored ordered_regex match.

    Returns end-exclusive index if op_types[start:end] matches pattern,
    else None.  Trailing skip-ops do NOT extend the match (they remain
    available for the next iteration of the scanner).
    Only ``ordered_regex`` patterns are supported.
    """
    if pattern.kind != "ordered_regex":
        return None
    regexes = pattern.op_regexes
    skip = pattern.skip_ops
    i, j = 0, start
    n_re, n_op = len(regexes), len(operator_types)
    while i < n_re and j < n_op:
        op = operator_types[j]
        try:
            hit = re.fullmatch(regexes[i], op) is not None
        except re.error:
            hit = False
        if hit:
            i += 1
            j += 1
        elif op in skip:
            j += 1
        else:
            return None
    if i != n_re:
        return None
    matched_len = j - start
    if matched_len < pattern.min_ops or matched_len > pattern.max_ops:
        return None
    return j


class SlidingWindowScanner:
    """Greedy left-to-right partial-match scanner.

    Walks operator_types from left to right.  At each position, tries
    every ordered_regex rule in priority order; picks the highest-priority
    match (ties broken by longest match, then registration order).
    Advances past the matched span.  When no rule matches, advances by 1.
    """

    def __init__(self, rules: Sequence[ModuleFusionRule]):
        # Only ordered_regex rules are scannable.
        self._rules = [
            r for r in rules
            if r.pattern is not None and r.pattern.kind == "ordered_regex"
        ]
        # Stable sort by priority desc — registration-order ties preserved.
        self._rules.sort(key=lambda r: -r.priority)

    def scan(
        self,
        operator_types: Sequence[str],
    ) -> list[tuple[ModuleFusionRule, int, int]]:
        """Return list of (rule, start_idx, end_idx_exclusive) for each match."""
        matches: list[tuple[ModuleFusionRule, int, int]] = []
        position = 0
        n = len(operator_types)
        while position < n:
            best: Optional[tuple[ModuleFusionRule, int]] = None  # (rule, end)
            for rule in self._rules:
                end = try_match_at(operator_types, position, rule.pattern)
                if end is None:
                    continue
                if best is None:
                    best = (rule, end)
                elif rule.priority > best[0].priority:
                    best = (rule, end)
                elif rule.priority == best[0].priority and end > best[1]:
                    best = (rule, end)
            if best is None:
                position += 1
                continue
            matches.append((best[0], position, best[1]))
            position = best[1]
        return matches
