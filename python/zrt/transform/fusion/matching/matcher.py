"""Three-tier fusion-rule matcher.

Contract: see ``docs/fusion_v2_rich_rules_zh.md`` §2.1.
"""
from __future__ import annotations

import re
from typing import Optional

from python.zrt.transform.fusion.core.pattern import MatchPattern
from python.zrt.transform.fusion.core.rule import ModuleFusionRule


# ─────────────────────────────────────────────────────────────────────────────
# target_class ↔ module_class
# ─────────────────────────────────────────────────────────────────────────────

def _target_candidates(target_class: type | str | tuple[str, ...]) -> tuple[str, ...]:
    """Normalize ``target_class`` to a tuple of candidate strings/regexes."""
    if isinstance(target_class, type):
        return (target_class.__name__,)
    if isinstance(target_class, tuple):
        return tuple(str(c) for c in target_class)
    return (str(target_class),)


def _class_matches(module_class: str, target_class: type | str | tuple[str, ...]) -> bool:
    """Return True iff ``module_class`` is matched by ``target_class``.

    Strategy per candidate:
        - literal ``==`` first (fast path for plain class names)
        - then ``re.fullmatch`` (regex fallback)
    """
    candidates = _target_candidates(target_class)
    for cand in candidates:
        if module_class == cand:
            return True
        try:
            if re.fullmatch(cand, module_class) is not None:
                return True
        except re.error:
            # Malformed regex → treat as literal-only; already failed above.
            continue
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Per-tier matchers
# ─────────────────────────────────────────────────────────────────────────────

def _check_size(operator_types: tuple[str, ...], pattern: MatchPattern) -> bool:
    n = len(operator_types)
    return pattern.min_ops <= n <= pattern.max_ops


def _match_class_only(operator_types: tuple[str, ...], pattern: MatchPattern) -> bool:
    return _check_size(operator_types, pattern)


def _match_ordered_regex(operator_types: tuple[str, ...], pattern: MatchPattern) -> bool:
    if not _check_size(operator_types, pattern):
        return False
    regexes = pattern.op_regexes
    skip = pattern.skip_ops
    i = 0  # index into regexes
    j = 0  # index into operator_types
    n_re = len(regexes)
    n_op = len(operator_types)
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
            return False
    if i != n_re:
        return False
    # Trailing ops past last regex must all be skip-ops.
    while j < n_op:
        if operator_types[j] not in skip:
            return False
        j += 1
    return True


def _match_dag_signature(operator_types: tuple[str, ...], pattern: MatchPattern) -> bool:
    if not _check_size(operator_types, pattern):
        return False
    for regex, min_count in pattern.op_multiset:
        try:
            count = sum(1 for op in operator_types if re.fullmatch(regex, op) is not None)
        except re.error:
            count = 0
        if count < min_count:
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def match_group(
    operator_types: tuple[str, ...],
    module_class: str,
    *,
    pattern: MatchPattern,
    target_class: type | str | tuple[str, ...],
) -> bool:
    """Return True iff the group matches ``pattern`` under ``target_class``.

    ``target_class`` only gates ``class_only`` rules (which have no
    other signal).  ``ordered_regex`` and ``dag_signature`` rules match
    on operator-sequence shape alone — the bucket's ``module_class`` is
    not required to match the rule's ``target_class``.  The
    ``target_class`` parameter is preserved in the signature for API
    stability (``best_rule`` still passes it).
    """
    kind = pattern.kind
    if kind == "class_only":
        if not _class_matches(module_class, target_class):
            return False
        return _match_class_only(operator_types, pattern)
    if kind == "ordered_regex":
        return _match_ordered_regex(operator_types, pattern)
    if kind == "dag_signature":
        return _match_dag_signature(operator_types, pattern)
    raise ValueError(f"unknown MatchPattern.kind: {kind!r}")


def best_rule(
    operator_types: tuple[str, ...],
    module_class: str,
    rules: list[ModuleFusionRule],
) -> Optional[ModuleFusionRule]:
    """Pick the highest-priority matching rule; ties broken by list order."""
    best: Optional[ModuleFusionRule] = None
    best_idx: int = -1
    for idx, rule in enumerate(rules):
        pattern = rule.pattern
        if pattern is None:
            continue
        if not match_group(
            operator_types,
            module_class,
            pattern=pattern,
            target_class=rule.target_class,
        ):
            continue
        if best is None or rule.priority > best.priority:
            best = rule
            best_idx = idx
        # ties: keep earlier (lower idx) — already satisfied because we
        # only replace on strictly greater priority.
        _ = best_idx
    return best
