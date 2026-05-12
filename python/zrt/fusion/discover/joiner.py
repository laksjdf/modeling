"""Join AST scan + runtime trace + templates into draft YAML rules.

The joiner is the only place that has to be aware of all three input
streams.  It produces:

  * a list of dicts ready for ``yaml.safe_dump`` (consumable by
    ``yaml_loader.load_yaml_rules``),
  * a list of human-readable review notes pointing out classes that
    fell back to the default template, missed runtime data, or have
    placeholder formulas.

Strategy:

  1. Iterate every ``nn.Module`` subclass plus every top-level helper
     in ``forward_calls``.  This keeps non-class fusion candidates
     (``sparse_attn``, ``apply_rotary_emb``) in scope.
  2. Pick a template — exact match, alias, or ``_default``.
  3. If runtime gave us an ordered op sequence and the template did
     **not** specify a ``class_only`` / ``dag_signature`` match, attach
     the longest observed sequence as ``op_regexes``.
  4. Always sort output by ``priority`` desc, then class name.
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Dict, List, Tuple

from .ast_scanner import AstClassInfo, AstScanResult
from .templates import TEMPLATES, get_template


_TOP_LEVEL_HELPERS = {"sparse_attn", "apply_rotary_emb"}


def join_rules(
    ast: AstScanResult,
    runtime: Dict[str, List[Tuple[str, ...]]],
) -> Tuple[List[dict], List[str]]:
    """Merge AST + runtime + templates into draft YAML rules.

    Returns ``(rules, review_notes)``.
    """
    rules: List[dict] = []
    notes: List[str] = []

    # Set of every class / helper that appears in any forward body
    # (used to decide whether a top-level function is referenced and
    # therefore worth a rule).
    referenced_helpers: set[str] = set()
    for cls in ast.classes:
        for c in cls.forward_calls:
            referenced_helpers.add(c.split(".")[-1])

    # ── nn.Module subclasses ────────────────────────────────────────────────
    for cls in ast.classes:
        if not cls.is_nn_module:
            continue
        rule = _build_rule_from_template(cls.name, runtime, notes)
        rules.append(rule)

    # ── Top-level helpers (functions only) ──────────────────────────────────
    seen_class_names = {c.name for c in ast.classes}
    for fn in ast.top_level_funcs:
        if fn.name in seen_class_names:
            continue
        # Only emit a rule if the helper is either an explicit known
        # template OR is referenced from at least one module.forward.
        if fn.name not in TEMPLATES and fn.name not in referenced_helpers:
            continue
        if fn.name not in TEMPLATES and fn.name not in _TOP_LEVEL_HELPERS:
            # Generic top-level helper — skip silently.  We'd otherwise
            # flood the output with utility-function rules.
            continue
        rule = _build_rule_from_template(fn.name, runtime, notes)
        rules.append(rule)

    # ── Stable sort: priority desc, then name ───────────────────────────────
    rules.sort(key=lambda r: (-int(r.get("priority", 10)),
                              str(r.get("target_class", ""))))
    return rules, notes


# ─── Per-class rule construction ──────────────────────────────────────────────

def _build_rule_from_template(
    name: str,
    runtime: Dict[str, List[Tuple[str, ...]]],
    notes: List[str],
) -> dict:
    tpl = get_template(name)
    if tpl is None:
        tpl = deepcopy(TEMPLATES["_default"])
        notes.append(
            f"{name}: fell back to _default template — please review "
            f"flops_formula / shape_derivation."
        )

    rule = dict(tpl)
    rule["target_class"] = name
    if rule.get("op_type") is None:
        rule["op_type"] = name

    # Attach runtime-observed op sequence when ordered_regex is needed.
    seqs = runtime.get(name, [])
    match = rule.get("match") or {"kind": "ordered_regex"}
    kind = match.get("kind", "ordered_regex")

    if kind == "ordered_regex":
        if seqs:
            chosen = seqs[0]  # already sorted longest-first
            match = dict(match)
            match["op_regexes"] = [re.escape(op) for op in chosen]
            match.setdefault("min_ops", len(chosen))
            rule["match"] = match
        else:
            # No runtime data — degrade to class_only so the loader
            # still accepts the rule (ordered_regex requires non-empty
            # op_regexes).
            rule["match"] = {"kind": "class_only"}
            notes.append(
                f"{name}: ordered_regex requested but no runtime sequence "
                f"observed — degraded to class_only."
            )
    # class_only / dag_signature: nothing to attach — template wins.

    if rule.get("annotations", {}).get("discover_status") == "fallback":
        notes.append(
            f"{name}: no template — please supply shape_derivation and "
            f"flops_formula manually."
        )

    return rule
