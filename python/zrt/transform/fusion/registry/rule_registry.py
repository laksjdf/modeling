"""``RuleRegistry``: encapsulates registered fusion rules.

Module-level free functions ``register_rule`` / ``clear_rules`` /
``all_rules`` / ``lookup_rule`` / ``iter_active_rules`` are forwarders
to the process-wide ``default_registry()`` singleton (see
``__init__.py``).
"""
from __future__ import annotations

import logging
from typing import Optional

from python.zrt.transform.fusion.core.rule import ModuleFusionRule

logger = logging.getLogger(__name__)


def _index_keys(rule: ModuleFusionRule) -> list[str]:
    """Return the literal class-name keys this rule should be indexed under."""
    tc = rule.target_class
    if isinstance(tc, type):
        return [tc.__name__]
    if isinstance(tc, tuple):
        return [str(c) for c in tc]
    return [str(tc)]


class RuleRegistry:
    """Container for registered ``ModuleFusionRule`` objects.

    Replaces the module-global lists that used to live in
    ``registry.py``.  Each instance keeps:

    * ``_all_rules``  — registration-order list (matchers iterate this).
    * ``_by_name``    — name → rule (uniqueness validated on register).
    * ``_class_index``— class-name → list of rules (fast narrow-pool).
    """

    def __init__(self) -> None:
        self._all_rules: list[ModuleFusionRule] = []
        self._class_index: dict[str, list[ModuleFusionRule]] = {}
        self._by_name: dict[str, ModuleFusionRule] = {}

    # ── Mutation ────────────────────────────────────────────────────────

    def register(self, rule: ModuleFusionRule) -> None:
        """Add a rule to this registry.

        Rule ``name`` must be unique within the active registry; conflicting
        registrations raise ``ValueError`` so YAML/builtin authors notice
        collisions immediately.
        """
        name = rule.name or rule.op_type or ""
        if not name:
            raise ValueError(
                f"ModuleFusionRule for target_class={rule.target_class!r} "
                "has no name and no op_type — cannot register"
            )
        if name in self._by_name and self._by_name[name] is not rule:
            raise ValueError(
                f"Duplicate fusion rule name {name!r} (existing target_class="
                f"{self._by_name[name].target_class!r}, new target_class="
                f"{rule.target_class!r})"
            )
        self._all_rules.append(rule)
        self._by_name[name] = rule
        for key in _index_keys(rule):
            self._class_index.setdefault(key, []).append(rule)

    def clear(self) -> None:
        """Remove all registered rules (for test isolation)."""
        self._all_rules.clear()
        self._class_index.clear()
        self._by_name.clear()

    # ── Read access ─────────────────────────────────────────────────────

    @property
    def all_rules_list(self) -> list[ModuleFusionRule]:
        """Return every registered rule (live reference)."""
        return self._all_rules

    @property
    def name_index(self) -> dict[str, ModuleFusionRule]:
        return self._by_name

    @property
    def class_index(self) -> dict[str, list[ModuleFusionRule]]:
        return self._class_index

    def all_rules(self) -> list[ModuleFusionRule]:
        return list(self._all_rules)

    def by_name(self, name: str) -> Optional[ModuleFusionRule]:
        return self._by_name.get(name)

    def all_rule_names(self) -> list[str]:
        return [r.name for r in self._all_rules]

    # ── Lookup (delegates to matching.best_rule) ────────────────────────

    def lookup(
        self,
        operator_types: tuple[str, ...],
        *,
        module_class: str | None = None,
        active_rules: list[ModuleFusionRule] | None = None,
    ) -> Optional[ModuleFusionRule]:
        """Find the best-matching rule via the three-tier matcher.

        ``_class_index`` is no longer pre-narrowed against — that was
        an unsound perf optimization once ``ordered_regex`` /
        ``dag_signature`` rules stopped class-gating.  ``best_rule`` is
        called against the full active list; ``_class_index`` is kept
        for diagnostics (e.g. ``--list-fusion-rules``) but not consulted
        here.
        """
        from python.zrt.transform.fusion.matching.matcher import best_rule

        candidates = active_rules if active_rules is not None else self._all_rules
        return best_rule(operator_types, module_class or "", candidates)

    def iter_active(
        self,
        fusion_cfg,
        phase: str,
    ) -> list[ModuleFusionRule]:
        """Resolve ``fusion_cfg`` + ``phase`` to the active rule list.

        * ``enabled_rules is None`` → all rules whose ``default_phases``
          contains ``phase``, minus ``disabled_rules``.
        * ``enabled_rules`` non-empty → exactly those names, minus
          ``disabled_rules``.

        Names not present in the registry raise ``ValueError`` to surface
        YAML typos early.
        """
        enabled = fusion_cfg.enabled_rules
        disabled = set(fusion_cfg.disabled_rules)

        if enabled is None:
            candidates = [r for r in self._all_rules if phase in r.default_phases]
        else:
            unknown_enabled = [n for n in enabled if n not in self._by_name]
            if unknown_enabled:
                raise ValueError(
                    f"Unknown fusion rule names in enabled_rules: {unknown_enabled}. "
                    f"Registered rules: {sorted(self._by_name)}"
                )
            candidates = [self._by_name[n] for n in enabled]

        # ``disabled_rules`` is a no-op for names the registry doesn't know
        # about — a global fallback YAML may legitimately list model-specific
        # rule names that aren't loaded for the current model.  Log it once
        # so genuine typos still surface, but don't fail the run.
        unknown_disabled = [n for n in disabled if n not in self._by_name]
        if unknown_disabled:
            logger.debug(
                "Fusion config disabled_rules references unregistered rule "
                "names (no-op for this run): %s", sorted(unknown_disabled),
            )
        return [r for r in candidates if r.name not in disabled]
