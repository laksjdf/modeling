"""Public registry API: process-wide singleton + module-level forwarders.

Tests and CLI call ``register_rule(...)`` / ``clear_rules()`` /
``all_rules()`` / ``lookup_rule(...)`` / ``iter_active_rules(...)``;
those forwarders delegate to the singleton ``default_registry()``.
"""
from __future__ import annotations

from typing import Optional

from python.zrt.transform.fusion.core.rule import ModuleFusionRule

from .rule_registry import RuleRegistry, _index_keys

# Process-wide singleton.
_DEFAULT_REGISTRY = RuleRegistry()


def default_registry() -> RuleRegistry:
    """Return the process-wide ``RuleRegistry`` singleton."""
    return _DEFAULT_REGISTRY


# ── Backward-compatible module-level globals (live references) ──────────────
# Tests don't read these directly, but they're kept as aliases so any
# legacy code that did ``registry._ALL_RULES`` continues to see the same
# list.  They alias the singleton's internal storage.
_ALL_RULES = _DEFAULT_REGISTRY._all_rules        # noqa: SLF001
_CLASS_INDEX = _DEFAULT_REGISTRY._class_index    # noqa: SLF001
_NAME_INDEX = _DEFAULT_REGISTRY._by_name         # noqa: SLF001


# ── Forwarder functions (preserve every existing public API) ────────────────

def register_rule(rule: ModuleFusionRule) -> None:
    """Add a rule to the default registry."""
    _DEFAULT_REGISTRY.register(rule)


def clear_rules() -> None:
    """Remove all registered rules from the default registry."""
    _DEFAULT_REGISTRY.clear()


def all_rules() -> list[ModuleFusionRule]:
    """Return every registered rule (flat list, registration order)."""
    return _DEFAULT_REGISTRY.all_rules()


def rule_by_name(name: str) -> Optional[ModuleFusionRule]:
    """Return the registered rule with ``name``, or ``None``."""
    return _DEFAULT_REGISTRY.by_name(name)


def all_rule_names() -> list[str]:
    """Return all registered rule names in registration order."""
    return _DEFAULT_REGISTRY.all_rule_names()


def lookup_rule(
    operator_types: tuple[str, ...],
    *,
    module_class: str | None = None,
    active_rules: list[ModuleFusionRule] | None = None,
) -> Optional[ModuleFusionRule]:
    """Find the best-matching rule via the three-tier matcher.

    Forwarder → ``default_registry().lookup(...)``.
    """
    return _DEFAULT_REGISTRY.lookup(
        operator_types,
        module_class=module_class,
        active_rules=active_rules,
    )


# Legacy alias — old call sites pass ``op_types`` of full aten names.
lookup_rule_by_op_sequence = lookup_rule


def iter_active_rules(
    fusion_cfg,                # FusionConfig (typed loosely to avoid import cycle)
    phase: str,
) -> list[ModuleFusionRule]:
    """Resolve the FusionConfig + phase to the active rule list.

    Forwarder → ``default_registry().iter_active(...)``.
    """
    return _DEFAULT_REGISTRY.iter_active(fusion_cfg, phase)


__all__ = [
    "RuleRegistry",
    "default_registry",
    "register_rule",
    "clear_rules",
    "all_rules",
    "rule_by_name",
    "all_rule_names",
    "lookup_rule",
    "lookup_rule_by_op_sequence",
    "iter_active_rules",
]
