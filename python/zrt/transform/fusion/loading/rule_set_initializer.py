"""Two-stage YAML rule registration for fusion.

Step-4: replaces the legacy ``registry/platforms/__init__.py`` if/elif
dispatcher and the Python-registered ``registry/builtins.py``.  The new
flow is:

1. ``clear_rules()`` — wipe the registry.
2. Always load ``rules/_common.yaml`` (the migrated builtins).
3. If ``model_id`` resolves to a known model slug, load
   ``rules/<slug>.yaml``.

No platform-specific Python rules are loaded any more — every rule lives
in YAML.
"""
from __future__ import annotations

import logging
from pathlib import Path

from python.zrt.transform.fusion.registry import clear_rules, register_rule

from .yaml_rule_loader import _model_id_to_key, load_yaml_rules

logger = logging.getLogger(__name__)


# Default directory for built-in YAML rule files.
_RULES_DIR = Path(__file__).resolve().parent.parent / "rules"
_COMMON_FILENAME = "_common.yaml"


def initialize_rules(model_id: str = "") -> None:
    """Load all fusion rules for ``model_id``.

    Always loads ``_common.yaml`` first, then ``<model_slug>.yaml`` if a
    matching file exists in :data:`_RULES_DIR`.

    The registry is cleared before loading so the result is deterministic
    regardless of prior state.
    """
    clear_rules()

    common_path = _RULES_DIR / _COMMON_FILENAME
    if common_path.exists():
        for rule in load_yaml_rules(common_path):
            try:
                register_rule(rule)
            except ValueError as exc:
                logger.warning("Skipping common rule: %s", exc)

    if not model_id:
        return

    slug = _model_id_to_key(model_id)
    if not slug:
        return

    model_path = _RULES_DIR / f"{slug}.yaml"
    if model_path.exists():
        for rule in load_yaml_rules(model_path):
            try:
                register_rule(rule)
            except ValueError as exc:
                logger.warning("Skipping model rule (%s): %s", slug, exc)
        logger.info("Loaded model-specific rules from %s", model_path)
        return

    # Prefix-match fallback (parity with the old ``load_model_yaml_rules``):
    # e.g. model_id="deepseek_v4_lite" still loads deepseek_v4.yaml.
    prefix = slug.split("_")[0].replace("-", "")
    for path in sorted(_RULES_DIR.glob("*.yaml")):
        if path.name == _COMMON_FILENAME:
            continue
        stem = path.stem.replace("_", "").replace("-", "")
        if prefix and stem.startswith(prefix):
            for rule in load_yaml_rules(path):
                try:
                    register_rule(rule)
                except ValueError as exc:
                    logger.warning("Skipping model rule (%s): %s", path.stem, exc)
            logger.info("Loaded model-specific rules from %s", path)
            return


__all__ = ["initialize_rules"]
