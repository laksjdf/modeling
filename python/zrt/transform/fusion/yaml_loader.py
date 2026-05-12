"""Backward-compat shim for ``python.zrt.transform.fusion.yaml_loader``.

The implementation moved to ``loading/`` (split between
``yaml_rule_loader.py`` and ``fusion_config.py``).  This module
re-exports the public surface so existing imports keep working.
"""
from __future__ import annotations

from .loading.fusion_config import (  # noqa: F401
    _FUSION_CONFIGS_DIR,
    _parse_fusion_config_dict,
    load_fusion_config_file,
    resolve_fusion_config,
)
from .loading.yaml_rule_loader import (  # noqa: F401
    _BUILTIN_YAML_DIR,
    _USER_YAML_DIRS,
    _expand_entry,
    _model_id_to_key,
    add_yaml_search_dir,
    load_model_yaml_rules,
    load_yaml_rules,
    load_yaml_rules_from_dir,
)

__all__ = [
    "add_yaml_search_dir",
    "load_fusion_config_file",
    "load_model_yaml_rules",
    "load_yaml_rules",
    "load_yaml_rules_from_dir",
    "resolve_fusion_config",
]
