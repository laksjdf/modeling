"""FusionConfig YAML resolution — picks YAML by model + phase, parses + validates."""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

from .yaml_rule_loader import _model_id_to_key

logger = logging.getLogger(__name__)


_FUSION_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _parse_fusion_config_dict(d: dict, source: Path):
    """Build a FusionConfig from a parsed YAML dict."""
    from python.zrt.transform.context import FusionConfig

    enabled = d.get("enabled_rules")
    if enabled is None:
        enabled_set = None
    elif isinstance(enabled, list):
        enabled_set = set(str(x) for x in enabled)
    else:
        raise ValueError(
            f"{source}: enabled_rules must be null or a list, got {type(enabled).__name__}"
        )

    disabled = d.get("disabled_rules") or []
    if not isinstance(disabled, list):
        raise ValueError(
            f"{source}: disabled_rules must be a list, got {type(disabled).__name__}"
        )

    siblings = d.get("merge_sibling_classes") or []
    if not isinstance(siblings, list):
        raise ValueError(
            f"{source}: merge_sibling_classes must be a list"
        )

    return FusionConfig(
        enabled_rules=enabled_set,
        disabled_rules=set(str(x) for x in disabled),
        allow_structural_collapse=bool(d.get("allow_structural_collapse", False)),
        merge_sibling_classes=set(str(x) for x in siblings),
    )


def load_fusion_config_file(path: Path, phase: str):
    """Load one fusion-config YAML file.

    Honours an optional ``training:`` / ``inference:`` top-level split:
    when present, the section matching ``phase`` is used.  Otherwise the
    whole document is treated as a single config.

    Returns ``None`` when the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return None

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: fusion config must be a YAML mapping at the top level"
        )

    if phase in raw and isinstance(raw[phase], dict):
        return _parse_fusion_config_dict(raw[phase], path)
    if "training" in raw or "inference" in raw:
        # Phase-aware doc but our phase has no section → empty config.
        return _parse_fusion_config_dict({}, path)
    return _parse_fusion_config_dict(raw, path)


def resolve_fusion_config(
    model_id: str,
    phase: str,
    explicit_path: str | Path | None = None,
):
    """Resolve the FusionConfig to use for one ``FusionPass`` run.

    Search order (first hit wins, except `explicit_path` which always wins):

      1. ``explicit_path`` — value of ``--fusion-config`` if given.
      2. ``configs/<model_slug>_<phase>.yaml``
      3. ``configs/<model_slug>.yaml``
      4. ``configs/<phase>_default.yaml``

    Where ``<model_slug>`` is :func:`_model_id_to_key`.  Anything missing
    is silently skipped.  When all four miss, returns the dataclass
    defaults (``enabled_rules=None``, no disables).
    """
    from python.zrt.transform.context import FusionConfig

    if explicit_path is not None:
        cfg = load_fusion_config_file(Path(explicit_path), phase)
        if cfg is None:
            raise FileNotFoundError(
                f"--fusion-config path does not exist: {explicit_path}"
            )
        logger.info("Loaded fusion config from %s [phase=%s]", explicit_path, phase)
        return cfg

    slug = _model_id_to_key(model_id) if model_id else ""
    candidates: list[Path] = []
    if slug:
        candidates.append(_FUSION_CONFIGS_DIR / f"{slug}_{phase}.yaml")
        candidates.append(_FUSION_CONFIGS_DIR / f"{slug}.yaml")
    candidates.append(_FUSION_CONFIGS_DIR / f"{phase}_default.yaml")

    for path in candidates:
        cfg = load_fusion_config_file(path, phase)
        if cfg is not None:
            logger.info("Loaded fusion config from %s [phase=%s]", path, phase)
            return cfg

    return FusionConfig()
