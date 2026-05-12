"""Load fusion rules from YAML files.

The classmethod ``ModuleFusionRule.from_yaml_dict`` delegates to
``rule_from_yaml_dict`` here so the dataclass module stays slim.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from python.zrt.transform.fusion.core.io_role import IORole, ShapeDerivation
from python.zrt.transform.fusion.core.pattern import (
    DEFAULT_SKIP_OPS,
    MatchPattern,
)

from .op_name_resolver import resolve_short_name, resolve_short_names

logger = logging.getLogger(__name__)


# Default directory for built-in YAML rule files.
_BUILTIN_YAML_DIR = Path(__file__).resolve().parent.parent / "rules"
_USER_YAML_DIRS: list[Path] = []


def add_yaml_search_dir(path: Path | str) -> None:
    """Add a directory to the YAML rule search path."""
    _USER_YAML_DIRS.append(Path(path))


def load_yaml_rules(path: Path):
    """Load rules from a single YAML file.

    Returns an empty list when the file does not exist or is empty.
    Malformed entries are skipped with a warning.
    """
    from python.zrt.transform.fusion.core.rule import ModuleFusionRule  # noqa: F401

    path = Path(path)
    if not path.exists():
        logger.warning("YAML rules file not found: %s", path)
        return []

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return []

    if not isinstance(data, list):
        logger.error("YAML rules file must contain a list at top level: %s", path)
        return []

    rules: list = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            logger.warning("Skipping non-dict entry %d in %s", i, path)
            continue
        if not entry.get("enabled", True):
            logger.info(
                "Skipping disabled rule: %s",
                entry.get("target_class", f"entry {i}"),
            )
            continue
        try:
            rules.extend(_expand_entry(entry))
        except Exception as e:
            logger.warning(
                "Failed to parse rule entry %d (%s) in %s: %s",
                i, entry.get("target_class", "?"), path, e,
            )

    return rules


def _expand_entry(entry: dict):
    """Expand one YAML entry into one or more rules.

    Multi-sequence ``observed_op_seqs`` is split into N rules.  Otherwise
    one rule is produced.
    """
    from python.zrt.transform.fusion.core.rule import ModuleFusionRule

    raw_seqs = entry.get("observed_op_seqs") or []
    has_match = entry.get("match") is not None

    # New schema or single-sequence legacy → just one rule.
    if has_match or len(raw_seqs) <= 1:
        return [ModuleFusionRule.from_yaml_dict(entry)]

    # Multi-sequence legacy → one rule per sequence.
    rules: list = []
    for seq in raw_seqs:
        sub: dict[str, Any] = dict(entry)
        sub["observed_op_seqs"] = [seq]
        rules.append(ModuleFusionRule.from_yaml_dict(sub))
    return rules


def load_yaml_rules_from_dir(directory: Path):
    """Load all .yaml/.yml files from a directory."""
    rules: list = []
    for path in sorted(directory.glob("*.yaml")):
        rules.extend(load_yaml_rules(path))
    for path in sorted(directory.glob("*.yml")):
        rules.extend(load_yaml_rules(path))
    return rules


def load_model_yaml_rules(model_id: str) -> None:
    """Load model-specific YAML rules and register them.

    Searches ``rules/`` and any user-registered directories for files
    matching the model_id (exact stem match first, then prefix match).
    """
    from python.zrt.transform.fusion.registry import register_rule

    model_key = _model_id_to_key(model_id)
    search_dirs = [_BUILTIN_YAML_DIR] + _USER_YAML_DIRS

    seen_files: set[Path] = set()

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Exact match: e.g. deepseek_v4.yaml
        exact = search_dir / f"{model_key}.yaml"
        if exact.exists() and exact not in seen_files:
            seen_files.add(exact)
            for rule in load_yaml_rules(exact):
                register_rule(rule)
            logger.info("Loaded model-specific rules from %s", exact)
            continue

        # Prefix match: e.g. deepseek_v4.yaml when model_key is "deepseek_v4_lite"
        for path in sorted(search_dir.glob("*.yaml")):
            if path in seen_files:
                continue
            stem = path.stem.replace("_", "").replace("-", "")
            prefix = model_key.split("_")[0].replace("-", "")
            if prefix and stem.startswith(prefix):
                seen_files.add(path)
                for rule in load_yaml_rules(path):
                    register_rule(rule)
                logger.info("Loaded model-specific rules from %s", path)


def _model_id_to_key(model_id: str) -> str:
    """Convert ``"deepseek-ai/DeepSeek-V4"`` → ``"deepseek_v4"``."""
    mid = model_id.lower()
    if "/" in mid:
        mid = mid.rsplit("/", 1)[-1]
    key = mid.replace("-", "_")
    noise = {"ai", "hf", "hub", "models", "instruct", "chat", "base"}
    parts = [p for p in key.split("_") if p and p not in noise]
    return "_".join(parts) if parts else key


# ─────────────────────────────────────────────────────────────────────────────
# YAML helpers used by ``ModuleFusionRule.from_yaml_dict`` (moved from rule.py)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_match_dict(d: dict) -> MatchPattern:
    """Build a MatchPattern from the YAML ``match:`` block."""
    kind = d.get("kind", "ordered_regex")

    # ordered_regex
    op_regexes: tuple[str, ...] = ()
    if d.get("op_regexes"):
        op_regexes = tuple(d["op_regexes"])

    # dag_signature
    op_multiset: tuple[tuple[str, int], ...] = ()
    raw_ms = d.get("op_multiset")
    if raw_ms:
        if isinstance(raw_ms, dict):
            op_multiset = tuple((k, int(v)) for k, v in raw_ms.items())
        else:
            op_multiset = tuple((entry[0], int(entry[1])) for entry in raw_ms)

    # skip_ops_extra: short or full names; merged with DEFAULT_SKIP_OPS
    skip = set(DEFAULT_SKIP_OPS)
    for raw in (d.get("skip_ops_extra") or []):
        try:
            skip.add(resolve_short_name(raw))
        except KeyError:
            skip.add(raw)
    # skip_ops_only: replace, not merge
    if d.get("skip_ops_only"):
        skip = set()
        for raw in d["skip_ops_only"]:
            try:
                skip.add(resolve_short_name(raw))
            except KeyError:
                skip.add(raw)

    return MatchPattern(
        kind=kind,
        op_regexes=op_regexes,
        op_multiset=op_multiset,
        skip_ops=frozenset(skip),
        min_ops=int(d.get("min_ops", 1)),
        max_ops=int(d.get("max_ops", 1024)),
    )


def _io_role_from_dict(d: dict) -> IORole:
    return IORole(
        role=d["role"],
        source_kind=d.get("source_kind", "input"),
        source_op_index=int(d.get("source_op_index", -1)),
        source_arg_index=int(d.get("source_arg_index", -1)),
        shape_role=d.get("shape_role"),
        expected_dtype=d.get("expected_dtype"),
    )


def _parse_shape_derivation(d: dict) -> ShapeDerivation:
    standard = ("batch_size", "seq_len", "hidden_in", "hidden_out",
                "num_heads", "head_dim", "intermediate_dim",
                "vocab_size", "num_experts", "topk")
    kwargs: dict[str, Any] = {k: d[k] for k in standard if k in d}
    extra_raw = d.get("extra") or []
    extra: tuple[tuple[str, str], ...] = tuple(
        (entry[0], entry[1]) if isinstance(entry, (list, tuple))
        else (entry["name"], entry["expr"])
        for entry in extra_raw
    )
    if extra:
        kwargs["extra"] = extra
    return ShapeDerivation(**kwargs)


def rule_from_yaml_dict(rule_cls, d: dict):
    """Construct a ``ModuleFusionRule`` from a parsed YAML dict.

    ``rule_cls`` is the dataclass to instantiate (passed in to avoid a
    circular import — the rule module delegates here from its
    classmethod).
    """
    # ── Pattern (new schema) ──
    pattern = None
    match_d = d.get("match")
    if match_d is not None:
        pattern = _parse_match_dict(match_d)

    # ── Legacy op_sequences ──
    raw_seqs = d.get("observed_op_seqs", []) or []
    resolved_seqs: tuple[tuple[str, ...], ...] = tuple(
        resolve_short_names(seq) for seq in raw_seqs
    )

    # ── IO roles ──
    roles_list: list[IORole] = []
    for entry in (d.get("io_roles") or []):
        roles_list.append(_io_role_from_dict(entry))
    for entry in (d.get("inputs") or []):
        entry = {**entry, "source_kind": entry.get("source_kind", "input")}
        roles_list.append(_io_role_from_dict(entry))
    for entry in (d.get("outputs") or []):
        entry = {**entry, "source_kind": entry.get("source_kind", "output")}
        roles_list.append(_io_role_from_dict(entry))

    # ── Shape derivation ──
    sd = None
    sd_d = d.get("shape_derivation")
    if sd_d:
        sd = _parse_shape_derivation(sd_d)

    # target_class — list (regex alternatives) → tuple of str
    tc_raw = d["target_class"]
    if isinstance(tc_raw, list):
        target_class: type | str | tuple[str, ...] = tuple(tc_raw)
    else:
        target_class = tc_raw

    # Phase metadata — accept list / tuple / single string / null.
    raw_phases = d.get("default_phases")
    if raw_phases is None:
        default_phases: tuple[str, ...] = ("inference", "training")
    elif isinstance(raw_phases, str):
        default_phases = (raw_phases,)
    else:
        default_phases = tuple(str(p) for p in raw_phases)

    return rule_cls(
        target_class=target_class,
        op_type=d.get("op_type"),
        pattern=pattern,
        op_sequences=resolved_seqs,
        io_roles=tuple(roles_list),
        shape_derivation=sd,
        flops_formula=d.get("flops_formula"),
        memory_formula=d.get("memory_formula"),
        flops_kind=d.get("flops_kind", "sum_children"),
        annotations=d.get("annotations", {}) or {},
        priority=d.get("priority", 10),
        name=d.get("name", "") or "",
        description=d.get("description", "") or "",
        default_phases=default_phases,
    )
