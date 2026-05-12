"""Three-tier fusion-rule matcher (whole-bucket + sliding-window helpers)."""
from __future__ import annotations

from .matcher import (
    _class_matches,
    _match_class_only,
    _match_dag_signature,
    _match_ordered_regex,
    best_rule,
    match_group,
)
from .sliding_window import SlidingWindowScanner, try_match_at

__all__ = [
    "SlidingWindowScanner",
    "_class_matches",
    "_match_class_only",
    "_match_dag_signature",
    "_match_ordered_regex",
    "best_rule",
    "match_group",
    "try_match_at",
]
