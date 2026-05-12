"""Child→parent annotation propagation for fused nodes."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.transform.fusion.bucketing.call_id_bucketer import FusionGroup


# Annotations that propagate child→parent when all children agree.
_SCALAR_PROPAGATE_KEYS = (
    "stage_id", "phase", "ep_experts_local", "ep_a2a_inserted", "recompute",
)
_DICT_PROPAGATE_KEYS = ("tp_split", "ep_needs_a2a", "cp_split")


def _propagated_annotations(group: "FusionGroup") -> dict:
    """Return annotations that all children of *group* agree on.

    Scalar (hashable) keys use set-based dedup; dict-valued keys
    compare element-wise (since dicts are unhashable).
    """
    propagated: dict = {}
    for key in _SCALAR_PROPAGATE_KEYS:
        vals = {op.annotations.get(key) for op in group.child_ops
                if key in op.annotations}
        if len(vals) == 1:
            propagated[key] = vals.pop()
    for key in _DICT_PROPAGATE_KEYS:
        seen = [op.annotations[key] for op in group.child_ops
                if key in op.annotations]
        if seen and all(d == seen[0] for d in seen):
            propagated[key] = seen[0]
    return propagated
