"""Performance efficiency lookup tables.

Phase 1: simple analytical heuristics.
Phase 4: empirical CSV lookup curves.
"""

from __future__ import annotations

from zrt.training.spec.dtype import Dtype


def achieved_flops_efficiency(
    gpu_name: str, dtype: Dtype, flops: float,
) -> float:
    """Achieved FLOP/s fraction of peak for a given matmul size.

    Phase 1 heuristic: larger matmuls achieve higher efficiency.
    """
    if flops <= 0:
        return 0.0

    # Use total FLOPs as proxy for matmul size
    # Small: < 1e9, medium: 1e9-1e11, large: > 1e11
    if flops < 1e9:
        return 0.50
    elif flops < 1e10:
        return 0.65
    elif flops < 1e11:
        return 0.75
    else:
        return 0.85


def achieved_bandwidth_efficiency(gpu_name: str, bytes_: float) -> float:
    """Achieved bandwidth fraction of peak for a given transfer size."""
    if bytes_ <= 0:
        return 0.0
    # Larger transfers approach peak BW
    if bytes_ < 1e6:      # < 1 MB
        return 0.40
    elif bytes_ < 1e8:    # < 100 MB
        return 0.70
    else:
        return 0.85
