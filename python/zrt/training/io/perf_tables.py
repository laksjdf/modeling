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

    Calibrated against:
      - NVIDIA MLPerf Training v4.0: 904 TFLOPs/s/GPU FP8 at 512 H100 GPUs
        → BF16 ~452 TFLOPs/s/GPU → eff ≈ 0.46 (source: NVIDIA blog, June 2024)
      - NVIDIA DGXC LLaMA 3.1 70B: ~461 TFLOPs/s/GPU BF16 on 1024 H100
        → eff ≈ 0.47 (source: NGC DGXC Benchmarking)

    Realistic large-GEMM BF16 efficiency on H100 at training scale: ~0.50,
    not 0.85.  The 0.85 figure reflects single-kernel microbenchmarks;
    sustained training throughput includes optimizer overheads, gradient
    accumulation, pipeline bubbles, and scheduling gaps that reduce
    effective utilization by ~40%.
    """
    if flops <= 0:
        return 0.0

    # Use total FLOPs as proxy for matmul size
    # Small: < 1e9, medium: 1e9-1e11, large: > 1e11
    if flops < 1e9:
        return 0.50
    elif flops < 1e10:
        return 0.55
    elif flops < 1e11:
        return 0.60
    else:
        return 0.50


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
