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


# ─────────────────────────────────────────────────────────────────────────────
# Unified effective-throughput entries.
#
# Single call site for every compute / memory-bandwidth time calculation.
# Precedence: explicit per-hardware override (YAML → GPU field) wins; when
# unset, fall back to the size-bucketed achieved_* heuristic above (which is
# the pre-existing, MLPerf-calibrated default behavior — preserved on purpose).
#
# NOT to be used for the MFU/HFU metric denominator (that is achieved-vs-
# *theoretical-peak* by definition) nor the structural roofline-knee check.
# ─────────────────────────────────────────────────────────────────────────────

def effective_flops(gpu, dtype: Dtype, flops: float) -> float:
    """Effective compute throughput (FLOP/s) = peak × utilization."""
    peak = peak_tflops_for(gpu, dtype)
    if peak <= 0 or flops <= 0:
        return 0.0
    override = getattr(gpu, "compute_efficiency", None)
    eff = (
        override if override is not None
        else achieved_flops_efficiency(gpu.name, dtype, flops)
    )
    return peak * eff


def effective_hbm_bw_bps(gpu, bytes_: float) -> float:
    """Effective HBM bandwidth (bytes/s) = peak × utilization."""
    bw = gpu.hbm_bw_gbps * 1e9
    if bw <= 0 or bytes_ <= 0:
        return 0.0
    override = getattr(gpu, "mem_bw_efficiency", None)
    eff = (
        override if override is not None
        else achieved_bandwidth_efficiency(gpu.name, bytes_)
    )
    return bw * eff


_FALLBACK_WARNED: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    """Emit a UserWarning at most once per process for a given key."""
    if key in _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED.add(key)
    import warnings
    warnings.warn(msg, UserWarning, stacklevel=3)


def peak_tflops_for(gpu, dtype: Dtype) -> float:
    """Return hardware peak FLOP/s (not TFLOP/s) for ``dtype`` on ``gpu``.

    Fallbacks (each emits a one-shot warning):
      FP4  → FP8 peak when gpu.flops_fp4 == 0
      FP8* → BF16 peak when gpu.flops_fp8 == 0
    BF16/FP16/FP32 all use gpu.flops_bf16 (no separate FP16 field).
    """
    if dtype is Dtype.FP4:
        if getattr(gpu, "flops_fp4", 0.0) > 0:
            return gpu.flops_fp4 * 1e12
        _warn_once(f"fp4_fallback_{gpu.name}",
                   f"GPU {gpu.name!r} declares no fp4_tops; falling back to FP8 peak")
        dtype = Dtype.FP8_E4M3
    if dtype in (Dtype.FP8_E4M3, Dtype.FP8_E5M2):
        if getattr(gpu, "flops_fp8", 0.0) > 0:
            return gpu.flops_fp8 * 1e12
        _warn_once(f"fp8_fallback_{gpu.name}",
                   f"GPU {gpu.name!r} declares no fp8_tops; falling back to BF16 peak")
    return gpu.flops_bf16 * 1e12
