"""Hardware specification dataclasses.

Purely describes hardware capabilities — no software/framework knowledge here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.ir.types import DType


# ─────────────────────────────────────────────────────────────────────────────
# Sub-specs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComputeSpec:
    """Peak compute throughput (theoretical).

    All values are in tera-ops/s (TFLOPS / TOPS).
    Simulators may apply an efficiency factor to get effective throughput.
    """
    fp16_tflops: float = 0.0
    bf16_tflops: float = 0.0
    fp32_tflops: float = 0.0
    int8_tops:   float = 0.0
    int4_tops:   float = 0.0
    fp8_tops:    float = 0.0   # covers both FP8_E4M3 and FP8_E5M2

    # Heterogeneous-core fields (matrix/Tensor/Cube vs scalar/vector split).
    # None = homogeneous hardware (fall back to unified peak_flops path).
    # Both peaks must be set before heterogeneous timing is used.
    cube_bf16_tflops: float | None = None
    vector_bf16_tflops: float | None = None
    overlap_ratio: dict[str, float] = field(default_factory=dict)

    # SRAM per SM for FlashAttention tile-level modeling (KB).
    # 0 = tile model disabled.
    sram_kb_per_sm: float = 0.0


@dataclass
class MemoryTier:
    """A single level of the memory hierarchy."""
    name: str
    bandwidth_gbps: float
    capacity_mb: float = 0.0   # 0 = effectively unbounded (e.g. HBM)


@dataclass
class MemorySpec:
    """On-device memory configuration."""
    capacity_gb: float
    hbm_bandwidth_gbps: float
    l2_cache_mb: float = 0.0
    tiers: list[MemoryTier] = field(default_factory=list)


@dataclass
class LinkSpec:
    """Point-to-point or collective interconnect properties.

    ``bandwidth_gbps`` is the **total aggregate bidirectional bandwidth** for
    this interconnect as computed by the registry from the YAML definition.

    For ring-style collectives, the effective per-direction bandwidth is
    typically ``bandwidth_gbps / 2``.
    """
    type: str                     # e.g. "HCCS", "NVLink", "RoCE", "IB"
    bandwidth_gbps: float         # total aggregate bidirectional BW (GB/s)
    latency_us: float
    topology: str = "point_to_point"
    num_devices: int = 1          # devices in the domain (e.g. 8 for a node)


@dataclass
class InterconnectSpec:
    intra_node: LinkSpec
    inter_node: LinkSpec


# ─────────────────────────────────────────────────────────────────────────────
# HardwareSpec
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HardwareSpec:
    """Complete hardware description for one accelerator device.

    Intentionally vendor-neutral: Ascend NPUs and NVIDIA GPUs share the
    same structure, identified by ``vendor`` / ``device_type``.
    """
    name: str
    vendor: str           # "huawei" | "nvidia" | "amd" | ...
    device_type: str      # "npu" | "gpu" | "cpu"
    compute: ComputeSpec
    memory: MemorySpec
    interconnect: InterconnectSpec

    # ── Convenience helpers ───────────────────────────────────────────────

    def peak_flops(self, dtype: "DType") -> float:
        """Return theoretical peak throughput in FLOPs/s (or Ops/s for INT).

        Returns 0.0 for dtypes not supported / not configured.
        """
        # import here to avoid circular imports at module level
        from python.zrt.ir.types import DType as _DType

        _map = {
            _DType.FP16:     self.compute.fp16_tflops,
            _DType.BF16:     self.compute.bf16_tflops,
            _DType.FP32:     self.compute.fp32_tflops,
            _DType.INT8:     self.compute.int8_tops,
            _DType.INT4:     self.compute.int4_tops,
            _DType.FP8_E4M3: self.compute.fp8_tops,
            _DType.FP8_E5M2: self.compute.fp8_tops,
        }
        tera_ops = _map.get(dtype, 0.0)
        return tera_ops * 1e12   # → ops/s

    def hbm_bandwidth(self) -> float:
        """Return HBM bandwidth in bytes/s."""
        return self.memory.hbm_bandwidth_gbps * 1e9

    def __repr__(self) -> str:
        return (
            f"HardwareSpec({self.name!r}, {self.device_type}, "
            f"bf16={self.compute.bf16_tflops}T, "
            f"hbm={self.memory.capacity_gb}GB@{self.memory.hbm_bandwidth_gbps}GB/s)"
        )
