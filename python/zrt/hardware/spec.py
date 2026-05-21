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
    fp4_tops:    float = 0.0   # Blackwell B200/B300+ (NVFP4/MXFP4); 0 = unsupported

    # Heterogeneous-core fields (matrix/Tensor/Cube vs scalar/vector split).
    # None = homogeneous hardware (fall back to unified peak_flops path).
    # Both peaks must be set before heterogeneous timing is used.
    cube_bf16_tflops: float | None = None
    vector_bf16_tflops: float | None = None
    overlap_ratio: dict[str, float] = field(default_factory=dict)

    # SRAM per SM for FlashAttention tile-level modeling (KB).
    # 0 = tile model disabled.
    sram_kb_per_sm: float = 0.0

    # Number of EP A2A overlap waves (K-wave technique, e.g. DeepSeek V3).
    # 0 = hardware does not support K-wave EP overlap (e.g. Ascend HCCS).
    # NVIDIA CUDA-capable hardware: typically 4.
    ep_overlap_waves: int = 0

    # Explicit compute utilization override (achieved/peak FLOPs).
    # None = fall back to the size-bucketed achieved_flops_efficiency() curve
    # (MLPerf-calibrated). A YAML-supplied value takes precedence over it.
    compute_efficiency: float | None = None


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

    # Explicit HBM-bandwidth utilization override (achieved/peak).
    # None = fall back to the size-bucketed achieved_bandwidth_efficiency()
    # curve. A YAML-supplied value takes precedence over it.
    mem_bw_efficiency: float | None = None


# Topology → cost-law class. Drives both the alpha-beta latency-step count
# (comm.py) and the bandwidth-derate eligibility (effective_bw_bps below).
# Unknown topologies fall back to "ring" (conservative: (N-1)-step latency).
#   switched_full : single-step full connectivity (NVSwitch / full mesh).
#                   Non-blocking → full per-card bandwidth, no scale derate.
#   clos          : non-blocking switched fabric. log2(N) latency like a tree,
#                   but full per-card bandwidth — NEVER scale-derated (the
#                   "ideal" reference other fabrics are compared against).
#   switched_tree : over-subscribable switched fabric (fat-tree); log2(N)
#                   latency, bandwidth derated by domain scale.
#   ring / torus  : link-local; (N-1)-step latency, scale-derated.
_TOPO_CLASS = {
    "nvswitch": "switched_full", "all_to_all": "switched_full",
    "full_mesh": "switched_full",
    "clos": "clos",
    "fat_tree": "switched_tree",
    "ring": "ring", "torus": "torus", "mesh": "torus",
}

# Classes whose per-card bandwidth degrades as the parallel domain grows
# beyond the non-blocking radix. clos / switched_full are non-blocking and
# excluded by construction.
_SCALE_DERATED = ("switched_tree", "ring", "torus")


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
    # Bandwidth utilization (effective/peak), (0,1]. Applies to ALL topologies
    # (clos included). Sole knob for the bandwidth-realism derate.
    kb_efficiency: float = 0.7
    # Spine over-subscription ratio for non-clos switched fabrics (e.g. 4 = 4:1).
    # 1.0 = non-blocking (clos / default) → no domain-size derate.
    oversubscription: float = 1.0

    @property
    def topology_class(self) -> str:
        return _TOPO_CLASS.get(self.topology, "ring")

    def effective_bw_bps(self, group_size: int = 1) -> float:
        """Effective bandwidth in bytes/s.

        = peak × ``kb_efficiency``, then — for the scale-derated classes
        (fat-tree / ring / torus) — divided by a domain-scale factor::

            s = 1 + (oversubscription - 1) · (1 - R/N)        (N > R)

        where N = ``group_size`` and R = non-blocking radix (``num_devices``
        if > 0, else 0 → the whole link is the over-subscribed tier, e.g.
        inter-node fat-tree). s is 1.0 at the radix and rises monotonically
        toward ``oversubscription`` as the domain grows, so bandwidth falls
        monotonically with scale. Default ``oversubscription`` = 1.0 → s ≡ 1
        (no derate, bit-identical to peak × kb_efficiency).

        clos and switched_full are non-blocking by construction: they keep
        full per-card bandwidth at any scale and are NEVER derated (clos is
        the ideal reference the scale-derated fabrics are measured against).
        The algorithmic (N-1)/N data-volume factor lives in comm.py.
        """
        base = self.bandwidth_gbps * 1e9 / 8.0 * self.kb_efficiency
        if base <= 0.0:
            return 0.0
        if (
            self.topology_class in _SCALE_DERATED
            and self.oversubscription > 1.0
        ):
            radix = self.num_devices if self.num_devices > 0 else 0
            if group_size > radix:
                s = 1.0 + (self.oversubscription - 1.0) * (1.0 - radix / group_size)
                return base / s
        return base


@dataclass
class TopologyTier:
    """One level of the interconnect hierarchy.

    Tiers are ordered innermost → outermost. The innermost tier is the
    smallest, fastest scale-up domain (NVLink island / HCCS super-node);
    the outermost tier is the largest, slowest scale-out fabric (IB
    spine), typically with ``link.num_devices == 0`` to mark "unbounded".

    ``link.num_devices`` semantics: number of GPUs in **one instance** of
    this tier (so 8 for a per-node NVLink island, 72 for an NVL72 rack,
    0 for the outermost unbounded fabric). Cumulative coverage grows
    monotonically from innermost to outermost.
    """
    name: str
    link: LinkSpec


class InterconnectSpec:
    """N-tier interconnect description (innermost → outermost).

    Two construction modes — kept compatible so existing YAMLs and test
    fixtures continue to work:

    1. **New**: ``InterconnectSpec(tiers=[t0, t1, t2, ...])`` for N ≥ 1
       tiers, ordered innermost to outermost.
    2. **Legacy**: ``InterconnectSpec(intra_node=L, inter_node=L2)`` —
       equivalent to ``tiers=[TopologyTier("intra_node", L),
                              TopologyTier("inter_node", L2)]``.

    The ``intra_node`` / ``inter_node`` *properties* always read
    ``tiers[0]`` / ``tiers[-1]`` respectively, so call sites that read
    by either name keep working regardless of how the object was built.
    """

    __slots__ = ("tiers",)

    def __init__(
        self,
        tiers: list[TopologyTier] | None = None,
        *,
        intra_node: LinkSpec | None = None,
        inter_node: LinkSpec | None = None,
    ) -> None:
        if tiers is not None and (intra_node is not None or inter_node is not None):
            raise ValueError(
                "InterconnectSpec: pass either `tiers` or "
                "`intra_node`/`inter_node`, not both"
            )
        if tiers is not None:
            self.tiers: list[TopologyTier] = list(tiers)
        else:
            tlist: list[TopologyTier] = []
            if intra_node is not None:
                tlist.append(TopologyTier(name="intra_node", link=intra_node))
            if inter_node is not None:
                tlist.append(TopologyTier(name="inter_node", link=inter_node))
            self.tiers = tlist
        self._validate_tiers()

    def _validate_tiers(self) -> None:
        """Validate tier ordering invariants shared by direct/YAML construction."""
        last_idx = len(self.tiers) - 1
        for i, tier in enumerate(self.tiers):
            if tier.link.num_devices == 0 and i != last_idx:
                raise ValueError(
                    "InterconnectSpec: only outermost tier may be unbounded "
                    f"(num_devices=0); got tier {i} {tier.name!r}"
                )

    # ── Back-compat accessors ─────────────────────────────────────────
    @property
    def intra_node(self) -> LinkSpec:
        if not self.tiers:
            raise ValueError("InterconnectSpec has no tiers configured")
        return self.tiers[0].link

    @property
    def inter_node(self) -> LinkSpec:
        if not self.tiers:
            raise ValueError("InterconnectSpec has no tiers configured")
        return self.tiers[-1].link

    # ── N-tier helpers ────────────────────────────────────────────────
    def innermost_tier_for(self, group_size: int) -> TopologyTier:
        """Smallest tier whose per-instance domain still contains the group.

        Iterates tiers innermost → outermost; returns the first tier
        whose ``link.num_devices`` is ≥ ``group_size`` (or the outermost
        tier, treated as unbounded, when none fit).

        A tier with ``link.num_devices == 0`` is treated as unbounded
        and matches any group size — appropriate for the outermost IB
        spine where the domain is the whole cluster.
        """
        if not self.tiers:
            raise ValueError("InterconnectSpec has no tiers configured")
        for tier in self.tiers:
            n = tier.link.num_devices
            if n == 0 or group_size <= n:
                return tier
        return self.tiers[-1]

    def __repr__(self) -> str:
        return f"InterconnectSpec(tiers={self.tiers!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InterconnectSpec):
            return NotImplemented
        return self.tiers == other.tiers

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(tuple(
            (
                t.name,
                t.link.type,
                t.link.bandwidth_gbps,
                t.link.latency_us,
                t.link.topology,
                t.link.num_devices,
                t.link.kb_efficiency,
                t.link.oversubscription,
            )
            for t in self.tiers
        ))


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
