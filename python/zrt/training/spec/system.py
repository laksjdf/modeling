from __future__ import annotations

from dataclasses import dataclass, field

from zrt.hardware.spec import InterconnectSpec, LinkSpec


@dataclass
class GPU:
    name: str
    flops_bf16: float   # peak TFLOP/s
    flops_fp8: float    # peak TFLOP/s
    hbm_gb: float
    hbm_bw_gbps: float  # aggregate HBM bandwidth (GB/s)

    flops_fp4: float = 0.0    # peak TFLOP/s (Blackwell B200/B300+); 0 = unsupported

    # Heterogeneous-core fields (None = homogeneous, use flops_bf16 for all ops).
    # "cube" is the matrix/Tensor/Cube peak; both peaks must be set to enable
    # heterogeneous timing.
    cube_tflops: float | None = None      # Matrix/Tensor/Cube peak TFLOP/s (bf16)
    vector_tflops: float | None = None    # Scalar/vector peak TFLOP/s (bf16)
    overlap_ratio: dict[str, float] = field(default_factory=dict)

    # SRAM per SM — used for FlashAttention tile-level modeling.
    # 0 = tile model disabled (preserve current single-read byte formula).
    sram_kb_per_sm: float = 0.0

    # Number of EP A2A overlap waves. 0 = hardware does not support K-wave EP
    # overlap (e.g. Ascend HCCS). Mirrors HardwareSpec.compute.ep_overlap_waves.
    ep_overlap_waves: int = 0

    # Explicit utilization overrides (None = use achieved_* heuristic curve).
    # Mirror HardwareSpec.compute.compute_efficiency / memory.mem_bw_efficiency.
    compute_efficiency: float | None = None
    mem_bw_efficiency: float | None = None


@dataclass
class SystemSpec:
    gpu: GPU
    host_mem_gb: float
    interconnect: InterconnectSpec
    nodes: int
    gpus_per_node: int
    world_size_override: int | None = None

    def __post_init__(self) -> None:
        if self.world_size_override is not None:
            if self.world_size_override <= 0:
                raise ValueError("world_size_override must be positive")
            if self.world_size_override > self.allocated_gpus:
                raise ValueError(
                    "world_size_override cannot exceed allocated GPUs "
                    f"({self.world_size_override} > {self.allocated_gpus})"
                )

    @property
    def allocated_gpus(self) -> int:
        return self.nodes * self.gpus_per_node

    @property
    def world_size(self) -> int:
        if self.world_size_override is not None:
            return self.world_size_override
        return self.allocated_gpus

    @property
    def idle_gpus(self) -> int:
        return self.allocated_gpus - self.world_size

    def intra_tier(self) -> LinkSpec:
        return self.interconnect.intra_node

    def inter_tier(self) -> LinkSpec:
        return self.interconnect.inter_node
