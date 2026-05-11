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

    # Heterogeneous-core fields (None = homogeneous, use flops_bf16 for all ops).
    # "cube" is the matrix/Tensor/Cube peak; both peaks must be set to enable
    # heterogeneous timing.
    cube_tflops: float | None = None      # Matrix/Tensor/Cube peak TFLOP/s (bf16)
    vector_tflops: float | None = None    # Scalar/vector peak TFLOP/s (bf16)
    overlap_ratio: dict[str, float] = field(default_factory=dict)


@dataclass
class SystemSpec:
    gpu: GPU
    host_mem_gb: float
    interconnect: InterconnectSpec
    nodes: int
    gpus_per_node: int

    @property
    def world_size(self) -> int:
        return self.nodes * self.gpus_per_node

    def intra_tier(self) -> LinkSpec:
        return self.interconnect.intra_node

    def inter_tier(self) -> LinkSpec:
        return self.interconnect.inter_node
