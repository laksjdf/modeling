from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GPU:
    name: str
    flops_bf16: float   # peak TFLOP/s
    flops_fp8: float    # peak TFLOP/s
    hbm_gb: float
    hbm_bw_gbps: float  # aggregate HBM bandwidth (GB/s)


@dataclass
class NetTier:
    scope: str          # "intra_node" | "inter_node"
    bw_gbps: float      # per-link unidirectional GB/s
    latency_us: float
    topology: str       # "ring" | "tree" | "nvswitch" | "fattree"


@dataclass
class SystemSpec:
    gpu: GPU
    host_mem_gb: float
    nets: list[NetTier]  # ordered: intra first
    nodes: int
    gpus_per_node: int

    @property
    def world_size(self) -> int:
        return self.nodes * self.gpus_per_node

    def intra_tier(self) -> NetTier | None:
        for t in self.nets:
            if t.scope == "intra_node":
                return t
        return None

    def inter_tier(self) -> NetTier | None:
        for t in self.nets:
            if t.scope == "inter_node":
                return t
        return None
