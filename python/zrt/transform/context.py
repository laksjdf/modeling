"""TransformContext and configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.hardware.spec import HardwareSpec


@dataclass
class ParallelConfig:
    tp: int = 1
    pp: int = 1
    ep: int = 1
    dp: int = 1
    cp: int = 1
    sp: bool = False

    @property
    def total_devices(self) -> int:
        return self.tp * self.pp * self.ep * self.dp * self.cp

    def describe(self) -> str:
        parts = []
        if self.tp > 1: parts.append(f"TP{self.tp}")
        if self.ep > 1: parts.append(f"EP{self.ep}")
        if self.pp > 1: parts.append(f"PP{self.pp}")
        if self.dp > 1: parts.append(f"DP{self.dp}")
        if self.cp > 1: parts.append(f"CP{self.cp}")
        if self.sp:     parts.append("SP")
        return "-".join(parts) or "single"


@dataclass
class StreamConfig:
    """Multi-stream execution configuration."""
    num_compute_streams: int = 1
    num_comm_streams: int = 1

    @property
    def total(self) -> int:
        return self.num_compute_streams + self.num_comm_streams

    def compute_stream_id(self, idx: int = 0) -> int:
        return idx % self.num_compute_streams

    def comm_stream_id(self, idx: int = 0) -> int:
        return self.num_compute_streams + (idx % self.num_comm_streams)


@dataclass
class QuantConfig:
    weight:     str = "bf16"   # "int8", "int4", "w8a8", "w4a16", ...
    activation: str = "bf16"
    kv_cache:   str = "bf16"

    @property
    def weight_bytes(self) -> float:
        _map = {"int4": 0.5, "int8": 1.0, "fp8": 1.0, "bf16": 2.0, "fp16": 2.0, "fp32": 4.0}
        return _map.get(self.weight.lower(), 2.0)


@dataclass
class TrainingConfig:
    """Training-specific configuration for performance modelling."""

    # Optimizer settings
    optimizer: str = "adam"  # "adam", "adamw", "muon"
    zero_stage: int = 1  # 0=none, 1=opt_state, 2=grads+opt, 3=weights+grads+opt

    # Batch size
    micro_batch: int = 1
    global_batch: int = 32

    # Recompute policy
    recompute_policy: str = "none"  # "none", "full", "selective"

    # Pipeline schedule
    pp_schedule: str = "1f1b"  # "1f1b", "interleaved", "dualpipe", "dualpipev", "zb"
    vpp_chunks: int = 1

    # Optional explicit layer→stage assignment for PP; length must equal
    # the number of traced transformer layers.  None → greedy bin-packing.
    pp_layer_assignment: list[int] | None = None

    # Context parallel strategy
    cp_kind: str = "ulysses"  # "ulysses", "ring", "hybrid", "none"

    # Overlap DP allreduce with PP bubble window
    dp_overlap_in_bubble: bool = True

    @property
    def num_microbatches(self) -> int:
        return self.global_batch // self.micro_batch


@dataclass
class TransformContext:
    hw_spec:      "HardwareSpec"
    parallel:     ParallelConfig  = field(default_factory=ParallelConfig)
    stream_config: StreamConfig   = field(default_factory=StreamConfig)
    quant:        QuantConfig | None = None
    training:     TrainingConfig | None = None  # Training-specific config
    optim_flags:  set[str]        = field(default_factory=set)
    phase:        str             = "prefill"
    profile:      Any             = None   # ModelProfile (optional)
    stack:        Any             = None   # SoftwareStack (optional)

    @property
    def is_training(self) -> bool:
        return self.training is not None
