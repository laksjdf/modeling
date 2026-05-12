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
class OffloadConfig:
    """Host-device memory offloading configuration."""
    pct: float = 0.0          # Fraction of each component to offload [0.0, 1.0]
    opt_state: bool = False   # Offload optimizer state (Adam momentum/variance)
    grads: bool = False       # Offload gradients after reduction
    params: bool = False      # Offload parameters (requires CPU-GPU sync)


@dataclass
class TrainingConfig:
    """Training-specific configuration for performance modelling."""

    # Optimizer settings
    optimizer: str = "adam"  # "adam", "adamw", "muon"
    zero_stage: int = 1  # 0=none, 1=opt_state, 2=grads+opt, 3=weights+grads+opt
    muon_ns_steps: int | None = None  # Newton-Schulz iterations for Muon
    muon_param_fraction: float | None = None  # Fraction of params using Muon
    muon_rotation: bool = True  # Moonshot rotation optimization for Muon

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

    # Memory offloading (optional, disabled by default)
    offload: OffloadConfig | None = None

    def effective_ns_steps(self, model_type: str | None = None) -> int:
        """Return effective NS steps, handling None fallback logic.

        Priority:
          1. self.muon_ns_steps (explicit config)
          2. _MUON_NS_STEPS_DEFAULTS[model_type] (model type lookup)
          3. Default 5
        """
        if self.muon_ns_steps is not None:
            return self.muon_ns_steps
        from zrt.training.spec.strategy import _MUON_NS_STEPS_DEFAULTS
        if model_type is not None:
            return _MUON_NS_STEPS_DEFAULTS.get(model_type, 5)
        return 5

    @property
    def num_microbatches(self) -> int:
        return self.global_batch // self.micro_batch


@dataclass
class FusionConfig:
    """User-facing operator-fusion controls.

    Loaded from YAML (default search path or ``--fusion-config``).  Filters
    which fusion rules fire during a single ``FusionPass`` run.

    enabled_rules
        ``None`` → use each rule's ``default_phases`` for the active phase.
        Non-empty set → only rules whose ``name`` is in this set are active.
    disabled_rules
        Always subtracted after ``enabled_rules`` resolution.  Useful for
        starting from defaults and removing a few specific rules.
    allow_structural_collapse
        Re-enables the legacy ``op_type = module_class`` fallback for
        unmatched multi-op buckets.  Off by default — unmatched ops stay
        as raw aten nodes.
    merge_sibling_classes
        Module class names for which ``_merge_parent_groups`` is allowed
        to fuse runs of identical sibling instances (e.g. ``Expert`` lists
        in MoE).  Off by default to keep buckets aligned with single
        forward calls.
    """

    enabled_rules: set[str] | None = None
    disabled_rules: set[str] = field(default_factory=set)
    allow_structural_collapse: bool = False
    merge_sibling_classes: set[str] = field(default_factory=set)

    @classmethod
    def merged(cls, base: "FusionConfig | None",
               override: "FusionConfig | None") -> "FusionConfig":
        """Layer ``override`` on top of ``base`` (override wins on every field)."""
        b = base or cls()
        if override is None:
            return cls(
                enabled_rules=set(b.enabled_rules) if b.enabled_rules is not None else None,
                disabled_rules=set(b.disabled_rules),
                allow_structural_collapse=b.allow_structural_collapse,
                merge_sibling_classes=set(b.merge_sibling_classes),
            )
        return cls(
            enabled_rules=(set(override.enabled_rules)
                           if override.enabled_rules is not None
                           else (set(b.enabled_rules) if b.enabled_rules is not None else None)),
            disabled_rules=set(b.disabled_rules) | set(override.disabled_rules),
            allow_structural_collapse=override.allow_structural_collapse,
            merge_sibling_classes=set(b.merge_sibling_classes) | set(override.merge_sibling_classes),
        )


@dataclass
class TransformContext:
    hw_spec:      "HardwareSpec"
    parallel:     ParallelConfig  = field(default_factory=ParallelConfig)
    stream_config: StreamConfig   = field(default_factory=StreamConfig)
    quant:        QuantConfig | None = None
    training:     TrainingConfig | None = None  # Training-specific config
    fusion:       FusionConfig    = field(default_factory=FusionConfig)
    optim_flags:  set[str]        = field(default_factory=set)
    phase:        str             = "prefill"
    profile:      Any             = None   # ModelProfile (optional)
    stack:        Any             = None   # SoftwareStack (optional)
    model_id:     str             = ""     # HF model id, used by FusionPass to load platform rules

    @property
    def is_training(self) -> bool:
        return self.training is not None

    def phase_for_fusion(self) -> str:
        """Return ``"training"`` when training, else ``"inference"``."""
        return "training" if self.is_training else "inference"
