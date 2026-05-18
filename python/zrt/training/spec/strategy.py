from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zrt.training.spec.model import ModelSpec
    from zrt.training.spec.system import SystemSpec


def rank_product(tp: int, cp: int, pp: int, ep: int, dp: int) -> int:
    """Calculate the rank product for a parallel configuration.

    TP * CP * PP * DP = world_size

    EP (Expert Parallelism) is handled *inside* the existing ranks and
    does not consume additional distinct ranks.
    """
    return tp * cp * pp * dp


class PPSched(Enum):
    ONE_F_ONE_B = "1f1b"
    INTERLEAVED = "i1f1b"
    ZERO_BUBBLE = "zb"
    DUALPIPE = "dualpipe"
    DUALPIPE_V = "dualpipev"


class CPKind(Enum):
    NONE = "none"
    ULYSSES = "ulysses"
    RING = "ring"
    HYBRID = "hybrid"
    COMPRESSED = "compressed"


class TPOverlap(Enum):
    NONE = "none"
    COC = "coc"
    MC2 = "mc2"


class OptKind(Enum):
    ADAM = "adam"
    MUON = "muon"


_MUON_NS_STEPS_DEFAULTS: dict[str, int] = {
    "deepseek_v4": 10,
    "deepseek_v3": 5,
    "default": 5,
}


@dataclass
class MuonConfig:
    ns_steps: int = 5
    ns_variant: str = "zolo_pd"
    rotation: bool = True
    adam_param_types: set[str] = field(
        default_factory=lambda: {"embed", "lm_head", "router", "bias"}
    )
    muon_param_fraction: float = 0.85


def resolve_muon_ns_steps(
    muon_config: MuonConfig | None,
    model: "ModelSpec",
) -> int:
    if muon_config is None:
        muon_config = MuonConfig()
    if muon_config.ns_steps != MuonConfig.__dataclass_fields__["ns_steps"].default:
        return muon_config.ns_steps
    if hasattr(model, "muon_ns_steps") and model.muon_ns_steps is not None:
        return model.muon_ns_steps
    model_type = getattr(model, "model_type", None)
    if model_type in _MUON_NS_STEPS_DEFAULTS:
        return _MUON_NS_STEPS_DEFAULTS[model_type]
    return _MUON_NS_STEPS_DEFAULTS["default"]


@dataclass
class RecomputePolicy:
    # Per-LayerKind value → set of recompute categories.
    # Canonical categories:
    #   "full"        : recompute entire layer (only input + output saved)
    #   "attn_core"   : selective recompute — only FA kernel + indexer +
    #                   compressor pool (Megatron-LM 'selective' flavor)
    #   "attn_block"  : attn_core scope + QKV / O linear projections
    #                   (the heavier 'rerun attention block' flavor)
    #   "ffn_swiglu"  : up/gate/down + swiGLU intermediates
    #   "ln"          : LayerNorm / RMSNorm intermediates
    #   "hc"          : DeepSeek-V4 mhc_pre / mhc_post / mhc_head
    # Deprecated alias (kept for backward compatibility, resolved by the
    # category-matching logic in flops._op_recompute_categories):
    #   "attn"  → "attn_block"
    per_layer: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class OffloadPolicy:
    opt_state: bool = False
    grads: bool = False
    params: bool = False  # ZeRO-Infinity style
    pct: float = 1.0      # fraction offloaded (0..1)


@dataclass
class Strategy:
    # parallelism degrees
    tp: int = 1
    cp: int = 1
    pp: int = 1
    ep: int = 1
    dp: int = 1

    # batch
    micro_batch: int = 1
    global_batch: int = 0  # 0 → derived from μbatch × dp × grad_accum

    # pipeline
    pp_schedule: PPSched = PPSched.ONE_F_ONE_B
    vpp_chunks: int = 1
    pp_layer_assignment: list[int] | None = None

    # context parallel
    cp_kind: CPKind = CPKind.NONE

    # memory
    zero_stage: int = 0  # 0/1/2/3; 3 == FSDP
    recompute: RecomputePolicy = field(default_factory=RecomputePolicy)
    offload: OffloadPolicy = field(default_factory=OffloadPolicy)

    # overlap
    tp_overlap: TPOverlap = TPOverlap.NONE
    ep_overlap: bool = False
    dualbatch: bool = False
    dp_overlap_in_bubble: bool = True
    # Fraction of steady-state backward compute that DP grad-reduce can overlap
    # (FSDP / Megatron grad bucket scheduling). 0.0 = legacy cooldown-only window;
    # 0.5 = conservative default; 1.0 = idealized full overlap.
    dp_steady_overlap_ratio: float = 0.5

    # optimizer
    optimizer: OptKind = OptKind.ADAM
    muon_config: MuonConfig | None = None

    def num_microbatches(self) -> int:
        """Number of microbatches per training step."""
        if self.global_batch > 0:
            return self.global_batch // (self.micro_batch * self.dp)
        return 1

    def validate(self, model: "ModelSpec", system: "SystemSpec") -> None:
        """Raise ValueError on invalid strategy + model + system combos."""
        errors: list[str] = []

        total = rank_product(self.tp, self.cp, self.pp, self.ep, self.dp)
        if total != system.world_size:
            errors.append(
                f"TP({self.tp})*CP({self.cp})*PP({self.pp})*DP({self.dp})="
                f"{total} != world_size({system.world_size})"
            )

        if model.num_heads % self.tp != 0:
            errors.append(
                f"num_heads({model.num_heads}) not divisible by TP({self.tp})"
            )
        if model.num_kv_heads % self.tp != 0 and model.num_kv_heads >= self.tp:
            errors.append(
                f"num_kv_heads({model.num_kv_heads}) not divisible by TP({self.tp})"
            )
        if model.ffn % self.tp != 0:
            errors.append(
                f"ffn({model.ffn}) not divisible by TP({self.tp})"
            )

        if self.cp_kind == CPKind.ULYSSES and model.num_heads % self.cp != 0:
            errors.append(
                f"Ulysses CP requires num_heads({model.num_heads}) % cp({self.cp}) == 0"
            )

        if self.ep > 1:
            if model.num_experts <= 0:
                errors.append("EP > 1 requires num_experts > 0")
            elif model.num_experts % self.ep != 0:
                errors.append(
                    f"num_experts({model.num_experts}) not divisible by EP({self.ep})"
                )

        if self.zero_stage >= 1 and self.dp <= 1:
            errors.append(f"ZeRO-{self.zero_stage} requires DP > 1")

        if self.global_batch > 0:
            if self.global_batch % (self.micro_batch * self.dp) != 0:
                errors.append(
                    f"global_batch({self.global_batch}) not divisible by "
                    f"micro_batch({self.micro_batch})*dp({self.dp})"
                )

        if self.pp > 1 and self.pp > len(model.layers):
            errors.append(
                f"PP({self.pp}) > num_layers({len(model.layers)})"
            )

        if errors:
            raise ValueError("Strategy validation failed:\n  " + "\n  ".join(errors))
