from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zrt.training.spec.model import ModelSpec
    from zrt.training.spec.system import SystemSpec


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


class TPOverlap(Enum):
    NONE = "none"
    COC = "coc"
    MC2 = "mc2"


class OptKind(Enum):
    ADAM = "adam"
    MUON = "muon"


@dataclass
class RecomputePolicy:
    # per-LayerKind value -> set of op categories to recompute
    # categories: "full" | "attn" | "attn_upscale" | "ffn_swiglu" | "ln"
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

    # optimizer
    optimizer: OptKind = OptKind.ADAM

    def num_microbatches(self) -> int:
        """Number of microbatches per training step."""
        if self.global_batch > 0:
            return self.global_batch // (self.micro_batch * self.dp)
        return 1

    def validate(self, model: "ModelSpec", system: "SystemSpec") -> None:
        """Raise ValueError on invalid strategy + model + system combos."""
        errors: list[str] = []

        total = self.tp * self.cp * self.pp * self.dp
        if total != system.world_size:
            errors.append(
                f"TP({self.tp})*CP({self.cp})*PP({self.pp})*DP({self.dp})="
                f"{total} != world_size({system.world_size})"
            )

        if model.num_heads % self.tp != 0:
            errors.append(
                f"num_heads({model.num_heads}) not divisible by TP({self.tp})"
            )
        if model.num_kv_heads % self.tp != 0:
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
