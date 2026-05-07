from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind


@dataclass
class Tensor:
    name: str
    shape_logical: tuple[int, ...]  # before sharding
    shape_local: tuple[int, ...]    # after sharding (per-rank)
    dtype: Dtype
    is_activation: bool
    is_param: bool = False

    def num_elements(self) -> int:
        result = 1
        for d in self.shape_local:
            result *= d
        return result

    def nbytes(self) -> int:
        return self.num_elements() * self.dtype.bytes


@dataclass
class Op:
    name: str
    kind: str   # "matmul" | "attn_core" | "softmax" | "ln" | "rope"
                # | "swiglu" | "router" | "dispatch" | "combine"
                # | "embed" | "lm_head" | "add"
    inputs: list[Tensor] = field(default_factory=list)
    outputs: list[Tensor] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    layer_id: int = -1
    layer_kind: LayerKind = LayerKind.DENSE


@dataclass
class Collective:
    name: str
    kind: str       # "AG" | "RS" | "AR" | "A2A" | "P2P"
    group: str      # "TP" | "CP" | "EP" | "DP" | "PP"
    bytes_: int     # per-rank payload in bytes
    inserted_after: str | None = None  # op name it follows
    inserted_before: str | None = None  # op name it precedes
    rounds: int = 1  # number of rounds for P2P (Ring CP)
    overlap: bool = False  # whether this communication overlaps with compute
    phase: str = "fwd"  # "fwd" | "bwd" | "both"

    @property
    def payload_mb(self) -> float:
        return self.bytes_ / (1024 * 1024)


@dataclass
class Graph:
    ops: list[Op] = field(default_factory=list)
    collectives: list[Collective] = field(default_factory=list)
    layer_index: dict[int, tuple[int, int]] = field(default_factory=dict)

    def ops_for_layer(self, layer_id: int) -> list[Op]:
        if layer_id not in self.layer_index:
            return []
        start, end = self.layer_index[layer_id]
        return self.ops[start:end]

    def ops_for_stage(self, stage_layer_ids: list[int]) -> list[Op]:
        result = []
        for lid in stage_layer_ids:
            result.extend(self.ops_for_layer(lid))
        return result
