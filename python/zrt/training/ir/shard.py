"""Sharding pass — apply TP/CP/EP sharding to IR and insert collectives."""

from __future__ import annotations

from zrt.training.ir.graph import Collective, Graph
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.strategy import CPKind, Strategy


class ShardPlan:
    """Holds sharding parameters derived from Strategy."""

    def __init__(self, strategy: Strategy):
        self.tp = strategy.tp
        self.cp = strategy.cp
        self.ep = strategy.ep
        self.dp = strategy.dp
        self.pp = strategy.pp
        self.cp_kind = strategy.cp_kind
        self.sp = strategy.tp > 1  # Megatron SP on when TP>1

    def shard_col_parallel(self, n: int) -> int:
        """Shard column dimension by TP."""
        return n // self.tp

    def shard_row_parallel(self, n: int) -> int:
        """Shard row dimension by TP (output dimension stays full via RS)."""
        return n // self.tp


def insert_collectives(graph: Graph, shard: ShardPlan, model: ModelSpec) -> None:
    """Insert TP collectives (AG/RS pairs) into the graph IN-PLACE.

    Megatron-SP pattern per dense layer:
      - AG before QKV (if input was RS'd by previous layer)
      - RS after O_proj
      - AG before FFN up/gate
      - RS after FFN down

    Phase 1: TP only. CP/EP added in Phase 2.
    """
    if shard.tp <= 1:
        return

    collectives: list[Collective] = []
    seq = model.seq_len
    h = model.hidden
    h_attn = model.num_heads * model.head_dim
    h_kv = model.num_kv_heads * model.head_dim
    ffn = model.ffn
    act_bytes = model.act_dtype.bytes

    for layer_id, (start, end) in graph.layer_index.items():
        # Compute payload sizes (before sharding, divided by TP for per-rank)
        qkv_bytes = seq * (h_attn + 2 * h_kv) * act_bytes // shard.tp
        o_proj_bytes = seq * h * act_bytes  # RS output is full size
        ag_attn_bytes = seq * h * act_bytes  # AG before QKV
        ag_ffn_bytes = seq * h * act_bytes   # AG before FFN up
        rs_attn_bytes = seq * h * act_bytes  # RS after O_proj
        rs_ffn_bytes = seq * h * act_bytes   # RS after FFN down

        for i in range(start, end):
            op = graph.ops[i]

            # AG before QKV projection (gathers seq-sharded input for col-parallel)
            if op.kind == "matmul" and "qkv" in op.name:
                collectives.append(Collective(
                    name=f"ag_{op.name}",
                    kind="AG", group="TP",
                    bytes_=ag_attn_bytes,
                    inserted_after=op.name,
                ))

            # RS after O projection
            if op.kind == "matmul" and "o_proj" in op.name:
                collectives.append(Collective(
                    name=f"rs_{op.name}",
                    kind="RS", group="TP",
                    bytes_=rs_attn_bytes,
                    inserted_after=op.name,
                ))

            # AG before FFN up projection
            if op.kind == "matmul" and "up_proj" in op.name:
                collectives.append(Collective(
                    name=f"ag_{op.name}",
                    kind="AG", group="TP",
                    bytes_=ag_ffn_bytes,
                    inserted_after=op.name,
                ))

            # RS after FFN down projection
            if op.kind == "matmul" and "down_proj" in op.name:
                collectives.append(Collective(
                    name=f"rs_{op.name}",
                    kind="RS", group="TP",
                    bytes_=rs_ffn_bytes,
                    inserted_after=op.name,
                ))

        # Adjust tensor shapes for TP sharding
        _apply_tp_sharding(graph, start, end, shard, h, h_attn, h_kv, ffn, seq, act_bytes)

    graph.collectives.extend(collectives)


def _apply_tp_sharding(
    graph: Graph, start: int, end: int, shard: ShardPlan,
    h: int, h_attn: int, h_kv: int, ffn: int, seq: int, act_bytes: int,
) -> None:
    """Adjust tensor shape_local for TP sharding on ops in [start, end)."""
    if shard.tp <= 1:
        return

    for i in range(start, end):
        op = graph.ops[i]

        if op.kind == "matmul":
            m, n, k = op.meta["m"], op.meta["n"], op.meta["k"]

            if "qkv" in op.name:
                # Col-parallel: shard n dimension (output) by TP
                n_local = n // shard.tp
                op.meta["n_local"] = n_local
                for t in op.outputs:
                    t.shape_local = (t.shape_logical[0], n_local)
            elif "o_proj" in op.name:
                # Row-parallel: shard k dimension (input) by TP
                k_local = k // shard.tp
                op.meta["k_local"] = k_local
                for t in op.inputs:
                    if t.shape_logical[-1] == k:
                        t.shape_local = (t.shape_logical[0], k_local)
            elif "up_proj" in op.name or "gate_proj" in op.name:
                # Col-parallel: shard n by TP
                n_local = n // shard.tp
                op.meta["n_local"] = n_local
                for t in op.outputs:
                    t.shape_local = (t.shape_logical[0], n_local)
            elif "down_proj" in op.name:
                # Row-parallel: shard k by TP
                k_local = k // shard.tp
                op.meta["k_local"] = k_local
                for t in op.inputs:
                    if t.shape_logical[-1] == k:
                        t.shape_local = (t.shape_logical[0], k_local)
