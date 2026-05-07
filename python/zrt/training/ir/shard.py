"""Sharding pass — apply TP/CP/EP sharding to IR and insert collectives."""

from __future__ import annotations

from zrt.training.ir.training_graph import Collective, Graph
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

    def shard_seq_parallel(self, s: int) -> int:
        """Shard sequence dimension by CP (Ulysses)."""
        return s // self.cp if self.cp > 1 else s

    def shard_expert_parallel(self, n: int) -> int:
        """Shard experts by EP (each rank gets n/experts)."""
        return n // self.ep if self.ep > 1 else n

    @property
    def has_cp(self) -> bool:
        """Whether context parallelism is enabled."""
        return self.cp > 1

    @property
    def has_ep(self) -> bool:
        """Whether expert parallelism is enabled."""
        return self.ep > 1

    @property
    def has_tp(self) -> bool:
        """Whether tensor parallelism is enabled."""
        return self.tp > 1


def insert_collectives(graph: Graph, model: ModelSpec, strategy: Strategy) -> None:
    """Insert TP/CP/EP collectives into the graph IN-PLACE.

    Megatron-TP pattern per dense layer:
      - AG before QKV (if input was RS'd by previous layer)
      - RS after O_proj
      - AG before FFN up/gate
      - RS after FFN down

    Ulysses-CP pattern:
      - A2A before attention (splits sequence across ranks)
      - A2A after attention (gathers sequence)

    EP pattern (MoE only):
      - All-to-All before routed expert FFN
      - All-to-All after routed expert FFN
    """
    shard = ShardPlan(strategy)
    collectives: list[Collective] = []

    # Insert TP collectives
    if shard.has_tp:
        _insert_tp_collectives(graph, shard, model, collectives)

    # Insert CP collectives (Ulysses sequence parallel)
    if shard.has_cp:
        _insert_cp_collectives(graph, shard, model, collectives)

    # Insert EP collectives (expert parallel for MoE)
    if shard.has_ep:
        _insert_ep_collectives(graph, shard, model, collectives)

    graph.collectives.extend(collectives)


def _insert_tp_collectives(
    graph: Graph, shard: ShardPlan, model: ModelSpec,
    collectives: list[Collective],
) -> None:
    """Insert TP collectives (AG/RS pairs) into the graph."""
    seq = model.seq_len
    h = model.hidden
    h_attn = model.num_heads * model.head_dim
    h_kv = model.num_kv_heads * model.head_dim
    ffn = model.ffn
    act_bytes = model.act_dtype.bytes
    
    # When CP is enabled, sequence is already sharded by CP
    # TP AG/RS should operate on the CP-sharded sequence
    if shard.cp > 1:
        seq = seq // shard.cp

    for layer_id, (start, end) in graph.layer_index.items():
        # Compute payload sizes (after CP sharding, before TP sharding)
        ag_attn_bytes = seq * h * act_bytes  # AG before QKV
        rs_attn_bytes = seq * h * act_bytes  # RS after O_proj
        ag_ffn_bytes = seq * h * act_bytes   # AG before FFN up
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
                    phase="both",
                ))

            # RS after O projection
            if op.kind == "matmul" and "o_proj" in op.name:
                collectives.append(Collective(
                    name=f"rs_{op.name}",
                    kind="RS", group="TP",
                    bytes_=rs_attn_bytes,
                    inserted_after=op.name,
                    phase="both",
                ))

            # AG before FFN up projection
            if op.kind == "matmul" and "up_proj" in op.name:
                collectives.append(Collective(
                    name=f"ag_{op.name}",
                    kind="AG", group="TP",
                    bytes_=ag_ffn_bytes,
                    inserted_after=op.name,
                    phase="both",
                ))

            # RS after FFN down projection
            if op.kind == "matmul" and "down_proj" in op.name:
                collectives.append(Collective(
                    name=f"rs_{op.name}",
                    kind="RS", group="TP",
                    bytes_=rs_ffn_bytes,
                    inserted_after=op.name,
                    phase="both",
                ))

        # Adjust tensor shapes for TP sharding
        _apply_tp_sharding(graph, start, end, shard, h, h_attn, h_kv, ffn, seq, act_bytes)


def _insert_cp_collectives(
    graph: Graph, shard: ShardPlan, model: ModelSpec,
    collectives: list[Collective],
) -> None:
    """Insert CP collectives for different CP strategies.

    Ulysses-CP pattern:
      - A2A before attention core (scatter-seq / gather-heads)
      - A2A after attention core (gather-seq / scatter-heads)
      - Each A2A transfers seq/cp × hidden_tp bytes

    Ring-CP pattern:
      - P2P send/recv pairs, cp rounds per attention
      - Each P2P round transfers seq/cp × hidden_tp bytes
      - P2P overlaps with FA tile computation

    Hybrid-CP pattern:
      - Combines Ulysses A2A with Ring P2P overlap
      - For now, modeled as Ring-CP with extra A2A

    Note: When TP is enabled, hidden dimension is already sharded by TP.
    CP communication should use hidden_tp = hidden / tp, not full hidden.
    """
    if shard.cp_kind == CPKind.NONE:
        return

    h = model.hidden
    if shard.tp > 1:
        h = h // shard.tp
    act_bytes = model.act_dtype.bytes
    cp = shard.cp

    for layer_id, (start, end) in graph.layer_index.items():
        # Note: Insert collectives BEFORE sharding metadata.
        # Collectives use model.seq_len directly (not op.meta["s"]),
        # so sharding order doesn't affect communication calculation.
        # Sharding metadata is for subsequent FLOPs calculation.
        for i in range(start, end):
            op = graph.ops[i]

            if op.kind != "attn_core":
                continue

            if shard.cp_kind == CPKind.ULYSSES:
                a2a_bytes = (model.seq_len // cp) * h * act_bytes
                collectives.append(Collective(
                    name=f"a2a_fwd_before_{op.name}",
                    kind="A2A", group="CP",
                    bytes_=a2a_bytes,
                    inserted_before=op.name,
                    phase="fwd",
                ))
                collectives.append(Collective(
                    name=f"a2a_fwd_after_{op.name}",
                    kind="A2A", group="CP",
                    bytes_=a2a_bytes,
                    inserted_after=op.name,
                    phase="fwd",
                ))
                collectives.append(Collective(
                    name=f"a2a_bwd_before_{op.name}",
                    kind="A2A", group="CP",
                    bytes_=a2a_bytes,
                    inserted_before=op.name,
                    phase="bwd",
                ))
                collectives.append(Collective(
                    name=f"a2a_bwd_after_{op.name}",
                    kind="A2A", group="CP",
                    bytes_=a2a_bytes,
                    inserted_after=op.name,
                    phase="bwd",
                ))

            elif shard.cp_kind == CPKind.RING:
                p2p_bytes = (model.seq_len // cp) * h * act_bytes
                # Forward: P2P overlaps with FA tile computation
                collectives.append(Collective(
                    name=f"p2p_ring_fwd_{op.name}",
                    kind="P2P", group="CP",
                    bytes_=p2p_bytes,
                    inserted_before=op.name,
                    rounds=cp,
                    overlap=True,
                    phase="fwd",
                ))
                # Backward: P2P for gradient propagation (similar pattern)
                collectives.append(Collective(
                    name=f"p2p_ring_bwd_{op.name}",
                    kind="P2P", group="CP",
                    bytes_=p2p_bytes,
                    inserted_before=op.name,
                    rounds=cp,
                    overlap=True,
                    phase="bwd",
                ))

            elif shard.cp_kind == CPKind.HYBRID:
                a2a_bytes = (model.seq_len // cp) * h * act_bytes
                collectives.append(Collective(
                    name=f"a2a_fwd_before_{op.name}",
                    kind="A2A", group="CP",
                    bytes_=a2a_bytes,
                    inserted_before=op.name,
                    phase="fwd",
                ))
                p2p_bytes = (model.seq_len // cp) * h * act_bytes
                collectives.append(Collective(
                    name=f"p2p_ring_{op.name}",
                    kind="P2P", group="CP",
                    bytes_=p2p_bytes,
                    inserted_before=op.name,
                    rounds=cp,
                    overlap=True,
                    phase="fwd",
                ))
                collectives.append(Collective(
                    name=f"a2a_fwd_after_{op.name}",
                    kind="A2A", group="CP",
                    bytes_=a2a_bytes,
                    inserted_after=op.name,
                    phase="fwd",
                ))
                collectives.append(Collective(
                    name=f"a2a_bwd_before_{op.name}",
                    kind="A2A", group="CP",
                    bytes_=a2a_bytes,
                    inserted_before=op.name,
                    phase="bwd",
                ))
                collectives.append(Collective(
                    name=f"a2a_bwd_after_{op.name}",
                    kind="A2A", group="CP",
                    bytes_=a2a_bytes,
                    inserted_after=op.name,
                    phase="bwd",
                ))
            
            elif shard.cp_kind == CPKind.COMPRESSED:
                # DeepSeek-V4 two-stage compressed CP
                # Determine layer type (CSA/HCA/SWA) from model spec
                cp_type = model.get_layer_cp_type(layer_id)
                
                # SWA-only layers don't participate in CP communication
                if cp_type == 'swa':
                    continue
                
                # Import compressed CP analyzer
                from zrt.training.models.compressed_cp import (
                    CompressedCPConfig, 
                    CompressedCPCommAnalyzer,
                )
                
                # Configure compressed CP based on layer type
                if cp_type == 'csa':
                    compression_ratio = 4  # CSA: m=4
                elif cp_type == 'hca':
                    compression_ratio = 128  # HCA: m'=128
                else:
                    continue  # Unknown type, skip
                
                # Create analyzer for this layer type
                cp_config = CompressedCPConfig(
                    cp_size=cp,
                    compression_ratio_csa=4 if cp_type == 'csa' else 4,
                    compression_ratio_hca=128 if cp_type == 'hca' else 128,
                    kv_head_dim=model.head_dim,
                )
                analyzer = CompressedCPCommAnalyzer(cp_config)
                
                # Stage 1: P2P boundary exchange (forward + backward)
                if cp_type == 'csa':
                    stage1_bytes = analyzer.stage1_comm_bytes_csa()
                else:  # hca
                    stage1_bytes = analyzer.stage1_comm_bytes_hca()
                
                collectives.append(Collective(
                    name=f"p2p_boundary_fwd_{op.name}_stage1_{cp_type}",
                    kind="P2P", group="CP",
                    bytes_=stage1_bytes,
                    inserted_before=op.name,
                    phase="fwd",
                ))
                collectives.append(Collective(
                    name=f"p2p_boundary_bwd_{op.name}_stage1_{cp_type}",
                    kind="P2P", group="CP",
                    bytes_=stage1_bytes,
                    inserted_before=op.name,
                    phase="bwd",
                ))
                
                # Stage 2: AllGather compressed KV (forward + backward)
                if cp_type == 'csa':
                    stage2_bytes = analyzer.stage2_comm_bytes_csa(model.seq_len)
                else:  # hca
                    stage2_bytes = analyzer.stage2_comm_bytes_hca(model.seq_len)
                
                collectives.append(Collective(
                    name=f"allgather_kv_fwd_{op.name}_stage2_{cp_type}",
                    kind="AG", group="CP",
                    bytes_=stage2_bytes,
                    inserted_before=op.name,
                    phase="fwd",
                ))
                collectives.append(Collective(
                    name=f"allgather_kv_bwd_{op.name}_stage2_{cp_type}",
                    kind="AG", group="CP",
                    bytes_=stage2_bytes,
                    inserted_before=op.name,
                    phase="bwd",
                ))

        _apply_cp_sharding(graph, start, end, shard, model.seq_len, model.hidden)


def _insert_ep_collectives(
    graph: Graph, shard: ShardPlan, model: ModelSpec,
    collectives: list[Collective],
) -> None:
    """Insert EP collectives (A2A pairs for expert parallel).

    EP pattern (MoE only):
      - A2A before routed expert FFN (routes tokens to expert ranks)
      - A2A after routed expert FFN (gathers expert outputs)
    """
    h = model.hidden
    if shard.tp > 1:
        h = h // shard.tp
    seq = model.seq_len
    if shard.cp > 1:
        seq = seq // shard.cp
    moe_ffn = model.moe_ffn if model.moe_ffn > 0 else model.ffn
    act_bytes = model.act_dtype.bytes

    for layer_id, (start, end) in graph.layer_index.items():
        # Check if this is an MoE layer
        if layer_id >= len(model.layers):
            continue
        if model.layers[layer_id].value != "moe":
            continue

        # A2A payload size: tokens × hidden (after TP and CP sharding)
        a2a_bytes = seq * h * act_bytes

        for i in range(start, end):
            op = graph.ops[i]

            # A2A before routed expert FFN
            if op.kind == "matmul" and "routed_expert" in op.name:
                collectives.append(Collective(
                    name=f"a2a_before_{op.name}",
                    kind="A2A", group="EP",
                    bytes_=a2a_bytes,
                    inserted_before=op.name,
                    phase="both",
                ))

            # A2A after routed expert FFN
            if op.kind == "matmul" and "routed_expert" in op.name:
                collectives.append(Collective(
                    name=f"a2a_after_{op.name}",
                    kind="A2A", group="EP",
                    bytes_=a2a_bytes,
                    inserted_after=op.name,
                    phase="both",
                ))

        # Adjust tensor shapes for EP sharding (experts sharded across ranks)
        _apply_ep_sharding(graph, start, end, shard, model.num_experts, model.top_k)


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
        elif op.kind == "attn_core":
            if "heads" in op.meta:
                heads_before_tp = op.meta["heads"]
                op.meta["heads"] = max(1, op.meta["heads"] // shard.tp)
                op.meta["heads_tp"] = op.meta["heads"]
                op.meta["heads_before_tp"] = heads_before_tp
            if "h_kv" in op.meta:
                op.meta["h_kv"] = max(1, op.meta["h_kv"] // shard.tp)
            for t in op.inputs + op.outputs:
                if t.shape_logical and t.shape_logical[-1] in (h_attn, h_kv):
                    t.shape_local = (t.shape_logical[0], max(1, t.shape_logical[-1] // shard.tp))
        elif op.kind in ("mhc_pre", "mhc_post", "mhc_head", "hc_expand"):
            # Hyper-Connections are token-local: TP does not shard hc or h.
            # The mixes-Linear operates on (hc·h) features that semantically
            # belong to *one* residual stream replicated across hc copies; the
            # learned hc_*_fn matrix is small (hc·h × mix_hc) and replicated on
            # every TP rank, so introducing TP here would only add comm without
            # reducing compute.  CP (sequence-parallel) handling — when added —
            # would scale the seq dim of meta["s"] independently.
            pass
        elif op.kind in ("ln", "rope", "swiglu", "add"):
            if "bytes_fwd" in op.meta:
                op.meta["bytes_fwd"] = int(op.meta["bytes_fwd"]) // shard.tp
            for t in op.inputs + op.outputs:
                if t.shape_logical and t.shape_logical[-1] in (h, h_attn, h_kv, ffn):
                    t.shape_local = (t.shape_logical[0], max(1, t.shape_logical[-1] // shard.tp))


def _apply_cp_sharding(
    graph: Graph, start: int, end: int, shard: ShardPlan, seq: int, hidden: int,
) -> None:
    """Adjust tensor shape_local for CP sharding.

    Ulysses-CP: sequence split by cp, heads multiplied by cp for attention.
    Ring-CP: sequence split by cp, heads unchanged (KV chunks sent round-robin).
    Hybrid-CP: both sequence and heads adjustments apply.

    For attention core:
      - Ulysses: heads_local = heads_tp * cp (heads gathered by A2A)
      - Ring: heads_local = heads_tp (unchanged)
      - seq_local = seq / cp for all CP kinds
    """
    if not shard.has_cp:
        return

    seq_local = seq // shard.cp

    for i in range(start, end):
        op = graph.ops[i]

        for t in op.inputs + op.outputs:
            if t.shape_logical and len(t.shape_logical) > 0:
                if t.shape_logical[0] == seq:
                    t.shape_local = (seq_local,) + t.shape_logical[1:]

        if op.kind == "matmul":
            if "m" in op.meta:
                op.meta["m"] = op.meta["m"] // shard.cp
        elif op.kind == "attn_core":
            if "s" in op.meta:
                op.meta["s"] = op.meta["s"] // shard.cp
            if shard.cp_kind == CPKind.ULYSSES or shard.cp_kind == CPKind.HYBRID:
                if "heads" in op.meta:
                    heads_tp = op.meta.get("heads_tp", op.meta["heads"])
                    op.meta["heads"] = heads_tp * shard.cp
                    op.meta["heads_gathered_by_cp"] = True
            if shard.cp_kind == CPKind.RING:
                op.meta["heads_gathered_by_cp"] = False
                op.meta["cp_tiles"] = shard.cp
        elif op.kind in ("ln", "rope", "swiglu", "add"):
            if "bytes_fwd" in op.meta:
                op.meta["bytes_fwd"] = int(op.meta["bytes_fwd"]) // shard.cp


def _apply_ep_sharding(
    graph: Graph, start: int, end: int, shard: ShardPlan,
    num_experts: int, top_k: int,
) -> None:
    """Adjust tensor shape_local for EP sharding (expert parallel).

    For EP, each rank handles num_experts/ep experts.
    The routed expert FFN only processes a subset of experts.
    """
    if not shard.has_ep:
        return

    experts_per_rank = num_experts // shard.ep

    for i in range(start, end):
        op = graph.ops[i]

        # Only routed expert FFN is affected by EP sharding
        if op.kind == "matmul" and "routed_expert" in op.name:
            # Update num_experts in metadata
            if "num_experts" in op.meta:
                op.meta["num_experts_local"] = experts_per_rank
            # FLOPs scale by local expert count
            # (top_k experts per token, but only experts_per_rank available locally)
            if "num_experts" in op.meta:
                original_experts = op.meta["num_experts"]
                # Effective FLOPs scale ratio: (experts_per_rank / original_experts)
                # But we keep top_k the same since each token still goes to top_k experts
                pass  # FLOPs scaling handled in flops.py

        # Router output: num_experts -> experts_per_rank
        if op.kind == "matmul" and "router" in op.name:
            for t in op.outputs:
                if len(t.shape_logical) > 1:
                    # Router output: (seq, num_experts) -> (seq, experts_per_rank)
                    if t.shape_logical[1] == num_experts:
                        t.shape_local = (t.shape_logical[0], experts_per_rank)
