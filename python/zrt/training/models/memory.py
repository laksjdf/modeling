"""Memory model — weights, grads, optimizer state, activations.

Reference: ZeRO (Rajbhandari et al. 2020), Korthikanti et al. 2022,
DeepSeek Memory Analysis (Yang et al. 2025).
"""

from __future__ import annotations

from dataclasses import dataclass

from zrt.training.ir.graph import Graph, Op
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import SystemSpec


@dataclass
class MemBreakdown:
    weights: float = 0.0       # bytes
    grads: float = 0.0         # bytes
    opt_state: float = 0.0     # bytes
    activations: float = 0.0   # bytes
    comm_buffers: float = 0.0  # bytes

    @property
    def total(self) -> float:
        return self.weights + self.grads + self.opt_state + self.activations + self.comm_buffers

    def to_gb(self) -> dict[str, float]:
        GB = 1024 ** 3
        return {
            "weights_gb": self.weights / GB,
            "grads_gb": self.grads / GB,
            "opt_state_gb": self.opt_state / GB,
            "activations_gb": self.activations / GB,
            "comm_buffers_gb": self.comm_buffers / GB,
            "total_gb": self.total / GB,
        }


def memory_breakdown(
    graph: Graph,
    model: ModelSpec,
    system: SystemSpec,
    strategy: Strategy,
    stage_layer_ids: list[int] | None = None,
) -> MemBreakdown:
    """Compute per-rank memory breakdown in bytes.

    Accounts for TP/PP sharding, ZeRO stages, and activation memory.
    """
    # ── Parameters on this rank ──────────────────────────────────────────
    P = _params_on_rank(model, strategy)

    weights = P * model.param_dtype.bytes
    grads = P * model.grad_dtype.bytes
    opt_state = _optimizer_state_bytes(P, model, strategy)

    # ZeRO sharding
    dp = strategy.dp
    if strategy.zero_stage >= 3:
        weights //= dp
        grads //= dp
        opt_state //= dp
    elif strategy.zero_stage >= 2:
        grads //= dp
        opt_state //= dp
    elif strategy.zero_stage >= 1:
        opt_state //= dp

    # ── Activations ──────────────────────────────────────────────────────
    if stage_layer_ids is not None:
        layer_ids = stage_layer_ids
    else:
        layer_ids = list(range(len(model.layers)))

    activations = _activation_memory(model, strategy, layer_ids)

    # ── Communication buffers ────────────────────────────────────────────
    comm_buffers = _comm_buffer_memory(model, strategy)

    # ── Offload ──────────────────────────────────────────────────────────
    off = strategy.offload
    if off.pct > 0:
        if off.opt_state:
            opt_state = int(opt_state * (1 - off.pct))
        if off.grads:
            grads = int(grads * (1 - off.pct))
        if off.params:
            weights = int(weights * (1 - off.pct))

    return MemBreakdown(
        weights=weights,
        grads=grads,
        opt_state=opt_state,
        activations=activations,
        comm_buffers=comm_buffers,
    )


def _params_on_rank(model: ModelSpec, strategy: Strategy) -> int:
    """Total parameters held on one rank after TP + PP sharding."""
    total = model.total_params()

    # TP: column-parallel and row-parallel shard params by TP
    # (roughly half the params in each layer are col-parallel, half row-parallel)
    if strategy.tp > 1:
        total //= strategy.tp

    # PP: only hold params for layers on this stage
    if strategy.pp > 1:
        n_layers = len(model.layers)
        layers_per_stage = n_layers / strategy.pp
        # Proportional: non-embedding params scale with layers
        embed_params = model.vocab * model.hidden * 2  # embed + lm_head
        non_embed = total - embed_params
        non_embed = int(non_embed * layers_per_stage / n_layers)
        total = non_embed + embed_params // strategy.pp

    return total


def _optimizer_state_bytes(P: int, model: ModelSpec, strategy: Strategy) -> int:
    """Optimizer state memory in bytes for P parameters."""
    master_bytes = model.master_dtype.bytes
    if strategy.optimizer.value == "adam":
        # Adam: master copy + momentum (m) + variance (v) = 3 × P × master_dtype
        return P * master_bytes * 3
    elif strategy.optimizer.value == "muon":
        # Muon: master copy + momentum matrix = 2 × P × master_dtype
        # Plus Newton-Schulz scratch (small, ~P * master_bytes * 0.1)
        return int(P * master_bytes * 2.1)
    return P * master_bytes * 3  # default: adam


def _activation_memory(
    model: ModelSpec, strategy: Strategy, layer_ids: list[int],
) -> int:
    """Activation memory per rank using Korthikanti-style estimation.

    Per layer: seq * hidden * dtype_bytes * coefficient(layer_kind)
    Coefficient accounts for number of activation tensors held simultaneously.
    """
    s = model.seq_len
    h = model.hidden
    act_bytes = model.act_dtype.bytes

    # Coefficient per layer kind (number of activation tensors that must be
    # materialized simultaneously for backward, roughly)
    # Dense: ~10 tensors (x, x_ln, q, k, v, attn_out, x_attn, x_ln2, up, gate, swiglu)
    COEFF_DENSE = 10
    COEFF_MOE = 14  # additional dispatch/combine tensors
    COEFF_MTP = 12

    total_act = 0
    for lid in layer_ids:
        if lid >= len(model.layers):
            continue
        lk = model.layers[lid]
        if lk.value == "dense":
            coeff = COEFF_DENSE
        elif lk.value == "moe":
            coeff = COEFF_MOE
        elif lk.value == "mtp":
            coeff = COEFF_MTP
        else:
            coeff = COEFF_DENSE

        # Base: seq * hidden * dtype_bytes * coeff
        layer_act = s * h * act_bytes * coeff

        # TP with SP reduces activation by factor of tp for seq-sharded portion
        if strategy.tp > 1:
            layer_act //= strategy.tp

        # CP reduces further
        if strategy.cp > 1:
            layer_act //= strategy.cp

        total_act += layer_act

    # Scale by microbatch
    total_act *= strategy.micro_batch

    # Scale by in-flight microbatches (PP bubble depth)
    if strategy.pp > 1:
        # 1F1B: worst case is first stage holding pp-1 microbatches
        # Average over training step: use (pp) // 2 as approximation
        in_flight = max(1, strategy.pp // 2)
        total_act *= in_flight

    return total_act


def _comm_buffer_memory(model: ModelSpec, strategy: Strategy) -> int:
    """Communication buffer memory (AG/RS buffers)."""
    if strategy.tp <= 1:
        return 0

    s = model.seq_len
    h = model.hidden
    act_bytes = model.act_dtype.bytes

    # Per layer: 2 AG buffers + 2 RS buffers (attn + FFN), each = seq * hidden * dtype
    # AG buffer: full tensor before sharding
    # RS buffer: full tensor after reduction
    per_layer = 4 * s * h * act_bytes

    n_layers = len(model.layers)
    if strategy.pp > 1:
        layers_per_stage = n_layers // strategy.pp
        n_layers = layers_per_stage

    return per_layer * n_layers * strategy.micro_batch
