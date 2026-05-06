"""IR validation — check divisibility, balance, and placement constraints."""

from __future__ import annotations

from zrt.training.spec.model import ModelSpec
from zrt.training.spec.strategy import CPKind, Strategy
from zrt.training.spec.system import SystemSpec


def validate(model: ModelSpec, system: SystemSpec, strategy: Strategy) -> list[str]:
    """Return list of warning/error strings. Empty list means valid."""
    warnings: list[str] = []

    n_layers = len(model.layers)

    # PP balance
    if strategy.pp > 1:
        if strategy.pp_layer_assignment is not None:
            if len(strategy.pp_layer_assignment) != n_layers:
                warnings.append(
                    f"pp_layer_assignment length ({len(strategy.pp_layer_assignment)}) "
                    f"!= num_layers ({n_layers})"
                )
        else:
            # Auto-balance: check if even distribution is possible
            if n_layers % strategy.pp != 0:
                warnings.append(
                    f"num_layers ({n_layers}) not evenly divisible by PP ({strategy.pp}); "
                    f"stages will be imbalanced"
                )

    # CP constraints
    if strategy.cp > 1:
        if strategy.cp_kind == CPKind.ULYSSES:
            if model.num_heads % strategy.cp != 0:
                warnings.append(
                    f"Ulysses CP requires num_heads % cp == 0, "
                    f"got {model.num_heads} % {strategy.cp}"
                )
        if strategy.cp_kind == CPKind.RING:
            block_size = 128  # typical flash-attn block size
            if model.seq_len % (strategy.cp * block_size) != 0:
                warnings.append(
                    f"Ring CP requires seq_len % (cp * block_size) == 0, "
                    f"got {model.seq_len} % ({strategy.cp} * {block_size})"
                )
        if strategy.cp_kind == CPKind.COMPRESSED:
            # DeepSeek-V4 compressed CP: requires seq_len divisible by cp
            # Compression ratios are fixed (CSA=4, HCA=128) or model-specific
            if model.seq_len % strategy.cp != 0:
                warnings.append(
                    f"Compressed CP requires seq_len % cp == 0, "
                    f"got {model.seq_len} % {strategy.cp}"
                )
            # Additional check: compression ratio should be compatible
            if hasattr(model, "attn_compression_ratio") and model.attn_compression_ratio < 1.0:
                # Model uses compressed attention (DeepSeek-V4 style)
                # No additional constraints beyond seq_len divisible by cp
                pass

    # EP × DP placement: EP crossing node boundaries is expensive
    if strategy.ep > 1 and strategy.ep > system.gpus_per_node:
        warnings.append(
            f"EP ({strategy.ep}) > gpus_per_node ({system.gpus_per_node}); "
            f"EP A2A will cross node boundaries (inter-node bandwidth)"
        )

    # TP across nodes: usually very bad
    if strategy.tp > system.gpus_per_node:
        warnings.append(
            f"TP ({strategy.tp}) > gpus_per_node ({system.gpus_per_node}); "
            f"TP communication will be inter-node (severe performance penalty)"
        )

    # VPP without interleaved schedule
    if strategy.vpp_chunks > 1 and strategy.pp_schedule.value != "i1f1b":
        warnings.append(
            f"vpp_chunks ({strategy.vpp_chunks}) > 1 but schedule is "
            f"{strategy.pp_schedule.value}; VPP requires i1f1b schedule"
        )

    return warnings
