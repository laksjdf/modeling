"""Search space definition for parallel configuration grid search."""
from __future__ import annotations

from dataclasses import dataclass, field

from zrt.training.spec.strategy import (
    CPKind, OffloadPolicy, OptKind, PPSched, RecomputePolicy, TPOverlap,
    rank_product,
)


@dataclass
class SearchSpace:
    """Defines the dimensions and constraints for parallel config search.

    Feature flags for pruning (phase-3 adaptation points):
      - enable_cross_node_tp_pruning: Skip TP > gpus_per_node (requires NVLink topology)
      - enable_cp_pruning: Only enable CP when seq_len >= threshold
      - enable_ep_pruning: Only enable EP when num_experts > threshold
      - cp_seq_len_threshold: Minimum seq_len for CP to be beneficial (default: 32768)

    TODO Phase 3: These pruning rules depend on CP/EP implementation status:
      - CP requires correct communication cost model (Ring vs Ulysses)
      - EP requires expert dispatch/all-to-all semantics
      - Cross-node TP requires topology awareness
      - Current defaults are conservative; adjust based on phase 3 findings
    """

    tp_values: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    cp_values: list[int] = field(default_factory=lambda: [1])
    pp_values: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    ep_values: list[int] = field(default_factory=lambda: [1, 8])
    dp_values: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    zero_stages: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    pp_schedules: list[PPSched] = field(default_factory=lambda: [PPSched.ONE_F_ONE_B, PPSched.INTERLEAVED, PPSched.DUALPIPE])
    recompute_policies: list[str] = field(default_factory=lambda: ["none", "selective", "full"])
    vpp_chunks_values: list[int] = field(default_factory=lambda: [1, 2, 4])
    optimizer_values: list[OptKind] = field(default_factory=lambda: [OptKind.ADAM, OptKind.MUON])
    muon_rotation_values: list[bool] = field(default_factory=lambda: [True, False])

    micro_batch: int = 1
    global_batch: int = 0
    max_memory_gb: float = 80.0

    # Pruning feature flags (phase-3 adaptation points)
    enable_cross_node_tp_pruning: bool = False
    enable_cp_pruning: bool = False
    enable_ep_pruning: bool = True
    cp_seq_len_threshold: int = 32768

    def strategies(self, world_size: int) -> list["Strategy"]:
        """Generate all valid Strategy instances for the given world_size."""
        from zrt.training.spec.strategy import Strategy

        results = []
        seen = set()

        for tp in self.tp_values:
            for cp in self.cp_values:
                for pp in self.pp_values:
                    for ep in self.ep_values:
                        total = tp * cp * pp * ep
                        if world_size % total != 0:
                            continue
                        dp = world_size // total
                        if dp not in self.dp_values and dp > 0:
                            continue

                        for zero_stage in self.zero_stages:
                            if zero_stage >= 1 and dp <= 1:
                                continue

                            for sched in self.pp_schedules:
                                for rc in self.recompute_policies:
                                    for opt in self.optimizer_values:
                                        if opt == OptKind.MUON:
                                            for rot in self.muon_rotation_values:
                                                if sched == PPSched.INTERLEAVED:
                                                    for vc in self.vpp_chunks_values:
                                                        s = self._make_strategy(
                                                            tp, cp, pp, ep, dp, zero_stage,
                                                            sched, rc, vc, opt, rot,
                                                        )
                                                        key = (tp, cp, pp, ep, dp, zero_stage, sched, rc, vc, opt, rot)
                                                        if key not in seen:
                                                            seen.add(key)
                                                            results.append(s)
                                                else:
                                                    s = self._make_strategy(
                                                        tp, cp, pp, ep, dp, zero_stage,
                                                        sched, rc, 1, opt, rot,
                                                    )
                                                    key = (tp, cp, pp, ep, dp, zero_stage, sched, rc, 1, opt, rot)
                                                    if key not in seen:
                                                        seen.add(key)
                                                        results.append(s)
                                        else:
                                            if sched == PPSched.INTERLEAVED:
                                                for vc in self.vpp_chunks_values:
                                                    s = self._make_strategy(
                                                        tp, cp, pp, ep, dp, zero_stage,
                                                        sched, rc, vc, opt, True,
                                                    )
                                                    key = (tp, cp, pp, ep, dp, zero_stage, sched, rc, vc, opt, True)
                                                    if key not in seen:
                                                        seen.add(key)
                                                        results.append(s)
                                            else:
                                                s = self._make_strategy(
                                                    tp, cp, pp, ep, dp, zero_stage,
                                                    sched, rc, 1, opt, True,
                                                )
                                                key = (tp, cp, pp, ep, dp, zero_stage, sched, rc, 1, opt, True)
                                                if key not in seen:
                                                    seen.add(key)
                                                    results.append(s)

        return results

    def _make_strategy(self, tp, cp, pp, ep, dp, zero_stage, sched, rc, vpp_chunks, opt, rotation):
        from zrt.training.spec.strategy import Strategy, RecomputePolicy, MuonConfig

        rc_policy = RecomputePolicy()
        if rc == "selective":
            rc_policy.per_layer = {"moe": {"attn"}, "dense": {"attn"}}
        elif rc == "full":
            rc_policy.per_layer = {"moe": {"full"}, "dense": {"full"}}

        muon_config = None
        if opt == OptKind.MUON:
            muon_config = MuonConfig(rotation=rotation)

        return Strategy(
            tp=tp, cp=cp, pp=pp, ep=ep, dp=dp,
            micro_batch=self.micro_batch,
            global_batch=self.global_batch,
            zero_stage=zero_stage,
            pp_schedule=sched,
            vpp_chunks=vpp_chunks,
            recompute=rc_policy,
            optimizer=opt,
            muon_config=muon_config,
        )
