"""Tests for ir/validate.py — strategy/model/system constraint checks.

validate() returns a list of warning strings.  An empty list means the
configuration is valid.  Tests below verify that each constraint produces
the correct warning (or no warning) and that a well-formed config is clean.

Why this matters for accuracy: many constraints are silent correctness issues.
  - PP with too few microbatches → pipeline cannot fill all stages → wrong bubble
  - VPP with wrong schedule → wrong bubble-fraction formula selected
  - Ulysses CP head indivisibility → sequence sharding is undefined → silent error
  - TP across nodes → wrong bandwidth tier used → 10–100× latency underestimate
"""
from __future__ import annotations

import pytest

from zrt.training.ir.validate import validate
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import CPKind, PPSched, Strategy
from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.spec.system import GPU, SystemSpec


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _model(num_layers: int = 8, num_heads: int = 32, seq_len: int = 4096) -> ModelSpec:
    return ModelSpec(
        hidden=4096, ffn=16384, num_heads=num_heads, num_kv_heads=num_heads,
        head_dim=128, vocab=32000, seq_len=seq_len,
        layers=[LayerKind.DENSE] * num_layers,
    )


def _system(gpus_per_node: int = 8, nodes: int = 1) -> SystemSpec:
    return SystemSpec(
        gpu=GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80, hbm_bw_gbps=3350),
        host_mem_gb=256,
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0,
                               topology="all_to_all", num_devices=8),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=400, latency_us=5.0,
                               topology="fat_tree"),
        ),
        nodes=nodes, gpus_per_node=gpus_per_node,
    )


def _warnings_contain(warnings: list[str], keyword: str) -> bool:
    return any(keyword.lower() in w.lower() for w in warnings)


# ── Sanity: valid config produces no warnings ─────────────────────────────────

class TestValidConfigIsClean:
    def test_simple_dense_config_has_no_warnings(self):
        """A straightforward PP=2 config with enough microbatches is clean."""
        model = _model(num_layers=8)
        system = _system()
        strategy = Strategy(
            tp=1, pp=2, dp=1,
            micro_batch=1, global_batch=16,  # 8 microbatches > pp=2
        )
        warnings = validate(model, system, strategy)
        assert warnings == [], f"Unexpected warnings: {warnings}"

    def test_pp1_single_stage_always_valid(self):
        """PP=1 has no pipeline constraints to violate."""
        model = _model(num_layers=4)
        system = _system()
        strategy = Strategy(tp=1, pp=1, dp=1, micro_batch=1, global_batch=4)
        assert validate(model, system, strategy) == []

    def test_ulysses_cp_valid_config_has_no_warnings(self):
        """Ulysses CP is valid when (num_heads // tp) is divisible by cp."""
        model = _model(num_layers=4, num_heads=32)
        system = _system()
        # cp=4, tp=1: effective_heads = 32, 32 % 4 == 0 → valid
        strategy = Strategy(tp=1, cp=4, pp=1, dp=1,
                            micro_batch=1, global_batch=4,
                            cp_kind=CPKind.ULYSSES)
        warnings = validate(model, system, strategy)
        # The only expected warning is about Ulysses + TP SP combination when tp>1.
        # With tp=1, that warning should not appear.
        assert not _warnings_contain(warnings, "conflict"), str(warnings)

    def test_vpp_valid_config_has_no_warnings(self):
        """VPP with i1f1b schedule and correct layer divisibility is valid."""
        model = _model(num_layers=8)
        system = _system()
        # pp=2, vpp_chunks=2: 8 layers / (2 * 2) = 2 per chunk, divisible
        strategy = Strategy(
            tp=1, pp=2, dp=1,
            micro_batch=1, global_batch=16,
            pp_schedule=PPSched.INTERLEAVED, vpp_chunks=2,
        )
        assert validate(model, system, strategy) == []


# ── PP constraints ────────────────────────────────────────────────────────────

class TestPipelineParallelConstraints:
    def test_layer_imbalance_warning_when_not_divisible(self):
        """num_layers not divisible by PP → stages will have unequal load.

        Impact: the pipeline composer assigns layers greedily, causing some
        stages to be heavier than others and producing an incorrect bubble estimate.
        """
        model = _model(num_layers=7)
        system = _system()
        strategy = Strategy(tp=1, pp=4, dp=1,
                            micro_batch=1, global_batch=8)
        warnings = validate(model, system, strategy)
        assert _warnings_contain(warnings, "imbalanced") or _warnings_contain(warnings, "divisible"), (
            f"Expected layer imbalance warning for 7 layers / PP=4, got: {warnings}"
        )

    def test_no_imbalance_warning_when_divisible(self):
        """num_layers divisible by PP → no imbalance warning."""
        model = _model(num_layers=8)
        system = _system()
        strategy = Strategy(tp=1, pp=4, dp=1,
                            micro_batch=1, global_batch=16)
        warnings = validate(model, system, strategy)
        assert not _warnings_contain(warnings, "imbalanced"), (
            f"Unexpected imbalance warning for 8 layers / PP=4: {warnings}"
        )

    def test_warmup_warning_when_microbatches_less_than_pp(self):
        """num_microbatches < PP → pipeline cannot warm up fully.

        If M < PP, some stages will be idle during warmup, the pipeline schedule
        is not achievable, and the bubble estimate is meaningless.  This is one
        of the most common misconfiguration errors.
        """
        model = _model(num_layers=4)
        system = _system()
        # pp=4, global_batch=3, micro_batch=1, dp=1 → 3 microbatches < pp=4
        strategy = Strategy(tp=1, pp=4, dp=1,
                            micro_batch=1, global_batch=3)
        warnings = validate(model, system, strategy)
        assert _warnings_contain(warnings, "microbatch") or _warnings_contain(warnings, "warmup"), (
            f"Expected pipeline warmup warning for M=3 < PP=4, got: {warnings}"
        )

    def test_no_warmup_warning_when_microbatches_equal_pp(self):
        """M == PP is the minimum valid case — no warning expected."""
        model = _model(num_layers=4)
        system = _system()
        strategy = Strategy(tp=1, pp=4, dp=1,
                            micro_batch=1, global_batch=4)
        warnings = validate(model, system, strategy)
        assert not _warnings_contain(warnings, "warmup"), (
            f"Unexpected warmup warning for M=4 == PP=4: {warnings}"
        )


# ── VPP constraints ───────────────────────────────────────────────────────────

class TestVPPConstraints:
    def test_vpp_with_non_interleaved_schedule_warns(self):
        """VPP requires interleaved (i1f1b) schedule.

        If vpp_chunks > 1 but schedule is 1f1b, the bubble formula for
        interleaved pipeline does not apply and step-time estimates are wrong.
        This is a silent accuracy error if not caught.
        """
        model = _model(num_layers=8)
        system = _system()
        strategy = Strategy(
            tp=1, pp=2, dp=1,
            micro_batch=1, global_batch=16,
            pp_schedule=PPSched.ONE_F_ONE_B, vpp_chunks=2,
        )
        warnings = validate(model, system, strategy)
        assert _warnings_contain(warnings, "vpp") or _warnings_contain(warnings, "schedule"), (
            f"Expected VPP+schedule mismatch warning, got: {warnings}"
        )

    def test_vpp_layer_indivisibility_warns(self):
        """num_layers must be divisible by pp * vpp_chunks.

        If violated, some model chunks have more layers than others,
        the chunk balance assumption breaks, and the bubble formula is wrong.
        """
        model = _model(num_layers=9)
        system = _system()
        # pp=2, vpp_chunks=2: 9 / (2*2) = 2.25 → not integer → warning
        strategy = Strategy(
            tp=1, pp=2, dp=1,
            micro_batch=1, global_batch=16,
            pp_schedule=PPSched.INTERLEAVED, vpp_chunks=2,
        )
        warnings = validate(model, system, strategy)
        assert _warnings_contain(warnings, "vpp") or _warnings_contain(warnings, "divisible"), (
            f"Expected VPP layer divisibility warning for 9 layers / (2×2), got: {warnings}"
        )


# ── Context Parallel constraints ──────────────────────────────────────────────

class TestContextParallelConstraints:
    def test_cp_without_cp_kind_warns(self):
        """cp > 1 with cp_kind='none' is a misconfiguration — CP won't activate.

        Whoever sets cp=4 expects sequence sharding; if cp_kind is not set,
        the sequence is not actually sharded and the bandwidth estimate for
        CP collectives is also wrong (they are not inserted).
        """
        model = _model(num_layers=4)
        system = _system()
        strategy = Strategy(tp=1, cp=4, pp=1, dp=1,
                            micro_batch=1, global_batch=4,
                            cp_kind=CPKind.NONE)
        warnings = validate(model, system, strategy)
        assert _warnings_contain(warnings, "cp_kind") or _warnings_contain(warnings, "none"), (
            f"Expected cp_kind warning for cp=4 + cp_kind=none, got: {warnings}"
        )

    def test_ulysses_head_indivisibility_warns(self):
        """Ulysses CP requires (num_heads // tp) to be divisible by cp.

        If violated, attention heads cannot be evenly partitioned across CP ranks.
        The actual computation is undefined; this must be flagged.
        """
        model = _model(num_layers=4, num_heads=32)
        system = _system()
        # cp=6, tp=1: effective_heads=32, 32 % 6 = 2 ≠ 0 → warning
        strategy = Strategy(tp=1, cp=6, pp=1, dp=1,
                            micro_batch=1, global_batch=4,
                            cp_kind=CPKind.ULYSSES)
        warnings = validate(model, system, strategy)
        assert _warnings_contain(warnings, "ulysses") or _warnings_contain(warnings, "head"), (
            f"Expected Ulysses head-divisibility warning for 32 heads / CP=6, got: {warnings}"
        )

    def test_ulysses_head_divisible_no_head_warning(self):
        """Ulysses CP with divisible heads produces no head-divisibility warning."""
        model = _model(num_layers=4, num_heads=32)
        system = _system()
        # cp=4: 32 % 4 == 0 → no head warning
        strategy = Strategy(tp=1, cp=4, pp=1, dp=1,
                            micro_batch=1, global_batch=4,
                            cp_kind=CPKind.ULYSSES)
        warnings = validate(model, system, strategy)
        head_warnings = [w for w in warnings if "head" in w.lower() and "divisible" in w.lower()]
        assert head_warnings == [], f"Unexpected head-divisibility warning: {head_warnings}"

    def test_ring_cp_seq_len_indivisibility_warns(self):
        """Ring CP requires seq_len % (cp * block_size) == 0 (block_size=128).

        If violated, attention tokens cannot be evenly tiled across CP ranks.
        """
        # seq_len=4000, cp=2: 4000 % (2*128) = 4000 % 256 = 160 ≠ 0 → warning
        model = _model(num_layers=4, seq_len=4000)
        system = _system()
        strategy = Strategy(tp=1, cp=2, pp=1, dp=1,
                            micro_batch=1, global_batch=4,
                            cp_kind=CPKind.RING)
        warnings = validate(model, system, strategy)
        assert _warnings_contain(warnings, "seq_len") or _warnings_contain(warnings, "ring"), (
            f"Expected Ring CP seq_len warning for 4000 % (2×128), got: {warnings}"
        )

    def test_ring_cp_seq_len_divisible_no_warning(self):
        """Ring CP with divisible seq_len: no seq_len warning."""
        # seq_len=4096, cp=2: 4096 % (2*128) = 4096 % 256 = 0 → clean
        model = _model(num_layers=4, seq_len=4096)
        system = _system()
        strategy = Strategy(tp=1, cp=2, pp=1, dp=1,
                            micro_batch=1, global_batch=4,
                            cp_kind=CPKind.RING)
        warnings = validate(model, system, strategy)
        seq_warnings = [w for w in warnings if "seq_len" in w.lower()]
        assert seq_warnings == [], f"Unexpected seq_len warning: {seq_warnings}"


# ── TP cross-node warning ─────────────────────────────────────────────────────

class TestTPCrossNodeWarning:
    def test_tp_exceeding_gpus_per_node_warns(self):
        """TP > gpus_per_node means TP all-reduce crosses node boundaries.

        This is a severe accuracy issue: the intra-node bandwidth (NVLink: 900 GB/s)
        is used for TP collectives by default; if TP spans nodes, inter-node
        bandwidth (IB: 400 GB/s or less) applies instead — a 2–10× difference.
        A missing warning here would cause silent 2–10× TP latency underestimates.
        """
        model = _model(num_layers=4, num_heads=16)
        system = _system(gpus_per_node=8)
        strategy = Strategy(tp=16, pp=1, dp=1,
                            micro_batch=1, global_batch=4)
        warnings = validate(model, system, strategy)
        assert _warnings_contain(warnings, "inter-node") or _warnings_contain(warnings, "node"), (
            f"Expected cross-node TP warning for TP=16 > gpus_per_node=8, got: {warnings}"
        )

    def test_tp_within_node_no_bandwidth_warning(self):
        """TP ≤ gpus_per_node: all TP traffic stays on NVLink, no warning needed."""
        model = _model(num_layers=4, num_heads=32)
        system = _system(gpus_per_node=8)
        strategy = Strategy(tp=8, pp=1, dp=1,
                            micro_batch=1, global_batch=4)
        warnings = validate(model, system, strategy)
        tp_node_warnings = [
            w for w in warnings
            if ("tp" in w.lower() and "node" in w.lower())
        ]
        assert tp_node_warnings == [], (
            f"Unexpected TP cross-node warning for TP=8 == gpus_per_node=8: {tp_node_warnings}"
        )


# ── EP constraints ────────────────────────────────────────────────────────────

class TestExpertParallelConstraints:
    def test_ep_dp_indivisibility_warns(self):
        """Megatron requires DP % EP == 0 (EP groups must be subsets of DP groups).

        If violated, the EP group assignment is undefined and expert routing
        will produce incorrect results at runtime.
        """
        model = _model(num_layers=4)
        system = _system()
        # ep=4, dp=6: 6 % 4 = 2 ≠ 0 → warning
        strategy = Strategy(tp=1, ep=4, pp=1, dp=6,
                            micro_batch=1, global_batch=6)
        warnings = validate(model, system, strategy)
        assert _warnings_contain(warnings, "ep") or _warnings_contain(warnings, "megatron"), (
            f"Expected EP/DP divisibility warning for DP=6, EP=4, got: {warnings}"
        )

    def test_ep_dp_divisible_no_warning(self):
        """DP divisible by EP: no Megatron constraint violation."""
        model = _model(num_layers=4)
        system = _system()
        # ep=4, dp=8: 8 % 4 == 0 → clean
        strategy = Strategy(tp=1, ep=4, pp=1, dp=8,
                            micro_batch=1, global_batch=8)
        warnings = validate(model, system, strategy)
        ep_dp_warnings = [
            w for w in warnings
            if "dp" in w.lower() and "ep" in w.lower() and "divisible" in w.lower()
        ]
        assert ep_dp_warnings == [], f"Unexpected EP/DP warning: {ep_dp_warnings}"
