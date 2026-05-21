"""Phase-1 bugfix regression tests.

Bug 1.1: 1F1B step-time formula
Bug 1.2: Activation memory (Korthikanti + recompute + ZeRO metadata)
Bug 1.3: TrainingFlopsPass per-node annotation priority
Bug 3: FlopsPass gates _calculate_grad_flops on node phase
Bug 4: TrainingFlopsPass phase-aware aggregation on stitched graphs
"""
from __future__ import annotations

import pytest

from python.zrt.ir.edge import Edge
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.transform.analysis.passes import FlopsPass, RooflinePass
from python.zrt.transform.analysis.training import (
    TrainingFlopsPass,
    TrainingMemoryPass,
    TrainingPipelinePass,
)
from python.zrt.transform.context import (
    ParallelConfig,
    TrainingConfig,
    TransformContext,
)
from python.zrt.transform.training.recompute import (
    has_internal_recompute,
    is_external_recompute_node,
)


def _make_graph(nodes=None, metadata=None):
    g = OpGraph(name="test", phase="prefill", metadata=metadata or {})
    if nodes:
        for n in nodes:
            g.nodes[n.id] = n
    return g


def _make_ctx(tp=1, pp=1, dp=1, cp=1, zero_stage=0, micro_batch=1,
              global_batch=32, recompute_policy="none", optimizer="adam"):
    from python.zrt.hardware.spec import (
        ComputeSpec,
        HardwareSpec,
        InterconnectSpec,
        LinkSpec,
        MemorySpec,
    )
    hw = HardwareSpec(
        name="test_h100",
        vendor="nvidia",
        device_type="gpu",
        compute=ComputeSpec(bf16_tflops=989, fp8_tops=1979),
        memory=MemorySpec(capacity_gb=80, hbm_bandwidth_gbps=3350),
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=100, latency_us=10.0),
        ),
    )
    return TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=tp, pp=pp, dp=dp, cp=cp),
        training=TrainingConfig(
            optimizer=optimizer,
            zero_stage=zero_stage,
            micro_batch=micro_batch,
            global_batch=global_batch,
            recompute_policy=recompute_policy,
        ),
    )


# ── Bug 1.3: TrainingFlopsPass ────────────────────────────────────────────────

def test_training_flops_uses_per_node_annotations():
    n1 = OpNode(id="mm", op_type="aten.mm.default",
                annotations={"flops_fwd": 100, "flops_dx": 50, "flops_dw": 50})
    n2 = OpNode(id="comm", op_type="comm.all_reduce",
                annotations={"flops_fwd": 0, "flops_dx": 0, "flops_dw": 0})
    g = _make_graph([n1, n2], {"num_layers": 4, "num_layers_traced": 4})
    ctx = _make_ctx(micro_batch=1, global_batch=4)

    result = TrainingFlopsPass().run(g, ctx)
    assert result.metadata["forward_flops"] == 100
    assert result.metadata["backward_flops"] == 100
    assert result.metadata["training_flops"] == 200


def test_training_flops_6p_fallback_when_no_annotations():
    g = _make_graph(metadata={
        "num_layers": 4, "num_layers_traced": 4,
        "seq_len": 128, "total_params": 1000,
    })
    ctx = _make_ctx(micro_batch=1, global_batch=4)

    result = TrainingFlopsPass().run(g, ctx)
    tokens = 128 * 1
    assert result.metadata["forward_flops"] == 2 * 1000 * tokens
    assert result.metadata["backward_flops"] == 4 * 1000 * tokens


# ── Bug 1.2: TrainingMemoryPass ───────────────────────────────────────────────

def test_activation_memory_applies_recompute_policy():
    metadata = {
        "seq_len": 2048, "hidden": 4096, "num_layers": 4,
        "num_layers_traced": 4,
    }

    ctx_none = _make_ctx(tp=1, pp=1, recompute_policy="none")
    ctx_sel = _make_ctx(tp=1, pp=1, recompute_policy="selective")
    ctx_full = _make_ctx(tp=1, pp=1, recompute_policy="full")

    g_none = _make_graph(metadata=metadata)
    g_sel = _make_graph(metadata=metadata)
    g_full = _make_graph(metadata=metadata)

    mem_none = TrainingMemoryPass().run(g_none, ctx_none).metadata["memory_breakdown"]
    mem_sel = TrainingMemoryPass().run(g_sel, ctx_sel).metadata["memory_breakdown"]
    mem_full = TrainingMemoryPass().run(g_full, ctx_full).metadata["memory_breakdown"]

    assert mem_sel.activations == pytest.approx(mem_none.activations * 0.5, rel=0.01)
    assert mem_full.activations == pytest.approx(mem_none.activations * 0.1, rel=0.01)


def test_activation_memory_reads_zero_metadata():
    g = _make_graph(metadata={
        "seq_len": 2048, "hidden": 4096, "num_layers": 4,
        "num_layers_traced": 4, "total_params": 100_000_000,
        "zero": {"weight_shard": 4, "grad_shard": 4, "optstate_shard": 4, "stage": 3},
    })
    ctx = _make_ctx(tp=1, dp=4, zero_stage=3)

    result = TrainingMemoryPass().run(g, ctx)
    mem = result.metadata["memory_breakdown"]

    assert mem.weights > 0
    assert mem.grads > 0
    assert mem.opt_state > 0


def test_activation_memory_pp_inflight():
    metadata = {
        "seq_len": 2048, "hidden": 4096, "num_layers": 4,
        "num_layers_traced": 4,
    }
    ctx_pp1 = _make_ctx(tp=1, pp=1, recompute_policy="none")
    ctx_pp4 = _make_ctx(tp=1, pp=4, recompute_policy="none")

    mem_pp1 = TrainingMemoryPass().run(
        _make_graph(metadata=metadata), ctx_pp1
    ).metadata["memory_breakdown"]
    mem_pp4 = TrainingMemoryPass().run(
        _make_graph(metadata=metadata), ctx_pp4
    ).metadata["memory_breakdown"]

    assert mem_pp4.activations == pytest.approx(mem_pp1.activations * 4, rel=0.01)


# ── Bug 1.1: 1F1B step-time formula ──────────────────────────────────────────

def test_step_time_matches_1f1b_formula():
    pp = 4
    M = 8
    per_stage_us = 1000.0
    stage_time_us = per_stage_us * pp

    metadata = {
        "num_layers": 4, "num_layers_traced": 4,
        "training_flops": 1e12,
    }
    g = _make_graph(metadata=metadata)
    ctx = _make_ctx(pp=pp, micro_batch=1, global_batch=M)

    from unittest.mock import patch, MagicMock
    mock_timeline = MagicMock()
    mock_timeline.total_latency_us = stage_time_us

    with patch("python.zrt.executor.scheduler.DAGScheduler") as MockSched:
        MockSched.return_value.schedule.return_value = mock_timeline
        result = TrainingPipelinePass().run(g, ctx)

    metrics = result.metadata["pipeline_metrics"]
    expected_step_ms = (M + pp - 1) * per_stage_us / 1000.0
    assert metrics.step_time_ms == pytest.approx(expected_step_ms, rel=0.01)


def test_bubble_fraction_correct():
    pp = 4
    M = 8
    per_stage_us = 1000.0
    stage_time_us = per_stage_us * pp

    metadata = {
        "num_layers": 4, "num_layers_traced": 4,
        "training_flops": 1e12,
    }
    g = _make_graph(metadata=metadata)
    ctx = _make_ctx(pp=pp, micro_batch=1, global_batch=M)

    from unittest.mock import patch, MagicMock
    mock_timeline = MagicMock()
    mock_timeline.total_latency_us = stage_time_us

    with patch("python.zrt.executor.scheduler.DAGScheduler") as MockSched:
        MockSched.return_value.schedule.return_value = mock_timeline
        result = TrainingPipelinePass().run(g, ctx)

    metrics = result.metadata["pipeline_metrics"]
    expected_bubble = (pp - 1) / (M + pp - 1)
    assert metrics.bubble_fraction == pytest.approx(expected_bubble, abs=0.01)


def test_steady_steps_equals_num_microbatches():
    pp = 4
    M = 8
    per_stage_us = 1000.0
    stage_time_us = per_stage_us * pp

    metadata = {
        "num_layers": 4, "num_layers_traced": 4,
        "training_flops": 1e12,
    }
    g = _make_graph(metadata=metadata)
    ctx = _make_ctx(pp=pp, micro_batch=1, global_batch=M)

    from unittest.mock import patch, MagicMock
    mock_timeline = MagicMock()
    mock_timeline.total_latency_us = stage_time_us

    with patch("python.zrt.executor.scheduler.DAGScheduler") as MockSched:
        MockSched.return_value.schedule.return_value = mock_timeline
        result = TrainingPipelinePass().run(g, ctx)

    metrics = result.metadata["pipeline_metrics"]
    assert metrics.steady_steps == M
    assert metrics.warmup_steps == pp - 1
    assert metrics.cooldown_steps == pp - 1


# ── Bug 3: FlopsPass gates _calculate_grad_flops on node phase ───────────

def test_recompute_compute_time_excludes_comm_nodes_defensively():
    fwd_compute = OpNode(
        id="fwd_compute",
        op_type="aten.silu.default",
        category="compute",
        annotations={
            "phase": "fwd",
            "recompute": True,
            "latency_us": 100.0,
            "base_latency_us": 80.0,
        },
    )
    bwd_compute = OpNode(
        id="bwd_compute",
        op_type="aten.silu.default",
        category="compute",
        annotations={
            "phase": "bwd",
            "recompute": True,
            "latency_us": 90.0,
            "base_latency_us": 70.0,
        },
    )
    comm = OpNode(
        id="comm",
        op_type="comm.all_to_all",
        category="communication",
        annotations={
            "phase": "fwd",
            "recompute": True,
            "latency_us": 1000.0,
            "base_latency_us": 900.0,
        },
    )
    g = _make_graph(
        [fwd_compute, bwd_compute, comm],
        {"num_layers": 1, "num_layers_traced": 1, "training_flops": 1e12},
    )
    ctx = _make_ctx(pp=1, micro_batch=1, global_batch=1)

    from unittest.mock import MagicMock, patch
    mock_timeline = MagicMock()
    mock_timeline.total_latency_us = 100.0

    with patch("python.zrt.executor.scheduler.DAGScheduler") as MockSched:
        MockSched.return_value.schedule.return_value = mock_timeline
        result = TrainingPipelinePass().run(g, ctx)

    assert result.metadata["recompute_compute_ms"] == pytest.approx(0.15)
    metrics = result.metadata["pipeline_metrics"]
    assert metrics.fwd_compute_ms == 0.0
    assert metrics.bwd_compute_ms == 0.0
    assert metrics.compute_time_ms == pytest.approx(metrics.recompute_compute_ms)


def test_train_flops_pass_zeros_dx_dw_for_backward_phase_nodes():
    """Bug 3: Bwd-phase nodes should have dx_flops=0 and dw_flops=0.

    In stitched graphs, bwd-phase nodes ARE the actual dx/dw computations
    captured by loss.backward(). Applying ratio-based flops_dx/dw on top
    creates phantom FLOPs that don't exist.
    """
    from python.zrt.ir.types import TensorMeta, DType

    tid = 0
    def _tm(shape, dtype=DType.BF16):
        nonlocal tid
        import math
        t = TensorMeta(id=f"t{tid}", shape=shape, dtype=dtype, mem_bytes=math.prod(shape) * 2)
        tid += 1
        return t

    # Forward-phase matmul node
    n_fwd = OpNode(
        id="mm_fwd",
        op_type="aten.mm.default",
        inputs=[_tm((128, 4096)), _tm((4096, 4096))],
        outputs=[_tm((128, 4096))],
        scope="model.layers.0.mlp.gate_proj",
        category="compute",
        layer="0",
    )
    n_fwd.annotations["phase"] = "fwd"

    # Backward-phase matmul node (actual backward computation)
    n_bwd = OpNode(
        id="mm_bwd",
        op_type="aten.mm.default",
        inputs=[_tm((128, 4096)), _tm((4096, 4096))],
        outputs=[_tm((128, 4096))],
        scope="model.layers.0.mlp.gate_proj",
        category="compute",
        layer="0",
    )
    n_bwd.annotations["phase"] = "bwd"

    g = _make_graph([n_fwd, n_bwd])
    result = FlopsPass().run(g, _make_ctx())

    # Forward node: should have dx/dw FLOPs from ratio-based calculation
    fwd_result = result.nodes["mm_fwd"]
    assert fwd_result.annotations["flops_fwd"] > 0
    assert fwd_result.annotations["flops_dx"] > 0
    assert fwd_result.annotations["flops_dw"] > 0

    # Backward node: dx/dw should be ZERO (no double-counting)
    bwd_result = result.nodes["mm_bwd"]
    assert bwd_result.annotations["flops_fwd"] > 0  # Actual backward op cost
    assert bwd_result.annotations["flops_dx"] == 0
    assert bwd_result.annotations["flops_dw"] == 0


def test_train_flops_pass_defaults_to_fwd_phase():
    """Nodes without phase annotation are treated as forward-phase."""
    from python.zrt.ir.types import TensorMeta, DType

    tid = 0
    def _tm(shape, dtype=DType.BF16):
        nonlocal tid
        import math
        t = TensorMeta(id=f"t{tid}", shape=shape, dtype=dtype, mem_bytes=math.prod(shape) * 2)
        tid += 1
        return t

    # Node without phase annotation (defaults to fwd)
    n_no_phase = OpNode(
        id="mm",
        op_type="aten.mm.default",
        inputs=[_tm((128, 4096)), _tm((4096, 4096))],
        outputs=[_tm((128, 4096))],
        scope="model.layers.0.mlp",
        category="compute",
    )

    g = _make_graph([n_no_phase])
    result = FlopsPass().run(g, _make_ctx())

    # Should calculate dx/dw for forward-phase nodes
    assert result.nodes["mm"].annotations["flops_dx"] > 0
    assert result.nodes["mm"].annotations["flops_dw"] > 0


# ── Bug 4: TrainingFlopsPass phase-aware aggregation on stitched graphs ─────

def test_training_flops_pass_phase_aware_on_stitched_graphs():
    """Bug 4: Stitched graphs should split FLOPs by phase, not flat sum."""
    from python.zrt.ir.types import TensorMeta, DType

    tid = 0
    def _tm(shape, dtype=DType.BF16):
        nonlocal tid
        import math
        t = TensorMeta(id=f"t{tid}", shape=shape, dtype=dtype, mem_bytes=math.prod(shape) * 2)
        tid += 1
        return t

    # Forward-phase node
    n1 = OpNode(
        id="mm_fwd",
        op_type="aten.mm.default",
        inputs=[_tm((128, 4096)), _tm((4096, 4096))],
        outputs=[_tm((128, 4096))],
        scope="model.layers.0.mlp",
        category="compute",
    )
    n1.annotations["phase"] = "fwd"
    n1.annotations["flops_fwd"] = 1000
    n1.annotations["flops_dx"] = 500
    n1.annotations["flops_dw"] = 500

    # Backward-phase node
    n2 = OpNode(
        id="mm_bwd",
        op_type="aten.mm.default",
        inputs=[_tm((128, 4096)), _tm((4096, 4096))],
        outputs=[_tm((128, 4096))],
        scope="model.layers.0.mlp",
        category="compute",
    )
    n2.annotations["phase"] = "bwd"
    n2.annotations["flops_fwd"] = 1500  # Actual backward op cost (already flops_dx+dwd)
    n2.annotations["flops_dx"] = 0  # Bug 3 fix ensures this is 0
    n2.annotations["flops_dw"] = 0

    # Stitched graph: fwd_bwd_stitched=True
    g_stitched = _make_graph([n1, n2], metadata={"fwd_bwd_stitched": True, "num_layers": 4, "num_layers_traced": 4})
    result_stitched = TrainingFlopsPass().run(g_stitched, _make_ctx(micro_batch=1, global_batch=4))

    # Stitched path: forward_flops = sum of fwd-phase flops_fwd only
    assert result_stitched.metadata["forward_flops"] == 1000
    # Stitched path: backward_flops = sum of bwd-phase flops_fwd only
    assert result_stitched.metadata["backward_flops"] == 1500
    assert result_stitched.metadata["training_flops"] == 2500

    # Non-stitched graph: should use legacy flat sum
    # Note: After Bug 3 fix, bwd nodes have flops_dx=flops_dw=0, so:
    # backward_flops = n1.flops_dx + n1.flops_dw + n2.flops_dx + n2.flops_dw
    #               = 500 + 500 + 0 + 0 = 1000
    g_legacy = _make_graph([n1, n2], metadata={"num_layers": 4, "num_layers_traced": 4})
    result_legacy = TrainingFlopsPass().run(g_legacy, _make_ctx(micro_batch=1, global_batch=4))

    # Legacy path: forward_flops = sum of ALL flops_fwd
    assert result_legacy.metadata["forward_flops"] == 2500
    # Legacy path: backward_flops = sum of ALL flops_dx + flops_dw
    # After Bug 3 fix: bwd nodes have flops_dx=flops_dw=0, so only fwd node's dx/dw count
    assert result_legacy.metadata["backward_flops"] == 1000


def test_training_flops_pass_phase_aliases_recognized():
    """Phase annotation should recognize common aliases: fwd, forward, train_forward."""
    from python.zrt.ir.types import TensorMeta, DType

    tid = 0
    def _tm(shape, dtype=DType.BF16):
        nonlocal tid
        import math
        t = TensorMeta(id=f"t{tid}", shape=shape, dtype=dtype, mem_bytes=math.prod(shape) * 2)
        tid += 1
        return t

    phases_fwd = ["fwd", "forward", "train_forward"]
    phases_bwd = ["bwd", "backward", "train_backward"]

    for phase in phases_fwd:
        n = OpNode(
            id=f"mm_{phase}",
            op_type="aten.mm.default",
            inputs=[_tm((128, 4096)), _tm((4096, 4096))],
            outputs=[_tm((128, 4096))],
            scope="model.layers.0.mlp",
            category="compute",
        )
        n.annotations["phase"] = phase
        n.annotations["flops_fwd"] = 1000

        g = _make_graph([n], metadata={"fwd_bwd_stitched": True, "num_layers": 4, "num_layers_traced": 4})
        result = TrainingFlopsPass().run(g, _make_ctx())
        # Fwd phases should count toward forward_flops
        assert result.metadata["forward_flops"] == 1000, f"Failed for phase={phase}"
        assert result.metadata["backward_flops"] == 0, f"Failed for phase={phase}"

    for phase in phases_bwd:
        n = OpNode(
            id=f"mm_{phase}",
            op_type="aten.mm.default",
            inputs=[_tm((128, 4096)), _tm((4096, 4096))],
            outputs=[_tm((128, 4096))],
            scope="model.layers.0.mlp",
            category="compute",
        )
        n.annotations["phase"] = phase
        n.annotations["flops_fwd"] = 1500

        g = _make_graph([n], metadata={"fwd_bwd_stitched": True, "num_layers": 4, "num_layers_traced": 4})
        result = TrainingFlopsPass().run(g, _make_ctx())
        # Bwd phases should count toward backward_flops
        assert result.metadata["forward_flops"] == 0, f"Failed for phase={phase}"
        assert result.metadata["backward_flops"] == 1500, f"Failed for phase={phase}"


# ── Recompute FLOPs only for fwd-phase nodes ───────────────────────────────────

def test_recompute_flops_only_for_fwd_phase_nodes():
    """Recompute overhead should only apply to forward-phase nodes.

    Bwd-phase nodes are never recomputed — recomputation is a forward-pass
    technique to trade compute for memory.
    """
    from python.zrt.ir.types import TensorMeta, DType

    tid = 0
    def _tm(shape, dtype=DType.BF16):
        nonlocal tid
        import math
        t = TensorMeta(id=f"t{tid}", shape=shape, dtype=dtype, mem_bytes=math.prod(shape) * 2)
        tid += 1
        return t

    # Fwd-phase node with recompute
    n1 = OpNode(
        id="mm_fwd",
        op_type="aten.mm.default",
        inputs=[_tm((1, 128, 4096))],
        outputs=[_tm((1, 128, 4096))],
        scope="model.layers.0.mlp",
        category="compute",
    )
    n1.annotations["phase"] = "fwd"
    n1.annotations["flops_fwd"] = 2000  # Already includes 2x multiplier
    n1.annotations["recompute"] = True

    # Bwd-phase node with recompute tag (should be ignored)
    n2 = OpNode(
        id="mm_bwd",
        op_type="aten.mm.default",
        inputs=[_tm((1, 128, 4096))],
        outputs=[_tm((1, 128, 4096))],
        scope="model.layers.0.mlp",
        category="compute",
    )
    n2.annotations["phase"] = "bwd"
    n2.annotations["flops_fwd"] = 3000
    n2.annotations["recompute"] = True  # Should be ignored for bwd nodes

    g = _make_graph([n1, n2], metadata={"num_layers": 4, "num_layers_traced": 4})
    result = TrainingFlopsPass().run(g, _make_ctx(micro_batch=1, global_batch=4))

    # Recompute FLOPs = flops_fwd / 2 for fwd-phase nodes only
    # n1: 2000 / 2 = 1000
    # n2: ignored (bwd-phase)
    assert result.metadata["recompute_flops"] == 1000


# ── Integration: FlopsPass → TrainingFlopsPass full pipeline ────────────

def test_integration_train_flops_then_training_flops_stitched():
    """Integration: FlopsPass then TrainingFlopsPass on a stitched graph.

    Verifies the full pipeline: FlopsPass annotates per-node FLOPs,
    then TrainingFlopsPass aggregates them with phase-aware splitting.
    """
    from python.zrt.ir.types import TensorMeta, DType

    tid = [0]
    def _tm(shape, dtype=DType.BF16):
        import math
        t = TensorMeta(id=f"t{tid[0]}", shape=shape, dtype=dtype, mem_bytes=math.prod(shape) * 2)
        tid[0] += 1
        return t

    # Forward-phase matmul node
    n_fwd = OpNode(
        id="mm_fwd",
        op_type="aten.mm.default",
        inputs=[_tm((128, 4096)), _tm((4096, 4096))],
        outputs=[_tm((128, 4096))],
        scope="model.layers.0.mlp.gate_proj",
        category="compute",
        layer="0",
    )
    n_fwd.annotations["phase"] = "fwd"

    # Backward-phase matmul node
    n_bwd = OpNode(
        id="mm_bwd",
        op_type="aten.mm.default",
        inputs=[_tm((128, 4096)), _tm((4096, 4096))],
        outputs=[_tm((128, 4096))],
        scope="model.layers.0.mlp.gate_proj",
        category="compute",
        layer="0",
    )
    n_bwd.annotations["phase"] = "bwd"

    g = _make_graph([n_fwd, n_bwd], metadata={"fwd_bwd_stitched": True, "num_layers": 4, "num_layers_traced": 4})

    # Run full pipeline: FlopsPass → TrainingFlopsPass
    g_annotated = FlopsPass().run(g, _make_ctx())
    result = TrainingFlopsPass().run(g_annotated, _make_ctx(micro_batch=1, global_batch=4))

    # Both nodes should have flops_fwd > 0 (computed by FlopsPass)
    assert g_annotated.nodes["mm_fwd"].annotations["flops_fwd"] > 0
    assert g_annotated.nodes["mm_bwd"].annotations["flops_fwd"] > 0

    # Bwd node should have zero dx/dw (Bug 3 fix)
    assert g_annotated.nodes["mm_bwd"].annotations["flops_dx"] == 0
    assert g_annotated.nodes["mm_bwd"].annotations["flops_dw"] == 0

    # Stitched path: forward_flops = fwd-phase flops_fwd only
    fwd_flops = g_annotated.nodes["mm_fwd"].annotations["flops_fwd"]
    bwd_flops = g_annotated.nodes["mm_bwd"].annotations["flops_fwd"]
    assert result.metadata["forward_flops"] == fwd_flops
    assert result.metadata["backward_flops"] == bwd_flops
    assert result.metadata["training_flops"] == fwd_flops + bwd_flops


def test_integration_recompute_multiplier_not_applied_to_bwd_nodes():
    """Integration: Recompute multiplier should NOT apply to backward-phase nodes.

    If a bwd node has annotations["recompute"], the 2x multiplier should
    NOT be applied because recomputation is a forward-pass technique.
    """
    from python.zrt.ir.types import TensorMeta, DType

    tid = [0]
    def _tm(shape, dtype=DType.BF16):
        import math
        t = TensorMeta(id=f"t{tid[0]}", shape=shape, dtype=dtype, mem_bytes=math.prod(shape) * 2)
        tid[0] += 1
        return t

    # Fwd-phase node with recompute
    n_fwd = OpNode(
        id="mm_fwd",
        op_type="aten.mm.default",
        inputs=[_tm((128, 4096)), _tm((4096, 4096))],
        outputs=[_tm((128, 4096))],
        scope="model.layers.0.mlp",
        category="compute",
    )
    n_fwd.annotations["phase"] = "fwd"
    n_fwd.annotations["recompute"] = True

    # Bwd-phase node with recompute tag (should NOT get 2x)
    n_bwd = OpNode(
        id="mm_bwd",
        op_type="aten.mm.default",
        inputs=[_tm((128, 4096)), _tm((4096, 4096))],
        outputs=[_tm((128, 4096))],
        scope="model.layers.0.mlp",
        category="compute",
    )
    n_bwd.annotations["phase"] = "bwd"
    n_bwd.annotations["recompute"] = True  # Should be ignored for bwd

    g = _make_graph([n_fwd, n_bwd])
    result = FlopsPass().run(g, _make_ctx())

    # Both nodes have same input shapes, so base flops should be equal
    # Fwd node: flops_fwd = base * 2 (recompute multiplier applied)
    # Bwd node: flops_fwd = base * 1 (recompute multiplier NOT applied)
    fwd_flops = result.nodes["mm_fwd"].annotations["flops_fwd"]
    bwd_flops = result.nodes["mm_bwd"].annotations["flops_fwd"]

    assert fwd_flops == 2 * bwd_flops, (
        f"Fwd node should have 2x flops due to recompute, got fwd={fwd_flops} bwd={bwd_flops}"
    )


def test_flash_attention_internal_recompute_is_not_external_checkpoint_replay():
    """FA/SDPA backward already includes internal recompute, so do not double it."""
    from python.zrt.ir.types import DType, TensorMeta

    q = TensorMeta.from_shape_dtype("q", (1, 128, 16, 64), DType.BF16)
    out = TensorMeta.from_shape_dtype("out", (1, 128, 16, 64), DType.BF16)
    grad = TensorMeta.from_shape_dtype("grad", (1, 128, 16, 64), DType.BF16)

    fa = OpNode(
        id="fa",
        op_type="aten._scaled_dot_product_flash_attention.default",
        inputs=[q],
        outputs=[out],
        category="compute",
        annotations={
            "phase": "fwd",
            "recompute": True,
            "sem_flops": 1_000_000,
            "sem_io": {
                "activation": {"bytes": q.mem_bytes},
                "output": {"bytes": out.mem_bytes},
            },
        },
    )
    bwd = OpNode(
        id="fa_bwd",
        op_type="aten.mm.default",
        inputs=[out],
        outputs=[grad],
        category="compute",
        annotations={"phase": "bwd"},
    )
    g = OpGraph(
        name="fa_internal_recompute",
        phase="train",
        nodes={fa.id: fa, bwd.id: bwd},
        edges=[Edge(src="fa", src_idx=0, dst="fa_bwd", dst_idx=0, tensor=out)],
        metadata={"fwd_bwd_stitched": True},
    )

    ctx = _make_ctx(recompute_policy="full")
    g = FlopsPass().run(g, ctx)
    assert g.nodes["fa"].annotations["flops_fwd"] == g.nodes["fa"].annotations["flops"]

    g = TrainingFlopsPass().run(g, ctx)
    assert g.metadata["recompute_flops"] == 0

    g = RooflinePass().run(g, ctx)
    assert g.nodes["fa_bwd"].annotations["recompute_flops"] == 0
    assert g.nodes["fa_bwd"].annotations["recompute_latency_us"] == 0.0


def test_recompute_helper_classifies_internal_attention_kernels():
    assert has_internal_recompute(OpNode(id="flash", op_type="flash_attn_fwd"))
    assert has_internal_recompute(OpNode(id="sdpa", op_type="aten.sdpa.default"))
    assert has_internal_recompute(
        OpNode(id="scaled_dot", op_type="aten.scaled_dot_product_attention.default")
    )
    assert not has_internal_recompute(OpNode(id="mask", op_type="custom_attn_mask_apply"))
    assert not has_internal_recompute(OpNode(id="mm", op_type="aten.mm.default"))
    assert not has_internal_recompute(OpNode(id="silu", op_type="aten.silu.default"))


def test_recompute_helper_classifies_external_checkpoint_replay():
    mm = OpNode(id="mm", op_type="aten.mm.default", annotations={"recompute": True})
    flash = OpNode(id="flash", op_type="flash_attn_fwd", annotations={"recompute": True})
    plain = OpNode(id="plain", op_type="aten.mm.default")

    assert is_external_recompute_node(mm)
    assert not is_external_recompute_node(flash)
    assert not is_external_recompute_node(plain)


def test_integration_flops_pass_computes_training_annotations():
    """FlopsPass in training mode computes raw flops + gradient annotations."""
    from python.zrt.ir.types import TensorMeta, DType

    tid = [0]
    def _tm(shape, dtype=DType.BF16):
        import math
        t = TensorMeta(id=f"t{tid[0]}", shape=shape, dtype=dtype, mem_bytes=math.prod(shape) * 2)
        tid[0] += 1
        return t

    n = OpNode(
        id="mm",
        op_type="aten.mm.default",
        inputs=[_tm((128, 4096)), _tm((4096, 4096))],
        outputs=[_tm((128, 4096))],
        scope="model.layers.0.mlp",
        category="compute",
    )

    g = _make_graph([n])
    result = FlopsPass().run(g, _make_ctx())

    # Raw forward FLOPs computed by sim._fmr()
    expected_flops = 2 * 128 * 4096 * 4096
    assert result.nodes["mm"].annotations["flops"] == expected_flops
    # Training mode: flops_fwd = raw (no recompute), flops_dx/dw from grad ratios
    assert result.nodes["mm"].annotations["flops_fwd"] == expected_flops
    assert result.nodes["mm"].annotations["flops_dx"] == expected_flops
    assert result.nodes["mm"].annotations["flops_dw"] == expected_flops


# ── Reviewer HIGH-1 regression: aten.addmm / aten.linear gradient FLOPs ──────

def test_flops_pass_addmm_default_produces_gradient_flops():
    """aten.addmm.default nodes must produce non-zero flops_dx/dw."""
    from python.zrt.ir.types import TensorMeta, DType

    tid = [0]
    def _tm(shape, dtype=DType.BF16):
        import math
        t = TensorMeta(id=f"t{tid[0]}", shape=shape, dtype=dtype, mem_bytes=math.prod(shape) * 2)
        tid[0] += 1
        return t

    n = OpNode(
        id="addmm",
        op_type="aten.addmm.default",
        inputs=[_tm((128,)), _tm((128, 4096)), _tm((4096, 4096))],
        outputs=[_tm((128, 4096))],
        scope="model.layers.0.mlp.gate_proj",
        category="compute",
    )
    g = _make_graph([n])
    result = FlopsPass().run(g, _make_ctx())

    assert result.nodes["addmm"].annotations["flops_fwd"] > 0
    assert result.nodes["addmm"].annotations["flops_dx"] > 0, \
        "aten.addmm.default must produce gradient dx FLOPs"
    assert result.nodes["addmm"].annotations["flops_dw"] > 0, \
        "aten.addmm.default must produce gradient dw FLOPs"


def test_flops_pass_linear_default_produces_gradient_flops():
    """aten.linear.default nodes must produce non-zero flops_dx/dw."""
    from python.zrt.ir.types import TensorMeta, DType

    tid = [0]
    def _tm(shape, dtype=DType.BF16):
        import math
        t = TensorMeta(id=f"t{tid[0]}", shape=shape, dtype=dtype, mem_bytes=math.prod(shape) * 2)
        tid[0] += 1
        return t

    n = OpNode(
        id="linear",
        op_type="aten.linear.default",
        inputs=[_tm((128, 4096)), _tm((4096, 4096))],
        outputs=[_tm((128, 4096))],
        scope="model.layers.0.mlp.gate_proj",
        category="compute",
    )
    g = _make_graph([n])
    result = FlopsPass().run(g, _make_ctx())

    assert result.nodes["linear"].annotations["flops_fwd"] > 0
    assert result.nodes["linear"].annotations["flops_dx"] > 0, \
        "aten.linear.default must produce gradient dx FLOPs"
    assert result.nodes["linear"].annotations["flops_dw"] > 0, \
        "aten.linear.default must produce gradient dw FLOPs"


# ── Reviewer HIGH-2 regression: attention compression ratio ──────────────────

def test_flops_pass_attention_compression_ratio_applied():
    """Attention nodes with compression ratio should scale flops_fwd/dx."""
    from python.zrt.ir.types import TensorMeta, DType

    tid = [0]
    def _tm(shape, dtype=DType.BF16):
        import math
        t = TensorMeta(id=f"t{tid[0]}", shape=shape, dtype=dtype, mem_bytes=math.prod(shape) * 2)
        tid[0] += 1
        return t

    n = OpNode(
        id="attn",
        op_type="aten.scaled_dot_product_attention",
        inputs=[_tm((1, 1024, 4096))],
        outputs=[_tm((1, 1024, 4096))],
        scope="model.layers.0.self_attn",
        category="compute",
    )
    n.annotations["attn_compression_ratio"] = 0.5

    g = _make_graph([n], metadata={"attn_compression_ratio": 0.27})
    result = FlopsPass().run(g, _make_ctx())

    # Node-level ratio (0.5) should override graph-level (0.27)
    raw_flops = result.nodes["attn"].annotations["flops"]
    fwd_flops = result.nodes["attn"].annotations["flops_fwd"]
    assert fwd_flops == int(raw_flops * 0.5), \
        f"Expected flops_fwd={int(raw_flops * 0.5)}, got {fwd_flops}"
    assert result.nodes["attn"].annotations["flops_dx"] == int(2.5 * raw_flops * 0.5)


def test_flops_pass_attention_graph_compression_ratio():
    """Graph-level compression ratio should apply when node has none."""
    from python.zrt.ir.types import TensorMeta, DType

    tid = [0]
    def _tm(shape, dtype=DType.BF16):
        import math
        t = TensorMeta(id=f"t{tid[0]}", shape=shape, dtype=dtype, mem_bytes=math.prod(shape) * 2)
        tid[0] += 1
        return t

    n = OpNode(
        id="attn",
        op_type="aten.scaled_dot_product_attention",
        inputs=[_tm((1, 512, 1024))],
        outputs=[_tm((1, 512, 1024))],
        scope="model.layers.0.self_attn",
        category="compute",
    )

    g = _make_graph([n], metadata={"attn_compression_ratio": 0.27})
    result = FlopsPass().run(g, _make_ctx())

    raw_flops = result.nodes["attn"].annotations["flops"]
    fwd_flops = result.nodes["attn"].annotations["flops_fwd"]
    assert fwd_flops == int(raw_flops * 0.27), \
        f"Expected flops_fwd={int(raw_flops * 0.27)}, got {fwd_flops}"
