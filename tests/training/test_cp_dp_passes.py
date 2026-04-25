"""Regression tests for Phase 3.1 (ContextParallelPass) and 3.2 (DataParallelPass)."""
from __future__ import annotations

import pytest

from zrt.hardware.spec import (
    ComputeSpec, HardwareSpec, InterconnectSpec, LinkSpec, MemorySpec,
)
from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.types import DType, TensorMeta
from zrt.transform.context import ParallelConfig, TrainingConfig, TransformContext
from zrt.transform.parallel.comm_inserter import CommInserterPass
from zrt.transform.parallel.context_parallel import ContextParallelPass
from zrt.transform.parallel.data_parallel import DataParallelPass


# ── helpers ──────────────────────────────────────────────────────────────────

def _hw():
    return HardwareSpec(
        name="test_h100", vendor="nvidia", device_type="gpu",
        compute=ComputeSpec(bf16_tflops=989, fp8_tops=1979),
        memory=MemorySpec(capacity_gb=80, hbm_bandwidth_gbps=3350),
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=100, latency_us=10.0),
        ),
    )


def _ctx(cp=1, cp_kind="ulysses", dp=1, zero_stage=0, dp_overlap=True):
    return TransformContext(
        hw_spec=_hw(),
        parallel=ParallelConfig(cp=cp, dp=dp),
        training=TrainingConfig(
            micro_batch=1, global_batch=8,
            cp_kind=cp_kind,
            zero_stage=zero_stage,
            dp_overlap_in_bubble=dp_overlap,
        ),
    )


def _attn_node(node_id="attn_0"):
    """Create a minimal attention node that ContextParallelPass will match."""
    t = TensorMeta(id="t0", shape=(1, 128, 4096), dtype=DType.BF16, mem_bytes=1 * 128 * 4096 * 2)
    return OpNode(
        id=node_id,
        op_type="aten._scaled_dot_product_attention",
        inputs=[t],
        outputs=[t],
        scope="model.layers.0.self_attn",
        category="compute",
    )


def _bwd_grad_node(node_id="grad_0"):
    """Create a backward/grad node for DataParallelPass."""
    t = TensorMeta(id="t1", shape=(4096, 4096), dtype=DType.BF16, mem_bytes=4096 * 4096 * 2)
    return OpNode(
        id=node_id,
        op_type="aten.mm_backward",
        inputs=[t],
        outputs=[t],
        scope="model.layers.0.mlp",
        category="compute",
    )


def _make_graph(nodes, phase="forward", metadata=None):
    return OpGraph(
        name="test", phase=phase,
        nodes={n.id: n for n in nodes},
        metadata=metadata or {},
    )


# ── Phase 3.1: ContextParallelPass ───────────────────────────────────────────

def test_cp_ulysses_annotates_attention():
    g = _make_graph([_attn_node()])
    ctx = _ctx(cp=2, cp_kind="ulysses")
    result = ContextParallelPass().run(g, ctx)
    ann = result.nodes["attn_0"].annotations.get("cp_split", {})
    assert ann["kind"] == "ulysses"
    assert ann["cp"] == 2


def test_cp_ring_annotates_attention():
    g = _make_graph([_attn_node()])
    ctx = _ctx(cp=2, cp_kind="ring")
    result = ContextParallelPass().run(g, ctx)
    ann = result.nodes["attn_0"].annotations.get("cp_split", {})
    assert ann["kind"] == "ring"
    assert ann["cp"] == 2
    assert ann["p2p_rounds"] == 2


def test_cp_pass_skipped_when_cp_1():
    g = _make_graph([_attn_node()])
    ctx = _ctx(cp=1)
    result = ContextParallelPass().run(g, ctx)
    # Graph returned unchanged (same object)
    assert "cp_split" not in result.nodes["attn_0"].annotations


# ── Phase 3.1: CommInserterPass CP insertion ─────────────────────────────────

def test_comm_inserter_cp_ulysses_inserts_a2a_pair():
    g = _make_graph([_attn_node()])
    ctx = _ctx(cp=2, cp_kind="ulysses")
    g = ContextParallelPass().run(g, ctx)
    result = CommInserterPass().run(g, ctx)

    a2a_nodes = [n for n in result.nodes.values() if n.op_type == "comm.all_to_all"]
    assert len(a2a_nodes) == 2
    roles = {n.attrs["role"] for n in a2a_nodes}
    assert "cp_ulysses_pre" in roles
    assert "cp_ulysses_post" in roles


def test_comm_inserter_cp_ring_inserts_p2p_with_overlap_target():
    cp = 3
    g = _make_graph([_attn_node()])
    ctx = _ctx(cp=cp, cp_kind="ring")
    g = ContextParallelPass().run(g, ctx)
    result = CommInserterPass().run(g, ctx)

    p2p_nodes = [n for n in result.nodes.values() if n.op_type == "comm.send_recv"]
    assert len(p2p_nodes) == cp
    for p2p in p2p_nodes:
        assert p2p.annotations.get("overlap_target") == "attn_0"


# ── Phase 3.2: DataParallelPass ───────────────────────────────────────────────

def test_dp_pass_inserts_all_reduce_zero0():
    g = _make_graph([_bwd_grad_node()], phase="train_backward")
    ctx = _ctx(dp=4, zero_stage=0)
    result = DataParallelPass().run(g, ctx)
    ar_nodes = [n for n in result.nodes.values() if n.op_type == "comm.all_reduce"]
    assert len(ar_nodes) == 1


def test_dp_pass_inserts_reduce_scatter_nonzero_zero():
    g = _make_graph([_bwd_grad_node()], phase="train_backward")
    ctx = _ctx(dp=4, zero_stage=2)
    result = DataParallelPass().run(g, ctx)
    rs_nodes = [n for n in result.nodes.values() if n.op_type == "comm.reduce_scatter"]
    assert len(rs_nodes) == 1


def test_dp_pass_dp_comm_annotation():
    g = _make_graph([_bwd_grad_node()], phase="train_backward")
    ctx = _ctx(dp=4, zero_stage=0)
    result = DataParallelPass().run(g, ctx)
    comm_nodes = [n for n in result.nodes.values() if n.category == "communication"]
    assert len(comm_nodes) == 1
    assert comm_nodes[0].annotations.get("dp_comm") is True


def test_dp_pass_overlap_in_bubble_annotation():
    g = _make_graph([_bwd_grad_node()], phase="train_backward")
    ctx = _ctx(dp=4, zero_stage=0, dp_overlap=True)
    result = DataParallelPass().run(g, ctx)
    comm_node = next(n for n in result.nodes.values() if n.category == "communication")
    assert comm_node.annotations.get("overlap_in_bubble") is True


def test_dp_pass_no_overlap_annotation_when_disabled():
    g = _make_graph([_bwd_grad_node()], phase="train_backward")
    ctx = _ctx(dp=4, zero_stage=0, dp_overlap=False)
    result = DataParallelPass().run(g, ctx)
    comm_node = next(n for n in result.nodes.values() if n.category == "communication")
    assert "overlap_in_bubble" not in comm_node.annotations


def test_dp_pass_skipped_when_dp_1():
    g = _make_graph([_bwd_grad_node()], phase="train_backward")
    ctx = _ctx(dp=1)
    result = DataParallelPass().run(g, ctx)
    assert len(result.nodes) == 1  # no comm node added


# ── Phase 3.2: DP-in-bubble in TrainingPipelinePass ──────────────────────────

def _ctx_pipeline(pp=4, dp=4, dp_overlap=True, inter_bw_gbps=100):
    hw = HardwareSpec(
        name="test", vendor="nvidia", device_type="gpu",
        compute=ComputeSpec(bf16_tflops=989, fp8_tops=1979),
        memory=MemorySpec(capacity_gb=80, hbm_bandwidth_gbps=3350),
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=inter_bw_gbps, latency_us=10.0),
        ),
    )
    return TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(pp=pp, dp=dp),
        training=TrainingConfig(
            micro_batch=1, global_batch=8,
            dp_overlap_in_bubble=dp_overlap,
        ),
    )


def test_pipeline_pass_dp_in_bubble_adds_exposed_time():
    """When DP AR > bubble window, the exposed time is added to step_time_ms."""
    from zrt.ir.node import OpNode as _OpNode
    from zrt.transform.analysis.training import TrainingPipelinePass

    pp = 2
    M = 4
    per_stage_us = 1000.0

    metadata = {
        "num_layers": 2, "num_layers_traced": 2, "training_flops": 1e12,
    }
    # Inject a pre-annotated DP comm node with a large bucket (forces AR >> bubble)
    # AR time: 2*(dp-1)/dp * 1e9 bytes / (100 Gbps * 1e9/8 / 1e6) = big
    large_bucket = 10_000_000_000  # 10 GB → AR >> bubble
    matmul = _OpNode(
        id="mm_0", op_type="aten.mm.default",
        inputs=[], outputs=[], scope="model.layers.0.mlp",
    )
    comm = _OpNode(
        id="comm_grad_reduce",
        op_type="comm.all_reduce",
        inputs=[], outputs=[],
        attrs={"group_size": 4, "collective": "all_reduce",
               "bucket_bytes": large_bucket},
        scope="data_parallel.grad_reduce",
        category="communication",
    )
    comm.annotations["dp_comm"] = True
    comm.annotations["overlap_in_bubble"] = True
    g = OpGraph(name="test", phase="forward",
                nodes={"mm_0": matmul, "comm_grad_reduce": comm},
                metadata=metadata)

    from unittest.mock import patch, MagicMock
    mock_tl = MagicMock()
    mock_tl.total_latency_us = per_stage_us * pp

    ctx_overlap = _ctx_pipeline(pp=pp, dp=4, dp_overlap=True, inter_bw_gbps=100)
    ctx_no_overlap = _ctx_pipeline(pp=pp, dp=4, dp_overlap=False, inter_bw_gbps=100)

    with patch("python.zrt.executor.scheduler.DAGScheduler") as MockSched:
        MockSched.return_value.schedule.return_value = mock_tl
        result_overlap = TrainingPipelinePass().run(g, ctx_overlap)
        result_no_overlap = TrainingPipelinePass().run(g, ctx_no_overlap)

    # With large DP AR and overlap enabled, step time > no-overlap case
    assert (result_overlap.metadata["pipeline_metrics"].step_time_ms >=
            result_no_overlap.metadata["pipeline_metrics"].step_time_ms)


def test_pipeline_pass_dp_in_bubble_no_penalty_when_ar_fits():
    """When DP AR fits inside bubble, step_time_ms is unchanged."""
    from zrt.ir.node import OpNode as _OpNode
    from zrt.transform.analysis.training import TrainingPipelinePass

    pp = 4
    M = 8
    per_stage_us = 1000.0

    metadata = {
        "num_layers": 4, "num_layers_traced": 4, "training_flops": 1e12,
    }
    # Tiny bucket: AR << bubble
    tiny_bucket = 1  # 1 byte → effectively zero latency
    matmul = _OpNode(id="mm_0", op_type="aten.mm.default", inputs=[], outputs=[])
    comm = _OpNode(
        id="comm_grad_reduce",
        op_type="comm.all_reduce",
        inputs=[], outputs=[],
        attrs={"group_size": 4, "collective": "all_reduce", "bucket_bytes": tiny_bucket},
        scope="data_parallel.grad_reduce",
        category="communication",
    )
    comm.annotations["dp_comm"] = True
    comm.annotations["overlap_in_bubble"] = True
    g = OpGraph(name="test", phase="forward",
                nodes={"mm_0": matmul, "comm_grad_reduce": comm},
                metadata=metadata)

    from unittest.mock import patch, MagicMock
    mock_tl = MagicMock()
    mock_tl.total_latency_us = per_stage_us * pp

    ctx = _ctx_pipeline(pp=pp, dp=4, dp_overlap=True)
    ctx_no = _ctx_pipeline(pp=pp, dp=4, dp_overlap=False)

    with patch("python.zrt.executor.scheduler.DAGScheduler") as MockSched:
        MockSched.return_value.schedule.return_value = mock_tl
        r_on = TrainingPipelinePass().run(g, ctx)
        r_off = TrainingPipelinePass().run(g, ctx_no)

    assert (r_on.metadata["pipeline_metrics"].step_time_ms ==
            pytest.approx(r_off.metadata["pipeline_metrics"].step_time_ms, rel=0.01))
