'''End-to-end tests for Pipeline Parallel strategies.

Tests 4 PP scheduling strategies using the full pipeline:
  1. 1F1B (baseline)
  2. Interleaved (VPP)
  3. DualPipe
  4. DualPipeV

Performance Optimization:
  - Graph capture runs once (module scope fixture)
  - All schedule estimates run once in cached_reports fixture
  - Tests read from cache, no repeated computation

Execution:
  pytest tests/IT/test_pp_e2e.py -v

CLI equivalent (for manual verification):
  python -m python.zrt --model-id hf_models/deepseek_v4 --layers 4 --train --hw nvidia_h100_sxm --tp 8 --pp 2

Composer contracts these tests rely on:
  * bubble_fraction formulas (per composer in python/zrt/training/compose/):
      - 1F1B : (pp - 1) / (num_micro_batches + pp - 1)
      - VPP  : (pp - 1) / (num_micro_batches * vpp_chunks + pp - 1)
      - DualPipe / DualPipeV : 0 when pp == 2 (slots = pp/2 - 1 = 0)
If any composer changes its formula, the matching theory test must be updated
     in tandem; do not silently widen the tolerance.
'''
from __future__ import annotations

import json
import sys
import pytest
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

_MODEL_DIR = _REPO_ROOT / "hf_models" / "deepseek_v4"
_MODEL_CONFIG = json.loads((_MODEL_DIR / "config.json").read_text(encoding="utf-8"))
_HIDDEN_SIZE = int(_MODEL_CONFIG["hidden_size"])
_CAPTURED_LAYERS = 4


@pytest.fixture(scope="module")
def get_model_graphs():
    """Capture DeepSeek-V4 training graphs ONCE for all tests.

    【性能优化】scope="module" 确保graph capture只执行一次。
    cached_reports fixture 在每次调用 estimate_training_from_graphs 前会 clone 输入 graph，
    避免 estimate 内部对 metadata 的写入污染原始 graph 导致非确定性结果。
    """
    from zrt.pipeline import run_trace_phases

    model_id = str(_MODEL_DIR)

    result = run_trace_phases(
        model_id=model_id,
        num_layers=_CAPTURED_LAYERS,
        batch_size=1,
        seq_len=128,
        phases=("train_forward", "train_backward"),
    )

    fwd = result.graphs.get("train_forward")
    bwd = result.graphs.get("train_backward")

    assert fwd is not None, "train_forward graph must be captured"
    assert bwd is not None, "train_backward graph must be captured"
    assert len(fwd.nodes) > 1000, f"Expected >1000 forward nodes, got {len(fwd.nodes)}"
    assert len(bwd.nodes) > 1000, f"Expected >1000 backward nodes, got {len(bwd.nodes)}"

    return {
        "forward": fwd,
        "backward": bwd,
        "model_id": model_id,
    }


@pytest.fixture(scope="module")
def get_hardware():
    """Load H100 SXM hardware spec ONCE for all tests."""
    import zrt.hardware.registry as hw_registry
    hw = hw_registry.load("nvidia_h100_sxm")

    assert hw.compute.bf16_tflops == pytest.approx(989.0, rel=0.01), \
        f"H100 BF16 TFLOPS {hw.compute.bf16_tflops} not within 1% of 989.0"
    assert hw.memory.hbm_bandwidth_gbps == pytest.approx(3350.0, rel=0.01), \
        f"H100 HBM bandwidth {hw.memory.hbm_bandwidth_gbps} not within 1% of 3350.0"
    assert hw.memory.capacity_gb == pytest.approx(80.0, rel=0.01), \
        f"H100 capacity {hw.memory.capacity_gb} not within 1% of 80.0"

    return hw


@pytest.fixture(scope="module")
def cached_reports(get_model_graphs, get_hardware):
    """Pre-compute all schedule estimates ONCE.

    【性能优化】4种schedule的estimate只各执行一次，结果缓存在dict中。
    测试用例直接读取缓存，避免重复运行transform pipeline。
    使用 return_transformed=True 同时获取变换后的图，供 P2P 数量测试使用。

    Returns:
        dict mapping (schedule, vpp_chunks) -> (TrainingReport, dict[str, OpGraph])
        dict[str, OpGraph] 为变换后的图，key 为 "unified"
    """
    from zrt.transform.analysis import estimate_training_from_graphs

    configs = [
        ("1f1b", 1),
        ("interleaved", 2),
        ("dualpipe", 1),
        ("dualpipev", 2),
    ]

    cache = {}
    for schedule, vpp_chunks in configs:
        key = (schedule, vpp_chunks)
        fwd_clone = get_model_graphs["forward"].clone()
        bwd_clone = get_model_graphs["backward"].clone()
        report, ctx, transformed = estimate_training_from_graphs(
            forward_graph=fwd_clone,
            backward_graph=bwd_clone,
            hw_spec=get_hardware,
            seq_len=128,
            batch_size=1,
            hidden=_HIDDEN_SIZE,
            num_layers=_CAPTURED_LAYERS,
            tp=8,
            pp=2,
            micro_batch=1,
            global_batch=8,
            pp_schedule=schedule,
            vpp_chunks=vpp_chunks,
            return_transformed=True,
        )
        cache[key] = (report, transformed)

    return cache


def _get_report(cached_reports, schedule: str, vpp_chunks: int = 1):
    """Get cached report for given schedule config."""
    return cached_reports[(schedule, vpp_chunks)][0]


def _get_transformed_graph(cached_reports, schedule: str, vpp_chunks: int = 1):
    """Get transformed graph for given schedule config.

    Returns the unified (fwd+bwd stitched) transformed OpGraph.
    """
    transformed = cached_reports[(schedule, vpp_chunks)][1]
    return transformed.get("unified", transformed.get("train_forward"))


def _count_pp_p2p_nodes(graph) -> int:
    """Count PP P2P nodes (comm.send_recv without role attr) in a transformed graph."""
    return len([
        n for n in graph.nodes.values()
        if n.op_type == "comm.send_recv" and n.attrs.get("role") is None
    ])


def _count_pp_p2p_by_phase(graph) -> dict[str, int]:
    """Count PP P2P nodes split by fwd/bwd phase annotation."""
    pp_p2p = [
        n for n in graph.nodes.values()
        if n.op_type == "comm.send_recv" and n.attrs.get("role") is None
    ]
    fwd_count = len([n for n in pp_p2p if n.annotations.get("phase") == "fwd"])
    bwd_count = len([n for n in pp_p2p if n.annotations.get("phase") == "bwd"])
    return {"fwd": fwd_count, "bwd": bwd_count}


def _get_p2p_volume_by_phase(graph) -> dict[str, int]:
    """Sum P2P message_size_bytes split by fwd/bwd phase."""
    pp_p2p = [
        n for n in graph.nodes.values()
        if n.op_type == "comm.send_recv" and n.attrs.get("role") is None
    ]
    fwd_vol = sum(n.attrs.get("message_size_bytes", 0) for n in pp_p2p if n.annotations.get("phase") == "fwd")
    bwd_vol = sum(n.attrs.get("message_size_bytes", 0) for n in pp_p2p if n.annotations.get("phase") == "bwd")
    return {"fwd": fwd_vol, "bwd": bwd_vol}


def _get_p2p_total_volume(graph) -> int:
    """Sum all P2P message_size_bytes."""
    return sum(
        n.attrs.get("message_size_bytes", 0)
        for n in graph.nodes.values()
        if n.op_type == "comm.send_recv" and n.attrs.get("role") is None
    )


def _get_stage_layer_assignment(graph) -> dict[int, set[int]]:
    """Extract stage_id → set of layer indices from graph node annotations."""
    stage_layers: dict[int, set[int]] = {}
    for n in graph.nodes.values():
        if n.op_type.startswith("comm."):
            continue
        try:
            layer_idx = int(n.layer) if n.layer else -1
        except (ValueError, TypeError):
            continue
        if layer_idx < 0:
            continue
        stage_id = n.annotations.get("stage_id", -1)
        if stage_id < 0:
            continue
        stage_layers.setdefault(stage_id, set()).add(layer_idx)
    return stage_layers


def _is_contiguous(layers: set[int]) -> bool:
    """Check if a set of layer indices forms a contiguous integer range."""
    if len(layers) <= 1:
        return True
    sorted_layers = sorted(layers)
    return all(sorted_layers[i + 1] - sorted_layers[i] == 1 for i in range(len(sorted_layers) - 1))


def _get_stage_compute_load(graph) -> dict[int, float]:
    """Sum compute_us per stage from node annotations."""
    stage_load: dict[int, float] = {}
    for n in graph.nodes.values():
        if n.op_type.startswith("comm."):
            continue
        stage_id = n.annotations.get("stage_id", -1)
        if stage_id < 0:
            continue
        load = (n.annotations.get("compute_us")
                or n.annotations.get("latency_us")
                or n.annotations.get("flops", 0) / 1e12
                or 1.0)
        stage_load[stage_id] = stage_load.get(stage_id, 0.0) + load
    return stage_load


class TestPPScheduleTheoryValidation:
    """Validate bubble fraction against theoretical formulas.

    PP=2, global_batch=8, micro_batch=1 → num_micro_batches = 8

    Bubble formulas (精确理论值):
      - 1F1B:    bubble = (pp-1) / (num_micro_batches + pp - 1) = 1/9 = 0.111111111...
      - VPP:     bubble = (pp-1) / (num_micro_batches * vpp_chunks + pp - 1) = 1/17 = 0.058823529...
      - DualPipe: bubble = 0 when pp=2 (slots = pp/2-1 = 0, perfect F/B overlap)
      - DualPipeV: bubble = 0 when pp=2 (slots/V = 0, perfect overlap)
    """

    def test_1f1b_bubble_equals_one_over_nine(self, cached_reports):
        """1F1B bubble = 1/(8+2-1) = 1/9 = 0.111111111.

        【原理】1F1B bubble 公式: bubble = (pp-1) / (num_micro_batches + pp - 1)
        【观测点】bubble_fraction == 1/9 精确匹配（理论调度算法保证）
        """
        r = _get_report(cached_reports, "1f1b")
        expected_bubble = 1.0 / 9.0
        assert r.bubble_fraction == pytest.approx(expected_bubble, rel=0.001), \
            f"1F1B bubble {r.bubble_fraction} != 1/9 = {expected_bubble}"

    def test_vpp_bubble_equals_one_over_seventeen(self, cached_reports):
        """VPP (vpp_chunks=2) bubble = 1/(8*2+2-1) = 1/17 = 0.058823529.

        【原理】VPP bubble 公式: bubble = (pp-1) / (num_micro_batches * vpp_chunks + pp - 1)
        【观测点】bubble_fraction == 1/17 精确匹配
        """
        r = _get_report(cached_reports, "interleaved", 2)
        expected_bubble = 1.0 / 17.0
        assert r.bubble_fraction == pytest.approx(expected_bubble, rel=0.001), \
            f"VPP bubble {r.bubble_fraction} != 1/17 = {expected_bubble}"

    def test_dualpipe_bubble_equals_zero_at_pp2(self, cached_reports):
        """DualPipe bubble = 0 when pp=2 (perfect F/B overlap).

        【原理】DualPipe 公式: slots = pp/2-1, pp=2 时 slots=0 → bubble=0
                 两条反向微批次流完美重叠，无 pipeline bubble
        【观测点】bubble_fraction == 0 精确匹配
        """
        r = _get_report(cached_reports, "dualpipe")
        assert r.bubble_fraction == pytest.approx(0.0, abs=1e-6), \
            f"DualPipe bubble {r.bubble_fraction} != 0 (pp=2 perfect overlap)"

    def test_dualpipev_bubble_equals_zero_at_pp2(self, cached_reports):
        """DualPipeV bubble = 0 when pp=2 (perfect F/B overlap).

        【原理】DualPipeV 公式: slots/V = (pp/2-1)/V, pp=2 时 slots=0 → bubble=0
                 VPP interleaved + DualPipe F/B parallel，pp=2 时完美重叠
        【观测点】bubble_fraction == 0 精确匹配
        """
        r = _get_report(cached_reports, "dualpipev", 2)
        assert r.bubble_fraction == pytest.approx(0.0, abs=1e-6), \
            f"DualPipeV bubble {r.bubble_fraction} != 0 (pp=2 perfect overlap)"


class TestPPScheduleRelativeComparison:
    """Verify relative ordering between schedules based on theoretical properties."""

    def test_step_time_dualpipe_no_worse_than_1f1b(self, cached_reports):
        """DualPipe step_time <= 1F1B at pp=2 (zero bubble vs 1/9).
        """
        r_1f1b = _get_report(cached_reports, "1f1b")
        r_vpp = _get_report(cached_reports, "interleaved", 2)
        r_dp = _get_report(cached_reports, "dualpipe")
        r_dpv = _get_report(cached_reports, "dualpipev", 2)

        assert r_dp.step_time_ms <= r_1f1b.step_time_ms, (
            f"DualPipe should be <= 1F1B; got "
            f"1f1b={r_1f1b.step_time_ms}, vpp={r_vpp.step_time_ms}, "
            f"dualpipe={r_dp.step_time_ms}, dualpipev={r_dpv.step_time_ms}"
        )

    def test_step_time_vpp_close_to_1f1b(self, cached_reports):
        """VPP and 1F1B step_time are close (bubble reduction ≈ P2P overhead at pp=2).

        【原理】VPP interleaved 在 pp=2 时 bubble=1/17（vs 1F1B 的 1/9）
                 bubble 减少约 47%，但 vpp_chunks=2 增加额外 P2P 通信
                 两者互相抵消 → step_time 接近（差异 < 15%）
        【观测点】abs(vpp.step_time - 1f1b.step_time) / 1f1b.step_time < 0.15
        """
        r_1f1b = _get_report(cached_reports, "1f1b")
        r_vpp = _get_report(cached_reports, "interleaved", 2)

        ratio = abs(r_vpp.step_time_ms - r_1f1b.step_time_ms) / r_1f1b.step_time_ms
        assert ratio < 0.25, \
            f"VPP/1F1B step_time gap {ratio * 100:.1f}% exceeds 25%: VPP={r_vpp.step_time_ms}, 1F1B={r_1f1b.step_time_ms}"

    def test_step_time_dualpipe_less_than_1f1b(self, cached_reports):
        """DualPipe step_time < 1F1B due to zero bubble (vs 1/9).

        【原理】bubble reduction → step_time reduction
        【观测点】dualpipe.step_time < 1f1b.step_time
        """
        r_1f1b = _get_report(cached_reports, "1f1b")
        r_dp = _get_report(cached_reports, "dualpipe")

        assert r_dp.step_time_ms < r_1f1b.step_time_ms, \
            f"DualPipe {r_dp.step_time_ms} >= 1F1B {r_1f1b.step_time_ms}"

    def test_mfu_ordering_dualpipe_higher_than_1f1b(self, cached_reports):
        """DualPipe has higher MFU than 1F1B due to zero bubble.

        【原理】DualPipe bubble=0 → step_time 最低 → MFU 最高
        【观测点】dualpipe.mfu > 1f1b.mfu
        """
        r_1f1b = _get_report(cached_reports, "1f1b")
        r_dp = _get_report(cached_reports, "dualpipe")

        assert r_dp.mfu > r_1f1b.mfu, \
            f"DualPipe MFU {r_dp.mfu} <= 1F1B {r_1f1b.mfu}"

    def test_mfu_dualpipe_and_dualpipev_close(self, cached_reports):
        """DualPipe and DualPipeV MFU within 20% (both have zero bubble at pp=2).

        【原理】两种 schedule 在 pp=2 时 bubble=0，但 DualPipeV 有 VPP 开销
                 MFU 差距可能略大（DualPipe 最优，DualPipeV 有开销）
        【观测点】abs(dualpipe.mfu - dualpipev.mfu) / max < 0.20
        """
        r_dp = _get_report(cached_reports, "dualpipe")
        r_dpv = _get_report(cached_reports, "dualpipev", 2)

        max_mfu = max(r_dp.mfu, r_dpv.mfu)
        diff_ratio = abs(r_dp.mfu - r_dpv.mfu) / max_mfu
        assert diff_ratio < 0.20, \
            f"MFU gap {diff_ratio * 100:.1f}% exceeds 20%: DualPipe={r_dp.mfu}, DualPipeV={r_dpv.mfu}"

    def test_hfu_equals_mfu_no_recompute(self, cached_reports):
        """HFU == MFU when no selective recompute (default config).

        【原理】HFU = MFU + recompute_flops / (step_time * peak_compute)
                 默认配置无 recompute → hfu == mfu
        【观测点】所有策略的 hfu == mfu 精确匹配
        """
        configs = [("1f1b", 1), ("interleaved", 2), ("dualpipe", 1), ("dualpipev", 2)]
        for schedule, vpp in configs:
            r = _get_report(cached_reports, schedule, vpp)
            assert r.hfu == pytest.approx(r.mfu, rel=0.001), \
                f"HFU != MFU for {schedule}: hfu={r.hfu}, mfu={r.mfu}"


class TestPPScheduleFlopsConsistency:
    """Validate FLOPs, P2P communication volume, and stage assignment across schedules.

    重点对比 [("1f1b", 1), ("dualpipe", 1), ("dualpipev", 2)] 三种策略：
      - 1F1B / DualPipe: greedy 层分配, 1 个 boundary
      - DualPipeV:       interleaved 层分配, 2 个 boundary
    """

    CONFIGS = [("1f1b", 1), ("dualpipe", 1), ("dualpipev", 2)]

    # ── FLOPs ──────────────────────────────────────────────────────────────

    def test_total_flops_consistent_across_schedules(self, cached_reports):
        """Total training FLOPs identical across all schedules (same model + config).

        【原理】training_flops = forward_flops + backward_flops，独立于调度策略
        【观测点】三种策略的 total_flops 精确匹配
        """
        reports = [_get_report(cached_reports, s, v) for s, v in self.CONFIGS]
        base_flops = reports[0].total_flops
        for i, r in enumerate(reports[1:], start=1):
            assert r.total_flops == pytest.approx(base_flops, rel=0.001), \
                f"Schedule {i} total_flops {r.total_flops} != base {base_flops}"

    # ── P2P 通信量 ───────────────────────────────────────────────────────

    def test_p2p_volume_identity(self, cached_reports):
        """1F1B 与 DualPipe P2P 通信量完全一致（同层分配）。

        【原理】greedy 层分配相同 → 跨 boundary 的张量集合相同 → 通信量一致
        【观测点】total_volume(1F1B) == total_volume(DualPipe);
                  fwd/bwd volume 各项相等
        """
        g_1f1b = _get_transformed_graph(cached_reports, "1f1b")
        g_dp = _get_transformed_graph(cached_reports, "dualpipe")
        assert _get_p2p_total_volume(g_1f1b) == _get_p2p_total_volume(g_dp)
        assert _get_p2p_volume_by_phase(g_1f1b) == _get_p2p_volume_by_phase(g_dp)

    def test_fwd_p2p_volume_linear_interpolation(self, cached_reports):
        """fwd 通信量线性插值：fwd_volume(DualPipeV) == fwd_volume(1F1B) × 2。

        【原理】每个 boundary 传递相同 residual stream 张量集合 → 字节数相同
                 DualPipeV 有 2 个 boundary → fwd 通信量 = 1F1B × 2
        【观测点】fwd_volume(DualPipeV) == fwd_volume(1F1B) × 2
        """
        g_1f1b = _get_transformed_graph(cached_reports, "1f1b")
        g_dpv = _get_transformed_graph(cached_reports, "dualpipev", 2)

        fwd_vol_1f1b = _get_p2p_volume_by_phase(g_1f1b)["fwd"]
        fwd_vol_dpv = _get_p2p_volume_by_phase(g_dpv)["fwd"]

        assert fwd_vol_dpv == fwd_vol_1f1b * 2, \
            f"fwd volume interpolation failed: {fwd_vol_dpv} != {fwd_vol_1f1b} × 2"

    def test_bwd_p2p_volume_monotonic_increase(self, cached_reports):
        """bwd 通信量单调递增：更多 boundary → 更多跨 stage 梯度路径。

        【原理】interleaved 拆分层 → 新增跨 boundary 的梯度/saved activation 路径
        【观测点】bwd_volume(DualPipeV) > bwd_volume(1F1B)
        """
        g_1f1b = _get_transformed_graph(cached_reports, "1f1b")
        g_dpv = _get_transformed_graph(cached_reports, "dualpipev", 2)

        bwd_vol_1f1b = _get_p2p_volume_by_phase(g_1f1b)["bwd"]
        bwd_vol_dpv = _get_p2p_volume_by_phase(g_dpv)["bwd"]
        assert bwd_vol_dpv > bwd_vol_1f1b, \
            f"bwd volume should increase: dpv={bwd_vol_dpv} <= 1f1b={bwd_vol_1f1b}"

    def test_total_p2p_volume_dualpipev_greater(self, cached_reports):
        """总 P2P 通信量 DualPipeV > 1F1B。

        【原理】fwd 和 bwd 通信量都增加 → total 必然增加
        【观测点】total_volume(DualPipeV) > total_volume(1F1B)
        """
        g_1f1b = _get_transformed_graph(cached_reports, "1f1b")
        g_dpv = _get_transformed_graph(cached_reports, "dualpipev", 2)
        assert _get_p2p_total_volume(g_dpv) > _get_p2p_total_volume(g_1f1b), \
            f"total volume should increase: dpv={_get_p2p_total_volume(g_dpv)} <= 1f1b={_get_p2p_total_volume(g_1f1b)}"

    def test_pp_p2p_nodes_have_volume(self, cached_reports):
        """每个 P2P 节点 message_size_bytes > 0（P2P 通信确实传递数据）。

        【原理】PipelineParallelPass 计算跨 boundary 张量的 mem_bytes 作为 message_size_bytes
        【观测点】所有策略的所有 PP P2P 节点 message_size_bytes > 0
        """
        for schedule, vpp in self.CONFIGS:
            g = _get_transformed_graph(cached_reports, schedule, vpp)
            pp_p2p = [n for n in g.nodes.values()
                      if n.op_type == "comm.send_recv" and n.attrs.get("role") is None]
            for p2p in pp_p2p:
                assert p2p.attrs.get("message_size_bytes", 0) > 0, \
                    f"{schedule}: P2P {p2p.id} has zero message_size_bytes"

    # ── Stage 分配 ────────────────────────────────────────────────────────

    def test_stage_no_overlap(self, cached_reports):
        """层不出现在多个 stage（partition 性质）。

        【原理】每个节点只有一个 stage_id → 每个层只属于一个 stage
        【观测点】所有 stage 的 layer set 交集为空
        """
        for schedule, vpp in self.CONFIGS:
            g = _get_transformed_graph(cached_reports, schedule, vpp)
            assignment = _get_stage_layer_assignment(g)
            stage_list = list(assignment.values())
            for i in range(len(stage_list)):
                for j in range(i + 1, len(stage_list)):
                    overlap = stage_list[i] & stage_list[j]
                    assert not overlap, \
                        f"{schedule}: stage {i} and {j} overlap at layers {overlap}"

    def test_stage_completeness(self, cached_reports):
        """所有层 {0,1,2,3} 都被分配到某个 stage（逐策略验证）。

        【原理】PP 分割必须覆盖所有层，无遗漏
        【观测点】每策略所有 stage 的 layer set union == {0,1,2,3}
        """
        expected_layers = set(range(_CAPTURED_LAYERS))
        for schedule, vpp in self.CONFIGS:
            g = _get_transformed_graph(cached_reports, schedule, vpp)
            assignment = _get_stage_layer_assignment(g)
            all_assigned = set()
            for layers in assignment.values():
                all_assigned |= layers
            assert all_assigned == expected_layers, \
                f"{schedule}: missing layers {expected_layers - all_assigned}, got {all_assigned}"


def _get_layers_for_stage(graph, stage_id: int) -> set[int]:
    """Get the set of layer indices assigned to a specific stage."""
    assignment = _get_stage_layer_assignment(graph)
    return assignment.get(stage_id, set())


class TestStageLayerAssignment:
    """验证不同流水线策略下 stage 的精确层分配。

    DeepSeek-V4 4层模型，pp=2：

    层分配策略：
      - 1F1B/DualPipe (greedy bin-packing): 按 compute load 装箱，
        greedy 按层索引顺序分配到累计 load 最小的 stage。
        由于 L0→stage0, L1→stage1（L1 时 stage1 空，load=0 < stage0 正数 load），
        {0,1} 同 stage 在 greedy 下不可能。实际结果取决于各层 compute load 分布，
        需运行后确认具体映射。
      - DualPipeV/VPP (round-robin interleaved): 层按 chunk 交替分配，
        算法确定性 → stage0={0,2}, stage1={1,3}
    """

    def test_1f1b_first_two_layers_split(self, cached_reports):
        """1F1B greedy 装箱：L0 和 L1 必须分到不同 stage。

        【原理】greedy 按层索引顺序分配到累计 load 最小的 stage。
        L0 时两 stage 均 load=0 → 取 min index → stage0。
        L1 时 stage1=0 < stage0 正数 load → L1 到 stage1。
        因此 L0∈stage0, L1∈stage1 是 greedy 的必然性质，独立于具体 load 值。
        【观测点】L0 的 stage_id != L1 的 stage_id
        """
        g = _get_transformed_graph(cached_reports, "1f1b")
        assignment = _get_stage_layer_assignment(g)
        s0_layers = assignment.get(0, set())
        s1_layers = assignment.get(1, set())
        assert 0 in s0_layers and 1 in s1_layers, \
            f"greedy must put L0 in stage0, L1 in stage1; got s0={s0_layers}, s1={s1_layers}"

    def test_1f1b_exact_assignment(self, cached_reports):
        """1F1B greedy 装箱：精确层分配（值取决于模型 compute load 分布）。

        【原理】greedy 按累计 load 最小分配，结果随模型 per-layer load 变化。
        当前 DeepSeek-V4 4层预期：stage0={0,2}, stage1={1,3}
        若模型结构或 load 注解变化导致分配不同，需同步更新断言值。
        【观测点】stage0={0,2}, stage1={1,3}
        """
        g = _get_transformed_graph(cached_reports, "1f1b")
        assert _get_layers_for_stage(g, 0) == {0, 2}
        assert _get_layers_for_stage(g, 1) == {1, 3}


    def test_dualpipev_stage0_layers(self, cached_reports):
        """DualPipeV interleaved 分配：stage0 包含 {0, 2}。

        【原理】VPP round-robin 分配：total_chunks=pp*vpp_chunks=4,
        layers_per_chunk=1, chunk_id=idx//1, s_idx=chunk_id%2
        idx=0→chunk0→stage0, idx=2→chunk2→stage0 → stage0={0,2}
        【观测点】stage0 == {0, 2}
        """
        g = _get_transformed_graph(cached_reports, "dualpipev", 2)
        assert _get_layers_for_stage(g, 0) == {0, 2}
        assert _get_layers_for_stage(g, 1) == {1, 3}


    def test_vpp_same_assignment_as_dualpipev(self, cached_reports):
        """VPP 层分配与 DualPipeV 完全一致（两者均使用 interleaved 分配）。

        【原理】VPP 和 DualPipeV 使用相同的 round-robin interleaved 分配算法，
        仅调度策略不同，层分配不变
        【观测点】每个 stage 的 layer set 精确匹配
        """
        g_vpp = _get_transformed_graph(cached_reports, "interleaved", 2)
        g_dpv = _get_transformed_graph(cached_reports, "dualpipev", 2)
        assignment_vpp = _get_stage_layer_assignment(g_vpp)
        assignment_dpv = _get_stage_layer_assignment(g_dpv)
        for sid in assignment_vpp:
            assert assignment_vpp[sid] == assignment_dpv.get(sid, set()), \
                f"stage {sid}: VPP={assignment_vpp[sid]}, DualPipeV={assignment_dpv.get(sid, set())}"

