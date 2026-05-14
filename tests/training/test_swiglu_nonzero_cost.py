"""Tests for SwiGLU op cost correctness in estimate-config mode.

Root cause of the bug:
  _build_moe_ffn_ops uses m = model.moe_ffn (default = 0).
  When moe_ffn=0 and n_shared_experts>0 (both defaults), shared_swiGLU
  is created with output shape (seq, 0) → num_elements()=0 → bytes_fwd=0
  → _elementwise_cost returns zero FLOPs and zero bytes → op_to_time returns 0.

Why this matters:
  This is a silent accuracy error: the shared expert SwiGLU disappears from
  the cost model, making MoE step-time estimates wrong by the fraction of time
  the shared expert contributes.
"""
from __future__ import annotations

import pytest

from zrt.training.ir.builders import build_graph
from zrt.training.ir.validate import validate
from zrt.training.models.flops import op_cost
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import Strategy
from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.spec.system import GPU, SystemSpec


def _moe_model(moe_ffn: int, n_shared_experts: int = 1) -> ModelSpec:
    return ModelSpec(
        hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
        head_dim=128, vocab=32000, seq_len=2048,
        layers=[LayerKind.MOE],
        moe_ffn=moe_ffn, num_experts=8, top_k=2,
        n_shared_experts=n_shared_experts,
    )


def _strategy(**kw) -> Strategy:
    defaults = dict(tp=1, pp=1, dp=1, micro_batch=1, global_batch=1)
    defaults.update(kw)
    return Strategy(**defaults)


def _system() -> SystemSpec:
    return SystemSpec(
        gpu=GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80, hbm_bw_gbps=3350),
        host_mem_gb=256,
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0,
                               topology="all_to_all", num_devices=8),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=400, latency_us=5.0,
                               topology="fat_tree"),
        ),
        nodes=1, gpus_per_node=8,
    )


class TestMoeFfnZeroRejected:
    def test_build_graph_raises_when_moe_ffn_zero_and_shared_experts(self):
        """build_graph must raise ValueError when moe_ffn=0 and n_shared_experts>0.

        Previously this silently created shared_swiGLU with output (seq, 0)
        producing zero FLOPs and zero latency.
        """
        model = _moe_model(moe_ffn=0, n_shared_experts=1)
        with pytest.raises(ValueError, match="moe_ffn"):
            build_graph(model, _strategy())

    def test_build_graph_raises_when_moe_ffn_zero_multi_shared_experts(self):
        """Same guard applies when n_shared_experts > 1."""
        model = _moe_model(moe_ffn=0, n_shared_experts=2)
        with pytest.raises(ValueError, match="moe_ffn"):
            build_graph(model, _strategy())

    def test_build_graph_ok_when_no_shared_experts(self):
        """moe_ffn=0 is allowed when there are no shared experts (nothing to compute)."""
        model = _moe_model(moe_ffn=0, n_shared_experts=0)
        graph = build_graph(model, _strategy())
        swiglu_ops = [op for op in graph.ops if op.kind == "swiglu"]
        assert swiglu_ops == [], "No SwiGLU expected when n_shared_experts=0"

    def test_validate_warns_moe_ffn_zero_with_shared_experts(self):
        """validate() must warn when moe_ffn=0 and n_shared_experts>0 with MOE layers."""
        model = _moe_model(moe_ffn=0, n_shared_experts=1)
        system = _system()
        strategy = _strategy()
        warnings = validate(model, system, strategy)
        assert any("moe_ffn" in w.lower() for w in warnings), (
            f"Expected moe_ffn warning for moe_ffn=0+n_shared_experts=1, got: {warnings}"
        )

    def test_validate_no_moe_ffn_warning_when_moe_ffn_valid(self):
        """validate() should not warn when moe_ffn>0 (correct config)."""
        model = _moe_model(moe_ffn=2048, n_shared_experts=1)
        system = _system()
        strategy = _strategy()
        warnings = validate(model, system, strategy)
        moe_ffn_warnings = [w for w in warnings if "moe_ffn" in w.lower()]
        assert moe_ffn_warnings == [], (
            f"Unexpected moe_ffn warning for valid config: {moe_ffn_warnings}"
        )


class TestSwiGLUCostNonZero:
    """With valid moe_ffn, shared SwiGLU must have non-zero FLOPs and bytes
    regardless of TP/CP/EP configuration."""

    def _check_swiglu_nonzero(self, model: ModelSpec, strategy: Strategy) -> None:
        system = _system()
        graph = build_graph(model, strategy)
        swiglu_ops = [op for op in graph.ops if op.kind == "swiglu"]
        assert swiglu_ops, "Expected at least one SwiGLU op in MoE model"
        for op in swiglu_ops:
            cost = op_cost(op, model, system)
            assert cost.fwd_vector_flops > 0, (
                f"SwiGLU '{op.name}': fwd_vector_flops=0 "
                f"(shape_local={[t.shape_local for t in op.outputs]})"
            )
            assert cost.fwd_bytes > 0, (
                f"SwiGLU '{op.name}': fwd_bytes=0 "
                f"(shape_local={[t.shape_local for t in op.outputs]}, "
                f"bytes_fwd={op.meta.get('bytes_fwd')})"
            )

    def test_tp1_ep1(self):
        """Baseline: no parallelism."""
        self._check_swiglu_nonzero(_moe_model(moe_ffn=2048), _strategy(tp=1, ep=1))

    def test_tp2_ep1(self):
        """TP=2: SwiGLU cost must remain non-zero after TP sharding."""
        self._check_swiglu_nonzero(_moe_model(moe_ffn=2048), _strategy(tp=2, ep=1))

    def test_tp4_ep1(self):
        """TP=4: SwiGLU cost must remain non-zero."""
        self._check_swiglu_nonzero(_moe_model(moe_ffn=2048), _strategy(tp=4, ep=1))

    def test_tp1_ep4(self):
        """EP=4: SwiGLU cost must remain non-zero after EP sharding."""
        self._check_swiglu_nonzero(_moe_model(moe_ffn=2048), _strategy(tp=1, ep=4))

    def test_tp2_ep4(self):
        """TP=2+EP=4: SwiGLU cost must remain non-zero."""
        self._check_swiglu_nonzero(_moe_model(moe_ffn=2048), _strategy(tp=2, ep=4))

    def test_deepseek_v3_like(self):
        """DeepSeek-V3 realistic: moe_ffn=2048, tp=8, ep=8."""
        model = ModelSpec(
            hidden=7168, ffn=18432, num_heads=128, num_kv_heads=128,
            head_dim=128, vocab=129280, seq_len=4096,
            layers=[LayerKind.MOE],
            moe_ffn=2048, num_experts=256, top_k=8,
            n_shared_experts=1,
        )
        self._check_swiglu_nonzero(model, _strategy(tp=8, ep=8, dp=1, global_batch=8))
