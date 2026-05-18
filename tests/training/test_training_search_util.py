"""Tests for training search utility."""
from __future__ import annotations

import pytest

from zrt.training.search.training_search_util import (
    TrainingConfigManager,
    _load_model_spec,
    _make_strategy_from_config,
    _make_system_from_config,
    format_results,
)
from zrt.training.spec.report import TrainingReport
from zrt.training.spec.strategy import (
    CPKind,
    OptKind,
    PPSched,
    RecomputePolicy,
    TPOverlap,
)


class TestLoadModelSpec:
    """Test ModelSpec loading from YAML."""

    def test_load_deepseek_v3_2(self):
        model = _load_model_spec("deepseek_v3_2")

        assert model.hidden == 7168
        assert model.num_heads == 128
        assert model.num_experts == 256
        assert model.seq_len == 4096
        assert len(model.layers) == 61

    def test_load_invalid_model_raises(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            _load_model_spec("nonexistent_model")

        assert "Model config not found" in str(exc_info.value)


class TestMakeSystemFromConfig:
    """Test SystemSpec construction from config dict."""

    def test_default_values(self):
        config = {}
        system = _make_system_from_config(config)

        assert system.nodes == 1
        assert system.gpus_per_node == 8
        assert system.world_size == 8
        assert "H100" in system.gpu.name

    def test_custom_values(self):
        config = {
            "hw": "nvidia_h100_sxm",
            "nodes": 8,
            "gpus_per_node": 8,
            "host_mem_gb": 2048.0,
        }
        system = _make_system_from_config(config)

        assert system.nodes == 8
        assert system.gpus_per_node == 8
        assert system.world_size == 64
        assert system.host_mem_gb == 2048.0


class TestMakeStrategyFromConfig:
    """Test Strategy construction from config dict."""

    def test_default_values(self):
        config = {}
        strategy = _make_strategy_from_config(config)

        assert strategy.tp == 1
        assert strategy.cp == 1
        assert strategy.pp == 1
        assert strategy.ep == 1
        assert strategy.dp == 1
        assert strategy.micro_batch == 1
        assert strategy.global_batch == 0
        assert strategy.pp_schedule == PPSched.ONE_F_ONE_B
        assert strategy.vpp_chunks == 1
        assert strategy.zero_stage == 0
        assert strategy.optimizer == OptKind.ADAM
        assert strategy.tp_overlap == TPOverlap.NONE
        assert strategy.cp_kind == CPKind.NONE
        assert strategy.ep_overlap is False
        assert strategy.dualbatch is False
        assert strategy.recompute.per_layer == {}

    def test_custom_parallel_config(self):
        config = {
            "tp": 8,
            "cp": 2,
            "pp": 4,
            "ep": 64,
            "dp": 2,
            "micro_batch": 4,
            "global_batch": 512,
        }
        strategy = _make_strategy_from_config(config)

        assert strategy.tp == 8
        assert strategy.cp == 2
        assert strategy.pp == 4
        assert strategy.ep == 64
        assert strategy.dp == 2
        assert strategy.micro_batch == 4
        assert strategy.global_batch == 512

    def test_recompute_selective(self):
        config = {"recompute": "selective"}
        strategy = _make_strategy_from_config(config)

        assert strategy.recompute.per_layer == {"moe": {"attn"}, "dense": {"attn"}}

    def test_recompute_full(self):
        config = {"recompute": "full"}
        strategy = _make_strategy_from_config(config)

        assert strategy.recompute.per_layer == {"moe": {"full"}, "dense": {"full"}}

    def test_optimizer_muon(self):
        config = {"optimizer": "muon", "muon_rotation": False}
        strategy = _make_strategy_from_config(config)

        assert strategy.optimizer == OptKind.MUON
        assert strategy.muon_config is not None
        assert strategy.muon_config.rotation is False

    def test_pp_schedule_interleaved_with_vpp(self):
        config = {"pp_schedule": "i1f1b", "vpp_chunks": 4}
        strategy = _make_strategy_from_config(config)

        assert strategy.pp_schedule == PPSched.INTERLEAVED
        assert strategy.vpp_chunks == 4

    def test_pp_schedule_non_interleaved_ignores_vpp(self):
        config = {"pp_schedule": "1f1b", "vpp_chunks": 4}
        strategy = _make_strategy_from_config(config)

        assert strategy.pp_schedule == PPSched.ONE_F_ONE_B
        assert strategy.vpp_chunks == 1

    def test_tp_overlap(self):
        config = {"tp_overlap": "coc"}
        strategy = _make_strategy_from_config(config)

        assert strategy.tp_overlap == TPOverlap.COC

    def test_cp_kind(self):
        config = {"cp_kind": "ulysses"}
        strategy = _make_strategy_from_config(config)

        assert strategy.cp_kind == CPKind.ULYSSES


class TestTrainingConfigManager:
    """Test TrainingConfigManager functionality."""

    def test_generate_static_configs_basic(self):
        manager = TrainingConfigManager(
            param_grid={"world_size": [1], "tp": [1], "cp": [1], "pp": [1], "dp": [1]},
        )
        configs = manager.generate_static_configs()

        assert len(configs) == 1
        assert configs[0]["tp"] == 1

    def test_generate_static_configs_grid_expansion(self):
        manager = TrainingConfigManager(
            param_grid={
                "world_size": [4],
                "tp": [1, 2, 4],
                "cp": [1],
                "pp": [1, 2],
                "dp": [1],
            }
        )
        configs = manager.generate_static_configs()

        valid = [(c["tp"], c["pp"]) for c in configs]
        assert (4, 1) in valid
        assert (2, 2) in valid

    def test_generate_static_configs_filters_invalid_world_size(self):
        manager = TrainingConfigManager(
            param_grid={
                "world_size": [8],
                "tp": [1, 2],
                "cp": [1],
                "pp": [1],
                "dp": [1, 2, 4, 8],
            }
        )
        configs = manager.generate_static_configs()

        for cfg in configs:
            product = cfg["tp"] * cfg["cp"] * cfg["pp"] * cfg["dp"]
            assert product == 8

    def test_generate_static_configs_valid_world_size(self):
        manager = TrainingConfigManager(
            param_grid={
                "world_size": [16],
                "tp": [1, 2, 4],
                "cp": [1, 2],
                "pp": [1, 2],
                "dp": [1, 2],
            }
        )
        configs = manager.generate_static_configs()

        for cfg in configs:
            ws = cfg["world_size"]
            product = cfg["tp"] * cfg["cp"] * cfg["pp"] * cfg["dp"]
            assert ws == product, f"world_size={ws} != tp*cp*pp*dp={product}"

    def test_output_path_generation(self):
        manager = TrainingConfigManager(
            param_grid={"model": ["test_model"]},
        )

        assert "test_model" in manager.output_path

    def test_output_path_with_world_size(self):
        manager = TrainingConfigManager(
            param_grid={"model": ["test_model"], "world_size": [64]},
        )

        assert "ws_64" in manager.output_path


class TestFormatResults:
    """Test format_results function."""

    def test_format_results_basic(self):
        report = TrainingReport(
            step_time_ms=100.0,
            pipeline_time_ms=95.0,
            mfu=0.45,
            hfu=0.50,
            bubble_fraction=0.1,
            tokens_per_sec=1000.0,
        )
        config = {
            "model": "test_model",
            "world_size": 8,
            "tp": 2,
            "cp": 1,
            "pp": 2,
            "ep": 1,
            "dp": 2,
            "zero_stage": 1,
            "pp_schedule": "1f1b",
            "recompute": "selective",
            "optimizer": "adam",
        }

        df = format_results([report], [config])

        assert len(df) == 1
        assert df.iloc[0]["model"] == "test_model"
        assert df.iloc[0]["world_size"] == 8
        assert df.iloc[0]["tp"] == 2
        assert df.iloc[0]["step_time_ms"] == 100.0
        assert df.iloc[0]["mfu"] == 0.45

    def test_format_results_sorted_by_mfu(self):
        reports = [
            TrainingReport(mfu=0.3, step_time_ms=200),
            TrainingReport(mfu=0.5, step_time_ms=100),
            TrainingReport(mfu=0.4, step_time_ms=150),
        ]
        configs = [{"model": f"model_{i}"} for i in range(3)]

        df = format_results(reports, configs)

        assert df.iloc[0]["mfu"] == 0.5
        assert df.iloc[1]["mfu"] == 0.4
        assert df.iloc[2]["mfu"] == 0.3

    def test_format_results_includes_comm_totals(self):
        """验证 format_results 包含各策略的通信总时间字段，且列按策略分组."""
        report = TrainingReport(
            step_time_ms=100.0,
            mfu=0.45,
            tp_exposed_ms=5.0,
            tp_hidden_ms=2.0,
            tp_total_ms=7.0,
            cp_exposed_ms=3.0,
            cp_total_ms=3.0,
            ep_exposed_ms=4.0,
            ep_hidden_ms=1.0,
            ep_total_ms=5.0,
            pp_exposed_ms=2.0,
            pp_total_ms=2.0,
            dp_exposed_ms=6.0,
            dp_hidden_ms=3.0,
            dp_total_ms=9.0,
        )
        config = {"model": "test", "tp": 8, "cp": 2, "ep": 64, "pp": 4}

        df = format_results([report], [config])

        assert "tp_total_ms" in df.columns
        assert "tp_exposed_ms" in df.columns
        assert "cp_total_ms" in df.columns
        assert "cp_exposed_ms" in df.columns
        assert "ep_total_ms" in df.columns
        assert "ep_exposed_ms" in df.columns
        assert "pp_total_ms" in df.columns
        assert "pp_exposed_ms" in df.columns
        assert "dp_total_ms" in df.columns
        assert "dp_exposed_ms" in df.columns
        assert df.iloc[0]["tp_total_ms"] == 7.0
        assert df.iloc[0]["tp_exposed_ms"] == 5.0
        assert df.iloc[0]["cp_total_ms"] == 3.0
        assert df.iloc[0]["cp_exposed_ms"] == 3.0
        assert df.iloc[0]["ep_total_ms"] == 5.0
        assert df.iloc[0]["ep_exposed_ms"] == 4.0
        assert df.iloc[0]["pp_total_ms"] == 2.0
        assert df.iloc[0]["pp_exposed_ms"] == 2.0
        assert df.iloc[0]["dp_total_ms"] == 9.0
        assert df.iloc[0]["dp_exposed_ms"] == 6.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])