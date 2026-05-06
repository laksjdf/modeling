"""Tests for DeepSeek-V4 two-stage compressed CP modeling."""

import pytest

from zrt.training.models.compressed_cp import (
    CompressedCPConfig,
    CompressedCPCommAnalyzer,
    CompressedCPTimeEstimator,
    HybridParallelConfig,
    HybridParallelCommEstimator,
    validate_shape_consistency,
    validate_comm_volume_conservation,
    validate_boundary_consistency,
)
from zrt.training.spec.strategy import CPKind


class TestCompressedCPConfig:
    def test_default_config(self):
        config = CompressedCPConfig(cp_size=8)
        assert config.cp_size == 8
        assert config.compression_ratio_csa == 4
        assert config.compression_ratio_hca == 128
        assert config.kv_head_dim == 512
        assert config.indexer_head_dim == 128

    def test_custom_config(self):
        config = CompressedCPConfig(
            cp_size=16,
            compression_ratio_csa=8,
            compression_ratio_hca=64,
            num_csa_layers=40,
            num_hca_layers=20,
        )
        assert config.cp_size == 16
        assert config.compression_ratio_csa == 8
        assert config.compression_ratio_hca == 64


class TestCompressedCPCommAnalyzer:
    @pytest.fixture
    def config(self):
        return CompressedCPConfig(cp_size=8)

    @pytest.fixture
    def analyzer(self, config):
        return CompressedCPCommAnalyzer(config)

    def test_compute_per_rank_tokens(self, analyzer):
        s = analyzer.compute_per_rank_tokens(1_048_576)
        assert s == 131_072

    def test_compute_effective_kv_bytes(self, analyzer):
        B = analyzer.compute_effective_kv_bytes()
        # RoPE: 64 * 2 = 128 bytes, non-RoPE: (512-64) * 1 = 448 bytes
        # weighted: (128 + 448) / 512 = 1.125
        assert B == pytest.approx(1.125, rel=0.01)

    def test_stage1_comm_bytes_csa(self, analyzer):
        bytes_sent = analyzer.stage1_comm_bytes_csa()
        # m=4, c=512, c_I=128, B=1.125, B_I=0.5
        # 4 * (2*512*1.125 + 2*512*1.125 + 128*0.5) = 4 * (1152 + 1152 + 64) = 4 * 2368 = 9472
        assert bytes_sent > 0
        assert bytes_sent == pytest.approx(9472, rel=0.1)

    def test_stage1_comm_bytes_hca(self, analyzer):
        bytes_sent = analyzer.stage1_comm_bytes_hca()
        # m'=128, c=512, B=1.125
        # 128 * (512*1.125 + 512*1.125) = 128 * 1152 = 147456
        assert bytes_sent > 0

    def test_stage2_comm_bytes_csa(self, analyzer):
        global_seq_len = 1_048_576
        bytes_recv = analyzer.stage2_comm_bytes_csa(global_seq_len)
        # s=131072, m=4, L=32769, cp=8, c=512, c_I=128, B=1.125, B_I=0.5
        # kv_bytes = (8-1) * 32769 * 512 * 1.125 ≈ 7 * 18.8M ≈ 132M
        assert bytes_recv > 1e6

    def test_stage2_comm_bytes_hca(self, analyzer):
        global_seq_len = 1_048_576
        bytes_recv = analyzer.stage2_comm_bytes_hca(global_seq_len)
        # m'=128, L=1025, much smaller
        assert bytes_recv > 0
        # HCA should be ~36x smaller than CSA
        csa_bytes = analyzer.stage2_comm_bytes_csa(global_seq_len)
        assert bytes_recv < csa_bytes / 30

    def test_total_comm_bytes_per_rank(self, analyzer):
        total = analyzer.total_comm_bytes_per_rank(1_048_576)
        assert "csa_stage1_per_layer" in total
        assert "csa_stage2_per_layer" in total
        assert "hca_stage1_per_layer" in total
        assert "hca_stage2_per_layer" in total
        assert "grand_total" in total
        assert total["grand_total"] > 0

    def test_post_compression_shape_csa(self, analyzer):
        s = 131072
        shape = analyzer.post_compression_shape_csa(s)
        m = analyzer.config.compression_ratio_csa
        expected_L = s // m + 1
        assert shape["compressed_kv"] == (expected_L, 512)
        assert shape["compressed_indexer_k"] == (expected_L, 128)

    def test_post_compression_shape_hca(self, analyzer):
        s = 131072
        shape = analyzer.post_compression_shape_hca(s)
        m_prime = analyzer.config.compression_ratio_hca
        expected_L = s // m_prime + 1
        assert shape["compressed_kv"] == (expected_L, 512)

    def test_stage2_global_shape(self, analyzer):
        global_seq_len = 1_048_576
        shape_csa = analyzer.stage2_global_shape_csa(global_seq_len)
        shape_hca = analyzer.stage2_global_shape_hca(global_seq_len)

        assert "before_reorg" in shape_csa
        assert "after_select_and_pad" in shape_csa
        assert "before_reorg" in shape_hca
        assert "after_select_and_pad" in shape_hca


class TestCompressedCPTimeEstimator:
    @pytest.fixture
    def estimator(self):
        config = CompressedCPConfig(cp_size=8)
        analyzer = CompressedCPCommAnalyzer(config)
        return CompressedCPTimeEstimator(
            analyzer,
            p2p_bandwidth_GBps=50.0,
            allgather_bandwidth_GBps=40.0,
            p2p_latency_us=5.0,
            allgather_latency_us=10.0,
        )

    def test_estimate_stage1_time(self, estimator):
        csa_time = estimator.estimate_stage1_time("csa")
        hca_time = estimator.estimate_stage1_time("hca")
        assert csa_time > 0
        assert hca_time > 0
        # HCA stage1 has larger data (m'=128 vs m=4)
        assert hca_time > csa_time

    def test_estimate_stage2_time(self, estimator):
        global_seq_len = 1_048_576
        csa_time = estimator.estimate_stage2_time("csa", global_seq_len)
        hca_time = estimator.estimate_stage2_time("hca", global_seq_len)
        assert csa_time > 0
        assert hca_time > 0
        # CSA stage2 dominates (larger data due to smaller compression ratio)
        assert csa_time > hca_time

    def test_estimate_total_cp_time(self, estimator):
        global_seq_len = 1_048_576
        total = estimator.estimate_total_cp_time(global_seq_len)
        assert total["csa_total_time_s"] > 0
        assert total["hca_total_time_s"] > 0
        assert total["grand_total_time_s"] > 0
        assert total["grand_total_time_ms"] == total["grand_total_time_s"] * 1000


class TestHybridParallelCommEstimator:
    @pytest.fixture
    def estimator(self):
        hybrid = HybridParallelConfig(
            tp_size=8,
            cp_size=8,
            ep_size=64,
            pp_size=4,
        )
        cp = CompressedCPConfig(cp_size=8)
        return HybridParallelCommEstimator(hybrid, cp, model_hidden_size=7168)

    def test_per_layer_comm_breakdown_csa(self, estimator):
        breakdown = estimator.per_layer_comm_breakdown(1_048_576, "csa")
        assert breakdown["sp_comm_bytes"] > 0
        assert breakdown["tp_comm_bytes"] > 0
        assert breakdown["cp_stage1_bytes"] > 0
        assert breakdown["cp_stage2_bytes"] > 0
        assert breakdown["total_bytes"] > 0
        assert 0 <= breakdown["cp_fraction"] <= 1

    def test_per_layer_comm_breakdown_hca(self, estimator):
        breakdown = estimator.per_layer_comm_breakdown(1_048_576, "hca")
        assert breakdown["cp_total_bytes"] > 0
        # HCA should have smaller CP fraction than CSA
        csa_breakdown = estimator.per_layer_comm_breakdown(1_048_576, "csa")
        assert breakdown["cp_total_bytes"] < csa_breakdown["cp_total_bytes"]


class TestCPKindEnum:
    def test_compressed_kind_exists(self):
        assert CPKind.COMPRESSED.value == "compressed"
        assert CPKind.COMPRESSED in CPKind

    def test_all_cp_kinds(self):
        kinds = list(CPKind)
        assert len(kinds) == 5
        assert CPKind.NONE in kinds
        assert CPKind.ULYSSES in kinds
        assert CPKind.RING in kinds
        assert CPKind.HYBRID in kinds
        assert CPKind.COMPRESSED in kinds


class TestValidationFunctions:
    @pytest.fixture
    def analyzer(self):
        config = CompressedCPConfig(cp_size=8)
        return CompressedCPCommAnalyzer(config)

    def test_validate_shape_consistency(self, analyzer):
        result = validate_shape_consistency(analyzer, 1_048_576)
        assert result["csa_shape_ok"] is True
        assert result["hca_shape_ok"] is True
        assert result["all_ok"] is True
        # CSA: N/m = 1M/4 = 256K
        assert result["csa_expected_len"] == 262144
        # HCA: N/m' = 1M/128 = 8K
        assert result["hca_expected_len"] == 8192

    def test_validate_comm_volume_conservation(self, analyzer):
        result = validate_comm_volume_conservation(analyzer, 1_048_576)
        assert result["csa_volume_conserved"] is True
        assert result["hca_volume_conserved"] is True
        assert result["all_ok"] is True

    def test_validate_boundary_consistency(self, analyzer):
        result = validate_boundary_consistency(analyzer)
        assert result["csa_symmetric"] is True
        assert result["hca_symmetric"] is True
        assert result["all_ok"] is True

    def test_total_comm_with_packing_efficiency(self, analyzer):
        # Default packing efficiency = 0.95
        result = analyzer.total_comm_bytes_per_rank(1_048_576)
        assert result["pack_efficiency"] == 0.95
        assert "swa_only_layers" in result
        # With packing, total should be slightly higher (divided by efficiency < 1)
        assert result["grand_total"] > 0

    def test_swa_only_layers_excluded(self):
        # Config with SWA-only layers
        config = CompressedCPConfig(
            cp_size=8,
            num_csa_layers=30,
            num_hca_layers=31,
            num_swa_only_layers=3,  # First 3 layers are SWA-only
        )
        analyzer = CompressedCPCommAnalyzer(config)
        result = analyzer.total_comm_bytes_per_rank(1_048_576)

        # SWA-only layers don't participate in CP communication
        # Effective CSA layers = 30 - 3 = 27
        expected_csa_layers = 30 - 3
        assert result["swa_only_layers"] == 3

        # Verify the total is reduced by excluding SWA-only layers
        config_no_swa = CompressedCPConfig(
            cp_size=8,
            num_csa_layers=30,
            num_hca_layers=31,
            num_swa_only_layers=0,
        )
        analyzer_no_swa = CompressedCPCommAnalyzer(config_no_swa)
        result_no_swa = analyzer_no_swa.total_comm_bytes_per_rank(1_048_576)

        # With SWA-only layers excluded, CSA total should be (27/30) of full CSA
        ratio = result["total_csa_all_layers"] / result_no_swa["total_csa_all_layers"]
        expected_ratio = expected_csa_layers / 30
        assert ratio == pytest.approx(expected_ratio, rel=0.01)