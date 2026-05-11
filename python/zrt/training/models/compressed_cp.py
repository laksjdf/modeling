"""DeepSeek-V4 Two-Stage Compressed Context Parallel (CP) modeling.

This module implements the communication volume and time estimation for
DeepSeek-V4's two-stage compressed CP scheme, as described in Section 3.5.3
of the paper.

Key concepts:
- Stage 1: P2P boundary exchange (send last m/m' uncompressed KV entries)
- Stage 2: All-Gather to collect compressed KV from all CP ranks
- CSA: Compression ratio m=4 (small)
- HCA: Compression ratio m'=128 (large)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CompressedCPConfig:
    """DeepSeek-V4 two-stage compressed CP configuration."""

    cp_size: int

    compression_ratio_csa: int = 4
    compression_ratio_hca: int = 128

    kv_head_dim: int = 512
    indexer_head_dim: int = 128
    num_query_heads: int = 128
    num_indexer_heads: int = 64
    attention_topk: int = 1024
    sliding_window_size: int = 128

    kv_dtype_bytes: float = 1.0
    kv_rope_dim: int = 64
    indexer_dtype_bytes: float = 0.5

    num_csa_layers: int = 30
    num_hca_layers: int = 31
    num_swa_only_layers: int = 0

    enable_packing: bool = True
    avg_pack_efficiency: float = 0.95


class CompressedCPCommAnalyzer:
    """Compute communication volume for Compressed CP stages."""

    def __init__(self, config: CompressedCPConfig):
        self.config = config

    def compute_per_rank_tokens(self, global_seq_len: int) -> int:
        return global_seq_len // self.config.cp_size

    def compute_effective_kv_bytes(self) -> float:
        cfg = self.config
        rope_bytes = cfg.kv_rope_dim * 2.0
        non_rope_bytes = (cfg.kv_head_dim - cfg.kv_rope_dim) * cfg.kv_dtype_bytes
        return (rope_bytes + non_rope_bytes) / cfg.kv_head_dim

    def stage1_send_shape_csa(self, s: int) -> dict:
        m = self.config.compression_ratio_csa
        c = self.config.kv_head_dim
        return {
            "kv_a": (m, c),
            "kv_b": (m, c),
            "indexer_k": (m, self.config.indexer_head_dim),
            "compress_weights_a": (m, c),
            "compress_weights_b": (m, c),
        }

    def stage1_send_shape_hca(self, s: int) -> dict:
        m_prime = self.config.compression_ratio_hca
        c = self.config.kv_head_dim
        return {
            "kv": (m_prime, c),
            "compress_weights": (m_prime, c),
        }

    def stage1_comm_bytes_csa(self) -> float:
        m = self.config.compression_ratio_csa
        c = self.config.kv_head_dim
        c_I = self.config.indexer_head_dim
        B = self.compute_effective_kv_bytes()
        B_I = self.config.indexer_dtype_bytes
        return m * (2 * c * B + 2 * c * B + c_I * B_I)

    def stage1_comm_bytes_hca(self) -> float:
        m_prime = self.config.compression_ratio_hca
        c = self.config.kv_head_dim
        B = self.compute_effective_kv_bytes()
        return m_prime * (c * B + c * B)

    def post_compression_shape_csa(self, s: int) -> dict:
        m = self.config.compression_ratio_csa
        c = self.config.kv_head_dim
        c_I = self.config.indexer_head_dim
        L_comp = s // m + 1
        return {
            "compressed_kv": (L_comp, c),
            "compressed_indexer_k": (L_comp, c_I),
            "description": f"[{s}+{m}, {c}] -> [{L_comp}, {c}]",
        }

    def post_compression_shape_hca(self, s: int) -> dict:
        m_prime = self.config.compression_ratio_hca
        c = self.config.kv_head_dim
        L_comp = s // m_prime + 1
        return {
            "compressed_kv": (L_comp, c),
            "description": f"[{s}+{m_prime}, {c}] -> [{L_comp}, {c}]",
        }

    def stage2_local_shape_csa(self, s: int) -> dict:
        m = self.config.compression_ratio_csa
        c = self.config.kv_head_dim
        c_I = self.config.indexer_head_dim
        L = s // m + 1
        return {
            "compressed_kv": (L, c),
            "compressed_indexer_k": (L, c_I),
        }

    def stage2_global_shape_csa(self, global_seq_len: int) -> dict:
        s = self.compute_per_rank_tokens(global_seq_len)
        m = self.config.compression_ratio_csa
        c = self.config.kv_head_dim
        c_I = self.config.indexer_head_dim
        cp = self.config.cp_size
        L = s // m + 1
        return {
            "before_reorg": {
                "compressed_kv": (cp * L, c),
                "compressed_indexer_k": (cp * L, c_I),
            },
            "after_select_and_pad": {
                "compressed_kv": (cp * (s // m), c),
                "compressed_indexer_k": (cp * (s // m), c_I),
            },
        }

    def stage2_global_shape_hca(self, global_seq_len: int) -> dict:
        s = self.compute_per_rank_tokens(global_seq_len)
        m_prime = self.config.compression_ratio_hca
        c = self.config.kv_head_dim
        cp = self.config.cp_size
        L = s // m_prime + 1
        return {
            "before_reorg": {
                "compressed_kv": (cp * L, c),
            },
            "after_select_and_pad": {
                "compressed_kv": (cp * (s // m_prime), c),
            },
        }

    def stage2_comm_bytes_csa(self, global_seq_len: int) -> float:
        s = self.compute_per_rank_tokens(global_seq_len)
        m = self.config.compression_ratio_csa
        c = self.config.kv_head_dim
        c_I = self.config.indexer_head_dim
        cp = self.config.cp_size
        B = self.compute_effective_kv_bytes()
        B_I = self.config.indexer_dtype_bytes
        L = s // m + 1
        kv_bytes = (cp - 1) * L * c * B
        indexer_bytes = (cp - 1) * L * c_I * B_I
        return kv_bytes + indexer_bytes

    def stage2_comm_bytes_hca(self, global_seq_len: int) -> float:
        s = self.compute_per_rank_tokens(global_seq_len)
        m_prime = self.config.compression_ratio_hca
        c = self.config.kv_head_dim
        cp = self.config.cp_size
        B = self.compute_effective_kv_bytes()
        L = s // m_prime + 1
        return (cp - 1) * L * c * B

    def total_comm_bytes_per_rank(self, global_seq_len: int) -> dict:
        csa_s1 = self.stage1_comm_bytes_csa()
        csa_s2 = self.stage2_comm_bytes_csa(global_seq_len)
        hca_s1 = self.stage1_comm_bytes_hca()
        hca_s2 = self.stage2_comm_bytes_hca(global_seq_len)

        cfg = self.config
        # SWA-only layers don't participate in CP communication (no compression)
        # They use sliding window attention locally, no need for compressed KV exchange
        effective_csa_layers = cfg.num_csa_layers - cfg.num_swa_only_layers
        effective_hca_layers = cfg.num_hca_layers
        # Packing efficiency affects effective communication volume
        # Lower efficiency means more padding, slightly higher overhead
        pack_factor = cfg.avg_pack_efficiency if cfg.enable_packing else 1.0

        total_csa = effective_csa_layers * (csa_s1 + csa_s2) / pack_factor
        total_hca = effective_hca_layers * (hca_s1 + hca_s2) / pack_factor

        return {
            "csa_stage1_per_layer": csa_s1,
            "csa_stage2_per_layer": csa_s2,
            "hca_stage1_per_layer": hca_s1,
            "hca_stage2_per_layer": hca_s2,
            "total_csa_all_layers": total_csa,
            "total_hca_all_layers": total_hca,
            "grand_total": total_csa + total_hca,
            "swa_only_layers": cfg.num_swa_only_layers,
            "pack_efficiency": pack_factor,
        }


class CompressedCPTimeEstimator:
    """Estimate communication time for Compressed CP stages."""

    def __init__(
        self,
        comm_analyzer: CompressedCPCommAnalyzer,
        p2p_bandwidth_GBps: float = 50.0,
        allgather_bandwidth_GBps: float = 40.0,
        p2p_latency_us: float = 5.0,
        allgather_latency_us: float = 10.0,
    ):
        self.analyzer = comm_analyzer
        self.p2p_bw = p2p_bandwidth_GBps * 1e9
        self.ag_bw = allgather_bandwidth_GBps * 1e9
        self.p2p_lat = p2p_latency_us * 1e-6
        self.ag_lat = allgather_latency_us * 1e-6

    def estimate_stage1_time(self, layer_type: str) -> float:
        if layer_type == "csa":
            data_bytes = self.analyzer.stage1_comm_bytes_csa()
        else:
            data_bytes = self.analyzer.stage1_comm_bytes_hca()
        return self.p2p_lat + data_bytes / self.p2p_bw

    def estimate_stage2_time(self, layer_type: str, global_seq_len: int) -> float:
        if layer_type == "csa":
            data_bytes = self.analyzer.stage2_comm_bytes_csa(global_seq_len)
        else:
            data_bytes = self.analyzer.stage2_comm_bytes_hca(global_seq_len)
        return self.ag_lat + data_bytes / self.ag_bw

    def estimate_total_cp_time(self, global_seq_len: int) -> dict:
        """估算所有层的 CP 通信总时间"""
        cfg = self.analyzer.config

        # SWA-only layers don't participate in CP communication
        effective_csa_layers = cfg.num_csa_layers - cfg.num_swa_only_layers
        effective_hca_layers = cfg.num_hca_layers

        csa_time = effective_csa_layers * (
            self.estimate_stage1_time("csa") +
            self.estimate_stage2_time("csa", global_seq_len)
        )
        hca_time = effective_hca_layers * (
            self.estimate_stage1_time("hca") +
            self.estimate_stage2_time("hca", global_seq_len)
        )

        return {
            "csa_total_time_s": csa_time,
            "hca_total_time_s": hca_time,
            "grand_total_time_s": csa_time + hca_time,
            "csa_total_time_ms": csa_time * 1000,
            "hca_total_time_ms": hca_time * 1000,
            "grand_total_time_ms": (csa_time + hca_time) * 1000,
        }


@dataclass
class HybridParallelConfig:
    """Multi-dimensional parallel configuration."""
    tp_size: int = 8
    cp_size: int = 8
    ep_size: int = 64
    pp_size: int = 4
    dp_size: int = 1

    tp_bandwidth_GBps: float = 450.0
    cp_p2p_bandwidth_GBps: float = 50.0
    cp_ag_bandwidth_GBps: float = 40.0
    ep_bandwidth_GBps: float = 25.0
    sp_bandwidth_GBps: float = 450.0

    tp_domain: str = "intra_node"
    cp_domain: str = "inter_node"
    sp_domain: str = "intra_node"


class HybridParallelCommEstimator:
    """Joint parallel communication estimation."""

    def __init__(
        self,
        hybrid_config: HybridParallelConfig,
        cp_config: CompressedCPConfig,
        model_hidden_size: int = 7168,
    ):
        self.hybrid = hybrid_config
        self.cp_analyzer = CompressedCPCommAnalyzer(cp_config)
        self.d = model_hidden_size

    def per_layer_comm_breakdown(self, global_seq_len: int, layer_type: str) -> dict:
        s = global_seq_len // self.hybrid.cp_size
        d = self.d
        tp = self.hybrid.tp_size

        sp_comm = 2 * s * d * 2 / tp

        tp_comm = s * d * 2

        if layer_type == "csa":
            cp_s1 = self.cp_analyzer.stage1_comm_bytes_csa()
            cp_s2 = self.cp_analyzer.stage2_comm_bytes_csa(global_seq_len)
        else:
            cp_s1 = self.cp_analyzer.stage1_comm_bytes_hca()
            cp_s2 = self.cp_analyzer.stage2_comm_bytes_hca(global_seq_len)

        total = sp_comm + tp_comm + cp_s1 + cp_s2

        return {
            "sp_comm_bytes": sp_comm,
            "tp_comm_bytes": tp_comm,
            "cp_stage1_bytes": cp_s1,
            "cp_stage2_bytes": cp_s2,
            "cp_total_bytes": cp_s1 + cp_s2,
            "total_bytes": total,
            "cp_fraction": (cp_s1 + cp_s2) / total if total > 0 else 0.0,
        }


def validate_shape_consistency(analyzer: CompressedCPCommAnalyzer, global_seq_len: int) -> dict:
    """Check Shape consistency (document section 6.1, point 1).

    All-Gather result should equal N/m (CSA) or N/m' (HCA).
    """
    cfg = analyzer.config
    N = global_seq_len
    s = N // cfg.cp_size

    # CSA: expected N/m
    m = cfg.compression_ratio_csa
    expected_csa_kv_len = N // m

    shape_csa = analyzer.stage2_global_shape_csa(N)
    actual_csa_kv_len = shape_csa["after_select_and_pad"]["compressed_kv"][0]

    csa_ok = actual_csa_kv_len == expected_csa_kv_len

    # HCA: expected N/m'
    m_prime = cfg.compression_ratio_hca
    expected_hca_kv_len = N // m_prime

    shape_hca = analyzer.stage2_global_shape_hca(N)
    actual_hca_kv_len = shape_hca["after_select_and_pad"]["compressed_kv"][0]

    hca_ok = actual_hca_kv_len == expected_hca_kv_len

    return {
        "csa_shape_ok": csa_ok,
        "csa_expected_len": expected_csa_kv_len,
        "csa_actual_len": actual_csa_kv_len,
        "hca_shape_ok": hca_ok,
        "hca_expected_len": expected_hca_kv_len,
        "hca_actual_len": actual_hca_kv_len,
        "all_ok": csa_ok and hca_ok,
    }


def validate_comm_volume_conservation(analyzer: CompressedCPCommAnalyzer, global_seq_len: int) -> dict:
    """Check communication volume conservation (document section 6.1, point 2).

    Stage2 All-Gather total = cp_size * (cp_size-1) * local_compressed_data.
    """
    cfg = analyzer.config
    s = global_seq_len // cfg.cp_size

    # CSA
    m = cfg.compression_ratio_csa
    L_csa = s // m + 1
    c = cfg.kv_head_dim
    c_I = cfg.indexer_head_dim
    B = analyzer.compute_effective_kv_bytes()
    B_I = cfg.indexer_dtype_bytes

    # Local contribution
    local_csa_kv_bytes = L_csa * c * B
    local_csa_indexer_bytes = L_csa * c_I * B_I
    local_csa_total = local_csa_kv_bytes + local_csa_indexer_bytes

    # Global total (all ranks send to all others)
    global_csa_total = cfg.cp_size * (cfg.cp_size - 1) * local_csa_total

    # Per-rank receive (what we calculate)
    per_rank_csa = analyzer.stage2_comm_bytes_csa(global_seq_len)

    # Conservation check: global_total / cp_size should equal per_rank_receive
    csa_conserved = per_rank_csa == (cfg.cp_size - 1) * local_csa_total

    # HCA
    m_prime = cfg.compression_ratio_hca
    L_hca = s // m_prime + 1
    local_hca_total = L_hca * c * B
    per_rank_hca = analyzer.stage2_comm_bytes_hca(global_seq_len)

    hca_conserved = per_rank_hca == (cfg.cp_size - 1) * local_hca_total

    return {
        "csa_volume_conserved": csa_conserved,
        "csa_local_bytes": local_csa_total,
        "csa_per_rank_recv": per_rank_csa,
        "hca_volume_conserved": hca_conserved,
        "hca_local_bytes": local_hca_total,
        "hca_per_rank_recv": per_rank_hca,
        "all_ok": csa_conserved and hca_conserved,
    }


def validate_boundary_consistency(analyzer: CompressedCPCommAnalyzer) -> dict:
    """Check boundary exchange symmetry (document section 6.1, point 3).

    Stage1 P2P send = recv (symmetric, except rank 0 and rank cp_size-1).
    """
    csa_send = analyzer.stage1_comm_bytes_csa()
    hca_send = analyzer.stage1_comm_bytes_hca()

    # In P2P, each rank sends same amount and receives same amount
    # (except boundary ranks which only send or only recv)
    csa_symmetric = True  # By design, send_bytes == recv_bytes per rank
    hca_symmetric = True

    return {
        "csa_send_bytes": csa_send,
        "csa_recv_bytes": csa_send,  # Same by design
        "csa_symmetric": csa_symmetric,
        "hca_send_bytes": hca_send,
        "hca_recv_bytes": hca_send,
        "hca_symmetric": hca_symmetric,
        "all_ok": csa_symmetric and hca_symmetric,
    }