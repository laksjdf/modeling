"""Test Compressed-CP integration with IR."""

import pytest
from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.strategy import Strategy, CPKind
from zrt.training.ir.builders import build_graph
from zrt.training.ir.shard import insert_collectives


class TestCompressedCPIntegration:
    """Tests for Compressed-CP integration in IR."""

    def test_compressed_cp_inserts_communication(self):
        """Compressed-CP should insert P2P+AG collectives."""
        model = ModelSpec(
            hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
            head_dim=128, vocab=32000, seq_len=2048,
            layers=[LayerKind.DENSE] * 2,
            num_csa_layers=2,
            num_hca_layers=0,
        )
        strategy = Strategy(tp=2, cp=4, pp=1, cp_kind=CPKind.COMPRESSED)
        graph = build_graph(model, strategy)
        
        cp_colls = [c for c in graph.collectives if c.group == 'CP']
        # Each CSA layer has 4 collectives: (Stage1 P2P + Stage2 AG) × (fwd + bwd)
        assert len(cp_colls) == 8  # 2 layers × 4 collectives
        
        # Check communication kinds
        kinds = [c.kind for c in cp_colls]
        assert 'P2P' in kinds
        assert 'AG' in kinds
        
        # Check phases
        fwd_count = len([c for c in cp_colls if c.phase == 'fwd'])
        bwd_count = len([c for c in cp_colls if c.phase == 'bwd'])
        assert fwd_count == 4  # 2 layers × 2 fwd collectives
        assert bwd_count == 4  # 2 layers × 2 bwd collectives

    def test_swa_only_layers_no_communication(self):
        """SWA-only layers should not participate in CP communication."""
        model = ModelSpec(
            hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
            head_dim=128, vocab=32000, seq_len=2048,
            layers=[LayerKind.DENSE] * 5,
            num_swa_only_layers=2,  # First 2 layers are SWA-only
            num_csa_layers=3,  # Last 3 layers are CSA
            num_hca_layers=0,
        )
        strategy = Strategy(tp=2, cp=4, pp=1, cp_kind=CPKind.COMPRESSED)
        graph = build_graph(model, strategy)
        
        cp_colls = [c for c in graph.collectives if c.group == 'CP']
        # Only CSA layers (3) should have communication, each with 4 collectives
        assert len(cp_colls) == 12  # 3 CSA layers × 4 collectives
        
        # Check that SWA layers have no communication
        for i in range(2):
            swa_comm = [c for c in cp_colls if f'L{i}' in c.name]
            assert len(swa_comm) == 0

    def test_csa_hca_layer_distribution(self):
        """CSA and HCA layers should have correct communication."""
        model = ModelSpec(
            hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
            head_dim=128, vocab=32000, seq_len=2048,
            layers=[LayerKind.DENSE] * 5,
            num_swa_only_layers=0,
            num_csa_layers=2,  # Layers 0-1
            num_hca_layers=3,  # Layers 2-4
        )
        strategy = Strategy(tp=2, cp=4, pp=1, cp_kind=CPKind.COMPRESSED)
        graph = build_graph(model, strategy)
        
        cp_colls = [c for c in graph.collectives if c.group == 'CP']
        # All layers have communication, each with 4 collectives
        assert len(cp_colls) == 20  # 5 layers × 4 collectives
        
        # Check CSA layers (0-1)
        for i in range(2):
            csa_comm = [c for c in cp_colls if f'L{i}' in c.name]
            assert len(csa_comm) == 4
            assert any('csa' in c.name for c in csa_comm)
        
        # Check HCA layers (2-4)
        for i in range(2, 5):
            hca_comm = [c for c in cp_colls if f'L{i}' in c.name]
            assert len(hca_comm) == 4
            assert any('hca' in c.name for c in hca_comm)

    def test_compressed_cp_communication_bytes(self):
        """Compressed-CP communication bytes should match analyzer."""
        from zrt.training.models.compressed_cp import (
            CompressedCPConfig,
            CompressedCPCommAnalyzer,
        )
        
        seq_len = 2048
        model = ModelSpec(
            hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
            head_dim=128, vocab=32000, seq_len=seq_len,
            layers=[LayerKind.DENSE] * 2,
            num_csa_layers=2,
            num_hca_layers=0,
        )
        strategy = Strategy(tp=2, cp=4, pp=1, cp_kind=CPKind.COMPRESSED)
        graph = build_graph(model, strategy)
        
        cp_colls = [c for c in graph.collectives if c.group == 'CP']
        
        # Get expected bytes from analyzer
        cp_config = CompressedCPConfig(cp_size=4, kv_head_dim=model.head_dim)
        analyzer = CompressedCPCommAnalyzer(cp_config)
        
        expected_stage1 = analyzer.stage1_comm_bytes_csa()
        expected_stage2 = analyzer.stage2_comm_bytes_csa(seq_len)
        
        # Check actual bytes in collectives
        p2p_colls = [c for c in cp_colls if c.kind == 'P2P']
        ag_colls = [c for c in cp_colls if c.kind == 'AG']
        
        assert len(p2p_colls) == 4  # 2 CSA layers × 2 phases (fwd+bwd)
        assert len(ag_colls) == 4  # 2 CSA layers × 2 phases
        
        # Each should match expected bytes
        for p2p in p2p_colls:
            assert p2p.bytes_ == pytest.approx(expected_stage1, rel=0.1)
        
        for ag in ag_colls:
            assert ag.bytes_ == pytest.approx(expected_stage2, rel=0.1)

    def test_get_layer_cp_type_method(self):
        """ModelSpec.get_layer_cp_type should return correct type."""
        model = ModelSpec(
            hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
            head_dim=128, vocab=32000, seq_len=2048,
            layers=[LayerKind.DENSE] * 10,
            num_swa_only_layers=3,
            num_csa_layers=4,
            num_hca_layers=3,
        )
        
        # SWA layers (0-2)
        for i in range(3):
            assert model.get_layer_cp_type(i) == 'swa'
        
        # CSA layers (3-6)
        for i in range(3, 7):
            assert model.get_layer_cp_type(i) == 'csa'
        
        # HCA layers (7-9)
        for i in range(7, 10):
            assert model.get_layer_cp_type(i) == 'hca'

    def test_compressed_cp_phase_forward_backward(self):
        """Compressed-CP collectives should have both fwd and bwd phases."""
        model = ModelSpec(
            hidden=4096, ffn=16384, num_heads=32, num_kv_heads=32,
            head_dim=128, vocab=32000, seq_len=2048,
            layers=[LayerKind.DENSE] * 2,
            num_csa_layers=2,
            num_hca_layers=0,
        )
        strategy = Strategy(tp=2, cp=4, pp=1, cp_kind=CPKind.COMPRESSED)
        graph = build_graph(model, strategy)
        
        cp_colls = [c for c in graph.collectives if c.group == 'CP']
        
        # Check that both fwd and bwd phases exist
        fwd_colls = [c for c in cp_colls if c.phase == 'fwd']
        bwd_colls = [c for c in cp_colls if c.phase == 'bwd']
        
        assert len(fwd_colls) == 4  # 2 layers × 2 collectives (P2P+AG)
        assert len(bwd_colls) == 4  # 2 layers × 2 collectives
        
        # Check that each layer has both fwd and bwd communications
        for i in range(2):
            layer_colls = [c for c in cp_colls if f'L{i}' in c.name]
            fwd_layer = [c for c in layer_colls if c.phase == 'fwd']
            bwd_layer = [c for c in layer_colls if c.phase == 'bwd']
            assert len(fwd_layer) == 2
            assert len(bwd_layer) == 2