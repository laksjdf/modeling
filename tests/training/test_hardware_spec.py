from __future__ import annotations

import pytest

from zrt.hardware.spec import InterconnectSpec, LinkSpec, TopologyTier


def _tier(name: str, num_devices: int) -> TopologyTier:
    return TopologyTier(
        name=name,
        link=LinkSpec(
            type=name,
            bandwidth_gbps=100,
            latency_us=1.0,
            topology="all_to_all" if num_devices > 0 else "fat_tree",
            num_devices=num_devices,
        ),
    )


def test_equal_interconnect_specs_have_equal_hashes():
    a = InterconnectSpec(tiers=[
        TopologyTier(
            name="tray",
            link=LinkSpec(
                type="NVLink",
                bandwidth_gbps=900,
                latency_us=1.0,
                topology="all_to_all",
                num_devices=8,
            ),
        ),
    ])
    b = InterconnectSpec(tiers=[
        TopologyTier(
            name="tray",
            link=LinkSpec(
                type="NVLink",
                bandwidth_gbps=900,
                latency_us=1.0,
                topology="all_to_all",
                num_devices=8,
            ),
        ),
    ])

    assert a == b
    assert hash(a) == hash(b)
    assert {a: "spec"}[b] == "spec"


def test_non_outermost_zero_num_devices_rejected():
    with pytest.raises(ValueError, match="only outermost tier may be unbounded"):
        InterconnectSpec(tiers=[
            _tier("tray", 4),
            _tier("rack", 0),
            _tier("spine", 0),
        ])


def test_outermost_zero_num_devices_allowed():
    interconnect = InterconnectSpec(tiers=[
        _tier("tray", 4),
        _tier("rack", 64),
        _tier("spine", 0),
    ])

    assert interconnect.innermost_tier_for(1024).name == "spine"


def test_legacy_inter_node_zero_num_devices_allowed():
    interconnect = InterconnectSpec(
        intra_node=_tier("intra_node", 8).link,
        inter_node=_tier("inter_node", 0).link,
    )

    assert interconnect.inter_node.num_devices == 0
