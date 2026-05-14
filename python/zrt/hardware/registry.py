"""Hardware registry: load HardwareSpec from YAML config files.

Usage::

    from python.zrt.hardware.registry import load, list_available

    hw = load("ascend_910b")        # by file stem
    hw = load("Ascend 910B")        # by display name (case-insensitive)
    print(list_available())         # ['ascend_910b', 'ascend_910c', ...]
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from python.zrt.hardware.spec import (
    ComputeSpec,
    HardwareSpec,
    InterconnectSpec,
    LinkSpec,
    MemorySpec,
    MemoryTier,
)

_CONFIGS_DIR = Path(__file__).parent / "configs"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load(name: str) -> HardwareSpec:
    """Load a HardwareSpec by config name or display name.

    ``name`` is matched (case-insensitive) against:
    1. YAML file stem  (e.g. ``"ascend_910b"`` → ``ascend_910b.yaml``)
    2. ``name`` field inside the YAML (e.g. ``"Ascend 910B"``)

    Raises ``KeyError`` if no match is found.
    """
    name_lower = name.lower().replace(" ", "_").replace("-", "_")

    # 1. Try exact stem match first (fast path)
    direct = _CONFIGS_DIR / f"{name_lower}.yaml"
    if direct.exists():
        return _load_file(direct)

    # 2. Scan all YAML files for display-name match
    for path in sorted(_CONFIGS_DIR.glob("*.yaml")):
        stem_norm = path.stem.lower().replace("-", "_")
        if stem_norm == name_lower:
            return _load_file(path)
        # peek at the 'name' field without parsing full YAML
        raw = path.read_text(encoding="utf-8")
        for line in raw.splitlines():
            if line.startswith("name:"):
                display = line.split(":", 1)[1].strip().strip('"').strip("'")
                if display.lower().replace(" ", "_").replace("-", "_") == name_lower:
                    return _load_file(path)
                break

    known = list_available()
    raise KeyError(
        f"Hardware {name!r} not found in {_CONFIGS_DIR}. "
        f"Available: {known}"
    )


def list_available() -> list[str]:
    """Return a sorted list of available hardware config stems."""
    return sorted(p.stem for p in _CONFIGS_DIR.glob("*.yaml"))


# ─────────────────────────────────────────────────────────────────────────────
# Internal parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_file(path: Path) -> HardwareSpec:
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return _parse_spec(data)


def _parse_spec(d: dict[str, Any]) -> HardwareSpec:
    return HardwareSpec(
        name=d["name"],
        vendor=d.get("vendor", "unknown"),
        device_type=d.get("device_type", "gpu"),
        compute=_parse_compute(d.get("compute", {})),
        memory=_parse_memory(d.get("memory", {})),
        interconnect=_parse_interconnect(d.get("interconnect", {})),
    )


def _parse_compute(c: dict[str, Any]) -> ComputeSpec:
    cube_raw = c.get("cube_bf16_tflops")
    vector_raw = c.get("vector_bf16_tflops")
    overlap_raw: dict[str, Any] = c.get("overlap_ratio", {})

    return ComputeSpec(
        fp16_tflops=float(c.get("fp16_tflops", 0.0)),
        bf16_tflops=float(c.get("bf16_tflops", 0.0)),
        fp32_tflops=float(c.get("fp32_tflops", 0.0)),
        int8_tops=float(c.get("int8_tops", 0.0)),
        int4_tops=float(c.get("int4_tops", 0.0)),
        fp8_tops=float(c.get("fp8_tops", 0.0)),
        cube_bf16_tflops=float(cube_raw) if cube_raw is not None else None,
        vector_bf16_tflops=float(vector_raw) if vector_raw is not None else None,
        overlap_ratio={k: float(v) for k, v in overlap_raw.items()},
        sram_kb_per_sm=float(c.get("sram_kb_per_sm", 0.0)),
    )


def _parse_memory(m: dict[str, Any]) -> MemorySpec:
    tiers_raw: list[dict] = m.get("tiers", [])
    tiers = [
        MemoryTier(
            name=t["name"],
            bandwidth_gbps=float(t["bandwidth_gbps"]),
            capacity_mb=float(t.get("capacity_mb", 0.0)),
        )
        for t in tiers_raw
    ]
    return MemorySpec(
        capacity_gb=float(m.get("capacity_gb", 0.0)),
        hbm_bandwidth_gbps=float(m.get("hbm_bandwidth_gbps", 0.0)),
        l2_cache_mb=float(m.get("l2_cache_mb", 0.0)),
        tiers=tiers,
    )


def _parse_interconnect(ic: dict[str, Any]) -> InterconnectSpec:
    return InterconnectSpec(
        intra_node=_parse_link(ic.get("intra_node", {})),
        inter_node=_parse_link(ic.get("inter_node", {})),
    )


def _parse_link(lk: dict[str, Any]) -> LinkSpec:
    """Parse a link spec dict into LinkSpec.

    Supports two YAML formats:

    1. ``bandwidth_gbps`` already given (bidirectional total).
    2. ``unidirectional_bw_gbps`` + ``num_links`` → computed as
       ``unidirectional_bw_gbps * num_links * 2`` (bidirectional total).
    """
    if not lk:
        return LinkSpec(type="none", bandwidth_gbps=0.0, latency_us=0.0)

    if "bandwidth_gbps" in lk:
        bw = float(lk["bandwidth_gbps"])
    else:
        uni = float(lk.get("unidirectional_bw_gbps", 0.0))
        n_links = int(lk.get("num_links", 1))
        bw = uni * n_links * 2   # bidirectional total

    return LinkSpec(
        type=lk.get("type", "unknown"),
        bandwidth_gbps=bw,
        latency_us=float(lk.get("latency_us", 0.0)),
        topology=lk.get("topology", "point_to_point"),
        num_devices=int(lk.get("num_devices", 1)),
    )
