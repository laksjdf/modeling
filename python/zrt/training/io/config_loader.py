"""YAML config loader — parse model + system + strategy from a single YAML file."""

from __future__ import annotations

from pathlib import Path

import yaml

from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import (
    CPKind, OffloadPolicy, OptKind, PPSched, RecomputePolicy, Strategy,
    TPOverlap,
)
from zrt.training.spec.system import GPU, NetTier, SystemSpec


def load_specs(config_path: str | Path) -> tuple[ModelSpec, SystemSpec, Strategy]:
    """Load model + system + strategy from a single YAML file."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model = _parse_model(cfg["model"])
    system = _parse_system(cfg["system"])
    strategy = _parse_strategy(cfg["strategy"])

    return model, system, strategy


def _parse_model(d: dict) -> ModelSpec:
    layers_str = d.get("layers", [])
    layers = _parse_layers(layers_str)

    return ModelSpec(
        hidden=d["hidden"],
        ffn=d["ffn"],
        num_heads=d["num_heads"],
        num_kv_heads=d.get("num_kv_heads", d["num_heads"]),
        head_dim=d.get("head_dim", d["hidden"] // d["num_heads"]),
        vocab=d["vocab"],
        seq_len=d["seq_len"],
        layers=layers,
        num_experts=d.get("num_experts", 0),
        moe_ffn=d.get("moe_ffn", 0),
        top_k=d.get("top_k", 0),
        capacity_factor=d.get("capacity_factor", 1.0),
        expert_imbalance=d.get("expert_imbalance", 0.0),
        mtp_depth=d.get("mtp_depth", 0),
        param_dtype=_parse_dtype(d.get("param_dtype", "bf16")),
        grad_dtype=_parse_dtype(d.get("grad_dtype", "fp32")),
        master_dtype=_parse_dtype(d.get("master_dtype", "fp32")),
        act_dtype=_parse_dtype(d.get("act_dtype", "bf16")),
    )


def _parse_system(d: dict) -> SystemSpec:
    gpu_d = d["gpu"]
    gpu = GPU(
        name=gpu_d["name"],
        flops_bf16=gpu_d["flops_bf16"],
        flops_fp8=gpu_d.get("flops_fp8", gpu_d["flops_bf16"] * 2),
        hbm_gb=gpu_d["hbm_gb"],
        hbm_bw_gbps=gpu_d["hbm_bw_gbps"],
    )

    nets = []
    for nd in d.get("nets", []):
        nets.append(NetTier(
            scope=nd["scope"],
            bw_gbps=nd["bw_gbps"],
            latency_us=nd["latency_us"],
            topology=nd["topology"],
        ))

    return SystemSpec(
        gpu=gpu,
        host_mem_gb=d.get("host_mem_gb", 256),
        nets=nets,
        nodes=d["nodes"],
        gpus_per_node=d["gpus_per_node"],
    )


def _parse_strategy(d: dict) -> Strategy:
    recompute = RecomputePolicy()
    if "recompute" in d:
        rc = d["recompute"]
        per_layer = {}
        for kind_str, tiers in rc.get("per_layer", {}).items():
            per_layer[kind_str] = set(tiers) if isinstance(tiers, list) else {tiers}
        recompute = RecomputePolicy(per_layer=per_layer)

    offload = OffloadPolicy()
    if "offload" in d:
        ol = d["offload"]
        offload = OffloadPolicy(
            opt_state=ol.get("opt_state", False),
            grads=ol.get("grads", False),
            params=ol.get("params", False),
            pct=ol.get("pct", 1.0),
        )

    return Strategy(
        tp=d.get("tp", 1),
        cp=d.get("cp", 1),
        pp=d.get("pp", 1),
        ep=d.get("ep", 1),
        dp=d.get("dp", 1),
        micro_batch=d.get("micro_batch", 1),
        global_batch=d.get("global_batch", 0),
        pp_schedule=PPSched(d.get("pp_schedule", "1f1b")),
        vpp_chunks=d.get("vpp_chunks", 1),
        pp_layer_assignment=d.get("pp_layer_assignment"),
        cp_kind=CPKind(d.get("cp_kind", "none")),
        zero_stage=d.get("zero_stage", 0),
        recompute=recompute,
        offload=offload,
        tp_overlap=TPOverlap(d.get("tp_overlap", "none")),
        ep_overlap=d.get("ep_overlap", False),
        dualbatch=d.get("dualbatch", False),
        dp_overlap_in_bubble=d.get("dp_overlap_in_bubble", True),
        optimizer=OptKind(d.get("optimizer", "adam")),
    )


def _parse_layers(layers_spec) -> list[LayerKind]:
    """Parse layers specification.

    Supports:
      - list of strings: ["dense", "moe", "mtp"]
      - string with repetition: "[dense]*3+[moe]*58+[mtp]"
    """
    if isinstance(layers_spec, list):
        return [LayerKind(s) for s in layers_spec]

    if isinstance(layers_spec, str):
        result = []
        for part in layers_spec.split("+"):
            part = part.strip()
            # Pattern: [kind]*N  (e.g. "[dense]*80")
            if "]*" in part:
                kind_str, count_str = part.split("]*", 1)
                kind_str = kind_str.lstrip("[")
                count = int(count_str)
                result.extend([LayerKind(kind_str)] * count)
            # Pattern: N*[kind]  (e.g. "3*[dense]")
            elif "*[" in part:
                count_str, kind_str = part.split("*[", 1)
                kind_str = kind_str.rstrip("]")
                count = int(count_str) if count_str else 1
                result.extend([LayerKind(kind_str)] * count)
            elif part.startswith("[") and part.endswith("]"):
                result.append(LayerKind(part[1:-1]))
            else:
                result.append(LayerKind(part))
        return result

    return []


def _parse_dtype(s: str) -> Dtype:
    s = s.lower().strip()
    mapping = {
        "fp32": Dtype.FP32, "float32": Dtype.FP32,
        "bf16": Dtype.BF16, "bfloat16": Dtype.BF16,
        "fp16": Dtype.FP16, "float16": Dtype.FP16,
        "fp8": Dtype.FP8, "float8": Dtype.FP8,
    }
    return mapping.get(s, Dtype.BF16)
