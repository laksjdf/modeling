"""YAML config loader — parse model + system + strategy from a single YAML file."""

from __future__ import annotations

from pathlib import Path

import yaml

from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import (
    CPKind, MuonConfig, OffloadPolicy, OptKind, PPSched, RecomputePolicy, Strategy,
    TPOverlap,
)
from zrt.training.spec.system import GPU, SystemSpec

_MODELS_DIR = Path(__file__).parent.parent / "configs" / "models"


def load_specs(config_path: str | Path) -> tuple[ModelSpec, SystemSpec, Strategy]:
    """Load model + system + strategy from a single YAML file."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = _resolve_model(cfg["model"])
    system = _parse_system(cfg["system"])
    strategy = _parse_strategy(cfg["strategy"])

    return model, system, strategy


def load_anchor_config(yaml_path: str | Path) -> tuple[ModelSpec, SystemSpec, Strategy]:
    """Load anchor YAML into ModelSpec, SystemSpec, Strategy.

    Anchor YAMLs have a special structure:
      - model: reference to model config (e.g., "deepseek-v3") or inline model spec
      - system: hardware configuration (hw, nodes, gpus_per_node)
      - config: parallel strategy (tp, cp, pp, ep, dp, etc.)

    Returns (ModelSpec, SystemSpec, Strategy) tuple for use with estimate().
    """
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Parse model (reference or inline)
    model_ref = cfg.get("model")
    if isinstance(model_ref, str):
        # Reference to model config in configs/models/
        model = _resolve_model(model_ref)
    elif isinstance(model_ref, dict):
        # Inline model spec
        model = _parse_model(model_ref)
    else:
        raise ValueError(
            f"Anchor {yaml_path}: 'model' must be a string (reference) "
            f"or dict (inline spec), got {type(model_ref)}"
        )

    # Parse system
    system_d = cfg.get("system", {})
    if not system_d:
        raise ValueError(f"Anchor {yaml_path}: missing 'system' section")
    system = _parse_system(system_d)

    # Parse strategy from "config" section (anchors use "config" not "strategy")
    config_d = cfg.get("config", {})
    if not config_d:
        raise ValueError(f"Anchor {yaml_path}: missing 'config' section")
    strategy = _parse_strategy(config_d)

    return model, system, strategy


def _resolve_model(model_ref: str | dict) -> ModelSpec:
    if isinstance(model_ref, str):
        path = _MODELS_DIR / f"{model_ref}.yaml"
        if not path.exists():
            raise KeyError(
                f"Model {model_ref!r} not found in {_MODELS_DIR}. "
                f"Available: {[p.stem for p in sorted(_MODELS_DIR.glob('*.yaml'))]}"
            )
        with open(path, encoding="utf-8") as f:
            model_d = yaml.safe_load(f)
        return _parse_model(model_d)
    return _parse_model(model_ref)


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
        attn_compression_ratio=d.get("attn_compression_ratio", 1.0),
        # MLA (V3 / V3.2)
        q_lora_rank=d.get("q_lora_rank", 0),
        kv_lora_rank=d.get("kv_lora_rank", 0),
        qk_nope_head_dim=d.get("qk_nope_head_dim", 0),
        qk_rope_head_dim=d.get("qk_rope_head_dim", 0),
        v_head_dim=d.get("v_head_dim", 0),
        # Indexer (V3.2 / V4 CSA)
        index_n_heads=d.get("index_n_heads", 0),
        index_head_dim=d.get("index_head_dim", 0),
        index_topk=d.get("index_topk", 0),
        # V4 attention
        o_lora_rank=d.get("o_lora_rank", 0),
        o_groups=d.get("o_groups", 0),
        compress_ratios=d.get("compress_ratios", []),
        swa_window=d.get("swa_window", 0),
        # V4 MoE
        n_hash_routed_layers=d.get("n_hash_routed_layers", 0),
        scoring_func=d.get("scoring_func", "sigmoid"),
        routed_expert_dtype=d.get("routed_expert_dtype", "bf16"),
        swiglu_clamp=d.get("swiglu_clamp", 0.0),
        # MoE
        num_experts=d.get("num_experts", 0),
        moe_ffn=d.get("moe_ffn") or d.get("route_expert_hidden", 0),
        top_k=d.get("top_k", 0),
        capacity_factor=d.get("capacity_factor", 1.0),
        expert_imbalance=d.get("expert_imbalance", 0.0),
        n_group=d.get("n_group", 0),
        n_shared_experts=d.get("n_shared_experts", 1),
        # Compressed-CP
        num_csa_layers=d.get("num_csa_layers", 0),
        num_hca_layers=d.get("num_hca_layers", 0),
        num_swa_only_layers=d.get("num_swa_only_layers", 0),
        # MTP / HC
        mtp_depth=d.get("mtp_depth", 0),
        hc_mult=d.get("hc_mult", 1),
        hc_sinkhorn_iters=d.get("hc_sinkhorn_iters", 20),
        # dtypes
        param_dtype=_parse_dtype(d.get("param_dtype", "bf16")),
        grad_dtype=_parse_dtype(d.get("grad_dtype", "fp32")),
        master_dtype=_parse_dtype(d.get("master_dtype", "fp32")),
        act_dtype=_parse_dtype(d.get("act_dtype", "bf16")),
        # normalization
        norm_kind=d.get("norm_kind", "rmsnorm"),
        model_type=d.get("model_type", "default"),
        muon_ns_steps=d.get("muon_ns_steps"),
    )


def _parse_system(d: dict) -> SystemSpec:
    from zrt.hardware import registry as hw_registry

    hw = hw_registry.load(d["hw"])

    gpu = GPU(
        name=hw.name,
        flops_bf16=hw.compute.bf16_tflops,
        flops_fp8=hw.compute.fp8_tops or hw.compute.bf16_tflops * 2,
        hbm_gb=hw.memory.capacity_gb,
        hbm_bw_gbps=hw.memory.hbm_bandwidth_gbps,
        cube_tflops=hw.compute.cube_bf16_tflops,
        vector_tflops=hw.compute.vector_bf16_tflops,
        overlap_ratio=dict(hw.compute.overlap_ratio),
        sram_kb_per_sm=hw.compute.sram_kb_per_sm,
    )
    return SystemSpec(
        gpu=gpu,
        host_mem_gb=d.get("host_mem_gb", 256),
        interconnect=hw.interconnect,
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

    muon_config = None
    if "muon_config" in d:
        mc = d["muon_config"]
        muon_config = MuonConfig(
            ns_steps=mc.get("ns_steps", 5),
            rotation=mc.get("rotation", True),
            adam_param_types=set(mc.get("adam_param_types", ["embed", "lm_head", "router", "bias"])),
            muon_param_fraction=mc.get("muon_param_fraction", 0.85),
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
        muon_config=muon_config,
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
