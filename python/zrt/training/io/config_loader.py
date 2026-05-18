"""YAML config loader — parse model + system + strategy from a single YAML file."""

from __future__ import annotations

import warnings as _warnings
from pathlib import Path

import yaml

# Canonical recompute categories. Updating here propagates to the loader's
# validation. Keep in sync with the docstring in
# ``zrt/training/spec/strategy.py::RecomputePolicy``.
_CANONICAL_RECOMPUTE_CATS = {
    "full", "attn_core", "attn_block", "ffn_swiglu", "ln", "hc",
}

# Deprecated → canonical alias map. We resolve at load time and emit a
# DeprecationWarning so users have time to migrate their YAMLs.
_RECOMPUTE_ALIASES = {"attn": "attn_block"}


def _normalize_recompute_categories(layer_kind: str, raw) -> set[str]:
    """Validate + alias-resolve a YAML recompute spec for one layer kind.

    Accepts either a single string or a list of strings. Unknown categories
    raise ``ValueError`` rather than being silently ignored.
    """
    if isinstance(raw, str):
        raw = [raw]
    out: set[str] = set()
    for cat in raw:
        if cat in _RECOMPUTE_ALIASES:
            new = _RECOMPUTE_ALIASES[cat]
            _warnings.warn(
                f"recompute category {cat!r} is deprecated; use {new!r}. "
                f"(layer kind: {layer_kind!r})",
                DeprecationWarning,
                stacklevel=4,
            )
            out.add(new)
        elif cat in _CANONICAL_RECOMPUTE_CATS:
            out.add(cat)
        else:
            raise ValueError(
                f"Unknown recompute category {cat!r} for layer kind "
                f"{layer_kind!r}. Valid: "
                f"{sorted(_CANONICAL_RECOMPUTE_CATS)} "
                f"(plus deprecated alias 'attn' → 'attn_block')."
            )
    return out

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
        # Either an inline spec or an overlay form ({base: <name>, ...}).
        # Dispatch through _resolve_model so the overlay branch is honored.
        model = _resolve_model(model_ref)
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
    if isinstance(model_ref, dict) and "base" in model_ref:
        # Overlay form: {base: <name>, <override keys>}.
        # Load the named base spec from configs/models/ and merge the
        # remaining keys on top (caller wins).
        base_name = model_ref["base"]
        path = _MODELS_DIR / f"{base_name}.yaml"
        if not path.exists():
            raise KeyError(
                f"Model {base_name!r} not found in {_MODELS_DIR}. "
                f"Available: {[p.stem for p in sorted(_MODELS_DIR.glob('*.yaml'))]}"
            )
        with open(path, encoding="utf-8") as f:
            base_d = yaml.safe_load(f)
        merged = {**base_d, **{k: v for k, v in model_ref.items() if k != "base"}}
        return _parse_model(merged)
    return _parse_model(model_ref)


def _parse_model(d: dict) -> ModelSpec:
    d = _expand_quant_preset(d)
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
        # NEW per-component dtypes (Task 3 added the ModelSpec fields)
        attn_compute_dtype=_parse_dtype(d.get("attn_compute_dtype", "bf16")),
        shared_expert_compute_dtype=_parse_dtype(d.get("shared_expert_compute_dtype", "bf16")),
        routed_expert_compute_dtype=_parse_dtype(d.get("routed_expert_compute_dtype", "bf16")),
        routed_expert_weight_dtype=_parse_dtype(d["routed_expert_weight_dtype"]) if "routed_expert_weight_dtype" in d else None,
        attn_act_dtype=_parse_dtype(d["attn_act_dtype"]) if "attn_act_dtype" in d else None,
        moe_act_dtype=_parse_dtype(d["moe_act_dtype"]) if "moe_act_dtype" in d else None,
        routed_expert_grad_dtype=_parse_dtype(d.get("routed_expert_grad_dtype", "fp32")),
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
        flops_fp4=hw.compute.fp4_tops,   # 0 -> peak_tflops_for falls back to fp8
        hbm_gb=hw.memory.capacity_gb,
        hbm_bw_gbps=hw.memory.hbm_bandwidth_gbps,
        cube_tflops=hw.compute.cube_bf16_tflops,
        vector_tflops=hw.compute.vector_bf16_tflops,
        overlap_ratio=dict(hw.compute.overlap_ratio),
        sram_kb_per_sm=hw.compute.sram_kb_per_sm,
        ep_overlap_waves=hw.compute.ep_overlap_waves,
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
            per_layer[kind_str] = _normalize_recompute_categories(kind_str, tiers)
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
        dp_steady_overlap_ratio=float(d.get("dp_steady_overlap_ratio", 0.5)),
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


_QUANT_PRESETS: dict[str, dict[str, str]] = {
    "bf16_baseline": {
        "attn_compute_dtype": "bf16",
        "routed_expert_compute_dtype": "bf16",
        "routed_expert_weight_dtype": "bf16",
        "shared_expert_compute_dtype": "bf16",
        "routed_expert_grad_dtype": "fp32",
        "act_dtype": "bf16",
    },
    "fp8_mixed": {   # DeepSeek-V3 style
        "attn_compute_dtype": "bf16",
        "routed_expert_compute_dtype": "fp8_e4m3",
        "routed_expert_weight_dtype": "bf16",
        "shared_expert_compute_dtype": "bf16",
        "routed_expert_grad_dtype": "fp32",
        "act_dtype": "bf16",
        "moe_act_dtype": "fp8_e4m3",
    },
    "deepseek_v4_fp8_fp4": {   # V4 main path
        "attn_compute_dtype": "bf16",
        "routed_expert_compute_dtype": "fp8_e4m3",
        "routed_expert_weight_dtype": "fp4",
        "shared_expert_compute_dtype": "bf16",
        "routed_expert_grad_dtype": "fp32",
        "act_dtype": "bf16",
        "moe_act_dtype": "fp8_e4m3",
    },
    "deepseek_v4_full_fp8": {   # V4 with FP8 shared experts
        "attn_compute_dtype": "bf16",
        "routed_expert_compute_dtype": "fp8_e4m3",
        "routed_expert_weight_dtype": "fp4",
        "shared_expert_compute_dtype": "fp8_e4m3",
        "routed_expert_grad_dtype": "fp32",
        "act_dtype": "bf16",
        "moe_act_dtype": "fp8_e4m3",
        "attn_act_dtype": "bf16",
    },
}


def _expand_quant_preset(d: dict) -> dict:
    """Expand a ``quant_preset`` shorthand into explicit dtype fields.

    Explicit fields in ``d`` override preset values. Removes the
    ``quant_preset`` key from the returned mapping. Returns a new dict
    (does not mutate input).
    """
    preset_name = d.get("quant_preset")
    out = {k: v for k, v in d.items() if k != "quant_preset"}
    if preset_name is None:
        return out
    if preset_name not in _QUANT_PRESETS:
        raise KeyError(
            f"unknown quant_preset {preset_name!r}; "
            f"valid options: {sorted(_QUANT_PRESETS)}"
        )
    for key, val in _QUANT_PRESETS[preset_name].items():
        out.setdefault(key, val)
    return out


def _parse_dtype(s: str) -> Dtype:
    s = s.lower().strip()
    mapping = {
        "fp32": Dtype.FP32, "float32": Dtype.FP32,
        "bf16": Dtype.BF16, "bfloat16": Dtype.BF16,
        "fp16": Dtype.FP16, "float16": Dtype.FP16,
        "fp8": Dtype.FP8_E4M3, "float8": Dtype.FP8_E4M3,
        "fp8_e4m3": Dtype.FP8_E4M3,
        "fp8_e5m2": Dtype.FP8_E5M2,
        "fp4": Dtype.FP4, "mxfp4": Dtype.FP4, "nvfp4": Dtype.FP4,
    }
    return mapping.get(s, Dtype.BF16)
