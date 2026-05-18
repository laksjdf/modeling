from __future__ import annotations

import concurrent.futures
import itertools
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Generator

import pandas as pd
from tqdm import tqdm

# Unit conventions:
# - Search filtering uses decimal GB (/1e9) for consistency with max_memory_gb parameter
# - report_to_dict() uses GiB (1024**3) via MemBreakdown.to_gb()
# - Difference: 1000 bytes / 1e9 = 1.00 decimal GB, 1000 bytes / 1024**3 = 0.93 GiB (~7%)

from zrt.hardware.registry import load as load_hw
from zrt.training.io.config_loader import _parse_model, _parse_system, _parse_strategy
from zrt.training.search.estimator import estimate
from zrt.training.search.report import report_summary
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.report import TrainingReport
from zrt.training.spec.strategy import (
    CPKind,
    MuonConfig,
    OptKind,
    PPSched,
    RecomputePolicy,
    Strategy,
    TPOverlap,
)
from zrt.training.spec.system import GPU, SystemSpec

logger = logging.getLogger(__name__)

_WORKER_MODEL_CACHE: Dict[Tuple[str, Optional[str]], ModelSpec] = {}
_WORKER_HW_CACHE: Dict[str, SystemSpec] = {}

_MODELS_DIR = Path(__file__).parent.parent / "configs" / "models"


def _load_model_spec(model_name: str, quant_preset: Optional[str] = None) -> ModelSpec:
    import yaml
    path = _MODELS_DIR / f"{model_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Model config not found: {path}. "
            f"Available: {[p.stem for p in _MODELS_DIR.glob('*.yaml')]}"
        )
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    if quant_preset:
        d["quant_preset"] = quant_preset
    return _parse_model(d)


def _make_system_from_config(config: Dict) -> SystemSpec:
    hw_name = config.get("hw", "nvidia_h100_sxm")
    hw = load_hw(hw_name)

    gpus_per_node = config.get("gpus_per_node", 8)
    if "world_size" in config:
        nodes = config["world_size"] // gpus_per_node
    else:
        nodes = config.get("nodes", 1)

    return SystemSpec(
        gpu=GPU(
            name=hw.name,
            flops_bf16=hw.compute.bf16_tflops,
            flops_fp8=hw.compute.fp8_tops or hw.compute.bf16_tflops * 2,
            flops_fp4=hw.compute.fp4_tops,
            hbm_gb=hw.memory.capacity_gb,
            hbm_bw_gbps=hw.memory.hbm_bandwidth_gbps,
            cube_tflops=hw.compute.cube_bf16_tflops,
            vector_tflops=hw.compute.vector_bf16_tflops,
            overlap_ratio=dict(hw.compute.overlap_ratio),
        ),
        interconnect=hw.interconnect,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        host_mem_gb=config.get("host_mem_gb", 256.0),
    )


def _make_strategy_from_config(config: Dict) -> Strategy:
    recompute = RecomputePolicy()
    rc_str = config.get("recompute", "none")
    if rc_str == "selective":
        recompute.per_layer = {"moe": {"attn"}, "dense": {"attn"}}
    elif rc_str == "full":
        recompute.per_layer = {"moe": {"full"}, "dense": {"full"}}
    elif rc_str == "mhc":
        recompute.per_layer = {"moe": {"hc"}}

    muon_config = None
    opt_str = config.get("optimizer", "adam")
    if opt_str == "muon":
        muon_config = MuonConfig(rotation=config.get("muon_rotation", True))

    pp_schedule = PPSched(config.get("pp_schedule", "1f1b"))
    vpp_chunks = config.get("vpp_chunks", 1)
    if pp_schedule != PPSched.INTERLEAVED:
        vpp_chunks = 1

    return Strategy(
        tp=config.get("tp", 1),
        cp=config.get("cp", 1),
        pp=config.get("pp", 1),
        ep=config.get("ep", 1),
        dp=config.get("dp", 1),
        micro_batch=config.get("micro_batch", 1),
        global_batch=config.get("global_batch", 0),
        pp_schedule=pp_schedule,
        vpp_chunks=vpp_chunks,
        zero_stage=config.get("zero_stage", 0),
        recompute=recompute,
        optimizer=OptKind(opt_str),
        muon_config=muon_config,
        tp_overlap=TPOverlap(config.get("tp_overlap", "none")),
        ep_overlap=config.get("ep_overlap", False),
        cp_kind=CPKind(config.get("cp_kind", "none")),
        dualbatch=config.get("dualbatch", False),
        dp_overlap_in_bubble=config.get("dp_overlap_in_bubble", True),
        dp_grad_buckets=config.get("dp_grad_buckets", 25),
    )


@dataclass
class TrainingConfigManager:
    param_grid: Dict[str, List[Any]]
    output_path: str = ""

    def __post_init__(self):
        model = self.param_grid.get("model", "unknown")
        if isinstance(model, list):
            model = model[0] if model else "unknown"
        world_sizes = self.param_grid.get("world_size", [1])
        max_world = max(world_sizes) if isinstance(world_sizes, list) else world_sizes
        self.output_path = os.path.join("output", "training_search", f"{model}_ws_{max_world}")

    def _get_divisors(self, n: int) -> List[int]:
        return [i for i in range(1, n + 1) if n % i == 0]

    def _build_strategy_for_validation(
            self, tp: int, cp: int, pp: int, ep: int, dp: int,
            other_config: Dict[str, Any]
    ) -> Strategy:
        pp_schedule = PPSched(other_config.get("pp_schedule", "1f1b"))
        vpp_chunks = other_config.get("vpp_chunks", 1)
        if pp_schedule != PPSched.INTERLEAVED:
            vpp_chunks = 1

        recompute = RecomputePolicy()
        rc_str = other_config.get("recompute", "none")
        if rc_str == "selective":
            recompute.per_layer = {"moe": {"attn"}, "dense": {"attn"}}
        elif rc_str == "full":
            recompute.per_layer = {"moe": {"full"}, "dense": {"full"}}
        elif rc_str == "mhc":
            recompute.per_layer = {"moe": {"hc"}}

        muon_config = None
        opt_str = other_config.get("optimizer", "adam")
        if opt_str == "muon":
            muon_config = MuonConfig(rotation=other_config.get("muon_rotation", True))

        return Strategy(
            tp=tp, cp=cp, pp=pp, ep=ep, dp=dp,
            micro_batch=other_config.get("micro_batch", 1),
            global_batch=other_config.get("global_batch", 0),
            pp_schedule=pp_schedule,
            vpp_chunks=vpp_chunks,
            zero_stage=other_config.get("zero_stage", 0),
            recompute=recompute,
            optimizer=OptKind(opt_str),
            muon_config=muon_config,
            tp_overlap=TPOverlap(other_config.get("tp_overlap", "none")),
            ep_overlap=other_config.get("ep_overlap", False),
            cp_kind=CPKind(other_config.get("cp_kind", "none")),
            dualbatch=other_config.get("dualbatch", False),
            dp_overlap_in_bubble=other_config.get("dp_overlap_in_bubble", True),
            dp_grad_buckets=other_config.get("dp_grad_buckets", 25),
        )

    def _expand_auto_values_optimized(self, grid: Dict[str, List[Any]], world_size: int) -> None:
        """
        核心优化：基于已知维度的最小值，动态裁剪并收紧 auto 变量的选择范围。
        避免盲目扩展成全量 world_size 的约数，将 auto 并行搜索空间缩减 90% 以上。
        
        注意：EP 不占用额外 rank，world_size = TP*CP*PP*DP
        """
        rank_keys = ["tp", "cp", "pp", "dp"]
        all_keys = ["tp", "cp", "pp", "ep", "dp"]

        # 确保所有并行维度都有默认值
        for key in all_keys:
            if key not in grid:
                grid[key] = [1]

        # 1. 整理出哪些键是固定值或具体范围，哪些键被设为了 "auto"
        explicit_rank_keys = []
        auto_rank_keys = []
        auto_ep = False

        for key in all_keys:
            vals = grid[key]
            is_auto = vals == "auto" or vals == ["auto"] or "auto" in vals
            if key == "ep":
                auto_ep = is_auto
            elif is_auto:
                auto_rank_keys.append(key)
            else:
                if key in rank_keys:
                    explicit_rank_keys.append(key)

        if not auto_rank_keys and not auto_ep:
            return

        # 2. 计算已知 rank 维度的乘积最小值（不包括 EP）
        min_explicit_prod = 1
        for key in explicit_rank_keys:
            vals = [v for v in grid[key] if v != "auto"]
            if vals:
                min_explicit_prod *= min(vals)

        # 3. 动态推导 auto rank 变量所允许的最大上限边界值
        max_allowed_val = world_size // min_explicit_prod

        # 提取当前合法边界内所有符合物理整除条件的约数
        all_divisors = self._get_divisors(world_size)
        optimized_divisors = [d for d in all_divisors if d <= max_allowed_val]

        # 4. 回填 rank 参数网格（tp, cp, pp, dp）
        for key in auto_rank_keys:
            vals = grid[key]
            if isinstance(vals, list):
                clean_vals = [v for v in vals if v != "auto"]
                grid[key] = sorted(list(set(clean_vals + optimized_divisors)))
            else:
                grid[key] = optimized_divisors

        # 5. EP 特殊处理：EP 独立扩展，不参与 rank 计算
        if auto_ep:
            vals = grid["ep"]
            if isinstance(vals, list):
                clean_vals = [v for v in vals if v != "auto"]
                # EP 可以是任意值，通常基于 num_experts 的约束
                grid["ep"] = sorted(list(set(clean_vals + optimized_divisors)))
            else:
                grid["ep"] = optimized_divisors

    def _enumerate_valid_parallel_configs(
            self, grid: Dict[str, List[Any]], target_ws: int,
            model: ModelSpec = None, system: SystemSpec = None,
            other_config: Dict[str, Any] = None
    ) -> Generator[Tuple[int, int, int, int, int], None, None]:
        tp_vals = grid.get("tp", [1])
        cp_vals = grid.get("cp", [1])
        pp_vals = grid.get("pp", [1])
        ep_vals = grid.get("ep", [1])
        global_batch = other_config.get("global_batch", 0) if other_config else 0
        micro_batch = other_config.get("micro_batch", 1) if other_config else 1
        zero_stage = other_config.get("zero_stage", 0) if other_config else 0
        cp_kind_str = other_config.get("cp_kind", "none") if other_config else "none"
        pp_schedule_str = other_config.get("pp_schedule", "1f1b") if other_config else "1f1b"

        for tp in tp_vals:
            if model is not None:
                if model.num_heads % tp != 0:
                    continue
                if model.num_kv_heads % tp != 0 and model.num_kv_heads >= tp:
                    continue
                if model.ffn % tp != 0:
                    continue

            for cp in cp_vals:
                if model is not None and cp_kind_str == "ulysses":
                    if model.num_heads % cp != 0:
                        continue

                for pp in pp_vals:
                    if model is not None and pp > len(model.layers):
                        continue
                    # Pipeline schedules other than 1F1B require pp > 1; at
                    # pp=1 they degenerate to OneF1B and just pollute the grid.
                    if pp == 1 and pp_schedule_str != "1f1b":
                        continue

                    remaining = target_ws // (tp * cp * pp)
                    if remaining <= 0 or target_ws % (tp * cp * pp) != 0:
                        continue

                    for dp in grid.get("dp", [1]):
                        if tp * cp * pp * dp != target_ws:
                            continue

                        if global_batch > 0:
                            if global_batch % (micro_batch * dp) != 0:
                                continue

                        if zero_stage >= 1 and dp <= 1:
                            continue

                        for ep in ep_vals:
                            if model is not None and ep > 1:
                                if model.num_experts <= 0:
                                    continue
                                if model.num_experts % ep != 0:
                                    continue
                                # Must have at least ep distinct ranks outside
                                # TP to place each expert on its own rank
                                # group; ep > rank_pool means we can't even
                                # cover one EP replica.
                                rank_pool = dp * pp * cp
                                if ep > rank_pool:
                                    continue

                            yield (tp, cp, pp, ep, dp)

    def get_valid_parallel_combos(self, grid: Dict[str, List[Any]], target_ws: int) -> List[Tuple[int, ...]]:
        return list(self._enumerate_valid_parallel_configs(grid, target_ws))

    def count_total_configs(self) -> int:
        grid = {k: (v if isinstance(v, list) else [v]) for k, v in self.param_grid.items()}
        world_sizes = grid.get("world_size", [1])
        target_ws = world_sizes[0]
        self._expand_auto_values_optimized(grid, target_ws)

        parallel_keys = ["tp", "cp", "pp", "ep", "dp"]
        
        total_token_vals = grid.get("total_token", [])
        has_total_token = total_token_vals and any(v is not None and v > 0 for v in total_token_vals)
        
        other_keys = [k for k in grid.keys() if k not in parallel_keys and k != "world_size"]
        if has_total_token and "seq_len" in other_keys:
            other_keys.remove("seq_len")

        model_name = grid.get("model", ["unknown"])[0] if grid.get("model") else "unknown"
        hw_name = grid.get("hw", ["nvidia_h100_sxm"])[0] if grid.get("hw") else "nvidia_h100_sxm"
        seq_len = grid.get("seq_len", [4096])[0] if grid.get("seq_len") else 4096

        try:
            model = _load_model_spec(model_name)
            model.seq_len = seq_len
            hw = load_hw(hw_name)
            gpus_per_node = grid.get("gpus_per_node", [8])[0] if grid.get("gpus_per_node") else 8
            nodes = target_ws // gpus_per_node
            system = SystemSpec(
                gpu=GPU(
                    name=hw.name,
                    flops_bf16=hw.compute.bf16_tflops,
                    flops_fp8=hw.compute.fp8_tops or hw.compute.bf16_tflops * 2,
                    flops_fp4=hw.compute.fp4_tops,
                    hbm_gb=hw.memory.capacity_gb,
                    hbm_bw_gbps=hw.memory.hbm_bandwidth_gbps,
                    cube_tflops=hw.compute.cube_bf16_tflops,
                    vector_tflops=hw.compute.vector_bf16_tflops,
                    overlap_ratio=dict(hw.compute.overlap_ratio),
                ),
                interconnect=hw.interconnect,
                nodes=nodes,
                gpus_per_node=gpus_per_node,
                host_mem_gb=grid.get("host_mem_gb", [256.0])[0] if grid.get("host_mem_gb") else 256.0,
            )
        except Exception:
            model = None
            system = None

        other_combinations = 1
        for k in other_keys:
            other_combinations *= len(grid[k])

        total = 0
        for other_vals in itertools.product(*[grid[k] for k in other_keys]):
            base_config = dict(zip(other_keys, other_vals))
            total += sum(1 for _ in self._enumerate_valid_parallel_configs(
                grid, target_ws, model, system, base_config
            ))

        return other_combinations * total if model is None else total

    def generate_static_configs_stream(self) -> Generator[Dict[str, Any], None, None]:
        grid = {k: (v if isinstance(v, list) else [v]) for k, v in self.param_grid.items()}
        world_sizes = grid.get("world_size", [1])
        if len(world_sizes) > 1:
            raise ValueError("Only single world_size is supported when using 'auto' parallel strategy")
        target_ws = world_sizes[0]

        self._expand_auto_values_optimized(grid, target_ws)

        parallel_keys = ["tp", "cp", "pp", "ep", "dp"]
        
        total_token_vals = grid.get("total_token", [])
        has_total_token = total_token_vals and any(v is not None and v > 0 for v in total_token_vals)
        
        other_keys = [k for k in grid.keys() if k not in parallel_keys and k != "world_size"]
        if has_total_token and "seq_len" in other_keys:
            other_keys.remove("seq_len")

        model_name = grid.get("model", ["unknown"])[0] if grid.get("model") else "unknown"
        hw_name = grid.get("hw", ["nvidia_h100_sxm"])[0] if grid.get("hw") else "nvidia_h100_sxm"
        seq_len = grid.get("seq_len", [4096])[0] if grid.get("seq_len") else 4096

        model = None
        system = None
        try:
            model = _load_model_spec(model_name)
            model.seq_len = seq_len
            hw = load_hw(hw_name)
            gpus_per_node = grid.get("gpus_per_node", [8])[0] if grid.get("gpus_per_node") else 8
            nodes = target_ws // gpus_per_node
            system = SystemSpec(
                gpu=GPU(
                    name=hw.name,
                    flops_bf16=hw.compute.bf16_tflops,
                    flops_fp8=hw.compute.fp8_tops or hw.compute.bf16_tflops * 2,
                    flops_fp4=hw.compute.fp4_tops,
                    hbm_gb=hw.memory.capacity_gb,
                    hbm_bw_gbps=hw.memory.hbm_bandwidth_gbps,
                    cube_tflops=hw.compute.cube_bf16_tflops,
                    vector_tflops=hw.compute.vector_bf16_tflops,
                    overlap_ratio=dict(hw.compute.overlap_ratio),
                ),
                interconnect=hw.interconnect,
                nodes=nodes,
                gpus_per_node=gpus_per_node,
                host_mem_gb=grid.get("host_mem_gb", [256.0])[0] if grid.get("host_mem_gb") else 256.0,
            )
        except FileNotFoundError:
            pass
        except Exception:
            pass

        other_grids = [grid[k] for k in other_keys]
        for other_vals in itertools.product(*other_grids):
            base_config = dict(zip(other_keys, other_vals))
            base_config["world_size"] = target_ws

            total_token = base_config.get("total_token")
            if total_token is not None and total_token > 0:
                global_batch = base_config.get("global_batch", 0)
                if global_batch > 0:
                    seq_len = int(total_token / global_batch)
                    base_config["seq_len"] = seq_len
                    if model is not None:
                        model.seq_len = seq_len
            else:
                seq_len = base_config.get("seq_len")
                if seq_len is not None and model is not None:
                    model.seq_len = seq_len

            for p_vals in self._enumerate_valid_parallel_configs(
                    grid, target_ws, model, system, base_config
            ):
                config = base_config.copy()
                config.update(dict(zip(parallel_keys, p_vals)))
                yield config

    def generate_static_configs(self) -> List[Dict[str, Any]]:
        return list(self.generate_static_configs_stream())


def _worker_initializer(model_name: str = "deepseek_v3_2"):
    global _WORKER_MODEL_CACHE, _WORKER_HW_CACHE
    _WORKER_MODEL_CACHE[(model_name, None)] = _load_model_spec(model_name)


def run_training_task_wrapper(config: Dict) -> Optional[Dict]:
    from zrt.training.ir.builders import build_graph

    model_name = config.get("model", "deepseek_v3_2")
    hw_name = config.get("hw", "nvidia_h100_sxm")
    quant_preset = config.get("quant_preset") or None

    try:
        model_key = (model_name, quant_preset)
        model = _WORKER_MODEL_CACHE.get(model_key)
        if model is None:
            model = _load_model_spec(model_name, quant_preset=quant_preset)
            _WORKER_MODEL_CACHE[model_key] = model
        # Per-config seq_len override: the cached ModelSpec carries the YAML
        # default, but the grid varies seq_len, and Strategy does not own this
        # field. Mutating the cached object is safe — each worker is a single
        # process running one task at a time.
        seq_len = config.get("seq_len")
        if seq_len is not None:
            model.seq_len = int(seq_len)

        # System cache must key on world_size / gpus_per_node, not just hw_name,
        # because nodes is derived from those — otherwise multi-world_size grids
        # silently reuse the first config's SystemSpec.
        sys_key = (
            hw_name,
            int(config.get("world_size", 0)),
            int(config.get("gpus_per_node", 8)),
        )
        system = _WORKER_HW_CACHE.get(sys_key)
        if system is None:
            system = _make_system_from_config(config)
            _WORKER_HW_CACHE[sys_key] = system

        strategy = _make_strategy_from_config(config)
        strategy.validate(model, system)
    except Exception as e:
        return {"status": "error", "config": config, "type": "validation_error", "message": str(e)}

    try:
        graph = build_graph(model, strategy)
        report = estimate(model, system, strategy, graph=graph)

        return {
            "status": "success",
            "config": config,
            "report": report,
            "model_name": model_name,
            "hw_name": hw_name,
        }
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return {"status": "error", "config": config, "type": "runtime_error"}


def format_results(reports: List[TrainingReport], configs: List[Dict]) -> pd.DataFrame:
    rows = []
    for cfg, report in zip(configs, reports):
        d = cfg.copy()
        if report.memory:
            memory_gb = round(report.memory.total / 1e9, 2)
            hw_name = cfg.get("hw", "nvidia_h100_sxm")
            hw = load_hw(hw_name)
            gpu_capacity_gb = hw.memory.capacity_gb
            # *0.8利用率
            if memory_gb > gpu_capacity_gb * 0.8:
                continue
        else:
            memory_gb = None

        d["fwd_compute_ms"] = round(report.fwd_compute_ms, 2)
        d["bwd_compute_ms"] = round(report.bwd_compute_ms, 2)
        d["exposed_comm_ms"] = round(report.exposed_comm_ms, 2)
        d["tp_total_ms"] = round(report.tp_total_ms, 2)
        d["tp_exposed_ms"] = round(report.tp_exposed_ms, 2)
        d["cp_total_ms"] = round(report.cp_total_ms, 2)
        d["cp_exposed_ms"] = round(report.cp_exposed_ms, 2)
        d["ep_total_ms"] = round(report.ep_total_ms, 2)
        d["ep_exposed_ms"] = round(report.ep_exposed_ms, 2)
        d["pp_total_ms"] = round(report.pp_total_ms, 2)
        d["pp_exposed_ms"] = round(report.pp_exposed_ms, 2)
        d["dp_total_ms"] = round(report.dp_total_ms, 2)
        d["dp_exposed_ms"] = round(report.dp_exposed_ms, 2)
        d["optimizer_compute_ms"] = round(report.optimizer_time_ms, 2)
        d["optimizer_total_ms"] = round(report.optimizer_comm_ms + report.optimizer_comm_hidden_ms, 2)
        d["optimizer_exposed_ms"] = round(report.optimizer_comm_ms, 2)
        d["step_time_ms"] = round(report.step_time_ms, 3)
        d["pipeline_time_ms"] = round(report.pipeline_time_ms, 3)
        d["mfu"] = round(report.mfu, 4)
        d["mfu_native"] = round(report.mfu_native, 4)
        d["hfu"] = round(report.hfu, 4)
        d["bubble_fraction"] = round(report.bubble_fraction, 4)
        d["tokens_per_sec"] = round(report.tokens_per_sec, 1)
        if report.memory:
            # Per-rank weight/grad/opt-state can drop to single-digit MB after
            # ZeRO-3 + EP + TP sharding; 2-decimal GB rounds those to 0.00 and
            # hides whether sharding is sane. Use 4 decimals (≈100 KB resolution).
            d["weights_gb"] = round(report.memory.weights / 1e9, 4)
            d["grads_gb"] = round(report.memory.grads / 1e9, 4)
            d["opt_state_gb"] = round(report.memory.opt_state / 1e9, 4)
            d["activations_gb"] = round(report.memory.activations / 1e9, 2)
            d["comm_buffers_gb"] = round(report.memory.comm_buffers / 1e9, 2)
            d["memory_gb"] = memory_gb
        d["mhc_recompute_ms"] = round(report.mhc_recompute_ms, 2) if report.mhc_recompute_ms > 0 else 0.0
        rows.append(d)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("mfu", ascending=False)

    metric_cols = ["fwd_compute_ms", "bwd_compute_ms", "exposed_comm_ms",
                   "tp_total_ms", "tp_exposed_ms", "cp_total_ms", "cp_exposed_ms",
                   "ep_total_ms", "ep_exposed_ms", "pp_total_ms", "pp_exposed_ms",
                   "dp_total_ms", "dp_exposed_ms",
                   "optimizer_compute_ms", "optimizer_total_ms", "optimizer_exposed_ms",
                   "step_time_ms", "pipeline_time_ms",
                   "mfu", "mfu_native", "hfu", "bubble_fraction", "tokens_per_sec",
                   "weights_gb", "grads_gb", "opt_state_gb", "activations_gb",
                   "comm_buffers_gb", "memory_gb", "mhc_recompute_ms"]
    config_cols = [k for k in rows[0].keys() if k not in metric_cols] if rows else []
    cols = config_cols + [c for c in metric_cols if c in df.columns]
    df = df[[c for c in cols if c in df.columns]]
    return df


def save_results(df: pd.DataFrame, output_path: str):
    os.makedirs(output_path, exist_ok=True)

    csv_path = os.path.join(output_path, "results_summary.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to: {csv_path}")

    print("=" * 60)
    print("Training Search Results")
    print("=" * 60)
    print(f"Total results: {len(df)} configs")
    print()
    print("Top 5 configs by MFU:")
    print(df.head(5).to_string())
    print("=" * 60)


def export_best_configs_excel(
        all_results: List[Dict],
        output_path: str
) -> None:
    from zrt.training.ir.builders import build_graph
    from zrt.training.models.flops import op_cost as _op_cost
    from zrt.training.io.excel_exporter import export_estimate_excel

    if not all_results:
        return

    df = pd.DataFrame([
        {
            "model": r["model_name"],
            "hw": r["hw_name"],
            "seq_len": r["config"].get("seq_len", 4096),
            "world_size": r["config"].get("world_size", 1),
            "mfu": r["report"].mfu,
            "config": r["config"],
            "report": r["report"],
        }
        for r in all_results
    ])

    grouped = df.groupby(["model", "hw", "seq_len", "world_size"])

    for (model_name, hw_name, seq_len, world_size), group in grouped:
        best_row = group.loc[group["mfu"].idxmax()]
        best_config = best_row["config"]
        best_report = best_row["report"]

        model = _load_model_spec(model_name)
        model.seq_len = seq_len

        hw = load_hw(hw_name)
        gpus_per_node = best_config.get("gpus_per_node", 8)
        nodes = world_size // gpus_per_node
        system = SystemSpec(
            gpu=GPU(
                name=hw.name,
                flops_bf16=hw.compute.bf16_tflops,
                flops_fp8=hw.compute.fp8_tops or hw.compute.bf16_tflops * 2,
                flops_fp4=hw.compute.fp4_tops,
                hbm_gb=hw.memory.capacity_gb,
                hbm_bw_gbps=hw.memory.hbm_bandwidth_gbps,
                cube_tflops=hw.compute.cube_bf16_tflops,
                vector_tflops=hw.compute.vector_bf16_tflops,
                overlap_ratio=dict(hw.compute.overlap_ratio),
            ),
            interconnect=hw.interconnect,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            host_mem_gb=best_config.get("host_mem_gb", 256.0),
        )

        strategy = _make_strategy_from_config(best_config)

        graph = build_graph(model, strategy)
        op_costs = {}
        for op in graph.ops:
            op_costs[op.name] = _op_cost(op, model, system)

        excel_name = f"{model_name}_{hw_name}_{seq_len}_best.xlsx"
        excel_path = os.path.join(output_path, excel_name)

        export_estimate_excel(
            report=best_report,
            graph=graph,
            model=model,
            system=system,
            strategy=strategy,
            op_costs=op_costs,
            output_path=excel_path,
        )
        logger.info(f"Best config Excel exported: {excel_path}")
        print(f"Best config Excel: {excel_path} (MFU={best_report.mfu:.4%})")


def run_training_search_parallel(
        param_grid: Dict[str, List[Any]],
        workers: int = 8,
        mfu_threshold: float = 0.0,
) -> pd.DataFrame:
    model_name = param_grid.get("model", ["unknown"])
    if isinstance(model_name, list):
        model_name = model_name[0] if model_name else "unknown"

    manager = TrainingConfigManager(
        param_grid=param_grid,
    )

    total_configs = manager.count_total_configs()
    adjusted_workers = min(workers, os.cpu_count() or 8)

    if os.path.exists(manager.output_path):
        for f in os.listdir(manager.output_path):
            fpath = os.path.join(manager.output_path, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
            elif os.path.isdir(fpath):
                import shutil
                shutil.rmtree(fpath)
    os.makedirs(manager.output_path, exist_ok=True)

    start_time = time.time()
    logger.info(f"Starting search: total_configs={total_configs}, workers={adjusted_workers}")

    all_results: List[Dict] = []
    error_count = 0

    config_generator = manager.generate_static_configs_stream()
    futures_map = {}

    with ProcessPoolExecutor(
            max_workers=adjusted_workers,
            initializer=_worker_initializer,
            initargs=(model_name,),
    ) as executor:
        with tqdm(total=total_configs, desc="Evaluating configs", unit="config") as pbar:
            for cfg in config_generator:
                fut = executor.submit(run_training_task_wrapper, cfg)
                futures_map[fut] = cfg

                while len(futures_map) >= adjusted_workers * 2:
                    done, _ = concurrent.futures.wait(
                        futures_map.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for fut in done:
                        futures_map.pop(fut)
                        pbar.update(1)
                        res = fut.result()
                        if res and res["status"] == "success":
                            all_results.append(res)
                        elif res and res["status"] == "error":
                            error_count += 1

            while futures_map:
                done, _ = concurrent.futures.wait(futures_map.keys())
                for fut in done:
                    futures_map.pop(fut)
                    pbar.update(1)
                    res = fut.result()
                    if res and res["status"] == "success":
                        all_results.append(res)
                    elif res and res["status"] == "error":
                        error_count += 1

    elapsed = time.time() - start_time
    logger.info(f"Search completed in {elapsed:.2f} seconds, success={len(all_results)}, errors={error_count}")

    if not all_results:
        logger.error("No valid configurations found.")
        return pd.DataFrame()

    # Memory-feasibility filter: drop configs whose per-rank memory exceeds
    # 80% of GPU HBM capacity. Apply before BOTH the CSV table and the Excel
    # "best" export so the latter cannot pick infeasible configs.
    feasible_results: List[Dict] = []
    for r in all_results:
        rep = r["report"]
        hw_name = r.get("hw_name") or r["config"].get("hw", "nvidia_h100_sxm")
        cap_gb = load_hw(hw_name).memory.capacity_gb
        if rep.memory is None:
            feasible_results.append(r)
            continue
        if rep.memory.total / 1e9 <= cap_gb * 0.8:
            feasible_results.append(r)
    logger.info(
        f"Memory feasibility: {len(feasible_results)}/{len(all_results)} configs "
        f"fit within 0.8 × HBM"
    )
    if not feasible_results:
        logger.error("No memory-feasible configurations found.")
        return pd.DataFrame()

    all_reports = [r["report"] for r in feasible_results]
    all_configs = [r["config"] for r in feasible_results]
    all_df = format_results(all_reports, all_configs)

    filtered_df = all_df[all_df["mfu"] > mfu_threshold] if mfu_threshold > 0 else all_df
    if filtered_df.empty:
        logger.warning(f"No results with MFU > {mfu_threshold}")
        return pd.DataFrame()

    save_results(filtered_df, manager.output_path)

    export_best_configs_excel(feasible_results, manager.output_path)

    if not filtered_df.empty:
        best = filtered_df.iloc[0]
        print("\nTop1 Result:")
        print(f"  MFU: {best['mfu']:.4f}, step_time: {best['step_time_ms']:.2f}ms")

    return filtered_df


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    training_param_grid = {
        "model": ["deepseek_v4_pro"],
        "hw": ["nvidia_b300"],
        "world_size": [8192],
        "tp": [1, 2, 4, 8, 16],
        "cp": [1, 2, 4, 8, 16],
        "pp": [1, 2, 4, 8, 16],
        "ep": [384],
        "dp": "auto",
        "micro_batch": [1, 16, 32],
        "global_batch": [512, 1024, 2048, 4096, 8192, 65536],
        "seq_len": [8192, 32768, 131072],
        "zero_stage": [3],
        "pp_schedule": ["dualpipev"],
        "cp_kind": ["ulysses"],
        "tp_overlap": ["coc"],
        "ep_overlap": [True],
        "dualbatch": [True],
        "dp_overlap_in_bubble": [True],
        "recompute": ["none", "mhc"],
        "optimizer": ["muon"],
        "quant_preset": ["deepseek_v4_fp8_fp4"],
    }

    df = run_training_search_parallel(
        param_grid=training_param_grid,
        workers=32,
        mfu_threshold=0.05,
    )
