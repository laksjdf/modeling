"""Top-level CLI entry point for ZRT-Sim.

Usage::

    python -m python.zrt --model-id Qwen/Qwen2.5-7B-Instruct --layers 4
    python -m python.zrt --model-id deepseek-ai/DeepSeek-V3-0324 --layers 4 --hw nvidia_h100_sxm --tp 8
    python -m python.zrt --model-id hf_models/llama3_8b --train --layers 2
    python -m python.zrt --estimate-config python/zrt/training/configs/llama3_70b_3d.yaml
    python -m python.zrt --search-config python/zrt/training/configs/llama3_70b_3d.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Lazy imports to avoid requiring torch at module load time
# These are imported only when needed:
#   from python.zrt.pipeline import run_trace_phases, _make_model_slug, _MODEL_DIRS, _PHASE_ALIASES


def _get_model_dirs():
    """Lazy import of _MODEL_DIRS to avoid requiring torch at module load time."""
    from python.zrt.pipeline import _MODEL_DIRS
    return _MODEL_DIRS


def _make_model_slug(model_id: str) -> str:
    """Lazy import of _make_model_slug to avoid requiring torch at module load time."""
    from python.zrt.pipeline import _make_model_slug as _impl
    return _impl(model_id)


def _run_trace_phases(**kwargs):
    """Lazy import of run_trace_phases to avoid requiring torch at module load time."""
    from python.zrt.pipeline import run_trace_phases
    return run_trace_phases(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace LLM operator sequences and write Excel + computation graph.")

    # ── Mode flags (mutually exclusive) ──────────────────────────────────────
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--estimate-config",
        metavar="YAML",
        help="Run spec-based training estimation from a YAML config (no graph capture). "
             "Example: --estimate-config python/zrt/training/configs/llama3_70b_3d.yaml",
    )
    mode_group.add_argument(
        "--search-config",
        metavar="YAML",
        help="Grid-search parallel strategies for a training config. "
             "Example: --search-config python/zrt/training/configs/llama3_70b_3d.yaml",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        default=False,
        help="Print a structured per-component time breakdown alongside the estimate. "
             "Shows compute, comm (by group), bubble, optimizer, and hidden comm.",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Write result to FILE. For --estimate-config: .xlsx for Excel report (default), "
             ".json for JSON. For --search-config: writes Pareto frontier JSON.",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model-id",
        metavar="MODEL",
        default=None,
        help="HF Hub model ID or local directory (e.g. deepseek-ai/DeepSeek-V3-0324). "
             "Required for graph capture modes.",
    )
    parser.add_argument(
        "--model",
        choices=_get_model_dirs().keys(),
        default=None,
        help="Shorthand for local DeepSeek model: v3 or v3.2 (maps to hf_models/).",
    )

    # ── Input & layers ────────────────────────────────────────────────────────
    parser.add_argument("--layers", type=int, default=4,
                        help="Number of transformer layers to trace (default: 4)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Dummy input batch size (default: 1)")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Prefill sequence length (default: 128)")

    _layer_group = parser.add_mutually_exclusive_group()
    _layer_group.add_argument(
        "--target-layers",
        metavar="IDX",
        help="Comma-separated layer indices to trace, e.g. '0,3'.",
    )
    _layer_group.add_argument(
        "--auto-layers",
        action="store_true",
        default=False,
        help="Automatically select the first dense and first sparse (MoE) layer.",
    )

    # ── Phases ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--phases", nargs="+", default=None,
        choices=["prefill", "decode", "forward",
                 "train_forward", "train_backward", "train"],
        metavar="PHASE",
        help="Phases to trace (default: prefill decode). "
             "Inference: prefill, decode. Training: train_forward, train_backward. "
             "'forward'/'train' are aliases for 'prefill'/'train_forward'.",
    )
    parser.add_argument(
        "--train", action="store_true", default=False,
        help="Trace training phases (train_forward + train_backward). "
             "Equivalent to --phases train_forward train_backward.",
    )

    # ── Capture mode ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--platform",
        default="generic",
        choices=["cuda", "ascend_npu", "cpu", "generic"],
        help="Target inference platform for fusion labelling (default: generic).",
    )
    parser.add_argument(
        "--graph-mode",
        action="store_true",
        default=False,
        help="Use torch.compile graph capture instead of TorchDispatchMode eager tracing.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="Enable activation checkpointing during training phases.",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument("--output-dir", "-o",
                        help="Output directory (default: output/<model_slug>)")

    # ── Parallel strategy (applies to both inference transforms and training modelling) ──
    parser.add_argument(
        "--tp", type=int, default=1,
        help="Tensor-parallel degree (default: 1).",
    )
    parser.add_argument(
        "--pp", type=int, default=1,
        help="Pipeline-parallel degree (default: 1).",
    )
    parser.add_argument(
        "--ep", type=int, default=1,
        help="Expert-parallel degree (default: 1).",
    )
    parser.add_argument(
        "--dp", type=int, default=1,
        help="Data-parallel degree (default: 1).",
    )
    parser.add_argument(
        "--cp", type=int, default=1,
        help="Context-parallel degree (default: 1).",
    )
    parser.add_argument(
        "--quant", default=None,
        metavar="DTYPE",
        help="Weight quantization dtype for analysis: int4, int8, fp8 (default: no quantization)",
    )

    # ── Hardware (triggers perf report / modelling) ───────────────────────────
    parser.add_argument(
        "--hw",
        metavar="HW",
        default=None,
        help="Hardware spec name for performance report (e.g. nvidia_h100_sxm). "
             f"Available: {', '.join(__import__('python.zrt.hardware.registry', fromlist=['list_available']).list_available())}",
    )

    # ── Training modelling extras (used with --train --hw or --estimate-config) ──
    parser.add_argument(
        "--zero-stage", type=int, default=1,
        help="ZeRO optimization stage 0-3 (training, default: 1).",
    )
    parser.add_argument(
        "--optimizer", default="adam",
        choices=["adam", "adamw", "muon"],
        help="Optimizer for training estimation (default: adam).",
    )
    parser.add_argument(
        "--muon-rotation", action="store_true", default=True,
        help="Enable Moonshot rotation optimization for Muon (default: True).",
    )
    parser.add_argument(
        "--muon-ns-steps", type=int, default=None,
        help="Newton-Schulz iteration steps for Muon (default: 5, DSV4: 10).",
    )
    parser.add_argument(
        "--micro-batch", type=int, default=1,
        help="Micro-batch size per GPU (training, default: 1).",
    )
    parser.add_argument(
        "--global-batch", type=int, default=32,
        help="Global batch size across DP ranks (training, default: 32).",
    )
    parser.add_argument(
        "--total-params", type=float, default=None,
        help="Full model param count, e.g. 671e9 (for scaling traced layers).",
    )
    parser.add_argument(
        "--hidden", type=int, default=7168,
        help="Hidden dimension for memory estimation (default: 7168).",
    )
    parser.add_argument(
        "--num-layers-full", type=int, default=None,
        help="Total layers in full model (defaults to --layers if not set).",
    )

    args = parser.parse_args()

    # ── Three independent modes ───────────────────────────────────────────────
    # 1. Spec-based estimation (--estimate-config)
    # 2. Grid search (--search-config)
    # 3. Graph capture + modelling (--model-id or --model)
    if args.estimate_config:
        _run_estimate(args.estimate_config, args.output, breakdown=args.breakdown)
        return

    if args.search_config:
        _run_search(args.search_config, args.output)
        return

    # ── Resolve model_id ──────────────────────────────────────────────────────
    if args.model_id:
        model_id = args.model_id
    elif args.model:
        model_dir_name = _get_model_dirs()[args.model]
        model_id = str(
            Path(__file__).parent.parent.parent / "hf_models" / model_dir_name)
    else:
        parser.error("Provide --model-id or --model v3/v3.2")

    output_dir = Path(args.output_dir) if args.output_dir else None

    # ── Phase resolution: --train > --phases > default ────────────────────────
    if args.train:
        phases = ["train_forward", "train_backward"]
    elif args.phases is not None:
        phases = args.phases
    else:
        phases = ["prefill", "decode"]

    target_layers: Optional[List[int]] = None
    if args.target_layers:
        try:
            target_layers = [int(x.strip()) for x in args.target_layers.split(",")]
        except ValueError:
            parser.error(
                f"--target-layers must be comma-separated integers, "
                f"got: {args.target_layers!r}"
            )

    effective_auto_layers = args.auto_layers or (target_layers is None)

    effective_platform = args.platform
    if effective_platform == "generic" and args.hw:
        import python.zrt.hardware.registry as hw_registry
        _hw = hw_registry.load(args.hw)
        _vendor = getattr(_hw, "vendor", "").lower()
        _device_type = getattr(_hw, "device_type", "").lower()
        if "nvidia" in _vendor or "cuda" in _vendor:
            effective_platform = "cuda"
        elif "huawei" in _vendor or "ascend" in _vendor or _device_type == "npu":
            effective_platform = "ascend_npu"

    result = _run_trace_phases(
        model_id=model_id,
        num_layers=args.layers,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        output_dir=output_dir,
        phases=tuple(phases),
        target_layers=target_layers,
        auto_layers=effective_auto_layers,
        platform=effective_platform,
        graph_mode=args.graph_mode,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    if args.hw:
        import python.zrt.hardware.registry as hw_registry
        hw = hw_registry.load(args.hw)

        if args.train:
            _run_training_modelling(args, model_id, hw, result)
        else:
            _run_inference_pipeline(args, model_id, hw, result)


def _build_model_profile(model_id: str, args) -> "SimpleNamespace":
    """Build a model profile with layer architecture info for report generation.

    Used by both inference and training pipelines so that the HTML report
    correctly distinguishes dense vs. MoE layers.
    """
    from types import SimpleNamespace

    _dense_indices: list[int] = []
    _sparse_indices: list[int] = []
    _n_exp = 0
    _topk = 0

    def _set_from_config_json(_raw: dict) -> bool:
        """Try to infer architecture from config.json fields.

        Returns True if successful, False if fields are missing.
        """
        nonlocal _dense_indices, _sparse_indices, _n_exp, _topk

        _first_k = _raw.get("first_k_dense_replace")
        _freq = _raw.get("moe_layer_freq")

        # V4-style config: no first_k_dense_replace/moe_layer_freq fields.
        # Don't compute — let the transformer API handle it.
        if _first_k is None and _freq is None:
            return False

        _first_k = _first_k or 0
        _freq = _freq or 1
        _total_layers = _raw.get("num_hidden_layers", 61)
        for i in range(_total_layers):
            if i < _first_k:
                _dense_indices.append(i)
            elif (i - _first_k) % _freq == 0:
                _sparse_indices.append(i)
            else:
                _dense_indices.append(i)
        _n_exp = _raw.get("n_routed_experts", 0) or _raw.get(
            "num_local_experts", 0)
        _topk = _raw.get("num_experts_per_tok", 0) or _raw.get(
            "moe_topk", 0) or _raw.get("top_k", 0)
        return True

    try:
        import json as _json2
        _cfg_path = Path(model_id) / "config.json"
        if not _cfg_path.is_absolute():
            _cfg_path = Path.cwd() / _cfg_path

        if _cfg_path.exists():
            with open(_cfg_path) as _f:
                _raw = _json2.load(_f)

            if not _set_from_config_json(_raw):
                # config.json lacks first_k_dense_replace / moe_layer_freq.
                # Fall through to transformer API below.
                _n_exp = _raw.get("n_routed_experts", 0) or _raw.get(
                    "num_local_experts", 0)
                _topk = _raw.get("num_experts_per_tok", 0) or _raw.get(
                    "moe_topk", 0) or _raw.get("top_k", 0)
                _dense_indices = []
                _sparse_indices = []
                # Continue to transformer API fallback below.
            else:
                logger.info(
                    "Architecture (from config.json): %d dense + %d sparse "
                    "layers (total %d), experts=%d, topk=%d",
                    len(_dense_indices), len(_sparse_indices),
                    _raw.get("num_hidden_layers", 61), _n_exp, _topk,
                )
    except Exception as _exc:
        logger.warning("Could not read config.json: %s", _exc)

    # Fallback: use transformers config API (handles V4, Mixtral, etc.)
    if not _dense_indices and not _sparse_indices:
        try:
            from python.zrt.graph.model_loader import infer_layer_types, _load_config
            _cfg, _ = _load_config(model_id)
            _types = infer_layer_types(_cfg)
            _dense_indices = _types["dense"]
            _sparse_indices = _types["sparse"]
            if not _n_exp:
                _n_exp = getattr(_cfg, "n_routed_experts", 0) or getattr(
                    _cfg, "num_local_experts", 0)
            if not _topk:
                _topk = getattr(_cfg, "num_experts_per_tok", 0) or getattr(
                    _cfg, "moe_topk", 0) or getattr(_cfg, "top_k", 0)
            logger.info(
                "Architecture (from transformers API): %d dense + %d sparse "
                "layers (total %d), experts=%d, topk=%d",
                len(_dense_indices), len(_sparse_indices),
                getattr(_cfg, "num_hidden_layers", 0), _n_exp, _topk,
            )
        except Exception as _exc:
            logger.warning("Could not infer layer architecture: %s", _exc)

    return SimpleNamespace(
        num_layers=getattr(args, "num_layers_full", None) or getattr(args, "layers", 0) or 0,
        total_param_count=getattr(args, "total_params", 0) or 0,
        hidden_size=getattr(args, "hidden", 7168) or 7168,
        is_moe=len(_sparse_indices) > 0,
        num_experts=_n_exp,
        moe_topk=_topk,
        dense_layer_indices=_dense_indices,
        sparse_layer_indices=_sparse_indices,
    )


def _run_inference_pipeline(args, model_id: str, hw, result) -> None:
    """Run the inference transform + simulate + report pipeline."""
    from python.zrt.transform import (
        build_default_pipeline, TransformContext,
        ParallelConfig, StreamConfig,
    )
    from python.zrt.transform.context import QuantConfig
    from python.zrt.report import export_reports
    from python.zrt.report.excel_writer import append_perf_summary

    quant = QuantConfig(weight=args.quant, activation=args.quant) if args.quant else None
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(
            tp=args.tp, pp=args.pp, ep=args.ep, dp=args.dp, cp=args.cp,
        ),
        stream_config=StreamConfig(num_compute_streams=1, num_comm_streams=1),
        quant=quant,
    )
    pipe = build_default_pipeline()

    slug = _make_model_slug(model_id)

    profile = _build_model_profile(model_id, args)

    for phase, (raw_graph, _) in result.graphs.items():
        g = pipe.run(raw_graph, ctx)

        # Single call: schedule + simulate + all exports
        try:
            report_dir = result.output_dir / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            rc, flat = export_reports(
                model=model_id, hardware=args.hw, phase=phase,
                batch_size=args.batch_size, seq_len=args.seq_len,
                graph=g, hw_spec=hw, ctx=ctx,
                output_dir=report_dir, slug=slug,
                flat_summary=True,
                profile=profile,
            )
        except Exception as exc:
            logger.warning("Report export failed: %s", exc)
            continue

        # Print console summary + append Excel
        if flat is not None:
            try:
                print(f"\n{flat}")
            except UnicodeEncodeError:
                logger.info("Performance summary: %s", flat)

            xlsx_path = result.output_dir / f"{slug}_{phase}_ops.xlsx"
            if xlsx_path.exists():
                try:
                    append_perf_summary(xlsx_path, flat)
                    logger.info("Performance summary written to %s", xlsx_path)
                except Exception as exc:
                    logger.warning("Excel summary append failed: %s", exc)


def _run_training_modelling(args, model_id: str, hw, result) -> None:
    """Run graph-native training modelling on captured training graphs."""
    from python.zrt.transform.analysis import estimate_training_from_graphs
    from python.zrt.transform.exporter import export_training_graphs

    fwd_pair = result.graphs.get("train_forward")
    if not fwd_pair:
        logger.error("--train --hw requires train_forward phase but none was captured.")
        return

    raw_fwd = fwd_pair[0]
    bwd_pair = result.graphs.get("train_backward")
    raw_bwd = bwd_pair[0] if bwd_pair else None

    if raw_bwd is None:
        logger.warning("No train_backward graph captured; backward metrics will use forward-only fallback.")

    # Load model config for MoE sizing (active experts, total experts)
    _moe_active = 1
    _moe_total = 0
    try:
        import json as _json3
        _cfg_path = Path(model_id) / "config.json"
        if not _cfg_path.is_absolute():
            _cfg_path = Path.cwd() / _cfg_path
        if _cfg_path.exists():
            with open(_cfg_path) as _f:
                _raw_cfg = _json3.load(_f)
            _moe_active = _raw_cfg.get("num_experts_per_tok", 0) or _raw_cfg.get(
                "moe_topk", 0) or _raw_cfg.get("top_k", 0) or 1
            _moe_total = _raw_cfg.get("n_routed_experts", 0) or _raw_cfg.get(
                "num_local_experts", 0) or 0
            if _moe_active > 1:
                logger.info(
                    "MoE config: %d total experts, %d active per token",
                    _moe_total, _moe_active)
    except Exception as _exc:
        logger.warning("Could not read MoE config: %s", _exc)

    report, ctx, transformed = estimate_training_from_graphs(
        forward_graph=raw_fwd,
        backward_graph=raw_bwd,
        output_dir=result.output_dir,
        hw_spec=hw,
        total_params=args.total_params,
        hidden=args.hidden,
        num_layers=args.layers,
        num_layers_full=args.num_layers_full,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        dp=args.dp,
        cp=args.cp,
        zero_stage=args.zero_stage,
        optimizer=args.optimizer,
        muon_rotation=args.muon_rotation,
        muon_ns_steps=args.muon_ns_steps,
        micro_batch=args.micro_batch,
        global_batch=args.global_batch,
        return_transformed=True,
        quant=args.quant,
        moe_total_experts=_moe_total,
        moe_active_experts=_moe_active,
    )

    try:
        print(f"\n{report.summary()}")
    except UnicodeEncodeError:
        logger.info("Training summary:\n%s", report.summary())

    slug = _make_model_slug(model_id)
    output_dir = result.output_dir

    # Export training Excel
    try:
        if "unified" in transformed:
            g = transformed["unified"]
            fwd_for_export = g
            bwd_for_export = g
        else:
            fwd_for_export = transformed.get("train_forward")
            bwd_for_export = None

        if fwd_for_export:
            export_training_graphs(
                fwd_graph=fwd_for_export,
                bwd_graph=bwd_for_export,
                ctx=ctx,
                output_dir=output_dir,
            )
            logger.info("Training Excel exported to %s", output_dir / f"{slug}_training.xlsx")
    except Exception as exc:
        logger.warning("Training Excel export failed: %s", exc)

    # Export training report JSON + hierarchical HTML
    try:
        import json as _json
        report_dir = output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        # JSON dump
        json_path = report_dir / f"{slug}_training_report.json"
        json_path.write_text(_json.dumps(report.to_dict(), indent=2))
        logger.info("Training report written to %s", json_path)

        # Hierarchical HTML + Chrome Trace (single export_reports call)
        train_graph = transformed.get("unified") or transformed.get("train_forward")
        if train_graph is not None:
            from python.zrt.report import export_reports

            cli_profile = _build_model_profile(model_id, args)
            export_reports(
                model=model_id, hardware=args.hw, phase="train",
                batch_size=args.batch_size, seq_len=args.seq_len,
                graph=train_graph, hw_spec=hw, ctx=ctx,
                output_dir=report_dir, slug=slug,
                flat_summary=False,
                profile=cli_profile,
            )
    except Exception as exc:
        logger.warning("Training report export failed: %s", exc)


def _run_estimate(config_path: str, output_path: str | None, *, breakdown: bool = False) -> None:
    """Run spec-based training estimation from a YAML config."""
    from python.zrt.training.io.config_loader import load_specs, load_anchor_config
    from python.zrt.training.search.estimator import estimate
    from python.zrt.training.search.report import report_summary, report_to_json
    from python.zrt.training.ir.builders import build_graph
    from python.zrt.training.models.flops import op_cost as _op_cost, total_training_flops

    try:
        model, system, strategy = load_specs(config_path)
    except (KeyError, TypeError):
        model, system, strategy = load_anchor_config(config_path)

    # Build graph for op-level details, then reuse it in estimate()
    # to avoid duplicate build_graph() calls.
    graph = build_graph(model, strategy)
    op_costs: dict[str, object] = {}
    for op in graph.ops:
        op_costs[op.name] = _op_cost(op, model)

    report = estimate(model, system, strategy, graph=graph)

    if output_path:
        # If output ends with .xlsx, write Excel; otherwise JSON
        if output_path.endswith((".xlsx", ".xls")):
            from python.zrt.training.io.excel_exporter import export_estimate_excel
            export_estimate_excel(
                report=report, graph=graph, model=model,
                system=system, strategy=strategy,
                op_costs=op_costs, output_path=output_path,
            )
            print(f"Excel report written to {output_path}")
        else:
            report_to_json(report, output_path)
            print(f"Report written to {output_path}")
        if breakdown:
            print()
            print(report_summary(report))
    else:
        # Default: write Excel to output/estimate with timestamp
        from datetime import datetime
        from python.zrt.training.io.excel_exporter import export_estimate_excel
        _slug = config_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _default_path = Path("output") / "estimate" / f"{_slug}_{_ts}.xlsx"
        export_estimate_excel(
            report=report, graph=graph, model=model,
            system=system, strategy=strategy,
            op_costs=op_costs, output_path=_default_path,
        )
        print(f"Excel report written to {_default_path}")
        print()
        print(report_summary(report))


def _run_search(config_path: str, output_path: str | None) -> None:
    """Grid-search parallel strategies for a training config."""
    from python.zrt.training.io.config_loader import load_specs
    from python.zrt.training.search.estimator import grid_search, pareto_frontier
    from python.zrt.training.search.space import SearchSpace
    from python.zrt.training.search.report import report_summary, report_to_dict

    model, system, strategy = load_specs(config_path)

    # Preserve config-level batch settings in search space
    space = SearchSpace(
        micro_batch=strategy.micro_batch,
        global_batch=strategy.global_batch,
    )

    print(f"Searching {len(space.strategies(system.world_size))} strategies...")
    reports = grid_search(model, system, space)
    print(f"Found {len(reports)} valid configurations.\n")

    frontier = pareto_frontier(reports)
    print(f"Pareto frontier: {len(frontier)} configurations\n")

    for i, r in enumerate(frontier, 1):
        print(f"--- Frontier config {i} ---")
        print(report_summary(r))
        print()

    if output_path and frontier:
        import json as _json
        frontier_data = [report_to_dict(r) for r in frontier]
        Path(output_path).write_text(_json.dumps(frontier_data, indent=2))
        print(f"Pareto frontier written to {output_path}")


if __name__ == "__main__":
    main()
