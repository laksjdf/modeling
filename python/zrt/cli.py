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
#   from python.zrt.graph.main import run_trace_phases, _make_model_slug, _MODEL_DIRS, _PHASE_ALIASES


def _get_model_dirs():
    """Lazy import of _MODEL_DIRS to avoid requiring torch at module load time."""
    from python.zrt.graph.main import _MODEL_DIRS
    return _MODEL_DIRS


def _make_model_slug(model_id: str) -> str:
    """Lazy import of _make_model_slug to avoid requiring torch at module load time."""
    from python.zrt.graph.main import _make_model_slug as _impl
    return _impl(model_id)


def _run_trace_phases(**kwargs):
    """Lazy import of run_trace_phases to avoid requiring torch at module load time."""
    from python.zrt.graph.main import run_trace_phases
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
        "--output",
        metavar="FILE",
        help="Write estimation/search result as JSON to FILE (used with --estimate-config or --search-config).",
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
        _run_estimate(args.estimate_config, args.output)
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

    result = _run_trace_phases(
        model_id=model_id,
        num_layers=args.layers,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        output_dir=output_dir,
        phases=tuple(phases),
        target_layers=target_layers,
        auto_layers=effective_auto_layers,
        platform=args.platform,
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


def _run_inference_pipeline(args, model_id: str, hw, result) -> None:
    """Run the inference transform + simulate + report pipeline."""
    from python.zrt.transform import (
        build_default_pipeline, TransformContext,
        ParallelConfig, StreamConfig,
    )
    from python.zrt.transform.context import QuantConfig
    from python.zrt.executor import DAGScheduler
    from python.zrt.simulator import SimulatorHub
    from python.zrt.report import build_summary, export_html_report, export_chrome_trace
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
    hub = SimulatorHub.default()
    scheduler = DAGScheduler(hw_spec=hw)

    slug = _make_model_slug(model_id)
    report_dir = result.output_dir / "reports"

    for phase, (raw_graph, _) in result.graphs.items():
        g = pipe.run(raw_graph, ctx)
        tl = scheduler.schedule(g)
        sim_results = hub.simulate_graph(g, hw)

        parallel_desc = ctx.parallel.describe()

        summary = build_summary(
            model=model_id,
            hardware=args.hw,
            phase=phase,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            graph=g,
            sim_results=sim_results,
            timeline=tl,
            hw_spec=hw,
            parallel_desc=parallel_desc,
        )
        try:
            print(f"\n{summary}")
        except UnicodeEncodeError:
            logger.info("Performance summary: %s", summary)

        xlsx_path = result.output_dir / f"{slug}_{phase}_ops.xlsx"
        if xlsx_path.exists():
            append_perf_summary(xlsx_path, summary)
            logger.info("Performance summary written to %s", xlsx_path)

        # Auto-export HTML + Chrome Trace
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            export_html_report(
                summary, report_dir / f"{slug}_{phase}_report.html",
                timeline_data=[
                    {"start": op.start_us, "end": op.end_us,
                     "stream": op.stream_id, "type": op.stream_type}
                    for op in tl.scheduled_ops
                ],
            )
            export_chrome_trace(
                tl, report_dir / f"{slug}_{phase}_trace.json",
                name=f"{model_id} | {phase}",
                metadata={"model": model_id, "hardware": args.hw,
                          "phase": phase, "parallel": parallel_desc},
            )
        except Exception as exc:
            logger.warning("Report export failed: %s", exc)


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

    report, ctx, transformed = estimate_training_from_graphs(
        forward_graph=raw_fwd,
        backward_graph=raw_bwd,
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

    # Export training report JSON
    try:
        import json as _json
        report_dir = output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{slug}_training_report.json"
        json_path.write_text(_json.dumps(report.to_dict(), indent=2))
        logger.info("Training report written to %s", json_path)
    except Exception as exc:
        logger.warning("Report export failed: %s", exc)


def _run_estimate(config_path: str, output_path: str | None) -> None:
    """Run spec-based training estimation from a YAML config."""
    from python.zrt.training.io.config_loader import load_specs
    from python.zrt.training.search.estimator import estimate
    from python.zrt.training.search.report import report_summary, report_to_json

    model, system, strategy = load_specs(config_path)
    report = estimate(model, system, strategy)

    if output_path:
        report_to_json(report, output_path)
        print(f"Report written to {output_path}")
    else:
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
