"""Top-level CLI entry point for ZRT-Sim.

Usage::

    python -m python.zrt <model_id> [options]
    python -m python.zrt.graph.main <model_id> [options]  # backward compat

Examples::

    python -m python.zrt Qwen/Qwen2.5-7B-Instruct --layers 4
    python -m python.zrt deepseek-ai/DeepSeek-V3-0324 --layers 4 --hw nvidia_h100_sxm --tp 8
    python -m python.zrt hf_models/llama3_8b --train --layers 2
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from python.zrt.graph.main import (
    run_trace_phases,
    _make_model_slug,
    _MODEL_DIRS,
    _PHASE_ALIASES,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace LLM operator sequences and write Excel + computation graph.")
    parser.add_argument(
        "model_id", nargs="?",
        help="HF Hub model ID or local directory (e.g. deepseek-ai/DeepSeek-V3-0324)")
    parser.add_argument(
        "--model", choices=_MODEL_DIRS.keys(),
        help="Shorthand for local DeepSeek model: v3 or v3.2 (backward compat)")
    parser.add_argument("--layers", type=int, default=4,
                        help="Number of transformer layers to trace (default: 4)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Dummy input batch size (default: 1)")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Prefill sequence length (default: 128)")
    parser.add_argument("--output-dir", "-o",
                        help="Output directory (default: output/graph/<model_slug>)")
    parser.add_argument(
        "--phases", nargs="+", default=["prefill", "decode"],
        choices=["prefill", "decode", "forward",
                 "train_forward", "train_backward", "train"],
        metavar="PHASE",
        help="Phases to trace (default: prefill decode). "
             "Inference: prefill, decode. Training: train_forward, train_backward. "
             "'forward'/'train' are aliases for 'prefill'/'train_forward'.")
    parser.add_argument(
        "--phase", default=None,
        help="(legacy) Trace a single phase. Overrides --phases when set.")
    parser.add_argument(
        "--train", action="store_true", default=False,
        help="Trace training phases (train_forward + train_backward). "
             "Equivalent to --phases train_forward train_backward.")
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
        "--hw",
        metavar="HW",
        default=None,
        help="Hardware spec name for performance report (e.g. nvidia_h100_sxm). "
             f"Available: {', '.join(__import__('python.zrt.hardware.registry', fromlist=['list_available']).list_available())}",
    )
    parser.add_argument(
        "--tp", type=int, default=1,
        help="Tensor-parallel degree used when --hw is set (default: 1).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="Enable activation checkpointing during training phases.",
    )

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
    args = parser.parse_args()

    # Resolve model_id
    if args.model_id:
        model_id = args.model_id
    elif args.model:
        model_dir_name = _MODEL_DIRS[args.model]
        model_id = str(
            Path(__file__).parent.parent.parent / "hf_models" / model_dir_name)
    else:
        parser.error("Provide a model_id argument or --model v3/v3.2")

    output_dir = Path(args.output_dir) if args.output_dir else None

    # Phase resolution: --train > --phase (legacy) > --phases
    if args.train:
        phases = ["train_forward", "train_backward"]
    elif args.phase is not None:
        phases = [args.phase]
    else:
        phases = args.phases

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

    result = run_trace_phases(
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
        from python.zrt.transform import (
            build_default_pipeline, TransformContext,
            ParallelConfig, StreamConfig,
        )
        from python.zrt.executor import DAGScheduler
        from python.zrt.simulator import SimulatorHub
        from python.zrt.report import build_summary
        from python.zrt.graph.excel_writer import append_perf_summary
        import python.zrt.hardware.registry as hw_registry

        hw = hw_registry.load(args.hw)
        ctx = TransformContext(
            hw_spec=hw,
            parallel=ParallelConfig(tp=args.tp),
            stream_config=StreamConfig(num_compute_streams=1, num_comm_streams=1),
        )
        pipe = build_default_pipeline()
        hub = SimulatorHub.default()
        scheduler = DAGScheduler(hw_spec=hw)

        for phase, (raw_graph, _) in result.graphs.items():
            g = pipe.run(raw_graph, ctx)
            tl = scheduler.schedule(g)
            sim_results = hub.simulate_graph(g, hw)
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
                parallel_desc=f"TP{args.tp}",
            )
            try:
                print(f"\n{summary}")
            except UnicodeEncodeError:
                logger.info("Performance summary: %s", summary)

            slug = _make_model_slug(model_id)
            xlsx_path = result.output_dir / f"{slug}_{phase}_ops.xlsx"
            if xlsx_path.exists():
                append_perf_summary(xlsx_path, summary)
                logger.info("Performance summary written to %s", xlsx_path)


if __name__ == "__main__":
    main()
