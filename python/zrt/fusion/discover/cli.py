"""Command-line entry point for the discover skill.

Example::

    python -m zrt.fusion.discover \\
        --model-file hf_models/deepseek_v4/inference/model.py \\
        --hf-id deepseek-ai/DeepSeek-V4 \\
        --num-layers 4 --train \\
        --output python/zrt/transform/fusion/rules/deepseek_v4.yaml \\
        --review-out review_deepseek_v4.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from .ast_scanner import scan_model_file
from .joiner import join_rules
from .runtime_tracer import run_runtime_trace


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m zrt.fusion.discover",
        description="Generate fusion-rule drafts from a model.py "
                    "(static AST + optional runtime trace).",
    )
    p.add_argument("--model-file", required=True, type=Path,
                   help="Path to the target model.py source file.")
    p.add_argument("--hf-id", default=None,
                   help="HuggingFace id (or local path) for runtime trace. "
                        "Required unless --skip-runtime is passed.")
    p.add_argument("--num-layers", type=int, default=4,
                   help="Number of transformer layers for the trace.")
    p.add_argument("--train", action="store_true",
                   help="Use train_forward phase instead of prefill "
                        "(needed for DeepSeek-V4 and other kernel-stub models).")
    p.add_argument("--skip-runtime", action="store_true",
                   help="Static-only — skip the runtime trace entirely.")
    p.add_argument("--output", "-o", type=Path,
                   default=Path("rules_draft.yaml"),
                   help="Output YAML path.")
    p.add_argument("--review-out", type=Path, default=None,
                   help="Optional path to write human review notes.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s %(message)s")
    args = _build_parser().parse_args(argv)

    ast_result = scan_model_file(str(args.model_file))

    runtime: dict = {}
    if not args.skip_runtime:
        if not args.hf_id:
            print("ERROR: --hf-id is required unless --skip-runtime is used.",
                  file=sys.stderr)
            return 2
        phase = "train_forward" if args.train else "prefill"
        runtime = run_runtime_trace(args.hf_id,
                                    num_layers=args.num_layers,
                                    phase=phase)

    rules, notes = join_rules(ast_result, runtime)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(rules, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {len(rules)} rules → {args.output}")

    if args.review_out is not None:
        args.review_out.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(f"- {n}" for n in notes) or "_(no notes)_"
        args.review_out.write_text(
            f"# Discover review notes\n\nSource: `{args.model_file}`\n\n{body}\n",
            encoding="utf-8",
        )
        print(f"Wrote {len(notes)} review notes → {args.review_out}")
    elif notes:
        print(f"\nReview notes ({len(notes)}):")
        for n in notes:
            print(f"  - {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
