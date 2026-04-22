"""CLI: python -m zrt.training.cli estimate --config config.yaml"""

from __future__ import annotations

import argparse
import sys

from zrt.training.io.config_loader import load_specs
from zrt.training.search.estimator import estimate
from zrt.training.search.report import report_summary, report_to_json


def main():
    parser = argparse.ArgumentParser(
        description="AI Training Infra Modeller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # estimate subcommand
    est = sub.add_parser("estimate", help="Estimate training performance for a config")
    est.add_argument("--config", required=True, help="Path to YAML config file")
    est.add_argument("--output", "-o", default=None, help="Output JSON path (default: stdout summary)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "estimate":
        _cmd_estimate(args.config, args.output)


def _cmd_estimate(config_path: str, output_path: str | None) -> None:
    model, system, strategy = load_specs(config_path)
    report = estimate(model, system, strategy)

    if output_path:
        report_to_json(report, output_path)
        print(f"Report written to {output_path}")
    else:
        print(report_summary(report))


if __name__ == "__main__":
    main()
