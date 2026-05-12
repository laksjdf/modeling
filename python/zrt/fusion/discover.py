"""Discover fusion rules from model forward pass.

Usage::

    PYTHONPATH=python python -m zrt.fusion.discover \\
        hf_models/deepseek_v4/inference/model.py \\
        --output rules_draft.yaml \\
        --num-layers 4

The tool captures the model's forward pass under FakeTensorMode,
groups aten ops by leaf module class, extracts observed op sequences,
and infers input/output sources to generate YAML rule drafts.

Output uses full aten op names in ``observed_op_seqs`` and matches
the schema expected by ``ModuleFusionRule.from_yaml_dict()``.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def discover_fusion_rules(model_id: str, num_layers: int,
                          training: bool = False) -> list[dict[str, Any]]:
    """Run model trace and extract fusion rule drafts.

    Only returns groups with **2+ ops** (single-op groups don't need fusion).

    When *training* is True, uses training mode (``train_forward`` phase)
    which applies ``patch_for_training_capture`` to handle kernel stubs
    and inference-mode decorators.  Required for V4 models.
    """
    from zrt.graph import run_trace_phases

    phase = "train_forward" if training else "prefill"
    output_dir, phase_records = run_trace_phases(
        model_id=model_id,
        num_layers=num_layers,
        phases=(phase,),
        output_dir=Path("/tmp/zrt_fusion_discover"),
    )

    records = phase_records.get(phase, [])

    # Group consecutive ops by module_class
    runs: list[tuple[str, str, tuple[str, ...]]] = []
    current_mc = None
    current_seq: list[str] = []

    for rec in records:
        mc = rec.get("module_class", "")
        ot = rec.get("op_type", "")
        if mc != current_mc:
            if current_mc and current_seq:
                runs.append((current_mc, "", tuple(current_seq)))
            current_mc = mc
            current_seq = [ot]
        else:
            current_seq.append(ot)
    if current_mc and current_seq:
        runs.append((current_mc, "", tuple(current_seq)))

    # Deduplicate: module_class → set of sequences
    seqs_by_class: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for mc, _, seq in runs:
        seqs_by_class[mc].add(seq)

    # Build rule drafts — only multi-op sequences
    rules: list[dict[str, Any]] = []
    for mc, seqs in sorted(seqs_by_class.items()):
        multi_op_seqs = [list(s) for s in sorted(seqs) if len(s) >= 2]
        if not multi_op_seqs:
            continue

        rule: dict[str, Any] = {
            "target_class": mc,
            "op_type": mc,
            "observed_op_seqs": multi_op_seqs,
            "inputs": [
                {"role": "activation", "source_op_index": 0, "source_arg_index": 0, "source_kind": "input"},
            ],
            "outputs": [
                {"role": "output", "source_op_index": -1, "source_arg_index": -1, "source_kind": "output"},
            ],
            "priority": 20,
        }
        rules.append(rule)

    return rules


def main():
    parser = argparse.ArgumentParser(
        description="Discover fusion rules from model forward pass"
    )
    parser.add_argument(
        "model_id",
        help="HF model ID or local path",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("rules_draft.yaml"),
        help="Output YAML file path",
    )
    parser.add_argument(
        "--num-layers", "-n",
        type=int,
        default=4,
        help="Number of transformer layers to trace",
    )
    parser.add_argument(
        "--rules-dir",
        type=Path,
        default=None,
        help="Write to the fusion rules directory instead of --output",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Trace in training mode (needed for V4 and other models with kernel stubs)",
    )

    args = parser.parse_args()

    rules = discover_fusion_rules(args.model_id, args.num_layers,
                                  training=args.train)

    # Choose output path
    if args.rules_dir:
        args.rules_dir.mkdir(parents=True, exist_ok=True)
        model_key = args.model_id.lower().replace("/", "_").replace("-", "_")
        out_path = args.rules_dir / f"{model_key}.yaml"
    else:
        out_path = args.output

    with open(out_path, "w") as f:
        yaml.dump(rules, f, default_flow_style=False, sort_keys=False)

    print(f"Discovered {len(rules)} rules -> {out_path}")
    for rule in rules[:10]:
        n_seqs = len(rule["observed_op_seqs"])
        print(f"  - {rule['target_class']}: {n_seqs} op sequences")


if __name__ == "__main__":
    main()
