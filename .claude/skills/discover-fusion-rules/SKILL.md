---
name: discover-fusion-rules
description: Generate fusion-rule YAML drafts for a new model by combining static AST scanning of `model.py` with optional runtime aten-op tracing. Use when adding fusion support for a new HF model (DeepSeek-V3.2/V4, new MoE variants), regenerating a stale rules file, or seeding the rich-rule schema from an existing legacy YAML.
---

# Skill: Discover Fusion Rules

This skill produces a draft fusion-rule YAML by joining two complementary signals:

1. **Static AST** of the model source (`hf_models/<model>/inference/model.py`) — gives class names, `__init__` parameters, the high-level callables used in each `forward`, submodule wiring (`self.X`), and `dist.*` collective usage.
2. **Runtime trace** via `run_trace_phases` — gives the actual aten op sequence each `module_class` emits under `FakeTensorMode`. Required to populate `match.op_regexes` for `ordered_regex` rules.

Neither signal alone is enough: AST cannot see aten serialization, runtime cannot see semantic intent. The joiner combines them with per-class templates from `zrt.fusion.discover.templates.TEMPLATES` to emit a schema-valid YAML draft plus a list of human review notes.

## When to use

- Adding fusion support for a new HF model (e.g. DeepSeek-V4, a new MoE variant, custom attention).
- The matching/semantics on an existing rules file feels stale (model surface changed).
- You want to bootstrap the rich-rule format from scratch and let templates fill in `flops_formula` / `shape_derivation` for known patterns.

Skip this skill for trivial one-off tweaks — edit the YAML directly.

## Quick start

```bash
# Full path: AST + runtime trace (requires HF env)
PYTHONPATH=python python -m zrt.fusion.discover \
    --model-file hf_models/deepseek_v4/inference/model.py \
    --hf-id deepseek-ai/DeepSeek-V4 \
    --num-layers 4 --train \
    --output python/zrt/transform/fusion/rules/deepseek_v4.yaml \
    --review-out review_deepseek_v4.md

# Static-only (no runtime, ordered_regex templates degrade to class_only)
PYTHONPATH=python python -m zrt.fusion.discover \
    --model-file hf_models/deepseek_v3_2/inference/model.py \
    --skip-runtime \
    --output /tmp/deepseek_v32_draft.yaml
```

`--train` is **required** for DeepSeek-V4 because its inference path uses kernel stubs that only resolve under the `train_forward` capture phase.

## What gets produced

For each `nn.Module` subclass and each known top-level helper (`apply_rotary_emb`, `sparse_attn`), the skill emits one YAML rule:

```yaml
- target_class: RMSNorm
  op_type: rms_norm
  priority: 20
  match:
    kind: ordered_regex
    op_regexes:
      - "aten\\.pow\\.Tensor_Scalar"
      - "aten\\.mean\\.dim"
      - "aten\\.add\\.Scalar"
      - "aten\\.rsqrt\\.default"
      - "aten\\.mul\\.Tensor"
      - "aten\\.mul\\.Tensor"
    min_ops: 6
  io_roles:
    - {role: activation, source_kind: input,  source_op_index: 0,  source_arg_index: 0,  shape_role: "[B,S,H]"}
    - {role: weight,     source_kind: input,  source_op_index: -1, source_arg_index: 0,  shape_role: "[H]"}
    - {role: output,     source_kind: output, source_op_index: -1, source_arg_index: -1, shape_role: "[B,S,H]"}
  shape_derivation:
    batch_size: "activation.shape[0]"
    seq_len:    "activation.shape[1]"
    hidden_in:  "activation.shape[-1]"
    hidden_out: "output.shape[-1]"
  flops_formula:  "4 * batch_size * seq_len * hidden_in"
  memory_formula: "activation.bytes + weight.bytes + output.bytes"
  annotations:
    layer_norm_kind: rms
```

Rules are sorted **priority desc → name asc** so high-priority leaves (Compressor, MoE, Attention) get matched before catch-alls.

## Template coverage

`TEMPLATES` covers 16 patterns out-of-the-box (see `python/zrt/fusion/discover/templates.py`):

| Group | Classes / helpers |
|---|---|
| Norms | `RMSNorm`, `LayerNorm` (+ aliases: `LlamaRMSNorm`, `Qwen2RMSNorm`, `MistralRMSNorm`) |
| Linear | `Linear`, `ColumnParallelLinear`, `RowParallelLinear`, `ParallelEmbedding` |
| MoE | `Gate`, `Expert`, `MoE`, `MLP` |
| Attention | `MLA`, `Attention`, `Indexer`, `Compressor` |
| Helpers | `apply_rotary_emb`, `sparse_attn` |
| Block | `Block` (low priority — defers to inner-rule matching) |
| Fallback | `_default` (class_only, no formulas — appears as a review note) |

If a class isn't in this table, the joiner uses `_default` and emits a review note like:

> `MyMystery: fell back to _default template — please review flops_formula / shape_derivation.`

## Review-and-merge workflow

1. Run the skill, write to a scratch path (`/tmp/<model>_draft.yaml`).
2. Read `--review-out` notes; for each entry, hand-fill the missing `flops_formula` / `shape_derivation`.
3. Diff against the existing rules file (if any) and merge.
4. Move the file into `python/zrt/transform/fusion/rules/<model>.yaml`.
5. Run the matching tests: `pytest tests/transform/fusion/ -v`.

## Programmatic API

```python
from zrt.fusion.discover import discover_fusion_rules

rules, notes = discover_fusion_rules(
    model_file="hf_models/deepseek_v3_2/inference/model.py",
    hf_id="deepseek-ai/DeepSeek-V3.2",
    train=False,
    num_layers=4,
)
```

Lower-level entry points (each independently unit-testable):

- `scan_model_file(path)` → `AstScanResult`
- `run_runtime_trace(model_id, *, num_layers, phase)` → `dict[module_class, list[seq]]`
- `join_rules(ast, runtime)` → `(rules, review_notes)`
- `get_template(class_name)` → template dict or `None`

## Limitations

- AST cannot see attribute aliases (`self.norm = self.layers[0].norm`) — the joiner does **not** chase these.
- The runtime trace is single-phase; if your model has different op sequences in prefill vs. decode you must run the skill twice and merge by hand.
- The default flops formulas are correct order-of-magnitude only; verify against a Roofline anchor before trusting them in performance reports.
