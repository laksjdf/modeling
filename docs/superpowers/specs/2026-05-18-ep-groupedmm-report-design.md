# EP GroupedMM and Report Validation Design

## Context

This work is strictly scoped to the graph-capture path for
`hf_models/deepseek_v4`. The project also has a separate spec-based path, but
this design does not change or validate that path. In the graph-capture path,
the pipeline runs:

1. `ExpertParallelPass` annotates routed expert operators.
2. `ExpertGroupedMMPass` fuses routed expert work when `ep > 1`.
3. `CommInserterPass` inserts EP dispatch and combine A2A operators.
4. Analysis and exporter passes produce the final training report artifacts.

The existing EP E2E test passes when run with `PYTHONPATH=python`, but it mainly
checks the in-memory transformed graph. The acceptance target for this work is
the final exported report path: especially the training Excel workbook's
intermediate result sheets, plus the JSON training report generated in the same
output directory.

## Goals

- With EP enabled, routed MoE expert computation is represented as GroupedMM:
  `dispatch -> GroupedMatMul(gate_up) -> silu/swiglu -> GroupedMatMul(down) -> combine`.
- The forward operator order in the exported training Excel is correct.
- GroupedMM nodes carry enough tensor metadata for FLOPs, bytes, latency, and
  formulas to be meaningful in the final report.
- The E2E test uses `hf_models/deepseek_v4` as the benchmark model.
- The final validation reads exported files from a test-owned temporary output
  directory, not pre-existing local output files.

## Non-Goals

- Do not redesign the whole EP rank policy.
- Do not implement a full MoE block IR rewrite.
- Do not require stable committed output artifacts under `output/`.
- Do not validate numerical model outputs; this project traces FakeTensor/meta
  operator graphs and report metrics.
- Do not modify spec-based estimation behavior or its report semantics.

## Proposed Approach

Use the current EP pass structure and make it report-correct.

`ExpertGroupedMMPass` remains between `ExpertParallelPass` and
`CommInserterPass`. It will build one gate/up GroupedMM and one down GroupedMM
per MoE layer block. The fused nodes must include synthetic grouped weight
metadata in addition to the activation input:

- Gate/up input: `(experts_per_rank, tokens_per_expert, hidden)`
- Gate/up weight: `(experts_per_rank, hidden, 2 * routed_ffn)`
- Gate/up output: `(experts_per_rank, tokens_per_expert, 2 * routed_ffn)`
- Down input: `(experts_per_rank, tokens_per_expert, routed_ffn)`
- Down weight: `(experts_per_rank, routed_ffn, hidden)`
- Down output: `(experts_per_rank, tokens_per_expert, hidden)`

For `hf_models/deepseek_v4` with `TP=8, EP=8`, this means
`experts_per_rank=48`, `tokens_per_expert=128*6/384=2`, and
`routed_ffn=3072`. TP is scoped to attention and shared experts; routed experts
are owned by EP and do not divide their intermediate dimension by TP.
`ep_tokens_per_rank` remains useful as an annotation for A2A volume, but it is
not the GroupedMM `M` dimension.

The synthetic weight tensors are not runtime tensors; they are report metadata
so `RooflineSimulator` and `get_op_formulas()` can calculate GroupedMM cost from
the same shapes exported to Excel.

The activation node between the two GroupedMM nodes must inherit the forward
phase and relevant recompute annotation. Only the entry GroupedMM should request
EP A2A insertion. The down GroupedMM remains linked through
`ep_block_down_id` so `CommInserterPass` can place the combine A2A after it.

## Report Validation

The E2E test must create a temporary output directory and run the normal
capture, transform, report, and export path. It then validates the generated
artifacts in that directory:

- `reports/*_training_report.json`
- `*_training.xlsx`

The Excel workbook is the primary source for graph/report checks because it is
the user-facing intermediate report artifact. The test must parse:

- `Forward Operators`
  - Contains `GroupedMatMul` rows.
  - GroupedMM rows have two input shapes: activation and weight.
  - GroupedMM FLOPs and formula columns use the grouped formula, not fallback
    output-element counts.
  - For each forward MoE layer, the row order is:
    dispatch A2A, grouped gate/up, silu/swiglu, grouped down, combine A2A.
- `Communication Ops`
  - Contains EP `all_to_all` rows with roles `dispatch` and `combine`.
  - Dispatch and combine counts match.
  - Group size equals the requested EP degree.
- `Training Summary`
  - Step latency and MFU are present and positive.
  - Parallelism contains the requested TP and EP degrees.

The graph-capture JSON report remains a secondary consistency check. It must exist in the
same temporary output path and contain positive step-time and FLOPs metrics.

## Missing Checks to Add

- Import-path robustness: direct pytest runs currently fail without
  `PYTHONPATH=python` because training modules use `zrt.*` imports. Either keep
  the documented `PYTHONPATH=python` test command or add a test/conftest path
  setup deliberately.
- GroupedMM cost correctness: current sampled output showed GroupedMM with only
  one input tensor, causing fallback FLOPs. Tests must fail if weight metadata is
  missing.
- Phase propagation: the fused activation node must carry forward phase so
  exported forward-only views and phase filters stay stable.
- Final artifact validation: current EP E2E checks in-memory graph objects but
  must assert the exported Excel/JSON contents from a temporary output dir.
- Message bytes vs Excel volume: `Communication Ops` currently writes tensor
  output bytes. If EP `msg_bytes` is the intended report volume, the exporter
  must prefer `attrs["msg_bytes"]` for EP A2A rows, matching
  `CommLatencyPass`.

## Test Command

Use the same interpreter and path setup that works in the local environment:

```powershell
$env:PYTHONPATH='python'; py -m pytest tests\IT\test_ep_e2e.py -q
```

Broader verification must include relevant report/formula tests after the
implementation changes.
