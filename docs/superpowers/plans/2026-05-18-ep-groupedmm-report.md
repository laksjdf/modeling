# EP GroupedMM Report Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make EP GroupedMM graph-capture output numerically meaningful in the exported training Excel/JSON reports for `hf_models/deepseek_v4`.

**Architecture:** Keep the existing graph-capture transform pipeline order and fix the local contracts between EP fusion, comm insertion, analysis, and export. `ExpertGroupedMMPass` produces report-complete GroupedMM nodes, the graph-capture exporter preserves EP message bytes and graph-native training summaries, and E2E validates final temporary output artifacts. The separate spec-based path is out of scope and must not be modified.

**Tech Stack:** Python 3.14, pytest, openpyxl, ZRT `OpGraph`/`OpNode`, graph-native training estimator, DeepSeek-V4 local HF config.

---

## File Structure

- Modify `python/zrt/transform/parallel/expert_grouped_mm.py`
  - Owns EP routed expert fusion into `GroupedMatMul`.
  - Add synthetic grouped weight tensor metadata.
  - Preserve forward phase on generated activation nodes.
- Modify `python/zrt/transform/exporter.py`
  - Export EP A2A `attrs["msg_bytes"]` as communication volume.
  - Allow `TrainingGraphExcelWriter` to write a `Training Summary` sheet from the graph-capture `TrainingReport` object passed by `estimate_training_from_graphs()`.
- Modify `python/zrt/cli.py`
  - In graph-capture `--train --hw` mode only, pass the graph-native `TrainingReport` into `export_training_graphs()` so CLI output Excel contains `Training Summary`.
- Modify `tests/IT/test_ep_e2e.py`
  - Add artifact export to a test-owned temporary directory.
  - Parse `*_training.xlsx` and `reports/*_training_report.json`.
  - Use the graph-capture training workbook's `Forward Operators` sheet for forward operator checks.
  - Assert GroupedMM shapes, formulas, EP A2A, forward row order, and summary metrics from exported files.
- Do not modify `python/zrt/training/**` spec-path modules for this task.

## Task 1: Failing Excel Artifact Tests

**Files:**
- Modify: `tests/IT/test_ep_e2e.py`

- [ ] **Step 1: Add openpyxl/json/path imports**

Add these imports near the top:

```python
import json
from pathlib import Path

import openpyxl
```

- [ ] **Step 2: Add artifact export fixture**

Add this helper and fixture after `ep1_all`:

```python
def _export_training_artifacts(tmp_path: Path, ep_all):
    from python.zrt.transform.exporter import export_training_graphs

    report, ctx, transformed = ep_all
    out = tmp_path / f"ep{ctx.parallel.ep}"
    out.mkdir(parents=True, exist_ok=True)
    graph = transformed["unified"]

    export_paths = export_training_graphs(
        fwd_graph=graph,
        bwd_graph=graph,
        ctx=ctx,
        output_dir=out,
        training_summary=report,
    )

    report_dir = out / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "deepseek_v4_training_report.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return {
        "output_dir": out,
        "excel": export_paths["excel"],
        "json": json_path,
    }


@pytest.fixture(scope="module")
def ep8_artifacts(tmp_path_factory, ep8_all):
    return _export_training_artifacts(tmp_path_factory.mktemp("ep8_artifacts"), ep8_all)
```

- [ ] **Step 3: Add Excel parsing helpers**

Add these helpers after the artifact fixture:

```python
def _sheet_rows(path: Path, sheet_name: str) -> list[dict[str, object]]:
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    try:
        ws = wb[sheet_name]
        header = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
        rows: list[dict[str, object]] = []
        for values in ws.iter_rows(min_row=2, values_only=True):
            if not any(v is not None for v in values):
                continue
            rows.append(dict(zip(header, values)))
        return rows
    finally:
        wb.close()


def _summary_map(path: Path) -> dict[str, object]:
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    try:
        ws = wb["Training Summary"]
        return {
            str(row[0]): row[1]
            for row in ws.iter_rows(values_only=True)
            if row and row[0] not in (None, "")
        }
    finally:
        wb.close()
```

- [ ] **Step 4: Add failing tests for exported Excel and JSON**

Add these tests to `TestEPE2E`:

```python
    def test_exported_training_artifacts_exist(self, ep8_artifacts):
        assert ep8_artifacts["excel"].exists()
        assert ep8_artifacts["json"].exists()

    def test_exported_excel_grouped_mm_has_weight_inputs_and_formula(self, ep8_artifacts):
        rows = _sheet_rows(ep8_artifacts["excel"], "Transformed Operators")
        grouped = [r for r in rows if r["Op Type"] == "GroupedMatMul"]
        assert grouped, "Expected GroupedMatMul rows in exported Excel"
        for row in grouped:
            input_shapes = str(row["Input Shapes"])
            assert input_shapes.count("(") >= 2, input_shapes
            assert "2" in str(row["FLOPs Formula (sym)"])
            assert "G" in str(row["FLOPs Formula (sym)"])
            assert int(row["FLOPs"]) > int(row["Activation (B)"])

    def test_exported_excel_ep_forward_order(self, ep8_artifacts):
        rows = _sheet_rows(ep8_artifacts["excel"], "Transformed Operators")
        fwd_ep = [
            r for r in rows
            if r["Op Type"] in ("comm.all_to_all", "GroupedMatMul", "aten.silu")
            and (
                r.get("Role") in ("dispatch", "combine")
                or "grouped_" in str(r.get("Node ID", ""))
            )
        ]
        ids = [str(r["Node ID"]) for r in fwd_ep]
        for layer in range(4):
            prefix = f"transformer_layers_{layer}_ffn"
            expected = [
                f"comm_a2a_dispatch_{prefix}_grouped_gate_up",
                f"{prefix}_grouped_gate_up",
                f"{prefix}_grouped_silu",
                f"{prefix}_grouped_down",
                f"comm_a2a_combine_{prefix}_grouped_down",
            ]
            positions = [ids.index(x) for x in expected]
            assert positions == sorted(positions), list(zip(expected, positions))

    def test_exported_excel_ep_a2a_communication(self, ep8_artifacts):
        rows = _sheet_rows(ep8_artifacts["excel"], "Communication Ops")
        ep_rows = [r for r in rows if r["Collective Op"] == "all_to_all"]
        roles = [r["Role"] for r in ep_rows]
        assert roles.count("dispatch") == roles.count("combine") > 0
        for row in ep_rows:
            assert row["Group Size"] == _EP
            assert int(row["Data Volume (bytes)"]) == _BATCH * _SEQ_LEN * _HIDDEN * _MOE_ACTIVE * 2 // _EP

    def test_exported_excel_training_summary(self, ep8_artifacts):
        summary = _summary_map(ep8_artifacts["excel"])
        assert "TP8" in str(summary["Parallelism"])
        assert "EP8" in str(summary["Parallelism"])
        assert float(summary["Step latency (ms)"]) > 0
        assert str(summary["MFU"]).endswith("%")

    def test_exported_json_report_metrics(self, ep8_artifacts):
        data = json.loads(ep8_artifacts["json"].read_text(encoding="utf-8"))
        assert data["step_time_ms"] > 0
        assert data["training_flops"] > 0
        assert data["mfu"] > 0
        assert "GroupedMatMul" in data["fused_ops_summary"]
```

- [ ] **Step 5: Run tests to verify they fail for the right reasons**

Run:

```powershell
$env:PYTHONPATH='python'; py -m pytest tests\IT\test_ep_e2e.py -q
```

Expected before implementation: failures showing missing `Training Summary`, GroupedMM with one input shape, GroupedMM fallback FLOPs, or EP A2A volume using tensor bytes instead of `msg_bytes`.

## Task 2: Report-Complete GroupedMM Nodes

**Files:**
- Modify: `python/zrt/transform/parallel/expert_grouped_mm.py`
- Test: `tests/IT/test_ep_e2e.py`

- [ ] **Step 1: Add grouped weight helper**

Add this helper near `_make_grouped_mm`:

```python
def _weight_tensor(name: str, shape: tuple[int, ...], dtype: DType = DType.BF16) -> TensorMeta:
    return TensorMeta.from_shape_dtype(name, shape, dtype)
```

- [ ] **Step 2: Preserve phase on activation nodes**

In the `act_node` creation block, add:

```python
            phase = gates[0].annotations.get("phase")
            if phase:
                act_node.annotations["phase"] = phase
            if gates[0].annotations.get("recompute"):
                act_node.annotations["recompute"] = True
```

Replace the existing recompute-only block with the snippet above.

- [ ] **Step 3: Build GroupedMM with activation and weight inputs**

Replace the current `gate_up = _make_grouped_mm(...)` input list with:

```python
                [
                    TensorMeta.from_shape_dtype("grouped_in", (G, tokens_per_expert, H_in), DType.BF16),
                    _weight_tensor("grouped_gate_up_weight", (G, H_in, gate_dim + up_dim), DType.BF16),
                ],
```

Replace the current `down = _make_grouped_mm(...)` input list with:

```python
                [
                    TensorMeta.from_shape_dtype("grouped_down_in", (G, tokens_per_expert, ffn), DType.BF16),
                    _weight_tensor("grouped_down_weight", (G, ffn, H_out), DType.BF16),
                ],
```

Derive these dimensions from routed expert matmul nodes, not from quantization
helper nodes under the same expert scope. TP should skip routed expert scopes,
so DeepSeek-V4 uses the full routed expert intermediate size: `w1` is gate,
`w3` is up, and `w2` is down.

- [ ] **Step 4: Add intent annotations for report/debug**

After setting `gate_up.component`, add:

```python
            gate_up.annotations["grouped_mm_role"] = "gate_up"
            gate_up.annotations["ep_tokens_per_rank"] = tokens_per_rank
            gate_up.annotations["ep_tokens_per_expert"] = tokens_per_expert
```

After setting `down.component`, add:

```python
            down.annotations["grouped_mm_role"] = "down"
            down.annotations["ep_tokens_per_rank"] = tokens_per_rank
            down.annotations["ep_tokens_per_expert"] = tokens_per_expert
```

- [ ] **Step 5: Run the focused failing test**

Run:

```powershell
$env:PYTHONPATH='python'; py -m pytest tests\IT\test_ep_e2e.py::TestEPE2E::test_exported_excel_grouped_mm_has_weight_inputs_and_formula -q
```

Expected after this task: the test advances past the two-input assertion and formula assertion. It may still fail on Excel `FLOPs` if exporter/report summary is not fixed yet.

- [ ] **Step 6: Commit**

Run:

```powershell
git add python\zrt\transform\parallel\expert_grouped_mm.py tests\IT\test_ep_e2e.py
git commit -m "fix: export report-complete EP grouped mm"
```

## Task 3: Excel Export Uses EP Message Bytes

**Files:**
- Modify: `python/zrt/transform/exporter.py`
- Test: `tests/IT/test_ep_e2e.py`

- [ ] **Step 1: Update `Transformed Operators` comm volume**

In `_write_transformed_ops_sheet`, replace:

```python
            comm_vol = sum(t.mem_bytes for t in node.outputs) if node.is_comm else ""
```

with:

```python
            comm_vol = ""
            if node.is_comm:
                comm_vol = node.attrs.get("msg_bytes")
                if comm_vol is None:
                    comm_vol = node.attrs.get("bytes")
                if comm_vol is None:
                    comm_vol = sum(t.mem_bytes for t in node.outputs)
```

- [ ] **Step 2: Update `Communication Ops` data volume**

In `_write_communication_sheet`, replace:

```python
            data_volume = sum(t.mem_bytes for t in node.outputs)
```

with:

```python
            data_volume = node.attrs.get("msg_bytes")
            if data_volume is None:
                data_volume = node.attrs.get("bytes")
            if data_volume is None:
                data_volume = sum(t.mem_bytes for t in node.outputs)
```

- [ ] **Step 3: Run the communication artifact test**

Run:

```powershell
$env:PYTHONPATH='python'; py -m pytest tests\IT\test_ep_e2e.py::TestEPE2E::test_exported_excel_ep_a2a_communication -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

Run:

```powershell
git add python\zrt\transform\exporter.py tests\IT\test_ep_e2e.py
git commit -m "fix: export EP message bytes in training workbook"
```

## Task 4: Graph-Capture TrainingReport Sheet in Training Excel

**Files:**
- Modify: `python/zrt/transform/exporter.py`
- Modify: `python/zrt/cli.py`
- Test: `tests/IT/test_ep_e2e.py`

- [ ] **Step 1: Let graph-capture `_write_training_summary_sheet` accept `TrainingReport`**

At the start of `_write_training_summary_sheet`, before it reads `ts.forward_ms`, add:

```python
        if hasattr(ts, "step_time_ms"):
            self._write_training_report_sheet(wb, ts)
            return
```

- [ ] **Step 2: Add `_write_training_report_sheet`**

Add this method immediately below `_write_training_summary_sheet`:

```python
    def _write_training_report_sheet(self, wb: openpyxl.Workbook, report) -> None:
        """Write graph-native TrainingReport metrics as a key-value sheet."""
        ws = wb.create_sheet("Training Summary")
        ws.append(["Training Step Summary"])
        ws["A1"].font = Font(bold=True, size=13)

        rows = [
            ("Model", ""),
            ("Hardware", ""),
            ("Parallelism", report.config_summary),
            ("Batch size", ""),
            ("Sequence length", ""),
            ("", ""),
            ("=== Step Timing ===", ""),
            ("Step latency (ms)", round(report.step_time_ms, 3)),
            ("Pipeline time (ms)", round(report.pipeline_time_ms, 3)),
            ("Compute time (ms)", round(report.compute_time_ms, 3)),
            ("Exposed comm (ms)", round(report.exposed_comm_ms, 3)),
            ("", ""),
            ("=== HW Efficiency ===", ""),
            ("MFU", f"{report.mfu:.2%}"),
            ("HFU", f"{report.hfu:.2%}"),
            ("Total FLOPs (T)", round(report.training_flops / 1e12, 3)),
            ("Forward FLOPs (T)", round(report.forward_flops / 1e12, 3)),
            ("Backward FLOPs (T)", round(report.backward_flops / 1e12, 3)),
            ("", ""),
            ("=== Communication ===", ""),
            ("TP exposed (ms)", round(report.tp_exposed_ms, 3)),
            ("CP exposed (ms)", round(report.cp_exposed_ms, 3)),
            ("EP exposed (ms)", round(report.ep_exposed_ms, 3)),
            ("EP hidden (ms)", round(report.ep_hidden_ms, 3)),
            ("Total comm volume (ms)", round(report.total_comm_volume_ms, 3)),
        ]

        memory = report.memory_breakdown or {}
        if memory:
            rows += [
                ("", ""),
                ("=== Memory (per GPU) ===", ""),
                ("Weights (GB)", round(memory.get("weights", 0) / 1e9, 3)),
                ("Gradients (GB)", round(memory.get("grads", 0) / 1e9, 3)),
                ("Opt states (GB)", round(memory.get("opt_state", 0) / 1e9, 3)),
                ("Activations (GB)", round(memory.get("activations", 0) / 1e9, 3)),
                ("Comm buffers (GB)", round(memory.get("comm_buffers", 0) / 1e9, 3)),
                ("Total (GB)", round(memory.get("total", 0) / 1e9, 3)),
            ]

        for key, val in rows:
            ws.append([key, val])
            if str(key).startswith("==="):
                row_num = ws.max_row
                ws.cell(row=row_num, column=1).font = Font(bold=True)

        ws.column_dimensions["A"].width = 32
        ws.column_dimensions["B"].width = 28
```

- [ ] **Step 3: Pass report into graph-capture CLI Excel export**

In `_run_training_modelling`, change:

```python
            export_training_graphs(
                fwd_graph=fwd_for_export,
                bwd_graph=bwd_for_export,
                ctx=ctx,
                output_dir=output_dir,
                fwd_records=fwd_records,
                bwd_records=bwd_records,
            )
```

to:

```python
            export_training_graphs(
                fwd_graph=fwd_for_export,
                bwd_graph=bwd_for_export,
                ctx=ctx,
                output_dir=output_dir,
                training_summary=report,
                fwd_records=fwd_records,
                bwd_records=bwd_records,
            )
```

- [ ] **Step 4: Run summary artifact test**

Run:

```powershell
$env:PYTHONPATH='python'; py -m pytest tests\IT\test_ep_e2e.py::TestEPE2E::test_exported_excel_training_summary -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add python\zrt\transform\exporter.py python\zrt\cli.py tests\IT\test_ep_e2e.py
git commit -m "fix: include graph training summary in Excel export"
```

## Task 5: Final E2E and Regression Verification

**Files:**
- No new code files unless tests expose a concrete regression.

- [ ] **Step 1: Run EP E2E**

Run:

```powershell
$env:PYTHONPATH='python'; py -m pytest tests\IT\test_ep_e2e.py -q
```

Expected: all EP tests pass.

- [ ] **Step 2: Run formula/export adjacent tests**

Run:

```powershell
$env:PYTHONPATH='python'; py -m pytest tests\report\test_formula_registry.py tests\transform\fusion\test_dsv4_rules.py -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Inspect generated EP artifact assertions if a test fails**

If `test_exported_excel_grouped_mm_has_weight_inputs_and_formula` fails, print the relevant rows:

```powershell
$env:PYTHONPATH='python'; py -c "from pathlib import Path; import openpyxl; p=next(Path('$env:TEMP').rglob('*_training.xlsx')); wb=openpyxl.load_workbook(p, data_only=True, read_only=True); ws=wb['Transformed Operators']; header=[c.value for c in next(ws.iter_rows(min_row=1,max_row=1))]; print(header); [print(r) for r in ws.iter_rows(min_row=2, values_only=True) if r[2]=='GroupedMatMul']; wb.close()"
```

Expected: each GroupedMatMul row has two input shapes and non-fallback grouped formula.

- [ ] **Step 4: Check worktree before final response**

Run:

```powershell
git status --short --branch
```

Expected: only intended files are modified or committed; pre-existing unrelated files remain untouched.

- [ ] **Step 5: Final commit if previous tasks were not committed**

Run:

```powershell
git add python\zrt\transform\parallel\expert_grouped_mm.py python\zrt\transform\exporter.py python\zrt\cli.py tests\IT\test_ep_e2e.py docs\superpowers\specs\2026-05-18-ep-groupedmm-report-design.md docs\superpowers\plans\2026-05-18-ep-groupedmm-report.md
git commit -m "test: validate EP GroupedMM exported reports"
```

Expected: a commit containing only graph-capture EP report validation work and docs. No `python/zrt/training/**` spec-path files are changed.

## Self-Review

- Spec coverage: graph-capture GroupedMM fusion, final Excel/JSON artifacts, forward order, benchmark model, and missing checks all map to tasks above.
- Placeholder scan: no task uses placeholder instructions; each code-edit step includes concrete snippets.
- Type consistency: graph-native `TrainingReport` is detected by `step_time_ms`; existing `TrainingSummary` remains supported by the old branch of `_write_training_summary_sheet`. Spec-path modules are not part of the plan.
