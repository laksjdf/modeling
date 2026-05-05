# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ZRT-Sim** — an LLM performance modeling and simulation system. Captures the operator sequence of any HuggingFace causal LM using `TorchDispatchMode` inside `FakeTensorMode` (no weights or real memory needed), then applies parallelization transforms and simulates performance across hardware configurations.

**Tech Stack**: Python 3.14+, PyTorch 2.0+, transformers 4.36+ (version-agnostic via compat shims), networkx, openpyxl, onnx

**Hot Paths** (frequently accessed files): `python/zrt/transform/analysis/modeller.py`, `python/zrt/training/compose/schedules.py`, `python/zrt/transform/context.py`, `tests/training/test_captured_graph_modelling.py`, `python/zrt/transform/analysis/training.py`

## Commands

```bash
# Install
pip install -r requirements.txt

# Run all tests
pytest tests/ -v 2>&1 | tail -n 50

# Run a single test function (preferred)
pytest tests/test_transform.py::test_tp_shape_modification -v
pytest tests/test_executor.py -v -k "overlap"
pytest tests/test_train_trace.py -v

# Run specific test files
pytest tests/test_transform.py tests/test_executor.py tests/test_simulator.py -v 2>&1 | tail -n 50

# Run training-specific tests
PYTHONPATH=python pytest tests/training/test_captured_graph_modelling.py -v
PYTHONPATH=python pytest tests/training/test_1f1b.py -v
PYTHONPATH=python pytest tests/training/test_graph_schedule.py -v     # PP schedule dispatch (VPP/DualPipe)
PYTHONPATH=python pytest tests/training/test_dualpipe.py -v            # DualPipe/DualPipeV composers
PYTHONPATH=python pytest tests/training/test_stream_overlap.py -v      # CoC/MC2 overlap
PYTHONPATH=python pytest tests/training/test_chrome_trace.py -v        # Chrome trace export
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -v     # MFU anchor regression (YAML fixtures)
PYTHONPATH=python pytest tests/training/test_anchor.py -v              # AnchorValidator unit tests

# Skip network tests
pytest tests/test_screenshot_ops.py -v -m "not network"

# CLI: trace a model and export Excel
python -m python.zrt Qwen/Qwen2.5-7B-Instruct --layers 4
python -m python.zrt deepseek-ai/DeepSeek-V3-0324 --layers 4 --hw nvidia_h100_sxm --tp 8

# Training modeling (captures train_forward + train_backward + estimates performance)
python -m python.zrt deepseek-ai/DeepSeek-V3 --layers 4 --train --hw nvidia_h100_sxm --tp 8 --pp 2
PYTHONPATH=python python -m zrt.training estimate --config python/zrt/training/configs/llama3_70b_3d.yaml
# Export Chrome Trace alongside training estimate
PYTHONPATH=python python -m zrt.training estimate --config python/zrt/training/configs/llama3_70b_3d.yaml --trace out.json

# Spec-based training estimate (no graph capture needed)
python -m python.zrt --estimate-config python/zrt/training/configs/llama3_70b_3d.yaml

# Graph-mode capture (torch.compile backend instead of TorchDispatchMode)
python -m python.zrt deepseek-ai/DeepSeek-V3 --layers 2 --phases train_backward --graph-mode

# End-to-end validation
python e2e_check.py
```

> **Note**: Training module commands require `PYTHONPATH=python` because the training subpackage uses `zrt.*` imports rather than `python.zrt.*`.

## Architecture

Four-stage pipeline, all centered on the `OpGraph` IR:

```
Graph Capture → Transform Pipeline → DAGScheduler → Report Generator
                                          ↑
                              MemoryModel (feasibility)
                              SimulatorHub (latency)
```

**Stage 1 — Graph Capture** (`python/zrt/graph/`)
- `model_loader.py`: loads HF model via `FakeTensorMode` — no real weights or memory allocated
- `dispatch.py` + `tracker.py`: intercept aten ops during forward pass via `TorchDispatchMode` + `ModuleTracker`
- `fusion.py`: two-stage fusion — group by leaf module, then merge up to parent if ≤30 child ops
- `graph_builder.py`: produces raw `OpGraph` and fused `OpGraph`
- `patches.py`: MoE meta patch (replaces `.cpu().numpy()` on meta tensors); Indexer patch for DeepSeek V3.2
- `compat.py`: version shims for transformers 4.x vs 5.x API differences

**Stage 2 — Transform Pipeline** (`python/zrt/transform/`)
- `context.py`: `TransformContext`, `ParallelConfig` (TP/EP/PP/DP/SP), `StreamConfig`
- `pipeline.py`: `build_default_pipeline()` — pluggable pass system
- `parallel/`: tensor_parallel.py, expert_parallel.py, pipeline_parallel.py, data_parallel.py, context_parallel.py, comm_inserter.py
- `fusion/`: fusion passes based on software stack rules
- `analysis/`: FLOPs, Roofline, communication latency, training modeling (modeller.py, training.py)
- `optim/`: optimization passes (quant, recomp, EPLB, MTP)
- `training/`: training-specific graph passes — `RecomputePass` (activation recompute annotation), `OffloadPass`, `OptimizerPass`, `ZeroFSDPPass`
- Pass order: Parallel split (TP → EP → SP → PP) → Comm insertion → Fusion → Optimization → Analysis
- Transforms always **clone before mutating** — functional style throughout

**Stage 3 — Executor** (`python/zrt/executor/`)
- `scheduler.py`: topological sort + greedy multi-stream assignment → `Timeline`
- `overlap.py`: compute-communication overlap analysis

**Stage 4 — Simulator** (`python/zrt/simulator/`)
- `hub.py`: `SimulatorHub` with fallback chain — Roofline → Regression → ProfileDB → Tiling
- `backends/`: pluggable simulator implementations

**Supporting modules**
- `python/zrt/ir/`: `OpGraph`, `OpNode`, `OpEdge`, `DType`, `TensorMeta`, `GraphHierarchy` — core IR types
- `python/zrt/hardware/`: `HardwareSpec`, `hw_registry.load("nvidia_h100_sxm")` — YAML-based configs in `configs/`
- `python/zrt/memory/`: memory feasibility + peak estimation
- `python/zrt/layers/`: abstract operator type layer — `OperatorBase` subclasses (mm, attention, comm, activation, embedding, quant, fused, triton) used by simulator backends for cost modeling
- `python/zrt/policy_model/`: `PolicyModelManager` — dispatches `OpNode` simulation to a registered cost-model policy (`PolicyType`); pluggable via `POLICY_MAP` in `policy_register.py`
- `python/zrt/report/summary.py`: Excel/HTML report generation; `chrome_trace.py` builds Chrome Trace JSON from a `Timeline`
- `python/zrt/training/`: training performance estimation — see **Training Module** section below

## Training Module (`python/zrt/training/`)

The training module is self-contained and substantially larger than the rest; it has its own IR, spec types, and pipeline composers.

**Sub-packages:**
- `spec/` — `ModelSpec` (geometry + layer list), `Strategy` (TP/CP/PP/EP/DP + ZeRO + recompute), `SystemSpec` (cluster hardware), `Dtype`, and enums:
  - `PPSched`: `ONE_F_ONE_B` / `INTERLEAVED` / `ZERO_BUBBLE` / `DUALPIPE` / `DUALPIPE_V`
  - `CPKind`: `ULYSSES` / `RING` / `HYBRID`
  - `TPOverlap`: `COC` / `MC2`
  - `OptKind`: `ADAM` / `MUON`
  - `Strategy.attn_compression_ratio`: scales attention FLOPs for sparse/compressed attention variants
- `ir/` — training-side `Graph` (layer shards + stage assignment), `builders`, `validate`
- `models/` — `comm.py` (collective time), `flops.py` (`recompute_overhead_flops()` for selective recompute categories), `memory.py` (`MemBreakdown`)
- `compose/` — `PipelineComposer` ABC + five concrete composers: `OneF1BComposer`, `InterleavedComposer` (VPP), `ZeroBubbleComposer`, `DualPipeComposer`, `DualPipeVComposer`. Each returns a `StepResult` with fields: `step_time`, `bubble_fraction`, `mfu`, `hfu`, `memory`.
  - **MFU vs HFU**: MFU excludes recompute overhead FLOPs; HFU includes them. `hfu > mfu` whenever selective recompute is active.
- `search/` — `SearchSpace` (grid over TP/CP/PP/EP/DP/ZeRO/PPSched/vpp_chunks) + `SearchEstimator` → Pareto-front `SearchReport`
- `anchor/` — `AnchorValidator` reads YAML fixtures from `tests/training/anchors/` and checks MFU within tolerance; used for regression testing
- `trace/` — `ChromeTraceExporter` converts a `Timeline` into Chrome Trace JSON (loaded via `chrome://tracing`)
- `io/` — `config_loader.py` (YAML → `ModelSpec`/`Strategy`/`SystemSpec`), `perf_tables.py`

**PP schedule dispatch** (`transform/analysis/training.py:285-298`): `ctx.training.pp_schedule` selects among interleaved/dualpipev/dualpipe/zb/1f1b. The unified path in `modeller.py:307-343` reads `pipeline_metrics.step_time_ms` directly from the chosen composer instead of recomputing.

**Anchor YAML fixtures** (`tests/training/anchors/*.yaml`): GPT-3 175B, LLaMA-3 70B, DeepSeek-V3 — each pins expected MFU and step time to guard against regressions. Run with `pytest tests/training/anchors/test_anchors.py`.

## Key Rules (from .clauderules)

### Token Compression
- **Never dump full test logs.** Always pipe through `2>&1 | tail -n 50` or `grep` for errors.
- Run single test functions, not whole files.
- Before running prefill/decode simulations, check NumPy version and Python path.

### Session State
- After each bug fix or feature implementation (tests passing), update `SESSION_PROGRESS.md` with: current file state, resolved issues, next steps.
- After updating, archive to `SESSION_HISTORY.md` (append with timestamp, no need to read first).
- Remind user: **"进度已同步至存档，请执行 /clear 开启新会话或者执行 /compact 压缩成摘要以节省 Token。"**

### Fixed Task Workflow
1. Read `SESSION_PROGRESS.md` and continue from last state
2. Execute task
3. Append `SESSION_PROGRESS.md` content (timestamped) to `SESSION_HISTORY.md`
4. Update `SESSION_PROGRESS.md` with new state

### Error Handling
- Before retrying a fix, check `SESSION_PROGRESS.md` for previously attempted (and failed) solutions.

## Model-Specific Notes

- **Dense models** (Llama, Qwen2, Mistral): 2 layers captures all operator patterns
- **DeepSeek-V3 / V3.2**: 4 layers needed (first 3 dense, layer 4 is MoE); requires `trust_remote_code`
- **MoE models** (Mixtral, Qwen3-MoE): 2 layers usually sufficient (first layer is already MoE)

Local model configs (no weights) live in `hf_models/` (deepseek_v3, llama3_8b, etc.).

## Configuration Files

- **Hardware specs**: `python/zrt/hardware/configs/*.yaml` — H100, A100, H800, Ascend 910B/C
  - Define compute (TFLOPS), memory (HBM bandwidth, capacity), interconnect (HCCS, RoCE)
- **Training configs**: `python/zrt/training/configs/*.yaml` — model + hardware + strategy combinations
  - Reference `models/` subdir for model profiles, use `hw_registry` for hardware
- **Software stack rules**: Fusion rules are software-stack-specific (MindIE, vLLM, etc.)

## Important Implementation Patterns

**Transform Pipeline Extensibility**: All transforms inherit from `TransformPass` base class, implement `__call__(ctx, graph) → graph`. Add new passes to `build_default_pipeline()` in `pipeline.py`.

**IR Mutability**: OpGraph nodes are immutable (frozen dataclasses) after creation. Transforms create new nodes/subgraphs and use `graph.clone()` for full copies.

**Hardware Registry Pattern**: `hw_registry.load(name)` loads YAML configs into `HardwareSpec` objects. Hardware details never hardcode in transform logic.

**Training Memory Estimation**: Training memory calculation uses per-component formulas (weights, gradients, optimizer states, activations) with ZeRO staging (0-3) and gradient checkpointing support.

**Recompute annotation flow**: `RecomputePass` (in `transform/training/recompute.py`) annotates forward-graph nodes with `node.annotations["recompute"] = True` scoped by `layer_kind`. Downstream, `TrainingFlopsPass` extracts `recompute_flops` from these annotations and `compose/pipeline.py::compute_hfu()` uses it to separate MFU from HFU.

## Design Documentation

Key design docs in `docs/`:
- `training_modeller_zh.md` — Training performance modeling design
- `phase0_improvement_plan.md` — Phase 0 improvements (forward/backward graph stitching)
- `training_integration_design.md` — Training integration architecture

> **Note on `ARCHITECTURE.md`**: This is an aspirational V2 design document describing a planned refactoring. Its module paths (`zrt/capture/`, `zrt/stacks/`, `zrt/comm/`, etc.) do **not** match the current codebase. The actual code lives under `python/zrt/` as described in the Architecture section above.

## Coding Conventions

- **Naming**: `snake_case` for files/functions/variables, `PascalCase` for classes
- **Testing**: Use `pytest`, name tests `test_<behavior>.py`, test functions `test_<expected_behavior>()`
- **Imports**: Group standard library, third-party, local imports; keep consistent with surrounding code
- **Commits**: Short, imperative subjects (e.g., `修复import报错`, `bug fix phase1`), include rationale for non-trivial changes
