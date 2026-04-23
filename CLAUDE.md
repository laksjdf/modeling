# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ZRT-Sim** — an LLM performance modeling and simulation system. Captures the operator sequence of any HuggingFace causal LM using `TorchDispatchMode` inside `FakeTensorMode` (no weights or real memory needed), then applies parallelization transforms and simulates performance across hardware configurations.

## Commands

```bash
# Install
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run a single test function (preferred — see rules below)
pytest tests/test_screenshot_ops.py::TestExtractLayerIdx -v
pytest tests/test_screenshot_ops.py -v -k "deepseek_v3"

# Skip network tests
pytest tests/test_screenshot_ops.py -v -m "not network"

# CLI: trace a model and export Excel
python -m python.zrt Qwen/Qwen2.5-7B-Instruct --layers 4
python -m python.zrt deepseek-ai/DeepSeek-V3-0324 --layers 4 --hw nvidia_h100_sxm --tp 8

# End-to-end validation
python e2e_check.py
python e2e_validate_with_public_data.py
```

## Architecture

Four-stage pipeline, all centered on the `OpGraph` IR:

```
Graph Capture → Transform Pipeline → DAGScheduler → Report Generator
                                          ↑
                              MemoryModel (feasibility)
                              SimulatorHub (latency)
```

**Stage 1 — Graph Capture** (`python/zrt/graph/`)
- `model_loader.py`: loads HF model via `FakeTensorMode` (`torch._subclasses.fake_tensor`) with runtime patches — no real weights or memory allocated
- `dispatch.py` + `tracker.py`: intercept aten ops during forward pass via `TorchDispatchMode` + `ModuleTracker`
- `fusion.py`: two-stage fusion — group by leaf module, then merge up to parent if ≤30 child ops
- `graph_builder.py`: produces `OpGraph` (raw) and fused `OpGraph`
- `patches.py`: MoE meta patch (replaces `.cpu().numpy()` on meta tensors); Indexer patch for DeepSeek V3.2

**Stage 2 — Transform Pipeline** (`python/zrt/transform/`)
- `context.py`: `TransformContext`, `ParallelConfig` (TP/EP/PP/DP/SP), `StreamConfig`
- `pipeline.py`: `build_default_pipeline()` — pluggable pass system
- Pass order: Parallel split (TP → EP → SP → PP) → Comm insertion → Fusion → Optimization (quant, recomp) → Analysis (FLOPs, Roofline)
- Transforms always **clone before mutating** — functional style throughout

**Stage 3 — Executor** (`python/zrt/executor/`)
- `dag_scheduler.py`: topological sort + greedy multi-stream assignment → `Timeline`

**Stage 4 — Simulator** (`python/zrt/simulator/`)
- `SimulatorHub`: fallback chain — Roofline → Regression → ProfileDB → Tiling

**Supporting modules**
- `python/zrt/ir/`: `OpGraph`, `OpNode`, `OpEdge`, `DType`, `TensorMeta`, `GraphHierarchy`
- `python/zrt/hardware/`: `HardwareSpec`, `hw_registry.load("nvidia_h100_sxm")`
- `python/zrt/memory/`: memory feasibility + peak estimation
- `python/zrt/report/`: Excel (6 sheets original, 5 sheets + JSON after transform) + HTML + ONNX

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
- **MoE models**: 2 layers usually sufficient

Local model configs (no weights) live in `hf_models/` (deepseek_v3, llama3_8b, etc.).