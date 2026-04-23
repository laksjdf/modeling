# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ZRT-Sim** — an LLM performance modeling and simulation system. Captures the operator sequence of any HuggingFace causal LM using `TorchDispatchMode` inside `FakeTensorMode` (no weights or real memory needed), then applies parallelization transforms and simulates performance across hardware configurations.

**Tech Stack**: Python 3.14+, PyTorch 2.0+, transformers 4.36+ (version-agnostic via compat shims), networkx, openpyxl, onnx

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
pytest tests/training/test_captured_graph_modelling.py -v
pytest tests/training/test_1f1b.py -v

# Skip network tests
pytest tests/test_screenshot_ops.py -v -m "not network"

# CLI: trace a model and export Excel
python -m python.zrt Qwen/Qwen2.5-7B-Instruct --layers 4
python -m python.zrt deepseek-ai/DeepSeek-V3-0324 --layers 4 --hw nvidia_h100_sxm --tp 8

# Training modeling (captures train_forward + train_backward + estimates performance)
python -m python.zrt deepseek-ai/DeepSeek-V3 --layers 4 --train --hw nvidia_h100_sxm --tp 8 --pp 2
PYTHONPATH=python python -m zrt.training estimate --config python/zrt/training/configs/llama3_70b_3d.yaml

# End-to-end validation
python e2e_check.py
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
- `python/zrt/report/summary.py`: Excel/HTML report generation (6 sheets original, 5 sheets + JSON after transform)
- `python/zrt/training/`: training performance estimation, YAML configs in `configs/`

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
