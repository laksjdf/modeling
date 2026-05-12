# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> General behavioral guidelines (think before coding, simplicity, surgical changes, goal-driven execution) live in the global `~/.claude/CLAUDE.md`. This file documents only project-specific facts.
>
> `AGENTS.md` (repo root) is a parallel guidance file aimed at coding agents in general. When the two diverge, **`CLAUDE.md` takes precedence for Claude Code**; treat `AGENTS.md` as supplementary.

## 核心原则（必读）

1. **`hf_models/` 目录只读**：其中所有 `.py` 和 `.json` 文件必须完全来自 HF 官网下载，禁止在此目录内修改任何代码。
   - 唯一例外：`config.json` 可添加 `auto_map` 字段（用于本地加载自定义架构），但需标注原因。
   - 所有运行时兼容性修复必须写在 `python/zrt/graph/patches.py`，通过 monkey-patch 注入。

2. **所有 patch 逻辑集中在 `patches.py`**：不得在 `model_loader.py`、`main.py` 或测试文件中散落 patch 代码。

3. **FakeTensorMode 生命周期**：`load_model()` 返回 3-tuple `(model, config, fake_mode)`，
   `fake_mode` 必须在 forward pass 期间保持激活，finally 块中调用 `fake_mode.__exit__(None, None, None)`。

---

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
pytest tests/ -v -m "not network" 2>&1 | tail -n 50

# CLI: trace a model and export Excel
# NOTE: model is passed via --model-id (a flag, not a positional argument)
python -m python.zrt --model-id Qwen/Qwen2.5-7B-Instruct --layers 4
python -m python.zrt --model-id deepseek-ai/DeepSeek-V3-0324 --layers 4 --hw nvidia_h100_sxm --tp 8

# Training modeling (captures train_forward + train_backward + estimates performance)
python -m python.zrt --model-id deepseek-ai/DeepSeek-V3 --layers 4 --train --hw nvidia_h100_sxm --tp 8 --pp 2
PYTHONPATH=python python -m zrt.training estimate --config python/zrt/training/configs/llama3_70b_3d.yaml
# Export Chrome Trace alongside training estimate
PYTHONPATH=python python -m zrt.training estimate --config python/zrt/training/configs/llama3_70b_3d.yaml --trace out.json

# Spec-based training estimate (no graph capture needed)
python -m python.zrt --estimate-config python/zrt/training/configs/llama3_70b_3d.yaml

# Graph-mode capture (torch.compile backend instead of TorchDispatchMode)
python -m python.zrt --model-id deepseek-ai/DeepSeek-V3 --layers 2 --phases train_backward --graph-mode

# End-to-end validation
python -m validation.cli
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
- `fusion/`: MRO-based fusion system (v2) — rules keyed by `type(module).__mro__`, single-pass algorithm, preserves `name` + `provenance`. See `docs/fusion_v2_zh.md` for rule registration guide.
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
- `server/`: FastAPI HTTP service exposing trace/estimate as REST endpoints (entry: `server/main.py`); separate dependency set in `server/requirements.txt`
- `validation/`: end-to-end validation scenarios (`python -m validation.cli`) — compares simulator output against public benchmarks in `validation/public_benchmark_data.json`

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

**Anchor YAML fixtures** (`tests/training/anchors/*.yaml`): GPT-3 175B, LLaMA-3 70B, DeepSeek-V3 / V3.2 (and V4 variants). Each `targets:` block pins `mfu`; the four "algebraic" anchors (V3 + V3.2 family) also pin `step_time_ms`. Two tests gate regressions: `test_anchor_mfu_strict` and `test_anchor_step_time_strict` — both honor `strict_mfu_check: false`, which puts an anchor in **calibration mode** (anchor MFU sourced from a published reference, simulator not yet calibrated to match — `gpt3_175b_megatron` and `llama3_70b_meta` use this today). Run with `pytest tests/training/anchors/test_anchors.py`.

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


## 环境

- Python 3.14, torch 2.11.0, transformers 5.4.0
- 主工作目录（WSL）：`/mnt/d/work/git/modeling`
- 运行时使用 `python -m pytest` 或直接 `pytest`
- CLI 入口：`python -m python.zrt --model-id <model_id> --layers 4`（详见 `python/zrt/cli.py`）


## 执行要求
- 每次执行记得查看.clauderules的要求
## 关键文件速查

```
python/zrt/graph/
├── compat.py         # transformers 版本 shim + 本地模型注册表（新）
├── patches.py        # 运行时 patch（MoE、Indexer、legacy 属性）
├── model_loader.py   # 分层加载器：load_model() -> (model, config, fake_mode)
├── main.py           # run_trace() 入口 + CLI
├── dispatch.py       # RecordingDispatch + TensorTracker（aten 拦截）
├── tracker.py        # ModuleTracker（forward hooks，模块路径）
├── fusion.py         # FusionEngine（两阶段算子融合）
├── classifier.py     # 组件分类 + 颜色映射
├── excel_writer.py   # Excel + JSON 输出
├── graph_builder.py  # build_op_graph / build_fused_op_graph
├── graph_exporter.py # export_all() -> JSON/ONNX
└── __init__.py       # 公开 API 导出

hf_models/            # 只读！来自 HF 官网
├── deepseek_v3/      # config.json + modeling_deepseek.py + configuration_deepseek.py
├── deepseek_v3_2/    # 同上（V3.2，含 Indexer 模块；config.json 加了 auto_map）
├── llama3_8b/        # 仅 config.json（标准架构无需 modeling 文件）
├── llama3_70b/
├── qwen2_7b/
├── qwen2_72b/
├── mistral_7b/
└── mixtral_8x7b/

tests/                   # pytest 测试套件（见上方 Commands 章节）
```

---

## 核心 API

### `load_model(model_id, num_hidden_layers=4)`
```python
from python.zrt.graph import load_model
model, config, fake_mode = load_model("deepseek-ai/DeepSeek-V3-0324", num_hidden_layers=4)
# fake_mode 已 __enter__，forward 结束后需手动 __exit__
fake_mode.__exit__(None, None, None)
```

### `run_trace_phases(...)` ← 推荐：prefill + decode 一次完成
```python
from python.zrt.graph import run_trace_phases
output_dir, phase_records = run_trace_phases(
    model_id="deepseek-ai/DeepSeek-V3-0324",
    num_layers=4,
    batch_size=1,
    seq_len=128,
    output_dir="output/graph/DeepSeek-V3-0324",  # 可选
    phases=("prefill", "decode"),                 # 默认同时抓两阶段
)
# phase_records["prefill"] / phase_records["decode"] → op-record 列表
# 输出文件：
#   DeepSeek_V3_0324_prefill_ops.xlsx / _prefill_raw_graph.json / .onnx / _prefill_fused_graph.*
#   DeepSeek_V3_0324_decode_ops.xlsx  / _decode_raw_graph.json  / .onnx / _decode_fused_graph.*
```

### `run_trace(...)` ← 单阶段（向后兼容）
```python
from python.zrt.graph import run_trace
output_dir, records = run_trace(
    model_id="deepseek-ai/DeepSeek-V3-0324",
    num_layers=4,
    batch_size=1,
    seq_len=128,
    output_dir="output/graph/DeepSeek-V3-0324",  # 可选，默认 output/graph/<slug>
    phase="prefill",   # "prefill"（默认）/ "decode" / "forward"（"prefill" 的别名）
)
```

### Prefill vs Decode 输入差异

| 参数 | Prefill | Decode |
|------|---------|--------|
| `input_ids` shape | `(B, seq_len)` | `(B, 1)` |
| `position_ids` | `[[0..seq_len-1]]` | `[[seq_len]]` |
| `attention_mask` | `(1,1,seq_len,seq_len)` causal | `(1,1,1,seq_len+1)` 全零 |
| `past_key_values` | None | prefill 输出的 KV cache |
| `use_cache` | True | True |

两阶段在**同一 FakeTensorMode 上下文**内运行，prefill 的 FakeTensor KV cache 直接传给 decode。

---

## 版本自适应机制（compat.py）

### 问题背景

不同模型的 `modeling_*.py` 依赖写入时的 transformers 版本，升级后会出现：
- `ImportError: cannot import name 'xxx' from 'transformers.utils'`（API 被移除）
- `ValueError: does not recognize this architecture`（新架构未注册）

### 解决方案（参考 vLLM/SGLang 设计）

| 机制 | 作用 |
|------|------|
| **版本 shim**（`apply_version_shims()`）| 在模型文件 import 之前，向 `transformers.*` 注入被移除的符号 |
| **本地注册表**（`_LOCAL_REGISTRY`）| `model_type` / Hub ID → `hf_models/` 目录映射 |
| **分层加载**（`model_loader.py`）| Hub 加载失败时自动 fallback 到本地注册表 |

### 已知 shim 清单

| 符号 | 模块/类 | 变更版本 | 替代方案 |
|------|---------|---------|---------|
| `is_flash_attn_greater_or_equal_2_10` | `transformers.utils` | 5.x 移除 | `is_flash_attn_greater_or_equal("2.1.0")` |
| `is_torch_fx_available` | `transformers.utils` | 部分 5.x 移除 | 返回 `True` |
| `DynamicCache.from_legacy_cache` | `transformers.DynamicCache` | 5.x 移除 | 用 `update()` 逐层填充 |
| `DynamicCache.to_legacy_cache` | `transformers.DynamicCache` | 5.x 移除 | 迭代 `layers[i].keys/values` |
| `DynamicCache.get_usable_length` | `transformers.DynamicCache` | 5.x 重命名 | `get_seq_length(layer_idx)` |
| `DynamicCache.seen_tokens` | `transformers.DynamicCache` | 5.x 移除 | `get_seq_length()` |
| `DynamicCache.get_max_length` | `transformers.DynamicCache` | 5.x 移除 | 返回 `None`（无界缓存）|

### 新增自定义架构

1. 将模型文件放入 `hf_models/<name>/`（含 `auto_map` 的 `config.json`）
2. 在 `compat._LOCAL_REGISTRY` 添加两条映射：`model_type` + Hub ID
3. 如有 transformers API 兼容问题，在 `compat.apply_version_shims()` 添加 shim

## patches.py 清单

| 函数 | 作用 |
|------|------|
| `apply_compat_patches()` | 先调 `apply_version_shims()`，再补 legacy 属性 |
| `patch_moe_for_fake(model)` | 替换 MoE forward，绕过 `.cpu().numpy()` / `torch.bincount()` |
| `patch_indexer_for_fake(model)` | 修正 DeepSeek-V3.2 Indexer 的 3D tensor 错误 transpose |
| `is_moe_module(module)` | 检测 MoE（有 `experts: nn.ModuleList` 且未被 patch）|
| `patch_moe_for_meta` | 向后兼容别名 → `patch_moe_for_fake` |

**Indexer patch 背景**：原始 `modeling_deepseek.py`（deepseek_v3_2）中
`k_nope.transpose(1,2).transpose(2,3)` 对 3D tensor 非法，
patch 在运行时替换 `Indexer.forward`，模型文件保持不变。

---

## 模型支持说明

| 本地路径 | HF Hub ID | 备注 |
|---------|-----------|------|
| `hf_models/deepseek_v3` | `deepseek-ai/DeepSeek-V3` | MoE patch 必需 |
| `hf_models/deepseek_v3_2` | `deepseek-ai/DeepSeek-V3.2` | MoE + Indexer patch|
| — | `deepseek-ai/DeepSeek-V3-0324` | V3 的 3 月 24 日更新版，**不是** V3.2 |
| `hf_models/llama3_8b` | `meta-llama/Llama-3.1-8B` | 需 HF 授权 |
| `hf_models/qwen2_7b` | `Qwen/Qwen2.5-7B-Instruct` | — |
| `hf_models/mixtral_8x7b` | `mistralai/Mixtral-8x7B-v0.1` | MoE patch |

---

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
