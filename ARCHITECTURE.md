# ZRT-Sim Architecture

ZRT-Sim is an LLM performance modeling and simulation system. It captures the operator sequence of any HuggingFace causal LM using `TorchDispatchMode` inside `FakeTensorMode` (no weights or real memory needed), applies parallelization transforms, and simulates performance across hardware configurations.

---

## Design Principles

| Principle | Meaning |
|-----------|---------|
| **Two estimation paths** | Inference trace and training estimation are first-class paths; both flow through the same transform pipeline |
| **No weights needed** | `FakeTensorMode` captures operator shapes and types without loading model parameters |
| **Split before fuse** | Parallel splits run first; fusion only operates within resulting sub-graphs, so rules never need to know about parallelism |
| **Hardware/software orthogonal** | Hardware specs (H100, 910B) and fusion rules are independent; new combinations need no code changes |
| **Memory as first-class citizen** | Memory estimation is standalone — used for feasibility gating before latency simulation |
| **Pluggable simulators** | `SimulatorHub` dispatches through a priority-ordered fallback chain: Tiling → ProfileDB → Roofline |

---

## Two Primary Paths

```
┌──────────────────────────────────────────────────────┐
│  PATH A — Inference Trace                            │
│                                                      │
│  python -m python.zrt --model-id ...                │
│                                                      │
│  HF model → FakeTensorMode capture → raw OpGraph    │
│           → Transform Pipeline → SimulatorHub        │
│           → DAGScheduler → Timeline                  │
│           → Report (Excel / HTML / ONNX / JSON)      │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  PATH B — Training Estimate: two sub-paths                   │
│                                                              │
│  B1 — Graph-driven (needs HF model, matches real ops):      │
│    python -m python.zrt --model-id ... --train              │
│    FakeTensorMode capture → annotated OpGraph               │
│    → transform/analysis/training.py (PP dispatch)            │
│    → training/compose/schedules.py (composer)                │
│    → transform/analysis/modeller.py → StepResult            │
│                                                              │
│  B2 — Spec-driven (no model needed, fast):                  │
│    python -m zrt.training estimate --config <yaml>          │
│    YAML → ModelSpec + Strategy + SystemSpec                  │
│    → training/ir/builders.py (training Graph)                │
│    → training/compose/schedules.py (composer directly)       │
│    → StepResult                                              │
└──────────────────────────────────────────────────────────────┘
```

B1 is used when a model is available and you want to match actual operator patterns. B2 is used for design-space exploration, config search, and anchor regression testing without needing HF model files.

---

## Transform Pipeline

Both inference and graph-driven training flow through the same four-stage pipeline:

```
raw OpGraph
    │
    ▼  ── stage: split ──────────────────────────────────────────
    │  TP → EP → CP → PP → DP                  (parallel split passes)
    │  CommInserter                             (insert comm.* nodes at boundaries)
    │  RecomputePass / OffloadPass / ZeroFSDPPass  (training-only annotations)
    │
    ▼  ── stage: fuse ───────────────────────────────────────────
    │  FusionPass                               (MRO-based YAML rules)
    │
    ▼  ── stage: optim ──────────────────────────────────────────
    │  QuantizationPass / EPLBPass / MTPPass / SharedExpertPass
    │
    ▼  ── stage: analyze ────────────────────────────────────────
    │  FlopsPass / RooflinePass / CommLatencyPass
    │  TrainingAnalysisPass  (graph-driven path: PP composer dispatch)
    │
    ▼
transformed OpGraph
```

Pass order within **split**: TP → EP → CP → PP → DP → CommInserter. Split passes annotate nodes; CommInserter reads annotations and inserts `comm.*` nodes — the decision about what to parallelize is separate from the act of inserting communication.

---

## Inference Pipeline Detail

```
python -m python.zrt --model-id <id> --layers 4 --hw nvidia_h100_sxm --tp 8
    │
    ├─ graph/model_loader.py       load_model() → (model, config, fake_mode)
    ├─ graph/patches.py            apply_compat_patches(), patch_moe_for_fake()
    ├─ graph/dispatch.py + tracker.py   capture aten ops + scope paths
    ├─ graph/graph_builder.py      build_op_graph() + build_fused_op_graph()
    │
    ├─ transform/pipeline.py       build_default_pipeline().run(graph, ctx)
    │      parallel/ → fuse/ → optim/ → analysis/
    │
    ├─ memory/model.py             feasibility check (can this config fit in HBM?)
    ├─ simulator/hub.py            simulate_graph() → dict[node_id → SimResult]
    ├─ executor/scheduler.py       DAGScheduler → Timeline
    └─ report/summary.py           Excel + HTML + JSON output
```

**FakeTensorMode lifecycle** — both prefill and decode run in the same context so that the FakeTensor KV-cache from prefill flows directly into decode:

```python
model, config, fake_mode = load_model(model_id, num_hidden_layers=4)
# fake_mode is already __enter__'d
try:
    model(**prefill_inputs)   # produces FakeTensor KV-cache
    model(**decode_inputs)    # consumes it
finally:
    fake_mode.__exit__(None, None, None)
```

**Prefill vs Decode inputs:**

| | Prefill | Decode |
|--|---------|--------|
| `input_ids` shape | `(B, seq_len)` | `(B, 1)` |
| `past_key_values` | `None` | prefill output |
| `attention_mask` | causal `(1,1,S,S)` | full-zero `(1,1,1,S+1)` |

---

## Training Estimate Detail (Spec-driven, Path B2)

```
PYTHONPATH=python python -m zrt.training estimate --config llama3_70b_3d.yaml
    │
    ├─ training/io/config_loader.py    YAML → (ModelSpec, Strategy, SystemSpec)
    ├─ training/ir/builders.py         build training Graph (layer shards + stages)
    ├─ transform/analysis/training.py  select PP composer via ctx.training.pp_schedule
    ├─ training/compose/schedules.py   chosen composer → StepResult
    └─ transform/analysis/modeller.py  read pipeline_metrics.step_time_ms → MFU / HFU
```

**PP schedules** — `ctx.training.pp_schedule` (`PPSched` enum) selects the composer:

| `PPSched` | Composer |
|-----------|----------|
| `ONE_F_ONE_B` | `OneF1BComposer` |
| `INTERLEAVED` | `InterleavedComposer` (VPP) |
| `ZERO_BUBBLE` | `ZeroBubbleComposer` |
| `DUALPIPE` | `DualPipeComposer` |
| `DUALPIPE_V` | `DualPipeVComposer` |

**`StepResult` invariants** (all in seconds):

```
step_time        = pipeline_time + optimizer_time + optimizer_comm
pipeline_time    = compute_time + exposed_comm
compute_time     = fwd_compute + bwd_compute + recompute_time
mfu              = actual_flops / (step_time × peak_flops)      — excludes recompute
hfu              = (actual + recompute_flops) / (step_time × peak_flops)  — hfu > mfu when recompute active
```

**Anchor regression** (`tests/training/anchors/*.yaml`): pins `mfu` and `step_time_ms` for GPT-3 175B, LLaMA-3 70B, DeepSeek-V3/V3.2, and V4 variants. `strict_mfu_check: false` puts an anchor in calibration mode (not a regression blocker).

---

## Key Types & Contracts

### OpGraph IR (`python/zrt/ir/`)

```
OpGraph
  name, phase           # "prefill" | "decode"
  nodes: dict[id → OpNode]
  edges: list[OpEdge]

OpNode  (frozen dataclass — clone() before mutating)
  op_type               # "aten.mm" | "comm.all_reduce" | ...
  scope                 # "model.layers.0.self_attn.q_proj"
  category              # "compute" | "communication" | "memory"
  annotations: dict     # set by transform passes: recompute, tp_split, ep_needs_a2a, ...

GraphHierarchy  (built from scope strings, lazy)
  at_depth(n)           # 0=model, 1=embed/layers, 2=per-layer, 3=attn/mlp/norm
  aggregate(node, metric, values) → float
```

### TransformContext (`python/zrt/transform/context.py`)

```
TransformContext
  hw_spec: HardwareSpec
  parallel: ParallelConfig    # tp / pp / ep / dp / cp / sp
  quant: QuantConfig          # weight / activation / kv_cache dtype
  training: TrainingConfig    # pp_schedule, recompute_layers, optimizer, zero_stage
  offload: OffloadConfig      # pct, opt_state, grads, params
  fusion: FusionConfig        # enabled_rules, disabled_rules
  phase: str                  # "prefill" | "decode"
  model_id: str
```

### SimulatorHub (`python/zrt/simulator/`)

Fallback chain (highest priority wins):

```
TileSimulator     priority=100   tiling-level, narrowest coverage
ProfileDBSimulator priority=30   exact lookup from profiling CSV
RooflineSimulator  priority=0    always available, theoretical bound
```

`SimResult` fields: `latency_us`, `compute_us`, `memory_us`, `flops`, `read_bytes`, `write_bytes`, `arithmetic_intensity`, `bound` ("compute"/"memory"), `hw_utilization`, `backend`, `confidence`.

### Hardware Registry (`python/zrt/hardware/`)

```python
from python.zrt.hardware.registry import hw_registry
hw = hw_registry.load("nvidia_h100_sxm")   # → HardwareSpec
```

Available: `nvidia_h100_sxm`, `nvidia_a100_80g`, `nvidia_h800`, `ascend_910b`, `ascend_910c`.

`HardwareSpec.compute` has both `cube_tflops` (Tensor Core / matrix engine) and `vector_tflops` (CUDA Core / vector engine) per dtype — both must be populated for accurate roofline modeling on NVIDIA GPUs.

---

## Critical Constraints

**`hf_models/` is read-only.** All `.py` and `.json` files must come verbatim from HF. The only permitted modification is adding an `auto_map` field to `config.json`. All runtime fixes go in `graph/patches.py` via monkey-patch.

**Import convention.** The main package uses `python.zrt.*` imports. The training subpackage uses `zrt.*` imports and requires `PYTHONPATH=python` when running directly.

**Graph capture requires 4 layers for DeepSeek models** (first 3 dense, layer 4 is MoE). Dense models (Llama, Qwen, Mistral) only need 2 layers.

---

## Directory Structure

```
python/zrt/
├── cli.py                        # CLI: python -m python.zrt
│
├── graph/                        # Stage 1 — Graph Capture
│   ├── model_loader.py           #   load_model() → (model, config, fake_mode)
│   ├── dispatch.py               #   RecordingDispatch: aten op interception
│   ├── tracker.py                #   ModuleTracker: scope path tracking
│   ├── graph_builder.py          #   op records → OpGraph (raw + fused)
│   ├── patches.py                #   ALL monkey-patches (MoE, Indexer, compat)
│   ├── compat.py                 #   transformers 4.x/5.x shims + local registry
│   ├── classifier.py             #   component classification + color mapping
│   ├── pattern_extractor.py      #   structural pattern extraction
│   ├── tensor_utils.py           #   shape/dtype utilities
│   ├── transform_runner.py       #   drives transform pipeline from graph/
│   └── v4_fake_kernels.py        #   fake kernels for V4 ops
│
├── ir/                           # Core IR — shared by all stages
│   ├── graph.py                  #   OpGraph (pure Python, no NetworkX)
│   ├── node.py                   #   OpNode (frozen dataclass)
│   ├── edge.py                   #   OpEdge
│   ├── types.py                  #   DType, TensorMeta
│   ├── hierarchy.py              #   GraphHierarchy: scope tree + aggregation
│   ├── adapter.py                #   IR format adapters
│   ├── serde.py                  #   JSON serialization
│   └── param_count.py            #   parameter count utilities
│
├── transform/                    # Stage 2 — Transform Pipeline
│   ├── pipeline.py               #   build_default_pipeline(), TransformPipeline
│   ├── context.py                #   TransformContext + config dataclasses
│   ├── base.py                   #   GraphPass ABC
│   ├── parallel/                 #   Split passes (TP/EP/CP/PP/DP) + CommInserter
│   ├── fusion/                   #   MRO-based fusion v2 (YAML rules)
│   ├── analysis/                 #   FLOPs, Roofline, comm latency, training stats ← HOT PATH
│   ├── optim/                    #   Quant, EPLB, MTP, shared-expert passes
│   └── training/                 #   Recompute, offload, optimizer, ZeRO passes
│
├── executor/                     # Stage 3 — DAG Scheduler
│   ├── scheduler.py              #   topological sort + greedy multi-stream → Timeline
│   ├── stream.py                 #   Stream abstraction
│   ├── timeline.py               #   Timeline + query API
│   └── overlap.py                #   compute-comm overlap (scan-line intersection)
│
├── simulator/                    # Stage 4 — Latency Simulation
│   ├── hub.py                    #   SimulatorHub: priority dispatch + content-hash cache
│   ├── base.py                   #   OpSimulator ABC
│   ├── result.py                 #   SimResult
│   ├── cache.py                  #   content-hash cache
│   └── backends/                 #   roofline (0), lookup (30), tilesim (100)
│
├── hardware/                     # Hardware specs (YAML-based)
│   ├── spec.py                   #   HardwareSpec, ComputeSpec, MemorySpec
│   ├── registry.py               #   hw_registry.load(name) → HardwareSpec
│   └── configs/                  #   nvidia_h100_sxm, a100, h800, ascend_910b/c
│
├── memory/                       # Memory feasibility + peak estimation
├── layers/                       # Operator cost primitives (op_mm, op_attn, ...)
├── policy_model/                 # Pluggable cost-model policy dispatch
├── report/                       # Excel, HTML, ONNX, Chrome Trace, DOT, JSON
│
├── fusion/                       # Fusion rule discovery tool (offline, not runtime)
│   └── discover/                 #   AST scanner + runtime tracer + YAML templates
│
└── training/                     # Self-contained training estimator (zrt.* imports)
    ├── spec/                     #   ModelSpec, Strategy, SystemSpec, Dtype + enums
    ├── ir/                       #   training Graph (layer shards + stage assignment)
    ├── models/                   #   comm, flops, memory, optimizer math
    ├── compose/                  #   PP composers → StepResult  ← HOT PATH
    ├── search/                   #   SearchSpace + SearchEstimator → Pareto front
    ├── anchor/                   #   AnchorValidator: YAML fixture regression
    ├── trace/                    #   ChromeTraceExporter → Chrome Trace JSON
    ├── io/                       #   YAML config loader + perf tables
    └── configs/                  #   ready-to-use YAML training configs

hf_models/         READ-ONLY — verbatim HF downloads
tests/             pytest suite (training/ needs PYTHONPATH=python)
server/            FastAPI: /trace, /estimate, /predict  (separate requirements.txt)
validation/        E2E validation against public benchmark data
docs/              Design documentation (fusion-architecture.md, training_modeller_zh.md, ...)
```
