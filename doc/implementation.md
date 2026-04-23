# Implementation Guide

## Overview

This project is an **LLM inference performance modeling platform**. It traces the operator sequence of any Hugging Face causal LM on meta tensors (no weights needed), transforms the resulting computation graph, and produces per-operator latency estimates, memory budgets, and end-to-end throughput projections.

---

## End-to-End Data Flow

```
HF config.json  (no weights downloaded)
       │
       ▼
  load_model()           ← FakeTensor context, num_layers override
       │
       ▼
  _trace_phase()         ← RecordingDispatch + ModuleTracker
  (per phase: prefill / decode)
       │  raw records: list[Dict]
       ▼
  records_to_opgraph()   ← adapter.py  →  OpGraph IR (raw)
       │
       ▼
  FusionEngine.fuse()    ← graph/fusion.py  →  OpGraph IR (fused)
       │
  ┌────┴────────────────────────────────────┐
  │      Optional: TransformPipeline        │
  │  Stage 1 – Parallel Split               │
  │  Stage 2 – Fusion Pass                  │
  │  Stage 3 – Optimization Annotations     │
  │  Stage 4 – Analysis (FLOPs / Roofline / Comm) │
  └────────────────┬────────────────────────┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
   SimulatorHub          DAGScheduler
   (per-op latency)      (multi-stream timeline)
         │                    │
         └─────────┬──────────┘
                   ▼
            E2ESummary / Excel / JSON / ONNX
```

---

## Module Map

```
python/zrt/
├── ir/                  Computation graph IR (central data bus)
│   ├── types.py         DType enum, TensorMeta (shape + dtype, no data)
│   ├── node.py          OpNode — one operator with inputs/outputs/attrs
│   ├── edge.py          Edge — data or control dependency
│   ├── graph.py         OpGraph — DAG container, topo sort, clone/mutate
│   ├── hierarchy.py     GraphHierarchy — scope tree for hierarchical analysis
│   ├── adapter.py       records ↔ OpGraph conversions, NetworkX bridge
│   └── serde.py         JSON serialization
│
├── hardware/            Hardware specifications (pure metadata)
│   ├── spec.py          HardwareSpec, ComputeSpec, MemorySpec, InterconnectSpec
│   ├── registry.py      load(name) → HardwareSpec from YAML
│   └── configs/         ascend_910b.yaml, nvidia_h100.yaml, …
│
├── graph/               Operator tracing pipeline
│   ├── main.py          Public API: run_trace(), run_trace_phases()
│   ├── dispatch.py      RecordingDispatch (TorchDispatchMode)
│   ├── tracker.py       ModuleTracker (scope path via forward hooks)
│   ├── model_loader.py  load_model() with FakeTensor setup
│   ├── graph_builder.py records → OpGraph construction
│   ├── fusion.py        FusionEngine (pattern matching on raw records)
│   ├── excel_writer.py  Raw / fused op tables → Excel
│   └── graph_exporter.py JSON / ONNX export
│
├── transform/           4-stage graph transformation pipeline
│   ├── pipeline.py      TransformPipeline orchestration
│   ├── base.py          GraphPass ABC (stateless, functional)
│   ├── context.py       TransformContext (hw_spec, parallel config)
│   ├── parallel/
│   │   ├── tensor_parallel.py   TP sharding — splits shapes, inserts all_reduce
│   │   ├── expert_parallel.py   EP distribution — inserts all_to_all
│   │   └── comm_inserter.py     Inserts comm nodes at split boundaries
│   ├── fusion/
│   │   ├── pass_.py     FusionPass: 3-phase (leaf → parent → label)
│   │   └── patterns.py  FusionRule, semantic label matching
│   ├── optim/
│   │   ├── quantization.py   W8A8 / W4A16 / KV-int8 annotation
│   │   ├── eplb.py           Expert load-balancing annotation
│   │   ├── shared_expert.py  Shared expert externalization
│   │   └── mtp.py            Multi-token prediction setup
│   └── analysis/
│       ├── flops.py          FLOPs / MACs calculation
│       ├── roofline.py       Compute-bound vs memory-bound annotation
│       ├── comm_latency.py   Interconnect-aware comm latency
│       └── passes.py         StreamAssignPass, etc.
│
├── memory/              Memory modeling (formula-based, no graph needed)
│   ├── model.py         MemoryModel: weights / KV / activation / comm estimates
│   ├── budget.py        MemoryBudget breakdown struct
│   └── activation.py    Activation lifetime analysis (peak during inference)
│
├── simulator/           Operator-level performance estimation
│   ├── hub.py           SimulatorHub: routes to best available backend
│   ├── base.py          OpSimulator ABC
│   ├── result.py        SimResult (latency_us, FLOPs, bound, confidence)
│   ├── cache.py         Content-hash caching layer
│   └── backends/
│       ├── roofline.py      Roofline model (theoretical formulas)
│       ├── profile_db.py    Real profiling data lookup
│       ├── regression.py    Fitted regression models
│       └── tiling_sim.py    Tiling-level simulation adapter
│
├── executor/            Multi-stream DAG scheduling
│   └── scheduler.py     DAGScheduler — list scheduling, overlap analysis
│
└── report/              Output generation
    ├── summary.py       E2ESummary builder
    └── excel_writer.py  Performance-annotated Excel output
```

---

## Key Code Flows

### 1. Operator Tracing

Entry: `python/zrt/graph/main.py`

```
run_trace_phases(model_id, num_layers, phases=("prefill","decode"))
 └─ load_model(model_id, num_layers)
     ├─ AutoConfig.from_pretrained(model_id)          # download config.json only
     ├─ override config.num_hidden_layers = num_layers
     └─ AutoModelForCausalLM.from_config(config)
        .to("meta")                                   # no weights, no memory
        → (model, config, fake_mode)

 └─ for phase in phases:
     _trace_phase(model, phase, seq_len, batch_size)
      ├─ build input tensors on meta device
      ├─ ModuleTracker(model).start()                 # forward hooks → scope paths
      ├─ with RecordingDispatch() as rd:              # TorchDispatchMode
      │    model(input_ids, ...)                      # forward pass
      │    # every aten::* op is intercepted:
      │    #   __torch_dispatch__() called per op
      │    #   extracts: op_name, input shapes/dtypes, output shapes
      │    #   annotates: module_scope, module_class, call_site
      │    #   appends to rd.records
      └─ return rd.records                            # list[Dict]

 └─ _save_phase_outputs(records, phase)
     ├─ raw_graph = records_to_opgraph(records)       # ir/adapter.py
     ├─ fused_records = FusionEngine(rules).fuse(records)
     ├─ fused_graph = fused_records_to_opgraph(...)
     ├─ write_excel(raw_graph, fused_graph, output_dir)
     └─ export_json / export_onnx
```

Key class: **`RecordingDispatch`** (`graph/dispatch.py`)

```python
class RecordingDispatch(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs):
        # 1. let op execute on meta tensors
        out = func(*args, **kwargs)
        # 2. snapshot input/output tensor metadata
        record = {
            "op": func._overloadpacket._qualified_name,
            "inputs":  [TensorMeta.from_tensor(t) for t in flat_args],
            "outputs": [TensorMeta.from_tensor(t) for t in flat_out],
            "scope":   ModuleTracker.current_scope(),
            "call_site": _get_call_site(),
        }
        self.records.append(record)
        return out
```

---

### 2. IR Construction

Entry: `ir/adapter.py :: records_to_opgraph(records)`

```
records_to_opgraph(records)
 ├─ for each record → OpNode(op, inputs, outputs, attrs, scope)
 ├─ TensorTracker assigns stable IDs to meta tensors
 ├─ for each (producer_node, consumer_node) sharing a tensor_id
 │    → Edge(src=producer, dst=consumer, tensor_meta)
 └─ OpGraph(nodes, edges)
     ├─ topo_sort()          # Kahn's algorithm
     ├─ hierarchy            # built lazily from scope paths
     └─ clone()              # used by all transform passes (functional style)
```

---

### 3. Transform Pipeline

Entry: `transform/pipeline.py :: TransformPipeline.run(graph, ctx)`

```
TransformPipeline.run(graph, ctx)
 │
 ├─ Stage 1: Parallel Split
 │   ├─ TensorParallelPass(tp=8)
 │   │   ├─ shards matmul input/output shapes by tp
 │   │   └─ inserts all_reduce nodes after column-parallel ops
 │   └─ CommInserterPass
 │       └─ adds explicit comm nodes at TP/EP boundaries
 │
 ├─ Stage 2: Fusion
 │   └─ FusionPass
 │       ├─ Phase 1 – Leaf Fusion
 │       │   group consecutive nodes sharing same (scope, layer)
 │       │   comm nodes always break groups
 │       ├─ Phase 2 – Parent Fusion
 │       │   merge leaf groups sharing a fusible parent scope
 │       │   bounded by max_parent_ops, max_children
 │       └─ Phase 3 – Semantic Labeling
 │           map module_class → label (attention / mlp / rms_norm / …)
 │
 ├─ Stage 3: Optimization Annotations
 │   ├─ QuantizationPass   → node.attrs["quant"] = "w8a8" | "w4a16" | …
 │   ├─ EPLBPass           → annotates expert load imbalance factor
 │   ├─ SharedExpertPass   → marks always-active shared experts
 │   └─ MTPPass            → sets up multi-token prediction heads
 │
 └─ Stage 4: Analysis
     ├─ FlopsPass          → node.attrs["flops"] = computed FLOPs
     ├─ RooflinePass       → node.attrs["latency_us"], "bound" (compute|memory)
     └─ CommLatencyPass    → overwrites latency_us for comm nodes
```

**Roofline formula** (`transform/analysis/roofline.py`):

```
arithmetic_intensity = FLOPs / memory_bytes       # ops per byte

compute_time  = FLOPs / hw.peak_flops
memory_time   = memory_bytes / hw.hbm_bandwidth
latency_us    = max(compute_time, memory_time)

bound = "compute" if compute_time > memory_time else "memory"
```

**Comm latency formula** (`transform/analysis/comm_latency.py`):

```
# Ring AllReduce (e.g., TP all_reduce after matmul):
latency = 2 * (n-1)/n * data_bytes / bandwidth
        + 2 * (n-1) * link_latency

# Uses intra_node BW (NVLink/HCCS) when group fits on one node,
# inter_node BW (IB/RoCE) otherwise.
```

---

### 4. Memory Modeling

Entry: `memory/model.py :: MemoryModel.estimate(config, hw, parallel, dtype)`

```
MemoryModel.estimate()
 ├─ weights_mb   = total_params * bytes_per_param / (tp * pp * ep)
 ├─ kv_cache_mb  = 2 * local_layers * kv_dim * seq_len * batch * dtype_bytes
 │                 (kv_dim = kv_lora_rank for MLA, kv_heads*head_dim for GQA)
 ├─ activation_mb = ~34 * batch * seq_len * hidden / tp * dtype_bytes  # empirical
 ├─ comm_buf_mb  = TP all_reduce buffer or EP expert buffer
 └─ overhead_mb  = 0.05 * total                    # framework overhead

 → MemoryBudget(breakdown, is_feasible = total <= hw.memory.capacity_gb)
```

---

### 5. Simulation & Scheduling

**SimulatorHub** (`simulator/hub.py`):

```
SimulatorHub.simulate_node(node, hw_spec)
 ├─ cache_key = hash(op_type, shapes, dtypes, hw.name)
 ├─ if cache_key in cache → return cached SimResult
 └─ for backend in [ProfileDB, Regression, Roofline]:   # priority order
     if backend.supports(node):
         result = backend.simulate(node, hw_spec)
         cache[cache_key] = result
         return result
```

**DAGScheduler** (`executor/scheduler.py`):

```
DAGScheduler.schedule(graph)
 ├─ assign nodes to streams: compute_stream | comm_stream
 ├─ list scheduling: process nodes in topo order
 │   for each node:
 │     earliest_start = max(finish_time of all predecessors)
 │     if node.stream == comm_stream:
 │         can overlap with compute_stream nodes
 │     scheduled_time = earliest_start + node.latency_us
 └─ return Timeline(scheduled_ops, total_latency_us, overlap_us)
```

---

## CLI Reference

```bash
# Basic tracing (downloads config only, not weights)
python -m python.zrt deepseek-ai/DeepSeek-V3-0324 --layers 4

# Specify phases, output directory, hardware target
python -m python.zrt Qwen/Qwen2.5-7B-Instruct \
    --layers 4 --phases prefill decode \
    --hw ascend_910b --tp 8 \
    -o output/qwen25_7b

# Local config directory
python -m python.zrt ./hf_models/deepseek_v3 --layers 4
```

Output per run:

| File | Description |
|------|-------------|
| `*_raw.xlsx` | Raw aten op table with shapes and scope |
| `*_fused.xlsx` | Fused op table with semantic labels |
| `*_raw.json` | OpGraph IR (raw) as JSON |
| `*_fused.json` | OpGraph IR (fused) as JSON |
| `*_fusion_rules.json` | Discovered fusion rules |

---

## Testing

```bash
pytest tests/                         # all tests
pytest tests/test_fusion_pass.py      # fusion logic
pytest tests/test_simulator.py        # roofline formulas
pytest tests/test_executor.py         # DAG scheduling
pytest tests/test_transform.py        # transform pipeline
pytest tests/test_memory.py           # memory estimates
```

Test helpers (used across all test files):

```python
_t(shape, dtype)   → TensorMeta
_node(op, ...)     → OpNode
_edge(src, dst)    → Edge
_graph(nodes, edges) → OpGraph
```

---

## Design Principles

- **No weights required** — FakeTensor/meta device enables tracing 671B models in seconds on a laptop.
- **Functional graph passes** — every `GraphPass` calls `graph.clone()` before mutating; the original is never modified.
- **Hardware/model orthogonal** — `HardwareSpec` and model configs are fully independent; swap either without touching the other.
- **Content-hash caching** — `SimulatorHub` caches by `hash(op, shapes, dtypes, hw)`, not by node identity.
- **Lazy IR hierarchy** — `OpGraph.hierarchy` is built on first access and invalidated on any structural mutation.
- **Priority-based simulation** — backends stack as `ProfileDB > Regression > Roofline`; the most accurate available is used.
