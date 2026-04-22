# Training Modelling — Integration into `zrt/transform`

> Extends the existing inference-focused infrastructure in `python/zrt/` (branch `main_dev_ytq`) to cover transformer **training** workloads: TP/CP/PP/EP/DP parallelism, ZeRO/FSDP, recompute, offload, comm-compute overlap, DualPipe schedules, and Adam/Muon optimizers — **reusing the IR, graph extraction, pass pipeline, scheduler, memory model, and hardware spec that already exist.**

Reference: [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) · existing repo audit in `ARCHITECTURE.md`.

---

## 1. What's already in the repo (reuse verbatim)

| Area | Module | Status | Training reuses it as-is? |
|---|---|---|---|
| **IR** | `zrt/ir/{graph,node,edge,types}.py` — `OpGraph`, `OpNode`, `Edge`, `TensorMeta`, `DType` | Mature, well-factored | ✅ Yes, no changes |
| **Graph extraction** | `zrt/graph/{main,graph_builder,model_loader,patches,classifier}.py` — `run_trace_phases` traces real HF models via `RecordingDispatch`; builds raw + fused DAGs | Mature; **already has `train_forward` / `train_backward` phases** via `loss.backward()` on FakeTensor (`_trace_phase` in `main.py:290-389`) | ✅ Yes |
| **IR adapter** | `zrt/ir/adapter.py` — `records_to_opgraph`, `fused_records_to_opgraph` | Mature | ✅ Yes |
| **Transform ABC** | `zrt/transform/base.py` — `GraphPass` | Minimal, clean | ✅ Yes |
| **Pipeline** | `zrt/transform/pipeline.py` — 4-stage `split → fuse → optim → analyze`, condition-guarded | Mature | ✅ Extend with training passes in the same stages |
| **TP pass** | `zrt/transform/parallel/tensor_parallel.py` | Col/row-parallel linears + annotations | ✅ Training reuses identically |
| **EP pass** | `zrt/transform/parallel/expert_parallel.py` | Sharding annotations for expert blocks | ✅ Reuses; needs an `expert_imbalance` extension |
| **Comm inserter** | `zrt/transform/parallel/comm_inserter.py` | Inserts TP all-reduce, EP dispatch/combine A2A | ✅ Extend for CP/DP collectives |
| **Fusion** | `zrt/transform/fusion/` | Pattern-based op fusion | ✅ Reuses |
| **Analysis** | `zrt/transform/analysis/passes.py` — `FlopsPass`, `RooflinePass`, `StreamAssignPass`; `comm_latency.py` — `CommLatencyPass` | Annotates `flops`, `read_bytes`, `write_bytes`, `compute_us`, `memory_us`, `latency_us`, `bound`, `stream_id` | ✅ `Flops` needs dx/dw extension; others reuse |
| **Scheduler** | `zrt/executor/scheduler.py` — `DAGScheduler`, `Timeline`, stream-based overlap | Mature; already models comm-compute overlap through stream assignment | ✅ Reuses directly for per-stage timing |
| **Memory model** | `zrt/memory/{model,activation,budget}.py` | Weights / KV / activations / comm-buffer; MLA and MoE aware | ✅ Extend with training buckets (grads, optimizer state, ZeRO sharding, recompute savings) |
| **Hardware spec** | `zrt/hardware/{spec,registry,configs}.py` | `HardwareSpec.peak_flops(dtype)`, `hbm_bandwidth()` | ✅ Reuses |
| **Context** | `zrt/transform/context.py` — `TransformContext`, `ParallelConfig`, `StreamConfig`, `QuantConfig` | Clean dataclasses | ✅ Extend with `TrainingConfig` field |

### Key insight from reading the code
The inference system already models communication–computation overlap the right way: `StreamAssignPass` puts comm nodes on a separate `stream_id`, and `DAGScheduler` lets them run concurrently with compute nodes, subject to data dependencies. Training just needs to **annotate more collectives and more overlap relationships** — the scheduling machinery is already there. That means **CoC/MC2, DualPipe dual-batch, and DP-in-bubble don't need a new "overlap model" module** — they become stream-assignment and edge-insertion rules inside existing passes.

---

## 2. What's missing for training (the delta)

All of the following are **new passes under `zrt/transform/`** slotting into the existing `split / fuse / optim / analyze` stages — not a parallel system.

| Missing piece | New module | Stage |
|---|---|---|
| Context parallel (Ulysses + Ring) sharding | `transform/parallel/context_parallel.py` | split |
| Pipeline parallel stage assignment + P2P | `transform/parallel/pipeline_parallel.py` | split |
| Data parallel gradient AR/RS insertion | `transform/parallel/data_parallel.py` | split |
| Extended comm inserter for CP / DP collectives | extend `transform/parallel/comm_inserter.py` | split |
| EP load imbalance (expert skew) | extend `transform/parallel/expert_parallel.py` | split |
| Recompute policy annotation | `transform/training/recompute.py` | optim |
| ZeRO / FSDP sharding annotations | `transform/training/zero_fsdp.py` | optim |
| Offload (host ↔ device) insertion | `transform/training/offload.py` | optim |
| Adam / Muon optimizer step annotation | `transform/training/optimizer.py` | optim |
| dx / dw FLOPs split (training-aware) | `transform/analysis/flops_train.py` | analyze |
| Training memory model (grads + opt state + ZeRO + recompute + offload) | `transform/analysis/memory_train.py` | analyze |
| Pipeline schedule composer (1F1B / VPP / DualPipe / DualPipeV) | `transform/schedule/{one_f_one_b,interleaved,dualpipe,dualpipev,composer}.py` | *post-pipeline* (new stage) |
| TP comm-compute overlap (CoC / MC2) stream rules | extend `transform/analysis/passes.py :: StreamAssignPass` (or new `training_overlap.py`) | analyze |
| `TrainingConfig` on `TransformContext` | extend `transform/context.py` | — |
| Training-default pipeline builder | extend `transform/pipeline.py` — add `build_training_pipeline()` | — |

The training system **does not** extend beyond the transform directory — everything plugs into the existing execution model.

---

## 3. Integration architecture

### 3.1 Data flow (training)

```
HF model (DeepSeek-V3 / Llama / Qwen / ...)
       │
       ▼
run_trace_phases(phases=("train_forward", "train_backward"))    # existing
       │
       ├── raw_opgraph["train_forward"]     (OpGraph)
       └── raw_opgraph["train_backward"]    (OpGraph, includes grad aten ops)
       │
       ▼
TransformPipeline = build_training_pipeline()                    # new helper
       │
       │  stage: split
       │    ├── TensorParallelPass           (existing)
       │    ├── ContextParallelPass          (new)
       │    ├── ExpertParallelPass           (existing, extended)
       │    ├── PipelineParallelPass         (new, annotates stage_id on nodes)
       │    ├── DataParallelPass             (new, inserts grad AR/RS)
       │    └── CommInserterPass             (existing, extended for CP/DP)
       │
       │  stage: fuse
       │    └── FusionPass                   (existing)
       │
       │  stage: optim
       │    ├── RecomputePass                (new)
       │    ├── ZeroFSDPPass                 (new, annotates shard factors)
       │    ├── OffloadPass                  (new, inserts H2D/D2H transfers)
       │    ├── OptimizerPass                (new, annotates Adam/Muon state)
       │    └── QuantizationPass             (existing)
       │
       │  stage: analyze
       │    ├── TrainFlopsPass               (new — dx/dw split; subclasses FlopsPass)
       │    ├── RooflinePass                 (existing)
       │    ├── CommLatencyPass              (existing)
       │    ├── StreamAssignPass             (existing, training-overlap rules added)
       │    └── TrainMemoryPass              (new, full training memory buckets)
       │
       ▼
Annotated OpGraph(s)  —  one per (phase, pp_stage)
       │
       ▼
DAGScheduler.schedule(graph)                                     # existing
       │
       ▼
PipelineScheduleComposer                                         # new
   - assembles per-stage Timelines across μbatches
   - 1F1B / VPP / DualPipe / DualPipeV / ZeroBubble (low priority)
   - folds DP AR into PP bubble window when dp_overlap_in_bubble
       │
       ▼
TrainingReport
   - iter_time_ms, mfu, hbm_highwater
   - per-stage Timeline
   - per-rank memory breakdown
   - (optional) Chrome trace
```

### 3.2 `TransformContext` extension

Add a new `TrainingConfig` field. Non-training pipelines leave it as `None`; training pipelines set it.

```python
# zrt/transform/context.py  (extension)

@dataclass
class RecomputeConfig:
    # per-op-kind → which tiers to recompute
    # tiers: "full" | "attn" | "attn_upscale" | "ffn_swiglu" | "ln"
    per_layer_kind: dict[str, set[str]] = field(default_factory=dict)

@dataclass
class OffloadConfig:
    opt_state: bool = False
    grads:     bool = False
    params:    bool = False
    pct:       float = 1.0     # fraction offloaded

@dataclass
class TrainingConfig:
    # batch
    micro_batch:  int = 1
    global_batch: int = 0      # 0 → derived from μbatch × dp × grad_accum
    grad_accum:   int = 1

    # pipeline schedule (degree lives in ParallelConfig.pp)
    pp_schedule: str = "1f1b"           # "1f1b" | "i1f1b" | "dualpipe" | "dualpipev" | "zb"
    vpp_chunks:  int = 1
    pp_layer_assignment: list[int] | None = None    # explicit stage→layers, else greedy bin-pack

    # CP
    cp_kind: str = "none"               # "none" | "ulysses" | "ring" | "hybrid"

    # memory
    zero_stage: int = 0                 # 0..3  (3 == FSDP)
    recompute:  RecomputeConfig = field(default_factory=RecomputeConfig)
    offload:    OffloadConfig   = field(default_factory=OffloadConfig)

    # overlap toggles  (→ stream rules in StreamAssignPass)
    tp_overlap:  str = "none"           # "none" | "coc" | "mc2"
    ep_overlap:  bool = False
    dualbatch:   bool = False
    dp_overlap_in_bubble: bool = True

    # optimizer
    optimizer: str = "adam"             # "adam" | "muon"


@dataclass
class TransformContext:
    hw_spec:       "HardwareSpec"
    parallel:      ParallelConfig  = field(default_factory=ParallelConfig)
    stream_config: StreamConfig    = field(default_factory=StreamConfig)
    quant:         QuantConfig | None = None
    training:      TrainingConfig | None = None        # NEW
    optim_flags:   set[str]        = field(default_factory=set)
    phase:         str             = "prefill"
    profile:       Any             = None
    stack:         Any             = None

    @property
    def is_training(self) -> bool:
        return self.training is not None and self.phase in ("train_forward", "train_backward")
```

`ParallelConfig` already has `tp/pp/ep/dp/sp` — CP is added:

```python
# zrt/transform/context.py  (extension)

@dataclass
class ParallelConfig:
    tp: int = 1
    pp: int = 1
    ep: int = 1
    dp: int = 1
    cp: int = 1        # NEW
    sp: bool = False

    @property
    def total_devices(self) -> int:
        return self.tp * self.pp * self.ep * self.dp * self.cp
```

### 3.3 Training pipeline builder

```python
# zrt/transform/pipeline.py  (new function alongside build_default_pipeline)

def build_training_pipeline() -> TransformPipeline:
    from python.zrt.transform.parallel import (
        TensorParallelPass, ExpertParallelPass, CommInserterPass,
        ContextParallelPass, PipelineParallelPass, DataParallelPass,
    )
    from python.zrt.transform.fusion import FusionPass
    from python.zrt.transform.optim import QuantizationPass, EPLBPass, SharedExpertPass, MTPPass
    from python.zrt.transform.training import (
        RecomputePass, ZeroFSDPPass, OffloadPass, OptimizerPass,
    )
    from python.zrt.transform.analysis import (
        TrainFlopsPass, RooflinePass, CommLatencyPass, StreamAssignPass, TrainMemoryPass,
    )

    pipe = TransformPipeline()
    is_train = lambda c: c.is_training

    # ── Stage 1: Split ────────────────────────────────────────────────────────
    pipe.add("split", TensorParallelPass(),   condition=lambda c: c.parallel.tp > 1)
    pipe.add("split", ContextParallelPass(),  condition=lambda c: c.parallel.cp > 1)
    pipe.add("split", ExpertParallelPass(),   condition=lambda c: c.parallel.ep > 1)
    pipe.add("split", PipelineParallelPass(), condition=lambda c: c.parallel.pp > 1)
    pipe.add("split", DataParallelPass(),     condition=lambda c: is_train(c) and c.parallel.dp > 1)
    pipe.add("split", CommInserterPass(),
             condition=lambda c: c.parallel.tp > 1 or c.parallel.cp > 1 or c.parallel.ep > 1
                                 or (is_train(c) and c.parallel.dp > 1))

    # ── Stage 2: Fuse ─────────────────────────────────────────────────────────
    pipe.add("fuse", FusionPass())

    # ── Stage 3: Optim ────────────────────────────────────────────────────────
    pipe.add("optim", RecomputePass(),  condition=is_train)
    pipe.add("optim", ZeroFSDPPass(),   condition=lambda c: is_train(c) and c.training.zero_stage >= 1)
    pipe.add("optim", OffloadPass(),    condition=lambda c: is_train(c) and c.training.offload.pct > 0)
    pipe.add("optim", OptimizerPass(),  condition=is_train)
    pipe.add("optim", QuantizationPass(), condition=lambda c: c.quant is not None)
    pipe.add("optim", EPLBPass(),       condition=lambda c: "eplb" in c.optim_flags)
    pipe.add("optim", SharedExpertPass(),condition=lambda c: "shared_expert_external" in c.optim_flags)
    pipe.add("optim", MTPPass(),        condition=lambda c: "mtp" in c.optim_flags)

    # ── Stage 4: Analyze ──────────────────────────────────────────────────────
    pipe.add("analyze", TrainFlopsPass(), condition=is_train)
    pipe.add("analyze", RooflinePass())
    pipe.add("analyze", CommLatencyPass())
    pipe.add("analyze", StreamAssignPass())     # reads training.tp_overlap etc.
    pipe.add("analyze", TrainMemoryPass(), condition=is_train)

    return pipe
```

---

## 4. New passes — specifications

### 4.1 `ContextParallelPass` — `parallel/context_parallel.py`

Mirrors `TensorParallelPass` in style. Reads `ctx.parallel.cp`, `ctx.training.cp_kind`.

```python
class ContextParallelPass(GraphPass):
    name = "context_parallel"

    def run(self, graph, ctx):
        if ctx.parallel.cp <= 1: return graph
        g = graph.clone()

        for node in g.topo_sort():
            if not _is_attention_op(node):      # attn_core / flash_attn / softmax-scaled-dot
                continue

            if ctx.training.cp_kind == "ulysses":
                # Head-dim sharding: inputs arrive seq-sharded; A2A scatter-seq / gather-heads
                # before attn; inverse A2A after.
                # Annotate; CommInserterPass (extended) inserts the A2A nodes.
                node.annotations["cp_split"] = {"kind": "ulysses", "cp": ctx.parallel.cp}
                # Halve heads dim on Q/K/V/O view; multiply by CP on seq dim for attn core input
                ...
            elif ctx.training.cp_kind == "ring":
                # Seq-dim sharding inside attention: N rounds of P2P send/recv of KV chunks
                # each round overlaps with a flash-attn tile computation
                node.annotations["cp_split"] = {"kind": "ring", "cp": ctx.parallel.cp,
                                                "p2p_rounds": ctx.parallel.cp}
                ...
        return g
```

Validation goes in `transform/parallel/__init__.py` or a shared `validate.py`: Ulysses requires `num_heads % cp == 0`; Ring requires `seq_len % (cp · block_size) == 0`.

### 4.2 `PipelineParallelPass` — `parallel/pipeline_parallel.py`

Does **not** split the graph. It annotates every node with its `pp_stage` and inserts `comm.send_recv` P2P nodes at stage boundaries. The schedule composer later takes the single annotated graph and replays it per stage per μbatch.

```python
class PipelineParallelPass(GraphPass):
    name = "pipeline_parallel"

    def run(self, graph, ctx):
        if ctx.parallel.pp <= 1: return graph
        g = graph.clone()

        # 1. collect layer_id ordering from g.hierarchy (uses existing GraphHierarchy)
        layers = _ordered_layers(g)   # [(layer_id, layer_kind, total_flops)]
        # 2. partition into pp stages — either explicit (ctx.training.pp_layer_assignment)
        #    or greedy bin-pack on (fwd + bwd) cost to balance stages.
        stages = _partition(layers, ctx.parallel.pp, ctx.training.pp_layer_assignment)
        # 3. annotate every op with pp_stage
        for op in g.nodes.values():
            op.annotations["pp_stage"] = _stage_of(op, stages)
        # 4. insert P2P comm nodes at the last op of stage s → first op of stage s+1
        _insert_pp_p2p(g, stages)
        # 5. warn if max(stage_cost)/min(stage_cost) > 1.10 for dense+moe+mtp mix
        return g
```

### 4.3 `DataParallelPass` — `parallel/data_parallel.py`

Appends a gradient-reduction op group at the tail of `train_backward` graphs. Choice of AR vs RS depends on `ctx.training.zero_stage` (ZeRO-0 → AR on full grad buffer; ZeRO-1/2/3 → RS since each DP rank only owns a shard).

```python
class DataParallelPass(GraphPass):
    name = "data_parallel"

    def run(self, graph, ctx):
        if not ctx.is_training or ctx.parallel.dp <= 1 or graph.phase != "train_backward":
            return graph
        g = graph.clone()

        # scan graph for grad tensors; sum sizes per bucket (fused buckets, one per PP stage)
        grad_bytes = _collect_grad_bytes(g)
        collective = "all_reduce" if ctx.training.zero_stage == 0 else "reduce_scatter"

        comm_node = OpNode(
            id="comm_grad_reduce",
            op_type=f"comm.{collective}",
            inputs=[], outputs=[],            # bucket is virtual
            attrs={"group_size": ctx.parallel.dp,
                   "collective": collective,
                   "role": "dp_grad_reduce",
                   "bucket_bytes": grad_bytes},
            scope="dp",
            category="communication",
        )
        comm_node.annotations["inserted_by"] = "dp_pass"
        _append_at_end(g, comm_node)
        return g
```

Stream assignment later puts this on a comm stream; if `dp_overlap_in_bubble` is set, the schedule composer subtracts `pp_bubble_time` from its exposed latency.

### 4.4 Extended `CommInserterPass`

Add two methods on the existing class:

```python
def _insert_cp_comm(self, g, ctx):
    cp = ctx.parallel.cp
    if cp <= 1: return
    for n in list(g.topo_sort()):
        cps = n.annotations.get("cp_split")
        if not cps: continue
        if cps["kind"] == "ulysses":
            # A2A scatter-seq/gather-heads BEFORE attn core; inverse A2A AFTER.
            _prepend_comm(g, n.id, _make_comm_node(f"a2a_cp_pre_{n.id}",  "all_to_all", n, cp))
            _rewire      (g, n.id, _make_comm_node(f"a2a_cp_post_{n.id}", "all_to_all", n, cp))
        elif cps["kind"] == "ring":
            # N rounds of P2P, each annotated as its own comm node for overlap accounting.
            for i in range(cps["p2p_rounds"]):
                _insert_ring_p2p(g, n.id, i, cp)
```

### 4.5 `RecomputePass` — `training/recompute.py`

Reads `ctx.training.recompute.per_layer_kind`, finds matching ops in `train_forward` graph, adds `recompute=True` annotation. `TrainMemoryPass` later drops those activations from the peak calculation; `TrainFlopsPass` adds a recompute multiplier to `fwd_flops` when computing total training flops.

```python
class RecomputePass(GraphPass):
    name = "recompute"

    def run(self, graph, ctx):
        if graph.phase != "train_forward":
            return graph       # recompute is a forward-graph annotation
        g = graph.clone()
        policy = ctx.training.recompute.per_layer_kind
        for node in g.nodes.values():
            lk = _layer_kind_of(node)              # "dense" | "moe" | "mtp"
            tiers = policy.get(lk, set())
            if _matches_any_tier(node, tiers):
                node.annotations["recompute"] = True
                node.annotations["recompute_tier"] = _matching_tier(node, tiers)
        return g
```

Tiers (implemented as op-kind matchers):
- `"full"` — all ops in layer
- `"attn"` — softmax + attn_core + O-proj backward inputs
- `"attn_upscale"` — just softmax, per Korthikanti §4
- `"ffn_swiglu"` — swiglu activation only
- `"ln"` — layernorm

### 4.6 `ZeroFSDPPass` — `training/zero_fsdp.py`

Pure annotation pass — no graph structure change. Writes shard factors to a side-channel the memory model consumes.

```python
class ZeroFSDPPass(GraphPass):
    name = "zero_fsdp"

    def run(self, graph, ctx):
        g = graph.clone()
        z = ctx.training.zero_stage
        dp = ctx.parallel.dp
        # shard factors per memory bucket
        g.metadata["zero"] = {
            "stage":          z,
            "weight_shard":   dp if z >= 3 else 1,
            "grad_shard":     dp if z >= 2 else 1,
            "optstate_shard": dp if z >= 1 else 1,
        }
        # FSDP-3 also adds a per-layer AG during fwd/bwd — insert those collectives
        if z >= 3:
            _insert_fsdp_ag(g, ctx)
        return g
```

### 4.7 `OffloadPass` — `training/offload.py`

Inserts `comm.d2h` / `comm.h2d` nodes (new category in `node.py :: _COMM_OPS` — see §6) for offloaded buckets. Latency estimated by `CommLatencyPass` using PCIe BW from `HardwareSpec` (new field — see §6).

### 4.8 `OptimizerPass` — `training/optimizer.py`

Appends a virtual `optimizer.step` op to `train_backward` graphs:

```python
class OptimizerPass(GraphPass):
    name = "optimizer"

    def run(self, graph, ctx):
        if graph.phase != "train_backward": return graph
        g = graph.clone()

        params = _total_params_on_rank(g, ctx)
        opt = ctx.training.optimizer  # "adam" | "muon"

        step_node = OpNode(
            id="optimizer_step",
            op_type=f"optimizer.{opt}",
            inputs=[], outputs=[],
            attrs={
                "optimizer": opt,
                "params":    params,
                "state_bytes": _opt_state_bytes(opt, params),  # Adam: 8P bytes; Muon: 4P + scratch
                "step_flops": _opt_step_flops(opt, params),    # Muon adds Newton-Schulz cost
            },
            scope="optimizer",
            category="compute",
        )
        _append_at_end(g, step_node)
        return g
```

### 4.9 `TrainFlopsPass` — `analysis/flops_train.py`

Subclasses the existing `FlopsPass`. Where the inference `FlopsPass` annotates a single `flops` number, this annotates three:

```python
class TrainFlopsPass(FlopsPass):
    name = "flops_train"

    def run(self, graph, ctx):
        from python.zrt.simulator.backends.roofline import RooflineSimulator
        sim = RooflineSimulator()
        g = graph.clone()

        for node in g.nodes.values():
            fwd, read_b, write_b = sim._fmr(node)
            dx, dw = _grad_flops(node, fwd)           # op-kind specific
            # recompute: if this op will be recomputed, add one extra fwd
            rec_mult = 2.0 if node.annotations.get("recompute") else 1.0

            node.annotations.update({
                "flops_fwd": int(fwd * rec_mult),
                "flops_dx":  int(dx),
                "flops_dw":  int(dw),
                "read_bytes":  int(read_b),
                "write_bytes": int(write_b),
                # legacy key — sum of fwd/dx/dw for the relevant phase
                "flops": int(fwd * rec_mult) if graph.phase == "train_forward"
                         else int(dx + dw),
            })
        return g
```

`_grad_flops(op, fwd)` dispatch rules:
- matmul (m, n, k): `dx = 2·m·n·k`, `dw = 2·m·n·k`
- flash attention: `dx ≈ 2.5 × fwd` (measured), `dw = 0`
- softmax/ln/swiglu/rope (bandwidth-bound): `dx = fwd` (byte-wise), `dw = 0`
- embed / lm_head: normal matmul rules
- comm/memory nodes: all zero

### 4.10 `TrainMemoryPass` — `analysis/memory_train.py`

Extends `zrt/memory/model.py`'s `MemoryModel` with three new buckets and sharding. Produces a new `TrainingMemoryBudget` that supersedes `MemoryBudget` during training.

```python
@dataclass
class TrainingMemoryBudget:
    weights_mb:        float
    grads_mb:          float
    opt_state_mb:      float
    activations_mb:    float      # respects recompute mask and in-flight μbatches
    comm_buffer_mb:    float
    offloaded_mb:      float      # resident on host (not counted toward HBM)
    overhead_mb:       float
    total_hbm_mb:      float      # sum minus offloaded
    host_resident_mb:  float
    capacity_mb:       float
    is_feasible:       bool

class TrainMemoryPass(GraphPass):
    name = "train_memory"

    def run(self, graph, ctx):
        from python.zrt.memory.model import MemoryModel
        g = graph.clone()
        shards = g.metadata.get("zero", {"weight_shard": 1, "grad_shard": 1, "optstate_shard": 1})

        P = _params_on_rank(g, ctx)                    # after TP/PP/EP sharding
        param_bytes  = P * _dtype_bytes(ctx.training.param_dtype)  / shards["weight_shard"]
        grad_bytes   = P * _dtype_bytes(ctx.training.grad_dtype)   / shards["grad_shard"]
        opt_bytes    = _opt_state_total_bytes(ctx.training.optimizer, P) / shards["optstate_shard"]

        # activations:  Korthikanti eq.  with
        #   - SP factor (1/tp on seq-sharded portion)
        #   - CP factor (1/cp on attn activation)
        #   - recompute mask (drop recomputed ops)
        #   - in-flight μbatches (depends on PP schedule)
        act_bytes = _activations_korthikanti(g, ctx)

        # offload reduces HBM residency but increases PCIe traffic (handled by CommLatencyPass)
        off = ctx.training.offload
        offloaded = 0.0
        if off.opt_state: offloaded += opt_bytes * off.pct; opt_bytes *= (1 - off.pct)
        if off.grads:     offloaded += grad_bytes * off.pct; grad_bytes *= (1 - off.pct)
        if off.params:    offloaded += param_bytes * off.pct; param_bytes *= (1 - off.pct)

        budget = TrainingMemoryBudget(...)
        g.metadata["train_memory"] = budget
        return g
```

### 4.11 `StreamAssignPass` extension — training overlap rules

The existing pass round-robins ops onto compute / comm streams. Training needs four additional assignment rules that, combined with the scheduler, produce the overlap behaviour needed by CoC / MC2 / Ring-CP / DualPipe:

```python
# in StreamAssignPass.run(), AFTER existing logic, add:
if ctx.is_training and ctx.training is not None:
    # Rule 1: TP CoC — tile AG/RS into k shards, each on its own comm stream slot
    if ctx.training.tp_overlap == "coc":
        _apply_coc_tiling(g, ctx, k=4)

    # Rule 2: TP MC2 — fuse comm into matmul (annotate matmul with coupled_comm_us)
    if ctx.training.tp_overlap == "mc2":
        _apply_mc2_fusion(g, ctx)

    # Rule 3: Ring-CP — per-round P2P on a rotating comm stream; each round paired
    # with an FA-tile compute op so the scheduler overlaps them
    _apply_ring_cp_overlap(g, ctx)

    # Rule 4: DualPipe dual-batch — μbatch_i's EP A2A on the SAME compute stream as
    # μbatch_{i+1}'s shared-expert compute so the scheduler interleaves them
    if ctx.training.dualbatch:
        _apply_dualbatch_pairing(g, ctx)
```

Each helper is ~30 lines. No new scheduler work.

---

## 5. Pipeline schedule composer (new module)

This is the one piece not covered by a single-graph `DAGScheduler` run. It lives in `zrt/transform/schedule/` (a sibling of `analysis/`) and runs **after** the pipeline.

### 5.1 `composer.py`

```python
# zrt/transform/schedule/composer.py

@dataclass
class PipelineStepResult:
    step_time_ms:   float
    per_stage:      list[Timeline]
    bubble_ms:      float
    warmup_ms:      float
    steady_ms:      float
    cooldown_ms:    float
    dp_ar_exposed_ms: float

class PipelineScheduleComposer:
    def __init__(self, scheduler: "DAGScheduler"):
        self._sch = scheduler

    def compose(self, fwd: OpGraph, bwd: OpGraph, ctx: TransformContext) -> PipelineStepResult:
        pp = ctx.parallel.pp
        # 1. split the single annotated graph into per-stage subgraphs using pp_stage annotation
        stages_fwd = [fwd.subgraph(_ids_on_stage(fwd, s)) for s in range(pp)]
        stages_bwd = [bwd.subgraph(_ids_on_stage(bwd, s)) for s in range(pp)]
        # 2. schedule each stage once (μbatch timing is the same by assumption)
        t_fwd = [self._sch.schedule(g).total_latency_us for g in stages_fwd]
        t_bwd = [self._sch.schedule(g).total_latency_us for g in stages_bwd]
        # 3. compose according to schedule kind
        sched = ctx.training.pp_schedule
        if sched == "1f1b":      return self._one_f_one_b(t_fwd, t_bwd, ctx)
        if sched == "i1f1b":     return self._interleaved(t_fwd, t_bwd, ctx)
        if sched == "dualpipe":  return self._dualpipe(t_fwd, t_bwd, ctx)
        if sched == "dualpipev": return self._dualpipev(t_fwd, t_bwd, ctx)
        if sched == "zb":        return self._zero_bubble(t_fwd, t_bwd, ctx)
        raise ValueError(sched)
```

### 5.2 Schedule formulas (implemented one per file)

**1F1B** (`one_f_one_b.py`) — standard Megatron:
```
warmup   = (pp - 1) · t_fwd[0]
steady   = M · max(t_fwd[s] + t_bwd[s])
cooldown = (pp - 1) · t_bwd[-1]
step = warmup + steady + cooldown + dp_ar_exposed
```

**Interleaved 1F1B / VPP** (`interleaved.py`):
```
bubble_fraction = (pp - 1) / (v · M)
step = (1 + bubble_fraction) · M · max(t_fwd[s] + t_bwd[s]) + dp_ar_exposed
```

**DualPipe** (`dualpipe.py`) — DeepSeek-V3 dual-direction schedule; bubble roughly halved. Cross-stage compute-comm overlap is already captured per-stage by the single-graph scheduler (via `dualbatch` rule §4.11); the composer just applies the smaller bubble constant.

**DualPipeV** (`dualpipev.py`) — Sea AI Lab variant; similar formula with different bubble constant.

**Zero Bubble** (`zero_bubble.py`) — stub returning 1F1B (noted as low priority in requirements).

### 5.3 DP AR in bubble

```python
def _fold_dp_into_bubble(step: PipelineStepResult, ctx) -> PipelineStepResult:
    if not ctx.training.dp_overlap_in_bubble: return step
    bubble = step.warmup_ms + step.cooldown_ms     # idle windows
    hidden = min(bubble, step.dp_ar_exposed_ms)
    step.step_time_ms  -= hidden
    step.dp_ar_exposed_ms -= hidden
    return step
```

---

## 6. Minor extensions to existing modules

| File | Change |
|---|---|
| `zrt/ir/node.py` | Add `"comm.d2h"`, `"comm.h2d"` to `_COMM_OPS` (for offload). |
| `zrt/hardware/spec.py` | Add `host_mem_gb`, `pcie_bw_gbps`, `pcie_latency_us` fields; `HardwareSpec.pcie_bandwidth()`. |
| `zrt/transform/analysis/comm_latency.py` | Add branch for `d2h` / `h2d` using `pcie_bandwidth()`. |
| `zrt/memory/model.py` | No change — training uses `TrainMemoryPass` which produces `TrainingMemoryBudget` in `graph.metadata["train_memory"]`. The existing `MemoryModel` continues to handle inference. |
| `zrt/transform/parallel/expert_parallel.py` | Read optional `ctx.training.expert_imbalance` (or a new `ParallelConfig.expert_imbalance` if simpler) and emit a load-factor multiplier on the bottleneck expert's compute time in an annotation. |
| `zrt/transform/context.py` | Add `cp` to `ParallelConfig`; add `TrainingConfig` dataclass and `training` field + `is_training` property on `TransformContext`. |
| `zrt/transform/pipeline.py` | Add `build_training_pipeline()`. |
| `zrt/graph/main.py` | No change — `run_trace_phases` already supports `train_forward`/`train_backward`. |

---

## 7. Public API

```python
# new high-level entry point — zrt/__init__.py or a new module
def estimate_training_step(
    model_id: str,
    hw_spec: HardwareSpec,
    parallel: ParallelConfig,
    training: TrainingConfig,
    stream_config: StreamConfig | None = None,
    num_layers: int | None = None,            # trace only N layers for speed
) -> TrainingReport:
    # 1. trace — reuses existing
    from python.zrt.graph import run_trace_phases
    result = run_trace_phases(
        model_id=model_id,
        num_layers=num_layers,
        batch_size=training.micro_batch,
        seq_len=_seq_len_from_model(model_id, training),
        phases=("train_forward", "train_backward"),
    )
    fwd_raw, _ = result.graphs["train_forward"]
    bwd_raw, _ = result.graphs["train_backward"]

    # 2. transform
    from python.zrt.transform import build_training_pipeline
    ctx = TransformContext(hw_spec=hw_spec, parallel=parallel,
                           stream_config=stream_config or StreamConfig(),
                           training=training, phase="train_forward")
    pipe = build_training_pipeline()
    fwd = pipe.run(fwd_raw, ctx)
    ctx_bwd = replace(ctx, phase="train_backward")
    bwd = pipe.run(bwd_raw, ctx_bwd)

    # 3. compose
    from python.zrt.executor.scheduler import DAGScheduler
    from python.zrt.transform.schedule.composer import PipelineScheduleComposer
    composer = PipelineScheduleComposer(DAGScheduler(hw_spec))
    step = composer.compose(fwd, bwd, ctx)

    # 4. report
    return TrainingReport(
        step_time_ms = step.step_time_ms,
        mfu          = _compute_mfu(fwd, bwd, step, hw_spec),
        memory       = bwd.metadata["train_memory"],
        per_stage    = step.per_stage,
    )
```

A `search.sweep(...)` helper builds on this for Pareto sweeps (unchanged from the previous design doc — just swaps in `estimate_training_step`).

---

## 8. Testing

| Test | Anchor | File |
|---|---|---|
| Training trace produces valid IR | DeepSeek-V3, 2 layers | already in `tests/test_train_trace.py` |
| CP pass preserves graph semantics | Llama-3 8B, CP=2 | new `tests/test_context_parallel.py` |
| PP stage balance ≤ 10% skew | DeepSeek-V3 (dense+moe+mtp mix), PP=8 | new `tests/test_pipeline_parallel.py` |
| ZeRO-1/2/3 memory matches paper | Llama-3 70B on 64×H100 | new `tests/test_zero_memory.py` |
| Recompute (full / attn / attn_upscale) reduces activation memory correctly | Llama-3 8B | new `tests/test_recompute.py` |
| 1F1B step time matches Megatron paper §3.2 within 10% | GPT-3 175B on Selene-scale | new `tests/test_pipeline_schedule.py` |
| DualPipe bubble is ~½ of I1F1B under the same config | DeepSeek-V3 | new `tests/test_dualpipe.py` |
| End-to-end estimator matches published MFU within 10% | Anchor table (GPT-3 175B, Llama-3 70B, DeepSeek-V3) | new `validation/anchors_training/` |

Pattern follows the existing `validation/` directory. Each anchor is a yaml config + expected ranges.

---

## 9. Implementation phasing

| Phase | Deliverables | Files touched |
|---|---|---|
| **Phase 1 — Spine (~1 week)** | `TrainingConfig` on `TransformContext`; `TrainFlopsPass` (dx/dw split) and `TrainMemoryPass` (grads + opt + ZeRO-0 + activations); `OptimizerPass` Adam only; `DataParallelPass`; single-graph training estimate (no PP composer). Anchor: Llama-3 70B DP-only. | `transform/context.py`, `transform/training/{optimizer,}.py`, `transform/analysis/{flops_train,memory_train}.py`, `transform/parallel/data_parallel.py`, `transform/pipeline.py` |
| **Phase 2 — Parallelism (~2 weeks)** | `ContextParallelPass` (Ulysses + Ring); `PipelineParallelPass` + stage bin-pack; `PipelineScheduleComposer` with 1F1B and VPP; extended `CommInserterPass`; `RecomputePass`; Muon optimizer. Anchor: Llama-3 70B 3D parallel. | `transform/parallel/{context_parallel,pipeline_parallel}.py`, `transform/parallel/comm_inserter.py` (extend), `transform/training/recompute.py`, `transform/schedule/{one_f_one_b,interleaved,composer}.py` |
| **Phase 3 — MoE & ZeRO (~2 weeks)** | Expert imbalance in `ExpertParallelPass`; ZeRO-1/2/3 / FSDP in `ZeroFSDPPass`; MTP layer support in existing `MTPPass`; DualPipe + DualPipeV schedules; `dualbatch` overlap rule. Anchor: DeepSeek-V3 DualPipe config. | `transform/parallel/expert_parallel.py` (extend), `transform/training/zero_fsdp.py`, `transform/schedule/{dualpipe,dualpipev}.py`, `transform/analysis/passes.py` (StreamAssignPass extension) |
| **Phase 4 — Overlap & offload (~2 weeks)** | CoC / MC2 TP overlap rules in `StreamAssignPass`; Ring-CP overlap pairing; `OffloadPass` with PCIe latency; DP-in-bubble folding; sweep driver. Anchor: DeepSeek-V3 full config. | `transform/analysis/passes.py` (extend), `transform/training/offload.py`, `transform/schedule/composer.py`, `hardware/spec.py` (PCIe fields) |

Zero-bubble PP is deferred per requirements.

---

## 10. Key references

- Codebase: [`laksjdf/modeling` @ `main_dev_ytq`](https://github.com/laksjdf/modeling/tree/main_dev_ytq)
- Megatron-LM PP+VPP — [Narayanan et al. 2021](https://arxiv.org/pdf/2104.04473)
- Activation recompute — [Korthikanti et al. 2022](https://arxiv.org/pdf/2205.05198)
- ZeRO / FSDP — [Rajbhandari et al. 2020](https://arxiv.org/pdf/1910.02054)
- ZeRO-Offload — [Ren et al. 2021](https://arxiv.org/pdf/2101.06840)
- Ulysses CP — [Jacobs et al. 2023](https://arxiv.org/pdf/2309.14509)
- Ring Attention — [Liu et al. 2023](https://arxiv.org/pdf/2310.01889)
- DeepSeek-V3 (DualPipe, MTP, EP) — [DeepSeek 2024](https://arxiv.org/pdf/2412.19437)
- DualPipeV — [Sea AI Lab](https://sail.sea.com/blog/articles/63)
- AdamW — [Loshchilov & Hutter 2017](https://arxiv.org/pdf/1711.05101)
- Muon — [苏剑林](https://kexue.fm/archives/10592)
- Memory analysis (DeepSeek) — [Yang et al. 2025](https://arxiv.org/pdf/2502.07846)
- Calculon — [Isaev et al. SC'23](https://dl.acm.org/doi/10.1145/3581784.3607102)
- MindSpeed CoC / MC2 — [CoC doc](https://gitee.com/ascend/MindSpeed-LLM/blob/2.1.0/docs/pytorch/features/communication-over-computation.md), [MC2 doc](https://gitee.com/ascend/MindSpeed-LLM/blob/2.1.0/docs/pytorch/features/mc2.md)
