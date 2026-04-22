# AI Training Infra Modeller — Software Design Spec

A Calculon-style analytical performance & memory modeller for transformer training, extended for modern parallelism (TP/CP/PP/EP/DP + VPP/DualPipe), memory optimizations (ZeRO/FSDP, recompute, offload), communication–compute overlap, and MoE/MTP workloads.

**Reference**: [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [Calculon](https://dl.acm.org/doi/10.1145/3581784.3607102).

---

## 1. Architectural Principles

1. **Single IR, many models.** Inputs (model/system/strategy) are compiled once into an op-level Intermediate Representation. FLOPs, memory, comm, and overlap models all read from the same IR — they never re-derive shapes.
2. **Ops, not layers, are atomic.** Each transformer block decomposes into ~15 ops (QKV-proj, RoPE, attn-core, O-proj, LN×2, up/gate/down, MoE dispatch/combine, router, AR/RS/A2A/P2P). This is what enables per-op recompute, precise dx/dw accounting, and overlap reasoning.
3. **Collectives are IR nodes.** AllGather / ReduceScatter / All-to-All / P2P are inserted during the sharding pass, so the overlap model can pair each collective with the compute op it hides behind.
4. **Composition is layered:** op → layer → stage → pipeline → step. Each layer has a single well-defined responsibility and is independently testable.
5. **Analytical first, empirical second.** All formulas are closed-form and fast (µs per config) so search can sweep millions of strategies. Empirical tables (achieved TFLOPs vs matmul size, achieved BW vs msg size) are injected as lookup curves, not simulations.
6. **Strategy is data, not code.** A `Strategy` dataclass is the search variable; the optimizer enumerates it.

---

## 2. Top-Level Layout

```
ai_infra_modeller/
├── spec/                    # L1 — Inputs (dataclasses)
│   ├── model.py             # ModelSpec, LayerKind
│   ├── system.py            # SystemSpec, GPU, NetTier, Topology
│   └── strategy.py          # Strategy, PPSched, CPKind, RecomputePolicy, ...
├── ir/                      # L2 — IR + sharding
│   ├── graph.py             # Graph, Op, Tensor, Collective
│   ├── builders.py          # dense_block(), moe_block(), mtp_block()
│   ├── shard.py             # ShardPlan, insert_collectives()
│   └── validate.py          # divisibility, PP balance, EP placement
├── models/                  # L3 — Analytical models
│   ├── flops.py             # per-op fwd / dx / dw
│   ├── memory.py            # weights · grads · opt · activations
│   ├── comm.py              # α-β per collective, per net tier
│   └── overlap.py           # CoC / MC2 / Ring-CP / DualPipe
├── compose/                 # L4 — Composition
│   ├── stage.py             # per-stage time for one μbatch
│   └── pipeline.py          # 1F1B / I1F1B / DualPipe / ZB → step time
├── search/                  # L5 — Outputs
│   ├── estimator.py         # single-point evaluation
│   ├── sweep.py             # grid / pareto / pruning
│   └── report.py            # json/csv/trace emitters
├── io/
│   ├── config_loader.py     # yaml → specs
│   └── perf_tables.py       # empirical flops/BW curves
├── cli.py
└── api.py
```

**Data flow:**

```
yaml ─► ModelSpec + SystemSpec + Strategy
              │
              ▼
        build_ir() ──► Graph (ops + collectives, sharded shapes)
              │
      ┌───────┼─────────┬──────────┐
      ▼       ▼         ▼          ▼
   flops   memory    comm       overlap
      │       │         │          │
      └───────┴────┬────┴──────────┘
                   ▼
            StageComposer ──► per-stage time
                   ▼
          PipelineComposer ──► step time + memory HWM
                   ▼
         Report / Sweep / Pareto
```

---

## 3. Input Specs (L1)

### 3.1 `ModelSpec`

```python
from dataclasses import dataclass, field
from enum import Enum

class Dtype(Enum):
    FP32 = 4; BF16 = 2; FP16 = 2; FP8 = 1
    @property
    def bytes(self) -> int: return self.value

class LayerKind(Enum):
    DENSE = "dense"
    MOE   = "moe"
    MTP   = "mtp"       # multi-token prediction (DeepSeek-V3)

@dataclass
class ModelSpec:
    # geometry
    hidden: int
    ffn: int
    num_heads: int
    num_kv_heads: int              # GQA/MQA; == num_heads for MHA
    head_dim: int
    vocab: int
    seq_len: int

    # layer composition — order matters for PP balance
    layers: list[LayerKind]

    # MoE (ignored if no MOE layers)
    num_experts: int = 0
    moe_ffn: int = 0               # per-expert FFN, often < dense ffn
    top_k: int = 0
    capacity_factor: float = 1.0
    expert_imbalance: float = 0.0  # empirical load skew, 0 = perfect

    # MTP
    mtp_depth: int = 0

    # dtypes
    param_dtype: Dtype = Dtype.BF16
    grad_dtype: Dtype = Dtype.FP32
    master_dtype: Dtype = Dtype.FP32
    act_dtype:  Dtype = Dtype.BF16
```

**Notes.** Parameter count is *derived*, never user-specified — avoids drift between "nominal 70B" and actual sharded shapes. Heterogeneous `layers` list is essential because dense, MoE, and MTP layers have different per-layer cost, and PP stage balance depends on it.

### 3.2 `SystemSpec`

```python
@dataclass
class GPU:
    name: str
    flops_bf16: float   # peak TFLOP/s
    flops_fp8:  float
    hbm_gb: float
    hbm_bw_gbps: float  # per-GPU aggregate HBM BW

@dataclass
class NetTier:
    scope: str          # "intra_node" | "inter_node" | "scale_out"
    bw_gbps: float      # per-link unidirectional
    latency_us: float
    topology: str       # "ring" | "tree" | "fattree" | "nvswitch"

@dataclass
class SystemSpec:
    gpu: GPU
    host_mem_gb: float          # for offload
    nets: list[NetTier]         # ordered: intra first
    nodes: int
    gpus_per_node: int
    @property
    def world_size(self) -> int: return self.nodes * self.gpus_per_node
```

**Notes.** Multi-tier networks matter — intra-node NVLink and inter-node IB have very different α, β. The comm model picks the tier based on which ranks participate in the collective (governed by the rank-grid layout in L4).

### 3.3 `Strategy`

```python
class PPSched(Enum):
    ONE_F_ONE_B = "1f1b"
    INTERLEAVED = "i1f1b"       # VPP
    ZERO_BUBBLE = "zb"          # low priority
    DUALPIPE    = "dualpipe"
    DUALPIPE_V  = "dualpipev"

class CPKind(Enum):
    NONE     = "none"
    ULYSSES  = "ulysses"        # split heads, A2A
    RING     = "ring"           # split seq, P2P + blockwise FA
    HYBRID   = "hybrid"

class TPOverlap(Enum):
    NONE = "none"
    COC  = "coc"                # tiled pipelining of AG/RS vs matmul
    MC2  = "mc2"                # fused communication+computation kernel

class OptKind(Enum):
    ADAM = "adam"
    MUON = "muon"

@dataclass
class RecomputePolicy:
    # per-LayerKind → set of op categories to recompute
    # op categories: "full" | "attn" | "attn_upscale" | "ffn_swiglu" | "ln"
    per_layer: dict[LayerKind, set[str]] = field(default_factory=dict)

@dataclass
class OffloadPolicy:
    opt_state: bool = False
    grads:     bool = False
    params:    bool = False     # ZeRO-Infinity style
    pct:       float = 1.0      # fraction offloaded (0..1)

@dataclass
class Strategy:
    # parallelism degrees
    tp: int; cp: int; pp: int; ep: int; dp: int

    # batch
    micro_batch: int
    global_batch: int

    # pipeline
    pp_schedule: PPSched = PPSched.ONE_F_ONE_B
    vpp_chunks: int = 1
    pp_layer_assignment: list[int] | None = None   # explicit stage→layers, else auto-balance

    # context parallel
    cp_kind: CPKind = CPKind.NONE

    # memory
    zero_stage: int = 0          # 0/1/2/3;  3 == FSDP
    recompute: RecomputePolicy = field(default_factory=RecomputePolicy)
    offload:   OffloadPolicy   = field(default_factory=OffloadPolicy)

    # overlap
    tp_overlap: TPOverlap = TPOverlap.NONE
    ep_overlap: bool = False     # dispatch/combine hidden behind shared-expert compute
    dualbatch:  bool = False     # DualPipe / chunked DualPipeV dual-μbatch overlap
    dp_overlap_in_bubble: bool = True

    # optimizer
    optimizer: OptKind = OptKind.ADAM

    def validate(self, model: ModelSpec, system: SystemSpec):
        assert self.tp * self.cp * self.pp * self.dp == system.world_size, \
            "TP·CP·PP·DP must equal world_size (EP is a sub-grid of DP×TP)"
        assert model.num_heads  % self.tp == 0
        assert model.num_kv_heads % self.tp in (0,)  # GQA edge case handled in IR
        if self.cp_kind == CPKind.ULYSSES:
            assert model.num_heads % self.cp == 0, "Ulysses CP shards heads"
        if self.ep > 1:
            assert model.num_experts % self.ep == 0
        # ZeRO ⊂ DP
        if self.zero_stage >= 1:
            assert self.dp > 1
```

---

## 4. IR Layer (L2)

### 4.1 Graph structure

```python
@dataclass
class Tensor:
    name: str
    shape_logical: tuple[int, ...]      # before sharding, per-sample
    shape_local:   tuple[int, ...]      # after TP/CP/EP sharding, per-rank
    dtype: Dtype
    is_activation: bool
    is_param: bool = False

@dataclass
class Op:
    name: str
    kind: str            # "matmul" | "attn_core" | "softmax" | "ln"
                         # | "swiglu" | "rope" | "router" | "dispatch"
                         # | "combine" | "embed" | "lm_head"
    inputs:  list[Tensor]
    outputs: list[Tensor]
    meta: dict           # op-specific (e.g. matmul m,n,k; attn causal flag)
    layer_id: int
    layer_kind: LayerKind

@dataclass
class Collective:
    name: str
    kind: str            # "AG" | "RS" | "AR" | "A2A" | "P2P"
    group: str           # "TP" | "CP" | "EP" | "DP" | "PP"
    bytes: int           # per-rank payload
    inserted_after: str  # op name it follows (for overlap pairing)

@dataclass
class Graph:
    ops: list[Op]
    collectives: list[Collective]
    layer_index: dict[int, tuple[int, int]]  # layer_id → (op_start, op_end)
```

### 4.2 Block builders

Implement one builder per `LayerKind`. Example signature:

```python
def dense_block(hidden, ffn, seq, num_heads, num_kv_heads,
                head_dim, shard: ShardPlan, layer_id: int) -> list[Op]:
    ops = []
    # 1. pre-attn LN
    ops.append(Op("ln1", "ln", ..., layer_id=layer_id, layer_kind=LayerKind.DENSE))
    # 2. QKV projection  — [h] × [h, 3·h_q]  (sharded on out-dim by TP)
    ops.append(matmul_op("qkv_proj", m=seq, n=3*num_heads*head_dim // shard.tp, k=hidden, ...))
    # 3. RoPE
    ops.append(Op("rope", "rope", ...))
    # 4. attn core (flash-attn)  — cost depends on causal, CP variant
    ops.append(attn_core_op("attn", seq, num_heads // shard.tp, head_dim, causal=True, cp=shard.cp_kind))
    # 5. O projection
    ops.append(matmul_op("o_proj", m=seq, n=hidden, k=num_heads*head_dim // shard.tp, ...))
    # 6. post-attn LN
    # 7. FFN up/gate (fused for SwiGLU)
    # 8. SwiGLU activation (vector)
    # 9. FFN down
    return ops
```

For MoE: router (small matmul) → top-k gate selection → dispatch (A2A) → per-expert FFN (sharded across EP) → combine (A2A) → weighted sum. Expert imbalance enters as a multiplier on the bottleneck expert's compute.

### 4.3 Sharding pass

```python
class ShardPlan:
    def __init__(self, s: Strategy):
        self.tp, self.cp, self.ep, self.dp, self.pp = s.tp, s.cp, s.ep, s.dp, s.pp
        self.cp_kind = s.cp_kind
        self.sp = (s.tp > 1)   # Megatron-SP always on when TP>1

    def shard_linear_rowparallel(self, t: Tensor) -> Tensor: ...
    def shard_linear_colparallel(self, t: Tensor) -> Tensor: ...
    def shard_attn_heads(self, t: Tensor) -> Tensor: ...    # TP and Ulysses CP
    def shard_seq(self, t: Tensor) -> Tensor: ...           # SP and Ring CP
    def shard_experts(self, t: Tensor) -> Tensor: ...       # EP

def insert_collectives(g: Graph, shard: ShardPlan):
    # After col-parallel matmul with SP: output is seq-sharded → AllGather for next op; or ReduceScatter if next is row-parallel.
    # Megatron-SP: AG before attn core, RS after O-proj; AG before FFN up, RS after FFN down.
    # Ring CP:  P2P send/recv per block of KV inside attn_core (modeled as N_blocks collectives).
    # Ulysses CP: A2A before attn_core (scatter seq, gather heads); A2A after (inverse).
    # EP: A2A dispatch before expert FFN; A2A combine after.
    # PP: P2P send/recv at stage boundaries (inserted at L4, not here).
    # DP: AR/RS inserted at optimizer step (inserted at L4).
    ...
```

### 4.4 Validation

- TP divisibility: `num_heads % tp == 0`, `ffn % tp == 0`.
- CP: Ulysses needs `num_heads % cp == 0`; Ring needs `seq_len % (cp · block_size) == 0`.
- PP balance: if `pp_layer_assignment` is None, assign layers by greedy bin-pack on per-layer cost (not count); warn if imbalance > 10%.
- EP × DP placement: EP group typically nested inside DP; detect and flag when EP crosses node boundaries (A2A becomes inter-node → expensive).

---

## 5. Analytical Models (L3)

Each returns a pure function of the IR + specs. No hidden state.

### 5.1 FLOPs model (`models/flops.py`)

```python
@dataclass
class OpCost:
    fwd_flops: float = 0.0
    dx_flops:  float = 0.0
    dw_flops:  float = 0.0
    fwd_bytes: float = 0.0     # for memory-bound ops
    dx_bytes:  float = 0.0
    dw_bytes:  float = 0.0
    bound: str = "compute"     # "compute" | "memory"

def op_cost(op: Op, model: ModelSpec) -> OpCost:
    if op.kind == "matmul":
        m, n, k = op.meta["m"], op.meta["n"], op.meta["k"]
        return OpCost(
            fwd_flops = 2*m*n*k,
            dx_flops  = 2*m*n*k,
            dw_flops  = 2*m*n*k,
        )
    if op.kind == "attn_core":
        # flash-attn: fwd ≈ 4·b·s²·h_local·d (causal halves it)
        b, s, h, d = op.meta["b"], op.meta["s"], op.meta["heads"], op.meta["head_dim"]
        mult = 2.0 if op.meta["causal"] else 4.0
        return OpCost(
            fwd_flops = mult * b * s * s * h * d,
            dx_flops  = mult * b * s * s * h * d,      # recompute-aware: flash attn bwd ≈ 2.5× fwd in practice
            dw_flops  = 0.0,
        )
    if op.kind in ("ln", "softmax", "rope", "swiglu"):
        # memory-bound: dominated by byte traffic
        return OpCost(
            fwd_bytes = op.meta["bytes_fwd"],
            dx_bytes  = op.meta["bytes_bwd"],
            bound="memory"
        )
    # ... router, dispatch/combine (memory-bound), embed, lm_head
```

**Recompute handling.** `flops.py` returns the *raw* cost per op. The stage composer later applies a multiplier `1 + r(op)` to `fwd_flops` based on the recompute policy mask. The paper reference is [Korthikanti et al. 2022](https://arxiv.org/pdf/2205.05198) §4.

### 5.2 Memory model (`models/memory.py`)

Four buckets, all per-rank, in bytes:

```python
@dataclass
class MemBreakdown:
    weights:    float
    grads:      float
    opt_state:  float
    activations: float
    comm_buffers: float = 0.0
    @property
    def total(self): return sum((self.weights, self.grads, self.opt_state,
                                 self.activations, self.comm_buffers))
```

Formulas (`P` = params held on this rank after ZeRO/TP/EP sharding):

| Bucket | Adam | Muon |
|---|---|---|
| `weights` | `P · param_dtype.bytes` | same |
| `grads`   | `P · grad_dtype.bytes`  | same |
| `opt_state` | `P · master_dtype.bytes  +  2·P·master_dtype.bytes` (master + m + v) | `P · master_dtype.bytes  +  P·master_dtype.bytes` (master + momentum matrix); add newton-schulz scratch |

ZeRO sharding divides each bucket by DP:

- ZeRO-1: `opt_state /= dp`
- ZeRO-2: `opt_state /= dp; grads /= dp`
- ZeRO-3 / FSDP: `opt_state /= dp; grads /= dp; weights /= dp` (+ AG buffer for current layer)

Activations per layer (Korthikanti eq. 2, generalized):

```
act_per_layer = seq · batch · hidden · dtype.bytes · C(layer_kind)
```

where `C` is an op-sum coefficient that depends on:
- TP (SP reduces by `1/tp` for the seq-sharded portion, not for the activation-inside-attn portion)
- CP (reduces by `1/cp`)
- recompute mask (zeroes out the ops being recomputed)

Total activations = Σ over layers on this stage × μbatches in flight (for 1F1B that's `pp - rank` for the first stage, `1` for the last).

Offload reduces HBM residency for opt_state (or grads/params) by `pct`, but adds a PCIe transfer time to the comm/overlap accounting.

**Reference:** [ZeRO paper](https://arxiv.org/pdf/1910.02054) for sharding formulas, [ZeRO-Offload](https://arxiv.org/pdf/2101.06840) for host residency, [DeepSeek Memory Analysis](https://arxiv.org/pdf/2502.07846) for MoE corrections.

### 5.3 Communication model (`models/comm.py`)

α-β latency model with topology-aware collective cost. `α` is per-link latency, `β = 1/bw` is per-byte time.

```python
def collective_time(c: Collective, group_size: int, tier: NetTier) -> float:
    alpha = tier.latency_us * 1e-6
    beta  = 1.0 / (tier.bw_gbps * 1e9 / 8)        # sec/byte
    N = group_size; S = c.bytes
    if c.kind == "AG" or c.kind == "RS":
        # ring algorithm:  (N-1) steps of S/N bytes each
        return (N-1) * (alpha + (S/N) * beta)
    if c.kind == "AR":
        # ring all-reduce = AG + RS
        return 2 * (N-1) * (alpha + (S/N) * beta)
    if c.kind == "A2A":
        # each rank sends S/N to each of N-1 peers; pairwise
        return (N-1) * (alpha + (S/N) * beta)
    if c.kind == "P2P":
        return alpha + S * beta
```

**Tier selection.** For each collective group, determine whether all ranks in the group are co-located on one node. If yes, use `intra_node`. Otherwise use `inter_node`. For mixed (e.g. TP=8 on 1 node vs TP=16 across 2 nodes) use the slower tier.

**Per-parallelism:**

- **TP + SP (Megatron):** per-layer = AG(in) + RS(out) for both attn block and FFN block. Size = `b · s/tp · h` per collective.
- **CP Ulysses:** A2A before + after attn. Size per step = `b · s/cp · h`.
- **CP Ring:** `cp` rounds of P2P send/recv of KV-chunks inside attn. Each P2P = `b · s/cp · h_kv · d · dtype`. Overlappable with the per-tile FA compute.
- **EP:** dispatch A2A `b · s · h · top_k / ep` + combine A2A same size. Group size = `ep`.
- **PP:** one P2P send + one recv per μbatch per stage boundary. Size = `b · s · h`.
- **DP:** at step end, one AR (ZeRO-0) or RS (ZeRO-2/3) of gradient buffer, size = `P · grad_dtype.bytes / dp`.

### 5.4 Overlap model (`models/overlap.py`)

Computes `t_exposed = max(0, t_comm - t_cover)` per collective.

```python
def compute_overlap(graph, flops_costs, comm_costs, strategy) -> dict[str, float]:
    exposed = {}
    # --- TP overlap (CoC / MC2) ---
    if strategy.tp_overlap == TPOverlap.COC:
        for c in tp_ag_rs_collectives:
            # tile matmul into k chunks; overlap c.bytes/k with (k-1)/k of matmul time
            mm = paired_matmul_time(c)
            exposed[c.name] = max(0, c.time - mm * (k-1)/k)
    elif strategy.tp_overlap == TPOverlap.MC2:
        # fused kernel: exposed ≈ max(c.time, mm.time) - mm.time  (effectively full hide if mm >= comm)
        ...

    # --- CP overlap ---
    if strategy.cp_kind == CPKind.RING:
        # per FA tile: compare P2P time vs block compute
        exposed["cp_p2p"] = sum(max(0, p2p - fa_tile) for each tile)
    elif strategy.cp_kind == CPKind.ULYSSES:
        exposed["cp_a2a"] = a2a_time   # no overlap possible

    # --- EP overlap (DualPipe dual-batch) ---
    if strategy.ep_overlap and strategy.dualbatch:
        # A2A of μbatch_i hidden by compute of μbatch_{i+1}
        exposed["ep_a2a"] = max(0, a2a_time - peer_compute)
    elif strategy.ep_overlap:
        # overlap dispatch with shared-expert / router compute (partial)
        exposed["ep_a2a"] = max(0, a2a_time - shared_expert_time)

    # --- DP overlap (hide in PP bubble) ---
    if strategy.dp_overlap_in_bubble:
        exposed["dp_ar"] = max(0, dp_ar_time - pp_bubble_time)

    # PP P2P: not covered per spec
    return exposed
```

**References.** CoC/MC2: [MindSpeed-LLM CoC](https://gitee.com/ascend/MindSpeed-LLM/blob/2.1.0/docs/pytorch/features/communication-over-computation.md) and [MC2](https://gitee.com/ascend/MindSpeed-LLM/blob/2.1.0/docs/pytorch/features/mc2.md). Ring CP overlap: [Liu et al. 2023](https://arxiv.org/pdf/2310.01889). DualPipe: [DeepSeek-V3 TR](https://arxiv.org/pdf/2412.19437). DualPipeV: [Sea AI Lab blog](https://sail.sea.com/blog/articles/63).

---

## 6. Composition Layer (L4)

### 6.1 Stage composer (`compose/stage.py`)

For one μbatch on one stage:

```python
def stage_time(stage_ops, flops, comm, overlap, system, strategy) -> StageTime:
    t_compute_fwd = sum(op_to_time(c.fwd_flops, c.fwd_bytes, bound, system) for c in costs)
    t_compute_dx  = sum(op_to_time(c.dx_flops,  c.dx_bytes,  bound, system) for c in costs)
    t_compute_dw  = sum(op_to_time(c.dw_flops,  c.dw_bytes,  bound, system) for c in costs)

    # recompute: extra forward before backward
    t_recompute = sum(op.fwd_flops for op in recomputed_ops) / achieved_flops

    t_comm_exposed = sum(exposed.values())   # from overlap model

    return StageTime(
        fwd = t_compute_fwd + tp_exposed_fwd + cp_exposed_fwd + ep_exposed_fwd,
        bwd = t_compute_dx + t_compute_dw + t_recompute + tp_exposed_bwd + cp_exposed_bwd,
    )

def op_to_time(flops, bytes_, bound, system) -> float:
    if bound == "compute":
        return flops / (system.gpu.flops_bf16 * 1e12 * achieved_eff(flops))
    else:
        return bytes_ / (system.gpu.hbm_bw_gbps * 1e9)
```

`achieved_eff` is a lookup curve from `io/perf_tables.py` — measured MFU vs matmul size for the target GPU. This is what keeps small-matmul corner cases honest.

### 6.2 Pipeline composer (`compose/pipeline.py`)

```python
def pipeline_step_time(stage_times, strategy, model, system) -> StepResult:
    M = strategy.global_batch // (strategy.micro_batch * strategy.dp)  # μbatches / step
    pp = strategy.pp
    t_stage_max = max(st.fwd + st.bwd for st in stage_times)  # bottleneck stage

    if strategy.pp_schedule == PPSched.ONE_F_ONE_B:
        warmup   = (pp - 1) * stage_times[0].fwd
        steady   = M * t_stage_max                 # 1F1B steady state
        cooldown = (pp - 1) * stage_times[-1].bwd
        step = warmup + steady + cooldown

    elif strategy.pp_schedule == PPSched.INTERLEAVED:
        v = strategy.vpp_chunks
        bubble_fraction = (pp - 1) / (v * M)
        step = (1 + bubble_fraction) * M * t_stage_max

    elif strategy.pp_schedule == PPSched.DUALPIPE:
        # Each stage runs two μbatches concurrently (fwd of one, bwd of other)
        # Bubble ~ halved vs I1F1B; compute-overlap of EP A2A with peer μbatch
        step = dualpipe_step(stage_times, M, pp, dualbatch=True)

    elif strategy.pp_schedule == PPSched.DUALPIPE_V:
        step = dualpipev_step(stage_times, M, pp)

    elif strategy.pp_schedule == PPSched.ZERO_BUBBLE:
        step = zb_step(stage_times, M, pp)  # low priority — stub returning 1F1B for now

    # DP allreduce at end
    t_dp = dp_allreduce_time(model, strategy, system)
    if strategy.dp_overlap_in_bubble:
        t_dp = max(0, t_dp - bubble_cover_time)
    step += t_dp

    return StepResult(step_time=step, stage_times=stage_times, ...)
```

---

## 7. Output Layer (L5)

### 7.1 Estimator (`search/estimator.py`)

Single-point evaluation.

```python
def estimate(model: ModelSpec, system: SystemSpec, strategy: Strategy) -> Report:
    strategy.validate(model, system)
    ir = build_ir(model, strategy)
    flops_costs = {op.name: op_cost(op, model) for op in ir.ops}
    comm_costs  = {c.name: collective_time(c, group_size_of(c, strategy), tier_of(c, system))
                   for c in ir.collectives}
    overlap     = compute_overlap(ir, flops_costs, comm_costs, strategy)
    mem         = memory_breakdown(ir, model, system, strategy)
    stage_times = [stage_time(stage_ops_of(s), flops_costs, comm_costs, overlap, system, strategy)
                   for s in range(strategy.pp)]
    step        = pipeline_step_time(stage_times, strategy, model, system)
    return Report(
        step_time = step.step_time,
        mfu       = compute_mfu(model, strategy, system, step.step_time),
        memory    = mem,
        timeline  = build_chrome_trace(ir, stage_times, overlap),  # optional
    )
```

### 7.2 Sweep (`search/sweep.py`)

```python
def sweep(model, system, search_space: SearchSpace,
          constraints: Constraints) -> list[Report]:
    results = []
    for strategy in enumerate_strategies(search_space, system.world_size):
        try:
            r = estimate(model, system, strategy)
        except ValidationError:
            continue
        if r.memory.total > system.gpu.hbm_gb * 1e9 * constraints.hbm_margin:
            continue
        results.append(r)
    return pareto(results, objectives=["step_time", "memory.total"])
```

**Pruning heuristics** for billion-config spaces:
- Fix `TP ≤ gpus_per_node` (no cross-node TP).
- `CP` only if `seq_len ≥ 32k`.
- `EP` only if `num_experts > 1`.
- Skip `ZeRO-3` if memory pressure is already satisfied by ZeRO-2 + recompute.

### 7.3 Report formats

- **JSON**: full structured report, includes per-op and per-stage breakdown.
- **CSV**: one row per sweep config, for pandas analysis.
- **Chrome trace** (`chrome://tracing`): timeline viewer for a single config — per-μbatch ops + collectives with overlap annotations.

---

## 8. Config File (yaml)

```yaml
model:
  name: deepseek-v3-like
  hidden: 7168
  ffn: 18432
  num_heads: 128
  num_kv_heads: 128
  head_dim: 128
  vocab: 128000
  seq_len: 4096
  layers: [dense, dense, dense] + [moe]*58 + [mtp]
  num_experts: 256
  moe_ffn: 2048
  top_k: 8
  capacity_factor: 1.0
  expert_imbalance: 0.10
  mtp_depth: 1
  param_dtype: bf16
  act_dtype: bf16

system:
  gpu:
    name: h100
    flops_bf16: 989
    flops_fp8:  1979
    hbm_gb: 80
    hbm_bw_gbps: 3350
  nodes: 32
  gpus_per_node: 8
  host_mem_gb: 2048
  nets:
    - {scope: intra_node, bw_gbps: 900, latency_us: 1.0, topology: nvswitch}
    - {scope: inter_node, bw_gbps: 400, latency_us: 5.0, topology: fattree}

strategy:
  tp: 8
  cp: 1
  pp: 8
  ep: 32
  dp: 4
  micro_batch: 1
  global_batch: 2048
  pp_schedule: dualpipe
  vpp_chunks: 1
  cp_kind: none
  zero_stage: 1
  recompute:
    per_layer:
      moe: ["attn"]
      dense: []
  offload: {opt_state: false, grads: false, params: false}
  tp_overlap: coc
  ep_overlap: true
  dualbatch: true
  optimizer: adam
```

---

## 9. Testing Strategy

| Layer | Test type | What it verifies |
|---|---|---|
| `spec` | Property tests | Divisibility, enum coverage, param-count derivation |
| `ir` | Golden files | Op count and sharded shapes match hand-computed cases (Llama-2 7B at TP=2, DeepSeek-V3 at TP=8×EP=32) |
| `flops` | Closed-form checks | `6ND` rule for standard dense transformer, Chinchilla test |
| `memory` | Table cross-check | ZeRO-1/2/3 reductions match [ZeRO paper Table 1]; DeepSeek numbers match [memory analysis paper] |
| `comm` | α-β microbench match | AR on 8-GPU NVLink ring, A2A on 16-GPU IB |
| `overlap` | Regression | DualPipe dual-batch overlap reduces EP A2A exposure by ≥80% when `compute ≥ comm` |
| `compose` | End-to-end | 1F1B bubble ratio matches [Megatron-LM paper] §3.2 |
| `estimator` | Anchor configs | Reproduce published MFU for GPT-3 175B on Selene-ish config within 10% (Calculon claims 3.4% avg) |

Keep an `anchors/` directory with ~10 published training runs (paper + measured numbers). CI runs estimator against all on every PR.

---

## 10. Implementation Phasing

**Phase 1 — Spine (2 weeks)**
`spec`, `ir` with dense-block only, `flops`, `memory` (Adam + ZeRO-1), `comm` (TP+DP only), 1F1B, single-point `estimator`. Validate against GPT-3 175B anchor.

**Phase 2 — Modern parallelism (2 weeks)**
CP (Ulysses + Ring), PP interleaved (VPP), EP + MoE block, expert imbalance, recompute policies (full / block / fine-grained), ZeRO-2/3, Muon optimizer.

**Phase 3 — Overlap + advanced pipeline (2 weeks)**
CoC/MC2 TP overlap, Ring-CP overlap, DualPipe + DualPipeV, DP-in-bubble, MTP layer, offload.

**Phase 4 — Search + UI (2 weeks)**
Sweep engine with pruning, pareto frontier, CLI/Python API, Chrome trace export, optional web viewer.

Zero-bubble PP is deferred — low priority per the requirements.

---

## 11. Key References

- **Megatron-LM PP+VPP** — [Narayanan et al. 2021](https://arxiv.org/pdf/2104.04473)
- **Activation recompute** — [Korthikanti et al. 2022](https://arxiv.org/pdf/2205.05198)
- **ZeRO / FSDP** — [Rajbhandari et al. 2020](https://arxiv.org/pdf/1910.02054)
- **ZeRO-Offload** — [Ren et al. 2021](https://arxiv.org/pdf/2101.06840)
- **Ulysses CP** — [Jacobs et al. 2023](https://arxiv.org/pdf/2309.14509)
- **Ring Attention** — [Liu et al. 2023](https://arxiv.org/pdf/2310.01889)
- **DeepSeek-V3 (DualPipe, MTP, EP)** — [DeepSeek 2024](https://arxiv.org/pdf/2412.19437)
- **DualPipeV** — [Sea AI Lab](https://sail.sea.com/blog/articles/63)
- **AdamW** — [Loshchilov & Hutter 2017](https://arxiv.org/pdf/1711.05101)
- **Muon** — [苏剑林 — Muon优化器赏析](https://kexue.fm/archives/10592)
- **Memory analysis (DeepSeek)** — [Yang et al. 2025](https://arxiv.org/pdf/2502.07846)
- **Calculon** — [Isaev et al. SC'23](https://dl.acm.org/doi/10.1145/3581784.3607102)
- **MindSpeed CoC / MC2** — [CoC doc](https://gitee.com/ascend/MindSpeed-LLM/blob/2.1.0/docs/pytorch/features/communication-over-computation.md), [MC2 doc](https://gitee.com/ascend/MindSpeed-LLM/blob/2.1.0/docs/pytorch/features/mc2.md)
