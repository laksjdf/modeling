# ZRT-Sim 完整架构文档

> 生成日期：2026-04-26

---

## 一、项目概览

**ZRT-Sim** 是一个 LLM 性能建模与仿真系统，四阶段流水线：

```
Graph Capture → Transform Pipeline → DAGScheduler → Report Generation
                                          ↑
                              MemoryModel (feasibility)
                              SimulatorHub (latency)
```

支持三条主路径：
- **推理图路径**：捕获 HF 模型的 aten op 序列 → 施加并行变换 → 调度 → 输出 Excel/HTML 报告
- **训练规格路径**（spec-based）：读取 YAML 配置 → 构造 training IR → 组合调度模型 → 输出性能/内存估算
- **内置模型库路径**（builtin）：一次捕获 OpGraph 持久化 → 按需加载 + retemplate → 聚合为 training IR → 估算（MoE/MTP 结构更精确）

---

## 二、入口点

### 2.1 主 CLI：`python -m python.zrt`

**入口文件**：`python/zrt/__main__.py`（转发到 `cli.py`）

```
main() [cli.py:33]
  ├─ argparse 参数解析 [cli.py:34-165]
  │
  ├─ --estimate-config → _run_estimate() [cli.py:168]        ← 训练规格路径
  │
  ├─ --capture-builtin → _run_capture_builtin() [cli.py]     ← 内置库捕获路径
  │   └─ 遍历 phases (prefill, train_forward)
  │       ├─ run_trace_phases() [graph/main.py]
  │       └─ builtin_registry.save(model_id, phase, graph, meta)
  │
  └─ 模型 ID → run_trace_phases() [graph/main.py]            ← 图捕获路径
       ├─ 推理 → _run_inference_pipeline() [cli.py:227]
       └─ 训练 → _run_training_modelling() [cli.py:297]
```

**关键 CLI 参数**（`cli.py:36-152`）：

| 参数 | 说明 |
|------|------|
| `model_id` | HF Hub 模型 ID 或本地路径 |
| `--layers N` | 追踪的层数（默认 4） |
| `--batch-size`, `--seq-len` | 虚拟输入配置 |
| `--phases` | prefill / decode / train_forward / train_backward |
| `--hw` | 硬件规格名称（e.g. `nvidia_h100_sxm`） |
| `--tp/pp/dp/ep/cp` | 各维度并行度 |
| `--train` | 启用训练图捕获 |
| `--estimate-config YAML` | 跳过图捕获，直接读 YAML 做规格估算 |
| `--capture-builtin MODEL_ID` | 捕获并持久化到内置模型库 |

**集中 CLI 运行方式汇总**：

```bash
# 1. 推理图追踪 → Excel/HTML 报告
python -m python.zrt Qwen/Qwen2.5-7B-Instruct --layers 4
python -m python.zrt deepseek-ai/DeepSeek-V3 --layers 4 --hw nvidia_h100_sxm --tp 8

# 2. 训练图追踪（捕获 train_forward + train_backward 并估算性能）
python -m python.zrt deepseek-ai/DeepSeek-V3 --layers 4 --train --hw nvidia_h100_sxm --tp 8 --pp 2

# 3. 规格路径估算（直接从 YAML 构造 training IR，无需加载模型）
PYTHONPATH=python python -m zrt.training estimate --config python/zrt/training/configs/llama3_70b_3d.yaml
# 可选输出 Chrome Trace
PYTHONPATH=python python -m zrt.training estimate --config python/zrt/training/configs/llama3_70b_3d.yaml --trace out.json

# 4. 内置模型库：捕获（一次）
python -m python.zrt hf_models/deepseek_v3 --layers 4 --seq-len 4096 --capture-builtin deepseek_v3
# 内置库存于 python/zrt/training/builtins/models/<model_id>.<phase>.json

# 5. 内置模型库：估算（加载预捕获图，在 YAML 中设置 builtin_model_id）
PYTHONPATH=python python -m zrt.training estimate --config python/zrt/training/configs/deepseek_v3_h100_8n_builtin.yaml
```

### 2.2 训练估计 CLI：`python -m zrt.training estimate`

**入口文件**：`python/zrt/training/__main__.py`

```
estimate()
  ├─ load_specs(config_path) [training/io/config_loader.py:20]
  │   └─ 返回 (ModelSpec, SystemSpec, Strategy)
  │
  ├─ estimate(model, system, strategy) [training/search/estimator.py:34]
  │
  └─ 输出：JSON / Excel / Chrome Trace
```

---

## 三、核心 IR 类型

### 3.1 推理侧 IR（`python/zrt/ir/`）

#### OpGraph（`ir/graph.py`）
```python
class OpGraph:
    name: str                          # e.g. "DeepSeek-V3_prefill"
    phase: str                         # "prefill"|"decode"|"train_forward"|"train_backward"
    nodes: dict[str, OpNode]
    edges: list[Edge]
    metadata: dict[str, Any]

    # 主要方法
    topo_sort() → list[OpNode]         # Kahn 拓扑排序
    clone() → OpGraph                  # 深拷贝（变换前必须调用）
    replace_subgraph(old_ids, new_node) # 融合核心操作
```

#### OpNode（`ir/node.py`）
```python
@dataclass
class OpNode:
    id: str                        # "op_0", "fused_3"
    op_type: str                   # "aten.mm.default", "comm.all_reduce"
    inputs: list[TensorMeta]
    outputs: list[TensorMeta]
    attrs: dict[str, Any]
    scope: str                     # "model.layers.0.self_attn"
    category: str                  # "compute" | "communication" | "memory"
    annotations: dict[str, Any]    # flops, latency_us, stream_id 等（由 pass 写入）

    # 融合元数据（仅融合节点）
    fused_from: list[str]
    num_sub_ops: int
    fusion_level: str              # "leaf" | "parent"

    # 来源信息
    src_file: str
    src_line: int
```

#### TensorMeta（`ir/types.py:145`）
```python
@dataclass(frozen=True)
class TensorMeta:
    id: str                        # "t42"
    shape: tuple[int, ...]
    dtype: DType                   # BF16, FP32, FP8_E4M3, ...
    mem_bytes: int                 # product(shape) * dtype.itemsize
    shape_template: tuple[int | str, ...] | None = None
    # 形状模板：静态维度为 int，动态维度为 "S"/"Q"/"B"/"BQ"/"BS"
    # 由 RecordingDispatch 捕获时写入，retemplate() 用于按新输入重算 shape
```

#### retemplate（`ir/retemplate.py`）
```python
def retemplate(graph: OpGraph, batch_size: int, seq_len: int,
               query_len: int | None = None) -> OpGraph:
    """按新 (batch_size, seq_len) 更新所有带 shape_template 的张量维度。"""
    # 标签绑定：B→batch_size, S→seq_len, Q→query_len, BS→B*S, BQ→B*Q
    # 对 graph 深拷贝后遍历所有 TensorMeta，按 template 重算 shape
```

#### DType 枚举（`ir/types.py:14`）
- FP32, FP16, BF16, FP8_E4M3, FP8_E5M2, INT8, INT4, INT32, INT64, UINT8, BOOL, UNKNOWN
- 每种有 `.itemsize` 和 `.bits` 属性

#### Edge（`ir/edge.py`）
```python
class Edge:
    src: str; src_idx: int
    dst: str; dst_idx: int
    tensor: TensorMeta | None
    tensor_id: str
```

---

### 3.2 训练侧 IR（`python/zrt/training/ir/`）

#### Graph / Op / Tensor / Collective（`training/ir/graph.py`）
```python
@dataclass
class Tensor:
    name: str
    shape_logical: tuple[int, ...]    # 分片前
    shape_local: tuple[int, ...]      # 分片后（单卡）
    dtype: Dtype
    is_activation: bool
    is_param: bool = False

@dataclass
class Op:
    name: str                         # "L5.qkv_proj"
    kind: str                         # "matmul"|"attn_core"|"softmax"|"ln"|...
    inputs: list[Tensor]
    outputs: list[Tensor]
    meta: dict[str, Any]              # {"m", "n", "k"} 等
    layer_id: int
    layer_kind: LayerKind             # DENSE | MOE | MTP

@dataclass
class Collective:
    name: str
    kind: str                         # "AG"|"RS"|"AR"|"A2A"|"P2P"
    group: str                        # "TP"|"CP"|"EP"|"DP"|"PP"
    bytes_: int                       # 单卡负载
    inserted_after: str

@dataclass
class Graph:
    ops: list[Op]
    collectives: list[Collective]
    layer_index: dict[int, tuple[int, int]]    # layer_id → (start_op_idx, end_op_idx)
```

---

### 3.3 规格类型（`python/zrt/training/spec/`）

#### ModelSpec（`training/spec/model.py:16`）
```python
@dataclass
class ModelSpec:
    hidden: int; ffn: int
    num_heads: int; num_kv_heads: int; head_dim: int
    vocab: int; seq_len: int
    layers: list[LayerKind]           # [DENSE, DENSE, MOE, ...] — 顺序决定 PP 分配
    num_experts: int = 0
    moe_ffn: int = 0; top_k: int = 0
    capacity_factor: float = 1.0
    expert_imbalance: float = 0.0
    param_dtype: Dtype = BF16
    attn_compression_ratio: float = 1.0   # P2 Compressed Attention
```

#### Strategy（`training/spec/strategy.py`）
```python
@dataclass
class Strategy:
    tp: int; cp: int; pp: int; ep: int; dp: int
    micro_batch: int; global_batch: int
    zero_stage: int                   # 0–3
    pp_schedule: PPSched              # ONE_F_ONE_B | INTERLEAVED | ZERO_BUBBLE | DUALPIPE | DUALPIPE_V
    recompute: RecomputePolicy        # full | selective(attn/ffn/ln) | none
    offload: OffloadPolicy
    dp_overlap_in_bubble: bool = True
    builtin_model_id: str | None = None
    # 设置后 estimator 走内置库路径；None 时走传统 build_graph() 路径
```

#### PPSched 枚举（`training/spec/strategy.py`）
- `ONE_F_ONE_B`：标准 1F1B
- `INTERLEAVED`：VPP（Virtual Pipeline Parallelism）
- `ZERO_BUBBLE`：ZB-1p 调度
- `DUALPIPE`：DualPipe
- `DUALPIPE_V`：DualPipe-V

---

## 四、四阶段流水线详解

### Stage 1：Graph Capture（`python/zrt/graph/`）

**完整调用链**：

```
run_trace_phases() [graph/main.py]
  │
  ├─ load_model() [graph/model_loader.py]
  │   └─ FakeTensorMode + HF AutoModel → 无权重加载
  │
  ├─ infer_layer_types() [graph/main.py:130]
  │   └─ 识别 Dense vs MoE 层
  │
  └─ _trace_phase() [graph/main.py:289]  ← 每个 phase 执行一次
      ├─ 准备虚拟输入张量
      ├─ RecordingDispatch.__torch_dispatch__() [graph/dispatch.py]
      │   └─ 拦截每个 aten op → 构建 record dict
      │      {aten_op, node_id, input_ids/shapes/dtypes,
      │       output_ids/shapes/dtypes, module_path, layer, src_file, src_line,
      │       input_shape_tags, output_shape_tags}  ← 形状模板标签（B/S/Q/BQ/BS/静态）
      │
      ├─ 前向 pass（train 模式时追加 loss.backward()）
      │
      ├─ build_op_graph() [graph/graph_builder.py]
      │   └─ records_to_opgraph() [ir/adapter.py:96]
      │       ├─ Pass 1：records → OpNode（含 TensorMeta）
      │       └─ Pass 2：output_id → consume 关系 → Edge
      │
      └─ build_fused_op_graph()
          └─ 两阶段融合（按 leaf module 分组，≤30 子 op 合并到 parent）
```

**模型特殊处理**（`graph/patches.py`）：
- MoE meta patch：替换 `.cpu().numpy()` 避免 FakeTensor 崩溃
- DeepSeek V3.2 Indexer patch

**版本兼容**（`graph/compat.py`）：transformers 4.x vs 5.x API 差异

---

### Stage 2：Transform Pipeline（`python/zrt/transform/`）

**Pass 顺序**（`transform/pipeline.py:53-127`）：

```
1. SPLIT — 并行分割
   ├─ DataParallelPass         (dp > 1 且 training)
   ├─ TensorParallelPass       (tp > 1)  → shard hidden dim
   ├─ ExpertParallelPass       (ep > 1)  → shard expert choice
   ├─ ContextParallelPass      (cp > 1)  → shard seq dim
   ├─ CommInserterPass         (tp|ep|cp > 1) → 插入 AG/RS/AR/A2A
   └─ PipelineParallelPass     (pp > 1)  → 按 layer 切 stage + P2P Send/Recv

2. FUSE — 算子融合
   └─ FusionPass               → 按软件栈规则（MindIE/vLLM）合并相邻 op

3. OPTIM — 优化
   ├─ QuantizationPass
   ├─ EPLBPass, SharedExpertPass, MTPPass
   └─ ZeroFSDPPass             (training)

4. ANALYZE — 性能注释（写入 node.annotations）
   ├─ FlopsPass                → flops, read_bytes, write_bytes
   ├─ RooflinePass             → compute_us, memory_us, latency_us, bound
   ├─ CommLatencyPass          → comm_latency_us
   ├─ StreamAssignPass         → stream_id, stream_type
   ├─ TrainFlopsPass           (training) → flops_fwd, flops_dx, flops_dw
   ├─ TrainingFlopsPass        (training)
   ├─ TrainingMemoryPass       (training)
   └─ TrainingPipelinePass     (training) → step_time_ms, mfu, hfu
```

**核心设计**：每个 pass 调用 `graph.clone()` 后再变异，确保函数式语义。

---

### Stage 3：DAGScheduler（`python/zrt/executor/`）

**贪心 list-scheduling 算法**（`executor/scheduler.py:134`）：

```python
for node in graph.topo_sort():
    stream_id = node.annotations["stream_id"]
    lat = node.annotations["latency_us"]

    pred_done = max(finish[p] for p in predecessors(node))
    stream_free = stream_avail[stream_id]

    start = max(pred_done, stream_free)
    end = start + lat

    scheduled.append(ScheduledOp(...))
    finish[node.id] = end
    stream_avail[stream_id] = end

return Timeline(scheduled_ops)
```

**输出数据结构**（`executor/scheduler.py:28`）：

```python
@dataclass
class ScheduledOp:
    node_id: str
    stream_id: int
    stream_type: str       # "compute" | "comm"
    start_us: float
    end_us: float
    latency_us: float

@dataclass
class Timeline:
    scheduled_ops: list[ScheduledOp]

    # 派生属性
    total_latency_us: float       # max(end_us)
    compute_time_us: float        # 计算流总和
    comm_time_us: float           # 通信流总和
    overlap_us: float             # compute + comm - total
```

**Compute-Comm 重叠分析**：`executor/overlap.py`

---

### Stage 4：Report Generation（`python/zrt/report/`）

**推理报告**（`report/summary.py:56`）：

```python
@dataclass
class E2ESummary:
    model: str; hardware: str; phase: str
    latency_ms: float; tokens_per_sec: float
    ttft_ms: float | None; tpot_ms: float | None
    compute_ms: float; comm_ms: float
    exposed_comm_ms: float; overlap_ratio: float
    mfu: float; hbm_bandwidth_util: float
    by_component: dict[str, float]   # component → % latency
    by_layer: list[float]
    top_bottleneck_ops: list[tuple[str, float]]
```

**输出格式**：
- Excel（6 sheet 原始图，5 sheet + JSON 变换后）
- HTML 可视化页面
- Chrome Trace JSON（`report/chrome_trace.py`）

---

## 五、训练建模子系统详解

### 5.1 模块结构（`python/zrt/training/`）

```
training/
├─ spec/          ModelSpec, Strategy, SystemSpec, Dtype, PPSched 等枚举
├─ ir/            Graph, Op, Tensor, Collective + builders + validate
│   └─ from_opgraph.py   ← NEW: OpGraph → training.ir.Graph 聚合 Pass
├─ models/        flops.py, memory.py, comm.py — 基础公式层
├─ compose/       PipelineComposer 及四种具体实现
├─ search/        SearchSpace + SearchEstimator → Pareto 报告
├─ anchor/        AnchorValidator — MFU 回归锚点测试
├─ trace/         ChromeTraceExporter → chrome://tracing JSON
├─ io/            config_loader.py (YAML → Spec), perf_tables.py
└─ builtins/      ← NEW: 内置模型库
    ├─ registry.py    builtin_registry.load/save/list_models/list_phases
    └─ models/        <model_id>.<phase>.json + <model_id>.meta.yaml
        ├─ deepseek_v3.prefill.json
        ├─ deepseek_v3.train_forward.json
        ├─ deepseek_v3.meta.yaml
        ├─ llama3_70b.prefill.json
        ├─ llama3_70b.train_forward.json
        ├─ llama3_70b.meta.yaml
        ├─ qwen2_7b.prefill.json
        ├─ qwen2_7b.train_forward.json
        └─ qwen2_7b.meta.yaml
```

#### 内置库路径数据流（`training/ir/from_opgraph.py`）

```
builtin_registry.load(model_id, phase="train_forward")
  → OpGraph (aten 粒度，376 nodes for DeepSeek-V3)
      ↓
retemplate(op_graph, batch_size=micro_batch, seq_len=model.seq_len)
  → 按 shape_template 重算所有动态维度
      ↓
aggregate_to_training_ir(op_graph, model)
  → training.ir.Graph (语义粒度，12 ops/层)
  按 OpNode.scope 关键字分桶：
    "q_proj/k_proj/v_proj" → matmul (layer{i}.qkv)
    "o_proj"               → matmul (layer{i}.o_proj)
    "gate_proj/up_proj"    → matmul (layer{i}.gate/up_proj)
    "down_proj"            → matmul (layer{i}.down_proj)
    "input_layernorm"      → ln (layer{i}.ln1)
    "post_attention_layernorm" → ln (layer{i}.ln2)
    "MoEGate"              → router (layer{i}.router)
    "embed_tokens"         → embed
    "lm_head"              → lm_head
      ↓
insert_collectives(graph, ShardPlan(strategy), model)
  → 插入 TP/EP/PP collective ops
```

---

### 5.2 FLOPs 计算（`training/models/flops.py`）

#### 矩阵乘法（`flops.py:41`）
```
fwd  = 2 * m * n * k
dx   = 2 * m * n * k    ← dX 梯度
dw   = 2 * m * n * k    ← dW 梯度
总计 = 6mnk（标准 6P 规则）
```

#### 注意力（`flops.py:53`）
```
fwd = (2.0 if causal else 4.0) * b * s² * h * d * compression_ratio
bwd ≈ 2.5 × fwd
```

#### Recompute 额外开销（`flops.py` P4 新增）
```
recompute_overhead = recompute_fwd_flops(layer_kind, policy)
HFU = (model_flops + recompute_overhead) / (peak × step_time)
MFU = model_flops / (peak × step_time)   ← 不含 recompute
```
- `_op_recompute_categories()` 按 `layer_kind` 限定作用域（避免 dense/MoE 混合时错误计数）

---

### 5.3 内存模型（`training/models/memory.py`）

```
单卡内存 = W + G + O + A + C_buf
```

| 项 | 公式 | ZeRO |
|----|------|------|
| 权重 W | P × param_dtype.bytes | stage≥3 → ÷ dp |
| 梯度 G | P × grad_dtype.bytes | stage≥2 → ÷ dp |
| 优化器 O | Adam: 3P × master_dtype.bytes | stage≥1 → ÷ dp |
| 激活 A | Korthikanti 公式 × pp_inflight_depth ÷ cp | recompute → ÷(1+pct) |
| 通信缓冲 C_buf | f(TP, EP, CP 集合通信缓冲) | — |

**激活内存 Korthikanti 公式**：
```
per_layer ≈ seq × hidden × act_bytes × coeff(layer_kind)
total = sum(per_layer) × pp_inflight_depth / cp
```

---

### 5.4 通信延迟（`training/models/comm.py`）

- AllReduce / AllGather / ReduceScatter / A2A / P2P
- 两层带宽层级：HCCS（节点内，NVLink 类）/ RoCE（跨节点）
- `tier_for_group(group, group_size, system)` 选取带宽层级

---

### 5.5 Pipeline Composer（`training/compose/`）

**基类**（`compose/pipeline.py`）：
```python
class PipelineComposer(ABC):
    def compose(stage_times, M, pp, dp_ar_time, strategy) → StepResult
```

**四种实现**：

| Composer | 调度策略 | Bubble 公式 |
|----------|---------|------------|
| `OneF1BComposer` | 标准 1F1B | `(pp-1)(t_fwd+t_bwd) / step_time` |
| `InterleavedComposer` | VPP（虚拟流水） | `(pp-1)(1-1/vpp_chunks)×...` |
| `DualPipeComposer` | DualPipe | 双向流水 |
| `DualPipeVComposer` | DualPipe-V | DualPipe 变种 |
| `ZeroBubbleComposer` | ZB-1p | ~0 bubble |

**标准 1F1B 时间模型**（`compose/pipeline.py:50`）：
```
warmup  = (pp-1) × t_fwd_max
steady  = M × max(t_fwd[s] + t_bwd[s])
cooldown = (pp-1) × t_bwd_max
step    = warmup + steady + cooldown + dp_ar_exposed

DP AllReduce 可隐藏在 bubble 中：
  hidden = min(bubble, dp_ar_time)
  dp_ar_exposed = dp_ar_time - hidden
```

**StepResult**（最终输出）：
```python
@dataclass
class StepResult:
    step_time: float          # ms
    bubble_fraction: float
    mfu: float
    hfu: float                # P4 新增
    schedule_name: str
    memory: MemBreakdown
```

---

### 5.6 搜索与 Pareto（`training/search/`）

```
SearchSpace.grid()
  → 遍历 (tp, cp, pp, ep, dp, zero_stage, pp_schedule, vpp_chunks) 组合
  → 过滤不合法组合（内存超限、集合通信拓扑约束）

SearchEstimator.estimate_all()
  → 对每个合法点调用 estimate(model, system, strategy)
  → 返回 SearchReport（含 Pareto 前沿：吞吐 vs 内存）
```

---

### 5.7 Anchor 验证（`training/anchor/`）

- YAML fixtures 位于 `tests/training/anchors/*.yaml`（GPT-3 175B、LLaMA-3 70B、DeepSeek-V3）
- 每个 anchor 固定 expected_mfu + expected_step_time + tolerance
- `AnchorValidator` 在 `pytest tests/training/anchors/test_anchors.py` 中运行
- GPT-3 175B 启用 strict gate；其余使用 calibration-mode

---

## 六、模块依赖关系图

```
cli.py
 ├─ [图捕获路径]
 │   graph/main.py
 │     ├─ graph/model_loader.py     (FakeTensorMode + HF)
 │     ├─ graph/dispatch.py         (TorchDispatchMode)
 │     │    └─ graph/tracker.py     (ModuleTracker)
 │     ├─ graph/graph_builder.py
 │     │    └─ ir/adapter.py        (records → OpGraph)
 │     │         ├─ ir/node.py
 │     │         ├─ ir/edge.py
 │     │         └─ ir/types.py
 │     ├─ graph/patches.py
 │     └─ graph/compat.py
 │
 │   hardware/registry.py → hardware/spec.py (HardwareSpec, YAML configs)
 │
 │   transform/pipeline.py (build_default_pipeline)
 │     ├─ transform/parallel/tensor_parallel.py
 │     ├─ transform/parallel/expert_parallel.py
 │     ├─ transform/parallel/pipeline_parallel.py
 │     ├─ transform/parallel/data_parallel.py
 │     ├─ transform/parallel/context_parallel.py
 │     ├─ transform/parallel/comm_inserter.py
 │     ├─ transform/fusion/
 │     ├─ transform/optim/
 │     └─ transform/analysis/
 │          ├─ passes.py            (Flops, Roofline, StreamAssign)
 │          ├─ training.py          (TrainingFlops, TrainingMemory, TrainingPipeline)
 │          └─ modeller.py          (TrainingReport, estimate_training)
 │
 │   executor/scheduler.py (DAGScheduler → Timeline)
 │     └─ executor/overlap.py
 │
 │   simulator/hub.py (SimulatorHub, Roofline→Regression→ProfileDB)
 │   report/summary.py, report/chrome_trace.py
 │
 └─ [训练规格路径 / 内置库路径]
     training/io/config_loader.py
       └─ training/spec/           (ModelSpec, SystemSpec, Strategy[builtin_model_id])
     training/search/estimator.py
       ├─ [spec 路径] training/ir/builders.py → training/ir/graph.py
       └─ [builtin 路径] training/builtins/registry.py
            ├─ ir/retemplate.py                (shape rebind)
            └─ training/ir/from_opgraph.py     (OpGraph → training.ir.Graph)
                 └─ training/ir/shard.py       (insert_collectives)
     training/models/flops.py
     training/models/memory.py
     training/models/comm.py
     training/compose/pipeline.py
       └─ training/compose/stage.py
     training/search/report.py
     training/anchor/
     training/trace/chrome_trace.py
```

---

## 七、关键设计模式

### 7.1 函数式变换（Immutable IR）
每个 Transform pass 调用 `graph.clone()` 后再修改，避免副作用，便于调试回溯。

### 7.2 上下文注入（TransformContext）
`TransformContext` 贯穿整个 pipeline，包含 `hw_spec`、`parallel`、`training` 配置。Pass 通过条件函数 `cond(ctx)` 选择性启用。

### 7.3 三层抽象
```
dispatch records（底层原始数据）
  ↓
OpGraph（中层通用 IR）
  ↓
annotations dict（顶层性能数据，由 pass 写入）
```

### 7.4 硬件注册表（YAML-driven）
`hw_registry.load("nvidia_h100_sxm")` 从 `hardware/configs/*.yaml` 加载 `HardwareSpec`，变换逻辑中不硬编码硬件参数。

### 7.5 Composer 策略模式
`PipelineComposer` 基类 + 四种具体调度实现，`PPSched` 枚举驱动分派（`training.py:285-298`）。

---

## 八、测试覆盖结构

```
tests/
├─ test_simulator.py              # RooflineSimulator, SimulatorHub
├─ test_transform.py              # Transform passes（含 TP shape 验证）
├─ test_fusion_pass.py            # FusionPass
├─ test_executor.py               # DAGScheduler + overlap
├─ test_memory.py                 # 内存估算
├─ test_report_summary.py         # 报告生成
├─ test_train_trace.py            # 训练图追踪
├─ graph/
│  └─ test_shape_tags.py          # tag_dims / tags_str / shape_template 集成
├─ ir/
│  └─ test_retemplate.py          # retemplate 形状重绑定
└─ training/
   ├─ test_flops.py               # FLOPs + HFU 回归（202 passed）
   ├─ test_1f1b.py                # 1F1B 调度
   ├─ test_dualpipe.py            # DualPipe 调度
   ├─ test_graph_schedule.py      # VPP/DualPipe PP 调度分派
   ├─ test_transform_integration.py
   ├─ test_captured_graph_modelling.py
   ├─ test_ir_dense.py
   ├─ test_comm.py
   ├─ test_builtin_path.py        # 内置库全流程（registry/retemplate/aggregate/estimate）
   └─ anchors/
       └─ test_anchors.py         # MFU anchor 回归（GPT-3 strict gate）
```

---

## 九、硬件配置

**配置文件**：`python/zrt/hardware/configs/*.yaml`

支持硬件：H100 SXM、A100、H800、Ascend 910B/C

每份 YAML 定义：
- 计算：`flops_bf16`（TFLOPS）、`flops_fp8`
- 内存：`hbm_bw_gbps`、`hbm_capacity_gb`
- 互联：HCCS（节点内）带宽、RoCE（跨节点）带宽

---

## 十、已知限制

- FakeTensor 限制：部分 backward() 会失败，回退到 6P 规则估算
- FakeTensor decode 阶段 KV cache broadcast 失败：内置库仅捕获 `prefill` + `train_forward`，不含 decode
- EP 不平衡因子使用 balls-into-bins 近似
- Recompute overhead 估计基于简化假设（未考虑精确的 fused kernel timing）
- 跨节点 TP 搜索未做 NVLink 拓扑感知剪枝
- MoE all_to_all dispatch/combine：HuggingFace 前向不含分布式 A2A 通信 op，内置库中 MoE 层只有 router，无 dispatch/combine（分布式框架才有）
- 内置库聚合规则基于 scope 字符串匹配，不同 HF 实现（如 Mixtral vs DeepSeek MoE）需按需扩展 `_SCOPE_RULES`
