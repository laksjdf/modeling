# 训练建模器 —— 双路径架构与实施路线

_2026-04-28。合并自 `training_modeller_zh.md`（2026-04-23 架构审查）与 `training_modeller_zh_v2.md`（2026-04-28 统一方案）。_

---

## 双路径现状

系统当前存在两条并行的训练性能估算路径，均收敛于同一组 `PipelineComposer` 类：

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║  Stack A：规格驱动路径（快速分析估算）                                            ║
║  入口：zrt.training.search.estimator.estimate()                                ║
║  特点：无需真实模型权重；速度快；适用于搜索/扫描/CI 锚点场景                          ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  YAML config (model + system + strategy)                                      ║
║      │ training/io/config_loader.py                                           ║
║      ▼                                                                        ║
║  ModelSpec + Strategy + SystemSpec                                            ║
║      │ strategy.validate() + ir_validate()                                    ║
║      ▼                                                                        ║
║  build_graph(model, strategy)           training/ir/builders.py               ║
║      embed + dense_block × layers + final_ln + lm_head                       ║
║      ShardPlan + insert_collectives → TP AG/RS 集合                           ║
║      MoE/MTP 使用专用 block；Ulysses-CP/EP collectives 已建模                 ║
║      → training.ir.Graph                                                      ║
║            ops:        list[Op]  (name, kind, inputs, outputs,                ║
║                                   meta, layer_id, layer_kind)                 ║
║            collectives: list[Collective]  (AG/RS/AR/A2A/P2P；TP/CP/EP 组)    ║
║            layer_index: dict[int, tuple[int, int]]                            ║
║      │                                                                        ║
║      ├── total_training_flops()          training/models/flops.py             ║
║      │     op_cost(op): matmul fwd=2mnk, dx=2.5×fwd, dw=2mnk                 ║
║      │                  attn fwd=2bs²hd × compression_ratio                  ║
║      │     sum(fwd+dx+dw) × M 微批数 → training_flops                         ║
║      │     recompute_overhead_flops() 按 per_layer_kind 策略累加               ║
║      │                                                                        ║
║      ├── memory_breakdown()              training/models/memory.py            ║
║      │     weights     = P × dtype_bytes / ZeRO_weight_shard                 ║
║      │     gradients   = P × dtype_bytes / ZeRO_grad_shard                   ║
║      │     opt_state   = P × (Adam:3× | Muon:2.1×) / ZeRO_optstate_shard    ║
║      │     activations = coeff(layer_kind) × hidden × seq × L / (tp × cp)    ║
║      │                   × max_inflight_microbatches                          ║
║      │     comm_buffers + offload                                             ║
║      │                                                                        ║
║      ├── collective_time()               training/models/comm.py              ║
║      │     α-β 模型：AG/RS = (N-1)·(α + S/N·β)；AR = 2·AG；A2A = (N-1)/N·...║
║      │     tier_for_group：group_size ≤ gpus_per_node → intra (HCCS)         ║
║      │                     group_size > gpus_per_node → inter (RoCE)         ║
║      │                                                                        ║
║      └── pipeline_step_time()            training/compose/pipeline.py         ║
║              stage_time(op, system, strategy):                                ║
║                compute_us = flops / (peak_tflops × achieved_flops_eff)       ║
║                memory_us  = bytes / (hbm_bw × achieved_bw_eff)               ║
║                + recompute_time + collective_time/2 + ep_imbalance_factor    ║
║              按 PP 分 stage，选 COMPOSER_BY_SCHED[pp_schedule]:               ║
║                1F1B:     step=(pp-1)·t_fwd+M·t_max+(pp-1)·t_bwd+dp_exposed  ║
║                VPP:      bubble=(pp-1)/(vpp×M)                               ║
║                DualPipe: bubble≈(pp-1)/2 · t_stage_max                       ║
║                ZeroBubble: bubble=(pp-1)·max(t_stage-t_w, 0)                ║
║              memory_breakdown / compute_mfu / compute_hfu(recompute)         ║
║                         │                                                     ║
║                         ▼                                                     ║
║               StepResult → TrainingReport                                     ║
║                 step_time_ms  mfu  hfu  bubble_fraction                       ║
║                 memory_breakdown  per_stage_ms  warnings                      ║
║                 (可选) grid_search → Pareto 前沿 (step_time, peak_hbm)        ║
╚═══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════════╗
║  Stack B：图捕获路径（主路径）                                                    ║
║  入口：estimate_training_from_graphs()  transform/analysis/modeller.py        ║
║  特点：真实算子序列；精确张量形状；精确内存生命周期；精确 overlap 建模              ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  HuggingFace 模型 + 硬件 YAML + 训练策略                                        ║
║      │                                                                        ║
║      ▼ load_model (graph/model_loader.py)                                     ║
║        FakeTensorMode + AutoModelForCausalLM.from_config                     ║
║        apply_compat_patches + patch_moe_for_fake + patch_indexer_for_fake    ║
║        失败时 fallback 到 hf_models/<model> 本地目录                            ║
║      │                                                                        ║
║      ▼ run_trace_phases("train_forward", "train_backward")  graph/main.py    ║
║        共享 TensorTracker（fwd/bwd tensor_id 全局唯一，是 stitch 的前提）        ║
║        train_backward：fwd 阶段 active=False（仅分配 id），                     ║
║                        bwd 阶段 active=True 后调 logits.sum().backward()      ║
║        RecordingDispatch (TorchDispatchMode) + ModuleTracker (hooks)         ║
║        records 字段：aten_op, op_short, module_path, layer, component,       ║
║                      input/output shapes+dtypes, _input_ids, _output_ids,    ║
║                      recompute (activation checkpointing 重新前向标记)         ║
║        FusionEngine 三 Pass 融合：                                             ║
║          Pass 1 (leaf):   连续相同 module_path+layer 聚组                     ║
║          Pass 2 (parent): 相邻 leaf 组合并至父 scope（≤30 算子，≤max_children）║
║          Pass 3 (label):  平台子模式 → SEMANTIC_LABELS → module_class 兜底    ║
║        records_to_opgraph / fused_records_to_opgraph                         ║
║        → OpGraph[fwd]  +  OpGraph[bwd]（各自独立，无跨图边）                   ║
║      │                                                                        ║
║      ▼ stitch_fwd_bwd(fwd_graph, bwd_graph)   ir/adapter.py:613–749          ║
║        bwd 节点 ID 加 "bwd_" 前缀；annotations["phase"] = "fwd"/"bwd"        ║
║        参数节点：is_param=True（scope 路径模式判断）                             ║
║        跨图边匹配：                                                             ║
║          ① 精确 tensor_id 匹配（O(1) 查找）                                   ║
║          ② 形状+dtype+同 layer/scope 启发式（_best_cross_match）               ║
║        → 统一 OpGraph  (metadata["fwd_bwd_stitched"] = True)                 ║
║      │                                                                        ║
║      ▼ TransformContext(hw_spec, ParallelConfig, TrainingConfig)              ║
║      │                                                                        ║
║      ▼ TransformPipeline.run(graph, ctx)    transform/pipeline.py             ║
║        ── SPLIT ──────────────────────────────────────────────────────────    ║
║        DataParallelPass    [dp>1]   bwd 梯度节点后插 AR/RS；dp_overlap 标注   ║
║        TensorParallelPass  [tp>1]   列/行并行切分；comm_after 注解             ║
║        ExpertParallelPass  [ep>1]   专家 FFN 分片；ep_needs_a2a 注解          ║
║        ContextParallelPass [cp>1]   Ulysses A2A / Ring send_recv 插入        ║
║        CommInserterPass             TP/EP/CP 通信集合接入图                   ║
║        PipelineParallelPass [pp>1]  stage_id 注解（按 compute_us 贪心分配）  ║
║                                     阶段边界插 comm.send_recv P2P 节点        ║
║        ── FUSE ───────────────────────────────────────────────────────────    ║
║        FusionPass          OpGraph 形态三 Pass 融合；保护 stage_id/phase 不变量║
║        ── OPTIM ──────────────────────────────────────────────────────────    ║
║        ZeroFSDPPass        metadata["zero"] = {stage, weight/grad/optstate_  ║
║                            shard}；ZeRO-3 时按层插 AG/RS                      ║
║        ── ANALYZE ────────────────────────────────────────────────────────    ║
║        FlopsPass           每节点 flops_fwd/dx/dw；attn 按 compression_ratio ║
║        RooflinePass        每节点 compute_us / memory_us / latency_us / bound║
║        CommLatencyPass     通信节点 α-β 公式；区分 intra/inter 层              ║
║        StreamAssignPass    stream_id / stream_type                            ║
║                            overlap_type: coc / mc2 / ring_cp / none          ║
║        TrainingFlopsPass   training_flops / forward_flops / backward_flops   ║
║                            recompute_flops = ½·fwd[recompute=True]           ║
║                            layer_scale 放大到完整模型层数                       ║
║                            6P 规则仅在 forward_flops==0 时作兜底                ║
║        TrainingMemoryPass  weights/grads/opt_state (ZeRO 缩放)               ║
║                            activations：优先 fwd→bwd 边活字节；               ║
║                                         退化到 Korthikanti 系数 × 在途深度    ║
║                                         recompute 注解 → 动态缩减系数          ║
║      │                                                                        ║
║      ▼ TrainingPipelinePass              transform/analysis/training.py       ║
║        PP>1 且节点有 stage_id：                                                ║
║          for s in range(pp):                                                  ║
║            subgraph    = graph.subgraph([n for n if stage_id==s])            ║
║            timeline[s] = DAGScheduler(hw).schedule(subgraph)                 ║
║            stage_fwd[s]    = timeline[s].phase_latency("fwd")                ║
║            stage_bwd[s]    = timeline[s].phase_latency("bwd")                ║
║            stage_bwd_dw[s] = stage_bwd[s] × (dW_flops / total_bwd_flops)    ║
║        否则：单图调度 + 按 pp 平均（fallback warning）                           ║
║        → StageTime 列表                                                       ║
║        → COMPOSER_BY_SCHED[pp_schedule]（共享五个 PipelineComposer）          ║
║        → overlap 修正：MC2 全部隐藏；CoC 隐藏 (k-1)/k；ring_cp 减 fa_tile    ║
║        → metadata["pipeline_metrics"]: step_time_ms, MFU, HFU, bubble        ║
║                         │                                                     ║
║                         ▼                                                     ║
║               StepResult → TrainingReport                                     ║
║                 step_time_ms      MFU           HFU        bubble_fraction   ║
║                 memory_breakdown  forward_flops  backward_flops               ║
║                 recompute_flops   per_stage_ms   total_params                 ║
║                 (可选) Chrome Trace JSON → chrome://tracing 可视化             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

                      ▲ 两条路径共享的组件 ▲
                      PipelineComposer 及五个具体实现
                      OneF1B / Interleaved(VPP) / DualPipe / DualPipeV / ZeroBubble
                      位于：python/zrt/training/compose/schedules.py
                      输入：list[StageTime]，strategy → StepResult
```

---

## 核心设计原则

**Stack B 是主路径。Stack A 是快速估算回退。**

- Stack B（图捕获）携带真实张量形状、真实算子序列、真实内存生命周期，是所有并行化建模的正确基础。
- Stack A（规格驱动）用于无需完整追踪时的快速分析：搜索空间扫描、初步可行性判断、CI 快速锚点校验。
- 两条路径**不应合并 IR**。Stack A 的 `Graph`（层级列表）和 Stack B 的 `OpGraph`（有向数据流图）服务于不同的抽象层次，强行合并会增加复杂度而无收益。
- **收敛点**：两条路径都通过 `PipelineComposer` 类生成 `StepResult`，并最终返回统一的 `TrainingReport`。

---

## 开发规划（模块拆解）

两条路径均需从头实现。收敛点是 `PipelineComposer → StepResult → TrainingReport`，所有模块的最终目标都是将数据送达这个汇合点。Stack A 和 Stack B 可并行开发，共享组件先行。

---

### 共享组件（优先实现）

#### 共-1：统一输出类型（`training/spec/report.py`）

两条路径必须返回同一类型，否则下游无法统一消费。

`TrainingReport` 字段：

```python
@dataclass
class TrainingReport:
    # 时间（ms）
    step_time_ms: float
    pipeline_time_ms: float
    compute_time_ms: float
    exposed_comm_ms: float
    optimizer_time_ms: float
    optimizer_comm_ms: float
    warmup_ms: float
    steady_ms: float
    cooldown_ms: float
    # 效率
    mfu: float
    hfu: float
    mfu_native: float       # 不含 bubble
    bubble_fraction: float
    schedule_name: str
    warmup_steps: int
    steady_steps: int
    cooldown_steps: int
    # FLOPs
    total_flops: float
    forward_flops: float
    backward_flops: float
    training_flops: float
    recompute_flops: float
    total_params: int
    # 内存
    memory: MemBreakdown
    # per-stage 明细
    per_stage: list[StageTime]
    # 通信分解（exposed / hidden / total，按策略）
    tp_exposed_ms: float;  tp_hidden_ms: float;  tp_total_ms: float
    cp_exposed_ms: float;  cp_total_ms: float
    ep_exposed_ms: float;  ep_hidden_ms: float;  ep_total_ms: float
    pp_exposed_ms: float;  pp_total_ms: float
    dp_exposed_ms: float;  dp_hidden_ms: float;  dp_total_ms: float
    hidden_comm_ms: float
    total_comm_volume_ms: float
```

不变量（在 `__post_init__` 断言）：
- `step_time_ms = pipeline_time_ms + optimizer_time_ms + optimizer_comm_ms`
- `pipeline_time_ms = compute_time_ms + exposed_comm_ms`
- `exposed_comm_ms = tp_exposed_ms + cp_exposed_ms + ep_exposed_ms + pp_exposed_ms + dp_exposed_ms`

Stack A 保留 `Report = TrainingReport` 别名以兼容旧调用方。

#### 共-2：PP Schedule Composers（`training/compose/schedules.py`）

两条路径共用同一批 Composer，输入 `list[StageTime]` + `Strategy`，输出 `StepResult`（`StepResult` 是 `TrainingReport` 的核心计算结果，可直接转换）。

需要实现五个 `PipelineComposer` 子类：

| Composer | 气泡公式 | 关键约束 |
|----------|---------|---------|
| `OneF1BComposer` | `(pp-1) × t_max / (M × t_max)` | 需要 `bwd_dx + bwd_dw` 分离 |
| `InterleavedComposer`（VPP） | `(pp-1) / (vpp_chunks × M)` | `vpp_chunks > 1` |
| `DualPipeComposer` | `≈ (pp-1)/2 × t_stage_max` | 双向流水 |
| `DualPipeVComposer` | DualPipe + VPP 叠加 | 需同时满足两者约束 |
| `ZeroBubbleComposer` | `(pp-1) × max(t_stage - t_w, 0)` | `t_w = bwd_dw` 时间，须从 `StageTime.bwd_dw` 读取 |

`StepResult` 不变量（在单元测试中断言）：
```
step_time     = pipeline_time + optimizer_time + optimizer_comm
pipeline_time = compute_time + exposed_comm
compute_time  = fwd_compute + bwd_compute + recompute_time
mfu           = actual_flops / (step_time × peak_flops)          # 不含重计算 flops
hfu           = (actual + recompute_flops) / (step_time × peak_flops)
```

`PP_SCHED_BY_NAME: dict[PPSched, type[PipelineComposer]]` 须完整注册所有五个实现。

#### 共-3：Stage 时延（`training/compose/stage.py`）

- `StageTime` dataclass，须将 bwd 拆成 `bwd_dx` 和 `bwd_dw`（ZeroBubble 调度依赖此分离）：
  ```
  fwd, bwd_dx, bwd_dw           # 计算时间
  comm_fwd, comm_bwd             # 通信时间（暴露部分）
  tp_hidden, ep_hidden           # 已被 overlap 隐藏的通信
  tp_exposed, ep_exposed, cp_exposed
  recompute                      # 重计算额外时间
  ```

- `stage_time(graph, model, system, strategy, stage_layer_ids) -> StageTime`（Stack A 使用）
- `op_to_time(flops, bytes_, system) -> float`：`max(flops / peak_tflops, bytes_ / hbm_bw) × efficiency`
- `op_to_time_hetero(cube_flops, vector_flops, bytes_, system)`：NVIDIA 需区分 Tensor Core（cube）和 CUDA Core（vector）；非 NVIDIA 回退到 `op_to_time`

`efficiency` 从 `perf_tables.py` 查表，不得硬编码。

---

### Stack A — 规格驱动路径

**目标**：基于 YAML 规格（无需真实权重），快速估算训练性能。入口：`zrt.training.search.estimator.estimate()`。

#### SA-1：训练 IR（`training/ir/`）

**`training_graph.py`** — 核心数据类型：

```python
Tensor(name, shape_logical, shape_local, dtype, is_activation, is_param)
Op(name, kind, inputs, outputs, meta, layer_id, layer_kind, component)
  # kind: matmul | attn_core | sparse_attn | router | combine | embed | lm_head
  #       swiglu | rmsnorm | rope | add | compressor_pool | indexer_topk | ...
Collective(name, kind, group, bytes_, inserted_after, inserted_before,
           rounds, overlap, phase)
  # kind: AG | RS | AR | A2A | P2P
Graph(ops: list[Op], collectives: list[Collective],
      layer_index: dict[int, tuple[int, int]])
  # layer_index 映射 layer_id → (start, end) 在 ops 中的切片
```

**`builders.py`** — 图构建：

- `build_graph(model: ModelSpec, strategy: Strategy) -> Graph`：按层类型构造 `embed + dense_block × N + final_ln + lm_head`；MoE 层使用专用 MoE block；MTP 层使用 MTP block
- `ShardPlan`：按 TP/CP/EP 维度计算各 Op 的 `shape_local`
- `insert_collectives()`：根据 ShardPlan 在 AG/RS/A2A 边界处插入 `Collective`

**`shard.py`** — 集合通信插入：Ulysses-CP 和 EP collectives 须在此建模；Ring-CP 暂为 stub（`CPKind.RING` 直接返回，见仍开放事项）。

**约束**：所有形状须在 build 时完全确定；`layer_kind` 枚举须与 `flops.py` 中的系数表匹配。

#### SA-2：FLOPs 建模（`training/models/flops.py`）

```python
OpCost(fwd_cube_flops, fwd_vector_flops,    # cube = Tensor Core / matrix engine
       dx_cube_flops,  dx_vector_flops,     # vector = CUDA Core / vector engine
       dw_cube_flops,  dw_vector_flops,
       fwd_bytes, dx_bytes, dw_bytes,
       bound: Literal["compute", "memory"])

op_cost(op: Op, model: ModelSpec, system: SystemSpec | None) -> OpCost
total_training_flops(graph, model, strategy, system) -> float
forward_backward_flops(graph, model, strategy, system) -> tuple[float, float]
recompute_overhead_flops(graph, model, strategy) -> float
```

关键公式：
- matmul fwd = 2mnk，dx = 2mnk（实际乘约 2.5 因子），dw = 2mnk
- attn fwd = 2·B·S²·H·D × `attn_compression_ratio`
- `attn_compression_ratio`：优先读 `op.meta["attn_compression_ratio"]`，fallback 到 `model.attn_compression_ratio`；须 validate > 0 且 ≤ 1

#### SA-3：通信建模（`training/models/comm.py`）

```python
collective_time(c: Collective, group_size: int, link: LinkSpec) -> float
collective_time_hierarchical(c, intra_group_size, inter_group_size,
                             intra_link, inter_link) -> float
```

α-β 公式：
- AG/RS：`(N-1) × (α + S/N × β)`
- AR：`2 × AG`
- A2A：`(N-1)/N × (α + S × β)`
- P2P：`α + S × β`

`tier_for_group(group_size, system)`：`group_size ≤ gpus_per_node` → intra（HCCS），否则 → inter（RoCE）；`α`/`β` 从 `HardwareSpec.interconnect` 读取，不硬编码。

#### SA-4：内存建模（`training/models/memory.py`）

```python
@dataclass
class MemBreakdown:
    weights: float; grads: float; opt_state: float
    activations: float; comm_buffers: float
    hc_overhead_bytes: float; muon_ns_buffer: float
    peak_forward: float; peak_backward: float
    peak_optimizer: float; peak_overall: float
    def to_gb(self) -> MemBreakdown: ...

memory_breakdown(graph, model, strategy, system) -> MemBreakdown
```

ZeRO 分片规则（`strategy.zero_stage`）：
- weights：`P × dtype_bytes / (dp if zero ≥ 3 else 1)`
- grads：`P × dtype_bytes / (dp if zero ≥ 2 else 1)`
- opt_state：`P × (3× Adam | 2.1× Muon) / (dp if zero ≥ 1 else 1)`
- activations：`coeff(layer_kind) × hidden × seq × L / (tp × cp) × max_inflight_microbatches`

Muon 特例：需计算 Newton-Schulz AllGather spike（`A = XᵀX` buffer）并建模为 `muon_ns_buffer`——此峰值出现在 optimizer step 期间，须单独追踪。

#### SA-5：估算入口（`training/search/estimator.py`）

```python
estimate(model: ModelSpec, system: SystemSpec, strategy: Strategy,
         graph: Graph | None = None) -> TrainingReport
```

调用链：`strategy.validate()` → `ir_validate()` → `build_graph()`（若 graph 为 None）→ `total_training_flops()` → `memory_breakdown()` → `pipeline_step_time()` → 组装 `TrainingReport`

`TrainingReport` 须从 `training/spec/report.py` 导入，不得在 estimator.py 内重复定义。

#### SA-6：配置加载（`training/io/config_loader.py`）

```python
load_specs(config_path) -> tuple[ModelSpec, SystemSpec, Strategy]
load_anchor_config(yaml_path) -> tuple[ModelSpec, SystemSpec, Strategy]
```

YAML 顶层结构：`model` / `system` / `strategy`；`model` 字段可以是字符串引用（指向 `training/configs/models/` 中的 YAML）或内联 dict。

Recompute 策略别名：`"attn"` → `"attn_block"`（向后兼容）。

---

### Stack B — 图捕获路径

**目标**：基于真实 HuggingFace 模型，捕获训练算子序列，经变换流水线建模并行性，输出 `TrainingReport`。入口：`estimate_training_from_graphs()`。

#### SB-1：图捕获（`graph/`）

使用 `TorchDispatchMode` + `FakeTensorMode`（无真实权重）捕获。两个 phase 须在同一 `FakeTensorMode` 上下文内运行：

- `"train_forward"`：`RecordingDispatch.active=True`，正常前向，记录所有 aten op
- `"train_backward"`：先静默前向（`active=False`，仅分配 tensor_id），再 `logits.sum().backward()`，`active=True` 记录反向 op

`TensorTracker` 须在两个 phase 间共享，确保 tensor_id 全局唯一（是 `stitch_fwd_bwd` 的前提）。

`FusionEngine` 三 Pass：
1. **Leaf**：连续相同 `module_path + layer` 聚组
2. **Parent**：相邻 leaf 组合并至父 scope（≤ 30 算子，≤ max_children）
3. **Label**：平台子模式 → `SEMANTIC_LABELS` → `module_class` 兜底

输出：`OpGraph[fwd]` + `OpGraph[bwd]`（各自独立，无跨图边）。

约束：`patches.py` 中的 MoE patch 和 Indexer patch 必须在捕获前应用；DeepSeek 类模型须 4 层（前 3 dense，第 4 MoE）；dense 模型（Llama/Qwen）2 层即可。

#### SB-2：前向/反向图缝合（`ir/adapter.py`）

```python
stitch_fwd_bwd(fwd_graph: OpGraph, bwd_graph: OpGraph,
               name: str | None = None) -> OpGraph
```

实现步骤：
1. 合并两图节点，反向节点 ID 加 `"bwd_"` 前缀
2. 标注：`node.annotations["phase"] = "fwd" | "bwd"`；参数节点标注 `is_param = True`（按 scope 路径模式判断）
3. 跨图边匹配：① 精确 tensor_id O(1) 查找；② 形状+dtype+同 layer/scope 启发式（`_best_cross_match`）回退
4. 写入 `metadata["fwd_bwd_stitched"] = True`

下游所有 pass 须能处理 stitched 和非 stitched 两种情况（通过 `metadata` 区分）。

#### SB-3：并行变换 Passes（`transform/parallel/`）

执行顺序（不可交换）：DP → TP → EP → CP → CommInserter → PP

| Pass | 文件 | 核心职责 |
|------|------|---------|
| `TensorParallelPass` | `tensor_parallel.py` | 列/行并行切分，标注 `comm_after`，修改 `local_shape` |
| `ExpertParallelPass` | `expert_parallel.py` | 专家 FFN 节点标注 `ep_needs_a2a`，调整分片系数 |
| `ContextParallelPass` | `context_parallel.py` | Ulysses：注意力前后插 `comm.all_to_all`；Ring：插 `cp` 轮 `comm.send_recv`，标 `overlap_target="fa_tile:<id>"` |
| `DataParallelPass` | `data_parallel.py` | 反向梯度节点后按 layer 插 `comm.all_reduce`（ZeRO-0）或 `comm.reduce_scatter`（ZeRO-2/3），可重叠通信标注 `overlap_in_bubble=True` |
| `CommInserterPass` | `comm_inserter.py` | 读取 `comm_after`/`ep_needs_a2a`/`cp_split` 注解，将集合通信节点实际插入图 |
| `PipelineParallelPass` | `pipeline_parallel.py` | 按 `compute_us`（→`latency_us`→`flops` 降级）贪心装箱分配 `stage_id`；VPP 额外分配 `virtual_stage_id`；跨 stage 边替换为 `comm.send_recv` P2P 节点（放在接收 stage） |

所有 pass 须实现 clone-before-mutate：输入图不可被修改。

#### SB-4：训练专用 Passes（`transform/training/`）

| Pass | 文件 | 核心职责 |
|------|------|---------|
| `RecomputePass` | `recompute.py` | 按 `ctx.training.recompute_policy` 标注 `node.annotations["recompute"] = True`；selective 模式仅标 softmax、attn output proj、flash_attn 等高激活算子；跳过 stitched 图的反向节点 |
| `OffloadPass` | `offload.py` | 按 `ctx.offload.pct` 标注可卸载到 CPU 的激活节点 |
| `OptimizerPass` | `optimizer.py` | 图末追加 optimizer step 节点；Muon 模式追加 Newton-Schulz 迭代节点和 AllGather 节点 |
| `ZeroFSDPPass` | `zero_fsdp.py` | 按 ZeRO stage 在各层插入权重 AG（ZeRO-3）和梯度 RS（ZeRO-2/3）；写入 `metadata["zero"]` |

#### SB-5：分析 Passes（`transform/analysis/`）

按执行顺序：

**`FlopsPass`**：为每个节点标注 `flops_fwd`、`dx_flops`、`dw_flops`；attn 节点读取 `attn_compression_ratio`。

**`RooflinePass`**：计算 `compute_us`、`memory_us`、`latency_us`、`bound`（"compute" | "memory"）。

**`CommLatencyPass`**：通信节点按 α-β 公式计算延迟；区分 intra/inter 层。

**`StreamAssignPass`**：分配 `stream_id` 和 `stream_type`；检测重叠类型：`coc | mc2 | ring_cp | none`。

**`TrainingFlopsPass`**（`training.py`）：
- 汇总 `training_flops / forward_flops / backward_flops`
- `recompute_flops = ½ × Σ flops_fwd[recompute=True]`
- 按 `layer_scale` 放大到完整模型层数
- 6P 兜底：仅在 `forward_flops == 0 and backward_flops == 0` 时触发

**`TrainingMemoryPass`**（`training.py`）：
- 优先路径：遍历 fwd→bwd 边，对存活张量字节求和（排除 `is_param=True` 和 `recompute=True` 节点），除以 `tp × cp`
- 退化路径：Korthikanti 系数 × 动态重计算乘数（`_derive_recompute_multiplier`）× CP 分片（`tp × max(cp, 1)`），按 `stage_id` 取峰值在途深度

**`TrainingPipelinePass`**（`training.py`）：
- PP > 1 且节点有 `stage_id` 时：对每个 stage 子图分别运行 `DAGScheduler`，取 `phase_latency("fwd")` 和 `phase_latency("bwd")`；`bwd_dw` 按 dW flops 占比从 `bwd` 中分离
- PP = 1 时：单图调度，构造单阶段 `StageTime`
- overlap 修正：MC2 全部隐藏，CoC 隐藏 `(k-1)/k`，Ring-CP 减去目标 FA tile 时间
- 调用 `PP_SCHED_BY_NAME[ctx.training.pp_schedule]` Composer，将结果写入 `metadata["pipeline_metrics"]`
- 仅在节点缺少 `stage_id` 注解时回退到 PP 平均（带 `warnings.warn`）

#### SB-6：主入口（`transform/analysis/modeller.py`）

```python
estimate_training_from_graphs(
    forward_graph: OpGraph,
    backward_graph: OpGraph | None = None,
    hw_spec: HardwareSpec,
    tp: int = 1, pp: int = 1, ep: int = 1, dp: int = 1, cp: int = 1,
    zero_stage: int = 0,
    optimizer: OptKind = OptKind.ADAM,
    recompute: RecomputePolicy = ...,
    muon_rotation: bool = False,
    pp_schedule: PPSched = PPSched.ONE_F_ONE_B,
    vpp_chunks: int = 1,
    return_transformed: bool = False,
) -> TrainingReport | tuple[TrainingReport, TransformContext, dict[str, OpGraph]]
```

调用链：`stitch_fwd_bwd()` → 构建 `TransformContext` → `TransformPipeline.run()` → 从 `metadata["pipeline_metrics"]` 读取 `StepResult` → 组装并返回 `TrainingReport`。

---

### 验收标准

以下条件须同时满足，方可认为两条路径开发完成：

1. `estimator.estimate()` 和 `estimate_training_from_graphs()` 均返回 `TrainingReport`，通过 `isinstance` 检查
2. `StepResult` 和 `TrainingReport` 的不变量在 `__post_init__` 中有断言，且单元测试覆盖所有 Composer
3. 所有 PP > 1 场景下 Stack B 使用逐阶段 `DAGScheduler`，不使用 PP 平均
4. Stack A 和 Stack B 保持独立执行路径，互不强依赖对方的运行时
5. 锚点测试 `tests/training/anchors/test_anchors.py` 通过（GPT-3 175B、LLaMA-3 70B、DeepSeek-V3）

---

## 仍开放事项

| 项目 | 位置 | 说明 |
|------|------|------|
| Stack A Ring-CP 仍未建模 | `training/ir/shard.py:162–164` | Ulysses-CP 与 EP 集合通信已实现；`CPKind.RING` 仍直接返回，尚未插入 Ring send/recv 或对应 overlap 语义 |
| 搜索空间默认不扫描 CP | `training/search/space.py` | `cp_values` 默认仍为 `[1]`；即使 Stack A IR 已具备 Ulysses-CP 基础建模，默认搜索仍需要显式打开 CP 维度 |
| `perf_tables.py` 为简易启发表 | `training/io/perf_tables.py` | 四档跳变阈值，无 GPU/dtype 区分；Phase 4 待引入实测曲线 |
| `EPLBPass` / `MTPPass` 为 stub | `transform/optim/passes.py` | `run()` 直接返回原图；Stack A 已有 MTP block，但 Stack B 优化 pass 尚未实现 |
| `OpenBoxModel` / `OperatorOptimizationModel` / `SystemDesignModel` | `policy_model/` | `predict()` 体为 `pass`；待对应政策模型设施建成后实现 |
| `LookupSimulator` / `TilesimSimulator` | `simulator/backends/` | `can_simulate = False`；待 lookup/tile 仿真设施建成后实现 |
| Offload 仍缺少 CLI 配置入口 | `cli.py`, `transform/context.py` | `OffloadPass` 已注册且 `TrainingConfig.offload` 已存在，但命令行尚未暴露 offload 比例/对象开关，当前只能由 API 调用方构造 `TransformContext` 启用 |

---

## 关键文件

| 文件 | 作用 | 所属路径 |
|------|------|---------|
| `python/zrt/training/ir/training_graph.py` | Stack A 的 `Graph` + `Op` + `Collective` | Stack A |
| `python/zrt/training/ir/builders.py` | `build_graph(ModelSpec, Strategy) → Graph` | Stack A |
| `python/zrt/training/models/flops.py` | 层级 FLOPs 公式 | Stack A |
| `python/zrt/training/models/comm.py` | α-β 集合通信模型 | Stack A |
| `python/zrt/training/models/memory.py` | Korthikanti 内存公式 | Stack A |
| `python/zrt/training/compose/schedules.py` | `PipelineComposer` + 五个实现（**两路共享**） | 共享 |
| `python/zrt/training/compose/stage.py` | `stage_time()` + `StageTime`（**两路共享**） | 共享 |
| `python/zrt/training/spec/report.py` | `TrainingReport`（**两路统一输出**） | 共享 |
| `python/zrt/training/search/estimator.py` | Stack A 入口 → 返回 `TrainingReport` | Stack A |
| `python/zrt/ir/graph.py` | `OpGraph` + `from_model_spec()` | Stack B / 共享 |
| `python/zrt/ir/adapter.py` | `stitch_fwd_bwd()` | Stack B |
| `python/zrt/transform/analysis/training.py` | `TrainingPipelinePass`（调度桥接） | Stack B |
| `python/zrt/transform/analysis/modeller.py` | Stack B 主入口 `estimate_training_from_graphs()` | Stack B |

---

## 验证策略

```bash
# 接口统一验证
PYTHONPATH=python pytest tests/training/ -v -k "estimator or report" 2>&1 | tail -n 20

# 合成 OpGraph 工厂
PYTHONPATH=python pytest tests/training/test_opgraph_from_spec.py -v

# 全量回归：所有训练测试通过
PYTHONPATH=python pytest tests/training/ -v 2>&1 | tail -n 30

# 锚点回归：MFU 不漂移
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -v
```

---

## TrainingReport 通信时间字段说明（2026-05-15 更新）

`TrainingReport` 提供两种通信时间视角：

### 按可见性分类（exposed / hidden）

- **暴露通信（exposed）**：位于关键路径上，直接增加步骤时间
  - `tp_exposed_ms`: TP RS/AG（经 CoC/MC2 减缩后的暴露部分）
  - `cp_exposed_ms`: CP A2A
  - `ep_exposed_ms`: EP A2A（经 wave-overlap 减缩后）
  - `pp_exposed_ms`: PP P2P
  - `dp_exposed_ms`: DP AR/RS
  - `exposed_comm_ms` = Σ 以上字段

- **隐藏通信（hidden）**：与计算重叠运行，不在关键路径
  - `tp_hidden_ms`: TP 被 CoC/MC2 隐藏
  - `ep_hidden_ms`: EP 被 wave-overlap 隐藏
  - `dp_hidden_ms`: DP AR 吸收在流水线气泡中
  - `hidden_comm_ms` = Σ 以上字段

### 按策略汇总（total）

- **各策略总通信时间** = exposed + hidden
  - `tp_total_ms` = tp_exposed_ms + tp_hidden_ms
  - `cp_total_ms` = cp_exposed_ms（CP 无隐藏）
  - `ep_total_ms` = ep_exposed_ms + ep_hidden_ms
  - `pp_total_ms` = pp_exposed_ms（PP 无隐藏）
  - `dp_total_ms` = dp_exposed_ms + dp_hidden_ms
  - `total_comm_volume_ms` = Σ 以上字段（与 exposed_comm_ms + hidden_comm_ms 相同）

### 使用场景

- **搜索表格汇总**：`training_search_util.py` 使用 `*_total_ms` 字段展示各策略通信开销
- **性能诊断**：`*_exposed_ms` 识别瓶颈，`*_hidden_ms` 评估重叠效率
