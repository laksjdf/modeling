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
║      MoE/MTP 当前回退到 dense_block（Phase 2 TODO）                            ║
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
║               StepResult → Report                                             ║
║                 step_time_ms  mfu  hfu  bubble_fraction                       ║
║                 memory_breakdown  per_stage_ms  warnings                      ║
║                 (可选) grid_search → Pareto 前沿 (step_time, peak_hbm)        ║
╚═══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════════╗
║  Stack B：图捕获路径（主路径 ✅）                                                 ║
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
║      ▼ stitch_fwd_bwd(fwd_graph, bwd_graph)   ✅ ir/adapter.py:613–749       ║
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
║        TrainingMemoryPass  weights/grads/opt_state (ZeRO 缩放)               ║
║                            activations：优先 fwd→bwd 边活字节；               ║
║                                         退化到 Korthikanti 系数 × 在途深度    ║
║      │                                                                        ║
║      ▼ TrainingPipelinePass              transform/analysis/training.py       ║
║        PP>1 且节点有 stage_id：                                                ║
║          for s in range(pp):                                                  ║
║            subgraph    = graph.subgraph([n for n if stage_id==s])            ║
║            timeline[s] = DAGScheduler(hw).schedule(subgraph)                 ║
║            stage_fwd[s]    = timeline[s].phase_latency("fwd")                ║
║            stage_bwd[s]    = timeline[s].phase_latency("bwd")                ║
║            stage_bwd_dw[s] = stage_bwd[s] × (dW_flops / total_bwd_flops)    ║
║        否则：单图调度 + 按 pp 平均（fallback warning，见阶段 A.1）              ║
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
- **收敛点**：两条路径都通过 `PipelineComposer` 类生成 `StepResult`，并最终输出统一的报告类型。

---

## 已完成工作

| 组件 | 文件 | 状态 |
|------|------|------|
| `stitch_fwd_bwd()` | `python/zrt/ir/adapter.py` | ✅ 已实现 |
| `PipelineComposer` + 五个具体实现 | `python/zrt/training/compose/schedules.py` | ✅ 两条路径共享 |
| `TrainingPipelinePass`（Stack B 调度桥接） | `python/zrt/transform/analysis/training.py` | ✅ 已实现 |
| `estimate_training_from_graphs()` | `python/zrt/transform/analysis/modeller.py` | ✅ Stack B 主入口 |

**`stitch_fwd_bwd()` 实现细节**（`ir/adapter.py:613–749`）：
- 合并两图节点（反向节点 ID 加 `bwd_` 前缀以避免冲突）
- 通过张量 ID 匹配插入 fwd→bwd 跨图依赖边（启发式回退：形状+dtype+同 layer/scope）
- 标注：`node.annotations["phase"] = "fwd" / "bwd"`；参数节点标注 `is_param = True`
- 结果：`metadata["fwd_bwd_stitched"] = True`

---

## 统一目标（接口契约）

不统一 IR，而统一**接口契约**：

| 统一项 | 当前状态 | 目标状态 |
|--------|---------|---------|
| 输出类型 | Stack A 返回 `Report`；Stack B 返回 `TrainingReport` | 两者均返回 `TrainingReport` |
| 合成 OpGraph | 不存在 | 新增 `OpGraph.from_model_spec()` 作为快速追踪回退 |
| 跨路径类型泄漏 | `TrainingPipelinePass` 用下划线别名导入 Stack A 类型 | 清理别名，保持导入语义清晰 |

---

## 近期实施计划（阶段 0–2）

### 阶段 0 — 输出类型统一

**目标**：两条路径的调用者无需区分入口，均可使用 `TrainingReport`。

**方案**：
1. 将 `TrainingReport` 移至共享位置 `python/zrt/training/spec/report.py`
2. Stack A 的 `estimator.estimate()` 改为返回 `TrainingReport`（填充可计算的字段子集）
3. Stack B 的 `modeller.py` 更新导入路径

**影响文件**：
- `python/zrt/training/search/estimator.py`（~10 行）
- `python/zrt/training/spec/report.py`（新建或扩展，~30 行）
- `python/zrt/transform/analysis/modeller.py`（~2 行）

---

### 阶段 1 — 新增 `OpGraph.from_model_spec()` 工厂方法

**目标**：为 Stack A 提供一个"合成 OpGraph"，使其可在有限情况下接入 Stack B 的变换流水线（无需完整图捕获）。

**设计**：

```python
# python/zrt/ir/graph.py
@classmethod
def from_model_spec(cls, model: ModelSpec, strategy: Strategy, phase: str = "training") -> "OpGraph":
    """从 ModelSpec 构建合成 OpGraph，节点对应 training.ir.Graph 中的算子。

    用途：无真实追踪时的快速估算回退。节点携带层级元数据但无真实张量数据流。
    """
    from zrt.training.ir.builders import build_graph
    training_g = build_graph(model, strategy)

    nodes = {}
    for op in training_g.ops:          # op 字段：name, kind, inputs, outputs, meta, layer_id, layer_kind
        nodes[op.name] = OpNode(
            id=op.name,
            op_type=op.kind,
            annotations={"layer_id": op.layer_id, "layer_kind": op.layer_kind},
            meta={**op.meta},
        )

    edges = []
    op_names = list(nodes.keys())
    for i in range(len(op_names) - 1):
        curr, nxt = op_names[i], op_names[i + 1]
        if nodes[curr].annotations.get("layer_id") == nodes[nxt].annotations.get("layer_id"):
            edges.append(Edge(src=curr, dst=nxt))

    return cls(
        name=f"{model.name}_{phase}",
        nodes=nodes,
        edges=edges,
        metadata={
            "source": "model_spec",
            "model": model.name,
            "strategy": strategy,
            "collectives": {c.name: c for c in training_g.collectives},
        }
    )
```

**影响文件**：`python/zrt/ir/graph.py`（+80 行）

**测试**：`tests/training/test_opgraph_from_spec.py`
- `len(opgraph.nodes) == len(training_g.ops)`
- op 类型逐一匹配
- `metadata["source"] == "model_spec"`

---

### 阶段 2 — 清理跨路径类型泄漏

**当前状态**：`TrainingPipelinePass` 用下划线别名导入 Stack A 类型：

```python
from python.zrt.training.compose.stage import StageTime as _StageTime
from python.zrt.training.compose.schedules import PP_SCHED_BY_NAME, COMPOSER_BY_SCHED
from python.zrt.training.spec.strategy import Strategy as _Strategy, OptKind
```

**处理方式**：
- `COMPOSER_BY_SCHED` 导入合理（共享组件），无需修改
- `_StageTime` 和 `_Strategy` 的下划线别名无实际意义，直接去掉别名前缀
- 不需要抽象为 Protocol —— 导入本身是合理的，仅是命名风格问题

**影响文件**：`python/zrt/transform/analysis/training.py`（~5 行）

---

## 明确不做的事项

| 计划项 | 放弃原因 |
|---------|---------|
| 将 `TrainingGraph` 改为包装 `OpGraph` | 反转依赖方向；为层级公式强加节点级遍历，复杂度净增 |
| 将 `flops.py` / `comm.py` / `memory.py` 改为接受 `OpGraph` | 这些模型服务于 Stack A 的解析公式；切换到节点级遍历会丢失清晰的分析结构 |
| 修改 `PipelineComposer` 签名使其接受 `graph: OpGraph` | Composer 已被两条路径共享且工作正常；强加图依赖会破坏 Stack A 的独立性 |
| 删除 `training.ir.training_graph.Graph` | Stack A 的 `Graph` 对规格驱动估算而言是正确的抽象，保留它 |

---

## 后续路线图（Stack B 完整实现）

以下阶段聚焦于完善 Stack B 的精确建模能力，与近期接口统一工作独立推进。

### 阶段 A — 修复正确性 Bug

三个已知问题影响当前 Stack B 输出精度：

**A.1 — 步骤时间：PP 平均化回退**（`training.py` 中 `per_stage_us = stage_time_us / pp`）

当 PP>1 但节点缺少 `stage_id` 注解时，整图调度结果按 pp 平均，忽略阶段异构性和 warmup/cooldown 结构。
- **修复**：依赖 `PipelineParallelPass` 正确标注 `stage_id`，对每个阶段视图分别运行 `DAGScheduler`；过渡期至少改正 1F1B 公式为 `M·t_stage + (pp-1)·t_fwd_stage + (pp-1)·t_bwd_stage`。

**A.2 — 激活内存：退化系数路径**（`TrainingMemoryPass` 回退公式）

退化路径使用固定系数 `10 × hidden × seq × layers / tp`，忽略 `RecomputePass` 层级标记（可减少 50–80%）、CP 分片（`/cp`）以及 `ZeroFSDPPass` 已写入的 ZeRO 因子。
- **修复**：优先读取 `metadata["zero"]`；对 `RecomputePass` 标记层级应用零化；拼接图可用时用实际 fwd→bwd 边活字节替代系数。

**A.3 — 总 FLOPs：覆盖逻辑**（`TrainingFlopsPass`）

`6P` 兜底估算有时覆盖 `FlopsPass` 已正确标注的逐节点 `flops_fwd/dx/dw`。
- **修复**：对所有节点求和逐节点 FLOPs；删除覆盖逻辑；仅在 `FlopsPass` 未运行时将 `6P` 作为回退。

---

### 阶段 B — 流水线并行阶段分配与合成

**B.1 — `PipelineParallelPass` 完善**（`transform/parallel/pipeline_parallel.py`）
- 按 `node.annotations["compute_us"]` 之和贪心装箱进行阶段分配（需先运行 `RooflinePass`）
- 在阶段边界插入 `comm.send_recv` P2P 节点（跨阶段激活传输，大小来自 `TensorMeta.mem_bytes`）

**B.2 — 逐阶段 DAGScheduler 调度**（`estimate_training_from_graphs()`）

变换流水线完成后：
```
stage_nodes[s] = [n for n in g.nodes if n.annotations["stage_id"] == s]
timeline[s] = DAGScheduler(hw).schedule(stage_nodes[s])
t_fwd[s] = timeline[s].phase_latency("fwd")
t_bwd[s] = timeline[s].phase_latency("bwd")
```

**B.3 — 1F1B 合成器修正**（`TrainingPipelinePass`）
```
t_stage = max(t_fwd[s] + t_bwd[s] for s in stages)   # 瓶颈阶段
step_us = (pp-1)*t_fwd[0] + M*t_stage + (pp-1)*t_bwd[pp-1]
bubble  = 2*(pp-1)*t_stage / step_us
```
使用真实阶段时延，而非基于计数的公式。

---

### 阶段 C — 并行度完整性与精确内存建模

**C.1 — `ContextParallelPass`**（`transform/parallel/context_parallel.py`）
- **Ulysses CP**：在注意力前插入 `comm.all_to_all`（分散序列、聚合头），在注意力后插入逆变换。组大小 = `cp`，数据量 = `b × s/cp × h`。
- **Ring CP**：在注意力内插入 `cp` 轮 `comm.send_recv`，每轮对应一个 KV 块。标记为可与 FA tile 计算重叠（`annotations["overlap_target"] = paired_fa_tile_id`）。
- 扩展 `CommInserterPass` 以支持 CP 通信。

**C.2 — `DataParallelPass`**（`transform/parallel/data_parallel.py`）
- 在反向节点后，对每个参数节点的梯度张量插入 `comm.all_reduce`（ZeRO-0）或 `comm.reduce_scatter`（ZeRO-2/3）。
- 标注 `annotations["dp_comm"] = True`；`dp_overlap_in_bubble` 为 True 时标注 `annotations["overlap_in_bubble"] = True`。
- 流水线合成器从 bubble 窗口减去 `t_dp_ar`：`t_exposed_dp = max(0, t_dp - bubble_duration)`。

**C.3 — `StreamAssignPass` 中的 CoC/MC2 重叠规则**
- **CoC**：`annotations["coc_tile_k"] = k`；暴露通信时间 = `max(0, t_comm - t_matmul * (k-1)/k)`。
- **MC2**：AG+matmul 融合节点，暴露通信时间为零。
- **Ring-CP**：相邻 FA tile 的 P2P 节点获得重叠标注；暴露时间 = `max(0, t_p2p - t_fa_tile)`。

**C.4 — 基于图的精确内存建模**（前向+反向图拼接后）
- **参数**：`is_param == True` 节点输出之和 / ZeRO 权重分片因子（来自 `ZeroFSDPPass`）
- **激活**：fwd→bwd 跨图边上的张量；对 `RecomputePass` 标记层级应用零化
- **梯度/优化器状态**：`OptimizerPass` 标注（`state_bytes`），除以 ZeRO 分片因子
- **在途 μbatch 深度**：1F1B 稳态下，阶段 `s` 同时持有 `pp - s` 个 μbatch 的激活

---

### 阶段 D — 高级调度、搜索与验证

**D.1 — VPP / 交错 1F1B**
bubble 公式：`(pp-1) / (vpp × M)`。需要 `TrainingConfig` 中的 `vpp_chunks`。

**D.2 — DualPipe / DualPipeV**
每个阶段并发运行 μbatch_i 前向与 μbatch_{i-1} 反向；bubble 约为 1F1B 的一半。`dualbatch=True` 时 EP A2A 由对端 μbatch 计算隐藏。

**D.3 — EP 负载不均衡**
`t_expert_bottleneck = t_avg * (1 + imbalance)`，作为乘数应用于专家矩阵乘节点时延。

**D.4 — 搜索与帕累托前沿**（`search/sweep.py`）
对 `(tp, cp, pp, ep, dp, zero_stage)` 进行网格搜索，附带剪枝规则；输出按 `(step_time, peak_hbm)` 的帕累托前沿。

**D.5 — 锚点验证**（`tests/training/anchors/`）
GPT-3 175B、Llama-3 70B、DeepSeek-V3 —— CI 断言估算 MFU 误差在 15% 以内。

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
| `python/zrt/training/search/estimator.py` | Stack A 入口 → 目标返回 `TrainingReport` | Stack A |
| `python/zrt/ir/graph.py` | `OpGraph` + 目标新增 `from_model_spec()` | Stack B / 共享 |
| `python/zrt/ir/adapter.py` | `stitch_fwd_bwd()`（已实现） | Stack B |
| `python/zrt/transform/analysis/training.py` | `TrainingPipelinePass`（调度桥接） | Stack B |
| `python/zrt/transform/analysis/modeller.py` | Stack B 主入口 `estimate_training_from_graphs()` | Stack B |

---

## 验证策略

```bash
# 近期：接口统一验证
PYTHONPATH=python pytest tests/training/ -v -k "estimator or report" 2>&1 | tail -n 20

# 近期：合成 OpGraph 工厂
PYTHONPATH=python pytest tests/training/test_opgraph_from_spec.py -v

# 全量回归：所有训练测试通过
PYTHONPATH=python pytest tests/training/ -v 2>&1 | tail -n 30

# 锚点回归：MFU 不漂移
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -v
```

---

## 成功标准

1. `estimator.estimate()` 和 `estimate_training_from_graphs()` 均返回 `TrainingReport`
2. `OpGraph.from_model_spec(model, strategy)` 产生节点数和类型与 `build_graph()` 一致的 OpGraph
3. 所有现有训练测试通过（无回归）
4. Stack A 和 Stack B 保持独立执行路径 —— 互不强依赖对方的运行时
