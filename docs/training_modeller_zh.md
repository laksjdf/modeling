# 训练建模器 —— 双路径架构与实施路线

_2026-04-28。合并自 `training_modeller_zh.md`（2026-04-23 架构审查）与 `training_modeller_zh_v2.md`（2026-04-28 统一方案）。最后更新：2026-04-28，对齐近期已完成实施。_

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

## 已完成工作

| 组件 | 文件 | 状态 |
|------|------|------|
| `stitch_fwd_bwd()` | `python/zrt/ir/adapter.py` | ✅ 已实现 |
| `PipelineComposer` + 五个具体实现 | `python/zrt/training/compose/schedules.py` | ✅ 两条路径共享 |
| `TrainingPipelinePass`（Stack B 调度桥接） | `python/zrt/transform/analysis/training.py` | ✅ 已实现 |
| `estimate_training_from_graphs()` | `python/zrt/transform/analysis/modeller.py` | ✅ Stack B 主入口 |
| `TrainingReport`（统一输出类型） | `python/zrt/training/spec/report.py` | ✅ 已实现（Phase 0）|
| `estimator.estimate()` 返回 `TrainingReport` | `python/zrt/training/search/estimator.py` | ✅ 已实现（Phase 0）|
| `OpGraph.from_model_spec()` 工厂方法 | `python/zrt/ir/graph.py:218–285` | ✅ 已实现（Phase 1）|
| `test_opgraph_from_spec.py` | `tests/training/test_opgraph_from_spec.py` | ✅ 5 个测试函数 |
| 清理 `training.py` 下划线类型别名 | `python/zrt/transform/analysis/training.py` | ✅ 已实现（Phase 2）|
| A.1 逐阶段 DAGScheduler（不再按 PP 平均）| `training.py:423–469` | ✅ 已实现 |
| A.2 动态重计算乘数激活内存公式 | `training.py:228–241, 293–326` | ✅ 已实现 |
| A.3 6P 规则改为真正兜底 | `training.py:79–92` | ✅ 已实现 |
| B.1 `PipelineParallelPass`：`compute_us` 贪心装箱 + P2P 插入 | `pipeline_parallel.py:84–234` | ✅ 已实现 |
| B.2–B.3 PP>1 时逐阶段 DAGScheduler | `training.py:415–469` | ✅ 已实现 |
| C.1 `ContextParallelPass`：Ulysses A2A + Ring P2P | `comm_inserter.py:208–268` | ✅ 已实现 |
| C.2 `DataParallelPass`：反向后插 AR/RS | `data_parallel.py:85–121` | ✅ 已实现 |
| C.3 `StreamAssignPass`：CoC/MC2/Ring-CP overlap 规则 | `passes.py:220–232`, `training.py:733–762` | ✅ 已实现 |
| C.4 基于图的精确激活内存建模 | `training.py:234–236, 328–361` | ✅ 已实现 |
| D.5 锚点 YAML 固件 + 测试框架 | `tests/training/anchors/`（3 个 YAML + 测试文件）| ✅ 已实现 |
| Recompute/ZeRO/Optimizer/Offload 训练 pass 注册 | `python/zrt/transform/pipeline.py:111–118`, `python/zrt/transform/context.py` | ✅ 已实现 |
| Stack A MoE / MTP block | `python/zrt/training/ir/builders.py` | ✅ 已实现 |
| Stack A Ulysses-CP / EP collectives | `python/zrt/training/ir/shard.py` | ✅ 已实现（Ring-CP 仍开放） |
| 量化 CLI / pass / Roofline dtype | `python/zrt/cli.py`, `python/zrt/transform/optim/passes.py`, `python/zrt/transform/analysis/passes.py` | ✅ 已实现 |

**`stitch_fwd_bwd()` 实现细节**（`ir/adapter.py:613–749`）：
- 合并两图节点（反向节点 ID 加 `bwd_` 前缀以避免冲突）
- 通过张量 ID 匹配插入 fwd→bwd 跨图依赖边（启发式回退：形状+dtype+同 layer/scope）
- 标注：`node.annotations["phase"] = "fwd" / "bwd"`；参数节点标注 `is_param = True`
- 结果：`metadata["fwd_bwd_stitched"] = True`

---

## 统一目标（接口契约）—— 已达成

| 统一项 | 当前状态 | 结果 |
|--------|---------|------|
| 输出类型 | 两条路径均返回 `TrainingReport`（Stack A 保留 `Report = TrainingReport` 向后兼容别名） | ✅ 达成 |
| 合成 OpGraph | `OpGraph.from_model_spec()` 已实现于 `ir/graph.py:218–285` | ✅ 达成 |
| 跨路径类型泄漏 | `training.py` 中已无下划线别名，导入语义清晰 | ✅ 达成 |

---

## 已完成实施路线图（Stack B）

以下各阶段均已完成，保留技术细节供参考。

### 阶段 0 — 输出类型统一 ✅

**目标**：两条路径的调用者均可使用 `TrainingReport`。

**已完成**：
1. `TrainingReport` 已移至共享位置 `python/zrt/training/spec/report.py`
2. Stack A 的 `estimator.estimate()` 已改为返回 `TrainingReport`（`estimator.py:25`）；保留 `Report = TrainingReport` 别名供向后兼容
3. Stack B 的 `modeller.py` 已更新为从 `zrt.training.spec.report` 导入

---

### 阶段 1 — `OpGraph.from_model_spec()` 工厂方法 ✅

**目标**：为 Stack A 提供"合成 OpGraph"，使其在有限情况下可接入 Stack B 的变换流水线。

**已完成**：`python/zrt/ir/graph.py:218–285`，从 `ModelSpec` + `Strategy` 构建合成 OpGraph，节点携带层级元数据但无真实张量数据流。测试：`tests/training/test_opgraph_from_spec.py`（5 个测试函数）。

---

### 阶段 2 — 清理跨路径类型泄漏 ✅

**已完成**：`TrainingPipelinePass` 中的 `_StageTime` / `_Strategy` 下划线别名已去除；`COMPOSER_BY_SCHED` 等共享组件导入保持不变，语义清晰。

---

### 阶段 A — 修复正确性 Bug ✅

**A.1 — 步骤时间：PP 平均化回退** ✅  
`training.py:423–469`：PP>1 且节点有 `stage_id` 时，对每个阶段子图分别运行 `DAGScheduler`，得到真实 per-stage latency。仅在节点缺少 `stage_id` 注解时回退到 PP 平均（带 warning）。

**A.2 — 激活内存：退化系数路径** ✅  
`training.py:228–241, 293–326`：优先使用图原生路径（fwd→bwd 边活字节）；Korthikanti 兜底公式已集成动态重计算乘数（`_derive_recompute_multiplier`）和 CP 分片（`shard = tp * max(cp, 1)`）；按 `stage_id` 计算峰值在途深度。

**A.3 — 总 FLOPs：覆盖逻辑** ✅  
`training.py:79–92`：逐节点 FLOPs 求和为主路径；`6P` 兜底仅在 `forward_flops == 0 and backward_flops == 0` 时触发。

---

### 阶段 B — 流水线并行阶段分配与合成 ✅

**B.1 — `PipelineParallelPass` 完善** ✅  
`pipeline_parallel.py:84–234`：按 `compute_us`（→ `latency_us` → `flops` 降级）贪心装箱分配 stage；跨 stage 边替换为 `comm.send_recv` P2P 节点，放在接收 stage。

**B.2–B.3 — 逐阶段调度 + 1F1B 合成器** ✅  
`training.py:415–469`：变换流水线完成后按 `stage_id` 切子图，各自运行 `DAGScheduler`，取 `phase_latency("fwd"/"bwd")`，构造 `StageTime` 列表传入 `PipelineComposer`。

---

### 阶段 C — 并行度完整性与精确内存建模 ✅

**C.1 — `ContextParallelPass`** ✅  
Ulysses CP：注意力前后各插 `comm.all_to_all`（`comm_inserter.py:208–244`）。Ring CP：插 `cp` 轮 `comm.send_recv`，标 `overlap_target = "fa_tile:<id>"`（`comm_inserter.py:246–268`）。

**C.2 — `DataParallelPass`** ✅  
`data_parallel.py:85–121`：按 layer 聚合梯度生产节点，插 layer 粒度的 `comm.all_reduce`（ZeRO-0）或 `comm.reduce_scatter`（ZeRO-2/3）；`dp_overlap_in_bubble` 标 `overlap_in_bubble=True`。

**C.3 — `StreamAssignPass` 中的 CoC/MC2 重叠规则** ✅  
`passes.py:220–232`：按节点属性检测 `ring_cp / mc2 / coc / none`。`training.py:733–762`：`compute_exposed_comm_time()` MC2 全部隐藏，CoC 隐藏 `(k-1)/k`，Ring-CP 减去目标 FA tile 时间。

**C.4 — 基于图的精确内存建模** ✅  
`training.py:234–236, 328–361`：拼接图可用时，`_graph_native_activations()` 遍历 fwd→bwd 边，按 `is_param / recompute` 注解过滤后对存活张量字节求和，除以 `tp × cp`。

---

### 阶段 D — 验证 ✅

**D.5 — 锚点验证** ✅  
`tests/training/anchors/`：GPT-3 175B、LLaMA-3 70B、DeepSeek-V3 三个 YAML 固件；`test_anchors.py` 含集成测试（`estimate()` + MFU 容忍度校验）。运行时需设置 `PYTHONPATH=python`。

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
| `python/zrt/ir/adapter.py` | `stitch_fwd_bwd()`（已实现） | Stack B |
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

## 成功标准 —— 已达成

1. ✅ `estimator.estimate()` 和 `estimate_training_from_graphs()` 均返回 `TrainingReport`
2. ✅ `OpGraph.from_model_spec(model, strategy)` 产生节点数和类型与 `build_graph()` 一致的 OpGraph
3. ✅ 所有现有训练测试通过（无回归）
4. ✅ Stack A 和 Stack B 保持独立执行路径 —— 互不强依赖对方的运行时

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
