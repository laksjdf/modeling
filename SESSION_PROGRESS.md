# Session Progress

## 当前文件状态

| 文件 | 状态 |
|------|------|
| `ARCHITECTURE.md` | ✅ V2 完成，含 10 个章节 |
| `python/zrt/graph/*` | ✅ 现有图抓取+融合引擎，可用 |
| `python/zrt/ir/` | ✅ OpGraph IR 完整实现 + NetworkX 适配器 |
| `python/zrt/graph/main.py` | ✅ capture 层已迁移：`run_trace/run_trace_phases` 输出 `OpGraph` |
| `python/zrt/hardware/` | ✅ 完整实现：spec.py + registry.py + 5 个 YAML 配置 |
| `python/zrt/simulator/` | ✅ 完整实现：Phase 1 核心 Roofline 仿真器 |
| `python/zrt/transform/` | ✅ 完整实现：4-stage Transform Pipeline |
| `python/zrt/executor/` | ✅ 完整实现：DAGScheduler + Timeline |

## 已解决的问题

### transform/ 模块实现（本次）

**方案**：方案 B（先抓图再变换），两者都要（拓扑变化 + stream 注解）

**新增文件**：
- `python/zrt/transform/__init__.py`：公开 API
- `python/zrt/transform/base.py`：GraphPass ABC
- `python/zrt/transform/context.py`：TransformContext, ParallelConfig, StreamConfig, QuantConfig
- `python/zrt/transform/pipeline.py`：TransformPipeline + build_default_pipeline()
- `python/zrt/transform/parallel/tensor_parallel.py`：TensorParallelPass（column/row parallel shape修改 + 注解）
- `python/zrt/transform/parallel/expert_parallel.py`：ExpertParallelPass（MoE EP 注解）
- `python/zrt/transform/parallel/comm_inserter.py`：CommInserterPass（插入 comm.all_reduce / comm.all_to_all 节点，边重连）
- `python/zrt/transform/fusion/pass_.py`：FusionPass（现有引擎的 stub 适配器）
- `python/zrt/transform/optim/passes.py`：QuantizationPass / EPLBPass / SharedExpertPass / MTPPass
- `python/zrt/transform/analysis/passes.py`：FlopsPass + RooflinePass + StreamAssignPass

**核心设计**：
- TP column parallel：outputs 最后一维 / tp，同步更新出边 tensor shape
- TP row parallel：inputs[0] 最后一维 / tp，标注 comm_after=all_reduce
- CommInserter：在 row-parallel 节点后插入 comm.all_reduce，EP expert 块前后插入 comm.all_to_all，边重连正确
- StreamAssignPass：compute 节点 → stream 0..num_compute-1，comm 节点 → stream num_compute..total-1，round-robin

**测试**：
- `tests/test_transform.py`：18 个测试全部通过（2.08s）
- TP shape 修改验证
- comm 节点插入和边重连验证
- 多流 stream_id 分配验证（含多 compute/comm 流配置）
- 端到端 pipeline 验证

## 已完成（本次）：Executor / DAGScheduler

**文件**：
- `python/zrt/executor/scheduler.py`：DAGScheduler + Timeline + ScheduledOp
- `python/zrt/executor/__init__.py`：公开 API

**核心设计**：
- 拓扑序 + list scheduling：`start = max(前驱完成时间, 所在 stream 可用时间)`
- `latency_us` 优先从 annotations 读取，fallback 到 Roofline 估算（需传 hw_spec），再 fallback 到 1 µs
- `Timeline.overlap_us = compute_time + comm_time - total_latency`（量化通算掩盖收益）
- `Timeline.ops_on_stream(id)` 返回指定 stream 的按时间排序的 op 列表

**测试**：
- `tests/test_executor.py`：14 个测试全部通过（1.28s）
- 覆盖：单节点、线性链、依赖顺序、同 stream 串行化、不同 stream 并行、overlap 量化、无 overlap 线性链、latency fallback、完整 pipeline 集成

## 下一步待办

1. ~~**FusionPass 真正接入 OpGraph IR**~~ ✅ 已完成

2. ~~**E2ESummary 报表**~~ ✅ 已完成
   - `python/zrt/report/summary.py`：E2ESummary + build_summary()
   - 16 个测试全部通过

3. **capture 层迁移（可选后续）**
   - 将 `export_all` 从 `nx.DiGraph` 改为直接接收 `OpGraph`

## 已完成（2026-04-23）：Phase 0 — stitch_fwd_bwd() 统一前向+反向图

**设计文档**：`docs/training_modeller_zh.md` Phase 0

**修改文件**：
- `python/zrt/ir/adapter.py`：新增 `stitch_fwd_bwd(fwd_graph, bwd_graph) -> OpGraph`（~130 行）
- `python/zrt/ir/__init__.py`：导出 `stitch_fwd_bwd`
- `python/zrt/transform/analysis/modeller.py`：`estimate_training_from_graphs()` 在有 fwd+bwd 图时先拼接再跑 pipeline
- `tests/training/test_captured_graph_modelling.py`：7 个新测试

**核心设计**：
- 反向节点 ID 加 `bwd_` 前缀避免冲突
- 所有节点标注 `annotations["phase"]` = `"fwd"` / `"bwd"`
- 权重读取节点标注 `annotations["is_param"] = True`
- 跨图依赖边通过 `(shape, dtype)` 匹配，同 layer 优先
- `estimate_training_from_graphs()` 统一路径：stitch → 单次 pipeline run → 提取 metrics

**测试**：65/65 training tests pass（含 7 个新 stitch 测试），零回归

**下一步待办（Phase 1）**：
- 修复步骤时间公式（training.py per_stage_us 不应简单除以 pp）
- 修复激活内存估算（读取 RecomputePass + ZeroFSDPPass 标注）
- 修复总 FLOPs（使用逐节点 flops_fwd/dx/dw 求和替代 6P 覆盖）

## 已完成（2026-04-23）：Phase 0 改进 — 4 个 Issue 全部修复

**参考文档**：`docs/phase0_improvement_plan.md`

**Issue 1（高优先级）：跨图边匹配改为 tensor ID 主键**
- `dispatch.py`：`RecordingDispatch.__torch_dispatch__` 在暂停状态下也调用 `get_id()`，确保前向张量获得 ID
- `main.py`：`_trace_phase` 接受可选 `tensor_tracker`；`run_trace_phases` 创建共享 tracker 传入训练阶段
- `adapter.py`：Phase 5 先按 tensor_id 精确匹配，再按 (shape, dtype) 回退；`_best_cross_match` 增加作用域邻近和 LIFO 启发

**Issue 2（中优先级）：参数检测泛化**
- `adapter.py`：`_PARAM_READ_OPS` 新增 `aten.embedding.default`、`aten._convolution.default`；embedding/conv 类算子不看 scope 即判为参数节点

**Issue 3（低优先级）：元数据合并**
- `adapter.py`：metadata 合并 bwd + fwd（fwd 优先），保留 namespaced 副本 `fwd_metadata` / `bwd_metadata`

**Issue 4（测试覆盖）**：
- `test_stitch_cross_edges_within_same_layer`：验证跨层不串边
- `test_stitch_preserves_both_metadata`：验证元数据双向保留
- `test_stitch_param_detection_covers_embedding`：验证 embedding 节点判为参数

**测试**：68/68 training tests pass（含 10 个 stitch 相关），零回归

**Follow-up A（已修复）**：新增 `test_stitch_cross_edges_use_tensor_ids` 验证 tensor ID 主键匹配路径（69/69 pass）
**Follow-up B（延迟）**：非 Llama/DeepSeek matmul 命名（c_attn, w1/w2/w3 等）延后至实际需要时处理
