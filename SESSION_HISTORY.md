# Session History

按时间戳记录每个里程碑的核心变更和结果。最新状态见 `SESSION_PROGRESS.md`。

---

## 2026-04-16 — prefill/decode 分阶段抓图 + DynamicCache shim

### 变更
- `python/zrt/graph/main.py`：新增 `run_trace_phases` / `_trace_phase` / `_save_phase_outputs`；CLI 支持 `--phases`
- `python/zrt/graph/compat.py`：新增 `_shim_dynamic_cache_legacy_api`（5 个 DynamicCache 4.x→5.x shim）
- `python/zrt/graph/__init__.py`：导出 `run_trace_phases`
- `test_screenshot_ops.py`：新增 8 个 prefill/decode 测试用例
- `CLAUDE.md`：更新 API 文档、shim 清单、测试命令

### 已解决
- prefill / decode 分阶段抓图（文件名含 `_prefill_` / `_decode_`）
- `DynamicCache.from_legacy_cache / to_legacy_cache / get_usable_length / seen_tokens / get_max_length` 在 transformers 5.x 缺失 → shim 修复
- decode 无 KV cache 时 mask shape 不匹配 → 改为 `(1,1,1,1)`

### 测试结果
**57 passed, 0 failed**（transformers 5.4.0）

### 下一步待办
- 无明确待办。等待新任务。

---

## 2026-04-17 — fusion_rules.py + fusion.py 重构

### 变更
- `python/zrt/graph/fusion_rules.py`（新建 ~180 行）：ALWAYS_TRANSPARENT / SHAPE_OPS / INIT_OPS 透明算子集合；SEMANTIC_LABELS 15 条正则；SubPattern + match_subsequence；PLATFORM_SUBPATTERNS + PLATFORM_SETTINGS
- `python/zrt/graph/fusion.py`（重构 ~270 行）：三阶段引擎 Pass1（leaf 分组）→ Pass2（parent 合并）→ Pass3（语义标签 + 子序列匹配 + AddRMSNorm）
- `python/zrt/graph/excel_writer.py`：ExcelWriter 增加 platform 参数
- `python/zrt/graph/main.py`：run_trace / run_trace_phases / CLI 均增加 platform 参数
- `python/zrt/graph/graph_exporter.py`：ONNX builder 重写（无 value_info、无 weight input boxes）
- `python/zrt/graph/__init__.py`：导出 fusion_rules 新符号

### 下一步待办
- 无明确待办（融合引擎完成）

---

## 2026-04-17 — ARCHITECTURE.md V2

### 变更
- `ARCHITECTURE.md`：V2 完成，含 10 个章节，9 项结构性改进全部落地

### 下一步待办
- 可按 Phase 1 路线开始实现：ir/ + model/ + capture/ + hardware/ + simulator/roofline + executor/单流 + report/

---

## 2026-04-17 14:27 — python/zrt/ir/ 模块实现

### 新增文件
| 文件 | 内容 |
|------|------|
| `types.py` | DType 枚举（12种）、TensorMeta frozen dataclass、shape/dtype 解析工具 |
| `node.py` | OpNode dataclass，保留 dispatch 全部字段 |
| `edge.py` | Edge dataclass，含 tensor_id 追踪 |
| `graph.py` | OpGraph：add_node/add_edge/topo_sort/subgraph/insert_after/replace_subgraph/clone + 懒加载 hierarchy |
| `hierarchy.py` | GraphHierarchy + HierNode：scope 树、at_depth/find/aggregate/module_breakdown |
| `serde.py` | JSON 序列化/反序列化 + save_json/load_json |
| `adapter.py` | records→OpGraph、fused_records→OpGraph、NetworkX DiGraph ↔ OpGraph |

### 验证
records_to_opgraph / nx 双向转换正确，所有 smoke test 通过。

### 下一步待办
1. **迁移 capture 层**（优先推荐）：`run_trace()` / `run_trace_phases()` 改为输出 `OpGraph`，调用 `records_to_opgraph()` 替换现有 `build_op_graph()`，保持 API 向后兼容
2. **实现 hardware/ 模块**（与 capture 迁移无依赖，可并行）：YAML 加载 + HardwareSpec dataclass；ascend_910b.yaml + nvidia_h100.yaml
3. **实现 simulator/roofline**：依赖 OpGraph IR（已完成）和 hardware/（待完成）

---

## 2026-04-17 — capture 层迁移

### 变更
- `python/zrt/graph/main.py`：新增 `TraceResult` / `TracePhaseResult` tuple 子类（向后兼容 2-tuple 拆包）；`_save_phase_outputs` 调用 `records_to_opgraph` + `fused_records_to_opgraph` 构建 OpGraph
- `python/zrt/graph/__init__.py`：导出 `OpGraph`、`TraceResult`、`TracePhaseResult`、`records_to_opgraph`、`fused_records_to_opgraph`

### 测试结果
**48 passed, 27 skipped, 0 failed**（-m "not network"）

### 下一步待办
- 实现 hardware/ 模块（Phase 1 下一里程碑）
- 将 `export_all` 从 `nx.DiGraph` 改为直接接收 `OpGraph`（可选后续）

---

## 2026-04-17 — python/zrt/hardware/ 模块实现

### 新增文件
- `python/zrt/hardware/__init__.py`：公开 API 导出
- `python/zrt/hardware/spec.py`：HardwareSpec / ComputeSpec / MemorySpec / MemoryTier / InterconnectSpec / LinkSpec
- `python/zrt/hardware/registry.py`：`load(name) → HardwareSpec`，支持 file-stem 和 display-name 查找
- `python/zrt/hardware/configs/`：ascend_910b/910c、nvidia_a100_80g/h100_sxm/h800 共 5 个 YAML

### 验证
5 个 config smoke test 全部通过，peak_flops/hbm_bandwidth 返回正确，KeyError 未知硬件时正确报错。

### 下一步待办
1. **simulator/roofline**（Phase 1 核心）：RooflineSimulator + SimResult + SimulatorHub，公式覆盖 matmul/elementwise/layernorm/softmax
2. capture 层迁移（可选）：将 `export_all` 改为直接接收 `OpGraph`

---

## 2026-04-18 — python/zrt/simulator/ 模块实现

### 新增文件
| 文件 | 内容 |
|------|------|
| `python/zrt/simulator/result.py` | SimResult dataclass（latency_us, compute_us, memory_us, flops, bytes, arithmetic_intensity, bound, hw_utilization, backend, confidence） |
| `python/zrt/simulator/base.py` | OpSimulator ABC（name, priority, can_simulate, simulate） |
| `python/zrt/simulator/cache.py` | SimCache + content_hash（op_type + shapes/dtypes + attrs + hw.name） |
| `python/zrt/simulator/hub.py` | SimulatorHub（优先级路由 + cache + simulate_graph） |
| `python/zrt/simulator/backends/roofline.py` | RooflineSimulator（priority=0，兜底），覆盖 matmul/attn/norm/softmax/elementwise/embedding/shape ops |

### 测试结果
**tests/test_simulator.py：30 passed, 0 failed（0.08s）**

### 下一步待办
1. **实现 transform/ pipeline（Phase 2 准备）**：TransformContext + GraphPass ABC + TransformPipeline，先实现基础骨架，再加 TP/EP 切分
2. capture 层迁移（可选）：将 `export_all` 从 `nx.DiGraph` 改为直接接收 `OpGraph`
# Session Progress

## 当前文件状态

| 文件 | 状态 |
|------|------|
| `ARCHITECTURE.md` | ✅ V2 完成，含 10 个章节 |
| `python/zrt/graph/*` | ✅ 现有图抓取+融合引擎，可用 |
| `python/zrt/ir/` | ✅ OpGraph IR 完整实现 + NetworkX 适配器 |
| `python/zrt/graph/main.py` | ✅ capture 层已迁移：`run_trace/run_trace_phases` 输出 `OpGraph` |
| `python/zrt/graph/__init__.py` | ✅ 导出 `OpGraph`、`TraceResult`、`TracePhaseResult`、`records_to_opgraph` |
| `python/zrt/hardware/` | ✅ 完整实现：spec.py + registry.py + 5 个 YAML 配置 |
| `python/zrt/simulator/` | ✅ 完整实现：Phase 1 核心 Roofline 仿真器 |

## 已解决的问题

### simulator/ 模块实现（本次）

**新增文件**：
- `python/zrt/simulator/__init__.py`：公开 API（SimResult, OpSimulator, SimulatorHub, RooflineSimulator, SimCache）
- `python/zrt/simulator/result.py`：SimResult dataclass（latency_us, compute_us, memory_us, flops, read_bytes, write_bytes, arithmetic_intensity, bound, hw_utilization, backend, confidence）
- `python/zrt/simulator/base.py`：OpSimulator ABC（name, priority, can_simulate, simulate）
- `python/zrt/simulator/cache.py`：SimCache + content_hash（基于 op_type + input shapes/dtypes + attrs + hw.name）
- `python/zrt/simulator/hub.py`：SimulatorHub（优先级路由 + cache + simulate_graph）
- `python/zrt/simulator/backends/__init__.py`：backends 包
- `python/zrt/simulator/backends/roofline.py`：RooflineSimulator（priority=0, 兜底）

**支持的算子公式**：
- matmul：mm / addmm / bmm / linear（FLOPs = 2MNK）
- attention：scaled_dot_product_attention（FLOPs = 4BHSSD + 5BHSS）
- norm：layer_norm（5N）/ rms_norm（4N）/ add_rms_norm（6N）
- softmax：5N
- elementwise：add/mul/sub/div/neg/abs（1N）；silu/gelu/sigmoid/tanh（4N）
- embedding：FLOPs = 0（纯 HBM 读）
- shape ops：view/reshape/permute 等 FLOPs = 0
- 融合语义标签：rms_norm, layer_norm, add_rms_norm, flash_attn, sdpa, gated_mlp, mlp, moe_gate, rope 等
- 未知 op 兜底：1 flop/output_element

**测试**：
- `tests/test_simulator.py`：30 个测试全部通过（0.08s）
- compute-bound vs memory-bound 判断正确
- cache hit 验证
- 自定义 backend 优先级路由验证
- 端到端 simulate_graph 验证

## 下一步待办

1. **实现 transform/ pipeline（Phase 2 准备）**
   - 依赖：simulator/ ✅ + ir/ ✅ + hardware/ ✅
   - TransformContext + GraphPass ABC + TransformPipeline
   - 先实现基础骨架，再加 TP/EP 切分

2. **capture 层迁移（可选后续）**
   - 将 `export_all` 从 `nx.DiGraph` 改为直接接收 `OpGraph`

--- 2026-04-23T22:07:15+08:00 ---

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

--- 2026-04-23T22:28:16+08:00 ---

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

--- 2026-04-23T22:33:55+08:00 ---

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
---
Archived at: 2026-04-24
---

---
## 2026-04-24 Phase 4 全部完成

### 本轮完成

**变更 1：图路径调度分派（CRITICAL）**
- `python/zrt/transform/context.py`：`TrainingConfig` 新增 `vpp_chunks: int = 1`
- `python/zrt/transform/analysis/training.py:285-298`：替换硬编码 1F1B → 按 `pp_schedule` 分派（interleaved/dualpipev/dualpipe/1f1b）
- `python/zrt/transform/analysis/modeller.py:307-343`：unified 路径直接读取 `pipeline_metrics.step_time_ms`，不再重新计算
- `tests/training/test_graph_schedule.py`：5 个图路径测试全通过

**变更 2：Chrome Trace 报告集成（MEDIUM）**
- `python/zrt/report/chrome_trace.py`：`build_chrome_trace(timeline) -> dict`
- `python/zrt/report/summary.py`：`TrainingSummary.chrome_trace: dict | None = None`
- `python/zrt/training/cli.py`：`--trace <path>` 参数

**变更 3：Anchor YAML（LOW）**
- `tests/training/anchors/`：3 个 YAML + `test_anchors.py`，8 个测试全通过

**全量测试：252 passed, 0 failed**

---
## 2026-04-25 modeller.py 恢复

### 本轮完成

- 从提交 `a4f8c6c6cb5b3ed6f4e23ec6bc0a8794fd11ecd7` 检查并恢复 `python/zrt/transform/analysis/modeller.py` 的 Phase 0 图原生训练建模入口。
- 恢复 `python/zrt/transform/analysis/__init__.py` 对 `estimate_training`、`estimate_training_from_graphs`、`model_training`、`TrainingReport` 的导出。
- `stitch_fwd_bwd()` 元数据补充 `fwd_bwd_stitched=True`，使恢复后的 unified graph 路径可以触发 graph-native activation memory 分支。
- 放弃被中断轮次里的无关 `StageTime` 临时改动，保持本轮 diff 聚焦在 modeller 恢复。

### 验证

- `python -m py_compile python/zrt/transform/analysis/modeller.py python/zrt/transform/analysis/__init__.py python/zrt/training/compose/stage.py`
- `PYTHONPATH=python python -c "from python.zrt.transform.analysis import TrainingReport, estimate_training, estimate_training_from_graphs, model_training; print('analysis exports ok')"`
- `PYTHONPATH=python pytest tests/training/test_captured_graph_modelling.py -q`：15 passed

---
## 2026-04-25 modeller CLI 接线

### 本轮完成

- 修复 `python/zrt/cli.py::_run_training_modelling()` 仍绕过 `modeller.py` 的问题。
- `--train --hw` 路径现在直接调用 `estimate_training_from_graphs()`，把捕获到的 `train_forward` / `train_backward` 图交给 graph-native bridge，由该 bridge 执行 `stitch_fwd_bwd()` 和训练 pipeline。
- `estimate_training_from_graphs()` / `model_training()` 增加 `cp` 参数并传入 `ParallelConfig`，避免 CLI `--cp` 在 graph-native 建模路径中丢失。
- 新增 `tests/training/test_cli_modeller_wiring.py`，用 monkeypatch 锁定 CLI 必须委派到 `estimate_training_from_graphs()`。

### 验证

- `python -m py_compile python/zrt/cli.py python/zrt/transform/analysis/modeller.py tests/training/test_cli_modeller_wiring.py`
- `PYTHONPATH=python pytest tests/training/test_cli_modeller_wiring.py -q`：1 passed
- `PYTHONPATH=python pytest tests/training/test_captured_graph_modelling.py -q`：15 passed
- `PYTHONPATH=python pytest tests/training/test_graph_schedule.py -q`：5 passed

---
## 2026-04-25 modeller step_time 修复

### 本轮完成

- 修复 `estimate_training_from_graphs()` 重新计算 `step_time_ms` 的结构性错误。
- modeller 现在直接读取 `TrainingPipelinePass` 产出的 `pipeline_metrics.step_time_ms`、`mfu`、`warmup_steps`、`cooldown_steps`、`steady_steps`、`bubble_fraction`，保留 VPP/DualPipe/DP-in-bubble/overlap deduction 等调度修正。
- `estimate_training_from_graphs()` / `model_training()` 增加 `pp_schedule` 与 `vpp_chunks` 参数，允许 graph-native modeller 入口实际选择 VPP/DualPipe 类 schedule。
- 新增回归测试覆盖 DualPipe schedule adjustment，确保 modeller 不再退回 `(M + pp - 1) * per_stage_ms` 简化公式。

### 验证

- `python -m py_compile python/zrt/transform/analysis/modeller.py tests/training/test_captured_graph_modelling.py`
- `PYTHONPATH=python pytest tests/training/test_captured_graph_modelling.py -q`：16 passed
- `PYTHONPATH=python pytest tests/training/test_graph_schedule.py tests/training/test_cli_modeller_wiring.py -q`：6 passed
- `PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -q`：12 passed, 1 failed（既有 GPT-3 strict MFU calibration gap：estimated=0.2264 vs anchor=0.5200）

---
## 2026-04-25 P1 ZeroBubble Composer

### 本轮完成

- 在 spec 训练路径新增 `ZeroBubbleComposer` 并注册到 `PPSched.ZERO_BUBBLE`。
- `StageTime` 增加 `bwd_dx` / `bwd_dw`，`stage_time()` 从 `OpCost.dx_flops` / `dw_flops` 分离 backward critical path 与 weight-gradient work，同时保持既有 `bwd` 聚合字段兼容。
- graph-native `TrainingPipelinePass` 新增 `pp_schedule in {"zb", "zero_bubble"}` 分派，按 per-stage `flops_dw` 比例估算 `stage_timelines_bwd_dw`，用 dW work 缩减 ZeroBubble 暴露 bubble。
- DP-in-bubble 逻辑改为复用当前 schedule 计算出的 `bubble_us`，避免 ZeroBubble/VPP/DualPipe 后续又退回 1F1B bubble window。
- `python/zrt/training/compose/__init__.py` 导出 ZeroBubble 与现有 Composer 类。

### 验证

- `python -m py_compile python/zrt/training/compose/__init__.py python/zrt/training/compose/stage.py python/zrt/training/compose/pipeline.py python/zrt/transform/analysis/training.py tests/training/test_dualpipe.py tests/training/test_graph_schedule.py`
- `PYTHONPATH=python pytest tests/training/test_dualpipe.py tests/training/test_graph_schedule.py tests/training/test_search.py tests/training/test_interleaved_1f1b.py tests/training/test_pipeline_parallel.py -q`：41 passed
- `PYTHONPATH=python pytest tests/training/test_captured_graph_modelling.py tests/training/test_cli_modeller_wiring.py -q`：17 passed
- `PYTHONPATH=python pytest tests/training -q --ignore=tests/training/anchors`：172 passed

已知剩余风险：anchor suite 仍保留上一轮记录的 GPT-3 strict MFU calibration gap，未在 P1 中处理。

---
## 2026-04-25 P1 ZeroBubble 公式修正

### 本轮完成

- 修正 `ZeroBubbleComposer` 的 ZB-H bubble 公式：`bubble = (pp - 1) * max(t_stage - t_w, 0)`，去掉原先 `/ 2.0` 的 DualPipe-like 保守近似。
- `t_w` 改为取自同一个 bottleneck stage，而不是全局最大 `bwd_dw`，避免异构 stage 下跨 stage 抵扣。
- graph-native `TrainingPipelinePass` 同步修正 ZB-H 公式与 bottleneck-stage dW 选择。
- graph-native 路径在缺少 stage/phase 或 `flops_dw` 信息时写 debug log，并退化为无 dW bubble fill。
- 更新 `TrainingPipelinePass` 与 composer module stale docstring，不再标注为 1F1B-only。
- 增加 spec 与 graph-native 异构 stage 回归测试，并覆盖缺失 `flops_dw` 的 debug log。

### 验证

- `python -m py_compile python/zrt/training/compose/pipeline.py python/zrt/transform/analysis/training.py tests/training/test_dualpipe.py tests/training/test_graph_schedule.py`
- `PYTHONPATH=python pytest tests/training/test_dualpipe.py tests/training/test_graph_schedule.py -q`：17 passed
- `PYTHONPATH=python pytest tests/training/test_dualpipe.py tests/training/test_graph_schedule.py tests/training/test_search.py tests/training/test_interleaved_1f1b.py tests/training/test_pipeline_parallel.py -q`：44 passed
- `PYTHONPATH=python pytest tests/training/test_captured_graph_modelling.py tests/training/test_cli_modeller_wiring.py -q`：17 passed
- `PYTHONPATH=python pytest tests/training -q --ignore=tests/training/anchors`：175 passed
- `git diff --check`：passed

已知剩余风险：anchor suite 仍保留 GPT-3 strict MFU calibration gap，未在 P1 公式修正中处理。

---
## 2026-04-25 P2 Compressed Attention FLOPs

### 本轮完成

- `ModelSpec` 新增 `attn_compression_ratio`，默认 1.0，校验范围 `(0, 1]`，并由 YAML config loader 解析。
- spec 训练 FLOPs 模型在 `attn_core` 上应用 compressed attention ratio，缩放 forward 与基于 forward 推导的 backward dx FLOPs；per-op metadata 可以覆盖 model-level ratio。
- graph-native `TrainFlopsPass` 支持从 node annotation、node attrs、graph metadata 读取 `attn_compression_ratio`，用于 captured attention FLOPs；默认 dense ratio 1.0 保持兼容。
- 新增 spec-side 与 graph-native compressed attention 回归测试，覆盖 graph/model 默认值和 per-op/node override。

### 验证

- `python -m py_compile python/zrt/training/spec/model.py python/zrt/training/io/config_loader.py python/zrt/training/models/flops.py python/zrt/transform/analysis/flops_train.py tests/training/test_flops.py tests/training/test_transform_integration.py`
- `PYTHONPATH=python pytest tests/training/test_flops.py tests/training/test_transform_integration.py -q`：21 passed
- `PYTHONPATH=python pytest tests/training/test_flops.py tests/training/test_transform_integration.py tests/training/test_search.py tests/training/test_dualpipe.py tests/training/test_graph_schedule.py -q`：42 passed
- `PYTHONPATH=python pytest tests/training -q --ignore=tests/training/anchors`：179 passed
- `git diff --check`：passed

已知剩余风险：anchor suite 仍保留 GPT-3 strict MFU calibration gap；P2 未处理 P3 anchor calibration 或 P4 HFU metric。

---
## 2026-04-25 P2 Compressed Attention follow-up

### 本轮完成

- 修复 graph-native `TrainFlopsPass` attention FLOPs 维度推导：不再硬编码 `head_dim = 64`，改为优先读取 node attrs、graph metadata、4D Q tensor shape，再回退到 hidden/head_dim 推导。
- 在 spec-side `_attn_cost()` 为 `dx = 2.5 * fwd` 增加注释，明确 backward dx 继承 compressed forward 的 CSA/HCA ratio。
- graph-native 非法 `attn_compression_ratio` 改为 warning + dense fallback 1.0，warning 包含 node id，避免单个坏 annotation 中断整图 pass。
- 新增非 64 head_dim 维度推导测试，以及非法 compression ratio warning/fallback 测试。

### 验证

- `python -m py_compile python/zrt/transform/analysis/flops_train.py python/zrt/training/models/flops.py tests/training/test_transform_integration.py`
- `PYTHONPATH=python pytest tests/training/test_transform_integration.py tests/training/test_flops.py -q`：23 passed
- `PYTHONPATH=python pytest tests/training -q --ignore=tests/training/anchors`：181 passed
- `git diff --check`：passed

已知剩余风险：anchor suite 仍保留 GPT-3 strict MFU calibration gap；P2 follow-up 未处理 P3 anchor calibration 或 P4 HFU metric。

---
## 2026-04-25 P3 Anchor Integration

### 本轮完成

- 将 anchor integration test 从 placeholder 改为真实 `estimate()` 集成：每个 anchor YAML 都会加载为 `ModelSpec/SystemSpec/Strategy`，执行 `strategy.validate()` 与 `estimate()`，load/estimate 异常不再被吞掉。
- 对 `strict_mfu_check=True` 的 anchor 启用严格 MFU gate；GPT-3 175B strict anchor 现在通过，estimated MFU 0.5075 vs target 0.5200，误差 2.40%（tolerance 10%）。
- 修正 spec IR TP sharding：`attn_core` heads/KV metadata 与 attention tensor local shape 现在按 TP 缩放；LN/RoPE/SwiGLU/add 等 memory-bound activation bytes 也按 TP rank 计入，避免每个 TP rank 重复完整 activation traffic。
- PP=1 schedule 现在允许 DP gradient reduce 在 backward window 中重叠；`dp_overlap_in_bubble=False` 仍保留 fully exposed 行为。
- P3 计划中的 context-parallel Ulysses/Ring insertion coverage 已存在，本轮纳入 focused regression。

### 验证

- `python -m py_compile python/zrt/training/ir/shard.py python/zrt/training/compose/pipeline.py tests/training/test_1f1b.py tests/training/test_ir_dense.py`
- `PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -q -s`：13 passed
- `PYTHONPATH=python pytest tests/training/test_1f1b.py tests/training/test_ir_dense.py tests/training/test_context_parallel.py tests/training/test_search.py -q`：24 passed
- `PYTHONPATH=python pytest tests/training -q`：196 passed
- `git diff --check`：passed

已知剩余风险：DeepSeek-V3 与 LLaMA-3 anchors 仍为 calibration-mode，当前 MFU 偏差分别约 92.69% 与 41.77%；未启用 strict gate。P3 未处理 P4 HFU metric。

---
## 2026-04-25 P3 bytes_fwd integer follow-up

### 本轮完成

- 修复 TP sharding 后 `bytes_fwd` 由 int 变 float 的问题：`shard.py` 使用 integer floor division 保留 byte count 整数类型。
- 更新 `test_build_graph_with_tp`，同时断言 `bytes_fwd` 的数值和 `int` 类型，避免回归。

### 验证

- `python -m py_compile python/zrt/training/ir/shard.py tests/training/test_ir_dense.py`
- `PYTHONPATH=python pytest tests/training/test_ir_dense.py tests/training/anchors/test_anchors.py -q`：19 passed
- `PYTHONPATH=python pytest tests/training -q`：196 passed
- `git diff --check`：passed

已知剩余风险：DeepSeek-V3 与 LLaMA-3 anchors 仍为 calibration-mode；P4 HFU metric 尚未实现。
# Session Progress

## 当前阶段：P3 Anchor Integration follow-up — 已完成

## 最新变更（2026-04-25）

- P0 graph-native modeller 恢复、CLI 接线、`pipeline_metrics.step_time_ms` 直读修复均已完成。
- P1 新增 `python/zrt/training/compose/pipeline.py::ZeroBubbleComposer`，注册到 `PPSched.ZERO_BUBBLE`，schedule 名称为 `zb`。
- `python/zrt/training/compose/stage.py::StageTime` 增加 `bwd_dx` / `bwd_dw`，`stage_time()` 使用 `OpCost.dx_flops` / `dw_flops` 分离 input-gradient 与 weight-gradient 时间，同时保留既有 `bwd` 聚合字段。
- `python/zrt/transform/analysis/training.py::TrainingPipelinePass` 新增 graph-native `pp_schedule in {"zb", "zero_bubble"}` 分派；按 `flops_dw / (flops_dx + flops_dw)` 估算 `stage_timelines_bwd_dw`，用 dW work 缩减 ZeroBubble 暴露 bubble。
- DP-in-bubble 逻辑现在复用当前 schedule 的 `bubble_us`，避免 ZeroBubble/VPP/DualPipe 后续退回 1F1B bubble window。
- `python/zrt/training/compose/__init__.py` 导出 ZeroBubble 与现有 Composer 类。
- `tests/training/test_dualpipe.py` 增加 spec-side ZeroBubble 公式回归测试。
- `tests/training/test_graph_schedule.py` 增加 graph-native ZeroBubble dW split 回归测试。
- 已根据 review 修正 ZB-H bubble 公式：`bubble = (pp - 1) * max(t_stage - t_w, 0)`，不再使用 `/ 2.0` 保守近似。
- `t_w` 现在取自同一个 bottleneck stage，而不是全局最大 `bwd_dw`，避免异构 stage 下错误抵扣 bubble。
- graph-native 路径在缺少 `flops_dw` / stage phase 信息时写 debug log，并退化为“无 dW bubble fill”，避免静默给出误导性 ZeroBubble 结果。
- 已清理 `TrainingPipelinePass` 与 composer module 的 1F1B-only stale docstring。
- P2 新增 `ModelSpec.attn_compression_ratio`，默认 1.0，允许 YAML 配置解析；非法值会抛出 `ValueError`。
- spec 训练路径 `python/zrt/training/models/flops.py::_attn_cost()` 使用 `attn_compression_ratio` 缩放 attention-core forward FLOPs，并同步缩放基于 forward 的 backward dx FLOPs；per-op metadata 可覆盖 model 默认值。
- graph-native `TrainFlopsPass` 使用 node annotation → node attrs → graph metadata 的优先级读取 `attn_compression_ratio`，缩放 captured attention FLOPs；默认 1.0 保持 dense 行为。
- `tests/training/test_flops.py` 与 `tests/training/test_transform_integration.py` 增加 compressed attention / override 回归测试。
- P2 follow-up：graph-native attention 维度推导不再硬编码 `head_dim = 64`；优先读取 node attrs / graph metadata / 4D Q tensor shape，再回退到 hidden/head_dim 推导。
- `_attn_cost()` 明确注释 backward dx 是从 compressed forward 推导，因此继承 CSA/HCA ratio。
- graph-native 非法 `attn_compression_ratio` 不再中断整图 pass；现在写出包含 node id 的 warning，并回退到 dense ratio 1.0。
- P3 anchor harness 不再是 placeholder：`tests/training/anchors/test_anchors.py` 现在真实加载每个 anchor、运行 `estimate()`，并对 `strict_mfu_check=True` 的 anchor 执行严格 MFU gate；load/estimate 异常不再被吞掉。
- GPT-3 175B strict anchor 已校准通过：estimated MFU 0.5075 vs target 0.5200，误差 2.40%（tolerance 10%）。
- spec IR TP sharding 现在覆盖 `attn_core` heads / KV metadata 和 LN/RoPE/SwiGLU/add 等 memory-bound activation bytes，避免每个 TP rank 计入完整 attention/activation work。
- PP=1 路径支持 DP gradient reduce 在 backward window 中重叠；`dp_overlap_in_bubble=False` 仍保留 fully exposed DP allreduce/reduce-scatter 行为。
- P3 计划中的 context parallel insertion coverage 已存在于 `tests/training/test_context_parallel.py`，本轮纳入验证。
- P3 follow-up：TP sharding of `bytes_fwd` now preserves integer type via floor integer division; regression asserts `bytes_fwd` remains `int`.

## 本轮验证

```
python -m py_compile python/zrt/training/ir/shard.py python/zrt/training/compose/pipeline.py tests/training/test_1f1b.py tests/training/test_ir_dense.py
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -q -s
PYTHONPATH=python pytest tests/training/test_1f1b.py tests/training/test_ir_dense.py tests/training/test_context_parallel.py tests/training/test_search.py -q
python -m py_compile python/zrt/training/ir/shard.py tests/training/test_ir_dense.py
PYTHONPATH=python pytest tests/training/test_ir_dense.py tests/training/anchors/test_anchors.py -q
PYTHONPATH=python pytest tests/training -q
git diff --check
```

结果：anchor suite 13 passed；1F1B/IR/context/search focused regression 24 passed；bytes_fwd focused regression 19 passed；full training suite 196 passed；`git diff --check` passed。

已知剩余风险：DeepSeek-V3 与 LLaMA-3 anchors 仍为 calibration-mode，当前 MFU 偏差分别约 92.69% 与 41.77%；未启用 strict gate。P3 未处理 P4 HFU metric。

参考计划：`.omc/plans/proud-wishing-puzzle.md`

## 所有子项完成状态

| 子项 | 状态 | 说明 |
|------|------|------|
| P0 graph-native path | ✅ 完成 | modeller 恢复、stitch metadata、CLI 接线、step_time 直读 |
| P1 ZeroBubble Composer | ✅ 完成 | `ZeroBubbleComposer` + graph-native `zb` dispatch |
| P2 compressed attention | ✅ 完成 | `attn_compression_ratio` 接入 spec 与 graph-native attention FLOPs |
| P3 anchor integration/calibration | ✅ 完成 | GPT-3 strict MFU gate 通过；其他 anchors 保持 calibration-mode |
| P4 HFU metric | ⏳ 待办 | MFU/HFU 区分尚未实现 |

## 本轮修改文件

- `python/zrt/training/ir/shard.py`
- `python/zrt/training/compose/pipeline.py`
- `tests/training/anchors/test_anchors.py`
- `tests/training/test_1f1b.py`
- `tests/training/test_ir_dense.py`
- `SESSION_HISTORY.md`
- `SESSION_PROGRESS.md`

## 历史里程碑摘要

- Phase 0：`stitch_fwd_bwd()` 前向+反向图拼接；graph-native modeller 入口恢复。
- Phase 1：步骤时间公式修复 + 激活内存 + FLOPs 修复。
- Phase 2：`PipelineParallelPass` + 逐阶段 `DAGScheduler` + 1F1B 公式。
- Phase 3：`context_parallel.py` / `data_parallel.py` / CoC/MC2 overlap 注解。
- Phase 4：spec 路径 Composer、Chrome Trace、图路径调度分派、EP 不均衡、搜索/Pareto、Anchor 验证。
- P1 follow-up：ZeroBubble Composer 已接入 spec 与 graph-native training pipeline。
- P2 follow-up：Compressed attention FLOPs ratio 已接入 spec 与 graph-native attention cost。
- P3 follow-up：Anchor estimate integration 已启用 strict gate，GPT-3 175B strict anchor 通过。
# Session Progress

## 当前阶段：P4 HFU Metric — 已完成

## 最新变更（2026-04-25）

- P0 graph-native modeller 恢复、CLI 接线、`pipeline_metrics.step_time_ms` 直读修复均已完成。
- P1 新增 `python/zrt/training/compose/pipeline.py::ZeroBubbleComposer`，注册到 `PPSched.ZERO_BUBBLE`，schedule 名称为 `zb`。
- `python/zrt/training/compose/stage.py::StageTime` 增加 `bwd_dx` / `bwd_dw`，`stage_time()` 使用 `OpCost.dx_flops` / `dw_flops` 分离 input-gradient 与 weight-gradient 时间，同时保留既有 `bwd` 聚合字段。
- `python/zrt/transform/analysis/training.py::TrainingPipelinePass` 新增 graph-native `pp_schedule in {"zb", "zero_bubble"}` 分派；按 `flops_dw / (flops_dx + flops_dw)` 估算 `stage_timelines_bwd_dw`，用 dW work 缩减 ZeroBubble 暴露 bubble。
- DP-in-bubble 逻辑现在复用当前 schedule 的 `bubble_us`，避免 ZeroBubble/VPP/DualPipe 后续退回 1F1B bubble window。
- `python/zrt/training/compose/__init__.py` 导出 ZeroBubble 与现有 Composer 类。
- `tests/training/test_dualpipe.py` 增加 spec-side ZeroBubble 公式回归测试。
- `tests/training/test_graph_schedule.py` 增加 graph-native ZeroBubble dW split 回归测试。
- 已根据 review 修正 ZB-H bubble 公式：`bubble = (pp - 1) * max(t_stage - t_w, 0)`，不再使用 `/ 2.0` 保守近似。
- `t_w` 现在取自同一个 bottleneck stage，而不是全局最大 `bwd_dw`，避免异构 stage 下错误抵扣 bubble。
- graph-native 路径在缺少 `flops_dw` / stage phase 信息时写 debug log，并退化为"无 dW bubble fill"，避免静默给出误导性 ZeroBubble 结果。
- 已清理 `TrainingPipelinePass` 与 composer module 的 1F1B-only stale docstring。
- P2 新增 `ModelSpec.attn_compression_ratio`，默认 1.0，允许 YAML 配置解析；非法值会抛出 `ValueError`。
- spec 训练路径 `python/zrt/training/models/flops.py::_attn_cost()` 使用 `attn_compression_ratio` 缩放 attention-core forward FLOPs，并同步缩放基于 forward 的 backward dx FLOPs；per-op metadata 可覆盖 model 默认值。
- graph-native `TrainFlopsPass` 使用 node annotation → node attrs → graph metadata 的优先级读取 `attn_compression_ratio`，缩放 captured attention FLOPs；默认 1.0 保持 dense 行为。
- `tests/training/test_flops.py` 与 `tests/training/test_transform_integration.py` 增加 compressed attention / override 回归测试。
- P2 follow-up：graph-native attention 维度推导不再硬编码 `head_dim = 64`；优先读取 node attrs / graph metadata / 4D Q tensor shape，再回退到 hidden/head_dim 推导。
- `_attn_cost()` 明确注释 backward dx 是从 compressed forward 推导，因此继承 CSA/HCA ratio。
- graph-native 非法 `attn_compression_ratio` 不再中断整图 pass；现在写出包含 node id 的 warning，并回退到 dense ratio 1.0。
- P3 anchor harness 不再是 placeholder：`tests/training/anchors/test_anchors.py` 现在真实加载每个 anchor、运行 `estimate()`，并对 `strict_mfu_check=True` 的 anchor 执行严格 MFU gate；load/estimate 异常不再被吞掉。
- GPT-3 175B strict anchor 已校准通过：estimated MFU 0.5075 vs target 0.5200，误差 2.40%（tolerance 10%）。
- spec IR TP sharding 现在覆盖 `attn_core` heads / KV metadata 和 LN/RoPE/SwiGLU/add 等 memory-bound activation bytes，避免每个 TP rank 计入完整 attention/activation work。
- PP=1 路径支持 DP gradient reduce 在 backward window 中重叠；`dp_overlap_in_bubble=False` 仍保留 fully exposed DP allreduce/reduce-scatter 行为。
- P3 计划中的 context parallel insertion coverage 已存在于 `tests/training/test_context_parallel.py`，本轮纳入验证。
- P3 follow-up：TP sharding of `bytes_fwd` now preserves integer type via floor integer division; regression asserts `bytes_fwd` remains `int`.
- **P4 HFU metric**：新增 `hfu` 字段到 `StepResult`（spec 路径）、`PipelineStepMetrics`（graph-native 路径）、`Report`（estimator）。
- `python/zrt/training/models/flops.py` 新增 `recompute_overhead_flops()` 和 `_op_recompute_categories()`，根据 `RecomputePolicy.per_layer` 计算 recompute 额外 FLOPs。
- `python/zrt/training/compose/pipeline.py` 新增 `compute_hfu()`，HFU = (model_flops + recompute_overhead) / (peak * step_time)。
- graph-native `TrainingFlopsPass` 新增 `recompute_flops` 到 metadata，从 recomputed nodes 的 `flops_fwd // 2` 提取 recompute 额外 FLOPs。
- graph-native `TrainingPipelinePass` MFU 计算修正：MFU 使用 `training_flops - recompute_flops`（不含 recompute），HFU 使用完整 `training_flops`。
- `tests/training/test_flops.py` 新增 4 个 HFU 回归测试：无 recompute 时 HFU==MFU、selective recompute 时 HFU>MFU、默认无额外 FLOPs、full recompute 覆盖所有 compute-bound ops。

## 本轮验证

```
python -m py_compile python/zrt/training/models/flops.py python/zrt/training/compose/pipeline.py python/zrt/training/search/estimator.py python/zrt/transform/analysis/training.py
PYTHONPATH=python pytest tests/training/test_flops.py -v
PYTHONPATH=python pytest tests/training/ -q
git diff --check
```

结果：flops 13 passed；full training suite 200 passed（含 4 个新增 HFU 测试）；`git diff --check` passed。

## 所有子项完成状态

| 子项 | 状态 | 说明 |
|------|------|------|
| P0 graph-native path | ✅ 完成 | modeller 恢复、stitch metadata、CLI 接线、step_time 直读 |
| P1 ZeroBubble Composer | ✅ 完成 | `ZeroBubbleComposer` + graph-native `zb` dispatch |
| P2 compressed attention | ✅ 完成 | `attn_compression_ratio` 接入 spec 与 graph-native attention FLOPs |
| P3 anchor integration/calibration | ✅ 完成 | GPT-3 strict MFU gate 通过；其他 anchors 保持 calibration-mode |
| P4 HFU metric | ✅ 完成 | MFU/HFU 区分已实现；spec + graph-native 双路径；4 个回归测试 |

## 本轮修改文件

- `python/zrt/training/models/flops.py`
- `python/zrt/training/compose/pipeline.py`
- `python/zrt/training/search/estimator.py`
- `python/zrt/transform/analysis/training.py`
- `tests/training/test_flops.py`
- `SESSION_HISTORY.md`
- `SESSION_PROGRESS.md`

## 历史里程碑摘要

- Phase 0：`stitch_fwd_bwd()` 前向+反向图拼接；graph-native modeller 入口恢复。
- Phase 1：步骤时间公式修复 + 激活内存 + FLOPs 修复。
- Phase 2：`PipelineParallelPass` + 逐阶段 `DAGScheduler` + 1F1B 公式。
- Phase 3：`context_parallel.py` / `data_parallel.py` / CoC/MC2 overlap 注解。
- Phase 4：spec 路径 Composer、Chrome Trace、图路径调度分派、EP 不均衡、搜索/Pareto、Anchor 验证。
- P1 follow-up：ZeroBubble Composer 已接入 spec 与 graph-native training pipeline。
- P2 follow-up：Compressed attention FLOPs ratio 已接入 spec 与 graph-native attention cost。
- P3 follow-up：Anchor estimate integration 已启用 strict gate，GPT-3 175B strict anchor 通过。
- P4 follow-up：HFU metric 已实现，spec + graph-native 双路径，recompute overhead 从 RecomputePolicy 推导。
