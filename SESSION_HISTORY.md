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
