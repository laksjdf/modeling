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
