# Session Progress

## 当前阶段：P4 HFU Metric follow-up — 已完成

## 最新变更（2026-04-25）

- P0–P3 所有子项已完成（详见 SESSION_HISTORY.md）。
- **P4 HFU metric**：新增 `hfu` 字段到 `StepResult`（spec 路径）、`PipelineStepMetrics`（graph-native 路径）、`Report`（estimator）。
- `python/zrt/training/models/flops.py` 新增 `recompute_overhead_flops()` 和 `_op_recompute_categories()`，根据 `RecomputePolicy.per_layer` 计算 recompute 额外 FLOPs；**已修正为按 layer_kind 限定作用域**，避免 mixed dense/MoE 时错误跨层计数。
- `python/zrt/training/compose/pipeline.py` 新增 `compute_hfu()`，HFU = (model_flops + recompute_overhead) / (peak * step_time)。
- `python/zrt/training/compose/stage.py::_recompute_time()` **已修正**：从仅处理 `"full"` 扩展为处理 selective categories（`"attn"`, `"ffn_swiglu"`, `"ln"`），确保 selective recompute 正确增加 step time，避免 HFU 虚高。
- graph-native `TrainingFlopsPass` 新增 `recompute_flops` 到 metadata，从 recomputed nodes 的 `flops_fwd // 2` 提取 recompute 额外 FLOPs。
- graph-native `TrainingPipelinePass` MFU 计算修正：MFU 使用 `training_flops - recompute_flops`（不含 recompute），HFU 使用完整 `training_flops`。
- **`python/zrt/transform/analysis/modeller.py`**：`TrainingReport` 新增 `hfu` 字段；`to_dict()` 和 `summary()` 输出 HFU；`estimate_training()` 和 `estimate_training_from_graphs()` 从 `PipelineStepMetrics` 传播 HFU。
- **`python/zrt/training/search/report.py`**：`report_to_dict()` 和 `report_summary()` 输出 HFU。
- `tests/training/test_flops.py` 新增 6 个 HFU 回归测试：无 recompute 时 HFU==MFU、selective recompute 时 HFU>MFU、默认无额外 FLOPs、full recompute 覆盖所有 compute-bound ops、selective recompute 增加 step time、layer_kind 限定 recompute 作用域。

## 本轮验证

```
python -m py_compile python/zrt/training/models/flops.py python/zrt/training/compose/stage.py python/zrt/training/compose/pipeline.py python/zrt/transform/analysis/training.py python/zrt/transform/analysis/modeller.py python/zrt/training/search/report.py python/zrt/training/search/estimator.py
PYTHONPATH=python pytest tests/training/test_flops.py -v
PYTHONPATH=python pytest tests/training/ -q
git diff --check
```

结果：flops 15 passed；full training suite 202 passed（含 6 个新增 HFU 测试）；`git diff --check` passed。

## 所有子项完成状态

| 子项 | 状态 | 说明 |
|------|------|------|
| P0 graph-native path | ✅ 完成 | modeller 恢复、stitch metadata、CLI 接线、step_time 直读 |
| P1 ZeroBubble Composer | ✅ 完成 | `ZeroBubbleComposer` + graph-native `zb` dispatch |
| P2 compressed attention | ✅ 完成 | `attn_compression_ratio` 接入 spec 与 graph-native attention FLOPs |
| P3 anchor integration/calibration | ✅ 完成 | GPT-3 strict MFU gate 通过；其他 anchors 保持 calibration-mode |
| P4 HFU metric | ✅ 完成 | MFU/HFU 区分已实现；spec + graph-native 双路径；6 个回归测试 |

## 本轮修改文件

- `python/zrt/training/models/flops.py`
- `python/zrt/training/compose/pipeline.py`
- `python/zrt/training/compose/stage.py`
- `python/zrt/training/search/estimator.py`
- `python/zrt/training/search/report.py`
- `python/zrt/transform/analysis/training.py`
- `python/zrt/transform/analysis/modeller.py`
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
- P4 follow-up：HFU metric 已实现，recompute overhead 从 RecomputePolicy 按 layer_kind 推导；selective recompute 正确影响 step time；modeller/report 输出管线完整。
