# ARCHITECTURE.md 设计目标盘点

基于 [`ARCHITECTURE.md`](D:/workspace/claude/modeling/ARCHITECTURE.md) 对当前项目代码进行盘点，目标是回答三个问题：

1. 文档中的架构目标是什么。
2. 当前仓库已经实现到什么程度。
3. 主要缺口和后续优先级是什么。

盘点时间：2026-04-18  
盘点范围：`python/zrt/*`、`tests/*`、`README.md`、`ARCHITECTURE.md`

---

## 总体结论

当前项目已经完成了从“LLM 算子抓图工具”向“LLM 性能建模框架”迁移的大半主干，但还没有达到 [`ARCHITECTURE.md`](D:/workspace/claude/modeling/ARCHITECTURE.md) 描述的完整 V2 状态。

更准确地说：

- `Foundation Layer` 和 `Core Layer` 的主脊梁已经搭起来了。
- `Application Layer` 基本停留在设计层，代码里尚未形成统一编排。
- `Extension Layer` 基本未开始。
- 当前最成熟的主线是：
  `graph capture -> OpGraph IR -> transform -> roofline simulate -> DAG schedule -> summary/report`

按文档路线图判断，当前大致处于 `Phase 1.5 ~ Phase 2.0`，而不是完整 V2。

---

## 现状概览

### 设计理念对齐

| 设计理念 | 当前状态 |
|---|---|
| 图驱动 | `OpGraph` 已成为新主线中枢 |
| 硬件 × 软件栈正交 | 硬件已实现，软件栈体系缺失 |
| 先切后融 | `split -> fuse -> optim -> analyze` 已固定 |
| 显存一等公民 | `MemoryModel` 缺失 |
| 仿真器可插拔 | `SimulatorHub` 已有，但仅 roofline 后端 |
| 无卡运行 | fake/meta tensor 主线已具备 |
| 模块独立可服务化 | 方向存在，但应用层未形成 |

### 分层状态

| 层级 | 现状 | 结论 |
|---|---|---|
| Foundation | [`python/zrt/ir`](D:/workspace/claude/modeling/python/zrt/ir)、[`python/zrt/hardware`](D:/workspace/claude/modeling/python/zrt/hardware) 已成型 | 基本完成 |
| Core | 抓图、变换、roofline、调度、summary 已可用 | 主干已成型 |
| Application | 无 `Orchestrator`、无统一 CLI、无 search/compare/bottleneck | 基本缺失 |
| Extension | serving、training、calibration、API、高精度后端均未落地 | 尚未开始 |

### 已实现主线

- 图抓取：[`python/zrt/graph/main.py`](D:/workspace/claude/modeling/python/zrt/graph/main.py)、[`python/zrt/graph/dispatch.py`](D:/workspace/claude/modeling/python/zrt/graph/dispatch.py)、[`python/zrt/graph/tracker.py`](D:/workspace/claude/modeling/python/zrt/graph/tracker.py)
- 统一 IR：[`python/zrt/ir`](D:/workspace/claude/modeling/python/zrt/ir)
- 图变换：[`python/zrt/transform/pipeline.py`](D:/workspace/claude/modeling/python/zrt/transform/pipeline.py) 及 `parallel/fusion/optim/analysis`
- 仿真：[`python/zrt/simulator/hub.py`](D:/workspace/claude/modeling/python/zrt/simulator/hub.py)、[`python/zrt/simulator/backends/roofline.py`](D:/workspace/claude/modeling/python/zrt/simulator/backends/roofline.py)
- 调度：[`python/zrt/executor/scheduler.py`](D:/workspace/claude/modeling/python/zrt/executor/scheduler.py)
- 汇总：[`python/zrt/report/summary.py`](D:/workspace/claude/modeling/python/zrt/report/summary.py)

### 主要未实现项

- `MemoryModel / MemoryBudget`
- `SoftwareStack` 及 `stacks/*`
- `CommModel`
- `Orchestrator / RunConfig / app/* / 统一 CLI`
- `profile_db / regression / tiling_sim / calibration`

---

## 路线图判断

| Phase | 当前判断 |
|---|---|
| Phase 1 | 大体完成，但缺 `MemoryModel` 和统一 `predict` 封装 |
| Phase 2 | 做到一半左右，TP/EP、融合、多流调度已有基础，但缺 `CommModel` 和完整 overlap / timeline 能力 |
| Phase 3 | 刚起步，SP/PP、EPLB/MTP、search/compare/bottleneck 基本未落地 |
| Phase 4 | 基本未开始 |
| Phase 5 | 基本未开始 |

---

## 最关键的 6 个缺口

### 1. `MemoryModel` 缺失

这是最关键的架构缺口之一。  
没有它，文档中的：

- 可行性判定
- 快速剪枝
- 搜索引擎
- `predict` 的前置检查

都无法按设计成立。

### 2. `Orchestrator` 缺失

当前仓库里有不少可用模块，但没有统一编排层。  
结果就是：

- 能力存在
- 产品形态不存在

### 3. `SoftwareStack` 缺失

文档强调“硬件 × 软件栈正交”，但当前只有硬件这一半真正成型。  
这会直接影响：

- 融合规则
- 平台算子映射
- 优化能力声明

### 4. `CommModel` 缺失

现在的多卡链路只有“通信节点”，没有“通信性能模型”。  
这会限制：

- 多卡 latency 预测可信度
- overlap 分析质量
- TP/EP 调优能力

### 5. 新旧链路尚未收口

当前同时存在：

- 旧 `graph/*` 导出与融合链
- 新 `ir/transform/simulator/report/*` 建模链

这说明迁移仍在中途，后续维护成本会持续上升。

### 6. `Application Layer` 未形成

这是从“研究原型”走向“完整系统”的最后一层。  
不补这一层，项目始终更像内部引擎集合，而不是面向使用者的完整工具。

---

## 建议的补齐顺序

如果目标是尽快对齐 [`ARCHITECTURE.md`](D:/workspace/claude/modeling/ARCHITECTURE.md)，建议按这个顺序推进：

1. 补 `MemoryModel`
2. 补 `Orchestrator + RunConfig`
3. 补 `SoftwareStack + generic/mindie/vllm`
4. 补 `CommModel`
5. 收口旧 `graph` 导出链到统一 `report` 体系
6. 再做 `search / compare / bottleneck`
7. 最后补 `profile_db / regression / calibration`

前四项决定系统是否能从“模块集合”变成真正的端到端工具；后三项属于精度和产品增强，应放在主干稳定之后。

---

## 最终判断

当前项目已经具备了 V2 架构的核心骨架，尤其是 `OpGraph + Transform + Roofline + Schedule + Summary` 这条主线；但它还不是 [`ARCHITECTURE.md`](D:/workspace/claude/modeling/ARCHITECTURE.md) 定义的完整系统，离“统一编排、可搜索、可对比、可校准”的目标仍有明显距离。

从工程状态看，它更像：

- 不是早期 PoC
- 也还不是完整产品
- 而是一个“主干已成型、应用层待补齐”的中间阶段架构

---

## 补充风险

### 1. `MemoryModel` 不应被当作硬剪枝依据

文档里 `MemoryModel` 仍是经验公式，但配置寻优又依赖它快速淘汰候选。  
如果把这一步当成硬约束，容易误剪可行配置，或者保留实际会 OOM 的配置。

更合理的定位是“保守预筛”，并明确误差边界与二次校验机制。

### 2. 当前“寻优”更像受限枚举，不是真正优化器

现在的流程本质是“枚举 -> 剪枝 -> 截断 -> 排序”，会直接依赖枚举顺序，不能保证找到最优配置。  
如果继续保留这个设计，建议明确写成“预算受限的启发式搜索”。

### 3. `DAGScheduler` 的模型不足以支撑多卡精度目标

现有调度器覆盖 DAG 依赖、compute/comm 分流和基础 overlap，但多卡场景还缺 rank/device 维度、collective 同步窗口、pipeline bubble、跨设备资源竞争和链路占用建模。  
如果不补这些，TP / EP / PP 的 latency 预测会偏乐观。

### 4. `SimulatorHub` 缓存键维度不完整

当前 `content-hash` 主要围绕算子形状、属性和硬件。  
这不足以区分不同软件栈、仿真后端、校准参数和并行上下文。后续一旦引入这些能力，缓存误命中的风险会明显上升。

### 5. `TrainingEstimator` 应明确标注为粗估能力

训练估算目前是基于推理结果乘经验系数。  
这适合作为快速估算工具，不适合默认被理解为训练侧高可信建模能力。

建议明确其适用范围、误差预期，以及“不宜用于高精度训练并行决策”。
