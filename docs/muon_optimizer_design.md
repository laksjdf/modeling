# Muon 优化器建模特性开发设计文档

**项目**: ZRT-Sim 训练性能建模  
**版本**: v1.0  
**日期**: 2026-04-28  
**作者**: yetianqi

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [理论基础：Adam vs Muon](#2-理论基础adam-vs-muon)
3. [ZeRO 分布式下的通信差异](#3-zero-分布式下的通信差异)
4. [现状分析](#4-现状分析)
5. [设计规格](#5-设计规格)
   - 5.1 [数据模型扩展](#51-数据模型扩展)
   - 5.2 [内存模型](#52-内存模型)
   - 5.3 [FLOPs 模型](#53-flops-模型)
   - 5.4 [通信模型](#54-通信模型)
   - 5.5 [Transform Pipeline 集成](#55-transform-pipeline-集成)
   - 5.6 [MoE 模型的混合优化器策略](#56-moe-模型的混合优化器策略)
6. [接口变更](#6-接口变更)
7. [开发计划](#7-开发计划)
8. [测试策略](#8-测试策略)

---

## 1. 背景与动机

### 1.1 Muon 在生产中的应用

DeepSeek-V4 (DSV4) 技术报告 §3.5.1 披露：其训练框架对绝大多数 Transformer 参数矩阵采用 **Muon 优化器**（Momentum + Newton-Schulz 正交化），仅对 Embedding、LM Head 以及 MoE 路由矩阵继续沿用 Adam。实测 Muon 相较 Adam 可提升 ~1.8% MFU（受益于更少的优化器状态内存，从而允许更大的 micro-batch 或更少的 activation recompute）。

Moonshot（Kimi）团队也公开了 Muon 与 ZeRO 结合的工程方案，揭示了 Muon 在分布式场景下与 Adam 有本质不同的通信模式。

### 1.2 当前建模缺口

ZRT-Sim 目前已有 `OptKind.MUON` 枚举，但以下建模均为占位符：

| 建模维度 | 现状 | 问题 |
|---------|------|------|
| 优化器状态内存 | `2.1 × P × 4B` 近似 | 未区分 Adam/Muon 参数覆盖范围差异 |
| 优化器 FLOPs | `16 FLOPs/param` 固定值 | 未建模 NS 迭代的矩阵乘法开销 |
| ZeRO 通信 | 仅梯度 AllReduce | 未建模 Muon 特有的动量 AllGather + ReduceScatter |
| 参数覆盖范围 | 全部参数 | 未区分 Muon 参数 vs Adam 参数（embed/head）|

本文档给出完整设计以填补以上缺口。

---

## 2. 理论基础：Adam vs Muon

### 2.1 算法对比

**Adam 更新规则**（逐元素，参数无关性）：

```
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t          # 一阶矩（动量）
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²          # 二阶矩（方差）
m̂_t = m_t / (1 - β₁ᵗ)                          # bias correction
v̂_t = v_t / (1 - β₂ᵗ)
W_t = W_{t-1} - lr · m̂_t / (√v̂_t + ε)
```

**Muon 更新规则**（矩阵整体正交化）：

```
M_t = β · M_{t-1} + G_t                         # 动量（与 Adam m 类似）
M̂_t = NewtonSchulzK(M_t, K)                        # K步 NS 迭代正交化
W_t = W_{t-1} - lr · M̂_t
```

其中 Newton-Schulz 迭代（K步 Zolo-PD 近似，K 可配置）：

```
for k in range(K):           # K 默认为 5；DeepSeek-V4 官方报告 §2.4 使用 K=10
    A = Xₖᵀ Xₖ              # n×n 矩阵
    Xₖ₊₁ = α·Xₖ + β·Xₖ·A   # 正交化更新
```

### 2.2 关键差异：参数独立性

| 维度 | Adam | Muon |
|------|------|------|
| 状态变量 | m（动量）+ v（方差）| M（动量矩阵）|
| 更新粒度 | 逐元素，完全独立 | 矩阵整体（NS 需要全行交互）|
| 参数类型要求 | 任意形状（含1D bias、embedding）| 必须为 2D 矩阵（1D 参数退化为 SGD）|
| 理论依据 | 自适应学习率 | Nesterov + 谱范数约束 |

**这一根本差异导致 ZeRO 分片后的通信模式完全不同。**

### 2.3 优化器状态内存对比

设模型参数量为 P，master copy 精度为 FP32（4B）：

| 优化器 | 状态组成 | 每参数字节数 | 相对 Adam |
|--------|---------|------------|----------|
| Adam | master + m + v | 3 × 4B = **12 B** | baseline |
| AdamW | master + m + v | 3 × 4B = **12 B** | 同上 |
| Muon | master + M（无 v）| 2 × 4B = **8 B** | **-33%** |

> **设计注意**：Muon 的内存节省（-4B/param）在数百亿参数量级下效果显著。
> DSV4 660B 总参数、Muon 覆盖约 80% → 节省约 211 GB 优化器状态。

---

## 3. ZeRO 分布式下的通信差异

### 3.1 Adam + ZeRO-1

ZeRO-1 将优化器状态均匀分片到各 DP rank，梯度 AllReduce 后各 rank 独立更新自己负责的参数分片。

```
设 P_rank = P / DP（每 rank 负责的参数量）

每 rank 通信量：
  梯度 AllReduce = 2 × (DP-1)/DP × P_rank × grad_bytes   # Ring-AllReduce
  优化器步骤：无额外通信（m/v 逐元素，rank 内独立计算）

总额外通信（优化器步骤）= 0
```

Adam 是**逐元素**操作，分片后各 rank 完全独立，**不需要跨 rank 通信**。

### 3.2 Muon + ZeRO-1（核心差异）

对于形状为 `W ∈ R^{m×n}` 的权重矩阵，其动量矩阵 `M ∈ R^{m×n}` 按行分片：

```
GPU-0: M[0:m/DP, :] ∈ R^{m/DP × n}
GPU-1: M[m/DP:2m/DP, :] ∈ R^{m/DP × n}
...
```

NS 迭代需要计算 `Xₖᵀ Xₖ`，其中 `(XₖXₖᵀ)ᵢⱼ = Σₖ Xᵢₖ Xⱼₖ`：
- 当 i 属于 GPU-0，j 属于 GPU-1 时，需要跨 GPU 数据 → **必须 AllGather**

**Muon + ZeRO-1 的完整工作流（每个参数矩阵）：**

```
Step 1 — AllGather 动量分片
  每 rank 广播自己的 M 行分片 → 每 rank 获得完整 M ∈ R^{m×n}
  通信量（per rank）= (DP-1)/DP × m×n × 4B

Step 2 — 本地 NS 迭代（K步）
  每 rank 独立执行完整正交化 → 得到 M̂ ∈ R^{m×n}
  计算量 = K × (2mn² + 2mn²) = K×4mn²   # 见 §3.3；K 由配置决定

Step 3 — ReduceScatter 结果（或直接按行切片）
  每 rank 只取自己负责的行更新 W
  通信量（per rank）= (DP-1)/DP × m×n × 4B（若需要 RS）
  或直接 slice 无通信（若 Step 2 各 rank 结果一致）
```

**通信对比总结：**

| 场景 | 梯度通信 | 优化器步骤额外通信 |
|------|---------|----------------|
| Adam + ZeRO-1 | 2×(DP-1)/DP × P_rank × 4B（AR）| **0** |
| Muon + ZeRO-1 | 2×(DP-1)/DP × P_rank × 4B（AR）| AllGather: (DP-1)/DP × P_muon × 4B |
| Muon + ZeRO-2 | ReduceScatter 梯度 | AllGather: (DP-1)/DP × P_muon × 4B |
| Muon + ZeRO-3 | ReduceScatter 梯度 | AllGather: (DP-1)/DP × P_muon × 4B（更大）|

> **关键结论**：Muon 在所有 ZeRO 阶段都引入额外的 AllGather，通信量与 Adam 梯度 AllReduce 相当。

### 3.3 Moonshot 优化方案：轮转分工

为避免所有 GPU 重复执行相同的 NS 迭代（纯计算冗余），Moonshot 团队采用分工策略：

```
每个 DP rank 负责不同参数矩阵的正交化：
  GPU-0: AllGather M_W1 → NS(M_W1) → ReduceScatter → 更新 W1
  GPU-1: AllGather M_W2 → NS(M_W2) → ReduceScatter → 更新 W2
  ...
```

这将 NS 计算均匀分摊到各 rank，同时 AllGather + ReduceScatter 仍然不可省略。

**建模影响**：ZRT-Sim 需要对 Moonshot 优化方案建立两套模型：
1. **Naïve 方案**（无轮转）：每个 rank 做全部 NS 迭代，计算开销 × DP
2. **Optimized 方案**（轮转）：每个 rank 只做 1/DP 的 NS 迭代

---

## 4. 现状分析

> **2026-05-15 更新**：P1–P6 全部已修复并通过测试（38/38 单元测试通过）。
> 本节保留为历史记录，标注各问题的修复状态。

### 4.1 代码路径（当前状态）

```
python/zrt/training/spec/strategy.py
  └── OptKind(Enum): ADAM | MUON         ✅ 枚举已定义
  └── MuonConfig(dataclass)              ✅ ns_steps / muon_param_fraction / rotation 等字段

python/zrt/training/models/memory.py
  └── _optimizer_state_bytes()           ✅ int(P × (12 - f_muon × 4))，混合参数正确分离

python/zrt/transform/training/optimizer.py
  └── OptimizerPass._opt_state_bytes()   ✅ int(params × (12 - f_muon × 4))，与 memory.py 一致
  └── OptimizerPass._opt_step_flops()    ✅ 调用 muon_optimizer_step_flops()，含 NS 矩阵乘法建模

python/zrt/transform/analysis/training.py
  └── TrainingMemoryPass                 ✅ 使用 muon_param_fraction 正确分离 Muon/Adam 参数
  └── TrainingPipelinePass               ✅ optimizer step time 已计入 step_time

python/zrt/training/models/comm.py
  └── total_comm_time()                  ✅ 调用 optimizer_comm_time()，建模 Muon AG+RS

python/zrt/training/compose/schedules.py
  └── _compute_optimizer_time()          ✅ EP 和 ZeRO-1/2 分片均已正确应用（a4a6d25）
  └── StepResult.optimizer_comm          ✅ Muon AG+RS 通信单独列出
```

### 4.2 问题清单（历史）

| # | 位置 | 原问题 | 状态 |
|---|------|--------|------|
| P1 | `memory.py:_optimizer_state_bytes` | 2.1 系数无根据 | ✅ 修复：`int(P × (12 - f_muon × 4))` |
| P2 | `optimizer.py:_opt_state_bytes` | `6B/param` 计算错误 | ✅ 修复：`int(params × (12 - f_muon × 4))` |
| P3 | `optimizer.py:_opt_step_flops` | `16 FLOPs/param` 无 NS 建模 | ✅ 修复：接入 `muon_optimizer_step_flops()` |
| P4 | `comm.py:total_comm_time` | Muon ZeRO AllGather 未建模 | ✅ 修复：新增 `optimizer_comm_time()` |
| P5 | 全局 | 无 Muon 参数覆盖范围配置 | ✅ 修复：`MuonConfig.muon_param_fraction` |
| P6 | `schedules.py:StepResult` | optimizer step time 未计入 step_time | ✅ 修复：`step_time += optimizer_time + optimizer_comm` |

### 4.3 验证结果（2026-05-15）

对 LLaMA-3 70B（TP=4, PP=4, DP=4, ZeRO-2, H100×64）运行内存验证，所有指标通过（≤10% 误差）：

| 指标 | 期望值 | 实测误差 | 结论 |
|------|--------|---------|------|
| 优化器状态内存（Muon, ZeRO-2） | `P_rank × 8.6B / DP` | 0.0% | ✅ |
| 峰值阶段（激活主导，peak = backward） | `peak_overall == peak_backward` | — | ✅ |
| 梯度内存（FP32, ZeRO-2） | `P_rank × 4B / DP` | 0.0% | ✅ |
| Muon vs Adam 内存节省 | 28.3%（预期 25–31%） | — | ✅ |

> 注：`grad_dtype` 默认为 **FP32（4B/param）**，非 fp16。验证脚本见 `.omc/specs/deep-dive-investigate-memory-consumption-muon-optimizer.md`。

---

## 5. 设计规格

### 5.1 数据模型扩展

#### 5.1.1 新增 `MuonConfig` 到 `strategy.py`

```python
# python/zrt/training/spec/strategy.py

# 各模型的 NS 迭代步数默认值（基于官方报告）
_MUON_NS_STEPS_DEFAULTS: dict[str, int] = {
    "deepseek_v4": 10,    # DSV4 官方报告 §2.4 明确使用 10 步
    "deepseek_v3": 5,     # 原始 Muon 论文默认值
    "default":     5,     # 其他模型回退值
}

@dataclass
class MuonConfig:
    """Muon-specific configuration for performance modeling."""
    # Newton-Schulz 迭代步数。
    # 优先级（从高到低）：
    #   1. 显式设置此字段（训练 YAML 中 strategy.muon_config.ns_steps）
    #   2. ModelSpec.muon_ns_steps（模型 YAML 中声明，若存在）
    #   3. _MUON_NS_STEPS_DEFAULTS 按 model_type 查表
    #   4. 全局默认值 5
    # DSV4 官方报告 §2.4 使用 10 步；原始 Muon 论文使用 5 步。
    ns_steps: int = 5
    ns_variant: str = "zolo_pd"            # "zolo_pd" | "power_iter"
    rotation: bool = True                  # 是否使用 Moonshot 轮转分工优化
    # 使用 Adam 的参数类型（不做 NS 正交化）
    adam_param_types: set[str] = field(
        default_factory=lambda: {"embed", "lm_head", "router", "bias"}
    )
    # Muon 参数的近似覆盖比例（若无法静态分析时用于估算）
    muon_param_fraction: float = 0.85      # 约 85% 参数使用 Muon（DSV4 参考值）


def resolve_muon_ns_steps(
    muon_config: "MuonConfig",
    model: "ModelSpec | None" = None,
) -> int:
    """按优先级解析最终使用的 NS 迭代步数。

    优先级：
      1. muon_config.ns_steps（显式覆盖，YAML 中 strategy.muon_config.ns_steps）
      2. model.muon_ns_steps（ModelSpec 字段，从模型 YAML 加载）
      3. _MUON_NS_STEPS_DEFAULTS[model.model_type]（按模型类型查表）
      4. _MUON_NS_STEPS_DEFAULTS["default"] = 5
    """
    # 若用户显式设置了非默认值，直接使用
    if muon_config.ns_steps != MuonConfig.__dataclass_fields__["ns_steps"].default:
        return muon_config.ns_steps
    # 从 ModelSpec 读取（需 ModelSpec 新增 muon_ns_steps 可选字段）
    if model is not None and getattr(model, "muon_ns_steps", None) is not None:
        return model.muon_ns_steps
    # 按模型类型查表
    if model is not None and hasattr(model, "model_type"):
        steps = _MUON_NS_STEPS_DEFAULTS.get(model.model_type)
        if steps is not None:
            return steps
    return _MUON_NS_DEFAULTS["default"]
```

在 `Strategy` 中加入：

```python
@dataclass
class Strategy:
    ...
    optimizer: OptKind = OptKind.ADAM
    muon_config: MuonConfig = field(default_factory=MuonConfig)
```

#### 5.1.2 `TransformContext.TrainingConfig` 扩展

`TrainingConfig` 中的 `muon_ns_steps` 不写死默认值，而是在运行时通过 `resolve_muon_ns_steps()` 解析：

```python
# python/zrt/transform/context.py

@dataclass
class TrainingConfig:
    optimizer: str = "adam"
    muon_ns_steps: int | None = None   # None = 交由 resolve_muon_ns_steps() 决定
    muon_rotation: bool = True
    muon_param_fraction: float = 0.85
    zero_stage: int = 1
    ...

    def effective_ns_steps(self, model_type: str | None = None) -> int:
        """返回实际使用的 NS 步数，处理 None 回退逻辑。"""
        if self.muon_ns_steps is not None:
            return self.muon_ns_steps
        # 按模型类型查表
        from zrt.training.spec.strategy import _MUON_NS_STEPS_DEFAULTS
        if model_type is not None:
            return _MUON_NS_STEPS_DEFAULTS.get(model_type, 5)
        return 5
```

**`config_loader._parse_strategy()` 扩展**（`training/io/config_loader.py`）：

```python
def _parse_strategy(d: dict) -> Strategy:
    ...
    # 解析 muon_config（若存在）
    muon_cfg = MuonConfig()
    if "muon_config" in d:
        mc = d["muon_config"]
        muon_cfg = MuonConfig(
            ns_steps=mc.get("ns_steps", 5),   # YAML 显式设置优先
            rotation=mc.get("rotation", True),
            muon_param_fraction=mc.get("muon_param_fraction", 0.85),
        )

    return Strategy(
        ...
        optimizer=OptKind(d.get("optimizer", "adam")),
        muon_config=muon_cfg,
    )
```

**`ModelSpec` 新增可选字段**（`training/spec/model.py`）：

```python
@dataclass
class ModelSpec:
    ...
    # Muon 优化器专属字段（可选，从模型 YAML 加载）
    muon_ns_steps: int | None = None    # 若模型 YAML 指定则优先于全局默认
    model_type: str = "default"         # 用于 _MUON_NS_STEPS_DEFAULTS 查表
```

对应的模型 YAML（`configs/models/deepseek_v4_pro.yaml`）新增：

```yaml
# DeepSeek-V4-Pro 模型配置
model_type: deepseek_v4
muon_ns_steps: 10        # DSV4 官方报告 §2.4 明确使用 10 步 NS 迭代
hidden: 7168
...
```

### 5.2 内存模型

#### 5.2.1 修正 `_optimizer_state_bytes()`（`training/models/memory.py`）

**当前（错误）：**
```python
elif strategy.optimizer.value == "muon":
    return int(P * master_bytes * 2.1)    # 2.1 无根据
```

**修正后：**
```python
def _optimizer_state_bytes(P: int, model: ModelSpec, strategy: Strategy) -> int:
    master_bytes = model.master_dtype.bytes   # 通常 FP32 = 4B

    if strategy.optimizer == OptKind.ADAM:
        # master copy + momentum m + variance v = 3P × 4B
        return P * master_bytes * 3

    elif strategy.optimizer == OptKind.MUON:
        mc = strategy.muon_config
        # 分离 Muon 参数 和 Adam 参数
        P_muon = int(P * mc.muon_param_fraction)
        P_adam = P - P_muon
        # Muon 参数：master + M（动量矩阵）= 2P_muon × 4B
        muon_state = P_muon * master_bytes * 2
        # Adam 参数（embed/head 等）：master + m + v = 3P_adam × 4B
        adam_state = P_adam * master_bytes * 3
        return muon_state + adam_state

    return P * master_bytes * 3   # fallback
```

**内存节省分析**（以 DSV4-660B, muon_param_fraction=0.85 为例）：

```
P_total = 660B
P_muon  = 561B,  P_adam = 99B

Adam 方案：660B × 12B = 7,920 GB
Muon 方案：561B × 8B + 99B × 12B = 4,488 + 1,188 = 5,676 GB
节省：2,244 GB（-28.3%）
```

#### 5.2.2 修正 `TrainingMemoryPass`（`transform/analysis/training.py`）

```python
# TrainingMemoryPass.run() 中替换固定系数：
muon_fraction = getattr(ctx.training, 'muon_param_fraction', 0.85) if optimizer == "muon" else 0.0
P_muon = int(total_params * muon_fraction)
P_adam = total_params - P_muon

if optimizer in ("adam", "adamw"):
    opt_bytes = (total_params * 12) / opt_shard
elif optimizer == "muon":
    opt_bytes = (P_muon * 8 + P_adam * 12) / opt_shard
```

#### 5.2.3 时序阶段排序与 PP 并行注意事项 (Temporal Phase Ordering and PP In-Flight Caveat)

训练内存模型采用**顺序阶段假设**：单个 microbatch 的前向、反向、优化器步骤在时间上不重叠。

##### 阶段内存组成

```
[Forward:  weights + activations + comm_buffers]
    ↓
[Backward:  weights + activations + grads + comm_buffers]
    ↓
[Optimizer: weights + grads + opt_state]
```

**关键原则**：`peak_overall = max(peak_forward, peak_backward, peak_optimizer)`，而非三者之和。每个阶段展示的是**单个 microbatch 同时驻留的内存分量**。

##### PP In-Flight 注意事项

在 `PP > 1` 的 steady-state 管线中（如 1F1B 调度），`PP` 个 microbatches 同时在不同 stage 中执行。`_pp_in_flight()` 返回最坏负载 rank 上同时进行的 microbatch 数量（对于 1F1B 为 `PP`）。

**激活内存已缩放**：`peak_backward` 中的激活内存已经乘以此数量（`memory.py:376-379`）。因此 `peak_backward` 反映的是 steady-state 重叠后的实际峰值，而非单个 microbatch 的贡献。

```
# memory.py 中的实现（简化示意）
num_inflight = _pp_in_flight(model, strategy)
peak_backward = weights + (activations * num_inflight) + grads + comm_buffers
```

**sequential 阶段标签**描述的是**单个 microbatch 的驻留语义**，而非 steady-state 重叠现实。

##### ZeRO 分级对峰值的影响

| ZeRO Stage | 权重 (weights) | 梯度 (grads) | 优化器状态 (opt_state) |
|------------|----------------|--------------|----------------------|
| ZeRO-0     | 复制 (replicated) | 复制 | 复制 |
| ZeRO-1     | 复制 | 复制 | **DP 分片** |
| ZeRO-2     | 复制 | **DP 分片** | **DP 分片** |
| ZeRO-3     | **DP 分片** | **DP 分片** | **DP 分片** |

**`peak_optimizer` 不包含激活内存**：优化器步骤在反向完成后运行，此时激活已释放。因此 `peak_optimizer = weights + grads + opt_state`（无 activations）。

**验证**：对于 LLaMA-3 70B 规模（激活 >> 优化器状态），`peak_overall == peak_backward` 恒成立（参见验证报告 `.omc/specs/deep-dive-muon-verification-report.md`）。

##### NS 执行期间的瞬时内存峰值

`peak_optimizer` 还需计入两类仅在优化器步骤期间存在的瞬时缓冲：

| 来源 | 大小 | 触发条件 |
|------|------|---------|
| AllGather 动量缓冲 | `P_muon × 4B × (DP-1)/DP` | ZeRO-1/2/3 且 DP > 1 |
| 中间矩阵 A = XₖᵀXₖ | `max_short_dim² × 4B`（最大权重矩阵，逐矩阵计算）| 始终存在 |

**为何不计入 `total`**：这两项是瞬时缓冲（NS 步骤结束即释放），
`total` 仅对各常驻分量求和作为保守上界。

**影响范围**：对大 DP（如 DP=64）模型，AllGather 缓冲趋近于 `P_muon × 4B`（接近完整动量），
而 ZeRO 分片后的 `opt_state` 仅为 `P_muon × 4B / 64`——差距约 63×。
在序列长度较短或全量重计算场景下，`peak_optimizer` 可能超过 `peak_backward`，
进而影响 `peak_overall`（即 OOM 相关的约束性峰值）。

### 5.3 FLOPs 模型

#### 5.3.1 Newton-Schulz 迭代 FLOPs 精确建模

对参数矩阵 `W ∈ R^{m×n}`（m ≥ n，tall matrix），每步 NS 迭代：

```
A = Xₖᵀ Xₖ    GEMM(n×m, m×n) → n×n:  2mn²   FLOPs
B = Xₖ · A    GEMM(m×n, n×n) → m×n:  2mn²   FLOPs
--------------------------------------------
每步合计：                               4mn²   FLOPs
K步合计：                               K×4mn² FLOPs
```

若 m < n（fat matrix），交换维度：K×4m²n FLOPs。

统一表达：`NS_FLOPs(m, n, K) = K×4 × max(m,n) × min(m,n)²`

> **K 的取值**：默认 K=5（原始 Muon 论文）；DeepSeek-V4 官方报告 §2.4 使用 K=10。
> K=10 使 NS FLOPs 精确翻倍，对大矩阵（hidden=7168）计算开销影响显著。

```python
# 新增文件：python/zrt/training/models/optimizer.py

def ns_flops(m: int, n: int, steps: int = 5) -> float:
    """Newton-Schulz 正交化 FLOPs（K步 × 2个GEMM）。
    
    steps 从调用方传入，不写死；DSV4 传 10，其他模型传 5。
    """
    # 约定 m >= n（tall matrix）；fat matrix 对称处理
    long_dim, short_dim = max(m, n), min(m, n)
    # 每步：2个 GEMM: short×long × long×short + long×short × short×short
    flops_per_step = 2 * long_dim * short_dim**2 + 2 * long_dim * short_dim**2
    return steps * flops_per_step


def adam_step_flops(P: int) -> float:
    """Adam 优化器步骤 FLOPs（逐元素，~12 FLOPs/param）。"""
    return P * 12.0
```

#### 5.3.2 典型矩阵 FLOPs 对比

以 LLaMA-3-70B（hidden=8192, ffn=28672）为例：

| 参数矩阵 | 形状 | P | Adam FLOPs | Muon NS FLOPs | 倍数 |
|---------|------|---|------------|--------------|------|
| q_proj | (8192, 8192) | 67M | 805M | K×4×8192³（K=5: 11T, K=10: 22T）| K=5: ~13,700×，K=10: ~27,400× |
| up_proj | (28672, 8192) | 235M | 2.8G | K×4×28672×8192²（K=5: 38T, K=10: 77T）| K=5: ~13,700×，K=10: ~27,400× |
| embed | (vocab, 8192) | - | 用 Adam | N/A | - |

> **建模含义**：Muon 的计算开销比 Adam 大 4 个数量级，但全部为矩阵乘法，可被 H100 的 GEMM 单元高效执行。优化器步骤从 memory-bound（Adam）变为 **compute-bound**（Muon）。

#### 5.3.3 `OptimizerPass` FLOPs 更新

```python
# python/zrt/transform/training/optimizer.py

def _opt_step_flops(self, optimizer: str, params: int,
                    ns_steps: int = 5,
                    weight_shapes: list[tuple[int, int]] | None = None) -> int:
    """计算优化器步骤 FLOPs。

    ns_steps 由调用方通过 ctx.training.effective_ns_steps() 传入，
    不在此函数内写死。DSV4 传 10，其他模型传 5。
    """
    if optimizer == "adam":
        return params * 12

    elif optimizer == "muon":
        if weight_shapes:
            # 精确模式：逐矩阵计算 NS FLOPs，步数由 ns_steps 决定
            ns_total = sum(ns_flops(m, n, steps=ns_steps) for m, n in weight_shapes)
            # Adam 参数（embed/head）额外 FLOPs
            P_adam = params - sum(m * n for m, n in weight_shapes)
            return int(ns_total + P_adam * 12)
        else:
            # 近似模式：假设平均矩阵维度（基于 hidden_size 估算）
            # NS FLOPs ≈ K×4 × P × sqrt(P) for square-ish matrices
            import math
            avg_dim = int(math.sqrt(params * 0.85))
            return int(ns_steps * 4 * params * 0.85 * avg_dim + params * 0.15 * 12)

    return params * 12
```

### 5.4 通信模型

#### 5.4.1 新增 Muon ZeRO 通信建模

```python
# python/zrt/training/models/optimizer.py （新增）

from zrt.training.models.comm import collective_time, tier_for_group
from zrt.training.spec.strategy import Strategy, OptKind
from zrt.training.spec.system import SystemSpec


def muon_optimizer_comm_time(
    P_muon: int,
    strategy: Strategy,
    system: SystemSpec,
) -> float:
    """
    计算 Muon + ZeRO 优化器步骤引入的额外通信时间（秒）。

    Muon 需要在 NS 迭代前 AllGather 完整动量矩阵。

    Args:
        P_muon: 使用 Muon 优化器的参数量（不含 embed/head）
        strategy: 并行策略
        system: 系统硬件规格

    Returns:
        额外通信时间（秒）；Adam 场景返回 0.0
    """
    if strategy.optimizer != OptKind.MUON:
        return 0.0

    dp = strategy.dp
    if dp <= 1:
        return 0.0

    master_bytes = 4  # FP32
    # 每 rank 持有 P_muon / dp 的动量分片（ZeRO-1 分片优化器状态）
    shard_bytes = (P_muon // dp) * master_bytes

    # AllGather：广播自己的分片，接收其他 rank 的分片
    # Ring AllGather 流量 = (dp-1)/dp × P_muon × master_bytes （per rank）
    ag_bytes = int((dp - 1) / dp * P_muon * master_bytes)

    from zrt.training.ir.graph import Collective
    ag_collective = Collective(
        name="muon_momentum_allgather",
        kind="AG",
        group="DP",
        bytes_=ag_bytes,
        inserted_after="grad_reduce",
    )

    tier = tier_for_group("DP", dp, system)
    ag_time = collective_time(ag_collective, dp, tier)

    # ReduceScatter（若 rotation=True 则需要将正交化结果分发回各 rank）
    if strategy.muon_config.rotation:
        rs_collective = Collective(
            name="muon_momentum_reducescatter",
            kind="RS",
            group="DP",
            bytes_=ag_bytes,
            inserted_after="muon_ns_step",
        )
        rs_time = collective_time(rs_collective, dp, tier)
    else:
        # 无 rotation：各 rank 独立计算，直接 slice，无需 RS
        rs_time = 0.0

    return ag_time + rs_time
```

#### 5.4.2 修正 `total_comm_time()` 加入 Muon 通信

```python
# python/zrt/training/models/comm.py
# 在 total_comm_time() 末尾追加：

    # Muon optimizer AllGather（仅当 optimizer=muon 且 dp > 1）
    if strategy.optimizer == OptKind.MUON and strategy.dp > 1:
        from zrt.training.models.optimizer import muon_optimizer_comm_time
        P = _params_on_rank_for_dp(model, strategy)
        P_muon = int(P * strategy.muon_config.muon_param_fraction)
        muon_opt_time = muon_optimizer_comm_time(P_muon, strategy, system)
        result["muon_optimizer_ag_rs"] = muon_opt_time
```

#### 5.4.3 通信量对比（数值示例）

以 LLaMA-3-70B，TP=8，DP=64，ZeRO-1，inter-node 400Gbps RoCE 为例：

```
P_per_tp_rank = 70B / 8 = 8.75B
P_per_dp_rank = 8.75B / 64 = 136.7M（ZeRO-1 每 rank 优化器状态）

梯度 AllReduce（Adam & Muon 相同）：
  2 × (64-1)/64 × 136.7M × 2B（BF16 grad）= ~536 MB/rank
  Ring-AR 时间 ≈ 536MB / (400Gbps/8) = 10.7ms

Muon 额外 AllGather（动量矩阵，FP32）：
  P_muon = 8.75B × 0.85 = 7.44B（全 rank 总量）
  ZeRO-1 分片：7.44B/64 = 116.2M per rank
  AG 流量：(64-1)/64 × 116.2M × 4B = ~456 MB/rank
  AG 时间 ≈ 456MB / (400Gbps/8) = 9.1ms

总额外通信：~9.1ms（AG）+ ~9.1ms（RS if rotation）= ~18.2ms/step
```

> **结论**：Muon + ZeRO-1 在 DP=64 的场景下引入约 18ms 额外通信，占 step_time 的比例取决于整体训练速度（需在报表中独立显示）。

### 5.5 Transform Pipeline 集成

#### 5.5.1 `OptimizerPass` 输出扩展

`OptimizerPass` 在图中插入的 `optimizer_step` 节点需携带更丰富的属性：

```python
# python/zrt/transform/training/optimizer.py

step_node = OpNode(
    id="optimizer_step",
    op_type=f"optimizer.{opt}",
    attrs={
        "optimizer": opt,
        "params_total": params,
        "params_muon": int(params * muon_fraction) if opt == "muon" else 0,
        "params_adam": params if opt == "adam" else int(params * (1 - muon_fraction)),
        "state_bytes": self._opt_state_bytes(opt, params, muon_fraction),
        # ns_steps 从 ctx 动态读取，不写死：
        #   优先 ctx.training.muon_ns_steps（YAML 显式配置）
        #   否则按 model_type 查表（DSV4→10, others→5）
        "step_flops": self._opt_step_flops(
            opt, params,
            ns_steps=ctx.training.effective_ns_steps(
                getattr(ctx.profile, "model_type", None)
            ) if ctx.training else 5,
        ),
        # Muon 专属属性
        "ns_steps": ctx.training.effective_ns_steps(
            getattr(ctx.profile, "model_type", None)
        ) if (opt == "muon" and ctx.training) else 0,
        "ns_rotation": True if opt == "muon" else False,
        "muon_ag_bytes": self._muon_ag_bytes(opt, params, muon_fraction, dp),
    },
    ...
)
```

#### 5.5.2 `TrainingPipelinePass` 加入 Optimizer Step Time

当前 `step_time` 仅包含 forward + backward + PP bubble + DP AR。需加入：

```python
# python/zrt/transform/analysis/training.py → TrainingPipelinePass.run()

# 新增：计算 optimizer step time
opt_step_time_us = self._compute_optimizer_step_time(g, hw, ctx)
step_time_us += opt_step_time_us
g.metadata["optimizer_step_time_us"] = opt_step_time_us

# ──────────────────────────────────────────────────────────
@staticmethod
def _compute_optimizer_step_time(g, hw, ctx) -> float:
    """计算优化器步骤耗时（微秒）。

    包含：
    1. 优化器计算 FLOPs（Adam：memory-bound；Muon：compute-bound）
    2. Muon AllGather/ReduceScatter 通信（仅 Muon + ZeRO + DP>1）
    """
    opt_node = g.nodes.get("optimizer_step")
    if opt_node is None:
        return 0.0

    optimizer = opt_node.attrs.get("optimizer", "adam")
    step_flops = float(opt_node.attrs.get("step_flops", 0))

    from python.zrt.ir.types import DType
    if optimizer == "muon":
        # Muon NS 是 compute-bound（大 GEMM）
        peak_flops = hw.peak_flops(DType.BF16)
        compute_time_us = (step_flops / peak_flops) * 1e6 if peak_flops > 0 else 0.0
    else:
        # Adam 是 memory-bound（逐元素）
        opt_state_bytes = float(opt_node.attrs.get("state_bytes", 0))
        hbm_bw = hw.memory_bandwidth_bytes_per_s
        compute_time_us = (opt_state_bytes / hbm_bw) * 1e6 if hbm_bw > 0 else 0.0

    # Muon 额外通信（AllGather 动量）
    comm_time_us = 0.0
    if optimizer == "muon":
        ag_bytes = float(opt_node.attrs.get("muon_ag_bytes", 0))
        if ag_bytes > 0:
            dp_bw = hw.interconnect.inter_node.bandwidth_gbps * 1e9 / 8
            dp = ctx.parallel.dp if ctx.parallel else 1
            ring_factor = 2.0 * (dp - 1) / dp
            comm_time_us = (ring_factor * ag_bytes / dp_bw) * 1e6 if dp_bw > 0 else 0.0

    return compute_time_us + comm_time_us
```

#### 5.5.3 `StepResult` 扩展

```python
# python/zrt/training/compose/pipeline.py

@dataclass
class StepResult:
    step_time: float = 0.0
    ...
    optimizer_time: float = 0.0    # 新增：优化器步骤耗时（秒）
    optimizer_comm: float = 0.0    # 新增：Muon AG+RS 通信耗时（秒）
```

### 5.6 MoE 模型的混合优化器策略

DSV4 对不同参数类型使用不同优化器，ZRT-Sim 需要在建模中正确区分：

| 参数类型 | 优化器 | 理由 |
|---------|--------|------|
| Attention 权重（Q/K/V/O proj） | **Muon** | 2D 矩阵，适合 NS 正交化 |
| FFN 权重（up/gate/down proj） | **Muon** | 同上 |
| Embedding（input/output） | **Adam** | 大 vocab 维度，稀疏更新，NS 不适用 |
| LM Head | **Adam** | 同 Embedding |
| MoE Router 矩阵 | **Adam** | 小矩阵 + 稀疏更新，NS 不稳定 |
| MoE Expert FFN 权重 | **Muon** | 2D 矩阵（DSV4 实践） |
| LayerNorm / bias | **Adam**（或 SGD）| 1D 参数，无法做 NS |

**建模参考值**（基于 DSV4 模型结构分析）：

```
muon_param_fraction ≈ 1 - (embed + lm_head + router) / P_total
                    ≈ 1 - (vocab × hidden × 2 + num_layers × hidden) / P_total
```

在 `MuonConfig.muon_param_fraction` 中可设置静态估算值，或由 `OptimizerPass` 在图分析阶段精确计算。

---

## 6. 接口变更

### 6.1 YAML 配置格式

```yaml
# python/zrt/training/configs/deepseek_v4_pro_h100_16n.yaml（示例）

strategy:
  tp: 8
  dp: 64
  pp: 16
  zero_stage: 1
  optimizer: muon             # 新字段值
  muon_config:
    ns_steps: 10              # DSV4 官方报告 §2.4 使用 10 步；若不填则从模型 YAML 或默认值读取
    rotation: true
    muon_param_fraction: 0.85
    adam_param_types: [embed, lm_head, router, bias]
```

### 6.2 CLI 扩展

```bash
# 指定 Muon 优化器（训练建模）
python -m zrt.training estimate \
  --config configs/deepseek_v4_pro.yaml \
  --optimizer muon \
  --muon-rotation
```

### 6.3 报表输出扩展

Excel/HTML 报表新增 **Optimizer** sheet，包含：

| 字段 | 说明 |
|------|------|
| `optimizer_type` | adam / muon |
| `muon_param_fraction` | Muon 参数比例 |
| `opt_state_gb` | 优化器状态显存（GB）|
| `opt_state_savings_gb` | 相较纯 Adam 的节省（GB）|
| `optimizer_step_ms` | 优化器步骤耗时（ms）|
| `muon_ag_rs_ms` | Muon AllGather+RS 通信耗时（ms）|
| `muon_ns_tflops` | NS 迭代 FLOPs（TFLOPs）|
| `optimizer_time_fraction` | 优化器耗时占 step_time 比例 |

---

## 7. 开发计划

### Phase 1 — 内存模型修正（1-2天）

**目标**：修复内存计算错误，引入混合参数支持

**涉及文件**：

- `python/zrt/training/models/memory.py`
  - 修正 `_optimizer_state_bytes()`（P1, P2）
  - 加入 `muon_param_fraction` 分离 Muon/Adam 参数
- `python/zrt/training/spec/strategy.py`
  - 新增 `MuonConfig` dataclass
  - `Strategy` 增加 `muon_config` 字段
- `python/zrt/transform/analysis/training.py`
  - `TrainingMemoryPass` 替换固定系数

**验收标准**：
```bash
PYTHONPATH=python pytest tests/training/test_captured_graph_modelling.py -v -k "muon"
# 验证：Adam opt_state = 12B/param，Muon = 8.5B/param（85%×8 + 15%×12）
```

---

### Phase 2 — FLOPs 模型精确化（2-3天）

**目标**：建模 Newton-Schulz NS 迭代的矩阵乘法开销

**涉及文件**：

- `python/zrt/training/models/optimizer.py`（**新建**）
  - `ns_flops(m, n, steps: int)` — 步数由调用方传入，不写死
  - `adam_step_flops(P)`
  - `muon_step_flops(weight_shapes, muon_fraction, ns_steps: int)`
- `python/zrt/transform/training/optimizer.py`
  - `OptimizerPass._opt_step_flops()` 接入 NS FLOPs
  - 节点属性扩展（`params_muon`, `ns_steps`, `muon_ag_bytes`）
- `python/zrt/training/models/flops.py`
  - 新增 `optimizer_step_flops()` 供 training 模块复用

**验收标准**：
```python
# 单元测试验证 NS FLOPs 计算（steps 必须显式传入）
assert ns_flops(4096, 4096, steps=5)  == 20 * 4096**3      # K=5（默认）= 1.37T
assert ns_flops(4096, 4096, steps=10) == 40 * 4096**3      # K=10（DSV4）= 2.75T
assert ns_flops(28672, 8192, steps=5) == 20 * 28672 * 8192**2   # = 38.5T
assert ns_flops(28672, 8192, steps=10) == 40 * 28672 * 8192**2  # DSV4 = 77T
```

---

### Phase 3 — 通信模型（3-4天）

**目标**：建模 Muon + ZeRO AllGather/ReduceScatter 额外通信

**涉及文件**：

- `python/zrt/training/models/optimizer.py`
  - `muon_optimizer_comm_time()` 函数
- `python/zrt/training/models/comm.py`
  - `total_comm_time()` 末尾追加 Muon AG 项
- `python/zrt/transform/analysis/training.py`
  - `TrainingPipelinePass._compute_optimizer_step_time()` 新增 Muon 通信分支
  - `step_time` 累加 optimizer step time

**验收标准**：
```bash
PYTHONPATH=python pytest tests/training/test_captured_graph_modelling.py -v -k "muon_comm"
# 验证：DP=64时 Muon 额外通信与 §3.4 数值示例吻合（误差 <5%）
```

---

### Phase 4 — 端到端集成与报表（2-3天）

**目标**：YAML 配置支持、报表新增 Optimizer sheet、Anchor 测试

**涉及文件**：

- `python/zrt/training/io/config_loader.py`
  - 解析 `strategy.muon_config` 字段
- `python/zrt/training/compose/pipeline.py`
  - `StepResult` 增加 `optimizer_time`, `optimizer_comm` 字段
- `python/zrt/report/summary.py`
  - 新增 Optimizer sheet 输出
- `python/zrt/training/configs/deepseek_v4_pro_h100_16n.yaml`
  - 加入 `optimizer: muon` + `muon_config` 配置
- `tests/training/anchors/deepseek_v4_pro.yaml`
  - 新增 Muon 场景的 MFU/step_time 锚点

**验收标准**：
```bash
PYTHONPATH=python python -m zrt.training estimate \
  --config python/zrt/training/configs/deepseek_v4_pro_h100_16n.yaml

PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -v -k "deepseek_v4"
```

---

### Phase 5 — Rotation 优化建模与搜索空间集成（1-2天）

**目标**：搜索空间支持 Muon rotation 开关，对比 Adam vs Muon 性能

**涉及文件**：

- `python/zrt/training/search/space.py`
  - `SearchSpace` 新增 `optimizer` 维度（adam / muon）
  - `muon_rotation` 开关
- `python/zrt/training/search/report.py`
  - Pareto front 图表支持 optimizer 着色

**验收标准**：
```bash
# 搜索 Adam vs Muon 帕累托前沿
PYTHONPATH=python python -m zrt.training search \
  --config configs/deepseek_v4_pro.yaml \
  --search-optimizer
```

---

### 里程碑总结

| Phase | 功能 | 预估工期 | 关键产出 |
|-------|------|---------|---------|
| 1 | 内存模型修正 | 1-2天 | 正确的优化器状态内存（修复 P1-P2）|
| 2 | FLOPs 精确化 | 2-3天 | NS 迭代 GEMM 建模（修复 P3）|
| 3 | 通信模型 | 3-4天 | Muon ZeRO AG/RS 建模（修复 P4-P5）|
| 4 | 端到端集成 | 2-3天 | YAML + 报表 + Anchor 测试 |
| 5 | 搜索空间 | 1-2天 | Adam vs Muon 帕累托搜索 |
| **合计** | | **9-14天** | |

---

## 8. 测试策略

### 8.1 单元测试

```python
# tests/training/test_muon_optimizer.py

def test_muon_memory_vs_adam():
    """Muon 优化器状态内存比 Adam 少 ~28%（85% muon_fraction 下）。"""
    ...

def test_ns_flops_square_matrix_k5():
    """正方形矩阵 K=5 时 NS FLOPs = 20 × dim³（K×4 = 20）。"""
    assert ns_flops(4096, 4096, steps=5) == 20 * 4096**3

def test_ns_flops_square_matrix_k10():
    """DSV4 K=10 时 NS FLOPs 精确翻倍 = 40 × dim³。"""
    assert ns_flops(4096, 4096, steps=10) == 40 * 4096**3

def test_ns_flops_tall_matrix():
    """高矩阵 K=5 时 NS FLOPs = 20 × m × n²（m > n）。"""
    assert ns_flops(28672, 8192, steps=5) == 20 * 28672 * 8192**2

def test_ns_steps_resolved_for_dsv4():
    """DSV4 模型类型下 resolve_muon_ns_steps 应返回 10。"""
    from zrt.training.spec.strategy import MuonConfig, resolve_muon_ns_steps
    assert resolve_muon_ns_steps(MuonConfig(), model_type="deepseek_v4") == 10

def test_ns_steps_explicit_overrides_table():
    """YAML 显式设置 ns_steps=7 应覆盖模型类型查表（DSV4 本应返回 10）。"""
    from zrt.training.spec.strategy import MuonConfig, resolve_muon_ns_steps
    assert resolve_muon_ns_steps(MuonConfig(ns_steps=7), model_type="deepseek_v4") == 7

def test_ns_steps_default_non_dsv4():
    """非 DSV4 模型默认返回 5。"""
    from zrt.training.spec.strategy import MuonConfig, resolve_muon_ns_steps
    assert resolve_muon_ns_steps(MuonConfig()) == 5

def test_muon_ag_comm_dp1():
    """DP=1 时 Muon 无额外通信。"""
    ...

def test_muon_ag_comm_dp64():
    """DP=64 时 Muon AG 通信量与理论值匹配（±5%）。"""
    ...
```

### 8.2 集成测试

```python
# tests/training/test_captured_graph_modelling.py 追加

def test_muon_optimizer_step_in_pipeline():
    """optimizer_step_time_us 在 Muon 场景下非零且合理。"""
    ...

def test_muon_step_time_larger_than_adam():
    """Muon step_time 应 >= Adam step_time（额外 AG 通信）。"""
    ...

def test_muon_memory_smaller_than_adam():
    """Muon opt_state < Adam opt_state（节省约 28%）。"""
    ...
```

### 8.3 Anchor 回归测试

新增 `tests/training/anchors/deepseek_v4_pro.yaml`：

```yaml
# DeepSeek-V4-Pro + Muon + ZeRO-1 参考基准
name: deepseek_v4_pro_muon_h100
model: deepseek_v4_pro
strategy:
  tp: 8
  dp: 64
  pp: 16
  zero_stage: 1
  optimizer: muon
expected:
  mfu_min: 0.38           # 参考 DSV4 报告 §3.5.1（含 Muon 收益）
  mfu_max: 0.50
  opt_state_gb_max: 6000  # 约 5,676 GB（见 §5.2）
  optimizer_step_ms_max: 50
tolerance: 0.05
```

---

## 附录：关键公式速查

| 公式 | 说明 |
|------|------|
| `NS_FLOPs(m,n,K) = K×4 × max(m,n) × min(m,n)²` | K步 NS 迭代 FLOPs；K=5（默认）或 10（DSV4）|
| `Adam opt_state = 3P × 4B` | master + m + v |
| `Muon opt_state = P_muon × 8B + P_adam × 12B` | 混合参数 |
| `Muon AG bytes = (DP-1)/DP × P_muon × 4B` | ZeRO-1 momentum AllGather |
| `Adam step: memory-bound; T = state_bytes / HBM_BW` | 逐元素运算 |
| `Muon step: compute-bound; T = NS_FLOPs / Peak_FLOPS` | 矩阵乘法 |
