# LLM Operator Screenshot Tool

使用 `TorchDispatchMode` 在 **meta 张量**上拦截 causal LM 前向传播的完整 aten 算子序列，并将结果写入格式化的 Excel 工作簿。全程无需下载模型权重。

---

## 核心原理

```
HF config.json (几 KB)
       ↓  AutoConfig.from_pretrained
   config 对象
       ↓  AutoModelForCausalLM.from_config  +  FakeTensorMode
  模型结构（无权重，全为 FakeTensor）
       ↓  TorchDispatchMode  +  ModuleTracker hooks
  aten 算子序列（形状 / dtype / 模块路径）
       ↓  两阶段融合 + 数据流分析
       ↓  openpyxl
  model_ops.xlsx  +  *_fusion_rules.json
```

**FakeTensor**：`torch._subclasses.fake_tensor.FakeTensorMode` 创建的轻量张量，正确追踪形状、dtype 和 strides，不分配实际内存，不执行数值计算。相比旧的 meta device 方案，FakeTensorMode 能正确传播 strides、模拟 cuda device，并减少所需的兼容补丁数量。整个追踪过程内存占用极低，即使是 671B 参数的 DeepSeek-V3 也能在几秒内完成。

---

## 安装

```bash
pip install -r requirements.txt
```

`requirements.txt` 内容：

```
torch>=2.0.0
transformers>=4.36.0,<5.0.0
openpyxl>=3.1.0
```

> **Python 版本**：已在 Python 3.14 + torch 2.11.0 + transformers 4.57.6 上验证。

---

## 快速开始

### 命令行

```bash
# HF Hub 模型（只下载 config.json，不下载权重）
python -m python.zrt deepseek-ai/DeepSeek-V3-0324 --layers 4
python -m python.zrt deepseek-ai/DeepSeek-V3      --layers 4
python -m python.zrt Qwen/Qwen2.5-7B-Instruct     --layers 4
python -m python.zrt mistralai/Mistral-7B-v0.1    --layers 2

# 本地目录（包含 config.json 即可）
python -m python.zrt ./hf_models/deepseek_v3   --layers 4
python -m python.zrt ./hf_models/llama3_8b     --layers 2

# 仅抓 prefill 阶段；指定输出目录
python -m python.zrt Qwen/Qwen2.5-7B-Instruct --layers 4 --phases prefill -o output/my_run

# 抓图 + 打印性能报告（需指定硬件）
python -m python.zrt Qwen/Qwen2.5-7B-Instruct --layers 4 --hw nvidia_h100_sxm
python -m python.zrt deepseek-ai/DeepSeek-V3-0324 --layers 4 --hw nvidia_h100_sxm --tp 8

# 训练建模：抓图 + 训练性能估算（--train --hw 组合）
python -m python.zrt deepseek-ai/DeepSeek-V3 --layers 4 --train --hw nvidia_h100_sxm --tp 8 --pp 2
python -m python.zrt hf_models/deepseek_v3 --train --hw nvidia_h100_sxm --tp 8 --pp 4 --dp 2 --global-batch 1024

# 向后兼容的 --model 简写
python -m python.zrt --model v3
python -m python.zrt --model v3.2
```

输出文件默认保存在 `output/graph/<model_slug>/`，每个阶段（prefill / decode）各生成一组文件：`_ops.xlsx`、`_raw_graph.json/.onnx`、`_fused_graph.json/.onnx`。

### Python API

```python
from python.zrt.graph import run_trace_phases

# 推荐：prefill + decode 一次完成
result = run_trace_phases(
    model_id="deepseek-ai/DeepSeek-V3-0324",
    num_layers=4,
    batch_size=1,
    seq_len=128,
    phases=("prefill", "decode"),   # 默认同时抓两阶段
)

# 访问 OpGraph IR
raw_graph, fused_graph = result.graphs["prefill"]
print(f"prefill 原始图: {raw_graph}")      # OpGraph(nodes=..., edges=...)
print(f"prefill 融合图: {fused_graph}")

# 算子记录（list[dict]）
records = result.phase_records["prefill"]
print(f"捕获算子数: {len(records)}")
print(f"输出目录:   {result.output_dir}")
```

```python
from python.zrt.graph import run_trace

# 单阶段（向后兼容）
output_dir, records = run_trace(
    model_id="deepseek-ai/DeepSeek-V3-0324",
    num_layers=4,
    phase="prefill",
)
```

---

## 训练抓图

在推理抓图（prefill / decode）之外，工具同样支持**训练阶段**的算子追踪：捕获前向传播中 `model.train()` 特有的算子（dropout、batch norm 等），并可继续追踪 `loss.backward()` 触发的梯度算子。

### 训练阶段说明

| 阶段 | 含义 | 梯度 | model 模式 |
|------|------|------|------------|
| `train_forward` | 训练前向（含 dropout 等训练特有算子）| 开启 | `train()` |
| `train_backward` | 前向 + backward（包含梯度算子）| 开启 | `train()` |
| `train` | `train_forward` 的简写别名 | — | — |

训练阶段与推理阶段完全独立，可以在同一次调用中混合使用，各自生成独立的输出文件。

### 命令行

```bash
# 快捷 flag：同时抓 train_forward + train_backward
python -m python.zrt deepseek-ai/DeepSeek-V3 --layers 2 --train

# 等价写法（显式指定阶段列表）
python -m python.zrt Qwen/Qwen2.5-7B-Instruct --layers 2 \
    --phases train_forward train_backward

# 仅抓训练前向
python -m python.zrt Qwen/Qwen2.5-7B-Instruct --layers 2 \
    --phases train_forward

# 推理 + 训练混合（四阶段一次完成）
python -m python.zrt Qwen/Qwen2.5-7B-Instruct --layers 4 \
    --phases prefill decode train_forward train_backward
```

输出文件与推理阶段命名规则一致：
```
output/graph/Qwen2.5-7B-Instruct/
├── Qwen2.5-7B-Instruct_train_forward_ops.xlsx
├── Qwen2.5-7B-Instruct_train_forward_raw_graph.json
├── Qwen2.5-7B-Instruct_train_forward_raw_graph.onnx
├── Qwen2.5-7B-Instruct_train_forward_fused_graph.json
├── Qwen2.5-7B-Instruct_train_backward_ops.xlsx
└── ...
```

### Python API

```python
from python.zrt.graph import run_trace_phases

# 仅训练前向
result = run_trace_phases(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    num_layers=4,
    batch_size=1,
    seq_len=128,
    phases=("train_forward",),
)
records = result.phase_records["train_forward"]
print(f"捕获算子数（训练前向）: {len(records)}")

# 训练前向 + 反向（完整梯度追踪）
result = run_trace_phases(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    num_layers=4,
    batch_size=1,
    seq_len=128,
    phases=("train_forward", "train_backward"),
    output_dir="output/graph/qwen_train",
)
fwd_records = result.phase_records["train_forward"]
bwd_records = result.phase_records["train_backward"]
print(f"前向算子数: {len(fwd_records)}")
print(f"反向算子数（含梯度算子）: {len(bwd_records)}")

# 推理 + 训练混合（各阶段独立输出）
result = run_trace_phases(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    num_layers=4,
    batch_size=1,
    seq_len=128,
    phases=("prefill", "decode", "train_forward", "train_backward"),
)
for phase, records in result.phase_records.items():
    print(f"{phase}: {len(records)} ops")

# 访问 OpGraph IR（训练阶段同样生成完整的原始图和融合图）
raw_g, fused_g = result.graphs["train_backward"]
print(f"反向图节点数: {raw_g.num_nodes()}")
```

### 与推理阶段的关键差异

| 特性 | 推理（prefill/decode） | 训练（train_forward/backward） |
|------|----------------------|-------------------------------|
| 梯度计算 | `torch.no_grad()` | 开启（`requires_grad=True`）|
| 模型模式 | `model.eval()` | `model.train()` |
| KV Cache | 支持（prefill→decode 传递）| 不使用 |
| Dropout 算子 | 不出现 | 出现（如有 dropout 配置）|
| 梯度算子 | — | `train_backward` 额外捕获 |

---

## 训练性能建模

在 `--train` 基础上叠加 `--hw` 即可在同一入口执行**训练性能建模**（FLOPs、MFU、内存、通信量、1F1B 流水线调度），无需切换到独立的训练 CLI。

### 命令行

```bash
# 基础训练建模
python -m python.zrt deepseek-ai/DeepSeek-V3 --layers 4 \
    --train --hw nvidia_h100_sxm --tp 8

# 指定流水线并行 + 数据并行 + ZeRO + 批次配置
python -m python.zrt hf_models/deepseek_v3 --layers 4 \
    --train --hw nvidia_h100_sxm \
    --tp 8 --pp 4 --dp 2 \
    --zero-stage 1 --optimizer adam \
    --micro-batch 1 --global-batch 1024 \
    --total-params 671e9 --num-layers-full 61

# 仅抓训练图（不加 --hw 则只做算子追踪，不做性能建模）
python -m python.zrt hf_models/llama3_8b --train --layers 2
```

### 训练建模 CLI 参数

除了统一入口 `python -m python.zrt`，训练建模也可通过独立 CLI 使用：

```bash
# 基于捕获图的训练建模
PYTHONPATH=python python -m zrt.training model-training \
    hf_models/llama3_8b --num-layers 2 \
    --hw nvidia_h100_sxm --tp 2 --dp 4

# 基于 YAML 配置的训练估算（引用 model + hw registry）
PYTHONPATH=python python -m zrt.training estimate \
    --config python/zrt/training/configs/llama3_70b_3d.yaml
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pp` | `1` | 流水线并行度 |
| `--ep` | `1` | 专家并行度 |
| `--dp` | `1` | 数据并行度 |
| `--zero-stage` | `1` | ZeRO 优化阶段（0–3）|
| `--optimizer` | `adam` | 优化器（`adam` / `adamw` / `muon`）|
| `--micro-batch` | `1` | 每 GPU micro-batch 大小 |
| `--global-batch` | `32` | 全局 batch 大小 |
| `--total-params` | — | 完整模型参数量（如 `671e9`，用于缩放）|
| `--hidden` | `7168` | 隐藏层维度（内存估算用）|
| `--num-layers-full` | — | 完整模型总层数（默认等于 `--layers`）|

### 报告内容

训练建模报告包含：

- **Step time**：含 1F1B 流水线调度的完整 step 耗时
- **MFU**：Model FLOPs Utilization
- **FLOPs 分解**：前向 / 反向 / 总计
- **内存（每 GPU）**：权重 / 梯度 / 优化器状态 / 激活值
- **流水线指标**：warmup / steady / cooldown 步数、bubble 占比

### Python API

```python
from python.zrt.graph import run_trace_phases
from python.zrt.transform.analysis.modeller import estimate_training_from_graphs
import python.zrt.hardware.registry as hw_registry

# Step 1: 抓训练图（与其他 CLI 参数共享同一 run_trace_phases）
result = run_trace_phases(
    model_id="hf_models/deepseek_v3",
    num_layers=4,
    phases=("train_forward", "train_backward"),
)

# Step 2: 训练建模（复用已抓的 OpGraph，无需重抓）
hw = hw_registry.load("nvidia_h100_sxm")
report = estimate_training_from_graphs(
    forward_graph=result.graphs["train_forward"][0],
    backward_graph=result.graphs["train_backward"][0] if "train_backward" in result.graphs else None,
    hw_spec=hw,
    tp=8, pp=4, dp=2,
    total_params=671e9,
    num_layers_full=61,
)
print(report.summary())
```

> 也可以使用端到端 API `model_training()`，它内部自动完成抓图 + 建模。详见 `python/zrt/transform/analysis/modeller.py`。

---

## 图模式（torch.compile）

默认的 eager 路径通过 `TorchDispatchMode` 逐 op 拦截，产生扁平的算子列表。  
`graph_mode=True` 改用 **`torch.compile` 自定义 backend** 捕获计算图，Dynamo 在 trace 时生成带显式数据流边的 FX `GraphModule`。

### 与 eager 模式的对比

| 特性 | eager（默认） | graph_mode=True |
|------|--------------|-----------------|
| 捕获机制 | `TorchDispatchMode` | `torch.compile` + Dynamo |
| 数据流边 | 后验推断（tensor id 匹配）| FX `node.args` 直接读取 |
| GEMM 表示 | `mm` / `addmm` 分散 | `aten.linear.default` 统一 |
| 模块路径标注 | 完整（forward hook）| 依赖 Dynamo `nn_module_stack`（保留） |
| 训练反向捕获 | `loss.backward()` eager 拦截 | Dynamo 自动捕获反向子图 |
| 子图数量 | 单一扁平序列 | 可能多个子图（graph break 时合并）|
| 输出格式 | Excel + JSON + ONNX | **相同**（完全兼容）|

### 命令行

```bash
# 图模式抓 DSv3 训练计算图（前向 + 反向）
python -m python.zrt deepseek-ai/DeepSeek-V3 --layers 2 --phases train_backward --graph-mode

# 图模式抓推理阶段
python -m python.zrt deepseek-ai/DeepSeek-V3-0324 --layers 4 --phases prefill decode --graph-mode
```

### Python API

```python
from python.zrt.graph import run_trace_phases

# 图模式：DSv3 训练前向 + 反向
result = run_trace_phases(
    model_id="deepseek-ai/DeepSeek-V3",
    num_layers=2,
    batch_size=1,
    seq_len=16,
    phases=("train_backward",),
    graph_mode=True,
)
records = result.phase_records["train_backward"]
print(f"捕获算子数: {len(records)}")

# 图模式：推理阶段
result = run_trace_phases(
    model_id="deepseek-ai/DeepSeek-V3-0324",
    num_layers=4,
    phases=("prefill", "decode"),
    graph_mode=True,
)
```

### 注意事项

- 模型含复杂控制流（如 MoE routing）时 Dynamo 可能产生多个子图（graph break），records 会按顺序合并。
- `graph_mode` 与 `--train` / `--phases` 等所有现有参数完全兼容。

---

## 命令行参数


| 参数              | 简写  | 默认值              | 说明                                                                            |
| ----------------- | ----- | ------------------- | ------------------------------------------------------------------------------- |
| `model_id`        | —    | 必填                | HF Hub 模型 ID 或本地路径                                                       |
| `--layers`        | —    | `4`                 | 追踪的 Transformer Block 数量（2–4 即可覆盖所有算子模式）                      |
| `--output`        | `-o` | 自动命名            | 输出目录路径                                                                    |
| `--batch-size`    | —    | `1`                 | dummy 输入的 batch size                                                         |
| `--seq-len`       | —    | `128`               | dummy 输入的序列长度                                                            |
| `--phases`        | —    | `prefill decode`    | 追踪的阶段列表，可选：`prefill`、`decode`、`train_forward`、`train_backward`    |
| `--train`         | —    | `False`             | 快捷 flag：等价于 `--phases train_forward train_backward`                       |
| `--platform`      | —    | `generic`           | 目标平台（`cuda`/`ascend_npu`/`cpu`/`generic`），影响融合 kernel 命名           |
| `--hw`            | —    | —                  | 硬件规格名称（如 `nvidia_h100_sxm`），用于打印性能报告                          |
| `--tp`            | —    | `1`                 | 张量并行度（配合 `--hw` 使用）                                                  |
| `--target-layers` | —    | —                  | 指定追踪的层编号，逗号分隔（如 `0,3`）                                          |
| `--auto-layers`   | —    | `True`（CLI 默认） | 自动选择第一个密集层和第一个稀疏（MoE）层                                       |
| `--graph-mode`    | —    | `False`             | 使用 `torch.compile` 图模式捕获，替代 `TorchDispatchMode` eager 路径            |
| `--gradient-checkpointing` | — | `False`      | 启用激活重计算（训练阶段）                                                      |
| `--pp`            | —    | `1`                 | 流水线并行度（训练建模用）                                                      |
| `--ep`            | —    | `1`                 | 专家并行度（训练建模用）                                                        |
| `--dp`            | —    | `1`                 | 数据并行度（训练建模用）                                                        |
| `--zero-stage`    | —    | `1`                 | ZeRO 优化阶段 0-3（训练建模用）                                                 |
| `--optimizer`     | —    | `adam`              | 优化器（训练建模用，`adam`/`adamw`/`muon`）                                    |
| `--micro-batch`   | —    | `1`                 | 每 GPU micro-batch 大小（训练建模用）                                           |
| `--global-batch`  | —    | `32`                | 全局 batch 大小（训练建模用）                                                   |
| `--total-params`  | —    | —                  | 完整模型参数量，如 `671e9`（用于缩放）                                          |
| `--hidden`        | —    | `7168`              | 隐藏层维度（内存估算用）                                                        |
| `--num-layers-full` | —  | —                  | 完整模型总层数（默认等于 `--layers`）                                           |
| `--model`         | —    | —                  | 向后兼容简写：`v3` 或 `v3.2`                                                    |

### 关于 `--layers` 的选择


| 场景                                            | 推荐值 |
| ----------------------------------------------- | ------ |
| 纯密集模型（Llama / Qwen2 / Mistral）           | 2      |
| DeepSeek-V3（前 3 层为密集层，第 4 层起为 MoE） | 4      |
| Qwen3-MoE / Mixtral（第 1 层即为 MoE）          | 2      |

---

## 支持的模型

### 开箱即用（在 transformers 注册表中）


| 架构                      | 示例模型 ID                                  | 注意       |
| ------------------------- | -------------------------------------------- | ---------- |
| LLaMA / LLaMA-2 / LLaMA-3 | `meta-llama/Llama-3.1-8B`                    | 需 HF 授权 |
| Qwen2 / Qwen2.5           | `Qwen/Qwen2.5-7B-Instruct`                   | —         |
| Qwen3 (dense)             | `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-8B`           | —         |
| Qwen3 (MoE)               | `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-235B-A22B` | —         |
| Mistral                   | `mistralai/Mistral-7B-v0.1`                  | —         |
| Mixtral (MoE)             | `mistralai/Mixtral-8x7B-v0.1`                | —         |
| Gemma / Gemma2            | `google/gemma-2-9b`                          | —         |
| Phi-3 / Phi-4             | `microsoft/Phi-4`                            | —         |

### 需要 `trust_remote_code`（自动处理）


| 架构                    | 示例模型 ID                    | 特殊处理                       |
| ----------------------- | ------------------------------ | ------------------------------ |
| DeepSeek-V3             | `deepseek-ai/DeepSeek-V3`      | MoE meta patch                 |
| DeepSeek-V3-0324 (V3.2) | `deepseek-ai/DeepSeek-V3-0324` | MoE meta patch + Indexer patch |

### 本地目录支持

本地目录只需包含 `config.json`：

- **标准模型类型**（llama / qwen2 / mistral / mixtral 等）：直接支持，无需额外文件
- **自定义架构**（如 deepseek_v3）：需要在 `config.json` 中有 `auto_map` 字段，并附带 `configuration_*.py` 和 `modeling_*.py`

```
hf_models/
├── deepseek_v3/          ← config.json 含 auto_map，附带建模代码
│   ├── config.json
│   ├── configuration_deepseek.py
│   └── modeling_deepseek.py
├── llama3_8b/            ← 只需 config.json（llama 已在 transformers 注册）
│   └── config.json
└── qwen2_7b/
    └── config.json
```

---

## Transform Pipeline + 导出（Python API）

### 完整工作流：抓图 → Transform → 导出

```python
from python.zrt.graph import run_trace_phases
from python.zrt.graph.transform_runner import run_transform
from python.zrt.transform import ParallelConfig, StreamConfig
from pathlib import Path

# Step 1: 抓原始图
result = run_trace_phases(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    num_layers=4,
    batch_size=1,
    seq_len=128,
    phases=("prefill",),
)
raw_graph, fused_graph = result.graphs["prefill"]
output_base = Path("output/graph/Qwen2.5-7B-Instruct")

# Step 2: Transform + 导出（TP=1 baseline）
output_dir, transformed_graph = run_transform(
    raw_graph=raw_graph,
    output_dir=output_base,
    parallel_config=ParallelConfig(tp=1),     # 单卡，无 TP/EP/PP/DP
    stream_config=StreamConfig(num_compute_streams=1, num_comm_streams=1),
    hw_spec=hw_spec,  # 可选，用于性能估算
)
# 输出：
#   - {output_base}/Qwen2.5-7B-Instruct_transformed_ops.xlsx  (5个Sheet)
#   - {output_base}/Qwen2.5-7B-Instruct_transformed_graph.json

# Step 3: Transform + 导出（TP=4）
_, transformed_graph_tp4 = run_transform(
    raw_graph=raw_graph,
    output_dir=output_base,
    parallel_config=ParallelConfig(tp=4),     # TP=4：4卡张量并行
    stream_config=StreamConfig(num_compute_streams=1, num_comm_streams=1),
    hw_spec=hw_spec,
)
```

### `run_transform()` 参数说明


| 参数              | 类型              | 默认值                     | 说明                                |
| ----------------- | ----------------- | -------------------------- | ----------------------------------- |
| `raw_graph`       | OpGraph           | —                         | 从`run_trace_phases()` 获取的原始图 |
| `output_dir`      | Path              | —                         | 输出目录（自动创建）                |
| `parallel_config` | ParallelConfig    | `ParallelConfig()`         | TP/EP/PP/DP/SP 配置                 |
| `stream_config`   | StreamConfig      | `StreamConfig(1, 1)`       | 计算流数 + 通信流数                 |
| `pipeline`        | TransformPipeline | `build_default_pipeline()` | 自定义 transform 管道               |
| `hw_spec`         | HardwareSpec      | `A100_40GB`                | GPU 规格（用于 FLOPs 估算）         |

### 导出文件详解

#### Excel：`*_transformed_ops.xlsx`（5个Sheet）


| Sheet                     | 内容                                                |
| ------------------------- | --------------------------------------------------- |
| **Metadata**              | 图的基本信息 + 并行策略 + 流配置                    |
| **Transformed Operators** | 所有算子（注入了 FLOPs / 延迟 / bound / stream_id） |
| **Communication Ops**     | 通信算子详情（all-reduce / all-to-all / send-recv） |
| **Parallelism Summary**   | 按层统计计算/通信/内存算子数 + 并行类型             |
| **Stream Assignment**     | 流分配统计（哪些节点在 compute/comm stream）        |

#### JSON：`*_transformed_graph.json`

结构化数据（用于自动化分析）：

```json
{
  "graph": {
    "name": "Qwen2.5-7B-Instruct",
    "phase": "prefill",
    "num_nodes": 150,
    "num_edges": 200
  },
  "parallelism": {
    "strategy": "TP=1, EP=1, PP=1, DP=1, SP=False",
    "tp": 1, "ep": 1, "pp": 1, "dp": 1, "sp": false
  },
  "stream_config": {
    "compute_streams": 1,
    "comm_streams": 1
  },
  "nodes": [
    {
      "id": "op_0",
      "op_type": "mm",
      "category": "compute",
      "scope": "layers.0.self_attn.q_proj",
      "layer": "0",
      "annotations": {
        "flops": 134217728,
        "latency_us": 12.5,
        "bound": "compute",
        "stream_id": 0,
        "stream_type": "compute"
      },
      "input_shapes": [[1, 128, 4096]],
      "output_shapes": [[1, 128, 4096]]
    },
    ...
  ],
  "edges": [...]
}
```

### ParallelConfig 常见配置

```python
from python.zrt.transform import ParallelConfig

# 单卡（baseline）
ParallelConfig(tp=1)

# TP=4（4卡张量并行，需要 4× all-reduce）
ParallelConfig(tp=4)

# TP=8 + EP=2（8卡 TP，2卡专家并行）
ParallelConfig(tp=8, ep=2)

# TP=4 + PP=2（4卡 TP，2阶段流水线）
ParallelConfig(tp=4, pp=2)

# TP=4 + DP=2（4卡 TP，2卡数据并行）
ParallelConfig(tp=4, dp=2)

# 序列并行（需 transformers 5.0+ 支持）
ParallelConfig(tp=4, sp=True)

# 获取配置描述
cfg = ParallelConfig(tp=4, ep=2, pp=2)
print(cfg.describe())  # "TP=4, EP=2, PP=2, DP=1, SP=False"
```

---

## 输出 Excel 说明

工作簿包含 6 个 Sheet：


| Sheet                 | 说明                                        |
| --------------------- | ------------------------------------------- |
| Model Config          | 模型配置摘要                                |
| Fused Operators       | 融合后的算子序列（主视图，含融合 I/O 映射） |
| Raw Operator Sequence | 原始 aten 算子完整序列                      |
| Summary               | 按融合算子聚合统计                          |
| By Layer              | 按层级聚合统计                              |
| Fusion Rules          | 自动发现的融合模式（含融合 I/O 映射）       |

同时生成 `*_fusion_rules.json`，记录自动发现的算子融合模式。

### Sheet：Model Config

模型关键配置一览，包含（有则展示，无则跳过）：


| 字段                           | 说明                       |
| ------------------------------ | -------------------------- |
| `model_id`                     | 模型来源                   |
| `model_type`                   | 架构类型                   |
| `hidden_size`                  | 隐藏层维度                 |
| `num_hidden_layers`            | 总层数 / 追踪层数          |
| `num_attention_heads`          | 注意力头数                 |
| `vocab_size`                   | 词表大小                   |
| `n_routed_experts`             | MoE 路由专家数（DeepSeek） |
| `num_local_experts`            | MoE 本地专家数（Mixtral）  |
| `q_lora_rank` / `kv_lora_rank` | MLA 低秩维度（DeepSeek）   |

### Sheet：Fused Operators（主视图）

两阶段自动融合后的算子序列，每行包含：

- **Fused Input Shapes/Dtypes**：融合 kernel 的外部输入张量信息
- **Input Sources**：每个输入来自哪个子算子的哪个输入端口
- **Fused Output Shapes/Dtypes**：融合 kernel 的外部输出张量信息
- **Output Sources**：每个输出由哪个子算子的哪个输出端口产生

---

## 组件标签体系

`Component` 列的标签由模块路径中的命名模式推断，与具体模型实现无关：


| 标签前缀                                          | 含义                               | 颜色 |
| ------------------------------------------------- | ---------------------------------- | ---- |
| `attn_norm`                                       | Attention 前的 LayerNorm / RMSNorm | 绿   |
| `ffn_norm`                                        | Attention 后的 LayerNorm / RMSNorm | 绿   |
| `final_norm`                                      | 最终 Norm 层                       | 绿   |
| `attn.q_proj` / `attn.k_proj` / …                | 标准 QKV 及输出投影                | 蓝   |
| `attn.q_a_proj` / `attn.kv_a_proj` / …           | MLA 低秩投影（DeepSeek）           | 蓝   |
| `attn.score`                                      | QK 内积计算                        | 蓝   |
| `attn.softmax`                                    | Softmax                            | 蓝   |
| `attn.rope`                                       | RoPE 位置编码                      | 蓝   |
| `moe.gate.*`                                      | MoE 路由 / 门控                    | 橙   |
| `moe.shared.*`                                    | MoE 共享专家（DeepSeek）           | 黄   |
| `moe.experts.*`                                   | MoE 路由专家 MLP                   | 粉   |
| `ffn.gate_proj` / `ffn.up_proj` / `ffn.down_proj` | 密集 FFN 投影                      | 紫   |
| `ffn.silu` / `ffn.mul`                            | 激活函数                           | 紫   |
| `embedding`                                       | Token Embedding                    | 灰   |
| `lm_head`                                         | 语言模型输出头                     | 灰   |

---

## V3 vs V3.2 对比


| 特性             | V3  | V3.2                  |
| ---------------- | --- | --------------------- |
| 原始算子数 (4层) | 400 | 468                   |
| 融合后算子数     | 75  | 87                    |
| 融合模式数       | 6   | 7                     |
| Indexer 模块     | 无  | 有 (MLA 注意力中新增) |

---

## 实现细节

### MoE Meta Patch

MoE 模块的 `forward` 通常在路由阶段调用 `.cpu().numpy()`（如 DeepSeek 的 `moe_infer`），在 meta 张量上会失败。工具会自动检测并替换为简化版 forward：

- **检测条件**：模块有 `experts: nn.ModuleList` 属性（duck typing，不绑定类名）
- **简化策略**：执行 gate（捕获路由算子）→ 运行 `experts[0]`（捕获专家 MLP 算子）→ 处理 shared expert（如有）
- **返回类型适配**：通过检查原始 `forward` 源码，自动判断是返回单 tensor（DeepSeek / Qwen-MoE 风格）还是 `(hidden, router_logits)` 二元组（Mixtral 风格）

### Autocast FakeTensor 兼容

transformers 4.50+ 的 RoPE 实现会将 tensor 的 device type 直接传给 `torch.autocast`，FakeTensor 模拟的 device（如 `cuda`）在某些 torch 版本中可能不受 autocast 支持。工具在 `apply_compat_patches()` 中将未知 device type 重定向到 `cpu`（对 FakeTensor 是 no-op）。

### 两阶段算子融合

1. **第一阶段**：将同一 leaf module 触发的连续算子归为一组
2. **第二阶段**：若 parent module 内只有 ≤30 个子算子，将其进一步合并到 parent 级别

每个融合组通过张量 ID 追踪，区分外部 I/O（跨组边界的张量）与内部传递张量。

### 过滤掉的算子

以下零开销算子默认跳过：

`view`, `reshape`, `expand`, `permute`, `contiguous`, `slice`, `unsqueeze`, `squeeze`, `split`, `clone`, `arange`, `zeros`, `ones`, `full`, `tril`, `triu`, …

---

## 项目结构

```
modeling/
├── e2e_check.py                   # 完整 e2e 流程（抓图 + transform + 导出 + 调度 + 性能报告）
├── test_screenshot_ops.py         # pytest 自验用例
├── requirements.txt               # 依赖声明
│
├── python/zrt/graph/              # 图追踪模块（抓原始图）
│   ├── __init__.py                # 公开 API: run_trace / run_trace_phases / load_model
│   ├── main.py                    # CLI 入口
│   ├── model_loader.py            # 通用 HF 模型加载 + 兼容性修补
│   ├── dispatch.py                # RecordingDispatch + TensorTracker（aten 拦截）
│   ├── tracker.py                 # ModuleTracker（forward hooks）
│   ├── fusion.py                  # FusionEngine（两阶段算子融合）
│   ├── classifier.py              # 组件分类 + 颜色映射
│   ├── graph_builder.py           # build_op_graph / build_fused_op_graph
│   ├── graph_exporter.py          # 导出 JSON / ONNX（原始图）
│   ├── excel_writer.py            # Excel + JSON（原始图）
│   ├── transform_runner.py        # run_transform: 原始图 → transform pipeline → 导出
│   ├── patches.py                 # 运行时 patch（MoE、Indexer、legacy 属性）
│   ├── compat.py                  # transformers 版本 shim + 本地模型注册表
│   └── tensor_utils.py            # 张量工具 + SKIP_OPS
│
├── python/zrt/transform/          # Transform 管道模块（注入并行/流/通信）
│   ├── __init__.py                # API 导出
│   ├── context.py                 # TransformContext / ParallelConfig / StreamConfig
│   ├── pipeline.py                # TransformPipeline / build_default_pipeline
│   ├── passes/                    # 各种 transform 遍历（FLOPs / Roofline / Communication 等）
│   ├── exporter.py                # export_transformed_graph: 导出 Excel/JSON（转换后的图）
│   └── …
│
├── python/zrt/executor/           # 执行调度模块
│   ├── dag_scheduler.py           # DAGScheduler（拓扑排序 + 流调度）
│   └── …
│
├── python/zrt/simulator/          # 性能模拟
│   └── …
│
├── python/zrt/report/             # 报告生成
│   └── …
│
├── python/zrt/hardware/           # 硬件规格
│   ├── spec.py                    # HardwareSpec 数据类
│   ├── gpu.py                     # GPU_SPECS
│   ├── registry.py                # hw_registry.load("nvidia_h100_sxm")
│   └── configs/                   # 硬件配置 YAML (H100, A100, Ascend 910B/C, ...)
│
├── python/zrt/ir/                 # 中间表示（IR）
│   ├── graph.py                   # OpGraph
│   ├── node.py                    # OpNode
│   ├── edge.py                    # OpEdge
│   └── tensor.py                  # Tensor（形状、dtype、内存占用）
│
├── python/zrt/training/           # 训练性能建模
│   ├── cli.py                     # CLI: model-training / estimate 子命令
│   ├── io/config_loader.py        # YAML 加载（model ref + hw registry + strategy）
│   ├── configs/models/            # 模型规格 YAML（llama3_70b, deepseek_v3, ...）
│   ├── configs/*.yaml             # 训练配置（引用 model + hw + strategy）
│   └── ...                        # search/ spec/ compose/ 模块
│
└── hf_models/                     # 模型本地副本（只读！）
    ├── deepseek_v3/               # config.json + 建模代码（含 auto_map）
    ├── deepseek_v3_2/
    ├── llama3_8b/                 # 仅 config.json
    ├── llama3_70b/
    ├── qwen2_7b/
    ├── qwen2_72b/
    ├── mistral_7b/
    ├── mixtral_8x7b/
    └── modeling_sources/
```

### 核心数据流

```
HF config.json
       ↓
    load_model()  ← model_loader.py
       ↓ (返回 model, config, fake_mode)
   forward pass  ← TorchDispatchMode (dispatch.py)
       ↓
 aten 算子序列   ← ModuleTracker (tracker.py)
       ↓
  raw OpGraph    ← graph_builder.py
       ↓
run_transform()  ← transform_runner.py [NEW]
       ↓ (应用 transform pipeline)
 transformed     ← pipeline.py
 OpGraph         ← passes/
       ↓
 export 结果     ← exporter.py [NEW]
       ↓
  Excel/JSON     输出: metadata + operators + communication + parallelism + streams
  
       ↓ (可选)
  DAGScheduler   ← executor.py (调度)
       ↓
  性能报告       ← report.py (TTFT / TPOT / MFU / 内存)
```

---

## 自验输出文件

### 方式一：运行 e2e_check.py（完整流程）

```bash
# 完整 e2e 检查：抓图 + transform (TP=1, TP=4) + 导出 Excel/JSON + 性能报告
python e2e_check.py

# 输出位置：
# - output/graph/Qwen2.5-7B-Instruct/
#   ├── Qwen2.5-7B-Instruct_transformed_ops.xlsx  (TP=1, 5个Sheet)
#   ├── Qwen2.5-7B-Instruct_transformed_graph.json  (TP=1)
#   ├── Qwen2.5-7B-Instruct_transformed_ops.xlsx  (TP=4 覆盖，需要改进)
#   └── Qwen2.5-7B-Instruct_transformed_graph.json  (TP=4 覆盖，需要改进)
```

> **提示**：e2e_check.py 的 Step 8 会自动调用 `run_transform()` 导出结果。

### 方式二：Python API（按需导出）

```python
from python.zrt.graph import run_trace_phases
from python.zrt.graph.transform_runner import run_transform
from python.zrt.transform import ParallelConfig, StreamConfig
import python.zrt.hardware.registry as hw_registry
from pathlib import Path

model_id = "Qwen/Qwen2.5-7B-Instruct"

# 步骤 1：抓图
result = run_trace_phases(
    model_id=model_id,
    num_layers=4,
    batch_size=1,
    seq_len=128,
    phases=("prefill",),
)
raw_graph, _ = result.graphs["prefill"]

# 步骤 2：获取硬件规格
hw = hw_registry.load("nvidia_h100_sxm")

# 步骤 3a：导出 TP=1 版本
output_dir = Path(f"output/graph/{model_id.split('/')[-1]}")
_, _ = run_transform(
    raw_graph=raw_graph,
    output_dir=output_dir,
    parallel_config=ParallelConfig(tp=1),
    stream_config=StreamConfig(num_compute_streams=1, num_comm_streams=1),
    hw_spec=hw,
)
print(f"✓ TP=1 导出到: {output_dir}")

# 步骤 3b：导出 TP=4 版本（输出文件名相同，会覆盖）
# 建议：为 TP=4 指定不同的输出目录
output_dir_tp4 = Path(f"output/graph/{model_id.split('/')[-1]}_tp4")
_, _ = run_transform(
    raw_graph=raw_graph,
    output_dir=output_dir_tp4,
    parallel_config=ParallelConfig(tp=4),
    stream_config=StreamConfig(num_compute_streams=1, num_comm_streams=1),
    hw_spec=hw,
)
print(f"✓ TP=4 导出到: {output_dir_tp4}")
```

### 方式三：命令行（仅抓图，不做 transform）

```bash
# 抓图 + 自动应用默认 transform（TP=1）
python -m python.zrt Qwen/Qwen2.5-7B-Instruct --layers 4 -o output/my_qwen

# 指定硬件规格（用于 FLOPs 估算）
python -m python.zrt Qwen/Qwen2.5-7B-Instruct --layers 4 --hw nvidia_h100_sxm

# 测试 TP=4 配置（仅 transform，不修改原始图）
python -m python.zrt Qwen/Qwen2.5-7B-Instruct --layers 4 --tp 4 --hw nvidia_h100_sxm
```

---

## 运行测试

```bash
# 安装测试依赖
pip install pytest
```

### 测试覆盖


| 测试组                            | 内容                                                            |
| --------------------------------- | --------------------------------------------------------------- |
| `TestExtractLayerIdx`             | 层编号提取逻辑                                                  |
| `TestCla[tests](tests)ssifyComponent`           | 组件分类（norm / attn / MLA / MoE / FFN / embedding / lm_head） |
| `TestMoEDetection`                | MoE 模块检测与 patch 替换                                       |
| `test_local_model[*]`             | 7 个本地模型端到端追踪（无网络）                                |
| `test_moe_components_present`     | MoE 算子出现验证（DeepSeek-V3、Mixtral）                        |
| `test_deepseek_v3_mla_components` | MLA 专属算子出现验证                                            |
| `test_layer_attribution`          | Block 内算子均有正确层号                                        |
| `test_config_summary_fields`      | Config 摘要必填字段完整性                                       |
| `test_hub_model[*]`               | HF Hub 端到端：DeepSeek-V3.2 / Qwen3 / Llama-3.1                |
| `test_hub_moe_detection`          | Hub 模型 MoE/非MoE 检测准确性                                   |


### 精度验证

##### 验证 DeepSeek-V3 在 H100 上的精度

```
python -m validation.cli run \n    --model-id deepseek-ai/DeepSeek-V3-0324 \n    --dataset lambada \n    --device h100 \n    --batch-size 32 \n    --output validation_results/deepseek_v3_h100/
```

##### 生成对比报告

```
python -m validation.cli compare \n    --baseline huggingface \n    --traced validation_results/deepseek_v3_h100/ \n    --metric accuracy perplexity latency \n    --report html
```
