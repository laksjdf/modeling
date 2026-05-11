# ZRT-Sim — LLM 算子截图与性能建模工具

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

### 核心依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 内容：

```
torch>=2.0.0
transformers>=4.36.0
openpyxl>=3.1.0
onnx>=1.14.0
```

> **Python 版本**：已在 Python 3.14 + torch 2.11.0 + transformers 5.4.0 上验证。

### HTTP 服务额外依赖

```bash
pip install -r server/requirements.txt
```

`server/requirements.txt` 内容：

```
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
```

---

## 快速开始

### 命令行

#### 两种建模方式

| 方式 | 入口 | 前提 | 适用场景 |
|------|------|------|----------|
| 图捕获建模 | `--model-id` [+ `--hw`] | HF model config（几 KB） | 推理/训练算子级分析、精确 FLOPs |
| 配置估算建模 | `--estimate-config` | YAML 规格文件 | 训练策略快速探索、并行搜索 |

```bash
# 推理抓图（HF Hub 模型，只下载 config.json）
python -m python.zrt --model-id deepseek-ai/DeepSeek-V3-0324 --layers 4
python -m python.zrt --model-id deepseek-ai/DeepSeek-V3 --layers 4
python -m python.zrt --model-id Qwen/Qwen2.5-7B-Instruct --layers 4
python -m python.zrt --model-id mistralai/Mistral-7B-v0.1 --layers 2

# 本地目录（包含 config.json 即可）
python -m python.zrt --model-id ./hf_models/deepseek_v3 --layers 4
python -m python.zrt --model-id ./hf_models/llama3_8b --layers 2

# 仅抓 prefill 阶段；指定输出目录
python -m python.zrt --model-id Qwen/Qwen2.5-7B-Instruct --layers 4 --phases prefill -o output/my_run

# 推理性能建模（抓图 + 调度 + 报告）
python -m python.zrt --model-id Qwen/Qwen2.5-7B-Instruct --layers 4 --hw nvidia_h100_sxm
python -m python.zrt --model-id deepseek-ai/DeepSeek-V3-0324 --layers 4 --hw nvidia_h100_sxm --tp 8

# 多并行推理分析（TP+EP+DP）
python -m python.zrt --model-id Qwen/Qwen2.5-7B-Instruct --layers 4 --hw nvidia_h100_sxm --tp 4 --ep 2 --dp 2

# 指定量化精度分析
python -m python.zrt --model-id Qwen/Qwen2.5-7B-Instruct --layers 4 --hw nvidia_h100_sxm --tp 8 --quant int8

# 训练抓图（快捷 flag）
python -m python.zrt --model-id deepseek-ai/DeepSeek-V3 --layers 2 --train

# 训练抓图（显式指定阶段）
python -m python.zrt --model-id Qwen/Qwen2.5-7B-Instruct --layers 2 --phases train_forward train_backward

# 仅抓训练前向
python -m python.zrt --model-id Qwen/Qwen2.5-7B-Instruct --layers 2 --phases train_forward

# 推理 + 训练混合（四阶段一次完成）
python -m python.zrt --model-id Qwen/Qwen2.5-7B-Instruct --layers 4 --phases prefill decode train_forward train_backward

# 训练性能建模（抓图 + 1F1B 调度 + MFU 估算）
python -m python.zrt --model-id deepseek-ai/DeepSeek-V3 --layers 4 --train --hw nvidia_h100_sxm --tp 8

# 完整 3D 并行 + ZeRO + 批次配置
python -m python.zrt --model-id hf_models/deepseek_v3 --layers 4 --train --hw nvidia_h100_sxm --tp 8 --pp 4 --ep 2 --dp 2 --zero-stage 1 --optimizer adam --micro-batch 1 --global-batch 1024 --total-params 671e9 --num-layers-full 61

# 启用激活重计算
python -m python.zrt --model-id Qwen/Qwen2.5-7B-Instruct --layers 4 --train --hw nvidia_h100_sxm --tp 8 --gradient-checkpointing

# Spec-based 训练估算（无需抓图，纯分析）→ 默认输出 Excel
python -m python.zrt --estimate-config python/zrt/training/configs/llama3_70b_3d.yaml

# 网格搜索（Pareto 前沿）
python -m python.zrt --search-config python/zrt/training/configs/llama3_70b_3d.yaml

# 指定输出路径（.xlsx → Excel，.json → JSON）
python -m python.zrt --estimate-config python/zrt/training/configs/llama3_70b_3d.yaml --output my_report.xlsx

# 图模式（torch.compile）
python -m python.zrt --model-id deepseek-ai/DeepSeek-V3 --layers 2 --phases train_backward --graph-mode
python -m python.zrt --model-id deepseek-ai/DeepSeek-V3-0324 --layers 4 --phases prefill decode --graph-mode

# 向后兼容：--model 简写
python -m python.zrt --model v3
python -m python.zrt --model v3.2
```

输出文件默认保存在 `output/<model_slug>/`，每个阶段（prefill / decode / train_forward / train_backward）各生成一组文件：`_ops.xlsx`、`_raw_graph.json/.onnx`、`_fused_graph.json/.onnx`。

当使用 `--hw` 时，额外生成性能报告：
- `output/<model_slug>/reports/<slug>_<phase>_report.html` — HTML 报告
- `output/<model_slug>/reports/<slug>_<phase>_trace.json` — Chrome Trace（可在 `chrome://tracing` 中加载）

`--estimate-config` 默认输出 Excel 到 `output/estimate/<config名>_<时间戳>.xlsx`，包含 5 个 sheet：
- **Summary** — Step time, MFU, HFU, FLOPs, Memory (per GPU), Pipeline schedule
- **Ops** — 每个算子的 name, kind, layer, FLOPs, bytes, bound, latency (μs)
- **Model** — 完整模型参数（几何、MoE、HC、MLA、压缩比、dtype 等）
- **Hardware** — GPU 算力/HBM/网络 tier
- **Strategy** — 并行配置、batch、schedule、recompute、offload

### Python API

```python
from python.zrt.pipeline import run_trace_phases

# 推荐：prefill + decode 一次完成
result = run_trace_phases(
    model_id="deepseek-ai/DeepSeek-V3-0324",
    num_layers=4,
    batch_size=1,
    seq_len=128,
    phases=("prefill", "decode"),
)

# 访问 OpGraph IR
raw_graph, fused_graph = result.graphs["prefill"]
print(f"prefill 原始图: {raw_graph}")
print(f"prefill 融合图: {fused_graph}")

# 算子记录（list[dict]）
records = result.phase_records["prefill"]
print(f"捕获算子数: {len(records)}")
print(f"输出目录:   {result.output_dir}")
```

```python
from python.zrt.pipeline import run_trace

# 单阶段（向后兼容）
output_dir, records = run_trace(
    model_id="deepseek-ai/DeepSeek-V3-0324",
    num_layers=4,
    phase="prefill",
)
```

---

## HTTP 服务接口

将 CLI 的三种运行模式封装为 RESTful 服务，所有耗时任务（图捕获、性能建模、网格搜索）均以**异步后台任务**运行，提交后立即返回 `job_id`，通过轮询获取结果。

### 启动服务

```bash
# 从项目根目录启动（默认 8000 端口）
uvicorn server.main:app --host 0.0.0.0 --port 8000

# 开发模式（自动重载）
uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

启动后访问交互式 API 文档：`http://localhost:8000/docs`

### 接口总览

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 服务状态检查 |
| `GET` | `/hardware` | 可用硬件规格列表 |
| `GET` | `/models` | 本地模型简写列表（可用于 `local:<name>`） |
| `GET` | `/jobs` | 全部任务列表 |
| `GET` | `/jobs/{job_id}` | 轮询任务状态与结果 |
| `POST` | `/trace` | 提交图捕获任务（可选接性能建模） |
| `POST` | `/estimate` | 提交 Spec-based 训练估算任务 |
| `POST` | `/search` | 提交并行策略网格搜索任务 |

所有 `POST` 接口返回 HTTP 202，响应体包含 `job_id`。任务完成后 `status` 字段由 `"running"` 变为 `"done"` 或 `"error"`。

### `/trace` — 图捕获 + 性能建模

等价 CLI：`python -m python.zrt --model-id ... --layers ... [--hw ...]`

**请求参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_id` | string | **必填** | HF Hub ID、本地路径，或 `local:<name>`（如 `local:v3`） |
| `layers` | int | `4` | 追踪的 Transformer Block 数 |
| `batch_size` | int | `1` | 批次大小 |
| `seq_len` | int | `128` | prefill 序列长度 |
| `phases` | list[str] | `["prefill","decode"]` | `prefill`/`decode`/`train_forward`/`train_backward` |
| `train` | bool | `false` | 等价于 `phases=["train_forward","train_backward"]` |
| `hw` | string | — | 硬件规格（触发性能报告，如 `nvidia_h100_sxm`） |
| `tp/pp/ep/dp/cp` | int | `1` | 各维并行度 |
| `quant` | string | — | `int4`/`int8`/`fp8` |
| `platform` | string | `generic` | `cuda`/`ascend_npu`/`cpu`/`generic` |
| `graph_mode` | bool | `false` | 使用 `torch.compile` 图模式 |
| `gradient_checkpointing` | bool | `false` | 启用激活重计算 |
| `output_dir` | string | — | 输出目录（默认 `output/<slug>`） |
| `target_layers` | string | — | 指定层号，逗号分隔（如 `"0,3"`） |
| `auto_layers` | bool | `true` | 自动选取首个密集层和首个 MoE 层 |
| `zero_stage` | int | `1` | ZeRO 阶段 0–3（训练建模用） |
| `optimizer` | string | `adam` | `adam`/`adamw`/`muon` |
| `micro_batch` | int | `1` | 每 GPU micro-batch 大小 |
| `global_batch` | int | `32` | 全局 batch 大小 |
| `total_params` | float | — | 完整模型参数量（如 `671e9`） |
| `hidden` | int | `7168` | 隐藏层维度（内存估算） |
| `num_layers_full` | int | — | 完整模型总层数 |

**示例：**

```bash
# 推理抓图（prefill + decode）
curl -X POST http://localhost:8000/trace \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "Qwen/Qwen2.5-7B-Instruct",
    "layers": 4
  }'

# 推理抓图 + 性能建模（TP=8）
curl -X POST http://localhost:8000/trace \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "deepseek-ai/DeepSeek-V3-0324",
    "layers": 4,
    "hw": "nvidia_h100_sxm",
    "tp": 8
  }'

# 训练建模（3D 并行）
curl -X POST http://localhost:8000/trace \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "local:v3",
    "layers": 4,
    "train": true,
    "hw": "nvidia_h100_sxm",
    "tp": 8,
    "pp": 4,
    "ep": 2,
    "dp": 2,
    "total_params": 671e9,
    "num_layers_full": 61
  }'
```

**响应体（提交时）：**

```json
{
  "id": "3f8a2c1d-...",
  "status": "pending",
  "result": null,
  "error": null,
  "created_at": "2025-05-06T10:00:00+00:00",
  "finished_at": null
}
```

**结果体（完成时 `result` 字段）：**

```json
{
  "output_dir": "output/Qwen2.5_7B_Instruct",
  "phases": ["prefill", "decode"],
  "summary": "..."
}
```

### `/estimate` — Spec-based 训练估算

等价 CLI：`python -m python.zrt --estimate-config <yaml>`

提供 `config_path`（服务端文件路径）或 `config_content`（YAML 字符串）之一。

```bash
# 通过文件路径
curl -X POST http://localhost:8000/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "python/zrt/training/configs/llama3_70b_3d.yaml"
  }'

# 通过内联 YAML（适合跨机器调用）
curl -X POST http://localhost:8000/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "config_content": "model:\n  name: llama3_70b\n..."
  }'
```

**结果体：**

```json
{
  "summary": "====================\nTraining Estimation...",
  "data": {
    "step_time_ms": 1234.5,
    "mfu": 0.512,
    "hfu": 0.538,
    "memory": { "weights_gb": 140.0, "total_gb": 312.0 }
  }
}
```

### `/search` — 并行策略网格搜索

等价 CLI：`python -m python.zrt --search-config <yaml>`

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "python/zrt/training/configs/llama3_70b_3d.yaml",
    "output": "output/pareto_frontier.json"
  }'
```

**结果体：**

```json
{
  "total_configs": 128,
  "pareto_count": 5,
  "pareto_frontier": [
    { "step_time_ms": 980.0, "mfu": 0.53, "hfu": 0.56, ... },
    ...
  ]
}
```

### 轮询任务状态

```bash
# 轮询（替换为实际 job_id）
curl http://localhost:8000/jobs/3f8a2c1d-...

# 查询所有任务
curl http://localhost:8000/jobs
```

**`status` 字段取值：**

| 值 | 含义 |
|----|------|
| `pending` | 已提交，等待执行 |
| `running` | 正在执行 |
| `done` | 执行完成，`result` 有效 |
| `error` | 执行失败，`error` 字段包含错误信息 |

### Python 客户端示例

```python
import time
import requests

BASE = "http://localhost:8000"

# 提交任务
resp = requests.post(f"{BASE}/trace", json={
    "model_id": "Qwen/Qwen2.5-7B-Instruct",
    "layers": 4,
    "hw": "nvidia_h100_sxm",
    "tp": 4,
})
job_id = resp.json()["id"]

# 轮询直到完成
while True:
    job = requests.get(f"{BASE}/jobs/{job_id}").json()
    if job["status"] in ("done", "error"):
        break
    time.sleep(5)

if job["status"] == "done":
    print(job["result"]["summary"])
else:
    print("Failed:", job["error"])
```

---

## 训练抓图

在推理抓图（prefill / decode）之外，工具同样支持**训练阶段**的算子追踪：捕获前向传播中 `model.train()` 特有的算子（dropout、batch norm 等），并可继续追踪 `loss.backward()` 触发的梯度算子。

### 训练阶段说明

| 阶段 | 含义 | 梯度 | model 模式 |
|------|------|------|------------|
| `train_forward` | 训练前向（含 dropout 等训练特有算子）| 开启 | `train()` |
| `train_backward` | 前向 + backward（包含梯度算子）| 开启 | `train()` |
| `train` | `train_forward` 的简写别名（`--phases` 中可用） | — | — |

训练阶段与推理阶段完全独立，可以在同一次调用中混合使用，各自生成独立的输出文件。

---

## 训练性能建模

在 `--train` 基础上叠加 `--hw` 即可在同一入口执行**训练性能建模**（FLOPs、MFU、内存、通信量、1F1B 流水线调度），无需切换到独立的训练 CLI。

### 报告内容

训练建模报告包含：

- **Step time**：含流水线调度的完整 step 耗时
- **MFU / HFU**：Model / Hardware FLOPs Utilization（HFU 含重计算开销）
- **FLOPs 分解**：前向 / 反向 / 总计
- **内存（每 GPU）**：权重 / 梯度 / 优化器状态 / 激活值 / 通信缓冲
- **流水线指标**：warmup / steady / cooldown 步数、bubble 占比
- **时延分解**：见下表

### 时延分解字段

Step time 按正交维度拆解，所有字段满足严格恒等式（单位 ms）：

| 字段 | 含义 | 恒等式 |
|------|------|--------|
| `step_time_ms` | 完整单步耗时 | `= pipeline_time + optimizer_time + optimizer_comm` |
| `pipeline_time_ms` | 流水线时间 | `= compute_time + exposed_comm` |
| `compute_time_ms` | 关键路径纯计算 | — |
| `exposed_comm_ms` | 关键路径通信（暴露，阻塞计算）| `= Σ *_exposed_ms` |
| `hidden_comm_ms` | 非关键路径通信（掩盖，与计算重叠）| `= Σ *_hidden_ms` |
| `total_comm_volume_ms` | 总通信量 | `= exposed_comm + hidden_comm` |
| `optimizer_time_ms` | 优化器步计算 | — |
| `optimizer_comm_ms` | 优化器通信（如 Muon AG+RS）| — |

各并行组的暴露 / 掩盖拆分（`exposed_comm = Σ`，`hidden_comm = Σ`）：

| 并行组 | 暴露字段 | 掩盖字段 | 掩盖机制 |
|--------|---------|---------|---------|
| TP | `tp_exposed_ms` | `tp_hidden_ms` | CoC（保留 10%）/ MC2（全掩盖）|
| CP | `cp_exposed_ms` | — | — |
| EP | `ep_exposed_ms` | `ep_hidden_ms` | wave-overlap（Expert GEMM 与 A2A 重叠）|
| PP | `pp_exposed_ms` | — | — |
| DP | `dp_exposed_ms` | `dp_hidden_ms` | 流水线 bubble 期间执行 AR/RS |

### Python API

```python
from python.zrt.graph import run_trace_phases
from python.zrt.transform.analysis.modeller import estimate_training_from_graphs
import python.zrt.hardware.registry as hw_registry

# Step 1: 抓训练图
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

---

## 图模式（torch.compile）

默认的 eager 路径通过 `TorchDispatchMode` 逐 op 拦截，产生扁平的算子列表。  
`--graph-mode` 改用 **`torch.compile` 自定义 backend** 捕获计算图，Dynamo 在 trace 时生成带显式数据流边的 FX `GraphModule`。

### 与 eager 模式的对比

| 特性 | eager（默认） | `--graph-mode` |
|------|--------------|-----------------|
| 捕获机制 | `TorchDispatchMode` | `torch.compile` + Dynamo |
| 数据流边 | 后验推断（tensor id 匹配）| FX `node.args` 直接读取 |
| GEMM 表示 | `mm` / `addmm` 分散 | `aten.linear.default` 统一 |
| 模块路径标注 | 完整（forward hook）| 依赖 Dynamo `nn_module_stack`（保留） |
| 训练反向捕获 | `loss.backward()` eager 拦截 | Dynamo 自动捕获反向子图 |
| 子图数量 | 单一扁平序列 | 可能多个子图（graph break 时合并）|
| 输出格式 | Excel + JSON + ONNX | **相同**（完全兼容）|

### Python API

```python
from python.zrt.graph import run_trace_phases

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
```

### 注意事项

- 模型含复杂控制流（如 MoE routing）时 Dynamo 可能产生多个子图（graph break），records 会按顺序合并。
- `--graph-mode` 与 `--train` / `--phases` 等所有现有参数完全兼容。

---

## 命令行参数全集

### 运行模式（互斥）

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--estimate-config` | — | str | — | 从 YAML 配置执行单点训练估算（无图捕获） |
| `--search-config` | — | str | — | 网格搜索并行策略（输出 Pareto 前沿） |
| `--output` | — | str | — | 将估算/搜索结果写入 JSON 文件（配合上述两个参数使用） |

> `--estimate-config` 和 `--search-config` 互斥，只能选其一。

### 模型与输入

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--model-id` | — | str | **必填** | HF Hub 模型 ID 或本地路径（如 `deepseek-ai/DeepSeek-V3-0324`） |
| `--model` | — | choice | — | 向后兼容简写：`v3` 或 `v3.2`（映射到 `hf_models/deepseek_v3` / `hf_models/deepseek_v3_2`） |
| `--layers` | — | int | `4` | 追踪的 Transformer Block 数量（2–4 即可覆盖所有算子模式） |
| `--batch-size` | — | int | `1` | dummy 输入的 batch size |
| `--seq-len` | — | int | `128` | prefill 序列长度 |
| `--target-layers` | — | str | — | 指定追踪的层编号，逗号分隔（如 `0,3`） |
| `--auto-layers` | — | flag | `False` | 自动选择第一个密集层和第一个稀疏（MoE）层 |

> `--target-layers` 与 `--auto-layers` 互斥。当两者都不指定时，默认启用 auto-layers 行为。
> `--model-id` 和 `--model` 任选其一，至少指定一个。

### 阶段与模式

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--phases` | — | choice+ | `prefill decode` | 追踪的阶段列表：`prefill`、`decode`、`forward`、`train`、`train_forward`、`train_backward` |
| `--train` | — | flag | `False` | 快捷 flag：等价于 `--phases train_forward train_backward` |
| `--platform` | — | choice | `generic` | 目标平台：`cuda` / `ascend_npu` / `cpu` / `generic`，影响融合 kernel 命名；指定 `--hw` 时自动推断，无需手动设置 |
| `--graph-mode` | — | flag | `False` | 使用 `torch.compile` 图模式捕获，替代 `TorchDispatchMode` eager 路径 |
| `--gradient-checkpointing` | — | flag | `False` | 启用激活重计算（训练阶段） |

### 输出

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--output-dir` | `-o` | str | `output/<model_slug>` | 输出目录路径 |

### 并行策略（推理 transforms + 训练建模通用）

以下参数同时适用于推理性能分析和训练建模：

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--tp` | — | int | `1` | 张量并行度 |
| `--pp` | — | int | `1` | 流水线并行度 |
| `--ep` | — | int | `1` | 专家并行度 |
| `--dp` | — | int | `1` | 数据并行度 |
| `--cp` | — | int | `1` | 上下文并行度 |
| `--quant` | — | choice | — | 权重/激活量化精度：`int4` / `int8` / `fp8`（默认不量化） |

> 推理模式下，这些参数影响 transform pipeline 的并行模拟（通信算子注入、FLOPs 拆分等）。

#### 并行策略约束规则（`--estimate-config` 模式，参考 Megatron 实现）

**总卡数公式**

```
TP × CP × PP × DP = world_size（nodes × gpus_per_node）
```

EP 不占用独立 rank，在 DP group 内部运行，不参与上述乘积。

**约束汇总**

| 维度 | 规则 | 级别 |
|------|------|------|
| 总卡数 | `TP × CP × PP × DP = world_size` | 错误 |
| TP | `num_heads % TP == 0` | 错误 |
| TP | `num_kv_heads % TP == 0`（当 `num_kv_heads >= TP` 时）| 错误 |
| TP | `ffn % TP == 0` | 错误 |
| TP | `TP > gpus_per_node`（跨节点 TP，性能损失严重）| 警告 |
| EP | `num_experts % EP == 0` | 错误 |
| EP | `DP % EP == 0`（EP group 必须是 DP group 的子集）| 警告 |
| EP | `EP > gpus_per_node`（跨节点 A2A）| 警告 |
| PP | `PP ≤ num_layers` | 错误 |
| PP | `num_layers % PP == 0`（否则各 stage 层数不均）| 警告 |
| PP | `num_microbatches ≥ PP`，其中 `num_microbatches = global_batch / (micro_batch × DP)` | 警告 |
| ZeRO | `zero_stage ≥ 1` 时要求 `DP > 1` | 错误 |
| batch | `global_batch % (micro_batch × DP) == 0` | 错误 |
| VPP | `vpp_chunks > 1` 时要求 `pp_schedule = i1f1b` | 警告 |
| VPP | `num_layers % (PP × vpp_chunks) == 0` | 警告 |
| CP(Ulysses) | `(num_heads // TP) % CP == 0` | 警告 |
| CP(Ring) | `seq_len % (CP × 128) == 0` | 警告 |
| CP(Hybrid) | 同时满足 Ulysses + Ring 两条规则 | 警告 |
| CP | `CP > gpus_per_node`（跨节点 CP 通信）| 警告 |
| CP + EP | 两者同时存在时 A2A pattern 需人工确认 | 警告 |

### 硬件规格

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--hw` | — | str | — | 硬件规格名称，用于性能报告。指定后触发 transform + 调度 + 报告生成 |

#### 可用硬件

| 名称 | 描述 |
|------|------|
| `nvidia_h100_sxm` | NVIDIA H100 SXM5 80GB |
| `nvidia_h800` | NVIDIA H800 80GB |
| `nvidia_a100_80g` | NVIDIA A100 80GB SXM |
| `ascend_910b` | 华为昇腾 910B |
| `ascend_910c` | 华为昇腾 910C |

### 训练建模专属参数（`--train --hw` 时生效）

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--zero-stage` | — | int | `1` | ZeRO 优化阶段（0–3） |
| `--optimizer` | — | choice | `adam` | 优化器：`adam` / `adamw` / `muon` |
| `--muon-rotation` | — | flag | `True` | 启用 Moonshot rotation 优化（Muon 专用） |
| `--muon-ns-steps` | — | int | `5`（DSV4: `10`） | Newton-Schulz 迭代步数（Muon 专用） |
| `--micro-batch` | — | int | `1` | 每 GPU micro-batch 大小 |
| `--global-batch` | — | int | `32` | 全局 batch 大小（跨 DP ranks） |
| `--total-params` | — | float | — | 完整模型参数量（如 `671e9`，用于缩放 traced layers） |
| `--hidden` | — | int | `7168` | 隐藏层维度（内存估算用） |
| `--num-layers-full` | — | int | — | 完整模型总层数（默认等于 `--layers`） |

### `--layers` 选择建议

| 场景 | 推荐值 |
|------|--------|
| 纯密集模型（Llama / Qwen2 / Mistral） | `2` |
| DeepSeek-V3（前 3 层为密集层，第 4 层起为 MoE） | `4` |
| Qwen3-MoE / Mixtral（第 1 层即为 MoE） | `2` |

---

## 支持的模型

### 开箱即用（在 transformers 注册表中）

| 架构 | 示例模型 ID | 注意 |
|------|------------|------|
| LLaMA / LLaMA-2 / LLaMA-3 | `meta-llama/Llama-3.1-8B` | 需 HF 授权 |
| Qwen2 / Qwen2.5 | `Qwen/Qwen2.5-7B-Instruct` | — |
| Qwen3 (dense) | `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-8B` | — |
| Qwen3 (MoE) | `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-235B-A22B` | — |
| Mistral | `mistralai/Mistral-7B-v0.1` | — |
| Mixtral (MoE) | `mistralai/Mixtral-8x7B-v0.1` | — |
| Gemma / Gemma2 | `google/gemma-2-9b` | — |
| Phi-3 / Phi-4 | `microsoft/Phi-4` | — |

### 需要 `trust_remote_code`（自动处理）

| 架构 | 示例模型 ID | 特殊处理 |
|------|------------|----------|
| DeepSeek-V3 | `deepseek-ai/DeepSeek-V3` | MoE meta patch |
| DeepSeek-V3-0324 (V3.2) | `deepseek-ai/DeepSeek-V3-0324` | MoE meta patch + Indexer patch |
| DeepSeek-V4 | `hf_models/deepseek_v4`（本地） | MoE meta patch；推理路径 `rotate_activation` noop patch；CUDA 下 Attention 融合为 `v4_q_norm` / `v4_kv_quant` / `v4_sparse_attn`；Ascend NPU 下融合为 `npu_sas` |

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
output_base = Path("output/Qwen2.5-7B-Instruct")

# Step 2: Transform + 导出（TP=1 baseline）
output_dir, transformed_graph = run_transform(
    raw_graph=raw_graph,
    output_dir=output_base,
    parallel_config=ParallelConfig(tp=1),
    stream_config=StreamConfig(num_compute_streams=1, num_comm_streams=1),
    hw_spec=hw_spec,
)

# Step 3: Transform + 导出（TP=4 + EP=2 + DP=2）
output_dir_multi = Path("output/Qwen2.5-7B-Instruct_multi")
_, _ = run_transform(
    raw_graph=raw_graph,
    output_dir=output_dir_multi,
    parallel_config=ParallelConfig(tp=4, ep=2, dp=2),
    stream_config=StreamConfig(num_compute_streams=1, num_comm_streams=1),
    hw_spec=hw_spec,
)
```

### `run_transform()` 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `raw_graph` | OpGraph | — | 从 `run_trace_phases()` 获取的原始图 |
| `output_dir` | Path | — | 输出目录（自动创建） |
| `parallel_config` | ParallelConfig | `ParallelConfig()` | TP/EP/PP/DP/CP/SP 配置 |
| `stream_config` | StreamConfig | `StreamConfig(1, 1)` | 计算流数 + 通信流数 |
| `pipeline` | TransformPipeline | `build_default_pipeline()` | 自定义 transform 管道 |
| `hw_spec` | HardwareSpec | `A100_40GB` | GPU 规格（用于 FLOPs 估算） |

### 导出文件详解

#### Excel：`*_transformed_ops.xlsx`（5 个 Sheet）

| Sheet | 内容 |
|-------|------|
| **Metadata** | 图的基本信息 + 并行策略 + 流配置 |
| **Transformed Operators** | 所有算子（注入了 FLOPs / 延迟 / bound / stream_id） |
| **Communication Ops** | 通信算子详情（all-reduce / all-to-all / send-recv） |
| **Parallelism Summary** | 按层统计计算/通信/内存算子数 + 并行类型 |
| **Stream Assignment** | 流分配统计（哪些节点在 compute/comm stream） |

#### JSON：`*_transformed_graph.json`

结构化数据（用于自动化分析），包含 `graph`、`parallelism`、`stream_config`、`nodes`、`edges` 字段。

### ParallelConfig 常见配置

```python
from python.zrt.transform import ParallelConfig

# 单卡（baseline）
ParallelConfig(tp=1)

# TP=4（4卡张量并行）
ParallelConfig(tp=4)

# TP=8 + EP=2（8卡 TP，2卡专家并行）
ParallelConfig(tp=8, ep=2)

# TP=4 + PP=2（4卡 TP，2阶段流水线）
ParallelConfig(tp=4, pp=2)

# TP=4 + DP=2（4卡 TP，2卡数据并行）
ParallelConfig(tp=4, dp=2)

# TP=4 + EP=2 + DP=2 + CP=2
ParallelConfig(tp=4, ep=2, dp=2, cp=2)

# 序列并行（需 transformers 5.0+ 支持）
ParallelConfig(tp=4, sp=True)

# 获取配置描述
cfg = ParallelConfig(tp=4, ep=2, pp=2)
print(cfg.describe())  # "TP4-EP2-PP2"
```

---

## 输出 Excel 说明

抓图阶段（不含 `--hw`）生成的 Excel 工作簿包含 6 个 Sheet：

| Sheet | 说明 |
|-------|------|
| Model Config | 模型配置摘要 |
| Fused Operators | 融合后的算子序列（主视图，含融合 I/O 映射） |
| Raw Operator Sequence | 原始 aten 算子完整序列 |
| Summary | 按融合算子聚合统计 |
| By Layer | 按层级聚合统计 |
| Fusion Rules | 自动发现的融合模式（含融合 I/O 映射） |

同时生成 `*_fusion_rules.json`，记录自动发现的算子融合模式。

### Sheet：Model Config

模型关键配置一览，包含（有则展示，无则跳过）：

| 字段 | 说明 |
|------|------|
| `model_id` | 模型来源 |
| `model_type` | 架构类型 |
| `hidden_size` | 隐藏层维度 |
| `num_hidden_layers` | 总层数 / 追踪层数 |
| `num_attention_heads` | 注意力头数 |
| `vocab_size` | 词表大小 |
| `n_routed_experts` | MoE 路由专家数（DeepSeek） |
| `num_local_experts` | MoE 本地专家数（Mixtral） |
| `q_lora_rank` / `kv_lora_rank` | MLA 低秩维度（DeepSeek） |

### Sheet：Fused Operators（主视图）

两阶段自动融合后的算子序列，每行包含：

- **Fused Input Shapes/Dtypes**：融合 kernel 的外部输入张量信息
- **Input Sources**：每个输入来自哪个子算子的哪个输入端口
- **Fused Output Shapes/Dtypes**：融合 kernel 的外部输出张量信息
- **Output Sources**：每个输出由哪个子算子的哪个输出端口产生

---

## 组件标签体系

`Component` 列的标签由模块路径中的命名模式推断，与具体模型实现无关：

| 标签前缀 | 含义 | 颜色 |
|----------|------|------|
| `attn_norm` | Attention 前的 LayerNorm / RMSNorm | 绿 |
| `ffn_norm` | Attention 后的 LayerNorm / RMSNorm | 绿 |
| `final_norm` | 最终 Norm 层 | 绿 |
| `attn.q_proj` / `attn.k_proj` / … | 标准 QKV 及输出投影 | 蓝 |
| `attn.q_a_proj` / `attn.kv_a_proj` / … | MLA 低秩投影（DeepSeek） | 蓝 |
| `attn.score` | QK 内积计算 | 蓝 |
| `attn.softmax` | Softmax | 蓝 |
| `attn.rope` | RoPE 位置编码 | 蓝 |
| `moe.gate.*` | MoE 路由 / 门控 | 橙 |
| `moe.shared.*` | MoE 共享专家（DeepSeek） | 黄 |
| `moe.experts.*` | MoE 路由专家 MLP | 粉 |
| `ffn.gate_proj` / `ffn.up_proj` / `ffn.down_proj` | 密集 FFN 投影 | 紫 |
| `ffn.silu` / `ffn.mul` | 激活函数 | 紫 |
| `embedding` | Token Embedding | 灰 |
| `lm_head` | 语言模型输出头 | 灰 |

---

## V3 vs V3.2 对比

| 特性 | V3 | V3.2 |
|------|----|------|
| 原始算子数 (4层) | 400 | 468 |
| 融合后算子数 | 75 | 87 |
| 融合模式数 | 6 | 7 |
| Indexer 模块 | 无 | 有 (MLA 注意力中新增) |

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

融合算法集中在 `transform/fusion/` 下（`core.py` 四遍算法、`rules.py` 平台规则、`_dict_bridge.py` Dict ↔ IR 桥接），`graph/` 模块只负责抓图，不再包含融合逻辑。

#### 平台特定融合规则（CUDA）

DeepSeek-V4 的 `Attention.forward` 在子模块调用之间有大量内联计算（q 归一化、RoPE、kv 量化等），无法通过第二阶段 parent merge 合并（因为 compressor/indexer 含子孙）。CUDA 平台通过三个 SubPattern（priority=46）精确标注这些内联段：

| `op_type` | 关键算子序列 | 含义 |
|-----------|------------|------|
| `v4_q_norm` | `square → rsqrt → view_as_complex` | 内联 q 第二 RMSNorm + RoPE |
| `v4_kv_quant` | `view_as_complex → amax\|clamp_min` | kv RoPE + act_quant + 窗口 topk + cache 写入 |
| `v4_sparse_attn` | `gather → mm\|bmm → softmax → mm\|bmm` | 核心稀疏注意力（`sparse_attn` 内核） |

每层预期 fusion 分布（4 层 decode trace）：
```
v4_q_norm:       4  （每层 1 个）
v4_kv_quant:     4  （每层 1 个）
v4_sparse_attn:  4  （每层 1 个）
attn:            1  （layer 2 仅 3 ops：cat+to.dtype+copy_，Indexer topk 合并残留，可接受）
```

#### 平台特定融合规则（Ascend NPU）

DeepSeek-V4 在昇腾 NPU 上使用 `npu_sparse_attn_sharedkv` 内核，对应三种 Attention 变体：

| `attn_type` | `compress_ratio` | 含义 |
|-------------|-----------------|------|
| `SWA` | 0 | Sliding Window Attention |
| `CSA` | 4 | Compressed Sparse Attention（4× 压缩） |
| `HCA` | 128 | Heavily Compressed Attention（128× 压缩） |

融合后节点 `op_type="npu_sas"`，`FusionPass` 自动从 `graph.metadata["compress_ratios"]` 读取 V4 config 并注解每个节点的 `attn_type` 与 `compress_ratio`。

### 过滤掉的算子

以下零开销算子默认跳过：

`view`, `reshape`, `expand`, `permute`, `contiguous`, `slice`, `unsqueeze`, `squeeze`, `split`, `clone`, `arange`, `zeros`, `ones`, `full`, `tril`, `triu`, …

---

## 项目结构

```
modeling/
├── requirements.txt                 # 核心依赖
├── server/                          # HTTP 服务（FastAPI）
│   ├── main.py                      # FastAPI 应用 + 路由 + 后台任务
│   ├── schemas.py                   # Pydantic 请求/响应模型
│   └── requirements.txt             # 服务依赖（fastapi + uvicorn）
│
├── python/zrt/
│   ├── __main__.py                  # CLI 入口: python -m python.zrt
│   ├── cli.py                       # CLI 参数解析
│   ├── graph/                       # 图追踪模块（抓原始图）
│   │   ├── __init__.py              # 公开 API: run_trace / run_trace_phases / load_model
│   │   ├── main.py                  # run_trace_phases 实现
│   │   ├── model_loader.py          # 通用 HF 模型加载 + 兼容性修补
│   │   ├── dispatch.py              # RecordingDispatch + TensorTracker（aten 拦截）
│   │   ├── tracker.py               # ModuleTracker（forward hooks）
│   │   ├── classifier.py            # 组件分类 + 颜色映射
│   │   ├── graph_builder.py         # build_op_graph / build_fused_op_graph
│   │   ├── graph_exporter.py        # 导出 JSON / ONNX（原始图）
│   │   ├── patches.py               # 运行时 patch（MoE、Indexer、V4 推理 stubs、legacy 属性）
│   │   ├── compat.py                # transformers 版本 shim + 本地模型注册表
│   │   └── tensor_utils.py          # 张量工具 + SKIP_OPS
│   │
│   ├── report/                      # 报告生成
│   │   ├── excel_writer.py          # Excel 工作簿（抓图阶段原始输出）
│   │   └── …
│   │
│   ├── transform/                   # Transform 管道模块（融合 + 并行 + 分析）
│   │   ├── __init__.py              # 轻量 API 导出（context / pipeline / passes）
│   │   ├── context.py               # TransformContext / ParallelConfig / StreamConfig
│   │   ├── pipeline.py              # TransformPipeline / build_default_pipeline
│   │   ├── fusion/                  # 算子融合（所有平台规则集中于此）
│   │   │   ├── core.py              # 四遍融合算法（Pass 1-4）
│   │   │   ├── rules.py             # 平台 SubPattern（cuda / ascend_npu / cpu / generic）
│   │   │   ├── pass_.py             # FusionPass（OpGraph IR）+ npu_sas 注解
│   │   │   └── _dict_bridge.py      # fuse_records()：Dict 记录 ↔ FusionItem 桥接
│   │   ├── parallel/                # TP / EP / PP / DP / CP passes
│   │   ├── analysis/                # FLOPs / Roofline / 通信延迟 / 训练建模
│   │   ├── optim/                   # Quant / EPLB / SharedExpert / MTP
│   │   ├── exporter.py              # export_transformed_graph / export_full_report
│   │   └── …
│   │
│   ├── executor/                    # 执行调度模块
│   │   ├── dag_scheduler.py         # DAGScheduler（拓扑排序 + 流调度）
│   │   └── …
│   │
│   ├── simulator/                   # 性能模拟
│   │   └── …
│   │
│   ├── report/                      # 报告生成
│   │   └── …
│   │
│   ├── hardware/                    # 硬件规格
│   │   ├── spec.py                  # HardwareSpec 数据类
│   │   ├── gpu.py                   # GPU_SPECS
│   │   ├── registry.py              # hw_registry.load("nvidia_h100_sxm")
│   │   └── configs/                 # 硬件配置 YAML (H100, A100, Ascend 910B/C, ...)
│   │
│   ├── ir/                          # 中间表示（IR）
│   │   ├── graph.py                 # OpGraph
│   │   ├── node.py                  # OpNode
│   │   ├── edge.py                  # OpEdge
│   │   └── tensor.py                # Tensor（形状、dtype、内存占用）
│   │
│   └── training/                    # 训练性能建模
│       ├── io/config_loader.py      # YAML 加载
│       ├── configs/models/          # 模型规格 YAML
│       ├── configs/*.yaml           # 训练配置
│       └── ...                      # search/ spec/ compose/ 模块
│
└── hf_models/                       # 模型本地副本（只读！）
    ├── deepseek_v3/
    ├── deepseek_v3_2/
    ├── deepseek_v4/                 # DeepSeek-V4（SWA/CSA/HCA 三种 Attention）
    ├── llama3_8b/
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
run_transform()  ← transform_runner.py
       ↓ (应用 transform pipeline)
 transformed     ← pipeline.py
 OpGraph         ← passes/
       ↓
 export 结果     ← exporter.py
       ↓
  Excel/JSON     输出: metadata + operators + communication + parallelism + streams

       ↓ (可选，当 --hw 指定时)
  DAGScheduler   ← executor.py (调度)
       ↓
  性能报告       ← report.py (TTFT / TPOT / MFU / 内存)
```

---

## 自验与测试

### 运行测试

```bash
# 安装测试依赖
pip install pytest

# 运行全量测试（428 个用例）
python -m pytest tests/ -v 2>&1 | tail -n 50

# 运行单组测试
python -m pytest tests/test_transform.py -v
python -m pytest tests/test_executor.py -v -k "overlap"
python -m pytest tests/training/test_captured_graph_modelling.py -v
python -m pytest tests/training/test_1f1b.py -v
```

### 测试覆盖

| 测试组 | 内容 |
|--------|------|
| `TestExtractLayerIdx` | 层编号提取逻辑 |
| `TestClassifyComponent` | 组件分类（norm / attn / MLA / MoE / FFN / embedding / lm_head） |
| `TestMoEDetection` | MoE 模块检测与 patch 替换 |
| `test_local_model[*]` | 7 个本地模型端到端追踪（无网络） |
| `test_moe_components_present` | MoE 算子出现验证（DeepSeek-V3、Mixtral） |
| `test_deepseek_v3_mla_components` | MLA 专属算子出现验证 |
| `test_layer_attribution` | Block 内算子均有正确层号 |
| `test_config_summary_fields` | Config 摘要必填字段完整性 |
| `test_hub_model[*]` | HF Hub 端到端：DeepSeek-V3.2 / Qwen3 / Llama-3.1 |
| `test_hub_moe_detection` | Hub 模型 MoE/非MoE 检测准确性 |
