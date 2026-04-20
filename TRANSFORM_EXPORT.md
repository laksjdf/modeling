# Transform Pipeline + 导出快速指南

本文档说明如何生成并自验 `transform_runner.py` 导出的文件。

---

## 🚀 三种自验方式

### 方式一：完整 e2e 检查（推荐用于开发）

**命令**：
```bash
python e2e_check.py
```

**执行步骤**：
1. 抓图（prefill 阶段，4层）
2. Transform TP=1（单卡 baseline）
3. Transform TP=4（4卡张量并行）
4. DAGScheduler 调度（TP=1 和 TP=4）
5. 性能报告（TTFT / TPOT / MFU / 内存）
6. **导出 Excel/JSON**（Step 8 新增）

**输出位置**：
```
output/graph/Qwen2.5-7B-Instruct/
├── Qwen2.5-7B-Instruct_transformed_ops.xlsx
└── Qwen2.5-7B-Instruct_transformed_graph.json
```

**Excel 包含的 5 个 Sheet**：
- **Metadata**：图的基本信息 + 并行配置 + 流配置
- **Transformed Operators**：所有算子（FLOPs / 延迟 / bound / stream_id）
- **Communication Ops**：通信算子（all-reduce / all-to-all）
- **Parallelism Summary**：按层统计
- **Stream Assignment**：流分配统计

### 方式二：按需导出（纯 Transform，无调度）

```python
from python.zrt.graph import run_trace_phases
from python.zrt.graph.transform_runner import run_transform
from python.zrt.transform import ParallelConfig, StreamConfig
import python.zrt.hardware.registry as hw_registry
from pathlib import Path

# 步骤 1：抓原始图
result = run_trace_phases(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    num_layers=4,
    batch_size=1,
    seq_len=128,
    phases=("prefill",),
)
raw_graph, _ = result.graphs["prefill"]

# 步骤 2：获取硬件规格
hw = hw_registry.load("nvidia_h100_sxm")

# 步骤 3a：导出 TP=1
output_dir_tp1 = Path("output/graph/Qwen2.5-7B-Instruct_tp1")
_, _ = run_transform(
    raw_graph=raw_graph,
    output_dir=output_dir_tp1,
    parallel_config=ParallelConfig(tp=1),
    stream_config=StreamConfig(num_compute_streams=1, num_comm_streams=1),
    hw_spec=hw,
)
print(f"✓ 导出 TP=1 到: {output_dir_tp1}")

# 步骤 3b：导出 TP=4（推荐不同的输出目录）
output_dir_tp4 = Path("output/graph/Qwen2.5-7B-Instruct_tp4")
_, _ = run_transform(
    raw_graph=raw_graph,
    output_dir=output_dir_tp4,
    parallel_config=ParallelConfig(tp=4),
    stream_config=StreamConfig(num_compute_streams=1, num_comm_streams=1),
    hw_spec=hw,
)
print(f"✓ 导出 TP=4 到: {output_dir_tp4}")
```

### 方式三：命令行（仅抓图，自动 transform）

```bash
# 基础：抓图 + 自动应用默认 transform
python -m python.zrt.graph.main Qwen/Qwen2.5-7B-Instruct --layers 4

# 指定硬件（用于 FLOPs 估算）
python -m python.zrt.graph.main Qwen/Qwen2.5-7B-Instruct --layers 4 --hw nvidia_h100_sxm

# 测试 TP=4
python -m python.zrt.graph.main Qwen/Qwen2.5-7B-Instruct --layers 4 --tp 4 --hw nvidia_h100_sxm

# 自定义输出目录
python -m python.zrt.graph.main Qwen/Qwen2.5-7B-Instruct --layers 4 -o output/custom_dir
```

---

## 📊 Excel 文件说明

### Sheet 1：Metadata

| 字段 | 说明 |
|------|------|
| Graph Name | 图的标识符 |
| Phase | prefill / decode |
| Total Nodes | 图中的总节点数 |
| Total Edges | 图中的总边数 |
| TP / EP / PP / DP / SP | 并行配置 |
| Strategy Description | 完整的并行描述（如 "TP=4, EP=1, PP=1, DP=1, SP=False"） |
| Compute Streams | 计算流数 |
| Comm Streams | 通信流数 |

### Sheet 2：Transformed Operators（主表）

| 列 | 说明 |
|---|---|
| Node ID | 节点唯一标识（如 `op_0`, `op_42`, `comm_allreduce_0`） |
| Op Type | 操作类型（如 `mm`, `softmax`, `all_reduce`） |
| Category | `compute` \| `communication` \| `memory` |
| Scope | Module 路径（如 `layers.0.self_attn.q_proj`） |
| Layer | 层索引（0, 1, 2, ...） |
| Component | 组件分类（attn_norm, attn.q_proj, ffn, moe, 等） |
| Parallelism Strategy | 并行策略（如 "TP=4, EP=1, ..."） |
| Collective Op | 通信类型（`all_reduce`, `all_to_all`, 等）—仅通信节点 |
| Group Size | 通信组大小（如 `4` for TP=4） |
| Role | `dispatch` / `combine` / ... |
| Pipeline Stage | Pipeline 并行时的阶段（如 `stage_0`, `stage_1`） |
| Stream Type | `compute` \| `comm` |
| Stream ID | 流的唯一 ID（如 `0`, `1`, `2`） |
| Input Shapes | 输入张量形状（如 `(1, 128, 4096)`) |
| Output Shapes | 输出张量形状 |
| Input Dtypes | 输入数据类型（如 `float32`） |
| Output Dtypes | 输出数据类型 |
| FLOPs | 理论运算量 |
| Compute (µs) | 计算耗时（微秒）= FLOPs / peak_flops_fp32 |
| Memory (µs) | 访存耗时 = (read_bytes + write_bytes) / hbm_bandwidth |
| Total Latency (µs) | 总延迟 = max(compute, memory) |
| Bound | `compute` / `memory` —哪个是性能瓶颈 |
| Arithmetic Intensity | FLOPs / 总字节数 |
| Annotations | 其他注解 |

**颜色标记**：
- 🟥 红色：通信操作
- 🟩 绿色：计算操作
- 🟧 橙色：内存操作

### Sheet 3：Communication Ops

| 列 | 说明 |
|---|---|
| Node ID | 通信节点 ID |
| Collective Op | `all_reduce` / `all_to_all` / `send_recv` |
| Role | 通信角色（`dispatch` / `combine` 等） |
| Group Size | 通信组大小 |
| Scope | 在哪个模块中插入 |
| Layer | 层索引 |
| Stream Type | `compute` / `comm` |
| Stream ID | 流 ID |
| Input Shapes | 输入形状 |
| Output Shapes | 输出形状 |
| Inserted By | 由哪个 pass 插入（如 `TensorParallelPass`, `ExpertParallelPass`） |
| Data Volume (bytes) | 通信数据量 |

### Sheet 4：Parallelism Summary

按层统计，汇总该层的计算/通信/内存操作。

| 列 | 说明 |
|---|---|
| Layer | 层索引 |
| Compute Ops | 计算操作数 |
| Memory Ops | 内存操作数 |
| Comm Ops | 通信操作数 |
| Comm Types | 通信类型列表（如 `all_reduce, all_to_all`） |
| Inserted By | 插入这些通信的 pass |
| Parallelism Strategy | 该层的并行类型（如 `TP/EP`） |

### Sheet 5：Stream Assignment

统计流分配情况。

| 列 | 说明 |
|---|---|
| Stream ID | 流的唯一 ID |
| Stream Type | `compute` / `comm` |
| Assigned Nodes | 分配给该流的所有节点 ID（用分号分隔） |
| Node Count | 该流上的节点数 |

---

## 📁 JSON 文件结构

```json
{
  "graph": {
    "name": "Qwen2.5-7B-Instruct",
    "phase": "prefill",
    "num_nodes": 150,
    "num_edges": 200
  },
  "parallelism": {
    "strategy": "TP=4, EP=1, PP=1, DP=1, SP=False",
    "tp": 4,
    "ep": 1,
    "pp": 1,
    "dp": 1,
    "sp": false
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
      "attrs": {},
      "annotations": {
        "flops": 1073741824,
        "latency_us": 12.5,
        "compute_us": 12.5,
        "bound": "compute",
        "arithmetic_intensity": 42.7,
        "stream_id": 0,
        "stream_type": "compute"
      },
      "input_shapes": [[1, 128, 4096]],
      "output_shapes": [[1, 128, 4096]]
    },
    {
      "id": "comm_0",
      "op_type": "all_reduce",
      "category": "communication",
      "scope": "layers.0.self_attn",
      "layer": "0",
      "attrs": {
        "collective": "all_reduce",
        "group_size": 4,
        "role": "rank_0"
      },
      "annotations": {
        "latency_us": 150.0,
        "stream_id": 1,
        "stream_type": "comm",
        "inserted_by": "TensorParallelPass"
      },
      "input_shapes": [[1, 128, 1024]],
      "output_shapes": [[1, 128, 4096]]
    }
  ],
  "edges": [
    {"src": "op_0", "dst": "op_1", "src_idx": 0, "dst_idx": 0}
  ]
}
```

---

## 🔧 常见问题

### Q1：为什么 TP=1 和 TP=4 的输出文件名相同？

**答**：导出文件名由图的 name 决定，与并行配置无关。

**解决方案**：使用不同的输出目录
```python
output_dir_tp1 = Path("output/graph/model_tp1")
output_dir_tp4 = Path("output/graph/model_tp4")
```

### Q2：如何跨多个模型对比？

```python
models = ["Qwen/Qwen2.5-7B-Instruct", "deepseek-ai/DeepSeek-V3-0324"]

for model_id in models:
    result = run_trace_phases(model_id=model_id, num_layers=4, phases=("prefill",))
    raw_graph, _ = result.graphs["prefill"]
    run_transform(
        raw_graph=raw_graph,
        output_dir=Path(f"output/{model_id.split('/')[-1]}"),
        ...
    )
```

### Q3：如何导出其他并行配置？

```python
# TP=4 + DP=2
run_transform(raw_graph=raw_graph, parallel_config=ParallelConfig(tp=4, dp=2), ...)

# TP=8 + PP=2（管道并行）
run_transform(raw_graph=raw_graph, parallel_config=ParallelConfig(tp=8, pp=2), ...)

# TP=4 + SP=True（序列并行）
run_transform(raw_graph=raw_graph, parallel_config=ParallelConfig(tp=4, sp=True), ...)
```

### Q4：Excel 的 Annotations 列为空？

这是正常的，该列存储额外的元数据。关键指标在其他列：FLOPs、Compute (µs)、Total Latency (µs)、Bound。

---

## ✅ 验证清单

- [ ] `e2e_check.py` 全部通过（无断言错误）
- [ ] Step 8 输出：✓ 导出文件完整性检查通过
- [ ] 输出目录存在：`*_transformed_ops.xlsx` 和 `*_transformed_graph.json`
- [ ] 打开 Excel，验证：
  - [ ] Metadata sheet：并行配置正确
  - [ ] Transformed Operators：所有节点有 latency_us 和 bound
  - [ ] Communication Ops：TP=1 应为空，TP>1 应有内容
  - [ ] Stream Assignment：节点均匀分配到流
