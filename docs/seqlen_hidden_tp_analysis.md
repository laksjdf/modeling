# seq_len=4096 / hidden=7168 算子图追踪与 TP 形变分析

> 基于 DeepSeek-V3 模型，追踪两个关键参数在代码中的传递链路及 TP 开启后的 shape 变化原理。

---

## 一、两条执行路径概述

ZRT-Sim 支持两条训练建模路径，`seq_len` 和 `hidden` 在两条路径中的使用方式不同：

```
路径 A（图捕获）:  CLI → run_trace_phases → _trace_phase → FakeTensorMode 前向 → 记录 aten op
                   ↓
              estimate_training_from_graphs → metadata 注入 → Transform Pipeline → Report

路径 B（规格推算）: CLI --estimate-config → load_specs → estimate → build_graph → analytical formula → Report
```

本文重点分析路径 A（算子图捕获），并兼述路径 B 的差异。

---

## 二、seq_len 和 hidden 的源头与传递链

### 2.1 CLI 入口

`seq_len` 和 `hidden` 在 CLI 中通过两个不同来源进入：

```python
# cli.py:144-151
parser.add_argument("--hidden", type=int, default=7168, ...)      # hidden → 默认 7168
parser.add_argument("--seq-len", type=int, default=128, ...)       # seq_len → 默认 128
```

- `seq_len` → 直接构造虚拟输入张量，决定每个算子的序列维度
- `hidden` → 仅在训练建模阶段注入 `graph.metadata`，用于分析 pass 的内存/通信估算

### 2.2 seq_len 在图捕获阶段的作用

**核心函数**: `_trace_phase()` (`graph/main.py:289-403`)

```python
# graph/main.py:322-325
if phase == "decode":
    query_len = 1
    pos_start = seq_len
else:
    query_len = seq_len    # ← prefill 阶段：query_len == seq_len
    pos_start = 0

# graph/main.py:325-330
input_ids = torch.randint(0, config.vocab_size, (batch_size, query_len))
# shape: (1, 4096)  ← seq_len 直接决定了 input_ids 的第二维

# graph/main.py:344-345
mask = torch.full((1, 1, seq_len, seq_len), float("-inf"))
mask = torch.triu(mask, diagonal=1)
# shape: (1, 1, 4096, 4096)  ← attention mask 由 seq_len 决定
```

**seq_len 通过两个张量进入模型**：

| 张量 | 形状 | 作用 |
|------|------|------|
| `input_ids` | `(batch, seq_len)` = `(1, 4096)` | 输入 token IDs |
| `attention_mask` | `(1, 1, seq_len, seq_len)` = `(1, 1, 4096, 4096)` | 因果注意力掩码 |

### 2.3 seq_len 如何在 FakeTensorMode 下传播到每一个算子

由于模型处于 `FakeTensorMode`，所有中间张量都是 FakeTensor（只有 shape/dtype/stride，无真实数据）。`TorchDispatchMode` 拦截每一个 aten op：

```python
# graph/dispatch.py:119-194
class RecordingDispatch(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **kwargs)  # ← 实际执行（FakeTensorMode 下无真实计算）

        input_tensors = collect_tensors(args, kwargs)
        output_tensors = collect_output_tensors(out)

        # 记录每个 tensor 的 shape 和 dtype
        input_shapes = [shape_str(t) for t in input_tensors]
        output_shapes = [shape_str(t) for t in output_tensors]

        self.records.append({
            "aten_op": func_name,
            "input_shapes": ", ".join(input_shapes),
            "output_shapes": ", ".join(output_shapes),
            ...
        })
```

**seq_len 对算子序列的影响**：

1. **Embedding 层**: `input_ids (1, 4096)` → `hidden_states (1, 4096, 7168)`
    - seq_len=4096 决定了 hidden_states 的中间维度

2. **Attention 模块**（以 QKV 投影为例）:
   - 输入: `(1, 4096, 7168)` — seq_len 维度为 4096
   - Q 投影 `aten.mm`: `(4096, 7168) × (7168, 16384)` → `(4096, 16384)`
   - 注意：在 `model.eval()` 模式下，线性层会将 batch 和 seq 合并：`(1*4096, 7168)`

3. **Attention Score 计算**:
   - Score = `Q @ K^T`: shape `(batch, heads, 4096, 4096)` — 由 seq_len 决定
   - Softmax 沿最后一维: shape 不变
   - Score @ V: `(batch, heads, 4096, 64)` → `(batch, heads, 4096, head_dim)`
   - Reshape 回 `(batch, 4096, hidden)` = `(1, 4096, 7168)`

4. **FFN 层**:
   - gate_proj (up): `(4096, 7168) × (7168, 18432)` → `(4096, 18432)`
   - up_proj: `(4096, 7168) × (7168, 18432)` → `(4096, 18432)`
   - SiLU(gate) * up: `(4096, 18432)` element-wise
   - down_proj: `(4096, 18432) × (18432, 7168)` → `(4096, 7168)`

5. **MoE 层**（V3 的 58 个 MoE 层）:
   - Router: `(4096, 7168) @ (7168, 256)` → `(4096, 256)` — 路由 logits
   - Top-k selection → 8 个专家被激活
   - 每个专家的 FFN: 同 dense FFN 结构，但 `moe_ffn=1536`

6. **MTP 层**: 额外的投影 + dense block

**关键结论**: `seq_len` 作为序列维度贯穿整个模型，决定了：
- 所有 matmul 的 m 维度（batch*seq 展开后）
- attention score 矩阵的 O(seq_len²) 空间
- KV cache 的序列长度维度
- decode 阶段 seq_len=1 时算子形状完全不同于 prefill（seq_len=4096）

### 2.4 hidden 在图捕获阶段的作用

在**图捕获阶段**，`hidden` 不直接作为参数传入 — 它来自模型 config 的 `hidden_size`（本例=7168）。它决定：

1. 所有线性层的权重形状: `(hidden, projection_dim)` 或 `(projection_dim, hidden)`
2. 残差流的张量形状: `(batch, seq_len, hidden)` = `(1, 4096, 7168)`
3. FFN 中间层形状: `(batch, seq_len, ffn)` = `(1, 4096, 18432)`
4. 专家路由维度: `hidden → num_experts`

### 2.5 hidden 和 seq_len 在训练分析阶段的作用

在 `estimate_training_from_graphs()` (`transform/analysis/modeller.py:177-308`) 中，`hidden` 和 `seq_len` 被注入 `graph.metadata`：

```python
# modeller.py:205-211
metadata = {
    "seq_len": seq_len,        # → TrainingFlopsPass, TrainingMemoryPass 使用
    "batch_size": batch_size,
    "num_layers": num_layers_full or num_layers,
    "num_layers_traced": num_layers,
    "hidden": hidden,           # → TrainingMemoryPass 使用
}
```

**使用位置**：

1. **TrainingFlopsPass** (`training.py:37-92`)：
   - 6P 规则回退时：`tokens = seq_len * batch_size`, `forward_flops = 2 * params * tokens`
   - 即 `seq_len` 直接影响 FLOPs 总量

2. **TrainingMemoryPass** (`training.py:143-206`)：
   - Korthikanti 激活内存公式：`base = 34 * hidden * seq_len * num_layers * batch_size`
   - 通信缓冲区：`comm_bytes = (2 * hidden * seq_len * num_layers) / tp`
   - **`hidden` 和 `seq_len` 都直接参与内存估算**

3. **CommInserterPass** (`comm_inserter.py:182-197`)：
   - CP 通信消息大小：`ulysses_msg_bytes = micro_batch * (seq_len // cp) * hidden * dtype_bytes`

---

## 三、TP（Tensor Parallelism）开启后的 Shape 变化

### 3.1 TP 的核心原理

Tensor Parallelism 将权重矩阵沿特定维度切分到多个 GPU，每个 GPU 只持有部分权重。对 Transformer 层，切分策略取决于矩阵类型：

| 矩阵类型 | 切分维度 | 通信需求 |
|----------|----------|----------|
| **Column Parallel**（Q/K/V, gate, up） | 输出维度 (dim=-1) | 无需通信（切分输出） |
| **Row Parallel**（O, down） | 输入维度 (dim=-1) | 需要 AllReduce（还原完整结果） |

### 3.2 TP 实现的代码路径

```python
# transform/parallel/tensor_parallel.py:47-95
class TensorParallelPass(GraphPass):
    def run(self, graph, ctx):
        tp = ctx.parallel.tp  # e.g., tp=8
        if tp <= 1:
            return graph

        g = graph.clone()
        for node in g.topo_sort():
            if node.op_type not in _MATMUL_OPS:  # 只处理 matmul/linear
                continue
            rule = _classify(node.scope)  # 按 scope 分类
            if rule is None:
                continue

            # 标记 annotation
            node.annotations["tp_split"] = {...}

            if rule.input_split:
                # Row Parallel: 输入最后一维 ÷ tp
                node.inputs[0].shape[-1] /= tp
            else:
                # Column Parallel: 输出最后一维 ÷ tp
                node.outputs[i].shape[-1] /= tp
                # 同时更新 edge 上的 tensor
                for e in g.out_edges(node.id):
                    e.tensor.shape[-1] /= tp
```

### 3.3 具体算子的 Shape 变化（以 TP=8, hidden=7168 为例）

#### Attention 模块

```
                    QKV 投影 (Column Parallel)
                    ─────────────────────────
原始: (4096, 7168) @ W_QKV (7168, 3×16384)
TP后: (4096, 7168) @ W_QKV_shard (7168, 3×2048) → (4096, 3×2048)
                                                    ↑ 16384 / 8 = 2048
Q/K/V 各取 2048/8=256 维, heads 按 128/8=16

                    O 投影 (Row Parallel)
                    ─────────────────────
原始: (4096, 16384) @ W_O (16384, 7168) → (4096, 7168)
TP后: (4096, 2048)  @ W_O_shard (2048, 7168) → (4096, 7168)
       ↑ 16384/8                          ↑ 实际各rank产出的是部分和
       AllReduce 将各rank结果加和
```

#### FFN 模块（Dense）

```
                    gate_proj / up_proj (Column Parallel)
                    ──────────────────────────────────────
原始: (4096, 7168) @ W_gate (7168, 18432) → (4096, 18432)
TP后: (4096, 7168) @ W_gate_shard (7168, 2304) → (4096, 2304)
                                                   ↑ 18432 / 8 = 2304

                    down_proj (Row Parallel)
                    ─────────────────────────
原始: (4096, 18432) @ W_down (18432, 7168) → (4096, 7168)
TP后: (4096, 2304)  @ W_down_shard (2304, 7168) → (4096, 7168)
       ↑ 18432/8                                 AllReduce
```

### 3.4 TP 分类规则

```python
# transform/parallel/tensor_parallel.py:22-28
_COL_PARALLEL = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "w1", "w3")
_ROW_PARALLEL = ("o_proj", "down_proj", "w2")

def _classify(scope: str) -> TPRule | None:
    s = scope.lower()
    if any(k in s for k in _COL_PARALLEL):
        return TPRule(split_dim=-1, comm_after=None, input_split=False)
    if any(k in s for k in _ROW_PARALLEL):
        return TPRule(split_dim=-1, comm_after="all_reduce", input_split=True)
    return None
```

**匹配方式**：通过 `OpNode.scope`（即 `nn.Module` 路径，如 `model.layers.0.self_attn.q_proj`）中的关键字判断。

### 3.5 通信插入（CommInserterPass）

TP 分类完成后，`CommInserterPass` 在 Row Parallel 节点后插入 `comm.all_reduce`：

```python
# comm_inserter.py:92-108
def _insert_tp_comm(self, g, ctx):
    tp_nodes = [
        n for n in g.topo_sort()
        if n.annotations.get("tp_split", {}).get("comm_after") == "all_reduce"
    ]
    for node in tp_nodes:
        comm_node = _make_comm_node(comm_id, "all_reduce", node, tp)
        _rewire(g, node.id, comm_node)  # 在 node 和它的所有后继之间插入
```

**AllReduce 通信量**：每个 Row Parallel 节点的输出 shape 决定了通信量。以 O 投影为例：

```
输出 shape: (4096, 7168), dtype=bf16 → 4096 * 7168 * 2 bytes ≈ 55.9 MB
这是 TP=8 下每个 micro-batch 的 AR 通信量
```

### 3.6 完整示例：一个 Attention 块在 TP=8 下的 shape 流

```
输入: hidden_states (1, 4096, 7168)
  ↓ reshape → (4096, 7168)

  ┌─ QKV (Column Parallel) ─────────────────────────────────────┐
  │ W_QKV: (7168, 3×16384) → W_QKV_shard: (7168, 3×2048)       │
  │ out: (4096, 3×16384)   → out_shard: (4096, 3×2048)         │
  │ Q = out[:, :, :16384/8] → Q_shard: (4096, 16heads, 256dim)  │
  │ K, V 同理                                                     │
  └──────────────────────────────────────────────────────────────┘

  ┌─ Attention Core ─────────────────────────────────────────────┐
  │ Q@K^T: (16, 4096, 256) @ (16, 256, 4096) → (16, 4096, 4096) │
  │ Score@V: (16, 4096, 4096) @ (16, 4096, 256) → (16, 4096, 256)│
  │ concat → (4096, 16×256=4096) ... 不对，是 16heads×256=4096   │
  │ 实际是: (4096, 16×128=2048) [head_dim/TP = 128/8...]         │
  └──────────────────────────────────────────────────────────────┘

  ┌─ O (Row Parallel) ───────────────────────────────────────────┐
  │ W_O: (16384, 7168) → W_O_shard: (2048, 7168)                 │
  │ out_shard: (4096, 7168)                                       │
  │ comm.all_reduce(output) ← 各 rank 的部分和在此规约            │
  └──────────────────────────────────────────────────────────────┘
  ↓ reshape → (1, 4096, 7168)
```

---

## 四、seq_len 变化对算子序列的影响

### Prefill (seq_len=4096) vs Decode (seq_len=1) 的 shape 对比

| 算子 | Prefill (seq_len=4096) | Decode (seq_len=1) |
|------|------------------------|---------------------|
| input_ids | `(1, 4096)` | `(1, 1)` |
| attention_mask | `(1, 1, 4096, 4096)` | `(1, 1, 1, seq_len+1)` |
| embed → hidden | `(1, 4096, 7168)` | `(1, 1, 7168)` |
| QKV matmul m 维 | 4096 | 1 |
| Attention Score | `(heads, 4096, 4096)` | `(heads, 1, total_len)` |
| FFN matmul m 维 | 4096 | 1 |
| 每层 FLOPs | O(seq_len²) attention + O(seq_len) FFN | O(seq_len) attention + FFN |

**对算子序列的影响**：
- Decode 阶段所有 matmul 的 m=1，kernel 启动开销主导，计算效率极低
- Decode 阶段 KV cache 存在时，attention score 的 k 维度会随历史长度增长
- Prefill 阶段 attention 的 O(seq_len²) 复杂度在长序列下成为主要瓶颈

---

## 五、附录：关键文件索引

| 文件 | 职责 |
|------|------|
| `graph/main.py:289-403` | `_trace_phase()` — 构造虚拟输入，驱动模型前向 |
| `graph/model_loader.py:178-235` | `load_model()` — FakeTensorMode 加载模型 |
| `graph/dispatch.py:101-194` | `RecordingDispatch.__torch_dispatch__()` — 拦截 aten op |
| `ir/adapter.py:96-208` | `records_to_opgraph()` — record dict → OpGraph IR |
| `transform/parallel/tensor_parallel.py:47-95` | `TensorParallelPass.run()` — shape 切分 |
| `transform/parallel/comm_inserter.py:92-108` | `_insert_tp_comm()` — 插入 AllReduce |
| `transform/context.py:98-108` | `TransformContext` — 携带 parallel config |
| `transform/analysis/training.py:37-92` | `TrainingFlopsPass` — 使用 seq_len 计算 FLOPs |
| `transform/analysis/training.py:143-206` | `TrainingMemoryPass` — 使用 hidden+seq_len 估算内存 |
| `transform/analysis/modeller.py:177-308` | `estimate_training_from_graphs()` — 组装入口 |
| `cli.py:297-365` | `_run_training_modelling()` — CLI 训练建模分派 |
