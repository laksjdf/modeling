# 规格推算路径中的算子 Shape 推导与 batch_size/seq_len 映射

> 本文追踪路径 B（规格推算）从 YAML 配置到最终性能估算的完整执行链路，重点说明算子 shape 如何生成、batch_size 和 seq_len 变化如何影响所有下游计算。

---

## 一、路径 B 总览

```
YAML 配置 (model + system + strategy)
    │
    ▼
load_specs()                     → ModelSpec, SystemSpec, Strategy
    │
    ▼
build_graph(model, strategy)     → training.ir.Graph (Op + Tensor + Collective)
    │                                 ↑ 关键：seq_len 和 hidden 在此被"固化"为 shape
    │
    ▼
estimate(model, system, strategy)
    ├─ total_training_flops(graph, model, strategy)  → FLOPs 总量
    ├─ pipeline_step_time(graph, model, system, strategy)
    │   ├─ stage_time(stage_ops, ...)    → 每个 stage 的 fwd/bwd 时间
    │   ├─ total_comm_time(graph, ...)   → 通信时间
    │   ├─ composer.compose(...)         → 流水线编排
    │   ├─ memory_breakdown(...)         → 内存分解
    │   ├─ compute_mfu(...)             → MFU
    │   └─ compute_hfu(...)             → HFU
    └─ → Report
```

**与路径 A 的本质区别**：路径 A 从真实模型 + FakeTensorMode 捕获真实的 aten op 序列及其 shape；路径 B 根据 `ModelSpec` 的几何参数（hidden, ffn, num_heads, head_dim, seq_len）**用公式构造**一个抽象的 `training.ir.Graph`，其中的 Op 和 Tensor 都是手工生成的，不经过任何 PyTorch 计算。

---

## 二、算子 Shape 的来源：`build_graph()`

### 2.1 核心函数

```python
# training/ir/builders.py:194-254
def build_graph(model: ModelSpec, strategy: Strategy) -> Graph:
    h = model.hidden      # ← 7168
    s = model.seq_len      # ← 4096
    act_dtype = model.act_dtype  # ← bf16

    # Embedding
    _embed_op(model.vocab, h, s, act_dtype)

    # 每个 transformer 层调用 dense_block()
    for i, lk in enumerate(model.layers):
        dense_block(hidden=h, ffn=model.ffn, seq=s, ...)
```

**关键参数传递**：

| 参数 | 来源 | 本例值 | 作用域 |
|------|------|--------|--------|
| `seq` (= `s`) | `ModelSpec.seq_len` | 4096 | 所有 tensor 的序列维度 |
| `hidden` (= `h`) | `ModelSpec.hidden` | 7168 | 残差流、线性层输入维度 |
| `ffn` | `ModelSpec.ffn` | 18432 | FFN 中间维度 |
| `num_heads` | `ModelSpec.num_heads` | 128 | Attention 头数 |
| `head_dim` | `ModelSpec.head_dim` | 128 | 每头维度 |
| `num_kv_heads` | `ModelSpec.num_kv_heads` | 128 | KV 头数 |
| `vocab` | `ModelSpec.vocab` | 129280 | 词表大小 |

### 2.2 Shape 生成规则：`dense_block()`

每个算子通过 `_tensor()` 辅助函数创建，shape 由 `seq`（== seq_len）和几何参数直接决定：

```python
# training/ir/builders.py:12-15
def _tensor(name: str, shape: tuple[int, ...], dtype: Dtype, ...) -> Tensor:
    return Tensor(name=name, shape_logical=shape, shape_local=shape, ...)
```

初始时 `shape_local == shape_logical`（未经 sharding），后续由 `ShardPlan` 修改 `shape_local`。

#### 完整算子序列及 Shape（以 seq_len=4096, hidden=7168 为例）

```
Layer N  (Dense Block)
────────────────────────────────────────────────────────────────────

1. LN (pre-attention)
   输入:  (4096, 7168)        ← seq=4096, hidden=7168
   输出:  (4096, 7168)

2. QKV 投影 (matmul)
   meta:  m=4096, n=16384, k=7168
          n = num_heads*head_dim + 2*num_kv_heads*head_dim
            = 128*128 + 2*128*128 = 16384 + 32768 = 49152
   实际:  h_attn = 128*128 = 16384, h_kv = 128*128 = 16384
   输入:  (4096, 7168)
   输出:  (4096, 49152)       ← seq 维度不变

3. RoPE (memory-bound)
   输入:  (4096, 16384) [Q部分], (4096, 16384) [K部分]
   输出:  (4096, 16384), (4096, 16384)

4. Attention Core (attn_core)
   meta:  b=1, s=4096, heads=128, head_dim=128, causal=True
   输入:  (4096, 16384), (4096, 16384), (4096, 16384) [Q, K, V]
   输出:  (4096, 16384)

5. O 投影 (matmul)
   meta:  m=4096, n=7168, k=16384
   输入:  (4096, 16384)
   输出:  (4096, 7168)        ← 恢复 residual shape

6. Residual Add (memory-bound)
   输入:  (4096, 7168), (4096, 7168)  [attn_out, residual]
   输出:  (4096, 7168)

7. LN (post-attention)
   输入:  (4096, 7168)
   输出:  (4096, 7168)

8. FFN up_proj (matmul)
   meta:  m=4096, n=18432, k=7168
   输入:  (4096, 7168)
   输出:  (4096, 18432)       ← ffn=18432

9. FFN gate_proj (matmul)
   meta:  m=4096, n=18432, k=7168
   输入:  (4096, 7168)
   输出:  (4096, 18432)

10. SwiGLU (memory-bound)
    输入:  (4096, 18432), (4096, 18432)  [up, gate]
    输出:  (4096, 18432)

11. FFN down_proj (matmul)
    meta:  m=4096, n=7168, k=18432
    输入:  (4096, 18432)
    输出:  (4096, 7168)

12. Residual Add (memory-bound)
    输入:  (4096, 7168), (4096, 7168)  [ffn_out, residual]
    输出:  (4096, 7168)
```

#### 嵌入和输出层

```
0. Embedding (matmul-like)
   meta:  m=4096, n=7168, k=129280
   输入:  (4096,)            ← 1D token IDs
   输出:  (4096, 7168)

N+1. Final LN (memory-bound)
   输入:  (4096, 7168)
   输出:  (4096, 7168)

N+2. LM Head (matmul-like)
   meta:  m=4096, n=129280, k=7168
   输入:  (4096, 7168)
   输出:  (4096, 129280)     ← logits
```

### 2.3 Shape 规则总结

对于任何算子，shape 遵循以下规律：

| 算子 | 输入 Shape | 输出 Shape | meta (m, n, k) |
|------|-----------|-----------|-----------------|
| Embed | `(seq,)` | `(seq, hidden)` | `m=seq, n=hidden, k=vocab` |
| QKV proj | `(seq, hidden)` | `(seq, h_attn+2h_kv)` | `m=seq, n=h_attn+2h_kv, k=hidden` |
| O proj | `(seq, h_attn)` | `(seq, hidden)` | `m=seq, n=hidden, k=h_attn` |
| Attn core | `(seq, h_attn)` ×3 | `(seq, h_attn)` | `b=1, s=seq, heads, head_dim` |
| up/gate proj | `(seq, hidden)` | `(seq, ffn)` | `m=seq, n=ffn, k=hidden` |
| down proj | `(seq, ffn)` | `(seq, hidden)` | `m=seq, n=hidden, k=ffn` |
| LN/Add/SwiGLU | `(seq, dim)` | `(seq, dim)` | `bytes_fwd ∝ seq * dim` |
| LM Head | `(seq, hidden)` | `(seq, vocab)` | `m=seq, n=vocab, k=hidden` |

**核心规律**：
- `seq_len` 决定所有 shape 的第 0 维（展开后即 matmul 的 m 维）
- `hidden` 决定残差流张量的第 1 维
- `ffn` 决定 FFN 中间层的第 1 维
- 全连接层的 (m,n,k) 中，**k 是输入特征维**，**n 是输出特征维**

---

## 三、batch_size 的映射机制

### 3.1 batch_size 不在算子 shape 中

在 `build_graph()` 中，所有算子的 shape 第一维都是 `seq_len`（而非 `batch * seq_len`）：

```python
# builders.py:36
b = 1  # batch handled at tensor level; micro_batch applied in memory/flops

# 所有 tensor shape 只有 seq 维度，没有 batch 维度
_tensor("x", (seq, h), act_dtype)   # (4096, 7168), 不是 (batch*4096, 7168)
```

**设计理由**：在路径 A（图捕获）中，PyTorch 会将 `(batch, seq, hidden)` 展开为 `(batch*seq, hidden)` 传给 matmul。但在路径 B 中，IR 是抽象的，"一个 microbatch" 的语义在算子的 `meta` 字典中单独维护，**不在 shape 中体现 batch**。

### 3.2 batch_size 如何影响 FLOPs

```python
# training/models/flops.py:104-124
def total_training_flops(graph, model, strategy):
    total = 0.0
    for op in graph.ops:
        cost = op_cost(op, model)
        total += cost.fwd_flops + cost.dx_flops + cost.dw_flops

    M = strategy.num_microbatches()  # ← batch 影响在此！
    total *= M                        # ← 乘以 microbatch 数
    return total
```

其中 `num_microbatches()` 的定义：

```python
# spec/strategy.py:103-106
def num_microbatches(self) -> int:
    if self.global_batch > 0:
        return self.global_batch // (self.micro_batch * self.dp)
    return 1
```

**映射关系**：

```
单个 op 的 FLOPs = f(seq_len, hidden, ffn, heads, ...)  ← shape 决定
每个 microbatch = sum(所有 op 的 FLOPs)
每个 step = 每个 microbatch × num_microbatches
          = 每个 microbatch × (global_batch / (micro_batch × dp))
```

### 3.3 batch_size 如何影响 Step Time

```python
# compose/pipeline.py:50-117 (OneF1BComposer)
def compose(stage_times, M, pp, dp_ar_time, strategy):
    # M = num_microbatches()
    warmup   = (pp - 1) * t_fwd_max
    steady   = M * t_stage_max         # ← batch 影响在 steady 阶段
    cooldown = (pp - 1) * t_bwd_max
    step     = warmup + steady + cooldown + dp_exposed
```

**batch 增大的影响**：
- `num_microbatches` 线性增长（假设 global_batch 增大）
- Steady 阶段时间线性增长（`M * t_stage_max`）
- Pipeline bubble 占比例减小（bubble = warmup+cooldown 是常数）
- MFU 理论上提升（因为 bubble 占比下降）

### 3.4 batch_size 如何影响内存

```python
# compose/pipeline.py:443-444 (compute_mfu 中的 tokens 计算)
tokens = strategy.global_batch * model.seq_len

# models/memory.py (memory_breakdown 中激活内存使用)
activations = 34 * hidden * seq_len * num_layers * (micro_batch / (tp * cp)) * ...
```

---

## 四、TP Sharding 后的 Shape 变化

### 4.1 ShardPlan 机制

```python
# training/ir/shard.py:107-161
def _apply_tp_sharding(graph, start, end, shard, h, h_attn, h_kv, ffn, seq, ...):
    for i in range(start, end):
        op = graph.ops[i]
        if op.kind == "matmul":
            m, n, k = op.meta["m"], op.meta["n"], op.meta["k"]

            if "qkv" in op.name:        # Column Parallel
                n_local = n // shard.tp  # 输出维 ÷ TP
                op.meta["n_local"] = n_local
                op.outputs[0].shape_local = (seq, n_local)

            elif "o_proj" in op.name:   # Row Parallel
                k_local = k // shard.tp  # 输入维 ÷ TP
                op.meta["k_local"] = k_local
                op.inputs[0].shape_local = (seq, k_local)
```

### 4.2 DeepSeek-V3 Attention 在 TP=8 下的 Shape 变化

```
                        逻辑 Shape                局部 Shape (TP=8)
                        ────────                  ──────────────

QKV proj (col-parallel):
  输入:                  (4096, 7168)             (4096, 7168)      ← 不切
  权重 W_QKV:            (7168, 49152)            (7168, 6144)      ← 49152/8
  输出:                  (4096, 49152)            (4096, 6144)      ← 49152/8
    其中 Q 部分:          (4096, 16384)            (4096, 2048)      ← 16384/8
    其中 K 部分:          (4096, 16384)            (4096, 2048)      ← 16384/8
    其中 V 部分:          (4096, 16384)            (4096, 2048)      ← 16384/8

Attn (TP shard heads):
  heads:                 128                      16               ← 128/8
  head_dim:              128                      128              ← 不变

O proj (row-parallel):
  输入:                  (4096, 16384)            (4096, 2048)     ← 16384/8
  权重 W_O:              (16384, 7168)            (2048, 7168)     ← 16384/8
  输出:                  (4096, 7168)             (4096, 7168)     ← 不变
  通信: AG before QKV + RS after O_proj

FFN up/gate_proj (col-parallel):
  输入:                  (4096, 7168)             (4096, 7168)
  权重:                  (7168, 18432)            (7168, 2304)    ← 18432/8
  输出:                  (4096, 18432)            (4096, 2304)    ← 18432/8

FFN down_proj (row-parallel):
  输入:                  (4096, 18432)            (4096, 2304)    ← 18432/8
  权重:                  (18432, 7168)            (2304, 7168)    ← 18432/8
  输出:                  (4096, 7168)             (4096, 7168)    ← 不变
  通信: AG before up + RS after down
```

### 4.3 通信量计算（Shape → bytes → time）

通信量直接从 shape 计算：

```python
# shard.py:55-60
qkv_bytes = seq * (h_attn + 2 * h_kv) * act_bytes // shard.tp
# = 4096 * (16384 + 2*16384) * 2 // 8
# = 4096 * 49152 * 2 / 8 = 50,331,648 bytes ≈ 48 MB

ag_attn_bytes = seq * h * act_bytes  # AG 通信量不减，要传输完整 seq*hidden
# = 4096 * 7168 * 2 = 58,720,256 bytes ≈ 56 MB
```

然后转换为时间（alpha-beta 模型）：

```python
# models/comm.py:14-43
def collective_time(c, group_size, tier):
    N = group_size         # TP=8 → N=8
    S = c.bytes_           # 约 56 MB
    alpha = tier.latency_us * 1e-6   # 延迟（微秒→秒）
    beta = 1.0 / bw_bytes              # 带宽倒数

    if c.kind == "AG":
        return (N-1) * (alpha + (S/N) * beta)   # Ring AG: 7 hops
    # ≈ 7 * (1e-6 + (56MB/8) / (450GB/s))
    # ≈ 7 * (1e-6 + 7MB / 56.25GB/s)
    # ≈ 7 * (1e-6 + 0.12ms) ≈ 0.84 ms
```

---

## 五、seq_len 变化的影响链路图

```
seq_len 变化
    │
    ├─→ [build_graph] 所有 tensor shape 第0维 = seq_len
    │   ├─ matmul 的 m 维度 = seq_len
    │   ├─ attn_core 的 s 维度 = seq_len → FLOPs ∝ seq²
    │   └─ memory-bound op 的 bytes ∝ seq
    │
    ├─→ [op_cost] FLOPs
    │   ├─ matmul: 2mnk = 2 * seq * n * k  → FLOPs ∝ seq
    │   ├─ attn_core: 2 * b * s² * h * d  → FLOPs ∝ seq²
    │   └─ memory-bound: bytes ∝ seq
    │
    ├─→ [stage_time] 时间
    │   ├─ compute-bound: flops / (peak * eff) → time ∝ seq
    │   ├─ memory-bound: bytes / (bw * eff)    → time ∝ seq
    │   └─ attn: time ∝ seq²
    │
    ├─→ [collective_time] 通信量
    │   └─ bytes = seq * dim * act_bytes → time ∝ seq
    │
    ├─→ [memory_breakdown] 内存
    │   └─ activations ∝ hidden * seq_len * num_layers
    │
    └─→ [compute_mfu] MFU 计算
        └─ tokens = global_batch * seq_len → model_flops ∝ seq_len
```

### 具体例子：seq_len 从 4096 → 8192

| 影响点 | seq=4096 | seq=8192 | 变化 |
|--------|----------|----------|------|
| QKV FLOPs | 2×4096×49152×7168 | 2×8192×49152×7168 | **×2** |
| Attn FLOPs | 2×1×4096²×128×128 | 2×1×8192²×128×128 | **×4** |
| FFN FLOPs | 2×4096×18432×7168 | 2×8192×18432×7168 | **×2** |
| 每层总 FLOPs | ∝4096² + 4096 | ∝8192² + 8192 | **≈×4 (attn 主导)** |
| 激活内存 | ∝4096 | ∝8192 | **×2** |
| TP AG 通信 | 56 MB | 112 MB | **×2** |
| Step time (无 attn 瓶颈) | ∝4096 | ∝8192 | **×2** |
| Step time (attn 瓶颈) | ∝4096² | ∝8192² | **×4** |

---

## 六、两条路径的 Shape 来源对比

| 维度 | 路径 A（图捕获） | 路径 B（规格推算） |
|------|-----------------|-------------------|
| Shape 来源 | FakeTensorMode 下真实 PyTorch 前向传播 | `build_graph()` 手工构造 |
| 算子序列 | 真实的 aten op 序列（含 MoE router、MLA 等） | 简化的 13-op 模板（dense_block） |
| MoE 层 | 真实的 MoE dispatch/combine 算子 | 暂用 dense_block 代替（代码注释 `# Phase 2: moe_block()`） |
| MTP 层 | 真实的 MTP 算子 | 暂用 dense_block 代替（代码注释 `# Phase 2: mtp_block()`） |
| Seq 维度 | 真实 shape（如 `(batch*seq, hidden)` 展开后） | IR 抽象 `(seq, hidden)` |
| Batch | 通过展开和多次 microbatch 迭代体现 | 通过 `num_microbatches()` 乘法体现 |
| 精度 | 精确到每个 aten op | 近似 13 个算子/层 |

---

## 七、当前限制与 TODO

1. **MoE 层尚未实现**：`build_graph()` 中 `LayerKind.MOE` 和 `LayerKind.MTP` 都 fallback 到 `dense_block()`，注释标记为 `# Phase 2`。这意味着当前路径 B 对 DeepSeek-V3 的 MoE 建模是不准确的（将 MoE 层当作 dense 层处理）。

2. **attn_compression_ratio 未使用**：路径 B 的 `dense_block` 中 attention FLOPs 未应用 MLA 压缩比，可能高估 attention 开销。

3. **EP（Expert Parallelism）不平衡因子**：`ep_imbalance_factor()` 使用 balls-into-bins 近似，实际负载可能更均匀。

4. **1F1B 假设同构 stage**：当前 `OneF1BComposer` 假设所有 stage 的 fwd/bwd 时间相同（取 max），不能精确处理 layer 数不均等导致的异构流水线。
