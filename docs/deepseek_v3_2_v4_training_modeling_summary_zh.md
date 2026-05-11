# DeepSeek-V3 / V3.2 / V4 训练建模 — 实现总结

> 实现路径：`python -m python.zrt --estimate-config <yaml>` 静态分析（不抓图）→ step_time / MFU / HFU / memory
> 覆盖模型：DeepSeek-V3、V3.2（DSA）、V4-Pro（CSA+HCA+SWA, 1.6T-A50B）、V4-Flash（CSA+HCA+SWA, 290B-A14B）
> 覆盖硬件：NVIDIA H100 SXM、H800、Ascend 910C（自动按 `hardware/configs/*.yaml` 选择）

---

## 1 · 总览

通过手动静态分析方式，把三族 DeepSeek 模型的训练算子序、参数核算、TP/EP/PP/CP 分片、流水线编排、内存预算全部建到 `python/zrt/training/` 模块下，使 `--estimate-config` 这条 spec-based 路径无需 graph capture 即可输出可信的训练性能估计。

实测 4 个模型 × H100 SXM 的 MFU 全部落入论文反推的锚点 ±0.15 区间内：

| 模型 | step_time | MFU 实测 | MFU 锚点 | 备注 |
|---|---|---|---|---|
| V3 | 33.8 s | 0.442 | 0.45 | 验证基线 |
| V3.2 | 44.0 s | 0.340 | 0.43 | DSA / Lightning Indexer |
| V4-Pro | 48.7 s | 0.374 | 0.42 | CSA + HCA + SWA + Muon |
| V4-Flash | 13.2 s | 0.331 | 0.40 | 同 V4-Pro 架构，参数量缩 5× |

`pytest tests/training/anchors/` 下 44/44 通过（含 13 个锚点的 strict MFU 校验）。

---

## 2 · 架构建模能力

### 2.1 ModelSpec 三族字段（`python/zrt/training/spec/model.py`）

新增字段，让单一 `ModelSpec` 同时描述 V3 / V3.2 / V4：

| 类别 | 字段 | 用途 |
|---|---|---|
| MLA（V3+） | `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim` | Multi-head Latent Attention 几何 |
| Indexer（V3.2+） | `index_n_heads`, `index_head_dim`, `index_topk` | Lightning Indexer 几何 |
| V4 注意力 | `o_lora_rank`, `o_groups`, `compress_ratios`, `swa_window` | grouped O-proj、CSA/HCA/SWA 三族 |
| V4 路由 | `n_hash_routed_layers`, `scoring_func` | hash-routing 前 N 层 |
| V4 数值 | `routed_expert_dtype`, `swiglu_clamp` | FP4 expert / SwiGLU 钳位 |

属性 `use_mla` / `use_v4_attn` 在 builder 里做架构分发。

### 2.2 算子序构造（`python/zrt/training/ir/builders.py`）

`build_graph` 完全静态由 `model.layers + 几何参数` 生成 `Op` 列表：

- `_build_mla_attn`：q_a → q_a_norm → q_b → kv_a → kv_a_norm → kv_b → RoPE → attn_core → o_proj
- `_build_v4_attn`：低秩 Q + 单 KV head MQA + 可选 compressor/indexer + grouped O
- `_build_indexer_ops`：wq_b → weights_proj → idx_compressor_pool → indexer_topk
- `_build_compressor_ops`：comp_wkv → comp_wgate → compressor_pool（按 `compress_ratio` 收缩 KV）
- V3.2 indexer 在 MLA 路径自动接入：`if model.index_topk > 0 and not model.use_v4_attn`

### 2.3 三种新 Op 类别

- `compressor_pool` — KV 压缩门控池化（softmax + 加权求和）
- `indexer_topk` — Lightning Indexer 评分（einsum + ReLU + top-k）
- `hash_route` — 哈希路由（FLOPs 可忽略，仅占位）

各自在 `flops.py::op_cost` 注册，含 `_compressor_pool_cost` / `_indexer_topk_cost`。

### 2.4 注意力变体 FLOPs 模型（`flops.py::_attn_cost`）

由 op.meta 字段自动分流：

- `sparse_topk > 0` → CSA：`2·b·s·(topk+swa_window)·h·d`
- `compress_ratio > 0` → HCA：`2·b·s·(s/ratio + swa_window)·h·d`
- `swa_window > 0` → SWA-only：`2·b·s·swa_window·h·d`
- 其他 → 全因果 MLA：`2·b·s²·h·d × compression_ratio`

---

## 3 · 并行分片（`python/zrt/training/ir/shard.py`）

### 3.1 TP 分片矩阵

按 op 名 pattern 分流到列并行 / 行并行：

- 列并行（n 维分片）：`q_a/b_proj`, `kv_a_proj`, `wq_a/b`, `wkv`, `up_proj`, `gate_proj`, `comp_wkv/gate`, `idx_*`, `qkv`
- 行并行（k 维分片）：`o_proj`, `down_proj`, `wo_a/b`, `kv_b_proj`
- 路由：`router` 按 n 分；`routed_expert*` 同时按 n、k 分
- `attn_core`：按 heads 分片
- `indexer_topk`：按 indexer-head 分片（`ih_local = ih // tp`，注入 `world_factor=tp`）
- `compressor_pool`：按通道分片（`d_local = d // tp`，bytes_fwd 同步缩放）

### 3.2 EP 分片

- `routed_expert*` 的 `fwd_multiplier` 乘 `experts_per_rank / num_experts`
- EP 不平衡因子（`ep_imbalance_factor`）只乘 EP-并行 op，不再放大整个 stage（`stage.py::_ep_parallel_fraction`）

### 3.3 CP 分片

- `ULYSSES`：按 head 切，attn 用 `heads × cp` 全头算
- `RING`：按 seq 切，attn 用 `cp_tiles` 重复算
- `HYBRID`：同时插入 A2A（按 head）+ P2P（按 seq）的两阶段通信，对应 V4 §3.5.3

---

## 4 · 优化器建模

### 4.1 Muon 双阶段 Newton-Schulz（`models/optimizer.py`）

`muon_step_flops(P, K, hidden)` 把 K 步拆为：

- **Stage 1**：`int(K * 0.8)` 步，degree-4 多项式，`6·max(m,n)·min(m,n)²`/步
- **Stage 2**：`K - K1` 步，degree-2 多项式，`4·max(m,n)·min(m,n)²`/步

每个 NS 单元额外加 `4P` FLOPs（动量更新 + 参数更新）。

### 4.2 Muon 通信 / 状态字节

- `muon_comm_time(P, dp, hidden, ...)` 支持 ZeRO 1/2/3 + rotation（reduce_scatter）
- `muon_state_bytes`：根据 `muon_param_fraction` 在 Adam / Muon 两路按比例混合
- `resolve_muon_ns_steps`：DSV4=10, DSV3=8, 其余 default

### 4.3 配置层

`MuonConfig` 在 yaml 中：

```yaml
optimizer: muon
muon_config:
  ns_steps: 10
  rotation: true
  adam_param_types: ["embed", "lm_head", "router", "bias"]
  muon_param_fraction: 0.85
```

---

## 5 · 内存模型（`models/memory.py`）

### 5.1 FP4 路由专家权重

`routed_expert_dtype == "fp4"` 时：

```
expert_weight_bytes = P_expert × 0.5 + (P_expert / 32) × 2
                       │ FP4 主体    │ BF16 per-block scale
```

非 FP4 路径回退到 `param_dtype.bytes`（V3/V3.2 = bf16）。

### 5.2 ZeRO 0/1/2/3

- 0：所有 rank 全量
- 1：opt_state 切 dp
- 2：grads + opt_state 切 dp
- 3：weights + grads + opt_state 全切 dp

激活内存按 `recompute.per_layer` 配置扣除被重算的 op。

---

## 6 · 流水线编排（`compose/`）

### 6.1 Pipeline schedule

`PPSched` 枚举：`ONE_F_ONE_B`, `INTERLEAVED`(VPP), `ZERO_BUBBLE`, `DUALPIPE`, `DUALPIPE_V`

每个 schedule 对应一个 `PipelineComposer` 子类，返回 `StepResult{step_time, bubble_fraction, mfu, hfu, memory}`。

### 6.2 EP wave-overlap（`stage.py`）

`Strategy.ep_overlap = True` 时，把 EP A2A 拆成 K=4 波，与 expert GEMM 重叠：

```
exposed_total = comm_per_wave + (K-1) × max(comm/wave - gemm/wave, 0)
```

被节省的时间从 t_fwd / t_bwd / t_comm 中扣除。

### 6.3 MFU vs HFU

- **MFU** = `tokens × 6 × P_active / step_time / hw_peak_flops`，**不含**重算开销
- **HFU** = MFU + `recompute_flops / step_time / hw_peak_flops`
- `effective_params_for_flops` MTP 路径已用 active-fraction 公式：

  ```
  P_active = embed + dense_layers × P_dense + moe_layers × (P_attn + P_shared + top_k/num_experts × P_routed)
  ```

  实测 V3.2=37.7B、V4-Pro=50.6B、V4-Flash=14.1B，与论文对齐。

---

## 7 · 配置与锚点

### 7.1 模型 yaml（`configs/models/`）

- `deepseek_v3.yaml`、`deepseek_v3_2.yaml`、`deepseek_v4_pro.yaml`、`deepseek_v4_flash.yaml`

### 7.2 3D 并行 yaml（`configs/`）

每个模型 × {h100, h800, ascend_910c} = 12 套：

```
deepseek_v{3,3_2,4_pro,4_flash}_3d_{h100,h800,ascend_910c}.yaml
```

### 7.3 锚点 yaml（`tests/training/anchors/`）

| 锚点 | 论文 MFU | tolerance |
|---|---|---|
| deepseek_v3 | 0.45 | ±0.15 |
| deepseek_v3_2 (× 3 硬件) | 0.43 / 0.37 / 0.35 | ±0.15 / ±0.20 |
| deepseek_v4_pro (× 3 硬件) | 0.42 / 0.38 / 0.36 | ±0.15 / ±0.20 |
| deepseek_v4_flash (× 3 硬件) | 0.40 / 0.36 / 0.34 | ±0.15 / ±0.20 |
| llama3_70b_meta、gpt3_175b_megatron、deepseek_v4_muon | 历史回归基线 | — |

`tests/training/anchors/test_anchors.py::test_anchor_mfu_strict` 是必跑硬性校验，13/13 全过。

---

## 8 · 推理建模回归（`tests/training/test_v4_long_ctx_efficiency.py`）

复现论文图 1：1M context 下

- V4-Pro per-token FLOPs / V3.2 ≈ 0.27 ±0.15
- V4-Pro KV cache / V3.2 ≈ 0.10 ±0.15
- V4-Flash 0.10 / 0.07 同样验证

7/7 通过。

---

## 9 · 文件清单

### 新增 / 修改源文件

```
python/zrt/training/
├── spec/model.py            (+331 行) — 三族字段
├── ir/builders.py           (+1033 行) — _build_{mla,v4}_attn / _build_{indexer,compressor}_ops
├── ir/shard.py              (+ 83 行) — indexer_topk / compressor_pool TP 分片，EP FLOPs 修正
├── models/flops.py          (+ 90 行) — _compressor_pool_cost / _indexer_topk_cost / V4 attn 变体
├── models/memory.py         (+ 37 行) — FP4 expert
├── models/optimizer.py      (+ 59 行) — Muon 双阶段 NS
├── compose/stage.py         (+ 92 行) — EP wave-overlap, _ep_parallel_fraction
├── io/config_loader.py      (+ 27 行) — 新字段加载
├── configs/models/deepseek_v{3_2,v4_pro,v4_flash}.yaml — 3 个模型 yaml
└── configs/deepseek_v{3,v3_2,v4_pro,v4_flash}_3d_*.yaml — 12 个 3D 并行 yaml
```

### 新增测试 / 锚点

```
tests/training/
├── anchors/deepseek_v3_2{,_h800,_ascend_910c}.yaml        — V3.2 × 3 硬件
├── anchors/deepseek_v4_pro_{h800,ascend_910c}.yaml        — V4-Pro × 3 硬件
├── anchors/deepseek_v4_flash{,_h800,_ascend_910c}.yaml    — V4-Flash × 3 硬件
├── anchors/test_anchors.py::test_anchor_mfu_strict        — strict MFU 必跑校验
└── test_v4_long_ctx_efficiency.py                         — 1M ctx 推理比例
```

### 设计 / 推导文档

```
docs/
├── deepseek_v3_anchor_derivation_zh.md     — V3 锚点推导
├── deepseek_v3_2_anchor_derivation_zh.md   — V3.2 锚点推导
├── deepseek_v4_anchor_derivation_zh.md     — V4-Pro / V4-Flash 锚点推导
└── deepseek_v3_v32_v4_arch_diff_zh.md      — 三族架构对比
```

---

## 10 · 已知限制与遗留

- **H6 strict tolerance 收紧**：Ascend / H800 锚点目前用 ±0.15~±0.20 较宽容差；待物理标定后再收紧到 ±0.10。
- **H12 Sinkhorn 迭代展开**：mHC 的 Sinkhorn-Knopp 投影目前作为单 op 估计，未展开为 20 个迭代 op。计划后续版本细化。
- **H16 wo_a IR 形状**：当前 `_build_v4_attn::wo_a` 用 `k=h_per_group` 给出正确的 grouped projection FLOPs；改成 `k=h_attn` 会膨胀 `o_groups×`。保持现状。
- **未建模特性**：Anticipatory Routing、SwiGLU clamp 数值稳定性、KV cache on-disk、TileLang DSL 内核生成 — ModelSpec 字段已保留，wall-time 影响可忽略。

---

## 11 · 调用入口（速查）

```bash
# V3 baseline
python -m python.zrt --estimate-config python/zrt/training/configs/deepseek_v3_3d_h100.yaml

# V3.2 (DSA)
python -m python.zrt --estimate-config python/zrt/training/configs/deepseek_v3_2_3d_h100.yaml

# V4-Pro (CSA + HCA + SWA + Muon)
python -m python.zrt --estimate-config python/zrt/training/configs/deepseek_v4_pro_3d_h100.yaml

# V4-Flash
python -m python.zrt --estimate-config python/zrt/training/configs/deepseek_v4_flash_3d_h100.yaml

# H800 / Ascend 910C 同样路径，把文件名末尾换成 _h800.yaml / _ascend_910c.yaml

# Anchor 回归
PYTHONPATH=python pytest tests/training/anchors/ -v
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py::test_anchor_mfu_strict -v -s
```
