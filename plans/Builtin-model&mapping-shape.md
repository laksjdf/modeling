# 计划：Path A 捕获图作为 Path B 内置模型库

## 一、目标与背景

### 背景
ZRT-Sim 当前有两条训练建模路径（详见 `docs/seqlen_hidden_tp_analysis.md`、`docs/spec_path_shape_analysis.md`）：

- **Path A（图捕获）**：`FakeTensorMode + TorchDispatchMode` 拦截真实 HuggingFace 模型前向，得到 aten 粒度的 `OpGraph`。**优点**：真实捕获 MoE dispatch/combine、MLA、MTP 等结构。**缺点**：每次都需加载模型并 trace。
- **Path B（规格推算）**：`training/ir/builders.py:build_graph()` 按 `ModelSpec` 几何参数手工构造 12-op/层的抽象 `training.ir.Graph`。**缺点**：MoE/MTP 都 fallback 到 `dense_block`（代码注释 `# Phase 2`），不准确；attn_compression_ratio 未应用。

### 用户提案
1. 把 Path A 产出的算子序列**持久化为内置模型库**（区分 prefill/decode），Path B 直接加载。
2. 捕获时记录**形状-输入映射**：与模型输入无关的维度（hidden、ffn、head_dim）固化；与输入相关的维度（seq_len、batch_size）参数化到 op meta，Path B 加载时按新输入重算 shape。

### 可行性结论
**可行**，但存在两个 IR 的**粒度落差**（Path A 数百 aten op/层 vs Path B 12 语义 op/层），需要新增一个**聚合+映射 Pass**，将捕获的 `OpGraph` 转换为 `training.ir.Graph`，而不是直接替换 IR。这样既能复用 Path B 下游所有代码（`flops.py`、`shard.py`、composer），又能让 Path B 受益于 Path A 的真实 MoE/MTP 结构。

### 预期收益
- Path B 对 DeepSeek-V3 等 MoE 模型的建模精度提升（不再用 dense_block 替代）
- 内置库一次捕获多次复用，省去重复加载模型成本
- prefill/decode 算子序列分别保留，支持推理建模差异化

---

## 二、关键设计

### 2.1 IR 桥接策略（核心）
新增 `python/zrt/training/ir/from_opgraph.py`：把 `OpGraph` 聚合为 `training.ir.Graph`。

**聚合规则**（基于 `OpNode.scope`/`module_class`/`component`）：
| OpNode.scope/component 关键字 | 映射到 training.ir.Op.kind | 目标 op.name |
|--------------------------------|----------------------------|--------------|
| `q_proj`/`k_proj`/`v_proj`/`qkv_proj` 内的 mm/addmm | `matmul` | `layer{i}.qkv` |
| `o_proj` 内 mm/addmm | `matmul` | `layer{i}.o_proj` |
| `rotary_emb`/RoPE 区域 | `rope` | `layer{i}.rope` |
| flash_attn / `aten._scaled_dot_product_*` / 手工 attn 链 | `attn_core` | `layer{i}.attn_core` |
| `gate_proj`/`up_proj` 内 mm | `matmul` | `layer{i}.up_proj`、`layer{i}.gate_proj` |
| `down_proj` 内 mm | `matmul` | `layer{i}.down_proj` |
| `silu`+`mul` 链 | `swiglu` | `layer{i}.swiglu` |
| `RMSNorm`/`LayerNorm` 模块 | `ln` | `layer{i}.ln_*` |
| `embed_tokens` | `embed` | `embed` |
| `lm_head` | `lm_head` | `lm_head` |
| MoE router (`gate` 模块) | `router` | `layer{i}.router` |
| MoE dispatch (a2a 前 reorder) | `dispatch` | `layer{i}.dispatch` |
| MoE combine | `combine` | `layer{i}.combine` |
| MoE 内专家 mm | `matmul` | `layer{i}.expert_{up/gate/down}` |

每个语义 op 的 `meta["m"]/["n"]/["k"]` 从聚合范围内 matmul 节点的 `inputs[0].shape` 反推；`bytes_fwd` 由聚合范围所有节点的输入字节加和。

**op.name 命名约定**必须保持与 `shard.py:_apply_tp_sharding()` 子串匹配规则兼容（`"qkv" in op.name`、`"o_proj" in op.name` 等）。

### 2.2 形状模板化策略
在捕获阶段（`graph/dispatch.py`）记录每个 tensor 维度的"来源标签"：

- 捕获前已知 `seq_len`、`batch_size`、`vocab`、`hidden`、`ffn`、`head_dim`、`num_heads` 等参数（来自 HF config + CLI）
- 在 `RecordingDispatch.__torch_dispatch__` 中，对每个 tensor 的 shape 逐维比对：
  - 若 `dim == seq_len` → 标记 `"S"`
  - 若 `dim == batch_size` → 标记 `"B"`
  - 若 `dim == seq_len * batch_size` → 标记 `"BS"`
  - 否则视为静态，标记具体值（如 `7168`）
- 标签存入 record 的新字段 `input_shape_tags`/`output_shape_tags`
- 序列化进 `TensorMeta` 时，新增 `shape_template: tuple[int|str, ...]` 字段（保持 `shape` 字段为捕获时的具体值，不破坏现有 serde）

**歧义处理**：若 `hidden == seq_len`（罕见），标签优先级 `S > B > 静态`；启动 trace 前在 `cli.py` 校验几何参数与 `seq_len` 不重合，否则警告。

### 2.3 内置模型库布局
```
python/zrt/training/builtins/
├── __init__.py
├── registry.py                # builtin_registry.load("deepseek_v3", phase="prefill")
├── models/
│   ├── deepseek_v3.prefill.json    # 序列化的 OpGraph (复用 ir/serde.py)
│   ├── deepseek_v3.decode.json
│   ├── deepseek_v3.train_forward.json
│   ├── deepseek_v3.train_backward.json
│   ├── deepseek_v3.meta.yaml       # 捕获时几何参数 + 模型元信息
│   ├── llama3_70b.prefill.json
│   └── ...
└── README.md
```

`<model>.meta.yaml` 内容示例：
```yaml
model_id: deepseek_v3
captured_with:
  seq_len: 4096
  batch_size: 1
  num_layers_traced: 4
  num_layers_full: 61
geometry:
  hidden: 7168
  ffn: 18432
  num_heads: 128
  head_dim: 128
  vocab: 129280
  num_experts: 256
  num_routed_experts: 8
phases: [prefill, decode, train_forward, train_backward]
zrt_sim_version: <git-sha>
```

### 2.4 Path B 加载入口
新增 `training/io/builtin_loader.py`：

```python
def load_builtin_graph(model_id: str, phase: str, model: ModelSpec) -> training.ir.Graph:
    op_graph = builtin_registry.load(model_id, phase)            # OpGraph
    op_graph = retemplate(op_graph, seq_len=model.seq_len, ...)  # 按 ModelSpec 重算 shape
    return aggregate_to_training_ir(op_graph, model)             # OpGraph → training.ir.Graph
```

`Strategy` 配置新增字段 `builtin_model_id: str | None`；`SearchEstimator` / `estimate()` 中若设置则走加载分支，否则走 `build_graph(model, strategy)` 老路。**保持双轨兼容**。

---

## 三、分步任务列表

> 每个任务的 `current_state` 字段反映当前进度（pending / in_progress / done / failed）。

### Phase 1：形状模板化（Path A 增强）

#### Task 1.1 — 在捕获阶段记录 shape tags
- **改动**：`python/zrt/graph/dispatch.py:RecordingDispatch.__init__` 增加 `geometry_params: dict[str, int]` 参数；`__torch_dispatch__` 中对每个 tensor 维度调用新增辅助函数 `_tag_dims(shape, geometry_params)` 生成 tag 列表，存入 record。
- **关联**：`graph/main.py:_trace_phase()` 调用 `RecordingDispatch` 时传入 `{"seq_len": ..., "batch_size": ..., "hidden": ..., "ffn": ..., ...}`（从 HF config + CLI 收集）。
- **验收**：单测 `tests/graph/test_shape_tags.py`：用 `seq_len=4096, hidden=7168` trace 一个 2 层 Llama，断言 record 中 QKV 投影 input shape `(4096, 7168)` 对应 tags `("S", 7168)`。
- **当前状态**：done

#### Task 1.2 — 扩展 TensorMeta 携带 shape_template
- **改动**：`python/zrt/ir/types.py:TensorMeta` 新增 `shape_template: tuple[int|str, ...] | None = None`（`@field(default=None)` 保持向后兼容）。`adapter.py:records_to_opgraph` 把 record 中的 tags 写入。
- **关联序列化**：`python/zrt/ir/serde.py` 中 TensorMeta 的 to_dict/from_dict 加上新字段（None 时省略，保持旧 JSON 可读）。
- **验收**：旧的 OpGraph JSON dump 仍能被 `load_json` 读入；新 trace 的 OpGraph 中 `tensor.shape_template` 非 None 且与 `shape` 同长度。
- **当前状态**：done

#### Task 1.3 — 实现 retemplate 函数
- **新增文件**：`python/zrt/ir/retemplate.py:retemplate(op_graph, **bindings) -> OpGraph`
- **逻辑**：遍历所有 TensorMeta，按 `shape_template` 把 `"S"/"B"/"BS"` 替换为新 `seq_len`/`batch_size` 值；recompute `mem_bytes`。
- **验收**：单测：原始 OpGraph 用 `seq_len=4096` 捕获，`retemplate(g, seq_len=8192)` 后所有标 `"S"` 的维度从 4096 → 8192，标静态的不变。
- **当前状态**：done

---

### Phase 2：内置模型库（持久化）

#### Task 2.1 — 设计 registry 接口
- **新增文件**：`python/zrt/training/builtins/__init__.py`、`registry.py`
- **API**：
  ```python
  builtin_registry.list_models() -> list[str]
  builtin_registry.list_phases(model_id: str) -> list[str]
  builtin_registry.load(model_id: str, phase: str) -> tuple[OpGraph, dict]  # (graph, meta)
  builtin_registry.save(model_id: str, phase: str, graph: OpGraph, meta: dict) -> None
  ```
- **路径解析**：基于 `python/zrt/training/builtins/models/` 目录下的 JSON + YAML 文件，扫描得到模型列表。
- **验收**：单测：往临时目录写一个假 OpGraph + meta，`registry.load` 取回后字段一一对应。
- **当前状态**：done

#### Task 2.2 — CLI 增加 capture 子命令
- **改动**：`python/zrt/cli.py` 新增 `--capture-builtin <model_id>` 模式，自动 trace `prefill`/`decode`/`train_forward`/`train_backward` 四个 phase 并写入 builtins 目录；同步生成 `<model>.meta.yaml`。
- **验收**：`python -m python.zrt deepseek-ai/DeepSeek-V3 --layers 4 --capture-builtin deepseek_v3` 后，`python/zrt/training/builtins/models/deepseek_v3.*.json` 全部生成。
- **当前状态**：done

#### Task 2.3 — 首批捕获三个模型
- **执行**：用 Task 2.2 工具捕获 `llama3_70b`、`deepseek_v3`、`qwen2_5_7b`，写入仓库。
- **验收**：`pytest tests/training/builtins/test_registry.py` 通过；`registry.list_models()` 返回包含这三个。
- **当前状态**：done

---

### Phase 3：IR 桥接（OpGraph → training.ir.Graph）

#### Task 3.1 — 实现聚合规则
- **新增文件**：`python/zrt/training/ir/from_opgraph.py`
- **核心函数**：`aggregate_to_training_ir(op_graph: OpGraph, model: ModelSpec) -> training.ir.Graph`
- **算法**：
  1. 按 `OpNode.layer` 分桶（每层一个 group）
  2. 在每层内按 `scope` 关键字（见 §2.1 表格）匹配子组
  3. 每个子组生成一个 `training.ir.Op`，从子组首个 matmul 的 input/output shape 反推 `meta["m/n/k"]`
  4. 非 matmul 类（ln/rope/swiglu）从聚合范围内输入张量大小求 `bytes_fwd`
  5. 处理 MoE 层：检测 `aten.scatter`/`aten.index_select` + `aten.all_to_all` 模式 → `dispatch`/`combine`
- **验收**：用 Llama-3 trace 出的 OpGraph 转换后，对比 `build_graph(llama3_spec, ...)` 的结果，每层 op 数与 kind 序列一致；matmul 的 m/n/k 一致。
- **当前状态**：done

#### Task 3.2 — 桥接器对接 sharding
- **验证**：`shard.py:_apply_tp_sharding` 现有 `"qkv" in op.name`、`"o_proj" in op.name` 等子串匹配在 Task 3.1 产出的 `Op` 上仍然命中。
- **改动**：必要时调整 op.name 命名（保留 §2.1 表格中的命名约定）。
- **验收**：单测：对桥接生成的 graph 调用 `_apply_tp_sharding(tp=8)`，所有 matmul 节点的 `meta["n_local"]`/`meta["k_local"]` 被正确设置。
- **当前状态**：done

#### Task 3.3 — 处理 MoE/MTP（关键收益点）
- **改动**：`from_opgraph.py` 内识别 DeepSeek-V3 的 MoE 模式（`MoEGate` scope → router；`group_gemm`/批量 expert mm → `matmul` with `moe=True` flag；a2a 通信节点保留）。MTP 模块识别为独立 layer_kind（需 `LayerKind.MTP` 在 `training.ir` 中已定义）。
- **验收**：DeepSeek-V3 经桥接后，MoE 层不再退化为 dense_block，能识别到 router + 8 个激活专家 + dispatch/combine。
- **当前状态**：done

---

### Phase 4：Path B 接入

#### Task 4.1 — Strategy/SystemSpec 加字段
- **改动**：`python/zrt/training/spec/strategy.py:Strategy` 加 `builtin_model_id: str | None = None`；`io/config_loader.py` 支持 YAML 中的 `builtin_model_id` 字段。
- **验收**：现有所有 YAML 配置（`configs/*.yaml`）解析仍通过；新加 `builtin_model_id: deepseek_v3` 后能加载。
- **当前状态**：done

#### Task 4.2 — 加载分支
- **改动**：`training/search/estimator.py` 和 `training/api/estimate.py`（如存在）中：
  ```python
  if strategy.builtin_model_id:
      op_graph, meta = builtin_registry.load(strategy.builtin_model_id, phase="train_forward")
      op_graph = retemplate(op_graph, seq_len=model.seq_len, batch_size=strategy.micro_batch)
      graph = aggregate_to_training_ir(op_graph, model)
  else:
      graph = build_graph(model, strategy)  # 旧路径
  ```
- **验收**：现有 anchor 测试 `pytest tests/training/anchors/test_anchors.py` 走老路径仍通过；改 `deepseek_v3.yaml` 设 `builtin_model_id: deepseek_v3`，重跑 estimate，MFU 与未启用前差距可解释（应该更接近真实值）。
- **当前状态**：done

#### Task 4.3 — 更新示例配置
- **改动**：`python/zrt/training/configs/deepseek_v3_h100_8n.yaml`、`deepseek_v3.2_h100_8n.yaml` 加 `builtin_model_id` 字段；保留一份禁用内置库的对照配置。
- **验收**：两份配置都能跑通 `python -m zrt.training estimate --config <yaml>`。
- **当前状态**：done

---

### Phase 5：测试与文档

#### Task 5.1 — 端到端集成测试
- **新增**：`tests/training/test_builtin_path.py`，覆盖：
  - 加载 + retemplate + aggregate 全流程
  - 与 spec 路径输出对比（dense 模型应当近似一致）
  - DeepSeek-V3 内置库 vs spec 路径的 MFU 偏差可解释
- **验收**：`pytest tests/training/test_builtin_path.py -v` 通过。
- **当前状态**：done

#### Task 5.2 — 更新文档
- **改动**：`docs/spec_path_shape_analysis.md` 第七节 TODO 标注 MoE/MTP 已通过内置库解决；新增 `docs/builtin_model_library.md` 描述捕获/加载流程。
- **验收**：文档与代码一致；CLAUDE.md 简要提到 `builtin_model_id` 字段。
- **当前状态**：done

---

## 四、关键文件清单（待修改/新增）

| 文件 | 类型 | 说明 |
|------|------|------|
| `python/zrt/graph/dispatch.py:101-195` | 修改 | RecordingDispatch 增加 geometry_params + shape tagging |
| `python/zrt/graph/main.py:289-403` | 修改 | _trace_phase 把几何参数透传给 dispatch |
| `python/zrt/ir/types.py:145-183` | 修改 | TensorMeta 加 shape_template |
| `python/zrt/ir/serde.py:189-198` | 修改 | TensorMeta 序列化包含 shape_template |
| `python/zrt/ir/adapter.py:96-208` | 修改 | records_to_opgraph 写入 shape_template |
| `python/zrt/ir/retemplate.py` | 新增 | retemplate 函数 |
| `python/zrt/training/builtins/registry.py` | 新增 | 内置库注册表 |
| `python/zrt/training/builtins/models/*.json` | 新增 | 持久化的 OpGraph |
| `python/zrt/training/ir/from_opgraph.py` | 新增 | OpGraph → training.ir.Graph 聚合 Pass |
| `python/zrt/training/spec/strategy.py` | 修改 | 加 builtin_model_id |
| `python/zrt/training/io/config_loader.py` | 修改 | 解析 builtin_model_id |
| `python/zrt/training/search/estimator.py` | 修改 | 加载分支 |
| `python/zrt/cli.py` | 修改 | --capture-builtin 子命令 |

---

## 五、验证（端到端）

按以下顺序自顶向下检验：

1. **形状模板化单测**：`pytest tests/graph/test_shape_tags.py tests/ir/test_retemplate.py -v`
2. **内置库单测**：`pytest tests/training/builtins/ -v`
3. **桥接器单测**：`pytest tests/training/ir/test_from_opgraph.py -v`
4. **回归测试**（不启用 builtin）：`pytest tests/training/anchors/test_anchors.py -v`（GPT-3、LLaMA-3、DeepSeek-V3 anchor 仍通过）
5. **集成测试**（启用 builtin）：`pytest tests/training/test_builtin_path.py -v`
6. **CLI 端到端**：
   ```bash
   # 捕获
   python -m python.zrt deepseek-ai/DeepSeek-V3 --layers 4 --capture-builtin deepseek_v3
   # 用 spec 路径估算（旧）
   PYTHONPATH=python python -m zrt.training estimate --config python/zrt/training/configs/deepseek_v3_h100_8n.yaml
   # 用 builtin 路径估算（新）
   PYTHONPATH=python python -m zrt.training estimate --config python/zrt/training/configs/deepseek_v3_h100_8n_builtin.yaml
   # 对比两份 report，MFU 应可解释（builtin 更精确）
   ```

---

## 六、风险与未决问题

1. **聚合规则脆弱**：scope 关键字匹配依赖模型作者命名；不同 HF 实现（如 Mixtral vs DeepSeek MoE）需逐一调试。**缓解**：从覆盖最广的几个 (Llama/Qwen/DeepSeek/Mixtral) 起步，新模型按需扩展。
2. **shape_template 歧义**：若几何参数恰巧等于 seq_len（如 hidden==seq_len），需要优先级处理 + 启动校验。
3. **decode 阶段 KV cache 复杂性**：decode 的 attention K/V shape 来自历史长度（动态），不止 query_len=1。需要把 `pos_start`/历史长度也作为 shape 来源标签的一种。
4. **版本漂移**：内置库 JSON 与代码版本绑定。引入 `zrt_sim_version` 字段并在加载时校验，不匹配时警告。
5. **粒度落差仍残留**：聚合后的 training.ir.Graph 仍是 12-op/层；某些 Path A 真实存在的细粒度算子（如 RMSNorm 的具体内核序列）在聚合后丢失。**缓解**：将 `bytes_fwd` 和 FLOPs 从聚合范围内累加，保证总量准确即可。

---

## 七、当前状态总览

- Phase 1（形状模板化）：done（3/3 Task 完成）
- Phase 2（内置模型库）：done（2/3 Task 完成，Task 2.3 首批模型捕获待后续执行）
- Phase 3（IR 桥接）：done（3/3 Task 完成）
- Phase 4（Path B 接入）：done（3/3 Task 完成）
- Phase 5（测试与文档）：done（18 new tests pass, 13 anchor regression tests pass）
