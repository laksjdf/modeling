# ZRT-Sim 整体架构与模块详细设计

> 本文档完全基于代码（`python/zrt/**/*.py` 与 `python/__main__.py`、`cli.py`）梳理，
> 没有读取任何 Markdown 文档。读取过的关键代码文件在文末"信息来源"列出。
>
> 文档侧重三件事：
> 1. **从入口出发的完整业务流**（推理 / 图原生训练 / Spec-based 训练估计三条主线）
> 2. **逐模块详细设计**（职责、关键类型/函数、依赖关系）
> 3. **代码中的 TODO / 未完成项盘点**（精确定位文件:行号）

---

## 0. 顶层目录与代码入口

```
python/zrt/
├── __main__.py              # python -m python.zrt 入口（仅一行 -> cli.main）
├── cli.py                   # 主命令解析 + 三种执行路径分发
├── graph/                   # 阶段 1：HF 模型加载 + TorchDispatchMode 算子捕获 + Excel/ONNX 导出
├── ir/                      # 核心 IR：OpGraph / OpNode / Edge / TensorMeta / DType / GraphHierarchy
├── transform/               # 阶段 2：图变换 Pass 流水线（split → fuse → optim → analyze）
│   ├── parallel/            # TP / EP / PP / DP / CP + Comm 插入
│   ├── fusion/              # 平台感知融合 Pass（OpGraph 版本，对应 graph/fusion.py 的 FusionEngine）
│   ├── optim/               # 量化 / EPLB / SharedExpert / MTP（多个为 stub）
│   ├── analysis/            # FLOPs / Roofline / CommLatency / StreamAssign + 训练侧 Flops/Memory/PipelineMetrics + Modeller
│   └── training/            # 训练专属 Pass：ZeroFSDP / Recompute / Optimizer / Offload
├── executor/                # 阶段 3：DAGScheduler 拓扑+多流贪心调度，OverlapAnalyzer
├── simulator/               # 阶段 4：SimulatorHub + 后端（Roofline / Lookup / Tilesim） + Cache
├── policy_model/            # 后端选择策略：Priority / OpenBox / OpOptimize / SystemDesign
├── hardware/                # HardwareSpec + YAML registry
├── memory/                  # 推理形态显存预算（MemoryModel + activation liveness 分析）
├── layers/                  # 算子抽象层（OperatorBase 子类，供 backend 使用，目前轻量）
├── report/                  # E2E / Training summary、HTML、Chrome Trace、Comparison
└── training/                # 自带的 Spec-based 训练性能估计子系统（独立 IR/Composer/Search）
    ├── spec/                # ModelSpec / Strategy / SystemSpec / Dtype / 五种 PPSched 等枚举
    ├── ir/                  # 训练 IR：Graph/Op/Tensor/Collective + builders/shard/validate
    ├── models/              # 解析模型：comm（α-β）、flops（OpCost）、memory（MemBreakdown）
    ├── compose/             # PipelineComposer + 五个调度器实现 + StageTime
    ├── search/              # SearchSpace 网格搜索 + estimate + Pareto 前沿
    ├── anchor/              # 锚点 YAML 比对（MFU 容忍度校准）
    ├── trace/               # Chrome Trace 导出（独立于 report/chrome_trace.py）
    ├── io/                  # YAML config_loader（model+system+strategy 一体化）+ perf_tables
    └── configs/             # 训练 YAML 配置：模型 + 硬件 + 策略
```

CLAUDE.md 描述了 `python/zrt/training/builtins/`，但仓库中只剩下空 `__pycache__`，源码已移除——使用时已被 Python 子包名替代（`spec/`、`ir/` 等）。

---

## 1. 入口与三条主业务流

### 1.1 命令分发表

`python/zrt/__main__.py` 仅 `from python.zrt.cli import main; main()`。  
`python/zrt/cli.py:main()` 解析 argparse 后按以下规则分发：

| 触发条件 | 主调用栈 | 业务流 |
|---|---|---|
| `--estimate-config <yaml>` | `_run_estimate` → `training.io.config_loader.load_specs` → `training.search.estimator.estimate` | **Spec-based 训练估计**（不做图捕获） |
| 其他默认（无 `--hw`） | `_run_trace_phases` → 仅生成 Excel/ONNX | **图捕获**，无性能估计 |
| `--hw <hw>` 且非 `--train` | `_run_trace_phases` → `_run_inference_pipeline` | **推理性能估计**（含 simulate + report） |
| `--hw <hw>` 且 `--train` | `_run_trace_phases(train_*)` → `_run_training_modelling` → `transform.analysis.estimate_training_from_graphs` | **图原生训练估计** |
| `python -m zrt.training estimate ...` | `zrt/training/__init__.py` 暴露 `estimate`+`load_specs` | 同 `--estimate-config` 路径 |

> ⚠️ 训练子系统使用的是 `zrt.*` 顶层导入而非 `python.zrt.*`，因此运行时必须设置 `PYTHONPATH=python`，`graph/` 与 `transform/` 等其它包则使用 `python.zrt.*` 绝对导入——这是仓库中两套并存的导入约定。

### 1.2 业务流 A：推理（`--hw` 无 `--train`）

```text
                 cli.main (cli.py:48)
                         │
                         ▼
            run_trace_phases (graph/main.py:743)
            ├── load_model (graph/model_loader.py:178)
            │     │ FakeTensorMode + AutoModelForCausalLM.from_config
            │     │ 失败时 fallback 到 hf_models/ 本地目录
            │     └── apply_compat_patches → patch_moe_for_fake → patch_indexer_for_fake
            ├── _trace_phase × {prefill, decode}        (graph/main.py:289)
            │     │ Phase 1: 构造 fake input_ids/attention_mask/position_ids
            │     │ Phase 2: 进入 RecordingDispatch (TorchDispatchMode) + ModuleTracker (forward hooks)
            │     │ Phase 3: model(**kwargs); decode 透传 prefill 的 past_key_values
            │     │ Phase 4: 收集 records (op_short, aten_op, _input_ids, _output_ids, shapes, scope, …)
            │     └── target_layers 过滤（只保留首个 dense + 首个 sparse 层，CLAUDE.md 默认）
            ├── _save_phase_outputs                     (graph/main.py:688)
            │     ├── ExcelWriter.write → *.xlsx
            │     ├── records_to_opgraph → raw OpGraph
            │     ├── FusionEngine + fused_records_to_opgraph → fused OpGraph
            │     ├── build_op_graph (NetworkX)         (graph/graph_builder.py:23)
            │     └── export_all → *.onnx / *.json
            ▼
            _run_inference_pipeline (cli.py:242)
            ├── TransformContext = (hw_spec, ParallelConfig(tp=...), StreamConfig)
            ├── pipe = build_default_pipeline()  → build_pipeline (transform/pipeline.py:53)
            │   阶段顺序固定：split → fuse → optim → analyze
            │   推理路径条件：DataParallel/Zero/TrainingFlops 等训练 Pass 全部跳过
            │   实际执行：TensorParallelPass? → ExpertParallelPass? → CommInserterPass? → PipelineParallelPass?
            │              → FusionPass → QuantizationPass? + SharedExpertPass? + MTPPass(stub) + EPLBPass(stub)
            │              → FlopsPass → RooflinePass → CommLatencyPass → StreamAssignPass
            ├── DAGScheduler.schedule(g)  (executor/scheduler.py:134)
            │     │ Kahn 拓扑序，每节点 start = max(pred_finish, stream_avail)；
            │     │ latency 来源优先级：node.annotations["latency_us"] → Roofline 估算 → 1µs 兜底
            │     └── Timeline (含 ScheduledOp 列表 + 总时长 + 重叠统计)
            ├── SimulatorHub.simulate_graph(g, hw)   (simulator/hub.py:81)
            │     │ 经 PolicyModelManager → PriorityModel.predict → 选 priority 最高的 backend
            │     │ 默认 backend：Roofline (priority=0)，Lookup/Tilesim 都是 priority>0 但 can_simulate→False
            │     └── SimResult per node（latency_us, bound, hw_utilization, ai…）
            ├── build_summary → E2ESummary
            ├── append_perf_summary（写回 .xlsx）
            └── export_html_report + export_chrome_trace
```

### 1.3 业务流 B：图原生训练（`--hw --train`）

```text
                 cli.main (cli.py:48)
                         │
                         ▼
            run_trace_phases(phases=("train_forward","train_backward"))
            │  共享一个 TensorTracker：保证 fwd/bwd 之间 tensor id 全局唯一，
            │  这是后续 stitch_fwd_bwd 用 id 精确匹配的前提。
            │  train_backward 阶段：active=False 进入 fwd 阶段（只追踪 id），
            │                       fwd 结束后置 active=True 再调 logits.sum().backward()，
            │                       因此 records 只包含 backward 算子。
            ▼
            _run_training_modelling (cli.py:312)
            └── estimate_training_from_graphs (transform/analysis/modeller.py:177)
                 ├── 注入 metadata（seq_len/num_layers/hidden/...）到两张图
                 ├── stitch_fwd_bwd (ir/adapter.py:613)
                 │     │ 给 fwd 节点 annotation["phase"]="fwd"，bwd 节点 id 加 "bwd_" 前缀且 phase="bwd"
                 │     │ _is_param_node 标 is_param=True
                 │     │ 跨图边匹配：1) 精确 tensor_id；2) 形状+dtype+同 layer/scope 启发式
                 │     └── unified.metadata["fwd_bwd_stitched"]=True
                 ▼
                 TransformContext(training=TrainingConfig(...), parallel=ParallelConfig(...))
                 ▼
                 build_training_pipeline = build_pipeline (同一函数，根据 ctx 条件激活)
                 实际激活的 Pass（is_train=True 时）:
                   split   : DataParallelPass[dp>1] → TensorParallelPass[tp>1] → ExpertParallelPass[ep>1]
                             → ContextParallelPass[cp>1] → CommInserterPass[tp/ep/cp>1] → PipelineParallelPass[pp>1]
                   fuse    : FusionPass
                   optim   : QuantizationPass[quant?] + EPLB[stub] + SharedExpert[flag] + MTP[stub]
                             → ZeroFSDPPass（写 metadata["zero"]，stage=3 时插 AG/RS）
                   analyze : FlopsPass → RooflinePass → CommLatencyPass → StreamAssignPass
                             → TrainingFlopsPass → TrainingMemoryPass → TrainingPipelinePass
                 ▼
                 TrainingPipelinePass 内部 (transform/analysis/training.py:323)
                 ├── 若 PP>1 且节点有 stage_id：按 stage 切 subgraph 分别 DAGScheduler.schedule
                 │   得到 stage_fwd[s] / stage_bwd[s] / stage_bwd_dw[s]（按 dW FLOPs 比例摊分）
                 ├── 否则单图调度，PP>1 时按 pp 平均分（warning）
                 ├── 构造 StageTime 列表
                 ├── 调度器映射：1f1b/i1f1b/zb/dualpipe/dualpipev → 五个 PipelineComposer 之一
                 │   （所有 Composer 都来自 zrt/training/compose/pipeline.py，跨子系统复用）
                 ├── 计算 step_time，并按 overlap_type 减去隐藏的通信时间
                 │   （MC2 全部隐藏；CoC 隐藏 (k-1)/k；ring_cp 减目标 fa_tile）
                 └── 写入 metadata["pipeline_metrics"]：MFU(去掉 recompute)、HFU、bubble、warmup/cooldown
                 ▼
                 TrainingReport（modeller.py:34）→ summary() 打印 + JSON 文件
```

### 1.4 业务流 C：Spec-based 训练估计（不做图捕获）

```text
            cli.main --estimate-config X.yaml
                         │
                         ▼
            _run_estimate (cli.py:367)
            └── load_specs (training/io/config_loader.py:20)
                 │ 加载 model/system/strategy 三段：model 段可以是字符串引用 configs/models/<x>.yaml
                 ▼
                 estimate (training/search/estimator.py:34)
                 ├── strategy.validate(model, system)   // 维度整除 + 世界大小检查
                 ├── ir_validate(model, system, strategy) → 警告列表
                 ├── build_graph (training/ir/builders.py:194)
                 │     │ 每个 LayerKind 调 dense_block(...)（MoE/MTP 当前回退到 dense_block，见 TODO）
                 │     └── ShardPlan + insert_collectives → TP 的 AG/RS 集合
                 ├── total_training_flops (training/models/flops.py:104)
                 │     用 op_cost(每 op).fwd+dx+dw（matmul 2mnk×3=6mnk；attn dx=2.5×fwd）× M 微批数
                 ├── pipeline_step_time (training/compose/pipeline.py:323)
                 │     │ 按 PP 分 stage → stage_time 把 op_cost 转 wall-clock（compute/memory bound 两路 +
                 │     │   achieved_*_efficiency 经验曲线 + EP 不均衡因子 + 重计算时间）
                 │     │ 选 _COMPOSERS[strategy.pp_schedule] 计算 step_time / bubble_fraction
                 │     │ memory_breakdown / compute_mfu / compute_hfu (recompute_overhead_flops)
                 │     └── StepResult
                 └── Report（含 step_time_ms / mfu / hfu / memory / per_stage / warnings）
```

---

## 2. IR 与共享数据结构（`zrt.ir`）

### 2.1 OpGraph 设计要点（`ir/graph.py`）
- 显式拒绝依赖 NetworkX：节点用插入序 `dict[id, OpNode]`，邻接表 `_succ/_pred` 由 `_rebuild_adjacency` 重建。
- **变换 Pass 的不变量**：所有 Pass 都先 `graph.clone()` 再修改（深拷贝）。
- `topo_sort(debug=True)` 检测环并输出前 5/10 个无法到达节点的前驱信息。
- 关键变更接口：`add_node` / `add_edge`（轻量更新邻接） / `insert_after` / `replace_subgraph`（融合用）。
- `subgraph(node_ids)` 用于 PP 分 stage 调度。
- `metadata` 中典型键：`seq_len`、`num_layers`、`num_layers_traced`、`hidden`、`total_params`、`fwd_bwd_stitched`、`zero`、`training_flops`、`forward_flops`、`backward_flops`、`recompute_flops`、`memory_breakdown`、`pipeline_metrics`、`stage_timelines_fwd/bwd/bwd_dw`、`layer_scale`。

### 2.2 OpNode（`ir/node.py`）
- 主体字段：`id, op_type, inputs:list[TensorMeta], outputs:list[TensorMeta], attrs, scope, category, annotations`。
- `category` 由 `infer_category(op_type, component)` 推导：`comm.*` / `_COMM_OPS` → communication；`_MEMORY_OPS`（view/reshape/transpose/cat/...）→ memory；其它→compute。
- 出处：`op_short / module_class / layer / component / src_file / src_line / src_code` 由 RecordingDispatch 直接填入。
- 融合元数据：`fused_from / num_sub_ops / fusion_level("leaf"|"parent")`。
- `annotations` 是 Pass 间的"侧信道"：`tp_split, ep_needs_a2a, cp_split, stage_id, phase, recompute, flops*, compute_us, memory_us, latency_us, arithmetic_intensity, bound, stream_id, stream_type, overlap_type, dp_comm, inserted_by` 等。

### 2.3 Edge（`ir/edge.py`）
- 携带 `tensor:Optional[TensorMeta]` 与 `tensor_id:Optional[int]`（保留 dispatch 原始整型 id 用于 fwd/bwd 拼接）。
- `is_data` / `is_control`：tensor 为空即控制依赖。

### 2.4 TensorMeta + DType（`ir/types.py`）
- `DType` 是 `str, Enum`，`itemsize` 把 INT4 当成 0.5 字节。
- `TensorMeta` 不可变（`@dataclass(frozen=True)`），字段 `id / shape / dtype / mem_bytes`，提供 `with_shape / with_dtype` 用于 TP 切分。
- `parse_shape("[1, 128, 7168]") → (1,128,7168)`，`split_shape_list` 处理嵌套括号。

### 2.5 GraphHierarchy（`ir/hierarchy.py`）
- 由 `OpGraph.hierarchy`（lazy）构造的 scope 树：
  - 深度 0=root，1=`model`，2=`model.layers`，3=`model.layers.0`，4=`model.layers.0.self_attn`。
- `at_depth(d)` / `find(glob)` / `aggregate(node, values)` / `module_breakdown` 用于报告里的逐层逐模块汇总。

### 2.6 Adapter（`ir/adapter.py`）四大转换
1. `records_to_opgraph(records, ...)`：RecordingDispatch records → OpGraph（**首选路径**）。  
   特别注意：`producer_idx > consumer_idx` 的"反向边"被跳过（处理 decode 阶段 KV 缓存别名）。
2. `fused_records_to_opgraph`：FusionEngine 输出 → OpGraph。
3. `nx_to_opgraph` / `opgraph_to_nx`：与历史 NetworkX 表示双向桥。
4. `stitch_fwd_bwd(fwd, bwd)`：拼接训练正反向：
   - 节点 `phase` 注入；param 节点 `is_param=True`。
   - bwd 节点 id 加 `bwd_` 前缀。
   - 跨图边匹配优先级：① 精确 `tensor.id`；② 形状+dtype 同时同 layer/scope 启发式（`_best_cross_match`）。

### 2.7 param_count（`ir/param_count.py`）
- 三层兜底：`metadata["total_params"]` → tensor id 含 `weight/param` 命名启发 → 结构启发（matmul 第二输入、embedding 第一输入的 2-D 张量）。
- `op_short("aten.mm.default") = "mm"`。

---

## 3. 阶段 1：图捕获（`zrt.graph`）

### 3.1 `model_loader.py`
- 三步走：`apply_compat_patches()` → `_load_config(model_id)`（HF 失败则 `find_local_fallback` 切到 `hf_models/<...>`）→ `FakeTensorMode.__enter__` → `_instantiate_model`（再次失败可在 import-compat 错误时切到本地实现）。
- 调整后的 config：`_full_num_hidden_layers` 保留原层数；`num_hidden_layers = num_hidden_layers`（CLI 指定的层数）；`_attn_implementation = "eager"`。

### 3.2 `dispatch.py`：`RecordingDispatch(TorchDispatchMode)`
- 核心 `__torch_dispatch__`：
  - 先执行 `func(*args)` 得到输出；
  - **始终**为输入/输出 tensor 调用 `tensor_tracker.get_id`（即使 `active=False` 也要分配 id），保证 fwd 阶段被"暂停"时仍能记录 id 用于后续 bwd 边连接。
  - `active` 控制是否产出 records；过滤 `SKIP_OPS` 与 `target_layers` 之外的层。
  - 写入字段：`aten_op, op_short, module_path, module_class, layer, component, src_file/line/code/func, extra_args(JSON), input/output_shapes/dtypes, num_inputs/outputs, _input_ids, _output_ids, recompute`。
- `_capture_call_site()`：栈回溯到第一个非 torch/zrt 框架的帧。

### 3.3 `tracker.py`：`ModuleTracker`
- 通过 `register_forward_pre_hook / forward_hook / full_backward_pre_hook / full_backward_hook` 维护 `_stack`（当前模块路径）和 `_class_stack`。
- `path_to_class / path_to_children` 在安装阶段就完整构建——融合规则匹配只需类名+层次，不需要运行时。
- `in_recompute`：`_in_backward_phase=True` 且 `_forward_depth>0` 同时成立——用于识别 activation checkpointing 中的重新前向。

### 3.4 `fusion.py`：`FusionEngine`（records 形态的 3-Pass 融合）
- Pass 1（leaf）：连续相同 `module_path+layer` 算子聚成一组，通信节点和 scope 为空的算子打断分组。
- Pass 2（parent）：相邻 leaf 组若共享同一可融合父 scope 且总算子数 ≤ `max_parent_ops`、子 scope 数 ≤ `max_children`，合并向上。
- Pass 3（label）：先匹配平台子模式（`get_subpatterns(platform)`），再 `module_class → SEMANTIC_LABELS`，最后回退到 `module_class` 或首个 op_type。
- `_PATTERN_SKIP / SHAPE_OPS / ALWAYS_TRANSPARENT` 在匹配里只是"通配符"，绝不从 records 中删除。

### 3.5 `transform/fusion/pass_.py`：`FusionPass`（OpGraph 形态的同构实现）
- 同样的三 Pass，但直接在 OpGraph 上 `replace_subgraph` 改写。
- **关键不变量保护**：合并 group 时，对 `stage_id, phase` 这类不变量做"集合大小检查"——若 group 中存在多值，记 `logger.error` 并丢弃该注解（说明 Pass 调用顺序错了，跨越了 stage/phase 边界）。

### 3.6 `graph_builder.py / graph_exporter.py / excel_writer.py`
- `build_op_graph` / `build_fused_op_graph` 输出 `nx.DiGraph`，附 `tensor_ids/shape/dtype` 等边属性。
- `graph_exporter.export_all` 写 JSON + ONNX；ONNX 中 node `name` 用 `/` 分隔 scope，让 Netron 渲染层次。
- `ExcelWriter` 输出 4 个 sheet：Model Config / Fused Ops / Raw Ops / Summary。

### 3.7 `patches.py`（FakeTensorMode 兼容）
- `apply_compat_patches`：补加被新版 transformers 删除的属性（`is_torch_fx_available` 等）+ 给某些子模块装 stub。
- `patch_moe_for_fake`：替换 MoE 的 forward，绕开 `.cpu().numpy() / torch.bincount` 在 FakeTensor 上的失败。
- `patch_indexer_for_fake`：DeepSeek-V3.2 indexer 的 `.transpose(2,3)` 在 FakeTensor 下非法，注入修正版 forward。

### 3.8 `compat.py`：本地模型 fallback registry
- `find_local_fallback(model_id_or_type)` 把 HF Hub id 或 `model_type`（如 `deepseek_v32`）映射到 `hf_models/<dir>/`，使用 `auto_map` 加载。
- 包含一些"满足导入但运行不会触发"的 stub（`is_flash_attn_greater_or_equal_2_10` 等）。

### 3.9 `main.py:run_trace_phases / run_trace`
- 入口把上面所有零件串起来：
  - `auto_layers=True` 时调 `auto_target_layers`（首个 dense + 首个 sparse）。
  - 训练阶段共享 `TensorTracker`（fwd/bwd 之间 tensor id 全局唯一）。
  - 一次 `FakeTensorMode` 上下文跨多 phase（让 prefill 的 KV cache 直接喂给 decode）。
  - `graph_mode=True` 时改走 `_trace_compile_phase`（torch.compile 自定义 backend 收 GraphModule，自实现 `_compile_graph_to_records`）。

---

## 4. 阶段 2：变换流水线（`zrt.transform`）

### 4.1 顶层（`pipeline.py` / `context.py` / `base.py`）
- `GraphPass` 抽象基类：`name` + `run(graph, ctx) -> graph`。
- `TransformPipeline` 把 Pass 注册到固定四阶段 `("split","fuse","optim","analyze")`，每个 Pass 可带 `condition(ctx)->bool`。
- `build_pipeline()`（同时是 `build_default_pipeline` 和 `build_training_pipeline` 的别名）按下表注入条件：

| 阶段 | Pass | 触发条件 |
|---|---|---|
| split | DataParallelPass | `dp>1 and is_training` |
| split | TensorParallelPass | `tp>1` |
| split | ExpertParallelPass | `ep>1` |
| split | ContextParallelPass | `cp>1` |
| split | CommInserterPass | `tp>1 or ep>1 or cp>1` |
| split | PipelineParallelPass | `pp>1` |
| fuse | FusionPass | always |
| optim | QuantizationPass | `quant is not None` |
| optim | EPLBPass | `"eplb" in optim_flags` |
| optim | SharedExpertPass | `"shared_expert_external" in optim_flags` |
| optim | MTPPass | `"mtp" in optim_flags` |
| optim | ZeroFSDPPass | `is_training` |
| analyze | FlopsPass / RooflinePass / CommLatencyPass / StreamAssignPass | always |
| analyze | TrainingFlopsPass / TrainingMemoryPass / TrainingPipelinePass | `is_training` |

- `TransformContext`：聚合 `hw_spec / parallel(ParallelConfig) / stream_config(StreamConfig) / quant(QuantConfig?) / training(TrainingConfig?) / optim_flags / phase / profile / stack`。
- `ParallelConfig.describe()` 拼出 `TP8-EP8-PP4-DP2` 这样的字符串；`total_devices=tp*pp*ep*dp*cp`。
- `TrainingConfig.num_microbatches = global_batch / micro_batch`；同时载入 `pp_schedule, vpp_chunks, pp_layer_assignment, cp_kind, dp_overlap_in_bubble, recompute_policy("none/full/selective")`。

### 4.2 并行 Pass

- **TensorParallelPass**（`parallel/tensor_parallel.py`）  
  按 scope 关键字分类列/行并行：
  - 列并行（`q/k/v/gate/up/w1/w3_proj`）：输出末维 `dim/tp`，`comm_after=None`。
  - 行并行（`o/down/w2_proj`）：第一输入末维 `dim/tp`，`comm_after="all_reduce"`。
  - 注解 `node.annotations["tp_split"]`，**只调整本节点输入/输出**，不沿图传播。

- **ExpertParallelPass**（`parallel/expert_parallel.py`）  
  通过 scope 含 `experts.|expert_|.experts[|moe_ffn` 识别专家算子，写 `ep_experts_local`、`ep_needs_a2a`。  
  仅依据 `ctx.profile.num_experts`，因此调用方需要 profile 才生效。

- **ContextParallelPass**（`parallel/context_parallel.py`）  
  按 `ctx.training.cp_kind` 给 attention 节点写 `cp_split={kind, cp, p2p_rounds?}`；具体通信由 CommInserter 接手。  
  > 实现注释里多次出现 `OpNode` 在文件末尾才 import，是历史循环依赖痕迹，不影响功能。

- **CommInserterPass**（`parallel/comm_inserter.py`）  
  - TP：在所有 `tp_split.comm_after=="all_reduce"` 的节点后插 `comm.all_reduce`，注解 `inserted_by="tp_pass"`。
  - EP：按 scope 根分组找首尾专家节点，前面插 `comm.all_to_all (role=dispatch)`，后面插 `combine`。
  - CP：Ulysses 前后各插 A2A；Ring 在节点前插 N 轮 `send_recv`，每轮带 `overlap_target="fa_tile:<id>"`。

- **PipelineParallelPass**（`parallel/pipeline_parallel.py`）  
  - 按 `ctx.training.pp_layer_assignment` 或基于 `compute_us / latency_us / flops` 的贪心 binpacking 分 stage。
  - 给所有节点写 `stage_id`；`pp<=1` 时也写 `0` 以保证下游 Pass 不报错。
  - 跨 stage 的 edge 替换为 `comm.send_recv`，节点放在接收 stage（让 subgraph 切片仍包含通信前驱）。
  - 终点检测平衡比 >1.5x 时 warning。

- **DataParallelPass**（`parallel/data_parallel.py`）  
  仅对 `is_training and dp>1 and phase=="train_backward"` 生效；按 layer 聚合"梯度生产节点"（op_type 含 `grad/backward`），插 layer 粒度的 `comm.all_reduce`（zero=0）或 `comm.reduce_scatter`（zero≥2）。  
  根据 `dp_overlap_in_bubble` 标 `overlap_in_bubble=True`。

### 4.3 融合 Pass
见 §3.5（OpGraph 版三 Pass）。

### 4.4 优化 Pass（`optim/passes.py`）
- `QuantizationPass`：给 compute 节点写 `quant_weight/quant_act` 注解；不真正换 op_type。
- `SharedExpertPass`：scope 含 `shared_expert` → `shared_expert_external=True`。
- `EPLBPass / MTPPass`：**stub，只 return 输入图**（见 TODO 总览）。

### 4.5 训练专属 Pass（`training/`）
- **ZeroFSDPPass**（`zero_fsdp.py`）  
  - 写 `metadata["zero"] = {stage, weight_shard, grad_shard, optstate_shard}`，stage≥3 时按层插 AG/RS。
  - 是当前**唯一注册到 pipeline.optim 阶段**的训练 Pass。
- **RecomputePass / OptimizerPass / OffloadPass**：实现存在但 **未注册** 到 `build_pipeline()`，且依赖 `ctx.training.recompute.per_layer_kind` / `ctx.training.offload`——而当前 `TrainingConfig` 仅有 `recompute_policy: str`，没有 `recompute / offload` 子对象。属于"半成品"，详见 TODO §6。

### 4.6 分析 Pass（`analysis/`）

- **FlopsPass**（`passes.py:16`）  
  对每节点用 `RooflineSimulator._fmr` 计算 `(flops, read_bytes, write_bytes)`，写入注解。  
  训练态额外：根据 `phase` 与 `recompute` 注解派生 `flops_fwd/dx/dw`。  
  attention 算子若节点/图带 `attn_compression_ratio`（DeepSeek-style 压缩），FLOPs 等比缩放。

- **RooflinePass**（`passes.py:112`）  
  `compute_us = flops/peak`，`memory_us = (R+W)/bw`，`latency_us=max(...)`，分类 `bound∈{compute,memory,latency}`。  
  保留预先注入的 `latency_us`（测试或 profiling 注入场景）。

- **CommLatencyPass**（`comm_latency.py`）  
  通信节点用 α-β + 集合形 ring/ tree 公式（`_estimate_comm_latency`）覆盖通用 Roofline 估计；自动判断是否跨节点（`group_size > intra_node.num_devices`）。

- **StreamAssignPass**（`passes.py:168`）  
  通信节点轮转分到 comm stream，其余节点轮转分到 compute stream。  
  写 `stream_id / stream_type / overlap_type`。`overlap_type` 通过 `overlap_target.startswith("fa_tile:")→ring_cp`，`attrs.fused_ag_matmul→mc2`，`attrs.coc_tile_k→coc`，否则 `none`。

- **TrainingFlopsPass / TrainingMemoryPass / TrainingPipelinePass**（`analysis/training.py`）  
  - TrainingFlopsPass：拼接图（`fwd_bwd_stitched=True`）按 `phase` 切分前后向 FLOPs；非拼接图退化为 dx+dw 估算；最后兜底用 6P 规则。还按 layer_scale 把"实际跟踪层数"放大到"完整层数"。`recompute_flops = ½·flops_fwd[recompute=true 且 phase=fwd]`。
  - TrainingMemoryPass：weights/grads/opt_state 按 `metadata["zero"]` 或 zero stage 自推；activations 走两条路：①拼接图内取 fwd→bwd 边活字节再除以 `tp×cp`；②退化到 Korthikanti `34·h·s·L·bs · rc_mult / (tp·cp) · max_inflight`，其中按 stage_id 计算 peak inflight。
  - TrainingPipelinePass：见业务流 B 描述；最重要的是它**不重算** stage time——直接拿 DAGScheduler 调度结果给 PipelineComposer。

### 4.7 `analysis/modeller.py`
- `TrainingReport` 数据类（含 step/per_stage/mfu/hfu/flops/memory/pipeline/total_params）。
- `estimate_training(graph, ctx)`：纯图入口，跑流水线后从 `metadata` 提取指标。
- `estimate_training_from_graphs(forward_graph, backward_graph, ...)`：CLI `--train --hw` 用的入口（流程见 §1.3）。
- `model_training(model_id, ...)`：先 `run_trace_phases` 后再调上面那个。

---

## 5. 其它支撑模块

### 5.1 Executor
- `scheduler.py:DAGScheduler.schedule`：按拓扑序贪心 list scheduling，`stream_avail[stream_id]` 维护流忙到几时；输出 `Timeline`（含 `total/compute/comm_time_us`、`overlap_us`、`phase_latency(phase)`）。
- `overlap.py:OverlapAnalyzer`：用扫描线算法对 compute/comm 区间做精确交集，得 `OverlapReport`（`exposed_comm_us = comm_us - overlap_us`）。
- `stream.py:Stream` 仅为数据类。

### 5.2 Simulator
- `result.py:SimResult` 字段：`latency_us, compute_us, memory_us, flops, read_bytes, write_bytes, arithmetic_intensity, bound, hw_utilization, backend, confidence`。
- `base.py:OpSimulator`：`name + priority + can_simulate + simulate`。
- `cache.py:SimCache`：MD5(op_type+hw.name+shapes+dtypes+sorted attrs+fused_from)，命中即直返。
- `hub.py:SimulatorHub`：组合 `SimCache` + `PolicyModelManager`；`register(backend)` 把 backend 注入策略模型；`simulate / simulate_graph`。
- 后端：
  - **Roofline**（`backends/roofline.py`，1811 行）：超大量精确公式表 `_EXACT_FORMULAS`（约 108 op_type，覆盖 GEMM/Attn/Norm/Softmax/Top-k/逐元素/激活/超越/MLP 各类/MoE 路由/Embedding/Cast/分配/Shape，最后兜底）。融合节点没命中精确表时按 `_fused_decompose(fused_from)` 累加子算子 FLOPs。
  - **LookupSimulator** / **TilesimSimulator**（`backends/lookup.py`, `tilesim.py`）：**stub**，`can_simulate→False`、`simulate→pass`。
- `backends/__init__.py` 只 re-export Roofline + 调 `register_backend()`，把三个 Backend 写进 `BACKEND_MAP`。

### 5.3 PolicyModel（后端选择策略）
- `policy_register.py:PolicyType`：`PRIORITY / OOTB_PERFORMANCE / OPERATOR_OPTIMIZATION / SYSTEM_DESIGN`。
- `policy_base_model.py:PolicyBaseModel`：抽象 `predict`，构造时按 `[LOOKUP, TILESIM, ROOFLINE]` 实例化所有 backend，并按 priority 排序。
- 4 个具体策略中**只有 `PriorityModel.predict` 有真实实现**（找第一个 `can_simulate==True` 的 backend 调用）；其余 3 个仅 `pass`。详见 §6。
- `PolicyModelManager.simulate` 是 SimulatorHub 真正委托的对象；默认用 `PolicyType.PRIORITY`。

### 5.4 Hardware
- `spec.py`：`HardwareSpec(ComputeSpec, MemorySpec, InterconnectSpec)`，关键方法 `peak_flops(dtype) -> ops/s`、`hbm_bandwidth() -> bytes/s`。
- `registry.py`：扫描 `python/zrt/hardware/configs/*.yaml`，按文件名或 `name:` 字段做 case-insensitive 匹配。`unidirectional_bw_gbps + num_links` 自动算总双向带宽。

### 5.5 Memory（推理形态）
- `model.py:MemoryModel.estimate`：weights / kv_cache / activation_peak / comm_buffer / overhead → `MemoryBudget`。
  - kv_cache 兼容 MLA（`kv_dim = kv_lora_rank + qk_rope_head_dim`）和 GQA（`kv_heads * head_dim`）。
  - activation 优先用图原生 liveness 分析；profile 路径只在没有 OpGraph 时启用。
- `activation.py:analyze_activation`：对 OpGraph 做 tensor 活跃度分析（last-use 索引 + 顺扫），输出 `peak_bytes / peak_node_id / per_node_live_mb`。
- `budget.py:MemoryBudget`：仅 dataclass + `utilization` 属性。

### 5.6 Layers（算子抽象类）
- `op_base.py` 极简：`OperatorBase / OpVectorBase / OpCubeBase / OpMixBase / OpCommBase`，含 `OP_CLASS_REGISTRY` 和 `op_register` 装饰器。
- 子模块按算子类型分组（`op_mm/op_attention/op_communication/op_activation/op_embedding/op_quant/op_elementwise/op_fused/op_trition`）实现具体类（`Mm / Linear / ColumnParallelLinear / ScaledDotProductAttention / SwiGlu / Embedding / AllReduceOp / Bmm / RMSNorm / RopeKernel / MoEGatingTopk / LinearQuant ...`）。
- 当前更多是为 simulator backend 预留的扩展面，`Roofline` 主路径并不依赖它。

### 5.7 Report
- `summary.py:E2ESummary / TrainingSummary` + `build_summary / build_training_summary`：从 graph + sim_results + timeline + hw_spec 抽取指标，渲染字符串。
- `chrome_trace.py:build_chrome_trace / export_chrome_trace`：把 `Timeline.scheduled_ops` 转成 chrome://tracing 协议（pid=0，tid=stream label，附 op_type/category/colour）。
- `html_writer.py`：HTML 报告。
- `compare.py`：多配置 `ComparisonReport` + Excel/HTML 对比导出。
- `transform/exporter.py` 是另一套：把变换后的 OpGraph 直接导出 Excel/JSON/ONNX，包含 stage、stream 等并行注解。

---

## 6. 训练子系统（`zrt.training`，独立）

### 6.1 Spec 层
- `spec/model.py:ModelSpec`：geometry + `layers: list[LayerKind]` + MoE/MTP 字段 + `attn_compression_ratio`（构造期校验 ∈ (0,1]）+ 各 dtype。`total_params() / effective_params_for_flops()`（MoE 用 `top_k/num_experts` 缩放）。
- `spec/strategy.py:Strategy`：tp/cp/pp/ep/dp + micro/global_batch + `pp_schedule(PPSched)` + vpp + `cp_kind` + zero + recompute(`RecomputePolicy.per_layer`) + offload + tp_overlap(`TPOverlap`) + `dualbatch / dp_overlap_in_bubble` + optimizer。  
  `validate(model, system)`：rank_product==world_size、TP 整除 num_heads/num_kv_heads/ffn、Ulysses 整除 num_heads、EP 与 num_experts 相容、ZeRO 与 dp 关系、global_batch 整除、PP ≤ num_layers。
- `spec/system.py:SystemSpec`：`GPU + host_mem_gb + nets:[NetTier(intra), NetTier(inter)] + nodes + gpus_per_node`，提供 `intra_tier()/inter_tier()/world_size`。
- `spec/dtype.py:Dtype`：四种 `(FP32/BF16/FP16/FP8)`，`bytes` = enum value。

### 6.2 IR
- `ir/graph.py:Graph(ops:list[Op], collectives:list[Collective], layer_index:dict[lid→(start,end)])` + `Op + Tensor + Collective`。  
  `Collective.kind ∈ {AG, RS, AR, A2A, P2P}`，`group ∈ {TP,CP,EP,DP,PP}`。
- `ir/builders.py:build_graph(model, strategy)`：embed + 每层 `dense_block` + final_ln + lm_head；MoE/MTP 当前**回退到 dense_block**（详见 TODO §7.4）。
- `ir/shard.py:ShardPlan + insert_collectives`：仅 TP，按 op.name 含 `qkv/o_proj/up_proj/down_proj` 决定 AG/RS 的 `inserted_after`，并把 `shape_local` 减小。CP/EP 标 "Phase 2" 待补。
- `ir/validate.py:validate(model, system, strategy) -> warnings[]`：跨 rank 拓扑警告（PP 不整除、Ulysses CP 整除、Ring CP 块大小、跨节点 EP/TP、VPP 与调度不一致）。

### 6.3 Models（解析模型）
- `models/flops.py`：`OpCost(fwd_flops, dx_flops, dw_flops, fwd/dx/dw_bytes, bound)`；matmul `2mnk`、attn `2bs²hd × compression_ratio`、`dx=2.5×fwd`、内存绑定算子 `bytes_bwd≈1.5×fwd`。  
  `total_training_flops = sum(fwd+dx+dw) × M`。  
  `recompute_overhead_flops` 按 `RecomputePolicy.per_layer[layer_kind]` 与 `_op_recompute_categories(op)` 交集决定哪些 op 累加。
- `models/memory.py:memory_breakdown`：weights+grads+opt_state（按 ZeRO 缩）+ activations（按 layer kind 用系数 10/14/12 估，TP/CP 分片，PP `in_flight=pp//2`）+ comm_buffers + offload。`_optimizer_state_bytes`：Adam 3×P，Muon 2.1×P。
- `models/comm.py`：α-β 模型，`AG/RS=(N-1)·(α+S/N·β)`、`AR=2·...`、`A2A=...`、`P2P=α+S·β`。`tier_for_group` 根据 `group_size ≤ gpus_per_node` 选 intra/inter。

### 6.4 Compose（流水线调度）
- `compose/stage.py:stage_time` 把每 op cost 转为时间：
  - compute：`flops / (peak * achieved_flops_efficiency)`；
  - memory：`bytes / (bw * achieved_bandwidth_efficiency)`；
  - 加上重计算时间 + 集合通信时间（split half-and-half）+ EP 不均衡因子。
- `compose/pipeline.py`：`StepResult` + 五种 `PipelineComposer`：
  - `OneF1BComposer`：`step = (pp-1)·t_fwd[0] + M·max(t_fwd+t_bwd) + (pp-1)·t_bwd[-1] + dp_exposed`。
  - `Interleaved1F1BComposer`：warmup/cooldown 缩 V 倍。
  - `DualPipeComposer`：bubble = (pp-1)/2 · t_stage_max。
  - `DualPipeVComposer`：bubble = (pp-1)/(2V) · t_stage_max。
  - `ZeroBubbleComposer`：bubble = (pp-1)·max(t_stage − t_w, 0)。
- 同一文件还有顶层 `pipeline_step_time(graph, model, system, strategy)`：业务流 C 用，封装 stage 划分 + composer 调度 + memory + MFU + HFU。
- `compose/__init__.py` 导出 `compute_mfu / compute_hfu`，被 `pipeline_step_time` 用，也被 `transform/analysis/training.py` 复用同一组 Composer 类。

### 6.5 Search
- `search/space.py:SearchSpace`：tp/cp/pp/ep/dp/zero/sched/recompute/vpp 维度组合；`strategies(world_size)` 生成所有合法 Strategy（去重）。包含 `enable_cross_node_tp_pruning / enable_cp_pruning / enable_ep_pruning / cp_seq_len_threshold` 4 个特性开关。
- `search/estimator.py:estimate / grid_search / pareto_frontier`：单点估计 + 网格 + Pareto 前沿（按 `(step_time_ms, peak_hbm)` 字典序确定性构造）。
- `search/report.py:Report`+`report_to_dict/json/summary`：JSON / 可读字符串。

### 6.6 IO
- `io/config_loader.py`：YAML 加载 `model`（字符串引用 `configs/models/*.yaml` 或行内 dict）+ `system`（用 `zrt.hardware.registry` 把 HW spec 映射成 `SystemSpec`）+ `strategy`（含 `RecomputePolicy.per_layer`、`OffloadPolicy`）。`_parse_layers` 接受 `["dense","moe",...]` 或形如 `"[dense]*3+[moe]*58+[mtp]"` 的紧凑写法。
- `io/perf_tables.py`：当前是 Phase 1 简单启发表（FLOPs ≥1e11→0.85 效率，bytes ≥1e8→0.85 带宽利用率）。

### 6.7 Anchor
- `anchor/validate.py:Anchor + validate_anchor`：把 `Report` 与 YAML 锚点比对（`step_time_ms / mfu / total_flops` 容忍度，默认 15%）。`strict_mfu_check=False` 时 MFU 偏差只标 `[CALIBRATION]`，不当失败处理。

### 6.8 Trace
- `trace/exporter.py:export_chrome_trace`：把 `Timeline.scheduled_ops` 转成 chrome://tracing 协议；与 `python/zrt/report/chrome_trace.py` 是两个独立实现（前者只接 Timeline，后者带模型/硬件/PP 元信息）。

---

## 7. TODO / 未完成项总览（精确定位）

### 7.1 显式 `TODO Phase 3` 标注（5 处）
| 文件:行 | 内容 |
|---|---|
| `python/zrt/training/trace/exporter.py:34` | TODO：当 graph-native per-stage timelines 可用后，给 chrome trace 加 stage_id/phase/CP/DP/EP overlap 标签。 |
| `python/zrt/training/compose/pipeline.py:331` | TODO：`pipeline_step_time` 应改为消费 executor 提供的 per-stage timelines（取代现在公式驱动的 stage_time）。 |
| `python/zrt/training/search/space.py:22` | TODO：`enable_cross_node_tp_pruning / enable_cp_pruning / enable_ep_pruning` 三个剪枝规则依赖 CP/EP 实现就绪，目前是保守默认值。 |
| `python/zrt/training/search/estimator.py:123` | TODO：`pareto_frontier` 目前只按 `(step_time, memory)` 简单求 Pareto；没强制 CP/EP/跨节点 TP 等通信成本约束。 |
| `python/zrt/training/anchor/validate.py:6` | TODO：`strict_mfu_check` 在校准完成前默认关闭，MFU 偏差只是 `[CALIBRATION]`。 |

### 7.2 Pass 实现仅为 stub（return graph 不变）
| 文件:行 | Pass | 现状 |
|---|---|---|
| `python/zrt/transform/optim/passes.py:30` | `EPLBPass` | docstring 标 `(stub)`，run 直接 return graph。 |
| `python/zrt/transform/optim/passes.py:52` | `MTPPass`  | docstring 标 `(stub)`，run 直接 return graph。 |

### 7.3 PolicyModel 4 中 3 是 stub
| 文件:行 | 类 | 问题 |
|---|---|---|
| `python/zrt/policy_model/open_box_model.py:11` | `OpenBoxModel.predict` | 函数体仅 `pass`，调用即返回 None。 |
| `python/zrt/policy_model/op_aptimize_model.py:11` | `OperatorOptimizationModel.predict` | 同上。`PolicyType.OPERATOR_OPTIMIZATION` 切换后无效。 |
| `python/zrt/policy_model/micro_architecture_model.py:11` | `SystemDesignModel.predict` | 同上。 |
| `python/zrt/policy_model/op_aptimize_model.py:` | 文件名错误：`op_aptimize_model.py`（应为 `op_optimize_model.py`），属于命名瑕疵。 |

只有 `PriorityModel` 实际可用；`SimulatorHub` 默认即 `PolicyType.PRIORITY`，所以三个 stub 平时不被触发——但一旦上层切策略会拿到 None 触发 `RuntimeError`（`hub.py:75`）。

### 7.4 Simulator backend 中两个 stub
| 文件:行 | 类 | 现状 |
|---|---|---|
| `python/zrt/simulator/backends/lookup.py:6` | `LookupSimulator` | `can_simulate→False`，`simulate→pass`，priority=1。 |
| `python/zrt/simulator/backends/tilesim.py:5` | `TilesimSimulator` | 同上，priority=2。 |

它们被 `BACKEND_MAP` 注册，被每个 PolicyBaseModel 自动加载，但 `can_simulate` 永远 False，所以始终回落到 Roofline——不会出错，但没有真正的 lookup/tile 性能模型。

### 7.5 训练专属 Pass 实现完整但**未注册** + **API 错配**
| 文件 | 类 | 问题 |
|---|---|---|
| `python/zrt/transform/training/recompute.py:30` | `RecomputePass` | 引用 `ctx.training.recompute.per_layer_kind`，而 `TransformContext.training: TrainingConfig` 仅有 `recompute_policy: str`。当前运行起来会 `AttributeError`。且**未在 `build_pipeline()` 注册**——属于半成品。 |
| `python/zrt/transform/training/optimizer.py:10` | `OptimizerPass` | 已实现，但未注册；还有 `# This is a simplified implementation` 注释（`optimizer.py:66`），`_total_params_on_rank` 在没有 profile 时硬编码 70B。 |
| `python/zrt/transform/training/offload.py:10` | `OffloadPass` | 引用 `ctx.training.offload`，同样在 `TransformContext` 上不存在；未注册。 |

> `transform/training/zero_fsdp.py` 是唯一注册到 pipeline 的训练 Pass，正常工作。

### 7.6 训练 IR 的 MoE / MTP 分支占位
- `python/zrt/training/ir/builders.py:221, 230` 显式注释 `# Phase 2: moe_block()` / `# Phase 2: mtp_block()`，当前直接复用 `dense_block`。  
  影响：MoE 模型在 Spec-based 估计里被当成同 hidden 大小的 dense 层处理（FLOPs/memory 计算与真实 MoE 路由不一致）。

### 7.7 训练 IR 的 sharding 仅 TP
- `python/zrt/training/ir/shard.py:40` 注释 `Phase 1: TP only. CP/EP added in Phase 2.`。CP/EP 的 `Collective` 不会从 builder 自动产生（`zrt.training.search` 网格里 `cp_values=[1]` 默认即此原因）。

### 7.8 perf_tables 仍为简易启发
- `python/zrt/training/io/perf_tables.py:3` 注释明确 `Phase 1: simple analytical heuristics. / Phase 4: empirical CSV lookup curves.`。
- 现有四档跳变阈值（`<1e9 / <1e10 / <1e11 / 大`），无 GPU/dtype 区分，可解释为占位。

### 7.9 占位 latency
- `python/zrt/executor/scheduler.py:13, 127` 文档说明：节点既无 `latency_us` 注解又无 `hw_spec` 时，`1µs` 占位（无报错）。

### 7.10 调度器关于 PP 的弱化路径
- `python/zrt/transform/analysis/training.py:393-405`：当 PP>1 但节点没有 `stage_id` 注解时，整图调度后 `fwd /= pp` 平均估计——会丢真实 stage 异质性与 warmup/cooldown 结构，仅作 fallback warning。

### 7.11 dispatch 阶段的反向边裁剪
- `python/zrt/ir/adapter.py:181-186, 326-331`：`producer_idx > consumer_idx` 时跳过该边，注释解释是为了规避 KV 缓存别名导致的"反向引用"。这是设计取舍，不算 bug，但若未来切换捕获逻辑可能误删合法环（OpGraph.topo_sort 只在显式环时报错，所以不会沉默崩溃）。

### 7.12 `compat.py` 中的 stub 函数
- `python/zrt/graph/compat.py:263-272` 定义 `is_flash_attn_greater_or_equal_2_10` stub 用于满足 `from transformers.xxx import yyy` 的导入期检查；FakeTensor 路径下永远不会被调用，因此功能上可视作"无害占位"。

### 7.13 builtins 子包消失
- `python/zrt/training/builtins/` 目录在源码层面只剩 `__pycache__`——CLAUDE.md 中提到的 `builtins/registry.py` 似乎被删除/迁移，仓库 grep 不到任何 `builtins.registry` 引用。属于"文档落后于代码"的痕迹。

---

## 8. 整体数据流总览图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ HF Hub / hf_models                                                           │
│   └── load_model (FakeTensorMode + auto_map fallback + MoE/Indexer patches)  │
│        │                                                                     │
│        ▼                                                                     │
│  ModuleTracker (forward/backward hooks)  ◄──┐                                │
│        │                                    │                                │
│        ▼                                    │                                │
│  RecordingDispatch (TorchDispatchMode)──────┘   per-op records dict          │
│        │                                                                     │
│        ▼                                                                     │
│  records_to_opgraph / FusionEngine + fused_records_to_opgraph                │
│        │                                  ExcelWriter / graph_exporter        │
│        ▼                                  → .xlsx / .json / .onnx            │
│  raw OpGraph + fused OpGraph                                                 │
│        │                                                                     │
│        ▼                                                                     │
│  TransformPipeline.run(graph, ctx)                                           │
│   split → fuse → optim → analyze                                             │
│   写入大量 annotations: tp_split, ep_a2a, cp_split, stage_id, phase,          │
│       flops*, compute_us, memory_us, latency_us, bound, stream_id,           │
│       overlap_type, dp_comm, ...                                             │
│        │                                                                     │
│        ├──► DAGScheduler.schedule(g) ──► Timeline (ScheduledOp ×)            │
│        │                                                                     │
│        ├──► SimulatorHub.simulate_graph ──► dict[node_id, SimResult]         │
│        │       (PolicyModelManager → PriorityModel → Roofline backend)       │
│        │                                                                     │
│        └──► metadata.pipeline_metrics / memory_breakdown / training_flops    │
│                                                                              │
│        ▼                                                                     │
│  build_summary / build_training_summary ──► E2ESummary / TrainingSummary     │
│  export_html_report / export_chrome_trace / append_perf_summary              │
└──────────────────────────────────────────────────────────────────────────────┘

      Spec-based (training only):
          YAML → load_specs → estimate → build_graph → op_cost → stage_time
          → pipeline_step_time → Report
```

---

## 9. 信息来源（已读代码文件）

入口与 CLI：
`python/zrt/__main__.py`、`python/zrt/cli.py`

graph（捕获）：
`graph/main.py`、`graph/model_loader.py`、`graph/dispatch.py`、`graph/tracker.py`、`graph/fusion.py`、`graph/fusion_rules.py`（顶部）、`graph/graph_builder.py`（顶部）、`graph/graph_exporter.py`（顶部）、`graph/excel_writer.py`（顶部）、`graph/patches.py`（顶部）、`graph/classifier.py`（顶部）、`graph/__init__.py`、`graph/compat.py`（grep）

ir：
`ir/__init__.py`、`ir/graph.py`、`ir/node.py`、`ir/edge.py`、`ir/types.py`、`ir/hierarchy.py`、`ir/adapter.py`、`ir/param_count.py`

transform：
`transform/__init__.py`、`transform/base.py`、`transform/context.py`、`transform/pipeline.py`、`transform/exporter.py`（顶部）、
`transform/parallel/__init__.py`、`transform/parallel/tensor_parallel.py`、`transform/parallel/expert_parallel.py`、`transform/parallel/pipeline_parallel.py`、`transform/parallel/data_parallel.py`、`transform/parallel/comm_inserter.py`、`transform/parallel/context_parallel.py`、
`transform/fusion/pass_.py`、`transform/optim/passes.py`、
`transform/analysis/__init__.py`、`transform/analysis/passes.py`、`transform/analysis/comm_latency.py`、`transform/analysis/training.py`、`transform/analysis/modeller.py`、
`transform/training/zero_fsdp.py`、`transform/training/recompute.py`、`transform/training/optimizer.py`、`transform/training/offload.py`

executor：
`executor/__init__.py`、`executor/scheduler.py`、`executor/overlap.py`、`executor/stream.py`

simulator：
`simulator/__init__.py`、`simulator/result.py`、`simulator/base.py`、`simulator/cache.py`、`simulator/hub.py`、
`simulator/backends/__init__.py`、`simulator/backends/backend_register.py`、`simulator/backends/roofline.py`（顶部）、`simulator/backends/lookup.py`、`simulator/backends/tilesim.py`

policy_model：
`policy_model/__init__.py`、`policy_model/policy_register.py`、`policy_model/policy_base_model.py`、`policy_model/policy_model_manager.py`、`policy_model/priority_model.py`、`policy_model/open_box_model.py`、`policy_model/op_aptimize_model.py`、`policy_model/micro_architecture_model.py`

hardware：
`hardware/spec.py`、`hardware/registry.py`

memory：
`memory/__init__.py`、`memory/model.py`、`memory/budget.py`、`memory/activation.py`

layers：
`layers/__init__.py`、`layers/op_base.py`

report：
`report/__init__.py`、`report/summary.py`（顶部）、`report/chrome_trace.py`（顶部）、`report/compare.py`（顶部）

training：
`training/__init__.py`、
`training/spec/__init__.py`、`training/spec/model.py`、`training/spec/strategy.py`、`training/spec/system.py`、`training/spec/dtype.py`、
`training/ir/__init__.py`、`training/ir/graph.py`、`training/ir/builders.py`、`training/ir/shard.py`、`training/ir/validate.py`、
`training/models/__init__.py`、`training/models/flops.py`、`training/models/memory.py`、`training/models/comm.py`、
`training/compose/__init__.py`、`training/compose/stage.py`、`training/compose/pipeline.py`、
`training/search/__init__.py`、`training/search/space.py`、`training/search/estimator.py`、`training/search/report.py`、
`training/anchor/__init__.py`、`training/anchor/validate.py`、
`training/trace/exporter.py`、
`training/io/config_loader.py`、`training/io/perf_tables.py`

辅助 grep：
`grep TODO|FIXME|XXX|HACK|stub|simplified|Phase` 全仓 + `grep RecomputePass|OptimizerPass|OffloadPass|per_layer_kind`。
