# 算子融合模块设计文档

> 目录：`python/zrt/transform/fusion/`
>
> 跟踪命令：`python -m python.zrt --model-id hf_models/deepseek_v4 --layers 4 --train --hw nvidia_h100_sxm --tp 8 --pp 2`
>
> 适用版本：fusion v2 rich rules（详见 `docs/fusion_v2_rich_rules_zh.md` §1–§6）

---

## 1. 总览与设计目标

### 1.1 模块定位

算子融合模块在 ZRT-Sim 的四阶段流水线中作为 **Transform Pipeline** 的一环：

```
Graph Capture → Transform Pipeline → DAGScheduler → Report Generator
                       ↑
              FusionPass 在这里
```

输入：一张 `OpGraph`，其中每个 `OpNode` 是一条 aten 算子记录（含 `inputs/outputs` 的 `TensorMeta`、`scope`、`module_class`、`call_id`、`category` 等元数据）。

输出：一张新的 `OpGraph`，把若干 aten 算子折叠成单个"语义级"融合算子（例如把 6 个 aten 算子折叠成一个 `rms_norm`），并附带 `sem_io / sem_shape / sem_flops / sem_memory_bytes` 等下游 simulator 直接使用的注解。

### 1.2 设计目标

| 目标 | 实现机制 |
|---|---|
| **声明式规则** | 所有融合规则写在 YAML 里（`rules/*.yaml`），不再用 Python 注册 |
| **多模型可扩展** | 一份 `_common.yaml`（跨模型）+ 一份 `<model_slug>.yaml`（模型专属），按 `--model-id` 自动选择 |
| **多阶段 fixed-point** | 融合最多迭代 `MAX_PASSES=5` 轮，让"小融合先做、大融合后做"（如 `rms_coef` → `hc_pre_attn`） |
| **三类匹配策略** | `class_only`（按模块类匹配整个 forward 调用）、`ordered_regex`（有序正则）、`dag_signature`（无序多重集） |
| **完整 IO 推导** | 用规则中声明的 `io_roles` 从 `child_ops` 直接读 `TensorMeta`，placeholder（权重 / `input_ids` / RMSNorm γ）天然能取到 |
| **跨阶段安全** | `pass N` 的融合节点参与 `pass N+1` 时，`source_op_ids` 与 `fused_from` 自动 flatten 回原始 aten id，避免 `replace_subgraph` 完整性检查失败 |
| **失败容错** | 任何规则评估失败（shape 公式 / FLOPs 公式 / IO 索引越界）只 `logger.warning`，绝不抛异常中断 fusion |

### 1.3 高层数据流

```
                       ┌─────────────────────────────────┐
                       │   FusionPass.run(graph, ctx)    │
                       └──────────────┬──────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
   initialize_rules()      resolve_fusion_config()    iter_active_rules()
   (loading/                 (loading/                 (registry/)
    rule_set_initializer.py)  fusion_config.py)
        ↓                          ↓                          ↓
   ─── 注册阶段 ──        ─── 配置阶段 ──         ─── 激活过滤 ──
   _common.yaml +         configs/<slug>.yaml      enabled/disabled
   <slug>.yaml            (training/inference)     default_phases
   写入 RuleRegistry
                                      │
                                      ▼
                       MultiPassFuser.fuse(graph, ctx)
                              (pipeline/fuser.py)
                                      │
              ┌───── for pass in range(MAX_PASSES) ─────┐
              │                                          │
              ▼                                          │
       bucket_into_groups                                │
       (bucketing/call_id_bucketer.py)                   │
              │                                          │
              ▼                                          │
       for group in groups:                              │
           _fuse_group(group)                            │
              ├── whole-bucket match  → build_fused_node │
              ├── sliding-window scan → build_fused_node │
              └── structural collapse (默认关闭)         │
              │                                          │
              ▼                                          │
       node_count 不再下降 → break                       │
                                      │
                                      ▼
                       _compose_add_norm(graph)
                       (pipeline/compositors.py)
                                      │
                                      ▼
                              transformed graph
```

---

## 2. 目录结构

```
python/zrt/transform/fusion/
├── __init__.py              公开 API：FusionPass / ModuleFusionRule / IORole ...
├── api.py                   FusionPass GraphPass 入口
│
├── core/                    数据类（无副作用）
│   ├── rule.py              ModuleFusionRule 数据类
│   ├── io_role.py           IORole / ShapeDerivation / IOSpec 别名
│   └── pattern.py           MatchPattern / MatchKind / DEFAULT_SKIP_OPS
│
├── loading/                 配置 / 规则加载
│   ├── rule_set_initializer.py   initialize_rules() — 注册 YAML 规则
│   ├── yaml_rule_loader.py       load_yaml_rules() — YAML → ModuleFusionRule
│   ├── fusion_config.py          resolve_fusion_config() — 激活配置
│   └── op_name_resolver.py       短名 → aten 全名（"mm" → "aten.mm.default"）
│
├── registry/                注册表
│   ├── __init__.py          模块级 forwarder + 进程级单例
│   └── rule_registry.py     RuleRegistry 类
│
├── bucketing/               输入图分桶
│   └── call_id_bucketer.py  FusionGroup + bucket_into_groups()
│
├── matching/                规则匹配
│   ├── matcher.py           best_rule / match_group / _match_*
│   └── sliding_window.py    SlidingWindowScanner（partial-match 兜底）
│
├── building/                融合节点构造
│   ├── node_builder.py      build_fused_node / _build_collapsed_node
│   ├── io_resolver.py       resolve_io_tensors / _child_ops_external_io / resolve_io
│   └── annotation_propagator.py   _propagated_annotations
│
├── semantics/               规则语义评估
│   ├── annotator.py         resolve_io_views / derive_shape_axes / compute_flops / compute_memory
│   └── safe_eval.py         沙箱 AST evaluator
│
├── pipeline/                融合流水线
│   ├── fuser.py             fuse() / MultiPassFuser / _fuse_group / _apply_partial_matches
│   └── compositors.py       _compose_add_norm（后置 Add+Norm 合成）
│
├── rules/                   YAML 规则
│   ├── _common.yaml         跨模型规则（rms_norm / dropout / cross_entropy / add_norm ...）
│   └── deepseek_v4.yaml     DSV4 专属规则（17 条）
│
└── configs/                 激活配置
    ├── deepseek_v4.yaml     DSV4 专属 enabled/disabled 列表
    ├── training_default.yaml
    └── inference_default.yaml
```

---

## 3. 端到端执行追踪

下面按 `python -m python.zrt --model-id hf_models/deepseek_v4 --layers 4 --train --hw nvidia_h100_sxm --tp 8 --pp 2` 执行顺序，从 FusionPass 入口逐步展开。

### Step 1 — `FusionPass.run`：模块入口

[`api.py`](../python/zrt/transform/fusion/api.py)

```python
class FusionPass(GraphPass):
    name = "fusion"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        from python.zrt.transform.fusion.loading import initialize_rules
        from python.zrt.transform.fusion.loading.fusion_config import resolve_fusion_config
        from python.zrt.transform.fusion.pipeline.fuser import MultiPassFuser
        from python.zrt.transform.fusion.registry import default_registry

        initialize_rules(getattr(ctx, "model_id", "") or "")

        existing = getattr(ctx, "fusion", None)
        is_default = (
            existing is None
            or (existing.enabled_rules is None and not existing.disabled_rules
                and not existing.allow_structural_collapse
                and not existing.merge_sibling_classes)
        )
        if is_default:
            phase = ctx.phase_for_fusion() if ctx is not None else "inference"
            ctx.fusion = resolve_fusion_config(
                getattr(ctx, "model_id", "") or "", phase, explicit_path=None,
            )

        fuser = MultiPassFuser(registry=default_registry())
        return fuser.fuse(graph, ctx)
```

关键点：
- `ctx.model_id == "hf_models/deepseek_v4"`，`phase = "training"`（因为命令行加了 `--train`）。
- 显式 `ctx.fusion` 没设 → 用 `resolve_fusion_config` 自动从 `configs/` 选配置。
- 真正的工作在 `MultiPassFuser.fuse(graph, ctx)` 里。

### Step 2 — `initialize_rules`：把 YAML 灌进 RuleRegistry

[`loading/rule_set_initializer.py`](../python/zrt/transform/fusion/loading/rule_set_initializer.py)

```python
def initialize_rules(model_id: str = "") -> None:
    clear_rules()

    common_path = _RULES_DIR / "_common.yaml"
    if common_path.exists():
        for rule in load_yaml_rules(common_path):
            try:
                register_rule(rule)
            except ValueError as exc:
                logger.warning("Skipping common rule: %s", exc)

    if not model_id:
        return

    slug = _model_id_to_key(model_id)             # "deepseek_v4"
    model_path = _RULES_DIR / f"{slug}.yaml"
    if model_path.exists():
        for rule in load_yaml_rules(model_path):
            register_rule(rule)
        return

    # prefix fallback：deepseek_v4_lite → deepseek_v4.yaml
    prefix = slug.split("_")[0].replace("-", "")
    for path in sorted(_RULES_DIR.glob("*.yaml")):
        ...
```

执行结果（DSV4 + training）：
1. `_common.yaml` → 7 条跨模型规则（rms_norm、rms_norm_nn、rms_norm_inline、rms_coef、cross_entropy、dropout、add_norm）。
2. `deepseek_v4.yaml` → 17 条 DSV4 专属规则（dsv4_rms_norm、parallel_embedding、linear、column_parallel_linear、row_parallel_linear、rotary_emb、kv_compressor、sparse_indexer、mla_sparse_attn、moe_gate、moe_expert_swiglu、moe_dispatch、hc_pre_attn_raw、hc_pre_attn、hc_post_attn、hc_head、sparse_attention_kernel）。

注册时 `RuleRegistry` 维护三个索引：[`registry/rule_registry.py:46`](../python/zrt/transform/fusion/registry/rule_registry.py)

```python
def register(self, rule: ModuleFusionRule) -> None:
    name = rule.name or rule.op_type or ""
    if name in self._by_name and self._by_name[name] is not rule:
        raise ValueError(f"Duplicate fusion rule name {name!r} ...")
    self._all_rules.append(rule)        # 注册顺序（matchers 按此遍历）
    self._by_name[name] = rule          # name → rule（唯一性）
    for key in _index_keys(rule):
        self._class_index.setdefault(key, []).append(rule)   # class → rules
```

### Step 3 — `resolve_fusion_config`：选配置

[`loading/fusion_config.py:80`](../python/zrt/transform/fusion/loading/fusion_config.py)

```python
def resolve_fusion_config(model_id, phase, explicit_path=None):
    slug = _model_id_to_key(model_id) if model_id else ""
    candidates: list[Path] = []
    if slug:
        candidates.append(_FUSION_CONFIGS_DIR / f"{slug}_{phase}.yaml")
        candidates.append(_FUSION_CONFIGS_DIR / f"{slug}.yaml")
    candidates.append(_FUSION_CONFIGS_DIR / f"{phase}_default.yaml")

    for path in candidates:
        cfg = load_fusion_config_file(path, phase)
        if cfg is not None:
            return cfg
    return FusionConfig()
```

查找顺序：
1. `configs/deepseek_v4_training.yaml`（不存在）
2. `configs/deepseek_v4.yaml`（命中）
3. `configs/training_default.yaml`（兜底）

命中的 [`configs/deepseek_v4.yaml`](../python/zrt/transform/fusion/configs/deepseek_v4.yaml) 包含两个 phase 段：

```yaml
training:
  enabled_rules: null         # null → "所有 default_phases 含 training 的规则"
  disabled_rules:
    - kv_compressor           # KV 缓存写入会破坏 autograd 链
    - sparse_indexer
  allow_structural_collapse: false
  merge_sibling_classes: []

inference:
  enabled_rules: null
  disabled_rules: []
  allow_structural_collapse: false
  merge_sibling_classes: []
```

解析结果是一个 `FusionConfig` 实例，挂到 `ctx.fusion`。

### Step 4 — `iter_active_rules`：phase + 配置 → 激活列表

[`registry/rule_registry.py:123`](../python/zrt/transform/fusion/registry/rule_registry.py)

```python
def iter_active(self, fusion_cfg, phase: str) -> list[ModuleFusionRule]:
    enabled = fusion_cfg.enabled_rules
    disabled = set(fusion_cfg.disabled_rules)

    if enabled is None:
        candidates = [r for r in self._all_rules if phase in r.default_phases]
    else:
        candidates = [self._by_name[n] for n in enabled]

    return [r for r in candidates if r.name not in disabled]
```

DSV4 training 的结果：24 - 2（kv_compressor + sparse_indexer）= 22 条激活规则。

### Step 5 — `fuse`：多 pass 主循环

[`pipeline/fuser.py:49`](../python/zrt/transform/fusion/pipeline/fuser.py)

```python
def fuse(graph: "OpGraph", ctx=None) -> "OpGraph":
    fusion_cfg = getattr(ctx, "fusion", None) or FusionConfig()
    phase = ctx.phase_for_fusion() if ctx is not None else "inference"
    active = iter_active_rules(fusion_cfg, phase)
    active_names = {r.name for r in active}
    scanner = SlidingWindowScanner(active)

    working_graph = graph.clone()
    fuse_idx = 0
    for _pass_n in range(MAX_PASSES):                    # MAX_PASSES = 5
        previous_count = len(working_graph.nodes)
        groups = bucket_into_groups(
            working_graph,
            merge_sibling_classes=set(fusion_cfg.merge_sibling_classes),
        )
        for group in groups:
            fuse_idx = _fuse_group(
                working_graph, group, active, fusion_cfg, scanner, fuse_idx,
            )
        if len(working_graph.nodes) >= previous_count:
            break                                         # fixed point

    if _ADD_NORM_RULE_NAME in active_names:
        working_graph = _compose_add_norm(working_graph)
    return working_graph
```

**为什么要 fixed-point？** 部分规则需要"先融合再融合"才能匹配。例如 `hc_pre_attn`（8 算子）的 op_regex 中第一个就是 `rms_coef`——这是 pass-1 才会出现的融合算子，pass-2 才能匹配 `hc_pre_attn`。

#### Step 5a — `bucket_into_groups`：按 call_id 分桶

[`bucketing/call_id_bucketer.py:28`](../python/zrt/transform/fusion/bucketing/call_id_bucketer.py)

```python
def bucket_nodes_by_leaf_module(graph: "OpGraph") -> list[FusionGroup]:
    groups: list[FusionGroup] = []
    current: Optional[FusionGroup] = None
    last_key: Any = None

    for node in graph.topo_sort():
        if node.category == "communication" or not node.scope:
            # 通信 / 无 scope 节点立刻独立成组
            groups.append(FusionGroup(scope=node.scope, ..., child_ops=[node], ...))
            last_key = None
            continue

        node_call_id = getattr(node, "call_id", 0) or 0
        if node_call_id > 0:
            key = ("call", node_call_id)
        else:
            key = ("legacy", node.scope, node.module_class, node.layer)

        if current is not None and key == last_key:
            current.child_ops.append(node)
            continue

        current = FusionGroup(scope=..., child_ops=[node], call_id=node_call_id, ...)
        groups.append(current)
        last_key = key

    # 标记 is_full_forward：当桶内 ops 数等于该 call_id 的全部 ops 数
    if any(g.call_id > 0 for g in groups):
        call_id_total: dict[int, int] = {}
        for n in graph.nodes.values():
            cid = getattr(n, "call_id", 0) or 0
            if cid:
                call_id_total[cid] = call_id_total.get(cid, 0) + 1
        for g in groups:
            if g.call_id > 0 and len(g.child_ops) == call_id_total.get(g.call_id, -1):
                g.is_full_forward = True
    return groups
```

为什么不直接按 `scope` 分？因为同一个 module 可能在 forward 里被调用多次（例如同一个 `RMSNorm` 实例在 attn_norm 和 ffn_norm 处被调用），按 `scope` 会把不同调用粘成一组。`call_id` 在 `ModuleTracker` 给每次 forward 调用分配唯一编号，确保**桶 = 一次 forward 调用**。

`is_full_forward` 字段对 `class_only` 规则至关重要：这类规则假定桶涵盖整个 forward 调用，碎片（例如 Attention.forward 里嵌入的 inline RMSNorm 子序列）不能 fire `class_only` 规则，否则会被错标成父类的语义 op_type。

#### Step 5b — `_fuse_group`：单桶分派

[`pipeline/fuser.py:97`](../python/zrt/transform/fusion/pipeline/fuser.py)

```python
def _fuse_group(graph, group, active, fusion_cfg, scanner, fuse_idx):
    # ── 单节点桶 ──
    if len(group.child_ops) <= 1:
        node = group.child_ops[0]
        if node.category == "communication" or not node.module_class:
            return fuse_idx
        rule = lookup_rule((node.op_type,), module_class=node.module_class, active_rules=active)
        if rule is None:
            return fuse_idx
        if rule.pattern is None or rule.pattern.kind != "class_only":
            return fuse_idx
        if not group.is_full_forward:
            return fuse_idx
        fused = build_fused_node(group, rule, graph, fuse_idx)
        _propagate_call_id_and_provenance(fused, group)
        fuse_idx += 1
        graph.replace_subgraph({node.id}, fused)
        return fuse_idx

    # ── 多节点桶 ──
    if not group.scope or not group.module_class:
        return fuse_idx
    operator_sequence = tuple(op.op_type for op in group.child_ops)
    rule = lookup_rule(operator_sequence, module_class=group.module_class, active_rules=active)

    # class_only 不能 fire 在 forward 调用的"片段"上
    if (rule is not None and rule.pattern is not None
            and rule.pattern.kind == "class_only"
            and not group.is_full_forward):
        rule = None

    if rule is not None:
        group_ids = {op.id for op in group.child_ops}
        replacement = build_fused_node(group, rule, graph, fuse_idx)
        _propagate_call_id_and_provenance(replacement, group)
        fuse_idx += 1
        graph.replace_subgraph(group_ids, replacement)
        return fuse_idx

    # ── 兜底：滑动窗口 partial match ──
    return _apply_partial_matches(graph, group, scanner, fusion_cfg, fuse_idx)
```

三层 fallback：
1. 单节点 + `class_only` 规则（整 forward 命中）
2. 多节点的整桶匹配（`best_rule`）
3. 滑动窗口找连续子序列

#### Step 5b-i — 整桶匹配：`best_rule` → `match_group`

[`matching/matcher.py:137`](../python/zrt/transform/fusion/matching/matcher.py)

```python
def best_rule(operator_types, module_class, rules) -> Optional[ModuleFusionRule]:
    best = None
    for idx, rule in enumerate(rules):
        pattern = rule.pattern
        if pattern is None:
            continue
        if not match_group(operator_types, module_class,
                           pattern=pattern, target_class=rule.target_class):
            continue
        if best is None or rule.priority > best.priority:
            best = rule
    return best
```

`match_group` 是三种模式的分派：

```python
def match_group(operator_types, module_class, *, pattern, target_class) -> bool:
    kind = pattern.kind
    if kind == "class_only":
        if not _class_matches(module_class, target_class):
            return False
        return _match_class_only(operator_types, pattern)
    if kind == "ordered_regex":
        return _match_ordered_regex(operator_types, pattern)
    if kind == "dag_signature":
        return _match_dag_signature(operator_types, pattern)
```

三种匹配的语义：

| kind | 语义 | 典型规则 |
|---|---|---|
| `class_only` | 只校验 `target_class` + 算子数量在 `[min_ops, max_ops]` | `parallel_embedding`、`mla_sparse_attn`、`moe_dispatch` |
| `ordered_regex` | 按顺序正则匹配，允许 `skip_ops`（reshape/transpose 等无 cost op）穿插 | `rms_norm`、`rms_coef`、`hc_pre_attn_raw` |
| `dag_signature` | 无序多重集 + min count（不在意算子顺序，只在意每个算子至少出现 N 次） | `linear`（容忍 FP8/FP4 量化导致的 mm 前后多种 abs/amax/div 排列） |

`_match_ordered_regex` 实现里关键的 `skip_ops` 处理：[`matching/matcher.py:_match_ordered_regex`](../python/zrt/transform/fusion/matching/matcher.py)：碰到 `aten.view.default` 等 skip op 时 cursor 前进但 regex 索引不变。

`DEFAULT_SKIP_OPS` 列表（[`core/pattern.py:13`](../python/zrt/transform/fusion/core/pattern.py)）包含 16 个纯 reshape/cast 算子，例如 `aten.view.default`、`aten.transpose.int`、`aten._to_copy.default`、`aten.contiguous.memory_format`。

#### Step 5b-ii — 滑动窗口兜底

[`pipeline/fuser.py:159`](../python/zrt/transform/fusion/pipeline/fuser.py)

```python
def _apply_partial_matches(graph, group, scanner, fusion_cfg, fuse_idx):
    operator_types = [op.op_type for op in group.child_ops]
    matches = scanner.scan(operator_types)
    if not matches:
        if fusion_cfg.allow_structural_collapse:
            group_ids = {op.id for op in group.child_ops}
            replacement = _build_collapsed_node(group, graph, fuse_idx)
            graph.replace_subgraph(group_ids, replacement)
            fuse_idx += 1
        return fuse_idx

    for rule, start, end in matches:
        sub_ops = group.child_ops[start:end]
        sub_group = FusionGroup(scope=sub_ops[0].scope, ..., child_ops=list(sub_ops),
                                call_id=group.call_id, is_full_forward=False)
        replacement = build_fused_node(sub_group, rule, graph, fuse_idx)
        _propagate_call_id_and_provenance(replacement, sub_group)
        fuse_idx += 1
        graph.replace_subgraph({op.id for op in sub_ops}, replacement)
    return fuse_idx
```

`SlidingWindowScanner` 的核心逻辑（[`matching/sliding_window.py:73`](../python/zrt/transform/fusion/matching/sliding_window.py)）：

```python
def scan(self, operator_types) -> list[tuple[ModuleFusionRule, int, int]]:
    matches = []
    position = 0
    n = len(operator_types)
    while position < n:
        best = None        # (rule, end)
        for rule in self._rules:                          # 按 priority desc 排序
            end = try_match_at(operator_types, position, rule.pattern)
            if end is None: continue
            if best is None: best = (rule, end)
            elif rule.priority > best[0].priority: best = (rule, end)
            elif rule.priority == best[0].priority and end > best[1]: best = (rule, end)
        if best is None:
            position += 1                                  # 无匹配 → 跳一格
            continue
        matches.append((best[0], position, best[1]))
        position = best[1]                                 # 跳过整段
    return matches
```

策略：贪心、从左到右、相同优先级取更长。只接受 `ordered_regex` 规则。

#### Step 5b-iii — `structural collapse`：legacy 兜底（默认关闭）

只有 `fusion_cfg.allow_structural_collapse = True` 才会走，DSV4 训练配置里关掉了。功能：当一个多算子桶完全没规则匹配时，把整桶坍缩成一个以 `module_class` 命名的"结构融合"节点。`_build_collapsed_node`（[`building/node_builder.py:98`](../python/zrt/transform/fusion/building/node_builder.py)）使用旧的 edge-based `_external_io` 推 IO（已知会漏 placeholder，文档化局限性）。

### Step 5c — `build_fused_node`：构造融合节点

这是融合的"实际产物"环节。[`building/node_builder.py:22`](../python/zrt/transform/fusion/building/node_builder.py)

```python
def build_fused_node(group, rule, graph, fuse_idx) -> "OpNode":
    from python.zrt.ir.node import OpNode
    from python.zrt.transform.fusion.semantics import annotate_fused_node

    first = group.child_ops[0]
    op_type = rule.op_type or (rule.target_class.__name__ if ... else str(rule.target_class))
    fused_from = list(dict.fromkeys(op.op_type for op in group.child_ops))

    # ── 1) IO 推导：rule.io_roles 优先，child_ops fallback 兜底 ──
    if rule.io_roles:
        ext_inputs, ext_outputs = resolve_io_tensors(group.child_ops, rule)
        if not ext_inputs or not ext_outputs:
            fb_in, fb_out = _child_ops_external_io(group.child_ops)
            if not ext_inputs:  ext_inputs = fb_in
            if not ext_outputs: ext_outputs = fb_out
    else:
        ext_inputs, ext_outputs = _child_ops_external_io(group.child_ops)

    # ── 2) Provenance：把 io_roles 解析成 FusedIOPort ──
    provenance_parts = []
    for spec in rule.io_roles:
        port = resolve_io(group.child_ops, spec)
        if port is not None:
            provenance_parts.append(port)

    # ── 3) Annotation 传播：子→父，仅当所有子节点一致 ──
    propagated = _propagated_annotations(group)
    level = "parent" if len(group.child_ops) > 3 else "leaf"

    # ── 4) 构造节点 ──
    node = OpNode(
        id=f"fused_{fuse_idx}_{first.id}",
        op_type=op_type,
        inputs=ext_inputs, outputs=ext_outputs,
        scope=group.scope, category=first.category,
        module_class=group.module_class, layer=first.layer,
        component=first.component, fused_from=fused_from,
        num_sub_ops=len(group.child_ops), fusion_level=level,
        name=group.leaf_attr, provenance=tuple(provenance_parts),
    )
    node.annotations.update(propagated)
    node.annotations.update(rule.annotations)
    node.annotations["source_op_ids"] = [op.id for op in group.child_ops]
    node.annotations["fused_by_rule"] = rule.name

    # ── 5) 语义注解（FLOPs / memory / sem_shape ...）──
    annotate_fused_node(node, group.child_ops, rule)
    return node
```

#### 5c-i — IO 解析的两条路径

[`building/io_resolver.py`](../python/zrt/transform/fusion/building/io_resolver.py) 提供三个 helper：

**`resolve_io_tensors(child_ops, rule)`** — 规则声明式，主路径

```python
def resolve_io_tensors(child_ops, rule) -> tuple[list, list]:
    inputs, outputs = [], []
    seen_in, seen_out = set(), set()

    for spec in rule.io_roles:
        op_idx = _normalize_idx(spec.source_op_index, len(child_ops))
        if op_idx < 0 or op_idx >= len(child_ops):
            logger.warning("fusion.io: role %r source_op_index %d out of range ...")
            continue
        op = child_ops[op_idx]
        tensors = op.inputs if spec.source_kind == "input" else op.outputs
        arg_idx = _normalize_idx(spec.source_arg_index, len(tensors))
        if arg_idx < 0 or arg_idx >= len(tensors):
            logger.warning("fusion.io: role %r source_arg_index %d out of range ...")
            continue
        meta = tensors[arg_idx]
        if spec.source_kind == "input":
            if meta.id in seen_in: continue
            seen_in.add(meta.id); inputs.append(meta)
        else:
            if meta.id in seen_out: continue
            seen_out.add(meta.id); outputs.append(meta)
    return inputs, outputs
```

**`_child_ops_external_io(child_ops)`** — 兜底路径（无 io_roles 或部分缺失时）

```python
def _child_ops_external_io(child_ops) -> tuple[list, list]:
    internal_produced = {t.id for op in child_ops for t in op.outputs}
    internal_consumed = {t.id for op in child_ops for t in op.inputs}

    inputs, outputs = [], []
    seen_in, seen_out = set(), set()
    for op in child_ops:
        for t in op.inputs:
            if t.id in internal_produced or t.id in seen_in: continue
            seen_in.add(t.id); inputs.append(t)        # 外部输入：不是组内任何子节点产的
        for t in op.outputs:
            if t.id in internal_consumed or t.id in seen_out: continue
            seen_out.add(t.id); outputs.append(t)      # 外部输出：未被组内消费

    if not outputs:                                     # 终端产物 fallback
        for t in child_ops[-1].outputs:
            if t.id in seen_out: continue
            seen_out.add(t.id); outputs.append(t)
    return inputs, outputs
```

**为什么不用图边？** `_external_io(graph, group_ids)`（edge-based）会漏掉 placeholder 张量（weights、`input_ids`、RMSNorm γ），它们没有 producer `OpNode`，因此不存在 Edge 记录。`_child_ops_external_io` 通过 `tensor.id` 直接对子节点 `.inputs/.outputs` 做集合差，placeholder 天然能取到。

**`resolve_io(child_ops, spec)`** — 给 provenance 用

```python
def resolve_io(child_ops, spec) -> Optional[FusedIOPort]:
    op_idx = spec.source_op_index if spec.source_op_index >= 0 else len(child_ops) + spec.source_op_index
    if op_idx < 0 or op_idx >= len(child_ops): return None
    op = child_ops[op_idx]
    tensors = op.inputs if spec.source_kind == "input" else op.outputs
    arg_idx = spec.source_arg_index if spec.source_arg_index >= 0 else len(tensors) + spec.source_arg_index
    if arg_idx < 0 or arg_idx >= len(tensors): return None
    return FusedIOPort(
        role=spec.role, origin_node_id=op.id, origin_op_type=op.op_type,
        origin_arg_index=arg_idx, origin_kind=spec.source_kind,
    )
```

返回的 `FusedIOPort` 写入 `node.provenance`，记录"这个角色对应的张量来自哪个 child op 的哪个槽"，下游 simulator 可以拿来追溯。

#### 5c-ii — Annotation 传播

[`building/annotation_propagator.py:17`](../python/zrt/transform/fusion/building/annotation_propagator.py)

```python
_SCALAR_PROPAGATE_KEYS = ("stage_id", "phase", "ep_experts_local", "ep_a2a_inserted", "recompute")
_DICT_PROPAGATE_KEYS   = ("tp_split", "ep_needs_a2a", "cp_split")

def _propagated_annotations(group: "FusionGroup") -> dict:
    propagated = {}
    for key in _SCALAR_PROPAGATE_KEYS:
        vals = {op.annotations.get(key) for op in group.child_ops if key in op.annotations}
        if len(vals) == 1:           # 所有子节点 agree
            propagated[key] = vals.pop()
    for key in _DICT_PROPAGATE_KEYS:
        seen = [op.annotations[key] for op in group.child_ops if key in op.annotations]
        if seen and all(d == seen[0] for d in seen):
            propagated[key] = seen[0]
    return propagated
```

规则：**只有所有子节点都 agree 的 annotation 才提升到父节点**。这是为了避免把"上游 pass 给某个子节点打的标签"错误地泛化到整组。例如 `tp_split` 是 TP pass 给具体 mm 节点打的，融合节点继承它必须确认所有 mm 子节点都同意；否则丢弃。

#### 5c-iii — 语义注解

[`semantics/annotator.py:324`](../python/zrt/transform/fusion/semantics/annotator.py)

```python
def annotate_fused_node(fused_node, child_ops, rule) -> None:
    try:
        io_views = resolve_io_views(child_ops, rule)          # role → TensorView
        shape_axes = derive_shape_axes(io_views, rule)        # 公式求值得 batch_size / seq_len / ...
        namespace = {**io_views, **shape_axes}
        flops = compute_flops(rule, namespace, child_ops)
        memory_bytes = compute_memory(rule, namespace, child_ops)

        ann = fused_node.annotations
        ann["sem_io"] = {role: _io_view_to_dict(v) for role, v in io_views.items()}
        ann["sem_shape"] = dict(shape_axes)
        ann["sem_flops"] = flops
        ann["sem_memory_bytes"] = memory_bytes
        main_view = io_views.get("activation") or io_views.get("output")
        if main_view is not None:
            ann["sem_dtype"] = main_view.dtype

        for axis in _FLATTEN_AXES:                            # 扁平化 batch_size 等到顶层
            if axis in shape_axes and axis not in ann:
                ann[axis] = shape_axes[axis]
    except Exception as e:
        logger.warning("fusion.semantics: annotate_fused_node failed: %s", e)
```

四阶段：
1. **`resolve_io_views`** — 把 `rule.io_roles` 解析成 `{role: TensorView}` 字典；`TensorView` 是只读的 `{shape, dtype, bytes, numel, itemsize}` 快照。
2. **`derive_shape_axes`** — 在 `io_views` 命名空间下，逐条求值 `rule.shape_derivation.items()` 里的表达式（"earlier axes visible in later ones"），得到 `{batch_size, seq_len, hidden_in, ...}`。
3. **`compute_flops`** — 按优先级：`flops_callable` > `flops_formula` > `flops_kind="from_io"` (2×total_input_numel) > `flops_kind="sum_children"` > None。
4. **`compute_memory`** — 类似优先级链，但没有 `sum_children` 等价物。

公式由 [`semantics/safe_eval.py`](../python/zrt/transform/fusion/semantics/safe_eval.py) 的沙箱 AST evaluator 评估，允许字面量 / 名字 / `x.shape[i]` / 二元 op / `min/max/abs/log/log2/sqrt/ceil/floor` / 三元条件 / tuple-list 字面量。**禁止函数调用之外的任何 callable、import、属性写入**。

### Step 6 — `_compose_add_norm`：后置 Add+Norm 合成

[`pipeline/compositors.py:13`](../python/zrt/transform/fusion/pipeline/compositors.py)

```python
def _compose_add_norm(graph: "OpGraph") -> "OpGraph":
    topo = graph.topo_sort()
    norm_types = {"rms_norm", "layer_norm", "RMSNorm", "LayerNorm"}
    add_types = {"add", "residual_add"}

    pairs = []
    for i in range(len(topo) - 1):
        a, b = topo[i], topo[i + 1]
        if ((a.op_type.lower() in add_types or a.op_type.lower().endswith("add"))
                and (b.op_type.lower() in norm_types or b.op_type.lower().endswith("norm"))):
            if any(e.src == a.id and e.dst == b.id for e in graph.edges):
                pairs.append((i, i + 1))

    for add_i, norm_i in reversed(pairs):
        add_node, norm_node = topo[add_i], topo[norm_i]
        merged = OpNode(
            id=f"composed_{add_node.id}_{norm_node.id}",
            op_type="AddNorm",
            inputs=add_node.inputs, outputs=norm_node.outputs,
            scope=norm_node.scope, ...,
            num_sub_ops=add_node.num_sub_ops + norm_node.num_sub_ops,
            fusion_level="parent",
        )
        merged.annotations["source_op_ids"] = (
            list(add_node.annotations.get("source_op_ids", [add_node.id]))
            + list(norm_node.annotations.get("source_op_ids", [norm_node.id]))
        )
        merged.annotations["fused_by_rule"] = "add_norm"
        graph.replace_subgraph({add_node.id, norm_node.id}, merged)
    return graph
```

这是融合的"最后一公里"——它**在已融合的图上**进一步合并：相邻的 add+norm 节点对（必须有真实数据边）合并成 `AddNorm`。激活由 `_common.yaml` 里 `add_norm` 规则的存在与否决定（虚拟规则，不走 matcher，只是为了让用户通过 `enabled_rules/disabled_rules` 开关）。

---

## 4. 规则 Schema 速查

每条 YAML 规则对应一个 `ModuleFusionRule` 实例（[`core/rule.py:21`](../python/zrt/transform/fusion/core/rule.py)）。完整字段（节选）：

```yaml
- name: dsv4_rms_norm               # 唯一标识（注册时校验冲突）
  op_type: rms_norm                 # 融合后节点的 op_type
  description: "..."
  default_phases: [inference, training]
  priority: 30                      # best_rule / SlidingWindowScanner 用，大者优先
  target_class: RMSNorm             # str / type / tuple[str, ...]，class_only 必填

  match:                            # → MatchPattern
    kind: ordered_regex             # class_only | ordered_regex | dag_signature
    op_regexes:
      - 'aten\.pow\.Tensor_Scalar'
      - 'aten\.mean\.dim'
      - 'aten\.add\.(Tensor|Scalar)'
      - 'aten\.rsqrt\.default'
      - 'aten\.mul\.Tensor'
      - 'aten\.mul\.Tensor'
    min_ops: 5
    max_ops: 16

  io_roles:                         # → IORole 列表
    - {role: activation, source_kind: input,  source_op_index: 0,  source_arg_index: 0, shape_role: "[B,S,H]"}
    - {role: weight,     source_kind: input,  source_op_index: -1, source_arg_index: 0, shape_role: "[H]"}
    - {role: output,     source_kind: output, source_op_index: -1, source_arg_index: -1, shape_role: "[B,S,H]"}

  shape_derivation:                 # → ShapeDerivation
    batch_size: "activation.shape[0]"
    seq_len:    "activation.shape[1]"
    hidden_in:  "activation.shape[-1]"
    hidden_out: "activation.shape[-1]"

  flops_formula:  "4 * batch_size * seq_len * hidden_in"
  memory_formula: "activation.bytes + weight.bytes + output.bytes"
  annotations:                      # 静态 K-V 注入到 node.annotations
    layer_norm_kind: rms
```

`source_op_index` / `source_arg_index` 支持负数（Python 风格倒数）：`-1` 表示最后一个 child op / 该 op 的最后一个槽。

`shape_role` 仅为文档化用途（让审阅者快速看出"这一槽应该是什么形状"），不参与运行时校验。

`flops_kind` 默认 `"sum_children"`；当指定了 `flops_formula` / `flops_callable` 时会自动升为 `"formula"`（`__post_init__` 里做的）。

---

## 5. 设计要点 & 不变量

### 5.1 IO 推导：声明优先 + 兜底完备

旧版 `_external_io(graph, group_ids)` 用图边推 IO，**漏 placeholder 是致命缺陷**（embedding 的 `input_ids`、所有 linear 的权重、RMSNorm 的 γ 全部丢失）。重构后：

- 有 `io_roles` 的规则：`resolve_io_tensors` 按声明顺序取，**完整、有序**。
- 无 `io_roles` 的规则：`_child_ops_external_io` 通过 `tensor.id` 集差自动推导。
- 规则只声明一面（如只有 input）：另一面自动用 `_child_ops_external_io` 补齐。

修复后 DSV4 训练报表的 `Fused Operators (fwd)` 552 行中，**0 个多算子融合节点空 IO**（剩余 12 行空输入全是 `aten.arange` / `aten.scalar_tensor` 等真无入参的 factory op）。

### 5.2 Class-only 规则的"完整 forward"门控

`class_only` 规则只校验 `target_class` + 算子数，没有顺序约束。如果允许它们 fire 在 forward 的"片段"上（例如 `Attention.forward` 内嵌的 inline RMSNorm 5-op 子序列），会被错误地标成父类的语义 `op_type`，产生错误的 FLOPs/shape。

防护机制：`bucket_into_groups` 给每个桶计算 `is_full_forward`（桶内 ops 数 == 该 call_id 的全部 ops 数），`_fuse_group` 内对 `class_only` 规则强制要求 `group.is_full_forward == True`。

### 5.3 Fixed-point 迭代的 source_op_ids 扁平化

pass-N 的融合节点参与 pass-N+1 时，要保证 `replace_subgraph({op.id for op in sub_ops}, ...)` 调用里的 `op.id` 是当前图上仍存在的节点 id（即融合节点 id），但 `source_op_ids` 始终指向**最原始的 raw aten 节点 id**——否则 Excel 报表里 raw-op-ids 列追溯不回去。

实现：[`pipeline/fuser.py::_propagate_call_id_and_provenance`](../python/zrt/transform/fusion/pipeline/fuser.py)

```python
flat_ids = []
for child in group.child_ops:
    child_src_ids = child.annotations.get("source_op_ids") or []
    if child_src_ids:                          # 子节点已经是融合节点 → 拿它的 source_op_ids
        flat_ids.extend(child_src_ids)
    else:
        flat_ids.append(child.id)              # 子节点是 raw → 用自己的 id
replacement.annotations["source_op_ids"] = flat_ids
replacement.num_sub_ops = len(flat_ids)        # 重新数总 raw 算子数
```

### 5.4 失败容错：永不抛异常

任何规则评估失败（IO 索引越界 / 公式求值失败 / annotate_fused_node 异常）都只走 `logger.warning`：

- `resolve_io_tensors` 越界：打 warning，跳过该 role，继续处理其他 roles。
- `derive_shape_axes` 表达式失败：当条轴跳过，后续轴仍尝试评估。
- `annotate_fused_node` 顶层包了 `try / except`：哪怕完全炸了，融合本身不受影响。

### 5.5 Provenance 与 IO 列表的解耦

`node.inputs / node.outputs` 必须**覆盖完整**（让 Excel 报表能展示所有外部 IO），所以走"声明优先 + 兜底"。

`node.provenance` 是**纯声明**（`FusedIOPort` 记录每个 role 的 origin_node_id + origin_arg_index），让下游能区分"这一槽是 weight 还是 activation"。两者职责分离，互不污染。

---

## 6. 调试技巧

### 6.1 检查实际激活的规则

```python
from python.zrt.transform.fusion.loading.rule_set_initializer import initialize_rules
from python.zrt.transform.fusion.loading.fusion_config import resolve_fusion_config
from python.zrt.transform.fusion.registry import iter_active_rules

initialize_rules("hf_models/deepseek_v4")
cfg = resolve_fusion_config("hf_models/deepseek_v4", "training")
active = iter_active_rules(cfg, "training")
print(sorted(r.name for r in active))
```

### 6.2 查看每个融合节点的"来源"

打开 `output/deepseek_v4/deepseek_v4_training.xlsx` 的 `Fused Operators (fwd)` 表，关注 4 列：
- `Rule Name` — 该节点由哪条 YAML 规则产生
- `Raw Op IDs` — 哪些原始 aten 节点被吸收（点回 `Raw Operators (fwd)` 表对应行）
- `Constituent Aten Ops` — 用 → 连起来的 fused_from 列表
- `Sub-ops` — 总 raw 算子数

### 6.3 验证 `io_roles` 是否取对了张量

跑实际 trace 后看 Excel：`Fused Input Shapes` 列应该和 `shape_role` 一致：
- `rms_norm`: `(1, 128, 7168), (7168,)` — `[B,S,H]` + `[H]`  ✓
- `parallel_embedding`: `(129280, 7168), (1, 128)` — `[V,H]` + `[B,S]`  ✓
- `linear`: `activation, weight` 两槽  ✓

不一致就是 `source_arg_index` 写反了。例如本仓库历史上踩过的坑：DSV4 RMSNorm `(self.weight * x)` 的 `aten.mul.Tensor` 是 `(weight, x)`，weight 在 slot 0；规则却写了 `source_arg_index: 1`，导致 weight 角色拿到的是 activation。

### 6.4 检查 fixed-point 是否提前收敛

`fuse()` 主循环最多 5 轮，每轮看 `len(working_graph.nodes)` 是否下降。如果某条规则该 fire 但没 fire，可能：
- 该规则的依赖（前序融合）没在更早的 pass 出现 → 调高/调低 priority
- 整桶匹配失败但 sliding-window 也没匹配上 → 算子序列有 skip op 没在 `DEFAULT_SKIP_OPS` 里，规则的 op_regex 把它当成必匹配的算子之一

### 6.5 单元测试入口

```bash
# 规则结构 + YAML 解析
pytest tests/transform/fusion/test_dsv4_rules.py -v

# 三种 match kind 的单元测试
pytest tests/transform/fusion/test_match.py -v

# 语义注解（resolve_io_views / derive_shape_axes / compute_flops）
pytest tests/transform/fusion/test_semantics.py -v

# IO 解析（新加的 resolve_io_tensors + _child_ops_external_io）
pytest tests/transform/fusion/test_iorole_resolution.py -v

# 端到端 FusionPass 图契约
pytest tests/test_fusion_pass.py -v

# MFU 回归（确认融合改动不影响下游性能估算）
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -v
```

---

## 7. 相关文档

- `docs/fusion_v2_rich_rules_zh.md` — 规则 YAML 完整 schema 文档（§2 字段定义、§3 三种 match kind、§4 公式语法、§5 YAML 示例、§6 注册流程）
- `docs/fusion_v2_zh.md` — fusion v2（MRO-based）整体设计
- `CLAUDE.md` — 项目级编码约束（fusion 部分见 "Stage 2 — Transform Pipeline / Pass order"）
