# 通信建模重构 Spec（待评审）

- 状态：**草案 v0.1 — 待评审**
- 范围：`estimate`（spec-based 训练估算）通信成本模型
- 主要代码：`python/zrt/training/models/comm.py`、`python/zrt/hardware/spec.py`、`python/zrt/hardware/registry.py`
- 相关路径（口径一致性）：`python/zrt/transform/analysis/comm_latency.py`

---

## 0. 评审指引

本 spec 把前序讨论中暴露的通信建模问题**整体**收拢，目标是评审一次过。阅读顺序建议：§2（问题）→ §5（决策点）→ §9（需你拍板的边界条件）。

- 需求项编号 `R*`、决策点 `D*`、边界条件 `B*`，便于你直接在编号上批注。
- `D*` 决策点均给出选项 / 优劣 / **推荐项（加粗）**。请在每个 `D*` 标注「采纳推荐」或选其它项。
- `B*` 是我无法替你决定、且会显著改变设计的输入（硬件范围、标定数据、回归预算等）。请在评审时填。
- 凡标 **⚠️ 非 no-op** 的改动，都会移动现有 anchor，必须走标定流程，不能伪装成零位移合入。

---

## 1. 现状（精确引用）

通信成本集中在 `comm.py::collective_time`（`comm.py:17-72`），alpha-beta 模型：

```
T = α·(延迟步数) + S·(N-1)/N·β        α = latency_us·1e-6   β = 1/(bandwidth_bytes)
```

拓扑只参与一处判定（`comm.py:49`）：

```python
full_connectivity = link.topology in ("all_to_all","nvswitch","full_mesh")
# AG/RS: full→1·α 否则 (N-1)·α      AR: full→2·α 否则 2(N-1)·α
# A2A:   full→1·α  N>16→log2(N)·α  否则 (N-1)·α
bw_term = S·(N-1)/N·β               # 与 topology 完全无关
```

互联数据结构（`spec.py:84-87`）严格两级：`InterconnectSpec.intra_node` / `inter_node`，各一个 `LinkSpec`（`spec.py:67-81`，字段：`type / bandwidth_gbps / latency_us / topology / num_devices`）。

层次分解 `collective_time_hierarchical`（`comm.py:75-118`）：仅 AG/RS/AR 做「节点内 D 卡 + 节点间 L=N/D 节点」两级；A2A/P2P 不分解；`tier_for_group`（`comm.py:121-137`）二选一，PP 强制 inter_node。

配置经 `config_loader._parse_system`（`config_loader.py:218-243`）只取 `bf16/fp8/fp4/cube/vector/sram/overlap` + `memory` + `interconnect` 进 `SystemSpec`；`type` 字符串仅 `excel_exporter.py:355` 用于展示。

trace 路径 `transform/analysis/comm_latency.py` **完全不读 `topology`**，延迟步数写死 `(n-1)`。

---

## 2. 问题清单（按严重度）

| ID | 问题 | 严重度 | 影响 |
|----|------|--------|------|
| **P1** | `topology` 只有二元「是否全连接」，`fat_tree`/`clos` 被错套 ring 的 `(N-1)·α` | **高** | 大集群（高 DP、小 grad bucket、频繁 PP P2P）延迟项被系统性高估约 `(N-1)/log2(N)` 倍（N=64 时 ~10×），estimate 偏保守 |
| **P2** | 带宽项 `S·(N-1)/N·β` 对 topology 完全不敏感；无 over-subscription、无 tree-root 瓶颈 | 中 | 胖树 spine 收敛比、tree 算法带宽都无法表达 |
| **P3** | 在网归约（NVLS / IB+SHARP）未建模 | 中 | 支持 NVLS 的 NVSwitch 硬件 AllReduce 被高估 ~2×；当前对所有硬件一视同仁 |
| **P4** | `type` 字符串（HCCS/RoCE/…）不进成本模型，纯展示 | 低 | 本身**正确**（参数驱动而非技术名驱动），但易被误解为「需要按 type 挂系数」——见 §4.3 的反模式说明 |
| **P5** | 互联硬编码两级；A2A/P2P 不做层次分解；PP 强制 inter | 中 | 单机内 PP 被高估；MoE EP all-to-all 跨节点一锅端走 inter；三级 fabric（NVL 域 / 超节点）表达不了 |
| **P6** | trace 路径与 estimate 路径口径不一致（前者完全不读 topology） | 中 | 同硬件两条路结果对不上 |

---

## 3. 目标 / 非目标

**目标**

- G1：`topology` 从二元 flag 升级为可扩展的「拓扑类 → 代价律」派发，至少正确区分 `switched_full / switched_tree / ring / torus`。
- G2：修正 `fat_tree/clos` 的延迟代价律（`(N-1)·α → ~2·log2(N)·α`）。
- G3：引入在网归约修正参数，**由 topology 能力派生默认 + 显式覆盖**，连续可标定。
- G4：引入可选 over-subscription，对 switched_tree 带宽项 derate。
- G5：全程遵守 anchor 纪律：把改动拆成「零位移 opt-in」与「**⚠️ 非 no-op** 需标定」两类，分 commit、分流程。

**非目标**

- N1：不做 dragonfly / 拥塞 / incast 建模（alpha-beta 框架装不下，边际收益低）。
- N2：本 spec 不强制统一 trace 路径（P6 列为可选分期，见 D7）。
- N3：不引入第三互联层级（除非 B1 指明有超节点硬件）。
- N4：不重写 alpha-beta 框架本身。

---

## 4. 设计提案

### 4.1 拓扑类派发（G1, G2 — 解 P1/P2）

`comm.py` 内新增小查表，**不引入新数据结构**：

```python
_TOPO_CLASS = {
    "nvswitch":"switched_full", "all_to_all":"switched_full", "full_mesh":"switched_full",
    "fat_tree":"switched_tree", "clos":"switched_tree",
    "ring":"ring", "torus":"torus", "mesh":"torus",
}   # 未知 → "ring"（保守，等同现默认分支）
```

AR 延迟步数按类派发（AG/RS 去掉 `2·` 系数；A2A 保留现有 Bruck）：

| 类 | 延迟步数 L（AR） | 带宽项系数 |
|----|------------------|------------|
| switched_full | `2` | `1.0` |
| switched_tree | `2·⌈log2(N)⌉` | `oversubscription`（默认 1.0） |
| ring | `2·(N-1)` | `1.0` |
| torus | `2·Σ_d(n_d−1)` | `1.0` |

- **R1**：实现 `_TOPO_CLASS` 派发，AR/AG/RS 三类按表取延迟步数。
- **R2**：`switched_full` 集合与现 `full_connectivity` 完全等价（zero-shift）。
- **R3**：未知 topology → `ring`（= 现默认分支，zero-shift）。
- **R4 ⚠️ 非 no-op**：`fat_tree/clos` → `switched_tree`（`2·log2(N)·α`），会移动所有 inter 为 fat_tree 的 anchor（昇腾/NV 配置 inter 全是 fat_tree）。走 §6 标定流程，单独 commit。

### 4.2 Over-subscription（G4 — 解 P2）

`LinkSpec` 加 `oversubscription: float = 1.0`，仅 derate `switched_tree` 带宽项（`bw_term *= oversubscription`）。默认 1.0 → no-op。

- **R5**：`spec.py` / `registry._parse_link` 增字段，默认 1.0。

### 4.3 在网归约（G3 — 解 P3，澄清 P4）

调研结论（前序已确认）：

| 互联 | 在网归约 |
|------|----------|
| NVLink+NVSwitch（NVLS） | 有（NCCL `NVLS`，AR ~2× 提速） |
| IB+SHARP | 有（需 SHARP 交换机 + 启用） |
| 经典 HCCS（910B/C） | **无**——优势纯在带宽/延迟/full_mesh，已被现模型刻画 |
| 普通 RoCE | 无 |

**反模式说明（重要）**：按 `type` 字符串挂系数会引入事实错误——经典 HCCS 与 RoCE 一样没有在网归约，差别已在带宽数值里。HCCS **不该**因「是 HCCS」拿额外加成，否则昇腾估算被系统性高估。

设计：`LinkSpec` 加 `ar_reduce_factor: float = 1.0`（连续，<1 表示交换机做 reduce）。默认**绑 `topology=="nvswitch"`** 派生（非 `type` 字面量），HCCS/RoCE 因 topology 是 full_mesh/fat_tree 自动保持 1.0。仅改 AR 分支：

```python
if c.kind == "AR":
    f = link.ar_reduce_factor
    if f < 1.0:
        return 2*alpha + 2*bw_term*f          # 单跳 + 流量塌缩
    # 否则走 §4.1 派发
```

`collective_time_hierarchical` 中 AR 拆 RS+AG 那条路（`comm.py:113-118`）：`f<1` 的层不拆，直接单层 AR。

- **R6**：`ar_reduce_factor` 字段 + AR 分支修正。
- **R7**：默认派生绑 `topology`，非 `type`；昇腾配置不动 → zero-shift。
- **R8 ⚠️ 非 no-op**：给 NVSwitch 硬件设 `<1` 时移动对应 anchor，走标定。

### 4.4 层次分解与 PP（解 P5 的可控部分）

- **R9**：修正 PP intra-node 检测——单机内 PP（相邻流水级 `cp*tp*2 ≤ intra_domain`）应走 intra_node 而非强制 inter（现 `comm.py:130-131` 强制 inter）。⚠️ 可能 **非 no-op**（影响单机 PP anchor），见 D5。
- A2A 层次分解、第三互联层级：列入非目标，见 D6 / B1。

---

## 5. 决策点（请逐项批注）

> 每项给选项 / 优劣 / **推荐**。默认按推荐执行，除非你改选。

**D1 — 在网归约 keying 方式**
- (a) 按 `type` 字符串：实现简单，但事实错误（HCCS 误获加成），脆弱
- (b) **绑 `topology=="nvswitch"` 派生默认 + YAML 显式覆盖**（推荐）：参数驱动，昇腾天然为 1.0，可覆盖 IB+SHARP
- (c) 仅显式字段、无任何自动默认：最安全但每个 NV 配置都要手写
- **推荐 (b)**。

**D2 — 在网归约因子取值**
- (a) 硬编码 0.5（理论上限）：简单但乐观，未标定
- (b) **连续参数默认 1.0，per-hardware 标定到匹配**（推荐）：可收敛、默认 no-op
- **推荐 (b)**。

**D3 — 拓扑类粒度**
- (a) 保持二元：不解决 P1
- (b) **4 类 switched_full/switched_tree/ring/torus**（推荐）：覆盖现实主流，复杂度可控
- (c) 全算法级派发（ring/RD/RHD/双二叉树/Bruck 分消息大小）：最准但过度工程，需消息大小阈值标定
- **推荐 (b)**；(c) 列为远期。

**D4 — fat_tree 修正（P1 核心，⚠️ 非 no-op）**
- (a) 维持现状 `(N-1)·α`：零风险但已知系统性高估保留
- (b) **修正为 `2·log2(N)·α`，走 anchor 标定流程**（推荐）：纠正主偏差
- (c) 做成 YAML 可选算法（fat_tree 可选 ring/RHD）：灵活但增配置面与认知负担
- **推荐 (b)**，前提是你接受 anchor 重标定（见 B3）。

**D5 — 单机内 PP 走 intra（R9，可能 ⚠️ 非 no-op）**
- (a) 维持强制 inter：保守、零风险，单机 PP 偏高估
- (b) **按 `cp*tp*2 ≤ intra_domain` 判定走 intra**（推荐）：更准
- **推荐 (b)**；若影响 anchor 则随 D4 一起标定。

**D6 — 第三互联层级 / A2A 层次分解**
- (a) **本期不做，维持两级 + A2A 单层**（推荐）：范围可控
- (b) 泛化为 N-tier list + A2A 层次：大改 `InterconnectSpec`，仅当 B1 有超节点硬件才值得
- **推荐 (a)**，除非 B1 指明需求。

**D7 — trace 路径口径统一（P6）**
- (a) **本 spec 不动，列为独立后续项**（推荐）：聚焦 estimate，降低本次风险
- (b) 让 `comm_latency.py` 委托 `comm.py`：彻底但牵连图捕获回归面
- (c) 复制拓扑逻辑到 trace 路径：快但双份维护
- **推荐 (a)**；是否纳入取决于 B4。

**D8 — over-subscription 范围**
- (a) **仅 switched_tree 带宽项 derate，默认 1.0**（推荐）：精准、no-op
- (b) 所有 inter 链路通用 derate：影响面大、易误用
- **推荐 (a)**。

**D9 — torus 维度表达**
- (a) **本期不实现 torus（关键字保留，落 ring 兜底）**（推荐，除非 B2 有 mesh 硬件）：现配置无 torus
- (b) `LinkSpec` 加 `topology_shape: list[int]`，实现 `Σ(n_d−1)`：仅 B2 确认有 mesh 硬件才做
- **推荐 (a)**，依 B2 决定。

**D10 — 配置迁移 / 向后兼容**
- (a) **新字段全部带默认值，旧 YAML 零改动；新行为靠默认派生触发**（推荐）
- (b) 要求显式声明能力字段：破坏现有 YAML
- **推荐 (a)**。

**D11 — 分期与 commit 切分**
- (a) **Phase 1 全零位移（R1/R2/R3/R5/R6/R7/D10）一个 PR；Phase 2 ⚠️ 非 no-op（R4/R8/R9）走标定单独 PR**（推荐）
- (b) 全量一个 PR：anchor 位移与重构混在一起，难定责
- **推荐 (a)**。

---

## 6. 标定与回归策略

- **零位移门槛**：Phase 1 合入前跑 `pytest tests/training/anchors/test_anchors.py`，`test_anchor_mfu_strict` / `test_anchor_step_time_strict` 必须**逐值零 diff**（按构造保证：默认参数下新分支不进、表达式等价）。
- **⚠️ 非 no-op 流程（Phase 2）**：受影响 anchor 先切 `strict_mfu_check: false` 标定模式 → 用参考基准把 `log2` 模型 / `ar_reduce_factor` / `oversubscription` 拟合到匹配 → 复核后转 strict。**禁止**把 R4/R8/R9 与零位移项混在同一 commit。
- **新单测**（`tests/training/test_comm.py`）：
  - 同 collective，`ar_reduce_factor` 1.0 vs 0.5 → AR 时间比 ≈ 输入比，AG/RS/A2A 不变；
  - `switched_tree` vs `ring` 在 N=64 → 延迟项比 ≈ `log2(N)/(N-1)`，带宽项相等（oversub=1）；
  - 未知 topology → 与现 `ring` 分支逐值相等。

---

## 7. 配置面与向后兼容

`LinkSpec` 新增（均带默认，旧 YAML 无需改）：

| 字段 | 默认 | 含义 |
|------|------|------|
| `ar_reduce_factor` | `1.0` | <1 = 交换机做 reduce |
| `oversubscription` | `1.0` | >1 = switched_tree spine 收敛比 |
| `topology_shape`（D9-b 时） | `None` | torus 维度形状 |

`registry._parse_link`：显式值优先；否则 `topology=="nvswitch"` → `ar_reduce_factor=0.5`，其余 1.0。`type` 字段保留（展示 + 不参与成本）。昇腾/现有 NV 配置 YAML **本期不改文件**（除非 D4/D9 决定后需要）。

---

## 8. 范围边界

- 做：estimate 路径 AR/AG/RS/PP 的拓扑代价律、在网归约、over-subscription。
- 不做：dragonfly / 拥塞 / incast（N1）；trace 路径统一（除非 B4）；第三层级 / A2A 层次（除非 B1）；torus（除非 B2）；alpha-beta 框架重写（N4）。

---

## 9. 需你拍板的边界条件（评审时填）

> 这些我无法替你决定，且会改变设计/范围。请直接在此填。

- **B1 — 超节点 / 三级 fabric**：是否需建模昇腾 UB 超节点或 NVL 域（节点内再分子域 / 第三层级）？  默认假设：**否**（维持两级，D6-a）。
- **B2 — mesh/torus 硬件**：是否有 torus/mesh 互联硬件需建模？  默认假设：**否**（D9-a，torus 落 ring 兜底）。
- **B3 — anchor 重标定预算**：是否接受 Phase 2 移动现有 anchor（fat_tree/PP/NVLS）并投入标定？  默认假设：**接受**（否则 D4/D5/D8/P3 全部退化为仅保留默认 no-op）。
- **B4 — trace 路径**：P6（trace 与 estimate 口径不一致）是否纳入本次范围？  默认假设：**否**（D7-a，独立后续项）。
- **B5 — 标定参考数据来源**：fat_tree 大集群 AllReduce、NVLS/SHARP 的标定基准用公开数据还是你方内部 benchmark？是否已有可用数据？  默认假设：**公开数据 + calibration 模式**，精度待数据确认。
- **B6 — 优先级排序**：P1（fat_tree）/ P3（在网归约）/ P5（PP/层次）/ P6（口径）哪些是 must-have、哪些可砍？  默认假设：P1 > P3 > P5 > P6。
- **B7 — 偏好补充**：其它偏好或硬约束（如「严禁动 spec.py 数据结构」「必须保持 trace 兼容」等）请在此列。

---

## 10. 验证策略

1. Phase 1 合入前：anchor strict 双测零 diff（§6）。
2. 新单测覆盖派发 / 在网归约 / oversub / 未知兜底（§6）。
3. Phase 2：标定后受影响 anchor 复核 + `validation/cli` 端到端对公开 benchmark 不回退。
4. 报表一致性：`excel_exporter.py` 展示新字段（`ar_reduce_factor`/`oversubscription`），不影响成本。

---

## 11. 分期建议（供后续 plan）

- **Phase 1（零位移，单 PR）**：R1/R2/R3/R5/R6/R7 + D10 + 新单测；门槛 = anchor 双测零 diff。
- **Phase 2（⚠️ 非 no-op，单 PR，走标定）**：R4（fat_tree）+ R8（NVLS）+ R9（PP intra）；门槛 = 标定后 anchor 复核 + 端到端不回退。
- **Phase 3（可选，依 B1/B2/B4）**：torus / 第三层级 / trace 口径统一。

> 评审通过后，我据此出实施 plan（落到具体文件/函数/diff/测试命令）。
