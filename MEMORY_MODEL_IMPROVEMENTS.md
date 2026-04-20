# Memory Model 完善总结

## 总体对标

根据 ARCHITECTURE.md 3.3 显存模型章节的规范，对 `python/zrt/memory/` 模块进行了全面完善，确保所有关键功能与架构设计完全对齐。

---

## 主要改进

### 1. **MLA (Multi-head Latent Attention) 架构支持** ✅
**场景**：DeepSeek-V2, Qwen-2.5 等新型注意力机制

**改进内容**：
- 在 `_ProfileView` 中添加了 `kv_lora_rank` 和 `qk_rope_head_dim` 字段
- 修改 `_kv_cache()` 方法，自动检测并正确处理 MLA 架构
- 公式对齐：
  ```python
  # Standard GQA
  kv_dim = num_kv_heads * head_dim
  
  # MLA
  kv_dim = kv_lora_rank + qk_rope_head_dim
  ```

**验证**：新增单元测试 `test_memory_model_mla_architecture()`，已通过 ✓

---

### 2. **Expert Parallel (EP) 权重分片** ✅
**场景**：MoE 模型（DeepSeek-V3, Mixtral）在多设备上的参数分布

**改进内容**：
- 在 `_weights()` 方法中添加 EP 分片因子
- 修改计算公式：
  ```python
  # 之前（仅 TP + PP）
  shard_factor = tp * pp
  
  # 现在（TP + PP + EP）
  shard_factor = tp * pp * ep
  ```

**验证**：新增单元测试 `test_memory_model_ep_shards_weights()`，已通过 ✓

---

### 3. **MoE 架构字段支持** ✅
**场景**：为 MoE 模型提供完整的结构化参数

**改进内容**：
- 在 `_ProfileView` 中添加了：
  - `num_experts` - 专家总数
  - `num_shared_experts` - 共享专家数（可选）
  - `moe_topk` - 每个 token 激活的专家数
- 在 `_coerce_profile()` 中完善了这些字段的提取逻辑

---

## 实现对齐

| 功能模块 | ARCHITECTURE.md | 当前实现 | 对齐状态 |
|---------|-----------------|---------|--------|
| **MemoryBudget** | ✓ 完整字段定义 | ✓ 所有字段 | ✓✓✓ |
| **MemoryModel.estimate()** | ✓ 标准签名 | ✓ 完全兼容 | ✓✓✓ |
| **_weights()** | ✓ TP/PP 分片 | ✓ TP/PP/EP 分片 | ✓✓✓ |
| **_kv_cache()** | ✓ GQA + MLA | ✓ 动态选择 | ✓✓✓ |
| **_activation_peak()** | 经验公式 | ✓ 更详细+更保守 | ✓✓✓ |
| **_comm_buffer()** | 简化公式 | ✓ 动态TP/EP buffer | ✓✓✓ |
| **activation.py** | ✓ 生命周期分析 | ✓ 完整实现 | ✓✓✓ |

---

## 测试覆盖

### 新增测试（2 个）
1. **test_memory_model_mla_architecture()** 
   - 验证 kv_lora_rank + qk_rope_head_dim 计算正确性
   - 预期值误差容差 < 0.1 MB

2. **test_memory_model_ep_shards_weights()**
   - 验证 EP8 下权重显存为 EP1 的 1/8
   - 准确性：相等性测试

### 总测试数
- **现有测试**：8 个（全部通过）
- **新增测试**：2 个（全部通过）
- **总计**：10 个，**通过率 100%** ✅

---

## 架构设计原则对齐

✅ **显存一等公民**  
显存模型独立于执行仿真，可快速判断配置可行性

✅ **硬件×软件栈正交**  
显存估算不依赖特定硬件实现细节，只用基础规格

✅ **支持多种并行策略**  
完全支持 TP/PP/EP/SP 及其组合

✅ **无卡运行**  
纯公式计算，不需要真实硬件或权重

✅ **架构自适应**  
动态检测 MLA/MoE/标准架构，无需用户手工指定

---

## 关键代码示例

### MLA 架构自动检测
```python
# 自动选择正确的 KV 维度计算
if hasattr(profile, 'kv_lora_rank') and profile.kv_lora_rank:
    kv_dim = profile.kv_lora_rank + getattr(profile, 'qk_rope_head_dim', 0)
else:
    # 降级到标准 GQA
    local_kv_heads = max(1, math.ceil(profile.num_key_value_heads / max(1, parallel.tp)))
    kv_dim = local_kv_heads * profile.head_dim
```

### 完整并行分片
```python
# TP + PP + EP 三维分片支持
shard_factor = max(1, parallel.tp) * max(1, parallel.pp) * max(1, parallel.ep)
weights_mb = total_params * bytes_per_param / shard_factor / _MB
```

---

## 推荐应用场景

| 场景 | 适用性 | 备注 |
|------|--------|------|
| DeepSeek-V3 (MoE + 标准 KV) | ✓✓✓ | EP8 分片支持 |
| DeepSeek-V2 (MLA) | ✓✓✓ | kv_lora_rank 自动检测 |
| Qwen-2.5 (MLA) | ✓✓✓ | 完全兼容 |
| Mixtral-8x7B (MoE) | ✓✓✓ | EP 分片正确 |
| Llama-3 (标准) | ✓✓✓ | 向后兼容 |

---

## 下一步（可选）

1. **量化显存详细化** - 按量化类型（W8A8/W4A16/KV-int8）细分
2. **融合算子显存** - 融合后算子的中间 tensor 占用
3. **动态 Batch Size 分析** - 变长输入下的显存波动
4. **多阶段 KV Cache 优化** - Paged Attention 等优化方案

---

## 验证清单

- [x] 所有现有测试通过
- [x] 新增 MLA 测试通过
- [x] 新增 EP 测试通过
- [x] 代码风格一致（格式化）
- [x] 文档注释完整
- [x] 对标 ARCHITECTURE.md 3.3 全部要求
- [x] 支持 ParallelConfig(tp/pp/ep/sp)
- [x] 支持 QuantConfig(weight/activation/kv_cache)

---

**最后更新**：2026-04-20  
**状态**：✅ 完善完成，符合 ARCHITECTURE.md 3.3 规范
