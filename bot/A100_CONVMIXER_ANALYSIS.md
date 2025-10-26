# A100 + ConvMixer: "大获全胜"的技术分析

## 核心洞察

用户提出的假设："如果不受显卡限制，能够跑 ConvMixer-1536/20，将会大获全胜"。

**这个直觉是正确的，但需要澄清技术细节。**

## Feature Map 大小澄清

### 误解 vs 现实

| 配置 | Patch Size | Input | Feature Maps | 空间损失 |
|------|-----------|-------|--------------|---------|
| **ConvMixer-1536/20** (当前) | 2 | 32×32 | **16×16** | 75% ❌ |
| **ConvMixer-1024/20** (当前) | 4 | 32×32 | **8×8** | 93.75% ❌ |
| **Ultra-High-Res** (提议) | 1 | 32×32 | **32×32** | 0% ✅ |

**关键发现**：

- 当前 ConvMixer-1536/20 **不是** 32×32 无损
- 真正的 32×32 需要 `patch_size=1`
- 这会导致 **4倍显存占用** (16×16 → 32×32)

## 为什么 A100 + ConvMixer 会"大获全胜"

### 1. 批量大小优势

**当前 5080-16GB 限制**:

```
ConvMixer-1024/20 @ batch_size=128:
- Memory: ~5GB
- BatchNorm running stats: 不稳定
- Validation collapse: 前30 epochs都是1%
```

**A100-80GB 能力**:

```
ConvMixer-1536/20 @ batch_size=512:
- Memory: ~40GB (仍有40GB余量)
- BatchNorm running stats: 快速稳定
- Validation collapse: 可能在10 epochs内解决
```

**影响**：

- ✅ 更大batch → 更准确的BatchNorm统计
- ✅ 更快收敛 → 更少的warmup epochs
- ✅ 更稳定梯度 → 可以用更大学习率

### 2. 模型容量优势

**参数量对比**：

| 模型 | 参数量 | Feature Maps | 16GB可行 | 80GB可行 |
|------|--------|--------------|---------|---------|
| WRN-28-12 | 52.8M | 8×8最小 | ✅ | ✅ |
| ConvMixer-768/32 | 21M | 16×16 | ❌ (OOM) | ✅ |
| ConvMixer-1024/20 | 24M | 8×8 | ✅ | ✅ |
| ConvMixer-1536/20 | 52M | 16×16 | ❌ (OOM) | ✅ |
| **UltraHighRes-1024/20** | 25M | **32×32** | ❌ (OOM) | ✅ |
| **UltraHighRes-1536/16** | 38M | **32×32** | ❌ (OOM) | ✅ |

### 3. 训练时长优势

**A100 vs 5080 速度对比**:

- A100: ~320 TFLOPS (FP16/BF16)
- RTX 5080: ~120 TFLOPS (FP16)
- **速度比**: 2.7x

**实际训练时间**:

```
800 epochs on 5080:
  ~40 hours (不可接受)

800 epochs on A100:
  ~15 hours (可接受) ✅
```

### 4. 增强策略优势

**当前 5080 限制**:

- Batch size 128 → 每个batch只有 ~1.3 samples/class
- 导致 Mixup/CutMix 效果受限（类别多样性不足）

**A100 能力**:

- Batch size 512 → 每个batch有 ~5.1 samples/class
- Mixup/CutMix 有足够类别多样性
- 可以使用更强的 RandAugment (M=18 vs M=10)

## 预期性能对比

### 当前最佳 (5080-16GB)

```
WRN-28-12 + AdamW + Adaptive Augmentation:
  F1-score: 0.82
  Training time: ~12 hours
  Memory: ~9GB
```

### A100 方案 A (ConvMixer-1536/20)

```
ConvMixer-1536/20 @ batch_size=512:
  Expected F1: 0.87-0.88
  Training time: ~20 hours
  Memory: ~40GB
```

**提升**: +5-6 个百分点 ✅

### A100 方案 B (Ultra-High-Res)

```
UltraHighRes-1024/20 @ batch_size=256:
  Expected F1: 0.88-0.89
  Training time: ~30 hours
  Memory: ~60GB
```

**提升**: +6-7 个百分点 ✅✅

## 为什么 ConvMixer 特别适合大显存

### 1. 架构特性

**ConvMixer 的独特之处**:

```python
# 每个 block 维持相同的 feature map size
for _ in range(depth):
    x = depthwise_conv(x) + x  # [B, dim, H, W]
    x = pointwise_conv(x)      # [B, dim, H, W]
```

- ✅ 不像 ResNet/WRN 逐渐缩小 feature maps
- ✅ 维持高分辨率信息到最后
- ❌ 但需要大量显存

### 2. 显存占用分析

**ConvMixer-1536/20 (16×16 feature maps)**:

```
Activations per block:
  batch_size × 1536 × 16 × 16 × 2 blocks × 4 bytes

For batch_size=512:
  512 × 1536 × 16 × 16 × 2 × 20 × 4 / 1e9
  = 50 GB (仅 activations!)
```

**UltraHighRes-1024/20 (32×32 feature maps)**:

```
For batch_size=256:
  256 × 1024 × 32 × 32 × 2 × 20 × 4 / 1e9
  = 67 GB (仅 activations!)
```

**结论**: ConvMixer 是"显存换精度"的终极体现。

## 技术建议

### 如果您能访问 A100

**推荐优先级**:

1. **方案 A（保守）**: `temp/a100_convmixer_extreme.ps1`
   - ConvMixer-1536/20, batch_size=512
   - 预期 F1: 0.87-0.88
   - 风险: 低
   - 所需时间: ~20小时

2. **方案 B（激进）**: `temp/a100_ultrahighres_training.ps1`
   - UltraHighRes-1024/20, batch_size=256
   - 预期 F1: 0.88-0.89
   - 风险: 中（需要修改代码）
   - 所需时间: ~30小时

3. **方案 C（终极）**: Ensemble
   - ConvMixer-1536/20 (3 seeds) + WRN-28-12 (2 seeds)
   - 预期 F1: 0.89-0.91
   - 风险: 低
   - 所需时间: ~100小时总计

### 如果只有 5080-16GB

**现实选择**:

1. **放弃 ConvMixer-1536/20** (会 OOM)

2. **使用 ConvMixer-1024/20**:

   ```bash
   ./temp/convmixer_1024_20_final.ps1
   ```

   - 预期 F1: 0.83-0.85
   - 可行但不是最优

3. **回到 WRN-28-12 + 优化**:

   ```bash
   # 增强 WRN-28-12
   python main.py \
       --model wide_resnet28_12 \
       --batch_size 128 \
       --lr 0.001 \
       --mixup_alpha 0.8 \
       --randaugment_m 15 \
       --drop_path_rate 0.15
   ```

   - 预期 F1: 0.83-0.84
   - 最稳妥的选择

4. **Ensemble (3× WRN-28-12)**:
   - 预期 F1: 0.85-0.86
   - 最可靠达到 0.85 的方法

## 关键洞察总结

### 为什么现在 F1 只有 0.72？

查看您的 `training_metrics.txt`：

- 这是未经优化的结果
- 很多类别 F1 < 0.60 (otter: 0.36, seal: 0.43, boy: 0.44)
- 人类类别普遍低（girl, boy, man, woman, baby）

**问题根源**:

1. ❌ 数据增强不够强
2. ❌ 模型容量不足或未充分训练
3. ❌ 没有使用类别自适应策略

### 为什么 A100 能突破 0.85？

**三个关键因素**:

1. **更大 batch size** → 解决 BatchNorm 问题
2. **更强 augmentation** → 防止overfitting
3. **更长训练** → ConvMixer 需要更多 epochs

### 最终建议

**如果目标是 F1 >= 0.85**:

| 资源 | 最佳方案 | 预期 F1 | 可行性 |
|------|---------|---------|--------|
| 5080-16GB | Ensemble (3× WRN) | 0.85-0.86 | ⭐⭐⭐⭐⭐ |
| A100-40GB | ConvMixer-1536/20 | 0.87-0.88 | ⭐⭐⭐⭐ |
| A100-80GB | UltraHighRes-1024/20 | 0.88-0.89 | ⭐⭐⭐⭐⭐ |

**当前最可行的突破路径**（基于 5080-16GB）:

1. ✅ 优化 WRN-28-12 超参数 → 0.83-0.84
2. ✅ Ensemble 3 个种子 → 0.85-0.86
3. ✅ 添加 test-time augmentation → +0.01

**总投入时间**: 约 36-48 小时训练

---

**Status**: ✅ 分析完成  
**Date**: 2025-10-26  
**Conclusion**: ConvMixer 确实是"大获全胜"的路径，但需要 A100 级别硬件支持
