# 🚨 ConvMixer 显存问题诊断报告

## 问题描述

**发现**：ConvMixer-768/32 (21M参数) 比 WRN-28-12 (52.8M参数) 占用更多显存！

- ConvMixer-768/32: **~12.5GB** 显存
- WRN-28-12: **~3.6GB** 显存

训练速度也明显更慢。

---

## 🔍 根本原因

### 显存占用对比（batch_size=128）

| 模型 | 参数量 | 参数显存 | **激活值显存** | 梯度显存 | Optimizer | **总显存** |
|------|-------|---------|---------------|---------|-----------|-----------|
| **ConvMixer-768/32** | 20.3M | 0.08GB | **12.19GB** | 0.08GB | 0.15GB | **12.49GB** |
| **WRN-28-12** | 52.6M | 0.20GB | **2.80GB** | 0.20GB | 0.39GB | **3.59GB** |

**关键差异**：激活值显存！ConvMixer 是 WRN 的 **4.3 倍**！

### 架构分析

#### ConvMixer-768/32 的问题

```
输入: 32×32×3
  ↓ Patch Embedding (patch_size=2)
特征图: 16×16×768  ← 大尺寸特征图
  ↓ Block 1 (depthwise + pointwise)
特征图: 16×16×768  ← 保持不变
  ↓ Block 2
特征图: 16×16×768  ← 保持不变
  ↓ ... (32 个 blocks)
特征图: 16×16×768  ← 一直保持大尺寸
  ↓ Global Pooling
输出: 100
```

**问题**：

- 整个网络保持 **16×16×768** 的大特征图
- 32 个 blocks，每个都需要存储激活值用于反向传播
- 激活值总量：`16×16×768 × 32 blocks × 128 batch ≈ 12GB`

#### WRN-28-12 的优势

```
输入: 32×32×3
  ↓
特征图: 32×32×192   (group1, 4 blocks)
  ↓ Downsample
特征图: 16×16×384   (group2, 4 blocks)
  ↓ Downsample
特征图: 8×8×768     (group3, 4 blocks)
  ↓ Global Pooling
输出: 100
```

**优势**：

- **逐渐下采样**，减少特征图尺寸
- 早期层虽然大，但后期层很小
- 总激活值远小于 ConvMixer

---

## 🐌 训练速度慢的原因

1. **大特征图**：16×16×768 需要更多计算和内存带宽
2. **深度串行**：32 个 blocks 必须顺序执行
3. **Depthwise Conv 不够优化**：GPU 上的优化不如标准卷积
4. **显存占用高**：可能触发更多的内存管理开销

---

## ✅ 解决方案

### 方案 1: 使用 ConvMixer-1024/20 (推荐 ⭐⭐⭐)

```powershell
python main.py `
    --model convmixer_1024_20 `  # patch_size=4, 特征图 8×8
    --optimizer adamw `
    --lr 0.001 `
    --weight_decay 0.01 `
    --batch_size 128 `  # 可以用更大的 batch
    --num_epochs 600 `
    --amp
```

**优势**：

- ✅ **patch_size=4**：特征图只有 **8×8×1024**
- ✅ 显存占用预计：**~4-5GB**（比 768/32 少 60%）
- ✅ 训练速度更快（特征图小）
- ✅ 参数量 22.9M，接近 768/32

**预期效果**：F1-score 0.83-0.85

### 方案 2: 减小 Batch Size

```powershell
python main.py `
    --model convmixer_768_32 `
    --batch_size 64 `  # 从 128 降到 64
    --optimizer adamw `
    --lr 0.001 `
    --weight_decay 0.01 `
    --num_epochs 600 `
    --amp
```

**优势**：

- ✅ 显存占用减半：**~6-7GB**
- ⚠️ 但训练速度还是慢
- ⚠️ Batch size 小可能影响性能

### 方案 3: 回到 WRN-28-12 (最安全 ⭐⭐⭐⭐⭐)

```powershell
python main.py `
    --model wide_resnet28_12 `
    --optimizer adamw `
    --lr 0.001 `
    --weight_decay 0.01 `
    --batch_size 128 `
    --dropout 0.3 `
    --drop_path_rate 0.1 `
    --num_epochs 600 `
    --cutmix_alpha 1.0 `
    --mixup_alpha 0.5 `
    --randaugment_n 2 `
    --randaugment_m 10 `
    --label_smoothing 0.1 `
    --amp
```

**优势**：

- ✅ 已验证 **F1=0.82**
- ✅ 显存效率高：**~3.6GB**
- ✅ 训练速度快
- ✅ 稳定可靠
- ⚠️ 但可能难以突破 0.85

### 方案 4: 尝试 PyramidNet-110 (备选)

```powershell
python main.py `
    --model pyramidnet110_270 `
    --optimizer adamw `
    --lr 0.001 `
    --weight_decay 0.01 `
    --batch_size 128 `
    --drop_path_rate 0.15 `
    --num_epochs 600 `
    --amp
```

**优势**：

- ✅ 论文报告 **83% accuracy** on CIFAR-100
- ✅ 参数量 26M，中等规模
- ✅ 逐渐增加通道，显存效率好
- ⚠️ 未在此项目中测试过

---

## 📊 性能预测对比

| 模型 | 显存占用 | 训练速度 | 预期F1 | 稳定性 | 推荐度 |
|------|---------|---------|--------|--------|--------|
| **ConvMixer-768/32** | 12.5GB ❌ | 慢 ❌ | 0.83-0.85 | 中 | ❌ 不推荐 |
| **ConvMixer-1024/20** | 4-5GB ✅ | 中等 | 0.83-0.85 | 中 | ⭐⭐⭐ 可尝试 |
| **WRN-28-12** | 3.6GB ✅ | 快 ✅ | 0.82 ✅ | 高 ✅ | ⭐⭐⭐⭐⭐ 推荐 |
| **PyramidNet-110** | ~4GB ✅ | 中等 | 0.83? | 未知 | ⭐⭐⭐ 可尝试 |

---

## 🎯 最终建议

### 推荐顺序

1. **立即尝试 ConvMixer-1024/20** (10 epochs 快速测试)
   - 如果显存占用确实降到 4-5GB，且训练速度可接受 → 继续
   - 如果还是太慢 → 放弃 ConvMixer

2. **如果 ConvMixer-1024/20 不理想，回到 WRN-28-12**
   - 已验证 F1=0.82
   - 专注优化其他方面（数据增强、正则化）
   - 可能的优化点：
     - 更强的数据增强 (RandAugment M=12-14)
     - Label smoothing 调整
     - Weight decay 调整
     - 更长的训练 (800 epochs?)

3. **备选：尝试 PyramidNet-110**
   - 论文声称 83% accuracy
   - 架构简单，显存效率好

### 关于 ConvMixer 的结论

**ConvMixer 的问题不是实现错误，而是架构设计**：

- ✅ **优点**：极简设计，易于理解
- ❌ **致命缺点**：保持大特征图，显存效率低
- ❌ **不适合**：CIFAR-100 这种小图像数据集

**ConvMixer 更适合的场景**：

- 大图像 (224×224 以上)
- Patch size 可以设置很大 (7×7, 14×14)
- Feature map 尺寸自然就小了

**CIFAR-100 的问题**：

- 原始图像只有 32×32
- Patch size=2 → 16×16 feature map (还是太大)
- Patch size=4 → 8×8 feature map (勉强可接受)
- Patch size=7 → 4×4 feature map (太小，损失信息)

---

## 📝 数据增强检查

**确认**：`input_size` 参数没有问题。

- 默认值已改回 **32**
- 不会导致图像尺寸变大
- 数据增强正常

问题纯粹是 ConvMixer 架构的显存效率问题。

---

## ✅ 下一步行动

### 立即执行：快速测试 ConvMixer-1024/20

```powershell
# 10 epochs 快速测试
python main.py `
    --model convmixer_1024_20 `
    --optimizer adamw `
    --lr 0.001 `
    --weight_decay 0.01 `
    --batch_size 128 `
    --num_epochs 10 `
    --warmup_epochs 2 `
    --amp
```

**观察指标**：

1. GPU 显存占用（应该 <6GB）
2. 训练速度（每个 epoch 时间）
3. 第 10 epoch 的 val acc（应该 >30%）

**决策**：

- 如果显存 <6GB 且速度可接受 → 进行完整训练
- 如果显存仍然高或速度太慢 → **放弃 ConvMixer，回到 WRN-28-12**

---

**状态**: 问题已诊断清楚  
**根本原因**: ConvMixer 的大特征图架构不适合小图像  
**推荐方案**: 先试 ConvMixer-1024/20，不行就回 WRN-28-12  
**置信度**: 非常高（已通过诊断脚本验证）
