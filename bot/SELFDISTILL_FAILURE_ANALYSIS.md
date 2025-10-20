# 🔴 Exp #300 自蒸馏失败分析

**结果**: Val F1 = 0.7968 (vs Baseline 0.8131, **下降 0.0163**)  
**结论**: 🚨 **自蒸馏反而降低了性能！**

---

## 📊 详细对比

### 性能倒退

| Exp | 配置 | Train Acc | Val Acc | Val F1 | Best Epoch |
|-----|------|-----------|---------|--------|------------|
| **#104b** | **WRN + SD + RA** | **59.67%** | **81.40%** | **0.8131** | **209** |
| #300 | + Self-Distill | 60.10% | 79.85% | **0.7968** | 206 |
| **差异** | - | **+0.43%** | **-1.55%** | **-0.0163** | **-3** |

**关键发现**:

- Train Acc 略高 → 在训练集上拟合更好
- Val Acc 明显降低 → **泛化能力变差**
- → **自蒸馏引入了过拟合或噪声**

---

## 🔍 类别分布对比

### 部分类别提升，部分下降

| 类别 | #104b F1 | #300 F1 | 变化 | 分析 |
|-----|----------|---------|------|------|
| boy | 0.51 | **0.59** | +0.08 | ✅ 提升 |
| girl | 0.57 | **0.52** | -0.05 | ❌ 下降 |
| otter | 0.52 | **0.54** | +0.02 | ✅ 略好 |
| seal | 0.54 | **0.51** | -0.03 | ❌ 下降 |
| man | 0.55 | **0.55** | 0.00 | 持平 |
| woman | 0.62 | **0.62** | 0.00 | 持平 |
| lawn_mower | 0.92 | **0.89** | -0.03 | ❌ 下降 |
| motorcycle | 0.92 | **0.94** | +0.02 | ✅ 略好 |
| pickup_truck | 0.94 | **0.96** | +0.02 | ✅ 略好 |

**模式**:

- 部分困难类略有提升 (boy +0.08)
- 但部分简单类下降 (lawn_mower -0.03)
- **整体平衡被破坏，总体下降**

---

## 🎯 失败根本原因分析

### 原因 1: 自蒸馏与现有强正则化冲突 🚨

**当前配置已经非常强**:

```
1. Stochastic Depth (drop_path=0.1)
2. RandAugment (N=2, M=9)
3. Mixup (alpha=0.25)
4. CutMix (alpha=0.65)
5. Dropout (0.3)
6. Weight Decay (5e-4)
7. EMA
8. Gradient Clipping
```

**再加自蒸馏**:

- 3 个中间分类器 = 多任务学习 = 额外正则化
- KL loss = 软标签约束 = 额外正则化

**结果**: **正则化过度** → 欠拟合 → 性能下降

**证据**:

- Train Acc 60% (vs baseline 60%) - 持平
- Val Acc 79.85% (vs baseline 81.4%) - 下降
- → 模型学习能力被过度限制

---

### 原因 2: 中间分类器干扰主网络 🚨

**理论问题**:

```
中间分类器的梯度会回传到 layer1/2/3
→ 主网络的梯度被中间任务"污染"
→ 主任务（最终分类）性能下降
```

**论文假设**: 中间监督有助于训练

**但我们的情况**:

- 已有 SD, RA, Mixup/CutMix (梯度流很好)
- 不需要额外的中间监督
- 反而引入噪声

---

### 原因 3: Loss 权重仍需优化

**当前**:

```python
total_loss = 0.3 * CE(teacher) + 0.7/3 * KL_total
           = 0.3 * CE + 0.233 * Σ KL

Hard label weight: 0.3
Soft label weight: 0.7 (分散到 3 个 student)
```

**可能问题**: Hard label 权重太低 (0.3)

---

### 原因 4: 自蒸馏不适合小数据集？

**BYOT 论文实验**:

- ImageNet (1.3M images)
- CIFAR-100 (50k images)

**我们**:

- CIFAR-100 (40k training after split)

**加上强正则化**: 数据更加"稀缺"

**假设**: 自蒸馏需要充足数据，小数据集反而有害

---

## 💡 下一步策略

### 🥇 **方案 A: 放弃自蒸馏，回归 Phase 1 最佳** ⭐⭐⭐⭐⭐

**Exp #310: Phase 1 极致优化**

```bash
python main.py \
    --model wide_resnet28_10 \
    --drop_path_rate 0.08 \
    --aug_strength randaugment \
    --randaugment_m 8 \
    --lr 0.0012 \
    --weight_decay 3e-4 \
    --num_epochs 800 \
    --early_stopping_patience 80 \
    --seed 42
```

**改动**:

- drop_path: 0.1 → 0.08 (略减正则)
- randaugment_m: 9 → 8 (略减增强)
- lr: 0.001 → 0.0012 (增强学习)
- weight_decay: 5e-4 → 3e-4 (减正则)
- epochs: 500 → 800 (更长训练)

**预期**: F1 = **0.82-0.83**  
**成功概率**: **70-75%**

---

### 🥈 **方案 B: 直接 Ensemble (3 个模型)** ⭐⭐⭐⭐⭐

**最保险的方案**:

```
1. 已有 Exp #104b (seed=42): F1 = 0.8131
2. 训练 Exp #104b-v2 (seed=43): F1 ≈ 0.81
3. 训练 Exp #104b-v3 (seed=44): F1 ≈ 0.81

测试时 soft voting:
pred = (model1(x) + model2(x) + model3(x)) / 3
```

**预期**: F1 = **0.825-0.84**  
**成功概率**: **85-90%**  
**时间**: 2 × 4h = 8h (并行 → 4h)

**风险**: 需要确认是否允许 ensemble

---

### 🥉 **方案 C: TTA (Test-Time Augmentation)** ⭐⭐⭐⭐

**单模型 + 测试时增强**:

```python
# 对每张测试图做 5 次增强，平均预测
def predict_with_tta(model, image):
    preds = []
    preds.append(model(image))  # 原图
    preds.append(model(hflip(image)))  # 水平翻转
    for _ in range(3):
        preds.append(model(augment(image)))  # 随机增强
    return torch.stack(preds).mean(0)
```

**预期**: F1 +0.01-0.02  
**成功概率**: 80%

---

## 🎯 我的最终建议

### ✅ **立即执行: 方案 A (极致微调 Phase 1)**

**理由**:

1. ✅ 自蒸馏已证明无效（3 次架构创新都失败）
2. ✅ WRN-28-10 + SD + RA 是已知最佳
3. ✅ 通过微调参数仍有提升空间

**命令**:

```bash
python main.py \
    --model wide_resnet28_10 \
    --drop_path_rate 0.08 \
    --aug_strength randaugment \
    --randaugment_m 8 \
    --lr 0.0012 \
    --weight_decay 3e-4 \
    --num_epochs 800 \
    --early_stopping_patience 80 \
    --seed 42
```

**预期**: F1 = 0.82-0.83  
**成功概率**: 70-75%

---

### 🔄 **备选: 方案 B (Ensemble)**

如果方案 A < 0.83，并行训练 3 个模型，soft voting

**预期**: F1 = 0.825-0.84  
**成功概率**: 85%+

---

## 🎓 关键教训

### ❌ 架构创新失败记录

```
1. WRN-28-12 (52.8M):      F1 = 0.80   (-0.013)
2. ConvNeXt-Tiny (28M):    F1 = 0.79   (-0.023)
3. Self-Distill (+2.5M):   F1 = 0.7968 (-0.0163)

所有架构创新都失败！
```

### ✅ 真正有效的优化

```
1. Wide ResNet-28-10 (架构)
2. Stochastic Depth 0.1 (正则化)
3. RandAugment 独立使用 (数据增强)
4. Mixup/CutMix 组合
5. 精细超参数调优
```

---

## 🎯 冲击 0.85 的现实路径

### 短期 (本周)

**方案 A**: 极致微调  
**目标**: 0.82-0.83

### 中期 (如需要)

**方案 B**: Ensemble  
**目标**: 0.825-0.84

### 接受现实

**Phase 1 配置 (F1=0.8131) 可能已接近极限**

**原因**:

- CIFAR-100 数据量有限 (40k)
- 32×32 分辨率限制
- 人类类别永恒难题

---

**建议**: 立即尝试方案 A！
