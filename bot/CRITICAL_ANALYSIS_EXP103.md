# 🚨 Exp #103 失败根因分析

**结果**: Val F1 = 0.75 (vs Baseline 0.78, **下降 0.03!**)  
**分析时间**: 2025-10-17 16:00

---

## 💥 核心问题识别

### 问题 1: 增强过度 (Critical!)

**证据**:

```
Train Acc: ~48% (epoch 300)
Val Acc:   ~78% (epoch 300)
差距:      30%！
```

**正常情况应该是**:

- Train Acc ≈ Val Acc (± 5%)
- 或 Train Acc 略高（模型见过训练数据）

**异常信号**: Train Acc **远低于** Val Acc = **训练数据被过度增强**！

---

### 根本原因: RandAugment 与现有增强的**重复叠加**

#### 当前增强栈 (Exp #103)

```python
1. RandAugment (N=2, M=9):
   - 从 14 种操作中随机选 2 个
   - 包括: Rotate, Shear, Translate, Brightness, Contrast, etc.

2. Medium Augmentation:
   - Rotate (limit=20, p=0.8)        ← 与 RA 重复!
   - ShiftScaleRotate (p=0.8)        ← 与 RA 重复!
   - ColorJitter (p=0.8)             ← 与 RA 重复!
   - GaussianBlur/MotionBlur (p=0.4)
   - RandomBrightnessContrast (p=0.5) ← 与 RA 重复!
   - CoarseDropout (p=0.3)

3. Mixup/CutMix (batch-level)

总计: 同一张图可能经历 10+ 次变换！
```

**结果**: 训练图像被扭曲得面目全非 → 模型根本学不到特征！

---

## 🔍 代码问题定位

### 问题代码 (scripts/train_utils.py:527-565)

```python
# Line 527: 错误的叠加方式
randaugment_transforms = [RandAugment(n=randaugment_n, m=randaugment_m)]

# Line 562-565: 与现有增强叠加
augmentation_pipeline = A.Compose(
    randaugment_transforms  # ← RandAugment
    + [
        A.Rotate(...),      # ← 与 RA 的 Rotate 重复!
        A.ColorJitter(...), # ← 与 RA 的 Brightness/Contrast 重复!
        ...
    ]
)
```

**问题**: RandAugment 应该**替代**而非**叠加**到现有增强！

---

## 📊 性能下降分析

### 对比

| 实验 | 增强策略 | Train Acc | Val Acc | Val F1 | 分析 |
|-----|---------|-----------|---------|--------|------|
| Exp #100 | Medium + Mixup/CutMix | ~68% | 78% | 0.7802 | ✅ 正常 |
| Exp #103 | RA + Medium + Mixup/CutMix | ~48% | ~78% | 0.75 | 🚨 过度 |

**Train Acc 下降 20%**: 训练图像被破坏得太严重！

**Val F1 下降 0.03**: 模型没学好特征 → 泛化变差

---

## 🔧 解决方案

### 方案 A: RandAugment 替代现有增强 ⭐⭐⭐⭐⭐

**修改策略**:

- **启用 RandAugment** 时，**禁用** Medium augmentation
- 只保留基础的 Normalize + ToTensor

**代码修改** (train_utils.py):

```python
if use_randaugment:
    # 只用 RandAugment + 基础变换
    augmentation_pipeline = A.Compose([
        RandAugment(n=2, m=9),
        A.HorizontalFlip(p=0.5),  # 基础几何变换
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
else:
    # 使用现有的 Medium augmentation
    augmentation_pipeline = A.Compose([
        A.Rotate(...),
        A.ColorJitter(...),
        ...
    ])
```

**预期**: Train Acc 恢复正常 → Val F1 ≥ 0.78

---

### 方案 B: 降低 RandAugment 强度 ⭐⭐⭐⭐

保持叠加，但降低强度：

- M: 9 → **5 或 6** (降低 40%)
- N: 2 → **1** (减少操作数量)

**预期**: 减轻过度增强，但仍可能不够

---

### 方案 C: 仅用 Stochastic Depth ⭐⭐⭐⭐⭐

**最安全方案**:

```bash
python main.py \
    --model wide_resnet28_10 \
    --drop_path_rate 0.2 \
    --no_randaugment \       # ← 禁用 RandAugment
    --aug_strength medium \  # ← 使用原来的 medium
    --seed 42
```

**预期**:

- Baseline (0.78) + SD (+0.015-0.02) = **0.795-0.80**
- 更稳健，风险低

---

## 🎯 我的强烈建议

### ✅ **立即执行方案 C + A**

**Step 1: 先测试纯 Stochastic Depth (Exp #103-SD-only)**

```bash
python main.py \
    --model wide_resnet28_10 \
    --dropout 0.3 \
    --drop_path_rate 0.2 \
    --no_randaugment \
    --aug_strength medium \
    --num_epochs 500 \
    --batch_size 128 \
    --seed 42
```

**预期**: Val F1 = 0.795-0.805 (高概率达到 0.80!)

**Step 2: 如果 Step 1 达标，修复 RandAugment 实现**

改为替代模式，而非叠加模式。

---

## 🔍 关于 Train Acc 计算

**检查代码** (train_utils.py:1346-1353):

```python
if (use_cutmix and cutmix_alpha > 0) or mixup_alpha > 0:
    # Mixup/CutMix 的加权 accuracy
    correct += (
        lam * predicted.eq(targets_a).sum().item()
        + (1 - lam) * predicted.eq(targets_b).sum().item()
    )
```

**计算逻辑**: ✅ **正确**

**问题不在计算**，而在于**训练数据被过度增强**！

---

## 🎨 关于 GridMask

### 现在加 GridMask？❌ **绝对不要！**

**理由**:

1. **当前问题** = 增强过度
2. **GridMask** = 再加一层遮挡
3. **结果** = 雪上加霜，更差

**正确时机**:

- 在增强不足时考虑
- 当前是增强**过度**，方向相反！

---

## 📋 立即行动计划

### 🔥 优先级 1: Exp #103-Revised (SD Only)

**禁用 RandAugment，只保留 SD**

```bash
python main.py \
    --model wide_resnet28_10 \
    --drop_path_rate 0.2 \
    --no_randaugment \
    --seed 42
```

**预期**:

- 恢复到正常 train/val acc 比例
- Val F1 ≥ 0.795
- 有望达到 0.80

**时间**: 今晚运行，明早结果

---

### 🔥 优先级 2: 修复 RandAugment (如需要)

如果 SD-only 达到 0.80 → 不需要修复  
如果 SD-only < 0.80 → 修复 RandAugment 为替代模式

---

## 🎓 关键教训

### ❌ 错误假设

"RandAugment 和 Medium aug 可以叠加" → **错误！**

RandAugment **设计为独立使用**，不是叠加组件！

### ✅ 正确理解

```
方案 1: RandAugment ONLY (不推荐 CIFAR)
  RandAugment + HorizontalFlip + Normalize

方案 2: Traditional Aug (当前 Exp #100)
  Rotate + Flip + ColorJitter + ... + Normalize

方案 3: 混合 (需要小心)
  RandAugment (N=1, M=5) + Light Aug
```

我们用的是错误的方案 4: RandAugment + Full Medium Aug = **灾难**

---

## 📊 预期修复效果

```
Exp #103 (错误): 0.75
  ↓ 移除 RandAugment
Exp #103-Revised: 0.795-0.805 (预期)
  ↓ 达到目标!
Phase 1 完成: F1 ≥ 0.80 ✅
```

---

**建议**: 立即运行 SD-only 版本！
