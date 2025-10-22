# Experiment #2: Three-Way Class-Based Loaders

**分支**: exp-2-three-loaders  
**日期**: 2025-10-17  
**状态**: ✅ 实现完成

---

## 🎯 完整的 Adaptive 策略实现

### 三类数据分割

**1. Detail-sensitive classes (10 个, ~10% samples)**:

```
人类类: baby, boy, girl, man, woman
小动物: beaver, mouse, otter, possum, shrew

策略: 0% Mixup, 0% CutMix
原因: 保留细节，避免混合破坏面部/纹理特征
```

**2. Local-feature classes (15 个, ~15% samples)**:

```
机械: bicycle, bus, motorcycle, pickup_truck, tank, tractor, train
植物: maple_tree, oak_tree, orchid, palm_tree, pine_tree, sunflower, tulip, willow_tree

策略: 20% Mixup, 80% CutMix
原因: CutMix 对局部特征极有效，增强轮廓/局部识别
```

**3. Mixed-strategy classes (75 个, ~75% samples)**:

```
其余所有类别

策略: 30% Mixup, 70% CutMix
原因: 平衡的正则化策略
```

---

## 🔧 实现细节

### 三个独立 DataLoader

```python
detail_loader = DataLoader(
    Subset(train_data, detail_sensitive_indices),
    batch_size=128, ...
)

local_loader = DataLoader(
    Subset(train_data, local_feature_indices),
    batch_size=128, ...
)

mixed_loader = DataLoader(
    Subset(train_data, mixed_indices),
    batch_size=128, ...
)
```

### 训练循环

```python
for step in range(total_steps):
    loader_type = step % 3
    
    if loader_type == 0:
        # Detail: NO mixing
        batch = next(detail_iter)
        aug_strategy = "detail"
    elif loader_type == 1:
        # Local: 20% Mixup, 80% CutMix
        batch = next(local_iter)
        aug_strategy = "local"
    else:
        # Mixed: 30% Mixup, 70% CutMix
        batch = next(mixed_iter)
        aug_strategy = "mixed"
    
    # Apply strategy-specific augmentation
    if aug_strategy == "detail":
        lam = 1.0  # No mixing
    elif aug_strategy == "local":
        if rand() < 0.2:
            Mixup(alpha=0.25)
        else:
            CutMix(alpha=0.65)
    else:  # mixed
        if rand() < 0.3:
            Mixup(alpha=0.25)
        else:
            CutMix(alpha=0.65)
```

---

## 🚀 运行配置

### PyramidNet-110 + 3-Way Loaders

```bash
python main.py \
    --model pyramidnet110_270 \
    --drop_path_rate 0.1 \
    --aug_strength randaugment \
    --weight_decay 1e-4 \
    --use_class_based_loader \
    --num_epochs 600 \
    --seed 51
```

### WRN-28-10 + 3-Way Loaders

```bash
python main.py \
    --use_class_based_loader \
    --seed 51
```

---

## 📊 预期效果

### 理论分析

**Detail classes (10 个)**:

```
当前 (adaptive, batch-level): F1 ≈ 0.59
新策略 (完全禁用 mix): F1 = 0.65-0.70
提升: +0.06-0.11
```

**Local classes (15 个)**:

```
当前 (adaptive, 80% CutMix): F1 ≈ 0.92
新策略 (精确 80% CutMix): F1 = 0.92-0.93
提升: 0-0.01
```

**Mixed classes (75 个)**:

```
当前 (adaptive, 70% CutMix): F1 ≈ 0.83
新策略 (精确 70% CutMix): F1 = 0.83
提升: 0
```

### Macro F1 计算

```
Total F1 = Σ(F1_i) / 100

Detail: 10 × 0.68 = 6.8
Local:  15 × 0.92 = 13.8
Mixed:  75 × 0.83 = 62.25

Total = (6.8 + 13.8 + 62.25) / 100 = 0.8285 ✨
```

**vs Baseline (PyramidNet 0.82)**: **+0.0085**  
**vs Adaptive (batch-level)**: **+0.005-0.01**

---

## 🎯 vs Adaptive Augmentation

### 当前 Adaptive (batch-level)

**问题**:

```python
if detail_count > local_count and detail_count > mixed_count:
    # 只有当 detail 类在 batch 中占多数才触发
    # 但 detail 类只有 10/100
    # 很难形成多数
    # 大部分时间仍用 CutMix
```

**实际效果**: Detail 类仍然被 CutMix 伤害

### 新方案 (sample-level)

**优势**:

```python
# 基于样本所属类别，100% 准确执行策略
if sample in detail_classes:
    NO mixing (100% 确定)
elif sample in local_classes:
    20% Mixup, 80% CutMix (100% 确定)
else:
    30% Mixup, 70% CutMix (100% 确定)
```

**预期**: 更精准的策略执行 → 更好的效果

---

## ✅ 实现完成检查

- [x] load_data_class_based() - 三路分割
- [x] train_epoch_class_based() - 三种策略
- [x] main.py 集成
- [x] Progress bar 显示策略
- [x] Linting 检查通过

---

## 🚀 立即运行

```bash
python main.py \
    --model pyramidnet110_270 \
    --use_class_based_loader \
    --seed 51
```

**预期**: F1 = **0.825-0.835**  
**成功概率**: 70-75%

---

**实现完成！准备测试！** 🎯
