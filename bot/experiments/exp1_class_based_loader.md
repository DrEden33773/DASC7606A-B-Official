# Experiment #1: Class-Based DataLoader

**分支**: exp-1-multi-data-loader  
**日期**: 2025-10-17  
**状态**: ✅ 实现完成，待测试

---

## 🎯 实验目标

**核心假设**: Mixup/CutMix 对细节敏感类（人类、小动物）造成伤害

**解决方案**:

- Detail-sensitive classes: **完全禁用** Mixup/CutMix
- Other classes: 正常使用 Mixup/CutMix

**预期**: 人类类 F1 提升 +0.06-0.10 → 整体 F1 +0.01-0.02

---

## 🔬 方案设计

### 类别划分

**Detail-sensitive classes (10 个, ~4k samples)**:

```
人类类 (5):
- baby, boy, girl, man, woman

小动物类 (5):
- beaver, mouse, otter, possum, shrew
```

**Other classes (90 个, ~36k samples)**:

- 其余所有类别

### 训练策略

**Detail-sensitive loader**:

```python
# 不使用 Mixup/CutMix
mixup_alpha = 0
cutmix_alpha = 0

→ 保留细节，避免混合破坏面部/纹理特征
```

**Normal loader**:

```python
# 正常使用 Mixup/CutMix
mixup_alpha = 0.25
cutmix_alpha = 0.65

→ 增强泛化，CutMix 对机械/植物类极有效
```

### 实现

**两个独立 DataLoader**:

```python
detail_loader = DataLoader(
    Subset(train_data, detail_sensitive_indices),
    batch_size=128,
    ...
)

normal_loader = DataLoader(
    Subset(train_data, normal_indices),
    batch_size=128,
    ...
)
```

**交替训练**:

```python
for step in range(total_steps):
    if step % 2 == 0:
        # Detail classes (no mixing)
        batch = next(detail_iter)
        loss = train_step(batch, mixup=False)
    else:
        # Normal classes (with mixing)
        batch = next(normal_iter)
        loss = train_step(batch, mixup=True)
```

---

## 📊 运行配置

### PyramidNet-110 + Class-Based Loader

```bash
python main.py \
    --model pyramidnet110_270 \
    --drop_path_rate 0.1 \
    --aug_strength randaugment \
    --weight_decay 1e-4 \
    --use_class_based_loader \
    --num_epochs 600 \
    --early_stopping_patience 60 \
    --seed 42
```

### Wide ResNet-28-10 + Class-Based Loader

```bash
python main.py \
    --model wide_resnet28_10 \
    --drop_path_rate 0.1 \
    --aug_strength randaugment \
    --use_class_based_loader \
    --num_epochs 300 \
    --seed 42
```

---

## 📈 预期效果分析

### 保守估计

**人类类 (5 个类, 5% 样本)**:

```
当前 F1: 0.59 (平均)
不混合后: 0.63 (+0.04)
贡献: 0.05 × 0.04 = +0.002
```

**其他类 (95 个类, 95% 样本)**:

```
当前 F1: 0.85
保持: 0.85 (Mixup/CutMix 仍正常使用)
贡献: 0.95 × 0.85 = 0.8075
```

**总 F1**: 0.002 + 0.8075 = **0.8095**  
vs 当前 0.82: -0.0105 ❌

**结论**: 可能略降（人类类提升不足以弥补样本量差距）

---

### 乐观估计

**人类类**:

```
当前 F1: 0.59
不混合后: 0.68 (+0.09)
贡献: 0.05 × 0.09 = +0.0045
```

**其他类**:

```
保持: 0.85
贡献: 0.95 × 0.85 = 0.8075
```

**总 F1**: 0.0045 + 0.8075 = **0.812**  
vs 当前 0.82: -0.008 ❌

**结论**: 仍可能略降

---

### 最乐观估计

**人类类**:

```
不混合 + RandAugment 专注细节:
F1: 0.59 → 0.72 (+0.13)
贡献: 0.05 × 0.13 = +0.0065
```

**其他类**:

```
保持: 0.85
贡献: 0.8075
```

**总 F1**: 0.0065 + 0.8075 = **0.814**  
vs 当前 0.82: -0.006 ❌

**仍然略降！**

---

## 🔍 核心问题

### 样本量失衡

**Detail-sensitive classes**:

- 10 个类
- ~4,000 samples (10%)
- 即使 F1 从 0.59 → 0.70 (+0.11)
- 整体贡献: 0.10 × 0.11 = +0.011

**Other classes**:

- 90 个类
- ~36,000 samples (90%)
- 如果 F1 略降: 0.85 → 0.84 (-0.01)
- 整体影响: 0.90 × (-0.01) = -0.009

**净效果**: +0.011 - 0.009 = **+0.002** (微乎其微)

---

## ⚠️ 风险

**1. 其他类性能下降**

- 失去 Mixup/CutMix 的正则化效果？
- 可能略微过拟合

**2. 训练不均衡**

- Detail loader 样本少 (4k)
- Normal loader 样本多 (36k)
- 可能导致训练不稳定

---

## 🎯 我的预测

### 最可能结果

**Test F1 = 0.81-0.82**

**vs 当前最佳**:

- PyramidNet-110: 0.82
- 可能持平或略降

**原因**:

- 人类类提升 (+0.09)
- 但样本量太少 (10%)
- 整体贡献 +0.009
- 可能被其他类的轻微下降抵消

---

## ✅ 值得尝试的原因

**1. 验证假设**

- 你的观察很深刻
- 值得实验验证

**2. 学术价值**

- 即使 F1 不提升
- 也能理解 Mixup/CutMix 的作用机制

**3. 实现成本低**

- 已经实现完成
- 只需运行验证 (3.6 小时)

---

## 🚀 立即运行

```bash
python main.py \
    --model pyramidnet110_270 \
    --drop_path_rate 0.1 \
    --aug_strength randaugment \
    --weight_decay 1e-4 \
    --use_class_based_loader \
    --num_epochs 600 \
    --seed 50
```

**监控要点**:

- 人类类 F1 (boy, girl, man, woman, baby)
- 整体 F1
- vs Baseline (0.82)

---

## 📝 实验日志

### 2025-10-17

- ✅ load_data_class_based() 实现
- ✅ train_epoch_class_based() 实现
- ✅ main.py 集成
- ✅ Linting 检查通过
- 🟡 待运行实验

---

**实验设计**: ✅ 完成  
**理论分析**: ✅ 完成  
**预期**: F1 = 0.81-0.82 (可能不提升，但值得验证)

**立即运行验证你的假设！** 🚀
