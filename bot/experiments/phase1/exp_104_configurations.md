# Exp #104 系列：优化配置搜索

**创建时间**: 2025-10-17 18:00  
**目标**: 突破 0.80  
**策略**: 测试不同的 drop_path_rate + augmentation 组合

---

## 🎯 实验矩阵

### 核心发现（来自之前实验）

```
Exp #100 (baseline):      dropout=0.3, drop_path=0.0, aug=medium  → F1=0.7802 ✅
Exp #103-Rev (SD too strong): dropout=0.3, drop_path=0.2, aug=medium  → F1=0.7791 ❌
```

**问题**: drop_path=0.2 过强  
**方向**: 降低到 0.1 或 0.15，同时测试 RandAugment

---

## 📋 推荐实验配置

### 🔥 Exp #104a: SD(0.1) + Medium Aug

**最保守方案，基于 baseline 微调**

```bash
python main.py \
    --model wide_resnet28_10 \
    --dropout 0.3 \
    --drop_path_rate 0.1 \
    --aug_strength medium \
    --lr 0.001 \
    --weight_decay 5e-4 \
    --num_epochs 500 \
    --batch_size 128 \
    --seed 42
```

**预期**: Val F1 = 0.785-0.795  
**优先级**: 🔥🔥🔥🔥🔥

---

### 🔥 Exp #104b: SD(0.1) + RandAugment

**新方案，使用纯 RandAugment**

```bash
python main.py \
    --model wide_resnet28_10 \
    --dropout 0.3 \
    --drop_path_rate 0.1 \
    --aug_strength randaugment \
    --randaugment_n 2 \
    --randaugment_m 9 \
    --lr 0.001 \
    --weight_decay 5e-4 \
    --num_epochs 500 \
    --batch_size 128 \
    --seed 42
```

**预期**: Val F1 = 0.79-0.82  
**优先级**: 🔥🔥🔥🔥🔥

---

### 🔥 Exp #104c: SD(0.15) + Medium Aug

**中等 SD 强度**

```bash
python main.py \
    --model wide_resnet28_10 \
    --dropout 0.3 \
    --drop_path_rate 0.15 \
    --aug_strength medium \
    --lr 0.001 \
    --weight_decay 5e-4 \
    --num_epochs 500 \
    --batch_size 128 \
    --seed 42
```

**预期**: Val F1 = 0.78-0.795  
**优先级**: 🔥🔥🔥🔥

---

### 🔥 Exp #104d: SD(0.1) + RandAugment(lighter)

**降低 RandAugment 强度**

```bash
python main.py \
    --model wide_resnet28_10 \
    --dropout 0.3 \
    --drop_path_rate 0.1 \
    --aug_strength randaugment \
    --randaugment_n 2 \
    --randaugment_m 7 \
    --lr 0.001 \
    --weight_decay 5e-4 \
    --num_epochs 500 \
    --batch_size 128 \
    --seed 42
```

**预期**: Val F1 = 0.79-0.81  
**优先级**: 🔥🔥🔥🔥

---

## 🚀 推荐执行顺序

### 方案 A: 保守稳健 (串行)

```
今晚: Exp #104b (SD 0.1 + RandAug)  ← 最有希望
  ↓
明早: 查看结果
  ↓
  如果 ≥ 0.80 → ✅ 完成
  如果 < 0.80 → Exp #104a (SD 0.1 + Medium)
```

**时间**: 1-2 天  
**成功概率**: 70%

---

### 方案 B: 快速并行

```
同时运行:
- Exp #104a: SD(0.1) + medium
- Exp #104b: SD(0.1) + randaug(M=9)  
- Exp #104d: SD(0.1) + randaug(M=7)

明早选最佳
```

**时间**: 1 天  
**成功概率**: 85%

---

## 📊 预期效果分析

### 为什么 RandAugment (pure) 可能成功？

**1. 避免重复叠加**

```
之前: RandAug (2 ops) + Medium (6 ops) = 8 ops
现在: RandAug (2 ops) only = 2 ops
```

**2. 自动优化**

- RandAugment 是通过搜索优化的
- 可能比手工设计的 medium 更适合

**3. 文献支持**

- NeurIPS 2020 论文证明在 CIFAR-100 有效
- 很多 SOTA 实现用 RandAugment

---

## ⚖️ 风险评估

| 配置 | 风险 | 缓解 |
|-----|------|------|
| SD(0.1) + medium | 低 | 基于 baseline 微调 |
| SD(0.1) + randaug | 中 | 纯 RA 未测试过 |
| SD(0.15) + medium | 低 | 可能仍偏强 |
| 并行测试 | 低 | 时间成本 |

---

## 🎯 我的最终推荐

### ✅ **优先尝试 Exp #104b (SD + pure RandAugment)**

**理由**:

1. 修复了叠加问题
2. RandAugment 是自动优化的策略
3. 可能带来惊喜

**命令**:

```bash
python main.py \
    --model wide_resnet28_10 \
    --dropout 0.3 \
    --drop_path_rate 0.1 \
    --aug_strength randaugment \
    --randaugment_n 2 \
    --randaugment_m 9 \
    --num_epochs 500 \
    --batch_size 128 \
    --seed 42
```

**如果时间充裕，并行运行**:

- #104b (M=9)
- #104d (M=7)
- #104a (medium aug)

明早选最佳！

---

**GridMask?** 还是不需要，先看这轮结果！
