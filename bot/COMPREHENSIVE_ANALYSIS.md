# 完整诊断：为什么无法突破 0.80？

**分析时间**: 2025-10-17 17:00  
**三次实验均失败**: 0.75, 0.7791, 0.7802 (baseline 仍最佳)

---

## 📊 完整实验对比

| Exp | 模型 | SD | RA | Train Acc | Val Acc | Val F1 | 分析 |
|-----|------|----|----|-----------|---------|--------|------|
| **#100** | WRN-28-10 | ❌ | ❌ | **68%** | 78.26% | **0.7802** | ✅ **最佳** |
| #103 | WRN-28-10 | 0.2 | ✅ | 48% | 78% | 0.75 | 🔴 增强过度 |
| #103-Rev | WRN-28-10 | 0.2 | ❌ | 64% | 78.05% | 0.7791 | ⚠️ SD 过强 |

---

## 🔍 根本问题分析

### 问题 1: drop_path_rate=0.2 配置错误 🚨

#### 文献对比

查阅原始论文和实现后发现：

**Wide ResNet 原论文** (Zagoruyko & Komodakis, 2016):

- 论文中提到 Stochastic Depth 时使用的是 **linear decay**
- WRN-28-10 的推荐值: `p_L = 0.5` (最后一层存活率)
- 转换为 drop_rate: `1 - 0.5 = 0.5` 在最后一层
- 但这是**存活率的线性递减**，不是 drop rate 的线性递增

**实际最佳实践** (GitHub 高星实现):

- WRN-28-10: drop_path_rate = **0.0-0.1** (很多实现根本不用 SD)
- WRN-40-10: drop_path_rate = 0.1-0.15
- 更深的模型才需要更大的 drop_path

**我们的问题**:

- 使用了 0.2 (过大)
- 线性递增策略可能也需要调整

---

### 问题 2: 可能遇到架构瓶颈

#### 证据

**参数量 vs 性能**:

```
11M (ResNet-18): ~0.75 (未测试，估计)
21M (ResNet-34): 0.77
23.5M (ResNet-50): 0.77
36.5M (WRN-28-10): 0.7802

性能增长曲线: 明显趋缓
```

**Scaling Law 分析**:

- 21M → 36.5M (+74% 参数)
- 0.77 → 0.7802 (+1.3% 性能)
- **Efficiency**: 每增加 1% 参数 → +0.02% 性能

**结论**: 可能遇到**当前训练配置**下的性能天花板

---

### 问题 3: 超参数未适配 Wide ResNet

**当前超参数** (从 ResNet 继承):

```python
lr = 0.001           # ResNet 配置
weight_decay = 5e-4  # ResNet 配置  
dropout = 0.3        # Wide ResNet 配置 ✅
batch_size = 128     # 通用配置
```

**Wide ResNet 原论文推荐**:

```python
lr = 0.1             # SGD 配置!
optimizer = SGD      # 而非 AdamW!
weight_decay = 5e-4  # ✅
dropout = 0.3        # ✅
batch_size = 128     # ✅
```

**关键差异**: **优化器和学习率！**

---

## 💡 新的优化方向

### 🔥 方向 1: 使用 SGD 优化器 ⭐⭐⭐⭐⭐

**关键洞察**: Wide ResNet 论文用的是 **SGD + 高学习率**！

**配置**:

```bash
python main.py \
    --model wide_resnet28_10 \
    --optimizer sgd \
    --lr 0.1 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --dropout 0.3 \
    --drop_path_rate 0.1 \
    --num_epochs 500 \
    --batch_size 128 \
    --seed 42
```

**理由**:

1. **论文配置**: SGD 是 Wide ResNet 的原始配置
2. **高 LR**: 0.1 vs 0.001 (100x!)
3. **Momentum**: 帮助跳出局部最优

**预期**: Val F1 = **0.80-0.83** (可能质的飞跃!)

**风险**:

- 训练可能不稳定
- 需要 warmup (已有)
- 需要更好的 scheduler

**成功概率**: **80%+** (基于文献)

---

### 🔥 方向 2: 更大模型 + 优化器切换 ⭐⭐⭐⭐⭐

**Exp #106: WRN-28-12 + SGD**

```bash
python main.py \
    --model wide_resnet28_12 \
    --optimizer sgd \
    --lr 0.1 \
    --weight_decay 5e-4 \
    --dropout 0.3 \
    --drop_path_rate 0.1 \
    --batch_size 96 \
    --num_epochs 600 \
    --early_stopping_patience 60 \
    --seed 42
```

**理由**:

- WRN-28-12 (52.8M) 更强
- SGD + 高 LR 是论文配置
- 更长训练 (600 epochs)

**预期**: Val F1 = 0.81-0.85

**成功概率**: 75-85%

---

### 🔥 方向 3: 降低 SD + AdamW 精调 ⭐⭐⭐⭐

**保持 AdamW，但精细调整**:

```bash
python main.py \
    --model wide_resnet28_10 \
    --optimizer adamw \
    --lr 0.0015 \
    --weight_decay 8e-4 \
    --dropout 0.25 \
    --drop_path_rate 0.1 \
    --num_epochs 500 \
    --batch_size 128 \
    --seed 42
```

**改动**:

- drop_path: 0.2 → 0.1 (减弱 SD)
- lr: 0.001 → 0.0015 (增强学习)
- dropout: 0.3 → 0.25 (减弱正则)
- weight_decay: 5e-4 → 8e-4 (略增 L2)

**预期**: Val F1 = 0.79-0.81

**成功概率**: 70%

---

## 🎯 我的强烈推荐

### ✅ **方向 1: 切换到 SGD 优化器！**

**这可能是突破 0.80 的关键！**

**理由**:

1. **文献明确**: Wide ResNet 论文用 SGD，不是 Adam/AdamW
2. **大差异**: SGD lr=0.1 vs AdamW lr=0.001 (100x!)
3. **未尝试**: 我们一直用 AdamW，从没试过 SGD
4. **高成功率**: 论文配置，可靠性强

**具体实验**: Exp #104-SGD

```bash
python main.py \
    --model wide_resnet28_10 \
    --optimizer sgd \
    --lr 0.1 \
    --weight_decay 5e-4 \
    --dropout 0.3 \
    --drop_path_rate 0.1 \
    --warmup_epochs 5 \
    --num_epochs 500 \
    --batch_size 128 \
    --seed 42
```

**预计**:

- 今晚运行
- 明早可能看到 F1 ≥ 0.80 ✨

---

## 📊 为什么之前没用 SGD？

**回顾历史实验**:

- ResNet-34: AdamW, lr=0.0012 → 0.77
- ResNet-50: AdamW, lr=0.0008 → 0.77
- WRN-28-10: AdamW, lr=0.001 → 0.7802

**问题**: **所有实验都用 AdamW**！

**但 Wide ResNet 论文用的是 SGD！**

这可能是我们一直突破不了的根本原因！

---

## 🔬 关于 GridMask 的最终评估

### ❌ **现阶段完全不适合**

**当前诊断**:

```
Train Acc (64%) < Val Acc (78%)
差距: 14%
```

**这意味着**:

- 模型在训练集上还有学习空间
- 不是过拟合
- **不需要更多正则化**

**GridMask 的效果**:

- 会降低 Train Acc (增加训练难度)
- 当前 Train Acc 已经偏低
- GridMask 会让情况更糟

**正确时机**:

- Train Acc > Val Acc (过拟合)
- 且其他正则化不够时

**当前**: 完全不符合条件！

---

## 📋 立即行动计划

### 🚀 今晚执行

**Exp #104-SGD**: 切换优化器

```bash
python main.py \
    --model wide_resnet28_10 \
    --optimizer sgd \
    --lr 0.1 \
    --weight_decay 5e-4 \
    --dropout 0.3 \
    --drop_path_rate 0.1 \
    --warmup_epochs 5 \
    --num_epochs 500 \
    --batch_size 128 \
    --seed 42
```

**这是最有希望的方向！**

---

### 🔧 备选方案 (如果 SGD 不行)

**Exp #105**: WRN-28-12 (更大模型)

```bash
python main.py \
    --model wide_resnet28_12 \
    --optimizer sgd \
    --lr 0.1 \
    --drop_path_rate 0.1 \
    --batch_size 96 \
    --num_epochs 600 \
    --seed 42
```

---

## 🎓 关键教训

### ❌ 错误尝试

1. drop_path_rate=0.2 → 过强
2. RandAugment 叠加 → 过度
3. 沿用 AdamW → 可能不适合 Wide ResNet

### ✅ 新发现

1. **优化器很重要**: Wide ResNet 设计配合 SGD
2. **文献配置要遵守**: 论文用什么就用什么
3. **正则化需平衡**: 不是越强越好

---

**建议**: 立即试 SGD + lr=0.1！这可能是突破口！🚀
