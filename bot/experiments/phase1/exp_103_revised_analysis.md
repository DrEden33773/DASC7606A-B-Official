# Exp #103-Revised (SD-only) 深度分析

**实验**: Wide ResNet-28-10 + Stochastic Depth (drop_path=0.2)  
**结果**: Val F1 = **0.7791** (vs Baseline 0.7802, **-0.0011**)  
**状态**: 🔴 **失败 - Stochastic Depth 未带来提升**

---

## 📊 关键数据对比

### 三个实验的详细对比

| 实验 | 配置 | Train Acc | Val Acc | Val F1 | T/V Gap | 状态 |
|-----|------|-----------|---------|--------|---------|------|
| **Exp #100** | WRN baseline | **68%** | 78.26% | **0.7802** | -10% | ✅ 基准 |
| Exp #103 (RA+SD) | + RandAug + SD | 48% | 78% | 0.75 | -30% | 🔴 崩溃 |
| **Exp #103-Rev** | + SD only | **64%** | 78.05% | **0.7791** | -14% | ⚠️ 无效 |

---

## 🔍 核心发现

### 发现 1: Stochastic Depth 正则化过强 🚨

**证据**:

```
Train Acc: 68% (baseline) → 64% (+ SD) = -4%
Val Acc:   78.26% → 78.05% = -0.21%
Val F1:    0.7802 → 0.7791 = -0.0011
```

**分析**:

- SD 成功降低了 Train Acc（正则化作用）
- 但 Val 性能**没有提升反而微降**
- → SD (drop_path_rate=0.2) **太强了**！

**正确的 SD 应该**:

- Train Acc ↓ (正则化)
- Val F1 ↑ (泛化提升)

**当前情况**:

- Train Acc ↓ ✅
- Val F1 ↔ 或 ↓ ❌

**结论**: 正则化过强 → 模型学习能力受限 → 欠拟合

---

### 发现 2: 类别分布几乎完全相同

**对比 Exp #100 vs Exp #103-Revised**:

| 类别 | Exp #100 F1 | Exp #103-Rev F1 | 变化 |
|-----|-------------|-----------------|------|
| boy | 0.51 | 0.51 | 0.00 |
| girl | 0.57 | 0.57 | 0.00 |
| otter | 0.52 | 0.52 | 0.00 |
| woman | 0.62 | 0.62 | 0.00 |
| seal | 0.54 | 0.54 | 0.00 |

**几乎所有类别的 F1 都一模一样！**

**结论**: SD 对所有类别的影响是**均匀的**，没有特别帮助困难类别

---

### 发现 3: Best Epoch 延后了

```
Exp #100: Best Epoch = 152
Exp #103-Rev: Best Epoch = 190 (+38 epochs)
```

**分析**:

- SD 减缓了学习速度
- 需要更多 epoch 才能收敛
- 但最终性能没变

---

## 🎯 根本问题诊断

### 问题: drop_path_rate=0.2 过强

**文献中的推荐值**:

- Wide ResNet-28-10: drop_path_rate = **0.1-0.15**
- Wide ResNet-40-10: drop_path_rate = 0.2-0.3

**我们用的**: 0.2 (可能偏高)

**证据**:

- Train Acc 下降 4% (68% → 64%)
- 学习速度变慢
- 无泛化提升

**诊断**: **正则化过头了，模型容量被过度限制**

---

## 💡 优化方案分析

### 方案 A: 降低 drop_path_rate ⭐⭐⭐⭐⭐

**配置**: drop_path_rate = 0.2 → **0.1**

**理由**:

1. 文献建议 WRN-28-10 用 0.1-0.15
2. 0.2 对这个模型太强
3. 降到 0.1 可以保留 SD 的好处，避免过强

**预期**:

- Train Acc 恢复到 ~66-67%
- Val F1 提升到 0.785-0.795

**实验**: Exp #104a  
**优先级**: 🔥🔥🔥🔥🔥  
**成功概率**: 70%

---

### 方案 B: 增强模型学习能力 ⭐⭐⭐⭐

**配置**: 在 SD (0.1) 基础上：

- 增大学习率: lr = 0.001 → **0.0012**
- 或减小 weight decay: 5e-4 → **3e-4**

**理由**:

- 当前可能欠拟合（Train Acc 只有 64%）
- 更大的 LR 帮助模型学得更充分
- 更小的 WD 减少正则化压力

**预期**: Train Acc ↑, Val F1 ↑

**实验**: Exp #104b  
**优先级**: 🔥🔥🔥🔥

---

### 方案 C: 更宽的模型 ⭐⭐⭐⭐

**配置**: Wide ResNet-28-**12** (vs 当前 28-10)

**理由**:

- 更宽的通道 (192/384/768 vs 160/320/640)
- 参数量 52.8M (vs 36.5M, +45%)
- 更强的表达能力可能抵消 SD 的限制

**预期**: Val F1 = 0.79-0.81

**实验**: Exp #105  
**优先级**: 🔥🔥🔥

---

### 方案 D: 组合优化 ⭐⭐⭐⭐⭐

**最佳组合**:

```bash
python main.py \
    --model wide_resnet28_10 \
    --dropout 0.3 \
    --drop_path_rate 0.1 \      # ← 降低 (0.2 → 0.1)
    --lr 0.0012 \               # ← 提高 (0.001 → 0.0012)
    --weight_decay 5e-4 \
    --num_epochs 500 \
    --batch_size 128 \
    --seed 42
```

**预期**: 平衡正则化和学习能力  
**Val F1**: 0.795-0.805  
**成功概率**: 75%+

---

## 🎨 关于 GridMask

### ❌ **仍然不建议使用 GridMask**

**原因更新**:

1. **当前不是增强不足的问题**
   - Train Acc (64%) < Val Acc (78%)
   - 差距 14% 是正常范围
   - 不需要更多增强

2. **核心问题是正则化过强**
   - 需要**减弱**正则化，而非加强
   - GridMask = 增加正则化
   - 方向错误！

3. **更有效的方向**
   - 调整已有参数（drop_path_rate, lr, dropout）
   - 比引入新技术更稳健

---

## 📋 推荐执行顺序

### 🥇 **优先级 1: Exp #104a - 降低 drop_path_rate**

```bash
python main.py \
    --model wide_resnet28_10 \
    --drop_path_rate 0.1 \
    --seed 42
```

**理由**: 最小改动，最可能成功  
**预期**: Val F1 = 0.785-0.795  
**时间**: 今晚

---

### 🥈 **优先级 2: Exp #104b - SD + 更大 LR**

```bash
python main.py \
    --model wide_resnet28_10 \
    --drop_path_rate 0.1 \
    --lr 0.0012 \
    --seed 42
```

**理由**: 如果 #104a < 0.80，增强学习  
**预期**: Val F1 = 0.795-0.805  
**时间**: 明晚

---

### 🥉 **优先级 3: Exp #104c - SD + Dropout 调低**

```bash
python main.py \
    --model wide_resnet28_10 \
    --drop_path_rate 0.1 \
    --dropout 0.25 \
    --seed 42
```

**理由**: 进一步减弱正则化  
**预期**: Val F1 = 0.79-0.80

---

## 🎓 关键洞察

### Wide ResNet + SD 的微妙平衡

```
drop_path_rate 太小 (0.0):
  → Train ↑, Val ↔ → 过拟合

drop_path_rate 适中 (0.1):
  → Train ↓, Val ↑ → 最优!

drop_path_rate 过大 (0.2):
  → Train ↓↓, Val ↔ → 欠拟合 ← 当前状态
```

**我们现在在"过大"区域！**

---

## 🎯 修正后的预期

### 悲观路径

```
Exp #104a (drop_path=0.1): 0.785
  ↓
Exp #104b (+ lr=0.0012):  0.795
  ↓
Exp #104c (+ dropout=0.25): 0.80 ✅
```

### 乐观路径

```
Exp #104a (drop_path=0.1): 0.795
  ↓
Exp #104b (+ lr=0.0012):  0.805 ✅
```

### 最快路径

```
Exp #104d (drop_path=0.1 + lr=0.0012 + dropout=0.25): 0.80-0.82 ✅
```

**建议**: 先试 #104a（保守），不行再组合优化

---

## 📊 不再考虑的方案

### ❌ GridMask

- 当前是欠拟合（train acc 太低）
- GridMask 会让 train acc 更低
- 方向错误

### ❌ RandAugment (当前实现)

- 叠加模式导致过度增强
- 需要重构为替代模式
- 优先级低于调参

### ❌ 更大的 drop_path_rate

- 0.2 已经太强
- 0.3 会更差

---

## ✅ 立即行动

**运行 Exp #104a**:

```bash
python main.py \
    --model wide_resnet28_10 \
    --dropout 0.3 \
    --drop_path_rate 0.1 \
    --no_randaugment \
    --num_epochs 500 \
    --batch_size 128 \
    --seed 42
```

**预期明早**: Val F1 = 0.785-0.795  
**如果 < 0.80**: 继续 Exp #104b (增大 LR)

---

**关键结论**: drop_path_rate=0.2 过强，降到 0.1 试试！
