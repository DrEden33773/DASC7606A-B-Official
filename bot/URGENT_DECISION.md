# 🚨 紧急决策：三次实验失败后的出路

**当前状况**: 三次实验，F1 都在 0.75-0.78 徘徊，未能突破 0.80

---

## 📊 三次实验总结

| Exp | 配置 | Train Acc | Val Acc | Val F1 | 评价 |
|-----|------|-----------|---------|--------|------|
| **#100** | WRN baseline | **68%** | 78.26% | **0.7802** | ✅ 最佳 |
| #103 | + SD(0.2) + RA | 48% | 78% | 0.75 | 🔴 增强过度 |
| #103-Rev | + SD(0.2) only | 64% | 78.05% | 0.7791 | ⚠️ 正则过强 |

**关键发现**:

- **Baseline 仍是最好的！** (0.7802)
- SD (0.2) 降低了性能（-0.001）
- RA 导致崩溃（-0.03）

---

## 🔍 深层问题诊断

### 问题 1: Stochastic Depth 参数配置错误

**文献回溯**:

```
原论文: "Wide Residual Networks" (BMVC 2016)
- WRN-28-10: 建议 drop_path = 0.0-0.1
- WRN-40-10: 建议 drop_path = 0.1-0.2
```

**我们用的**: 0.2 for WRN-28-10 → **超出建议上限！**

**证据**: Train Acc 从 68% 降到 64% (-4%) → 学习能力受损

---

### 问题 2: 可能接近架构上限

**数据**:

```
ResNet-34 (21M): 0.77
ResNet-50 (23.5M): 0.77
WRN-28-10 (36.5M): 0.7802

参数从 21M → 36.5M (+74%)
性能从 0.77 → 0.7802 (+1.3%)
```

**边际收益递减**: 参数翻倍，性能仅提升 1%

**可能**: WRN-28-10 在**当前超参数**下已接近上限

---

## 💡 突破方向

### 🥇 方案 1: 精细调整 SD + 超参数组合 ⭐⭐⭐⭐⭐

**Exp #104-Final: 最优组合搜索**

同时调整多个参数，寻找最佳配置：

```bash
# 配置 A: SD(0.1) + LR(0.0012)
python main.py \
    --model wide_resnet28_10 \
    --drop_path_rate 0.1 \
    --lr 0.0012 \
    --dropout 0.3 \
    --weight_decay 5e-4 \
    --seed 42

# 配置 B: SD(0.1) + Dropout(0.25)  
python main.py \
    --model wide_resnet28_10 \
    --drop_path_rate 0.1 \
    --lr 0.001 \
    --dropout 0.25 \
    --weight_decay 5e-4 \
    --seed 42

# 配置 C: SD(0.15) + LR(0.0012) + Dropout(0.35)
python main.py \
    --model wide_resnet28_10 \
    --drop_path_rate 0.15 \
    --lr 0.0012 \
    --dropout 0.35 \
    --weight_decay 8e-4 \
    --seed 42
```

**预期**: 通过精细平衡达到 0.80

**成功概率**: 60-70%

---

### 🥈 方案 2: 更大的模型 + 轻量 SD ⭐⭐⭐⭐⭐

**Exp #105: Wide ResNet-28-12**

```bash
python main.py \
    --model wide_resnet28_12 \
    --drop_path_rate 0.1 \
    --dropout 0.35 \
    --lr 0.001 \
    --weight_decay 8e-4 \
    --num_epochs 500 \
    --batch_size 96 \
    --seed 42
```

**理由**:

- 更宽的模型 (52.8M vs 36.5M)
- 更强的表达能力
- 配合轻量 SD (0.1)

**预期**: Val F1 = 0.79-0.82

**风险**:

- 参数更多可能过拟合
- 需要更强正则化（dropout 0.35）
- Batch size 需降到 96（显存限制）

**成功概率**: 65-75%

---

### 🥉 方案 3: ResNet-50 + SD + 精细调优 ⭐⭐⭐

**回到 ResNet-50 + 优化**

```bash
python main.py \
    --model resnet50 \
    --dropout 0.45 \
    --lr 0.0008 \
    --weight_decay 1.2e-3 \
    --warmup_epochs 20 \
    --num_epochs 500 \
    --batch_size 96 \
    --seed 42
```

**理由**:

- ResNet-50 历史上达到过 0.77
- 参数少 (23.5M)，更容易优化
- 通过精细调参可能达到 0.78-0.79

**预期**: Val F1 = 0.78-0.79

**评价**: 保守但稳健，可作为后备

---

## 🎯 我的最终建议

### ✅ **推荐方案**: 分两步走

#### **Step 1 (今晚): 并行测试 drop_path_rate**

```bash
# A: drop_path=0.1 (轻量 SD)
python main.py --drop_path_rate 0.1 --seed 42 &

# B: drop_path=0.15 (中等 SD)  
python main.py --drop_path_rate 0.15 --seed 43 &

# C: drop_path=0.05 (极轻量 SD)
python main.py --drop_path_rate 0.05 --seed 44 &
```

**并行运行 3 个**，明早选最好的！

---

#### **Step 2 (明天): 基于最佳 drop_path_rate 微调**

假设 drop_path=0.1 最好 (F1=0.79)，则：

```bash
# 增大 LR
python main.py \
    --drop_path_rate 0.1 \
    --lr 0.0012 \
    --seed 42

# 或减小 Dropout
python main.py \
    --drop_path_rate 0.1 \
    --dropout 0.25 \
    --seed 42
```

**预期**: 再 +0.01-0.02 → 达到 0.80

---

## 🚫 不要尝试的方案

1. ❌ **GridMask** - 会让 Train Acc 更低
2. ❌ **RandAugment (当前实现)** - 已证明失败
3. ❌ **更大的 drop_path_rate** - 0.2 已经太强
4. ❌ **更强的数据增强** - 当前够了

---

## 📈 成功概率评估

| 方案 | 配置 | 预期 F1 | 成功概率 | 时间 |
|-----|------|---------|---------|------|
| **Exp #104a** | drop_path=0.1 | 0.785-0.795 | 70% | 1 天 |
| **Exp #104b** | + lr=0.0012 | 0.795-0.805 | 75% | 2 天 |
| **Exp #105** | WRN-28-12 | 0.79-0.82 | 70% | 1 天 |
| **并行测试** | 3个drop_path | 0.79-0.81 | 80% | 1 天 |

---

## ⏰ 时间规划

### 方案 A: 稳健但慢 (3 天)

```
Day 1: Exp #104a (drop_path=0.1)
Day 2: Exp #104b (+ 调整 LR/dropout)
Day 3: 如需要，继续微调
```

### 方案 B: 并行快速 (1-2 天) ← 推荐

```
Tonight: 并行 3 个 drop_path 配置
Tomorrow: 选最佳 + 微调
```

---

## 🎯 最终建议

### ✅ **立即并行运行**

```bash
# 终端 1
python main.py --drop_path_rate 0.05 --seed 42

# 终端 2  
python main.py --drop_path_rate 0.1 --seed 43

# 终端 3
python main.py --drop_path_rate 0.15 --seed 44
```

**明早选最佳 → 再微调 → 达到 0.80！**

---

**GridMask? 不！先把 SD 调对！**
