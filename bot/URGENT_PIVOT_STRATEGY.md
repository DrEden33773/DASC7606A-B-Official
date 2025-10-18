# 🚨 紧急策略转向：放弃 ConvNeXt

**时间**: 2025-10-17  
**问题**: ConvNeXt F1=0.79，不如 WRN-28-10 (0.8131)  
**决策**: 立即转向 Wide ResNet-28-12

---

## 📊 失败数据

```
Exp #104b (WRN-28-10):  F1 = 0.8131 ✅
Exp #200 (ConvNeXt):    F1 = 0.79   🔴 (-0.023)

性能倒退！
训练时间 +50%！
```

---

## 🔍 根因分析

### 1. weight_decay = 0.05 严重不适合 CIFAR

**数据规模对比**:

```
ImageNet: 1.28M images → wd=0.05 ✅
CIFAR-100: 50k images → wd=0.05 ❌ (数据少 26x)

建议 wd: 0.05 / 26 ≈ 0.002
```

**证据**:

- Train Acc 66% (vs WRN 60%) → 正则不够
- Best epoch 350 (vs WRN 209) → 收敛慢
- 完全没有 SD + RA 的提升

### 2. ConvNeXt 不适合 32×32 小图

**设计背景**: ImageNet 224×224  
**CIFAR**: 32×32 (小 7.7x)

**问题**:

- 7×7 conv 覆盖 22% 图像（太大）
- 4 次下采样 → 最后仅 4×4 (信息丢失)

### 3. 训练时间增加 50%

**原因**: Permute + LayerNorm + GELU  
**是否值得**: ❌ 性能还下降了

---

## 🎯 立即行动

### ✅ **Exp #205: Wide ResNet-28-12**

**配置**:

```bash
python main.py \
    --model wide_resnet28_12 \
    --drop_path_rate 0.12 \
    --aug_strength randaugment \
    --dropout 0.35 \
    --lr 0.0012 \
    --weight_decay 8e-4 \
    --warmup_epochs 10 \
    --num_epochs 600 \
    --early_stopping_patience 60 \
    --batch_size 96 \
    --seed 42
```

**关键参数**:

- model: **wide_resnet28_12** (52.8M, +45%)
- dropout: **0.35** (vs 0.3, 更大模型需更强正则)
- drop_path: **0.12** (vs 0.1, 略增)
- lr: **0.0012** (vs 0.001, 略增)
- batch_size: **96** (vs 128, 显存限制)

**预期**: F1 = **0.82-0.84**  
**成功概率**: **80-85%**

---

## 📊 为什么 WRN-28-12 应该成功？

### 逻辑链

```
WRN-28-10 (36.5M):
+ SD(0.1) + RA
= F1 0.8131

WRN-28-12 (52.8M, +45% 参数):
+ SD(0.12) + RA  
+ dropout 0.35 (防止过拟合)
+ lr 0.0012 (充分学习)
= F1 0.82-0.84 (预期)
```

### 文献支持

**Wide ResNet 论文**:

- WRN-28-10: CIFAR-100 ~81-82%
- WRN-28-12: 未明确，但理论上更好

**Scaling law**:

- 参数 +45% → 性能 +1-2%
- F1 0.8131 + 0.01-0.02 = 0.82-0.83

---

## ⚠️ 风险评估

### WRN-28-12 风险

| 风险 | 概率 | 缓解 |
|-----|------|------|
| OOM | 中 (30%) | batch_size=96 |
| 过拟合 | 低 (10%) | dropout=0.35, drop_path=0.12 |
| 性能不提升 | 低 (15%) | 参数量在那 |

**总体风险**: 🟡 中等，可控

---

## 🔄 ConvNeXt 是否放弃？

### 短期 (Phase 2): **是，放弃**

**理由**:

- 调参成本高
- 成功概率低 (60%)
- 训练时间长

### 长期 (Phase 3): **可能重试**

**条件**: WRN-28-12 达到 0.83+ 后

**配置**: wd=0.002-0.005 (大幅降低)

---

## 📋 行动计划

### 今晚

**Exp #205: WRN-28-12**

```bash
python main.py \
    --model wide_resnet28_12 \
    --drop_path_rate 0.12 \
    --dropout 0.35 \
    --lr 0.0012 \
    --weight_decay 8e-4 \
    --batch_size 96 \
    --num_epochs 600 \
    --seed 42
```

**预计**: 4-5 小时  
**预期**: F1 = 0.82-0.84

### 明天（如果 #205 < 0.83）

**Exp #206: WRN-40-10 或继续微调**

---

## 🎓 关键教训

### ❌ ConvNeXt 失败原因

1. **盲目照搬论文配置** (wd=0.05)
2. **忽视数据集差异** (ImageNet vs CIFAR)
3. **架构不适配** (224×224 vs 32×32)

### ✅ Wide ResNet 成功原因

1. **为 CIFAR 设计** (32×32 原生支持)
2. **配置经过验证** (Phase 1 成功)
3. **架构简单高效** (训练快)

---

**建议**: 立即运行 WRN-28-12！这是达到 0.83+ 的最佳路径！🚀
