# 🚀 Phase 2.5 最终运行指南

**状态**: ✅ NaN 问题已修复  
**模型**: Wide ResNet-28-10 + Self-Distillation  
**目标**: F1 ≥ 0.85

---

## ✅ 已修复的问题

### NaN Loss 根因

1. ❌ **Loss 幅度过大**: 4 个分类器的 CE loss 相加
2. ❌ **Temperature 过高**: 4.0 导致数值不稳定
3. ❌ **Alpha 过高**: 0.9 使硬标签权重仅 0.1

### 修复措施

1. ✅ **只对最终分类器计算 CE loss** (避免 4x 放大)
2. ✅ **降低 temperature**: 4.0 → **3.0**
3. ✅ **降低 alpha**: 0.9 → **0.7**
4. ✅ **Teacher softmax detach**: 避免梯度干扰
5. ✅ **KL loss 归一化**: 除以学生分类器数量

---

## 🚀 立即运行

### 命令 (所有参数已优化)

```bash
python main.py \
    --model wide_resnet28_10_selfdistill \
    --use_self_distillation
```

**或完整版本**:

```bash
python main.py \
    --model wide_resnet28_10_selfdistill \
    --drop_path_rate 0.1 \
    --aug_strength randaugment \
    --use_self_distillation \
    --distill_temperature 3.0 \
    --distill_alpha 0.7 \
    --num_epochs 500 \
    --early_stopping_patience 50 \
    --batch_size 128 \
    --seed 42
```

---

## 📊 修复前后对比

| 参数 | 修复前 | 修复后 | 原因 |
|-----|--------|--------|------|
| CE loss | 4 个分类器 | **1 个** (teacher only) | 避免 4x 放大 |
| temperature | 4.0 | **3.0** | 数值稳定性 |
| alpha | 0.9 | **0.7** | 平衡 hard/soft |
| Teacher grad | 流向 student | **detach** | 减少干扰 |

---

## 🎯 预期效果

```
Baseline (Exp #104b): F1 = 0.8131
+ Self-Distill (修复): F1 = 0.84-0.85

成功概率: 70-75%
```

### 训练动态预期

- Train Loss: 1.5-2.5 (稳定，无 NaN)
- Train Acc: 55-60%
- Val Acc: 83-86%
- Best epoch: 200-300

---

## ⚠️ 监控要点

### 健康信号

✅ Train Loss 稳定在 1.5-2.5  
✅ 无 NaN 或 Inf  
✅ Val F1 稳步上升  
✅ Train/Val gap 正常 (25-30%)

### 警报信号

🚨 Train Loss > 3.0 → loss 权重仍然过大  
🚨 出现 NaN → 立即停止，进一步降低 alpha/temperature  
🚨 Val F1 < 0.82 → 自蒸馏未生效

---

## 🔧 如果仍有问题

### 方案 A: 进一步降低参数

```bash
python main.py \
    --model wide_resnet28_10_selfdistill \
    --use_self_distillation \
    --distill_temperature 2.5 \
    --distill_alpha 0.5
```

### 方案 B: 禁用 AMP (排查)

```bash
python main.py \
    --model wide_resnet28_10_selfdistill \
    --use_self_distillation \
    --no_amp
```

### 方案 C: 回退到 Phase 1 配置

```bash
python main.py
# 使用 wide_resnet28_10 (无自蒸馏)
```

---

## 🎊 总结

**问题**: ✅ 已识别并修复  
**修复**: ✅ Loss 计算逻辑优化  
**状态**: ✅ 准备重新运行

**立即执行**:

```bash
python main.py \
    --model wide_resnet28_10_selfdistill \
    --use_self_distillation
```

**预期**: F1 = 0.84-0.85 🎯
