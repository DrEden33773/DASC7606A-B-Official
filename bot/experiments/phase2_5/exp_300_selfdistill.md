# Experiment #300 - Wide ResNet-28-10 + Self-Distillation (BYOT)

**日期**: 2025-10-17  
**阶段**: Phase 2.5 (Self-Distillation)  
**优先级**: 🔥🔥🔥🔥🔥  
**状态**: 🟡 待运行

---

## 🎯 实验目标

使用自蒸馏 (Be Your Own Teacher) 提升 Wide ResNet-28-10 性能，冲击满分标准。

**基准 F1**: 0.8131 (Exp #104b)  
**Phase 2.5 目标**: F1 ≥ 0.85 (满分)  
**预期提升**: +2.5-3.5% (基于 BYOT 论文)

---

## 🔬 自蒸馏原理

### 架构设计

```
Wide ResNet-28-10 + Self-Distillation:

Input (32×32×3)
  ↓
conv1 (16 channels)
  ↓
layer1 (4 blocks, 160 channels) → Classifier1 (student) → Logits1
  ↓
layer2 (4 blocks, 320 channels) → Classifier2 (student) → Logits2
  ↓
layer3 (4 blocks, 640 channels) → Classifier3 (student) → Logits3
  ↓
Final FC (teacher) → Logits4

训练: 使用所有 4 个分类器 (深层指导浅层)
推理: 只用 Logits4 (中间分类器可移除)
```

### Loss 函数

```python
Loss = Σ[(1-α) * CE(Logitsi, y)]          # Hard labels (所有分类器)
     + Σ[α * KL(Logitsi, Logits4)]        # Soft labels (浅层学习深层)

α = 0.9  (软标签权重，论文建议)
Temperature = 4.0 (知识蒸馏温度)
```

---

## 🔧 配置详情

### 模型配置

```python
model = "wide_resnet28_10_selfdistill"
depth = 28
widen_factor = 10
dropout = 0.3
drop_path_rate = 0.1
num_classes = 100

# 自蒸馏参数
use_self_distillation = True
temperature = 4.0
alpha = 0.9
```

**参数量**: ~39M (vs 36.5M baseline, +2.5M 来自中间分类器)

### 数据增强

```python
aug_strength = "randaugment"
randaugment_n = 2
randaugment_m = 9
mixup_alpha = 0.25
cutmix_alpha = 0.65
```

### 训练超参数

```python
lr = 0.001
weight_decay = 5e-4
optimizer = "adamw"
scheduler = "cosine"
warmup_epochs = 10
num_epochs = 500
batch_size = 128
```

---

## 📊 运行命令

### PowerShell

```powershell
python main.py `
    --model wide_resnet28_10_selfdistill `
    --drop_path_rate 0.1 `
    --aug_strength randaugment `
    --use_self_distillation `
    --distill_temperature 4.0 `
    --distill_alpha 0.9 `
    --lr 0.001 `
    --weight_decay 5e-4 `
    --num_epochs 500 `
    --early_stopping_patience 50 `
    --batch_size 128 `
    --seed 42
```

### Bash

```bash
python main.py \
    --model wide_resnet28_10_selfdistill \
    --drop_path_rate 0.1 \
    --aug_strength randaugment \
    --use_self_distillation \
    --distill_temperature 4.0 \
    --distill_alpha 0.9 \
    --lr 0.001 \
    --weight_decay 5e-4 \
    --num_epochs 500 \
    --early_stopping_patience 50 \
    --batch_size 128 \
    --seed 42
```

---

## 📈 预期结果

### 基于 BYOT 论文

**论文数据** (CIFAR-100):

```
ResNet-32:  72% → 76% (+4%)
ResNet-110: 75% → 78% (+3%)
VGG19:      72.6% → 76.7% (+4.1%)

平均提升: +2.65-4.07%
```

**我们的预期**:

```
WRN-28-10 baseline: 81.31%
+ Self-Distill (+3%): 84.31%
预期 F1: 0.843-0.85 ✅
```

### 成功标准

| 结果 | 评价 | 状态 |
|-----|------|------|
| F1 ≥ 0.85 | ✅ 满分达成！| Phase 2.5 完美收官 |
| F1 = 0.84-0.85 | ✅ 接近满分 | 微调冲击 0.85 |
| F1 = 0.83-0.84 | 🟡 Phase 2 达成 | 继续优化 |
| F1 < 0.83 | ⚠️ 未达预期 | 重新评估 |

---

## 🔍 关键技术细节

### 中间分类器设计

**Bottleneck 结构**:

```python
Conv 1×1 (降维 50%)
→ BatchNorm
→ ReLU
→ Global Average Pooling
→ Flatten
→ Linear (→ 100 classes)
```

**作用**: 减少与主网络的干扰，同时提供分类监督

### Loss 权重分析

**α = 0.9** (论文建议):

```
CE loss (hard labels): 0.1 × 4 = 0.4
KL loss (soft labels): 0.9 × 3 = 2.7

软标签占主导 (87%)
```

**合理性**: 深层分类器提供更丰富的知识

---

## 🎓 优势分析

### vs 继续增大模型 (WRN-28-12)

| 方案 | 参数 | 训练时间 | F1 | 分析 |
|-----|------|---------|-----|------|
| WRN-28-12 | 52.8M | 4h | 0.80 | 过拟合，性能下降 |
| WRN-28-10 + SD | 39M | 4h | 0.84-0.85? | 不增大单模型，利用内部知识 |

**优势**:

- ✅ 避免过拟合（容量适中）
- ✅ 训练时间相同
- ✅ 论文证明有效

### vs 传统蒸馏 (Ensemble → Student)

| 方案 | 训练次数 | 总时间 | 预期 F1 |
|-----|---------|--------|---------|
| 传统蒸馏 | 3 teachers + 1 student | 16h | 0.84-0.86 |
| 自蒸馏 | 1次 | 4h | 0.84-0.85 |

**优势**:

- ✅ 节省 75% 训练时间
- ✅ 一次训练完成

---

## 🔍 预期训练动态

### 健康信号

- Train Acc: 55-60% (自蒸馏增加训练难度)
- Val Acc: 84-86%
- Best epoch: 250-300 (可能略延后)

### 警报信号

- Train Acc < 50% → loss 权重不当
- Val F1 < 0.82 → 自蒸馏未起作用
- 中间分类器 acc 差距过大 → bottleneck 设计问题

---

## 🏷️ 标签

`phase-2.5` `self-distillation` `byot` `wrn-28-10` `pending`

---

## 📝 实验日志

### 2025-10-17 Evening

- ✅ WideResNetSelfDistill 实现完成
- ✅ self_distillation_loss 实现完成
- ✅ train_epoch 修改支持自蒸馏
- ✅ main.py 参数集成
- ✅ Linting 检查通过
- 🟡 待运行完整训练

---

**实验人员**: AI Assistant + User  
**最后更新**: 2025-10-17  
**预计完成**: 明天上午
