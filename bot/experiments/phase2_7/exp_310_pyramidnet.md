# Experiment #310 - PyramidNet-110 (α=270) + SD + RA

**日期**: 2025-10-17  
**阶段**: Phase 2.7 (PyramidNet)  
**优先级**: 🔥🔥🔥🔥🔥  
**状态**: 🟡 待运行

---

## 🎯 实验目标

使用 PyramidNet-110 架构 + Phase 1 最佳优化 (SD + RA)，冲击满分目标。

**基准 F1**: 0.8131 (Exp #104b - WRN-28-10)  
**论文基线**: CIFAR-100 ~83% acc (PyramidNet-110)  
**Phase 2.7 目标**: F1 ≥ 0.85 (满分)  
**预期提升**: +0.015-0.035 F1

---

## 🏗️ PyramidNet 架构特点

### 核心创新

**渐进式通道增长** (vs ResNet/WideResNet 的突然跳跃):

```
ResNet:     64 → 64 → 128 (×2) → 256 (×2) → 512 (×2)
Wide ResNet: 16 → 160 (×10) → 320 (×2) → 640 (×2)

PyramidNet:  16 → 23 → 31 → 38 → 46 → ... → 286
            (每个 block 增加 α/total_blocks = 270/54 = 5 channels)
```

**优势**: 更平滑的特征维度演化，避免信息瓶颈

### PyramidNet-110 规格

```python
Depth: 110 layers (54 blocks)
Alpha: 270 (widening factor)
Initial channels: 16
Final channels: 16 + 270 = 286
Blocks per group: 18-18-18
Parameters: ~26M
```

---

## 🔧 配置详情

### 模型配置

```python
model = "pyramidnet110_270"
depth = 110
alpha = 270
drop_path_rate = 0.1  # Phase 1 最佳
num_classes = 100
```

**参数量**: ~26M (vs WRN 36.5M, 更少但论文证明更强)

### 数据增强

```python
aug_strength = "randaugment"  # Phase 1 最佳
randaugment_n = 2
randaugment_m = 9
mixup_alpha = 0.25
cutmix_alpha = 0.65
```

### 训练超参数

**基于论文 + Phase 1 优化**:

```python
lr = 0.001  # AdamW
weight_decay = 1e-4  # 论文建议 (vs WRN 5e-4)
optimizer = "adamw"
scheduler = "cosine"
warmup_epochs = 10
num_epochs = 600  # 深网络需要更长训练
early_stopping_patience = 60
batch_size = 128
```

---

## 📊 运行命令

### PowerShell

```powershell
python main.py `
    --model pyramidnet110_270 `
    --drop_path_rate 0.1 `
    --aug_strength randaugment `
    --lr 0.001 `
    --weight_decay 1e-4 `
    --warmup_epochs 10 `
    --num_epochs 600 `
    --early_stopping_patience 60 `
    --batch_size 128 `
    --seed 42
```

### Bash

```bash
python main.py \
    --model pyramidnet110_270 \
    --drop_path_rate 0.1 \
    --aug_strength randaugment \
    --lr 0.001 \
    --weight_decay 1e-4 \
    --warmup_epochs 10 \
    --num_epochs 600 \
    --early_stopping_patience 60 \
    --batch_size 128 \
    --seed 42
```

---

## 📈 预期结果

### 基于论文 + Phase 1 优化

**PyramidNet-110 论文结果**:

```
CIFAR-100: 82.99% accuracy (~F1 0.83)
```

**我们的优化**:

```
PyramidNet-110 baseline: ~83% acc
+ Stochastic Depth (0.1): +0.01
+ RandAugment (优化): +0.01-0.02
+ Mixup/CutMix: (已包含在论文中)
= 85-86% acc (F1 0.85-0.86)
```

### 成功标准

| 结果 | 评价 | 后续 |
|-----|------|------|
| F1 ≥ 0.85 | ✅ 满分达成！| 提交准备 |
| F1 = 0.84-0.85 | ✅ 接近满分 | TTA 微调 |
| F1 = 0.83-0.84 | 🟡 Phase 2 达成 | 继续优化 |
| F1 < 0.83 | ⚠️ 未达预期 | Ensemble 保底 |

---

## 🔍 PyramidNet vs WRN-28-10

### 为什么 PyramidNet 可能更好？

**1. 论文证明** (CVPR 2017):

```
PyramidNet-110: 83% acc > WRN-28-10: 81% acc
差距: +2%
```

**2. 架构优势**:

- 渐进式通道增长 → 更平滑的特征演化
- 110 layers (深) → 更强的表达能力
- Zero-padded shortcut → 无参数开销

**3. 参数效率**:

- PyramidNet: 26M → 83% acc
- WRN: 36.5M → 81.31% acc
- PyramidNet 参数效率更高！

---

## ⚖️ 风险评估

| 风险 | 概率 | 缓解措施 |
|-----|------|---------|
| 深网络训练不稳定 | 中 (30%) | Stochastic Depth + Warmup |
| 参数配置需调整 | 低 (15%) | 基于论文配置 |
| 收敛慢 | 中 (25%) | 600 epochs 充足 |

**总体风险**: 🟡 中等，可接受

---

## 🎓 与现有优化的兼容性

### 完全兼容 ✅

- ✅ **Stochastic Depth**: PyramidNet 有残差连接，DropPath 直接适用
- ✅ **RandAugment**: 数据增强，与架构无关
- ✅ **Mixup/CutMix**: Batch-level 增强
- ✅ **EMA**: 适用所有模型
- ✅ **AMP**: 兼容
- ✅ **Gradient Clipping**: 兼容

**无互斥问题！**

---

## 🏷️ 标签

`phase-2.7` `pyramidnet` `pyramidnet-110` `deep-network` `pending`

---

## 📝 实验日志

### 2025-10-17

- ✅ PyramidNet-110 实现完成
- ✅ 集成到 create_model()
- ✅ main.py 参数更新
- ✅ Linting 检查通过
- 🟡 待运行完整训练

---

**实验人员**: AI Assistant + User  
**参考**: [PyramidNet 论文](https://arxiv.org/abs/1610.02915), [官方实现](https://github.com/dyhan0920/PyramidNet-PyTorch)  
**预计完成**: 明天上午
