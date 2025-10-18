# Experiment #200 - ConvNeXt-Tiny Baseline

**日期**: 2025-10-17  
**阶段**: Phase 2  
**优先级**: 🔥🔥🔥🔥🔥  
**状态**: 🟡 待运行

---

## 🎯 实验目标

验证 ConvNeXt-Tiny 在 CIFAR-100 上的性能，冲击 Phase 2 目标。

**基准 F1**: 0.8131 (Exp #104b - WRN-28-10)  
**Phase 2 目标**: F1 ≥ 0.83  
**预期提升**: +0.015-0.030 F1

---

## 🔧 配置详情

### 模型配置

```python
model = "convnext_tiny"
depths = [3, 3, 9, 3]  # 18 blocks total
dims = [96, 192, 384, 768]
drop_path_rate = 0.1  # Same as Phase 1 best
num_classes = 100
```

**参数量**: ~28M (vs WRN-28-10 36.5M)

### 数据增强

```python
aug_strength = "randaugment"  # Phase 1 最佳
randaugment_n = 2
randaugment_m = 9
mixup_alpha = 0.25
cutmix_alpha = 0.65
```

### 训练超参数

**基于 ConvNeXt 论文调整**:

```python
lr = 0.001  # AdamW (ConvNeXt 论文建议 4e-3，但我们 batch 小)
weight_decay = 0.05  # ConvNeXt 建议 (远大于 Wide ResNet)
optimizer = "adamw"
scheduler = "cosine"
warmup_epochs = 20  # ConvNeXt 建议
num_epochs = 600  # 更长训练
early_stopping_patience = 60
batch_size = 128
```

### 技术栈

- [x] Stochastic Depth (drop_path=0.1)
- [x] RandAugment (N=2, M=9)
- [x] Layer Scale (1e-6)
- [x] Mixup + CutMix
- [x] EMA
- [x] AMP

---

## 📊 运行命令

### PowerShell

```powershell
python main.py `
    --model convnext_tiny `
    --drop_path_rate 0.1 `
    --aug_strength randaugment `
    --lr 0.001 `
    --weight_decay 0.05 `
    --warmup_epochs 20 `
    --num_epochs 600 `
    --early_stopping_patience 60 `
    --batch_size 128 `
    --seed 42
```

### Bash

```bash
python main.py \
    --model convnext_tiny \
    --drop_path_rate 0.1 \
    --aug_strength randaugment \
    --lr 0.001 \
    --weight_decay 0.05 \
    --warmup_epochs 20 \
    --num_epochs 600 \
    --early_stopping_patience 60 \
    --batch_size 128 \
    --seed 42
```

---

## 📈 预期结果

### 基于文献和经验

**ConvNeXt 论文** (CVPR 2022):

- 在 ImageNet 上显著优于 ResNet/Wide ResNet
- 纯卷积架构，易于训练
- 性能接近 ViT

**预期提升**:

```
Baseline (WRN-28-10): 0.8131
+ ConvNeXt 架构优势: +0.015-0.025
+ 更强正则化 (wd=0.05): +0.005-0.010
= 0.833-0.848
```

### 成功标准

| 结果 | 评价 | 后续 |
|-----|------|------|
| F1 ≥ 0.85 | ✅ 满分标准！ | Phase 3 尝试更大模型 |
| F1 ≥ 0.83 | ✅ Phase 2 达成 | 微调冲击 0.85 |
| F1 = 0.82-0.83 | 🟡 接近 | 超参数优化 |
| F1 < 0.82 | ⚠️ 未达预期 | 重新评估 |

---

## 🔍 ConvNeXt vs Wide ResNet

### 架构对比

| 特性 | Wide ResNet | ConvNeXt |
|-----|------------|----------|
| 基础单元 | BasicBlock | ConvNeXt Block |
| 激活函数 | ReLU | GELU |
| 归一化 | BatchNorm | LayerNorm |
| 卷积类型 | 标准 conv | Depthwise + Pointwise |
| 特殊技术 | Dropout | Layer Scale |

### 理论优势

**ConvNeXt**:

1. **更现代**: 2022 vs 2016
2. **更强表达**: Depthwise + Pointwise 分离
3. **更稳定**: LayerNorm + Layer Scale
4. **文献支持**: ImageNet SOTA (纯 CNN)

**预期**: 同等参数下性能更好

---

## 🎯 关键配置说明

### weight_decay = 0.05 (重要!)

**ConvNeXt 论文配置**:

- ImageNet: weight_decay = 0.05
- 远大于 Wide ResNet (5e-4)

**原因**:

- ConvNeXt 容量大，需要更强 L2 正则
- LayerNorm 配合大 WD 效果好

### warmup_epochs = 20

**ConvNeXt 论文建议**:

- Linear warmup 20 epochs
- 帮助稳定初期训练

### num_epochs = 600

**理由**:

- ConvNeXt 收敛可能较慢
- 给足时间充分训练

---

## 📊 预期训练动态

### 与 Wide ResNet 对比

**Wide ResNet**:

- Best epoch: ~200
- Train Acc: 60%
- Val Acc: 81%

**ConvNeXt (预期)**:

- Best epoch: ~250-300
- Train Acc: 55-60% (更强正则化)
- Val Acc: 83-85%

---

## 🏷️ 标签

`phase-2` `convnext-tiny` `modern-cnn` `pending`

---

**准备运行 Phase 2 首个实验！** 🚀
