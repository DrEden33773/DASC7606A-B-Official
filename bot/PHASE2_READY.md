# 🚀 Phase 2 实现完成 - ConvNeXt-Tiny

**完成时间**: 2025-10-17  
**新增模型**: ConvNeXt-Tiny (28M params)  
**目标**: F1 ≥ 0.83 → 0.85

---

## ✅ 实现完成

### ConvNeXt-Tiny 核心组件

**文件**: `scripts/model_architectures.py`

**新增类**:

1. ✅ `LayerNorm2d` - Channels-first LayerNorm
2. ✅ `ConvNeXtBlock` - 核心 building block
3. ✅ `ConvNeXt` - 主模型类
4. ✅ `convnext_tiny()` - 工厂函数 (28M params)
5. ✅ `convnext_small()` - 工厂函数 (50M params, Phase 3)

### 关键特性

**适配 CIFAR (32×32)**:

- Stem: 3×3 conv stride=1 (保持 32×32)
- 4 stages: 32→16→8→4
- Depths: [3, 3, 9, 3] = 18 blocks
- Channels: [96, 192, 384, 768]

**现代技术**:

- Depthwise Separable Conv (7×7)
- LayerNorm (替代 BatchNorm)
- GELU (替代 ReLU)
- Layer Scale (1e-6)
- Stochastic Depth (0.1)

---

## 🎯 Phase 2 目标

```
Phase 1 达成: F1 = 0.8131 ✅
  ↓
Phase 2 目标: F1 ≥ 0.83
  ↓
最终目标: F1 ≥ 0.85 (满分)
```

**预期提升**: +0.02-0.03 F1

---

## 🚀 立即运行 Exp #200

### 完整命令

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

### 关键参数说明

| 参数 | 值 | 说明 |
|-----|-----|------|
| model | convnext_tiny | 28M params, 现代架构 |
| drop_path_rate | 0.1 | Phase 1 最佳值 |
| aug_strength | randaugment | Phase 1 最佳 |
| weight_decay | **0.05** | ConvNeXt 论文建议 (10x!) |
| warmup_epochs | **20** | ConvNeXt 建议 (vs WRN 10) |
| num_epochs | **600** | 更长训练 (vs WRN 500) |

---

## 📊 预期效果

### 保守估计

```
Baseline (WRN-28-10): 0.8131
+ ConvNeXt 架构: +0.015
= 0.8281
```

**评价**: 未达 Phase 2 目标 (0.83)

### 乐观估计

```
Baseline (WRN-28-10): 0.8131
+ ConvNeXt 架构: +0.025
+ 更强正则化 (wd=0.05): +0.005
= 0.8431
```

**评价**: 达成 Phase 2，接近满分 ✨

### 现实估计

```
预期: F1 = 0.83-0.84
成功概率: 70-75%
```

---

## 📋 如果未达标

### 优化方向

**1. 降低 weight_decay** (如果欠拟合)

- 0.05 → 0.03 or 0.02
- ConvNeXt 论文的 0.05 可能对 CIFAR 太强

**2. 调整 drop_path_rate**

- 0.1 → 0.15 (略增)
- 或 0.08 (略减)

**3. 更大模型**

- ConvNeXt-Small (50M params)

**4. 更长训练**

- 600 → 800 epochs

---

## 🎓 ConvNeXt 训练技巧

### 与 Wide ResNet 的不同

| 方面 | Wide ResNet | ConvNeXt |
|-----|-------------|----------|
| weight_decay | 5e-4 | **0.05** (100x!) |
| warmup | 10 epochs | **20 epochs** |
| optimizer | AdamW | AdamW |
| lr | 0.001 | 0.001 (已调整) |

### 训练建议

1. **耐心等待**: ConvNeXt 可能收敛较慢
2. **监控 Val F1**: 应该稳步上升
3. **Train/Val gap**: 可能更大（wd=0.05 很强）

---

## 🔗 相关资源

### 论文

- "A ConvNet for the 2020s" (CVPR 2022)
- <https://arxiv.org/abs/2201.03545>

### 官方实现

- <https://github.com/facebookresearch/ConvNeXt>

---

**准备就绪！开始 Phase 2 训练！** 🚀

**预计时间**: 5-6 小时 (600 epochs)  
**预计完成**: 明天上午
