# EfficientNet 快速启动指南

**目标**: 突破 F1 = 0.85  
**方案**: EfficientNet-B0 + 64×64 + 强增强  
**预期**: F1 = 0.85-0.87 ✅

---

## 🚀 立即执行

### 完整训练命令

```bash
python main.py \
    --model efficientnet_b0 \
    --input_size 64 \
    --dropout 0.2 \
    --drop_path_rate 0.2 \
    --aug_strength randaugment \
    --randaugment_n 2 \
    --randaugment_m 10 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --optimizer adamw \
    --scheduler cosine \
    --warmup_epochs 10 \
    --batch_size 96 \
    --num_epochs 600 \
    --use_amp \
    --use_ema \
    --mixup_alpha 0.2 \
    --use_cutmix \
    --cutmix_alpha 0.8 \
    --seed 42
```

**预期时间**: 6-8 小时  
**预期结果**: Test F1 = 0.85-0.87 ✅

---

## 📋 实施步骤

### Step 1: 实现代码 (2-3h)

参考: `bot/experiments/phase3/exp_400_efficientnet_implementation.md`

**关键文件**:

1. `scripts/model_architectures.py` - 添加 EfficientNet
2. `scripts/train_utils.py` - 支持 input_size
3. `main.py` - 添加参数

### Step 2: 测试 (10min)

```bash
# 测试前向传播
python -c "from scripts.model_architectures import efficientnet_b0; import torch; model = efficientnet_b0(input_size=64); print(model(torch.randn(2,3,64,64)).shape)"

# 快速训练测试 (5 epochs)
python main.py --model efficientnet_b0 --input_size 64 --num_epochs 5 --batch_size 64
```

### Step 3: 完整训练 (6-8h)

运行上面的完整命令

### Step 4: 监控

```bash
# 实时查看 Val F1
tail -f cifar_pipeline.log | grep "Val F1"

# 检查点:
# Epoch 50:  Val F1 > 0.70
# Epoch 100: Val F1 > 0.75
# Epoch 200: Val F1 > 0.80
```

---

## 🎯 核心改进点

### 1. 分辨率提升 (关键!)

```
32×32 → 64×64
- 像素: 1024 → 4096 (4×)
- 人类类 F1: 0.63 → 0.72-0.76 (+0.09-0.13)
```

### 2. EfficientNet 架构

- SE block (通道注意力)
- Swish 激活
- Mobile Inverted Bottleneck
- Compound Scaling

### 3. 更强数据增强

- RandAugment M=10 (从 9)
- CutMix α=0.8 (从 0.65)
- Cutout 8-16 (从 4-8)

---

## 📊 预期结果

### 整体指标

```
Test F1:  0.85-0.87 (vs 0.82, +0.03-0.05) ✅
Test Acc: 85-87%    (vs 82%, +3-5%)
```

### 类别提升

```
人类类平均: 0.72-0.76 (vs 0.63, +0.09-0.13)
- boy:   0.68-0.72 (vs 0.59)
- girl:  0.68-0.72 (vs 0.57)
- man:   0.70-0.74 (vs 0.62)
- woman: 0.73-0.77 (vs 0.65)
- baby:  0.78-0.82 (vs 0.72)

小动物类: 0.68-0.72 (vs 0.61, +0.07-0.11)
机械类:   0.93-0.94 (保持)
```

---

## ⚠️ 注意事项

### 显存管理

- batch_size=96 (从 128 降低)
- 64×64 显存占用约 10-12GB
- RTX 5080/4080 Super (16GB): 充足 ✅

### 训练时间

- 64×64 约为 32×32 的 2-4×
- 预计 6-8h < 12h 限制 ✅

### 备选方案 (如果 F1 < 0.85)

1. **96×96 分辨率**: F1 预期 0.86-0.88
2. **EfficientNet-B1**: 更大模型
3. **模型集成**: 3 个 EfficientNet-B0

---

## 🔗 详细文档

- [完整分析](./analysis/efficientnet_breakthrough_strategy.md)
- [实验计划](./experiments/phase3/exp_400_efficientnet_b0_plan.md)
- [实现指南](./experiments/phase3/exp_400_efficientnet_implementation.md)
- [EfficientNet 论文](https://arxiv.org/abs/1905.11946)

---

**准备开始！目标 F1 ≥ 0.85!** 🚀
