# Exp #400 - EfficientNet-B0 突破 0.85

**实验日期**: 2025-01-26  
**实验阶段**: Phase 3 - 最终冲刺  
**目标**: Test F1 ≥ 0.85  
**策略**: EfficientNet-B0 + 64×64 分辨率 + 强数据增强

---

## 🎯 实验目标

### 主要目标

- **Test F1 ≥ 0.85** (满分标准)
- 人类类 F1: 0.72-0.76 (当前 0.63, 提升 +0.09-0.13)
- 小动物类 F1: 0.68-0.72 (当前 0.61, 提升 +0.07-0.11)

### 次要目标

- 训练时间 < 12h (预计 6-8h)
- 参数量 < 10M (EfficientNet-B0: 5.3M ✅)
- 显存占用 < 14GB (batch_size=96 ✅)

---

## 📊 实验假设

### 核心假设

**H1**: 分辨率提升 (32→64) 能显著改善细节类别性能

- **预期**: 人类类 F1 从 0.63 提升到 0.72-0.76 (+0.09-0.13)
- **依据**: 64×64 = 4× 像素，面部细节从 10×10px → 20×20px

**H2**: EfficientNet 架构优于 Wide ResNet

- **预期**: 相同分辨率下，EfficientNet F1 高 +0.01-0.02
- **依据**: SE block, Swish, Compound Scaling

**H3**: 强数据增强在大分辨率下更有效

- **预期**: RandAugment M=10 (vs M=9) 提升 +0.01-0.02 F1
- **依据**: 更大图像需要更强正则化

### 风险假设

**R1**: 从头训练 EfficientNet 可能不稳定

- **缓解**: 使用 He 初始化 + Warmup + EMA

**R2**: 64×64 训练时间可能超出预算

- **缓解**: 预计 6-8h < 12h 限制

**R3**: 显存可能不足 (batch_size=128, 64×64)

- **缓解**: 降低到 batch_size=96

---

## ⚙️ 实验配置

### 模型配置

```python
model_type = "efficientnet_b0"
num_classes = 100
input_size = 64  # 关键: 从 32 提升到 64
dropout_rate = 0.2
drop_path_rate = 0.2  # Stochastic Depth

# 架构参数
width_mult = 1.0  # B0 配置
depth_mult = 1.0  # B0 配置
# 预期参数量: 5.3M
```

### 数据增强配置

```python
# Resize (关键改动)
input_size = 64  # 32×32 → 64×64

# RandAugment (增强版)
randaugment_n = 2
randaugment_m = 10  # 从 9 提升到 10

# Mixup/CutMix
mixup_alpha = 0.2  # 略降低 (从 0.25)
use_cutmix = True
cutmix_alpha = 0.8  # 提高 (从 0.65)

# CoarseDropout (适配 64×64)
hole_height_range = (8, 16)  # 从 (4, 8) 增大
hole_width_range = (8, 16)
dropout_prob = 0.5
```

### 训练配置

```python
# 优化器
optimizer = "adamw"
lr = 0.001
weight_decay = 1e-5  # EfficientNet 推荐更小的 WD

# 调度器
scheduler = "cosine"
warmup_epochs = 10
num_epochs = 600

# Batch size (关键调整)
batch_size = 96  # 从 128 降低 (64×64 显存需求大)

# 训练技巧
use_amp = True  # 混合精度
use_ema = True  # EMA
ema_decay = 0.9999
max_grad_norm = 1.0  # 梯度裁剪

# 早停
early_stopping_patience = 60
early_stopping_min_delta = 0.001
early_stopping_warmup = 50
```

### 硬件配置

```python
device = "cuda"
num_workers = 4
seed = 42
```

---

## 🚀 执行命令

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
    --early_stopping_patience 60 \
    --use_amp \
    --use_ema \
    --ema_decay 0.9999 \
    --max_grad_norm 1.0 \
    --mixup_alpha 0.2 \
    --use_cutmix \
    --cutmix_alpha 0.8 \
    --seed 42
```

### 快速测试命令 (5 epochs)

```bash
python main.py \
    --model efficientnet_b0 \
    --input_size 64 \
    --batch_size 64 \
    --num_epochs 5 \
    --seed 42
```

---

## 📋 实施步骤

### Phase 1: 代码实现 (2-3h)

**Step 1.1: 实现 EfficientNet-B0**

- [ ] 创建 `Swish` 激活函数
- [ ] 实现 `SEBlock` (Squeeze-and-Excitation)
- [ ] 实现 `MBConvBlock` (Mobile Inverted Bottleneck)
- [ ] 实现 `EfficientNet` 主体网络
- [ ] 添加 `efficientnet_b0` 工厂函数
- [ ] 集成到 `create_model` 函数

**Step 1.2: 修改数据加载**

- [ ] `get_train_transforms` 添加 `input_size` 参数
- [ ] 添加 `A.Resize(input_size, input_size)`
- [ ] 更新 `RandAugment` 强度 (M=10)
- [ ] 调整 `CoarseDropout` 尺寸 (8-16)
- [ ] `load_transforms` 添加 `input_size` 支持

**Step 1.3: 更新 main.py**

- [ ] 添加 `--model efficientnet_b0` 选项
- [ ] 添加 `--input_size` 参数 (default=64)
- [ ] 更新 `--randaugment_m` default=10

**Step 1.4: 测试**

```python
# 测试前向传播
python -c "
from scripts.model_architectures import efficientnet_b0
import torch
model = efficientnet_b0(input_size=64)
x = torch.randn(2, 3, 64, 64)
y = model(x)
print(f'Input: {x.shape}')
print(f'Output: {y.shape}')
print(f'Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
"
```

**预期输出**:

```
Input: torch.Size([2, 3, 64, 64])
Output: torch.Size([2, 100])
Parameters: 5.30M
```

### Phase 2: 训练 (6-8h)

**Step 2.1: 启动训练**

- [ ] 检查 CUDA 可用: `torch.cuda.is_available()`
- [ ] 清理旧日志: `rm cifar_pipeline.log`
- [ ] 启动训练 (使用完整命令)
- [ ] 记录开始时间

**Step 2.2: 实时监控**

- [ ] 每 30min 查看日志

  ```bash
  tail -f cifar_pipeline.log | grep "Val F1"
  ```

- [ ] 监控关键指标:
  - Epoch 50: Val F1 应该 > 0.70
  - Epoch 100: Val F1 应该 > 0.75
  - Epoch 150: Val F1 应该 > 0.78
  - Epoch 200: Val F1 应该 > 0.80

**Step 2.3: 异常处理**

- [ ] 如果 Loss = NaN: 降低 lr 到 0.0005
- [ ] 如果 OOM: 降低 batch_size 到 64
- [ ] 如果 Val F1 不提升: 检查数据增强

### Phase 3: 评估 (1h)

**Step 3.1: 加载最佳模型**

- [ ] 检查 `results/models/best_model.pth` 存在
- [ ] 查看 Best Epoch 和 Val F1

**Step 3.2: 测试集评估**

- [ ] 运行评估 (main.py 自动执行)
- [ ] 查看 `training_metrics.txt`

**Step 3.3: 分析结果**

- [ ] 整体 Test F1
- [ ] 人类类 F1 (boy, girl, man, woman, baby)
- [ ] 小动物类 F1 (otter, seal, mouse, shrew)
- [ ] Top-5 最佳类别
- [ ] Top-5 最差类别

---

## 📊 预期结果

### 乐观情况 (75% 概率)

```
Test F1 = 0.85-0.87 ✅ 目标达成！

类别性能:
- 人类类平均 F1: 0.72-0.76 (vs 0.63, +0.09-0.13)
  - boy:   0.68-0.72 (vs 0.59, +0.09-0.13)
  - girl:  0.68-0.72 (vs 0.57, +0.11-0.15)
  - man:   0.70-0.74 (vs 0.62, +0.08-0.12)
  - woman: 0.73-0.77 (vs 0.65, +0.08-0.12)
  - baby:  0.78-0.82 (vs 0.72, +0.06-0.10)

- 小动物类平均 F1: 0.68-0.72 (vs 0.61, +0.07-0.11)
- 机械类平均 F1: 0.93-0.94 (保持)
- 植物类平均 F1: 0.91-0.92 (保持)
```

**训练指标**:

- Best Epoch: 250-350 / 600
- Train Acc: 60-65% (Mixup/CutMix)
- Val Acc: 85-87%
- 训练时间: 6-8 小时

### 中等情况 (20% 概率)

```
Test F1 = 0.83-0.85 ⚠️ 接近目标

类别性能:
- 人类类平均 F1: 0.68-0.72 (+0.05-0.09)
- 小动物类平均 F1: 0.65-0.68 (+0.04-0.07)
```

**应对策略**:

- 方案 A: 提升到 96×96 分辨率
- 方案 B: 训练 EfficientNet-B1
- 方案 C: 模型集成

### 悲观情况 (5% 概率)

```
Test F1 = 0.80-0.83 ❌ 未达标

可能原因:
- EfficientNet 从头训练不稳定
- 数据增强过强/过弱
- 学习率设置不当
```

**应对策略**:

- 回退到 WRN-28-12 + 64×64
- 或模型集成 (WRN-28-10 × 3)

---

## 🔍 评估指标

### 主要指标

1. **Test F1 (macro avg)**: ≥ 0.85 ✅
2. **Test Accuracy**: ≥ 85%
3. **人类类 F1**: ≥ 0.70
4. **训练时间**: < 12h

### 次要指标

1. **参数量**: 5.3M (记录)
2. **FLOPs**: 记录 (可选)
3. **Best Epoch**: 记录
4. **Val F1**: 记录

### 分析维度

1. **类别分组分析**:
   - 人类类 (5 个)
   - 小动物类 (4 个)
   - 机械类 (7 个)
   - 植物类 (8 个)
   - 其他类 (76 个)

2. **F1 分布分析**:
   - F1 ≥ 0.90: X 个类别
   - F1 ≥ 0.80: X 个类别
   - F1 < 0.70: X 个类别

3. **混淆矩阵分析**:
   - 最易混淆的类别对
   - 人类类内部混淆情况

---

## 🎓 实验关键点

### 成功关键因素

1. **分辨率提升**: 32→64 是核心突破点
2. **架构选择**: EfficientNet > ResNet (论文验证)
3. **数据增强**: RandAugment M=10 + CutMix
4. **训练稳定性**: EMA + Warmup + AMP

### 失败风险点

1. **显存 OOM**: batch_size=96 应该安全
2. **训练不稳定**: 监控 Loss 曲线
3. **过拟合**: Train-Val gap 过大
4. **欠拟合**: Val F1 长期不提升

### 调试检查点

**Epoch 50**:

- Val F1 应该 > 0.70
- Loss 应该 < 2.0
- 如果不满足: 降低 lr 或调整 WD

**Epoch 100**:

- Val F1 应该 > 0.75
- 如果不满足: 检查数据增强

**Epoch 200**:

- Val F1 应该 > 0.80
- 接近目标，继续训练

---

## 📝 实验记录模板

### 训练日志

```markdown
## 训练过程

**开始时间**: YYYY-MM-DD HH:MM
**结束时间**: YYYY-MM-DD HH:MM
**总时长**: X.X 小时

**关键 Epoch**:
- Epoch 50: Val F1 = 0.XXX
- Epoch 100: Val F1 = 0.XXX
- Epoch 150: Val F1 = 0.XXX
- Best Epoch: XXX, Val F1 = 0.XXX

**异常情况**: (如果有)
- 无 / [描述]
```

### 最终结果

```markdown
## 最终结果

**Test F1**: 0.XXX
**Test Accuracy**: XX.XX%

**类别性能**:
- 人类类平均 F1: 0.XXX (+0.XXX vs 当前)
  - boy: 0.XXX (vs 0.59)
  - girl: 0.XXX (vs 0.57)
  - man: 0.XXX (vs 0.62)
  - woman: 0.XXX (vs 0.65)
  - baby: 0.XXX (vs 0.72)

- 小动物类平均 F1: 0.XXX
- Top-5 最佳: [列出]
- Top-5 最差: [列出]
```

### 结论与分析

```markdown
## 结论

**目标达成**: ✅ / ❌

**关键发现**:
1. 分辨率提升对人类类的影响: [描述]
2. EfficientNet vs WRN-28-10: [对比]
3. 数据增强效果: [描述]

**下一步**:
- 如果达标: 准备提交
- 如果未达标: [列出备选方案]
```

---

## 🔗 相关文档

- [EfficientNet 完整分析](../analysis/efficientnet_breakthrough_strategy.md)
- [EfficientNet 实现指南](./exp_400_efficientnet_implementation.md)
- [实验追踪表](../experiment_tracker.md)
- [Phase 1 总结](../../PHASE1_FINAL_SUMMARY.md)

---

**准备执行实验！目标: F1 ≥ 0.85!** 🚀
