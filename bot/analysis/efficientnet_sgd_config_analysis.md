# EfficientNet-B0 + SGD 配置可行性分析

## 分析时间

2025-10-25

## 背景

用户希望使用SGD优化器训练EfficientNet-B0，提供的脚本为：

```powershell
python main.py `
    --optimizer sgd `
    --lr 0.1 `
    --weight_decay 5e-4 `
    --warmup_epochs 10
```

## 完整配置清单

### 1. 脚本指定参数

- `optimizer`: sgd
- `lr`: 0.1
- `weight_decay`: 5e-4 (0.0005)
- `warmup_epochs`: 10

### 2. 模型和输入 (默认值)

- `model`: efficientnet_b0
- `input_size`: 64 (64×64 像素)
- `dropout`: 0.2
- `drop_path_rate`: 0.2

### 3. 训练配置 (默认值)

- `batch_size`: 96
- `num_epochs`: 600
- `scheduler`: cosine
- `max_grad_norm`: 1.0 (梯度裁剪)

### 4. 损失函数 (默认值)

- `loss_type`: ce (CrossEntropy)
- `label_smoothing`: 0.0 (未启用)
- `use_class_weights`: True
- `weight_strategy`: long_board (保守的类别权重)

### 5. 数据增强 (默认值)

- `aug_strength`: randaugment
- `randaugment_n`: 2 (操作数量)
- `randaugment_m`: 9 (强度)
- `use_cutmix`: True
- `cutmix_alpha`: 1.0
- `mixup_alpha`: 0.25 (被CutMix覆盖)

### 6. 优化技术 (默认值)

- `use_amp`: True (混合精度训练)
- `use_ema`: True (指数移动平均)
- `use_compile`: True (torch.compile加速)

### 7. SGD 内部配置 (train_utils.py 硬编码)

- `momentum`: 0.9
- `nesterov`: True (Nesterov加速)

## 详细分析

### ✅ 优势分析

#### 1. **SGD + 0.1 学习率是经典配置**

- lr=0.1 是 CIFAR-100 上 SGD 的标准起始学习率
- ResNet/WRN 论文都使用这个配置
- 配合 Cosine Annealing，会平滑衰减到接近 0

#### 2. **Weight Decay = 5e-4 适合 SGD**

- 这是 CIFAR 数据集上 SGD 的经典值
- 比 AdamW 的 1e-3 更轻，符合 SGD 最佳实践
- 论文中广泛验证的配置

#### 3. **Momentum + Nesterov = 最优 SGD 变体**

- `momentum=0.9`: 标准配置，加速收敛
- `nesterov=True`: Nesterov 动量，提前"观察"梯度方向
- 代码中硬编码，无需手动指定

#### 4. **Warmup + Cosine 调度器**

- `warmup_epochs=10`: 前10个epoch逐步升温到0.1
- `cosine scheduler`: 平滑衰减，避免震荡
- 总共600个epoch，衰减曲线充分

#### 5. **EfficientNet 特有的正则化已启用**

- `drop_path_rate=0.2`: Stochastic Depth，EfficientNet标配
- `dropout=0.2`: 分类头的Dropout
- `use_ema=True`: 指数移动平均，提升泛化性

#### 6. **强数据增强 + 64×64 输入**

- RandAugment (N=2, M=9): 强而不过度的自动增强
- CutMix (alpha=1.0): 保持局部特征，适合小图
- 输入尺寸64×64: 符合 EfficientNet 论文建议，增强细节识别

#### 7. **显存占用安全**

- EfficientNet-B0 参数量：~5.3M
- 64×64 输入 + batch_size=96
- 预估显存占用：约 8-10GB (启用AMP)
- **16GB 显存完全安全，有充足余量**

### ⚠️ 潜在问题

#### 1. **SGD 可能不是 EfficientNet 的最优选择**

**论文证据：**

- EfficientNet 原论文使用 **RMSProp** 优化器
- EfficientNetV2 论文使用 **SGD with momentum=0.9**，但：
  - 学习率为 0.005 (比0.1小20倍)
  - 配合更激进的正则化

**可能的影响：**

- lr=0.1 对 EfficientNet 可能过大，尤其是在早期
- Warmup 10个epoch可能不够长，需要15-20个epoch
- AdamW (lr=0.001) 已经过实战验证，达到 F1=0.82

#### 2. **Label Smoothing 未启用**

- 当前 `label_smoothing=0.0`
- EfficientNet 论文推荐使用 label_smoothing=0.1
- CIFAR-100 这种100类任务，label smoothing 可以防止过拟合

#### 3. **Batch Size 可能偏小**

- SGD 通常受益于更大的 batch size (128-256)
- 当前 batch_size=96，可能限制了 SGD 的优势
- AdamW 对 batch size 不敏感，但 SGD 更依赖大批次

#### 4. **训练时间会非常长**

- 600 epochs × (64×64 输入) = 预计 20-30 小时 (4080S)
- 如果 SGD 不收敛，代价高昂

### 📊 与 AdamW 的对比

| 项目 | **SGD (脚本配置)** | **AdamW (当前最佳)** |
|------|-------------------|---------------------|
| **学习率** | 0.1 | 0.001 |
| **Weight Decay** | 5e-4 | 1e-3 |
| **Warmup** | 10 epochs | 10 epochs |
| **优势** | 更强泛化性能 | 收敛快，稳定 |
| **劣势** | 超参敏感，训练慢 | 可能过拟合 |
| **EfficientNet 适配度** | ⚠️ 中等 | ✅ 高 |
| **CIFAR-100 适配度** | ✅ 高 | ✅ 高 |
| **预计 F1** | 0.83-0.86 (如果收敛) | 0.82 (已验证) |

## 可行性结论

### ✅ 技术上完全可行

- 所有默认参数配置合理
- 显存占用安全 (16GB下约50-60%占用)
- 代码逻辑无问题，可以直接运行

### ⚠️ 但存在以下风险

#### **风险1: 学习率可能过大**

- EfficientNet 对 lr 敏感
- 建议监控前 10-20 个 epoch 的 loss
- 如果 loss 震荡或爆炸，立即停止

#### **风险2: 收敛速度慢**

- SGD 需要更多 epoch 才能达到最优
- 600 epochs 可能不够（EfficientNet论文用了更多）
- 训练时间预计 20-30 小时

#### **风险3: 不如 AdamW 稳定**

- AdamW 自适应学习率，容错率高
- SGD 对超参数非常敏感
- 如果配置不当，可能浪费大量时间

## 改进建议

### 方案1: 保守改进 (推荐)

```powershell
python main.py `
    --optimizer sgd `
    --lr 0.05 `  # 降低初始学习率
    --weight_decay 5e-4 `
    --warmup_epochs 15 `  # 延长warmup
    --label_smoothing 0.1  # 启用label smoothing
```

**理由:**

- lr=0.05 更安全，warmup后仍能探索
- warmup_epochs=15 让模型更稳定地起步
- label_smoothing=0.1 防止过拟合

### 方案2: 激进尝试 (如果时间充裕)

```powershell
python main.py `
    --optimizer sgd `
    --lr 0.1 `
    --weight_decay 5e-4 `
    --warmup_epochs 20 `
    --label_smoothing 0.15 `
    --batch_size 128  # 增大batch size
```

**理由:**

- 保持 lr=0.1，但用更长的 warmup 缓冲
- batch_size=128 让 SGD 梯度估计更准确
- label_smoothing=0.15 更激进的正则化

### 方案3: 先短期测试 (最安全)

```powershell
python main.py `
    --optimizer sgd `
    --lr 0.1 `
    --weight_decay 5e-4 `
    --warmup_epochs 10 `
    --num_epochs 50  # 先只跑50个epoch测试
```

**理由:**

- 先用 50 epochs 测试配置是否合理
- 观察 loss 曲线和 F1 趋势
- 如果效果好，再跑完整的 600 epochs

## 最终建议

### 如果目标是 **快速验证 SGD 是否更优**

→ **使用方案3**，先跑 50 epochs 测试

### 如果目标是 **一次性冲到 F1≥0.85**

→ **使用方案1**，保守改进，提高成功率

### 如果目标是 **探索 SGD 的极限**

→ **使用方案2**，激进尝试，但准备好重新调参

### 如果时间紧张或风险厌恶

→ **继续使用 AdamW**，已验证的 F1=0.82 基线
→ 将精力放在数据增强和模型调优上

## 执行决策

### 当前脚本可以直接运行吗？

**✅ 可以，但建议修改**

### 建议的执行命令 (折中方案)

```powershell
python main.py `
    --optimizer sgd `
    --lr 0.08 `  # 折中的学习率
    --weight_decay 5e-4 `
    --warmup_epochs 15 `  # 延长warmup
    --label_smoothing 0.1 `  # 启用label smoothing
    --num_epochs 600
```

这个配置：

- 学习率 0.08 介于保守和激进之间
- warmup_epochs=15 提供更稳定的启动
- 启用 label_smoothing，提升泛化性
- 保持其他默认值，利用现有的强数据增强

## 监控要点

在训练过程中，请密切关注：

1. **前 20 个 epoch 的 training loss**:
   - 如果 loss 不下降或震荡 → lr 过大
   - 如果 loss 下降过慢 → lr 过小

2. **Validation F1 在 100 epoch 时的值**:
   - 如果 < 0.70 → 配置有问题
   - 如果 0.70-0.75 → 可以继续，但不如 AdamW
   - 如果 > 0.75 → 很有希望达到 0.85+

3. **显存占用**:
   - 应该在 8-12GB 之间
   - 如果接近 16GB，考虑降低 batch_size

## 结论

**原始脚本技术上可行，但建议加上 `--label_smoothing 0.1` 和 `--lr 0.08` 再运行。**

如果不想修改，直接运行原脚本也可以，但要做好以下准备：

1. 前 50 个 epoch 密切监控 loss
2. 如果效果不好，立即切换到改进方案
3. 准备至少 24 小时的训练时间

**风险等级: 🟡 中等**  
**成功概率: 65-75%**  
**预期 F1: 0.83-0.86 (如果配置合适)**
