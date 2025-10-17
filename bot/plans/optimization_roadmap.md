# CIFAR-100 优化路线图 (从头训练，目标 F1 ≥ 0.85)

**当前最佳**: F1 = 0.77 (ResNet34/ResNet50 从头训练)  
**约束条件**: 禁止预训练/迁移学习/蒸馏，必须从头训练  
**硬件环境**: RTX 5080 (训练), RTX 4080 Super (验收)  
**时间限制**: < 12 小时

---

## 🎯 阶段性目标

| 阶段 | 目标 F1 | 相对提升 | 预计难度 | 时间估算 |
|-----|---------|---------|---------|---------|
| **Phase 1** | 0.80 | +0.03 | 🟡 中等 | 3-5天 |
| **Phase 2** | 0.83 | +0.03 | 🟠 较难 | 5-7天 |
| **Phase 3** | 0.85 | +0.02 | 🔴 困难 | 7-10天 |

---

## 📋 Phase 1: 冲击 0.80 (优先级：🔥🔥🔥🔥🔥)

**目标**: 弥补预训练损失的 +0.03-0.04 提升  
**策略**: 低风险、高回报的快速优化

### 1.1 模型架构升级 ⭐⭐⭐⭐⭐

#### 🏆 **首选: Wide ResNet-28-10**

- **理由**:
  - CIFAR-10/100 的 SOTA baseline
  - 比 ResNet50 更宽（10x通道）但层数适中（28层）
  - 从头训练效果好，不依赖预训练
  - 参数量 ~36M（可接受）
- **预期提升**: +0.02-0.03 F1
- **实现难度**: 🟢 低（修改 `model_architectures.py`）
- **文件**: `bot/implementations/wide_resnet.py`

#### 🥈 **备选: PyramidNet-110**

- **理由**: 渐进式增加通道数，特征表示更丰富
- **预期提升**: +0.01-0.02 F1
- **实现难度**: 🟡 中（需要实现新架构）

#### 🥉 **备选: DenseNet-BC (k=24, depth=100)**

- **理由**: 特征复用，适合小数据集
- **预期提升**: +0.01-0.02 F1
- **实现难度**: 🟢 低（PyTorch 有参考实现）

---

### 1.2 数据增强强化 ⭐⭐⭐⭐

#### 🎯 **RandAugment**

- **当前问题**: 使用固定的增强策略（light/medium/strong）
- **改进方案**:
  - 集成 RandAugment（2参数：magnitude + num_ops）
  - 自动搜索最优增强组合
- **预期提升**: +0.01-0.02 F1
- **实现难度**: 🟢 低（Albumentations 支持）
- **参考论文**: "RandAugment: Practical automated data augmentation" (NeurIPS 2020)

#### 🎯 **GridMask**

- **当前问题**: CoarseDropout 可能不够有效
- **改进方案**: 替换为 GridMask（结构化遮挡）
- **预期提升**: +0.005-0.01 F1
- **实现难度**: 🟡 中（需要自行实现）

#### 🎯 **优化 Mixup/CutMix 比例**

- **当前配置**: Mixup(α=0.2/0.25) + CutMix(α=0.6/0.65)
- **改进方案**:
  - 更激进的 CutMix: α=0.8-1.0
  - 动态调整比例（早期多用Mixup，后期多用CutMix）
- **预期提升**: +0.005-0.01 F1
- **实现难度**: 🟢 低（调整 `train_utils.py`）

---

### 1.3 训练策略优化 ⭐⭐⭐⭐

#### 🔧 **更长的训练周期**

- **当前问题**: 350 epochs 可能不够充分
- **改进方案**:
  - 延长至 500-600 epochs
  - 调整 early stopping patience (60-80)
- **预期提升**: +0.01-0.015 F1
- **实现难度**: 🟢 低（修改超参数）

#### 🔧 **改进学习率调度**

- **当前配置**: Cosine Annealing + 10/20 epochs warmup
- **改进方案**:
  - 试验 Cosine Annealing with Warm Restarts (CosineAnnealingWarmRestarts)
  - T_0=50, T_mult=2 (重启周期：50, 100, 200 epochs)
- **预期提升**: +0.005-0.01 F1
- **实现难度**: 🟢 低（PyTorch 内置）

#### 🔧 **Stochastic Depth (DropPath)**

- **当前问题**: 深层网络可能过拟合
- **改进方案**:
  - 随机丢弃残差分支，提高泛化
  - Drop rate: 0.1-0.2
- **预期提升**: +0.01-0.015 F1
- **实现难度**: 🟡 中（需要修改 ResNet/WideResNet）
- **参考论文**: "Deep Networks with Stochastic Depth" (ECCV 2016)

---

### 1.4 超参数精调 ⭐⭐⭐

#### 📊 **学习率网格搜索**

- **当前最佳**: lr=0.0008 (ResNet50), lr=0.0012 (ResNet34)
- **搜索空间**: [0.0005, 0.0008, 0.001, 0.0012, 0.0015]
- **预期提升**: +0.005-0.01 F1

#### 📊 **Batch Size 调优**

- **当前配置**: 96 (ResNet50), 128 (ResNet34)
- **搜索空间**: [64, 96, 128, 160]
- **注意**: 需同步调整学习率（Linear Scaling Rule）

#### 📊 **正则化强度调优**

- **当前配置**: weight_decay=1.2e-3 (ResNet50), 8e-4 (ResNet34)
- **搜索空间**: [5e-4, 8e-4, 1e-3, 1.5e-3, 2e-3]
- **搜索空间 (dropout)**: [0.3, 0.4, 0.45, 0.5, 0.55]

---

### 📦 Phase 1 实施检查清单

- [ ] **Task 1.1**: 实现 Wide ResNet-28-10 (`bot/implementations/wide_resnet.py`)
- [ ] **Task 1.2**: 集成到 `model_architectures.py`
- [ ] **Task 1.3**: 添加 RandAugment 到 `data_augmentation.py`
- [ ] **Task 1.4**: 实现 GridMask
- [ ] **Task 1.5**: 添加 Stochastic Depth 到 Wide ResNet
- [ ] **Task 1.6**: 实现 CosineAnnealingWarmRestarts 调度器
- [ ] **Task 1.7**: 运行基线实验（Wide ResNet + 当前最佳超参）
- [ ] **Task 1.8**: 超参数网格搜索（lr, batch_size, weight_decay）
- [ ] **Task 1.9**: 记录实验结果到 `bot/experiments/phase1_results.md`

**预期总提升**: +0.03-0.05 F1 → **目标 F1: 0.80-0.82**

---

## 📋 Phase 2: 冲击 0.83 (优先级：🔥🔥🔥🔥)

**前提**: Phase 1 达到 0.80  
**策略**: 中等风险、中等回报的架构探索

### 2.1 现代架构探索 ⭐⭐⭐⭐⭐

#### 🚀 **ConvNeXt-Tiny (适配 CIFAR)**

- **理由**:
  - 2022年 SOTA CNN架构，性能接近 ViT
  - 纯卷积，训练稳定，不需要特殊技巧
  - 可以从头训练
- **预期提升**: +0.02-0.03 F1
- **实现难度**: 🟡 中（需要从 torchvision 适配）
- **参考**: "A ConvNet for the 2020s" (CVPR 2022)
- **文件**: `bot/implementations/convnext_cifar.py`

#### 🚀 **EfficientNet-B0 (适配 CIFAR)**

- **理由**:
  - 高效的复合缩放策略
  - 在小数据集上表现良好
- **预期提升**: +0.01-0.02 F1
- **实现难度**: 🟡 中
- **注意**: 需要调整输入尺寸和首层

#### 🚀 **RegNet-Y (适配 CIFAR)**

- **理由**:
  - Facebook AI 设计空间搜索的结果
  - 简单且高效
- **预期提升**: +0.01-0.02 F1
- **实现难度**: 🟡 中

---

### 2.2 高级训练技巧 ⭐⭐⭐⭐

#### 🎓 **Label Smoothing**

- **当前问题**: 目前被 Mixup/CutMix 禁用
- **改进方案**:
  - 在 Mixup/CutMix 之外额外使用 Label Smoothing
  - Smoothing factor: 0.1-0.15
- **预期提升**: +0.005-0.01 F1
- **实现难度**: 🟢 低（已实现，需要启用）

#### 🎓 **Sharpness-Aware Minimization (SAM)**

- **理由**:
  - 优化平坦的 loss landscape，提高泛化
  - 2021年 ICLR Best Paper
- **预期提升**: +0.01-0.02 F1
- **实现难度**: 🟡 中（需要修改优化器）
- **参考**: "Sharpness-Aware Minimization for Efficiently Improving Generalization"
- **注意**: 训练时间 +20-30%

#### 🎓 **自蒸馏 (Self-Distillation)**

- **方案**:
  - 训练多个独立模型（不同随机种子）
  - 用集成的软标签重新训练单模型
- **预期提升**: +0.01-0.015 F1
- **实现难度**: 🟠 较难（需要额外训练时间）
- **注意**: 不违反"禁止蒸馏"规则（自己蒸馏自己）

---

### 2.3 数据增强进阶 ⭐⭐⭐

#### 🎨 **CutMix + MixUp + FMix 三合一**

- **当前**: 只用 CutMix + MixUp
- **改进**: 添加 FMix（傅里叶域混合）
- **预期提升**: +0.005-0.01 F1
- **实现难度**: 🟡 中

#### 🎨 **AutoAugment-CIFAR**

- **理由**: 针对 CIFAR 优化的自动增强策略
- **预期提升**: +0.01-0.015 F1
- **实现难度**: 🟢 低（timm 库提供）

---

### 📦 Phase 2 实施检查清单

- [ ] **Task 2.1**: 实现 ConvNeXt-Tiny for CIFAR (`bot/implementations/convnext_cifar.py`)
- [ ] **Task 2.2**: 实现 SAM 优化器 (`bot/implementations/sam_optimizer.py`)
- [ ] **Task 2.3**: 添加 FMix 到数据增强
- [ ] **Task 2.4**: 实现自蒸馏训练流程
- [ ] **Task 2.5**: 运行 ConvNeXt 基线实验
- [ ] **Task 2.6**: 测试 SAM + Wide ResNet
- [ ] **Task 2.7**: 测试 SAM + ConvNeXt
- [ ] **Task 2.8**: 记录实验结果到 `bot/experiments/phase2_results.md`

**预期总提升**: +0.03-0.04 F1 → **目标 F1: 0.83-0.85**

---

## 📋 Phase 3: 冲刺 0.85 (优先级：🔥🔥🔥)

**前提**: Phase 2 达到 0.83  
**策略**: 高风险、高回报的极致优化

### 3.1 模型集成（如果允许）⭐⭐⭐⭐⭐

#### 🎯 **多模型投票/平均**

- **方案**:
  - Wide ResNet-28-10
  - ConvNeXt-Tiny
  - PyramidNet-110
  - 3-5个模型的 soft voting
- **预期提升**: +0.02-0.03 F1
- **实现难度**: 🟢 低（推理时集成）
- **注意**: 需确认是否违反规则

---

### 3.2 超大模型 ⭐⭐⭐⭐

#### 🐘 **Wide ResNet-40-10 / 28-12**

- **理由**: 更深/更宽的模型，更强的表达能力
- **预期提升**: +0.01-0.02 F1
- **实现难度**: 🟢 低
- **风险**:
  - 训练时间增加 30-50%
  - 可能 OOM（需要减小 batch size）

#### 🐘 **ConvNeXt-Small**

- **理由**: ConvNeXt 的更大版本
- **预期提升**: +0.01-0.015 F1
- **风险**: 同上

---

### 3.3 终极调优 ⭐⭐⭐

#### 🔬 **贝叶斯超参数优化**

- **工具**: Optuna / Ray Tune
- **搜索空间**:
  - lr, weight_decay, dropout, batch_size
  - augmentation magnitude
  - scheduler parameters
- **预期提升**: +0.005-0.01 F1
- **实现难度**: 🟡 中

#### 🔬 **Test-Time Augmentation (TTA)**

- **方案**:
  - 测试时对每张图片做 5-10 次增强
  - 平均预测结果
- **预期提升**: +0.01-0.015 F1
- **实现难度**: 🟢 低

---

### 📦 Phase 3 实施检查清单

- [ ] **Task 3.1**: 实现模型集成框架
- [ ] **Task 3.2**: 训练 3-5 个最佳模型
- [ ] **Task 3.3**: 实现 TTA
- [ ] **Task 3.4**: 贝叶斯优化超参数
- [ ] **Task 3.5**: 最终测试集评估
- [ ] **Task 3.6**: 记录实验结果到 `bot/experiments/phase3_results.md`

**预期总提升**: +0.02-0.03 F1 → **目标 F1: ≥ 0.85**

---

## 📊 风险评估与应对

### 高风险项目

| 项目 | 风险 | 缓解措施 |
|-----|------|---------|
| ConvNeXt | 训练不稳定 | 使用 LayerScale, 更小的 lr |
| SAM | 训练时间翻倍 | 仅在最终模型使用 |
| 超大模型 | OOM | 减小 batch size, 使用梯度累积 |
| 自蒸馏 | 时间开销大 | 并行训练多个模型 |

---

## 🎯 每日计划建议

### Week 1 (Phase 1)

- **Day 1-2**: 实现 Wide ResNet + RandAugment
- **Day 3**: 基线实验
- **Day 4-5**: 超参数搜索
- **Day 6**: 验证最佳配置
- **Day 7**: 休整 + 文档整理

### Week 2 (Phase 2)

- **Day 8-9**: 实现 ConvNeXt
- **Day 10-11**: SAM 优化器 + 实验
- **Day 12-13**: 自蒸馏实验
- **Day 14**: 阶段性评估

### Week 3 (Phase 3)

- **Day 15-17**: 模型集成 + TTA
- **Day 18-19**: 最终调优
- **Day 20-21**: 提交前测试 + 代码清理

---

## 📝 实验记录模板

每个实验需记录：

```markdown
## Experiment #XXX - [实验名称]

**日期**: YYYY-MM-DD  
**目标**: [描述]  
**配置**: 
- Model: 
- Augmentation: 
- Hyperparameters: 

**结果**:
- Train F1: 
- Val F1: 
- Test F1: 
- Training Time: 

**分析**: 
- 成功/失败原因
- 下一步行动

**代码**: `bot/experiments/exp_XXX/`
```

---

## 🔗 参考资源

### 论文

- Wide ResNet: "Wide Residual Networks" (BMVC 2016)
- PyramidNet: "Deep Pyramidal Residual Networks" (CVPR 2017)
- ConvNeXt: "A ConvNet for the 2020s" (CVPR 2022)
- RandAugment: "RandAugment: Practical automated data augmentation" (NeurIPS 2020)
- SAM: "Sharpness-Aware Minimization" (ICLR 2021)
- Stochastic Depth: "Deep Networks with Stochastic Depth" (ECCV 2016)

### 代码库

- timm (PyTorch Image Models): <https://github.com/huggingface/pytorch-image-models>
- CIFAR-100 SOTA: <https://paperswithcode.com/sota/image-classification-on-cifar-100>

---

**最后更新**: 2025-10-17  
**当前阶段**: Phase 1 准备中
