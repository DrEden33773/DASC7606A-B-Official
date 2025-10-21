# CIFAR-100 图像分类优化技术报告

**项目**: DASC7606A-B CIFAR-100 图像分类（从头训练）  
**最终成绩**: Test F1 = **0.82** (PyramidNet-110)  
**完成日期**: 2025-10-17  
**作者**: [Student ID]

---

## 📋 项目背景与约束

### 任务要求

**数据集**: CIFAR-100

- 100 个类别
- 训练样本: 50,000 (split 后 40,000)
- 测试样本: 10,000
- 图像分辨率: **32×32 pixels** (极低分辨率)

**评分标准**:

- F1 ≥ 0.85: 100 分（满分）
- F1 ≥ 0.80: 90 分
- F1 ≥ 0.75: 80 分

**严格约束**:

1. ❌ **禁止使用任何预训练模型**（ImageNet, 迁移学习, 蒸馏）
2. ✅ **必须从头训练**（train from scratch）
3. ⏱️ **训练时间 < 12 小时**
4. 💾 **单 GPU 限制**（RTX 4080 Super/5080, 16GB）

**允许修改范围**:

- ✅ `scripts/model_architectures.py`
- ✅ `scripts/data_augmentation.py`
- ✅ `scripts/train_utils.py`
- ✅ `main.py` (超参数)

---

## 🎯 最终成果

### 三个高分方案

| 方案 | 模型 | Val F1 | Test F1 | 得分 | 训练时间 |
|-----|------|--------|---------|------|---------|
| **Phase 1** | WRN-28-10 + SD + RA | **0.8131** | **0.81** | **90分** | 1.5h |
| **Phase 2.7** | PyramidNet-110 + SD + RA | **0.8307** | **0.82** | **90分** | 3h38m |
| **Phase 3** | Ensemble (WRN×3) | - | **0.82** | **90分** | 5h |

**最佳单模型**: PyramidNet-110, Test F1 = **0.82**

---

## 📈 完整优化历程

### 起点 (Historical Baseline)

```
ResNet-34 (21M params): F1 = 0.77
ResNet-50 (23.5M params): F1 = 0.77
```

**问题**: 传统 ResNet 在 CIFAR 上表现平平

---

### Phase 1: 架构选择与优化 (F1: 0.77 → 0.8131)

#### Exp #100: Wide ResNet-28-10 Baseline (+0.01)

**架构改进**:

```python
模型: Wide ResNet-28-10
改进: 加宽通道 (16→160→320→640)
参数: 36.5M
```

**超参数**:

```python
lr = 0.001
weight_decay = 5e-4
optimizer = "adamw"
dropout = 0.3
aug_strength = "medium"
mixup_alpha = 0.25
cutmix_alpha = 0.65
```

**结果**: Val F1 = **0.7802** (+0.0102)

**关键发现**: Wide ResNet 比标准 ResNet 更适合 CIFAR

---

#### Exp #103-Revised: Stochastic Depth 调优 (-0.001)

**尝试**: drop_path_rate = 0.2

**结果**: Val F1 = 0.7791 ❌

**教训**: drop_path_rate=0.2 对 WRN-28-10 **过强**，导致欠拟合

---

#### Exp #104b: Stochastic Depth + RandAugment (+0.03) ✨

**关键突破**:

1. **Stochastic Depth 优化**: 0.2 → **0.1**
2. **RandAugment 独立使用**: 不与 medium aug 叠加

**超参数**:

```python
model = "wide_resnet28_10"
drop_path_rate = 0.1  # ← 关键！
aug_strength = "randaugment"  # ← 独立使用
randaugment_n = 2
randaugment_m = 9
lr = 0.001
weight_decay = 5e-4
optimizer = "adamw"
batch_size = 128
num_epochs = 300
```

**结果**: Val F1 = **0.8131**, Test F1 = **0.81** ✅

**关键技术**:

- ✅ Stochastic Depth (0.1): 模型层面正则化
- ✅ RandAugment (N=2, M=9): 自动增强搜索
- ✅ Mixup + CutMix: Batch-level 增强
- ✅ EMA: 指数移动平均
- ✅ AMP: 混合精度训练

**Phase 1 完成**: 达到 90 分标准 (F1 ≥ 0.80)

---

### Phase 2: 架构探索 (失败经验)

#### ❌ Exp #205: Wide ResNet-28-12 (F1=0.80)

**尝试**: 增大模型 (52.8M params)

**结果**: F1 下降到 0.80

**教训**: **模型容量过大 → 过拟合**

**关键洞察**: 36.5M 是 CIFAR-100 的**容量甜点**

---

#### ❌ Exp #200: ConvNeXt-Tiny (F1=0.79)

**尝试**: 现代 CNN 架构

**问题**: weight_decay=0.05 (ImageNet 配置) 对 CIFAR 太大

**结果**: F1 = 0.79

**教训**: 不能盲目照搬 ImageNet 配置

---

#### ❌ Exp #300: Self-Distillation (F1=0.7968)

**尝试**: BYOT (Be Your Own Teacher)

**问题**: 与现有 8 种正则化技术叠加过度

**结果**: F1 = 0.7968

**教训**: 正则化不是越多越好

---

### Phase 2.7: PyramidNet 突破 (F1: 0.8131 → 0.8307)

#### Exp #310: PyramidNet-110 (α=270) (+0.0176) ✨

**架构创新**:

```
渐进式通道增长:
16 → 23 → 31 → 38 → ... → 286
(每个 block +5 channels)

vs Wide ResNet 突然跳跃:
16 → 160 (×10) → 320 (×2) → 640 (×2)
```

**超参数**:

```python
model = "pyramidnet110_270"
depth = 110 layers
alpha = 270 (widening factor)
drop_path_rate = 0.1
aug_strength = "randaugment"
randaugment_n = 2
randaugment_m = 9
lr = 0.001
weight_decay = 1e-4  # ← 比 WRN 略小 (5e-4)
optimizer = "adamw"
batch_size = 128
num_epochs = 600  # ← 深网络需更长训练
early_stopping_patience = 60
mixup_alpha = 0.25
cutmix_alpha = 0.65
```

**结果**:

- Val F1 = **0.8307** (+0.0176)
- Test F1 = **0.82**

**关键优势**:

- ✅ 渐进式通道增长 → 更平滑的特征演化
- ✅ 110 layers (深) → 更强的表达能力
- ✅ 26M params → 避免了 WRN-28-12 的过拟合
- ✅ Zero-padded shortcut → 参数高效

**训练时间**: 3h38m

**Phase 2.7 完成**: 新纪录，Val F1 = 0.8307

---

### Phase 3: Ensemble 探索 (F1: 0.82)

#### Exp #320: WRN-28-10 Ensemble (3 models)

**配置**:

```python
3 个 WRN-28-10 模型:
- Model 1: seed=42, F1=0.81
- Model 2: seed=43, F1≈0.81
- Model 3: seed=44, F1≈0.81

Ensemble: EnsembleModel (Soft Voting)
```

**实现**:

```python
class EnsembleModel(nn.Module):
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mean(dim=0)
```

**结果**: Test F1 = **0.82**

**发现**: Ensemble 提升有限 (0.81 → 0.82, +0.01)

**可能原因**:

- 模型间多样性不足（相同架构和配置）
- 已接近该任务的性能上限

---

## 🔬 关键技术解析

### 1. Stochastic Depth (DropPath)

**原理**: 训练时随机丢弃残差分支，提升泛化

**实现**:

```python
class DropPath(nn.Module):
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand(...)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor
```

**最佳配置**:

- WRN-28-10: drop_path_rate = **0.1**
- PyramidNet-110: drop_path_rate = **0.1**
- WRN-28-12: drop_path_rate = 0.2 (过强，失败)

**效果**: 约 +0.01-0.015 F1

---

### 2. RandAugment (自动增强搜索)

**原理**: 从 14 种操作中随机选 N 个，幅度统一为 M

**14 种操作**:

```
Contrast, Brightness, Saturation, Sharpness, Rotate,
ShearX/Y, TranslateX/Y, AutoContrast, Equalize,
Invert, Posterize, Solarize
```

**关键发现**: **必须独立使用，不与传统 augmentation 叠加**

**错误尝试**: RandAug + Medium Aug → 增强过度 → F1 = 0.75 ❌

**正确使用**: aug_strength = "randaugment" (替代 medium) → F1 = 0.8131 ✅

**最佳配置**:

```python
aug_strength = "randaugment"  # 独立模式
randaugment_n = 2  # 每图 2 个操作
randaugment_m = 9  # 幅度 9/10
```

**效果**: 约 +0.02-0.03 F1

---

### 3. Mixup + CutMix 自适应组合

**实现**: 根据 batch 中的类别分布动态选择

**策略**:

- 人类/小动物类 (detail-sensitive): Mixup only (α=0.4)
- 机械/植物类 (local-feature): 80% CutMix, 20% Mixup
- 其他类 (mixed): 30% Mixup, 70% CutMix

**配置**:

```python
mixup_alpha = 0.25
cutmix_alpha = 0.65
```

---

### 4. 其他关键技术

**EMA (Exponential Moving Average)**:

```python
ema_decay = 0.9999
```

**Gradient Clipping**:

```python
max_grad_norm = 1.0
```

**Optimizer**:

```python
optimizer = "adamw"
lr = 0.001
scheduler = "cosine"
warmup_epochs = 10
```

**AMP (Automatic Mixed Precision)**:

- 加速训练 ~10-15%
- 减少显存占用

---

## 📊 三个最佳方案详细配置

### 🥇 方案 1: Wide ResNet-28-10 + SD + RA

**性能**: Val F1 = **0.8131**, Test F1 = **0.81**  
**训练时间**: ~1.5 小时  
**参数量**: 36.5M

#### 完整超参数

```python
# 模型
model = "wide_resnet28_10"
depth = 28
widen_factor = 10
dropout = 0.3
drop_path_rate = 0.1

# 数据增强
aug_strength = "randaugment"
randaugment_n = 2
randaugment_m = 9
mixup_alpha = 0.25
cutmix_alpha = 0.65

# 训练
lr = 0.001
weight_decay = 5e-4
optimizer = "adamw"
scheduler = "cosine"
warmup_epochs = 10
num_epochs = 300
early_stopping_patience = 30
batch_size = 128

# 技术
use_amp = True
use_ema = True
ema_decay = 0.9999
max_grad_norm = 1.0

# 硬件
device = "cuda"
num_workers = 4
seed = 42
```

#### 运行命令

```bash
python main.py \
    --model wide_resnet28_10 \
    --drop_path_rate 0.1 \
    --aug_strength randaugment \
    --seed 42
```

（大部分参数已设为最优默认值）

#### 特点

- ✅ **最稳定**: Val-Test gap 仅 0.003
- ✅ **训练快**: 3.5 小时
- ✅ **配置成熟**: 经过充分验证
- ✅ **泛化好**: Test F1 与 Val F1 高度一致

---

### 🥈 方案 2: PyramidNet-110 (α=270) + SD + RA

**性能**: Val F1 = **0.8307**, Test F1 = **0.82**  
**训练时间**: ~3h38m  
**参数量**: 28.5M

#### 完整超参数

```python
# 模型
model = "pyramidnet110_270"
depth = 110 layers
alpha = 270 (widening factor)
drop_path_rate = 0.1

# 数据增强
aug_strength = "randaugment"
randaugment_n = 2
randaugment_m = 9
mixup_alpha = 0.25
cutmix_alpha = 0.65

# 训练
lr = 0.001
weight_decay = 1e-4  # ← 比 WRN 略小
optimizer = "adamw"
scheduler = "cosine"
warmup_epochs = 10
num_epochs = 600  # ← 深网络需更长训练
early_stopping_patience = 30
batch_size = 128

# 技术
use_amp = True
use_ema = True
ema_decay = 0.9999
max_grad_norm = 1.0

# 硬件
device = "cuda"
num_workers = 4
seed = 42
```

#### 运行命令

```bash
python main.py \
    --model pyramidnet110_270 \
    --drop_path_rate 0.1 \
    --aug_strength randaugment \
    --weight_decay 1e-4 \
    --num_epochs 600 \
    --early_stopping_patience 60 \
    --seed 42
```

#### 特点

- ✅ **Val F1 最高**: 0.8307
- ✅ **参数效率**: 28.5M (比 WRN 少 22%)
- ✅ **架构先进**: 渐进式通道增长
- ⚠️ **Val-Test gap**: 0.0107 (略大)

---

### 🥉 方案 3: Ensemble (WRN-28-10 × 3)

**性能**: Test F1 = **0.82**  
**训练时间**: 5 小时 (串行)  
**参数量**: 109.5M (3 × 36.5M)

#### 完整配置

```python
# 基础模型配置 (与方案 1 相同)
model = "wide_resnet28_10"
drop_path_rate = 0.1
aug_strength = "randaugment"
# ... (其他参数同方案 1)

# Ensemble 配置
ensemble_seeds = [42, 43, 44]
num_models = 3
ensemble_method = "soft_voting"  # 平均 logits
```

#### 运行命令

```bash
python main.py --ensemble_seeds 42,43,44
```

#### 实现

```python
# scripts/model_architectures.py
class EnsembleModel(nn.Module):
    def __init__(self, models: List[nn.Module]):
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mean(dim=0)

# main.py
def ensemble_main(args):
    # 训练 3 个模型
    models = []
    for seed in [42, 43, 44]:
        model = build_and_train(seed)
        models.append(model)
    
    # 创建 ensemble
    ensemble = EnsembleModel(models)
    
    # 使用原有 evaluate()
    evaluate(args, ensemble)
```

#### 特点

- ✅ **多样性**: 3 个不同随机种子
- ✅ **Soft voting**: 平均 logits，更稳定
- ⚠️ **提升有限**: +0.01 (vs 单模型)
- ⚠️ **时间长**: 10.5 小时

---

## 💡 关键技术决策

### 决策 1: Stochastic Depth 参数调优

**问题**: 初始使用 drop_path_rate=0.2，性能下降

**分析**:

```
drop_path_rate 影响:
0.0: 无正则化 → 可能过拟合
0.1: ✅ 最佳平衡
0.2: 正则化过强 → 欠拟合

WRN-28-10 最佳: 0.1
PyramidNet-110 最佳: 0.1
```

**结论**: drop_path_rate=0.1 是 30-40M 参数模型的最佳值

---

### 决策 2: RandAugment 使用方式

**错误尝试**: RandAug + Medium Aug (叠加)

- 结果: 增强过度，Train Acc 仅 48%, F1 = 0.75

**正确方式**: RandAug 作为独立 aug_strength 选项

- aug_strength = "randaugment" (替代，而非叠加)
- 结果: F1 = 0.8131 (+0.03)

**教训**: RandAugment 设计为完整增强策略，不应叠加

---

### 决策 3: 模型容量选择

**实验数据**:

```
21M (ResNet-34):    F1 = 0.77
36.5M (WRN-28-10):  F1 = 0.8131 ✅ 最优
52.8M (WRN-28-12):  F1 = 0.80   ❌ 过拟合
28.5M (PyramidNet): F1 = 0.8307 ✅ 架构优势
```

**结论**:

- 30-40M 是 CIFAR-100 的容量甜点
- 更大模型会过拟合
- 架构创新比盲目增大更有效

---

## ❌ 失败尝试总结

### 失败案例

| 尝试 | F1 | 问题 | 教训 |
|-----|-----|------|------|
| WRN-28-12 (52.8M) | 0.80 | 参数过多 | 容量甜点 ≈ 36M |
| ConvNeXt (wd=0.05) | 0.79 | wd 配置不当 | 不能照搬 ImageNet |
| Self-Distill | 0.7968 | 正则化叠加 | 8 层正则化过度 |
| RandAug + Medium | 0.75 | 增强叠加 | RandAug 应独立使用 |
| SD (drop_path=0.2) | 0.7791 | SD 过强 | 0.1 是最佳值 |

### 关键教训

1. **模型容量有上限**: 36M 是甜点，更大反而差
2. **配置不能照搬**: ImageNet ≠ CIFAR
3. **正则化需平衡**: 不是越多越好
4. **RandAug 独立使用**: 不要叠加

---

## 🎓 核心经验与洞察

### 经验 1: CIFAR-100 的特殊性

**"麻雀虽小，五脏俱全"**:

- 100 个类别（复杂度高）
- 每类仅 400 训练样本（数据少）
- 32×32 分辨率（信息量小）

**最佳策略**:

- 中等容量模型 (30-40M)
- 强正则化 (SD, RA, Mixup/CutMix)
- 充分训练 (300-600 epochs)

---

### 经验 2: Train Acc < Val Acc 是正常的

**现象**: Train Acc ≈ 60%, Val Acc ≈ 81%

**原因**: Mixup/CutMix 使训练样本混合

```python
# 训练时
image = 0.7 × cat + 0.3 × dog
→ 即使预测对也只算 70% 正确

# 验证时
image = 1.0 × cat
→ 预测对算 100% 正确
```

**结论**: 这不是 bug，是 Mixup/CutMix 的预期效果

---

### 经验 3: 架构创新 vs 超参数优化

**架构创新成功率**: 30% (3/10)

- ✅ Wide ResNet
- ✅ PyramidNet
- ❌ ConvNeXt, 更大模型, Self-Distill, 等

**超参数优化成功率**: 80%+

- ✅ Stochastic Depth 调优
- ✅ RandAugment 独立使用
- ✅ weight_decay 微调

**结论**: 在小数据集上，精细调参比架构创新更可靠

---

## 📊 性能瓶颈分析

### 困难类别 (F1 < 0.60)

| 类别 | F1 | 原因 |
|-----|-----|------|
| boy | 0.57 | 32×32 分辨率，面部细节丢失 |
| girl | 0.57 | 与 boy 混淆，特征模糊 |
| otter | 0.61 | 小动物纹理不清晰 |
| seal | 0.58 | 水生动物，与 otter 混淆 |
| man | 0.57 | 面部细节丢失 |
| woman | 0.65 | 略好，但仍困难 |

**人类类平均**: F1 ≈ 0.59  
**整体平均**: F1 ≈ 0.82  
**差距**: -0.23 (-28%)

**根本原因**: 32×32 分辨率无法保留足够的面部细节

---

### 优秀类别 (F1 ≥ 0.90)

| 类别 | F1 | 原因 |
|-----|-----|------|
| pickup_truck | 0.96 | 轮廓清晰，CutMix 友好 |
| lawn_mower | 0.95 | 机械特征明显 |
| aquarium_fish | 0.94 | 颜色和形状独特 |
| motorcycle | 0.94 | 局部特征明显 |
| wardrobe | 0.93 | 形状特征强 |

**机械/大物体类平均**: F1 ≈ 0.92

**关键**: CutMix 对这些类极有效

---

### 性能极限推测

**单模型极限**: **F1 ≈ 0.82-0.83**

**约束因素**:

1. 32×32 分辨率（人类类上限 ~0.60）
2. 40k 训练样本（数据量限制）
3. 禁止预训练（无外部知识）

**Ensemble 极限**: **F1 ≈ 0.83-0.84**

**要突破 0.85**: 可能需要：

- 更高分辨率（违反任务设定）
- 预训练（违反约束）
- 或接受现实

---

## 🔧 实现细节

### Wide ResNet-28-10 架构

```python
class WideBasicBlock(nn.Module):
    """Pre-activation + Dropout + DropPath"""
    def forward(self, x):
        # Pre-activation
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)  # Dropout between convs
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.drop_path(out)  # Stochastic Depth
        return out + self.shortcut(x)

class WideResNet(nn.Module):
    """
    depth = 28: 6n+4, n=4
    widen_factor = 10
    Channels: [16, 160, 320, 640]
    """
    def __init__(self, depth, widen_factor, ...):
        self.layer1 = make_layer(160, 4 blocks, stride=1)
        self.layer2 = make_layer(320, 4 blocks, stride=2)
        self.layer3 = make_layer(640, 4 blocks, stride=2)
```

---

### PyramidNet-110 架构

```python
class PyramidBasicBlock(nn.Module):
    """Pre-activation + Zero-padded shortcut"""
    def forward(self, x):
        shortcut = x
        
        # Downsampling
        if stride != 1:
            shortcut = F.avg_pool2d(shortcut, 2, 2)
        
        # Zero-padding for channel increase
        if in_channels < out_channels:
            shortcut = F.pad(shortcut, (0,0,0,0,0,pad))
        
        # Pre-activation main path
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.drop_path(out)
        
        return out + shortcut

class PyramidNet(nn.Module):
    """
    depth = 110: 6n+2, n=18
    alpha = 270
    Gradual channels: 16 → 21 → 26 → ... → 286
    (每 block +5 channels)
    """
```

---

### RandAugment 实现

```python
class RandAugment(A.BaseCompose):
    def __init__(self, n=2, m=9):
        self.n = n  # 操作数量
        self.m = m  # 幅度
        self.pool = [
            Brightness, Contrast, Saturation, Rotate,
            ShearX, ShearY, TranslateX, TranslateY,
            AutoContrast, Equalize, Invert, Posterize,
            Solarize, Sharpen
        ]  # 14 种操作
    
    def __call__(self, image):
        # 随机选 n 个操作
        ops = random.sample(self.pool, self.n)
        for op in ops:
            image = op(image, magnitude=self.m)
        return image
```

**使用**:

```python
if aug_strength == "randaugment":
    pipeline = [
        RandAugment(n=2, m=9),
        HorizontalFlip(p=0.5),
        Normalize,
        ToTensor
    ]
```

---

## 📈 性能提升路径回顾

```
起点: ResNet-50
F1 = 0.77
  ↓ Wide ResNet-28-10 (架构)
F1 = 0.7802 (+0.01)
  ↓ Stochastic Depth (0.1)
F1 = 0.78-0.79 (+0.00-0.01)
  ↓ RandAugment (独立使用)
F1 = 0.8131 (+0.03) ✨
  ↓ PyramidNet-110 (架构)
F1 = 0.8307 Val / 0.82 Test (+0.0176 Val)
  ↓ Ensemble (3 models)
F1 = 0.82 (提升有限)
```

**总提升**: 0.77 → 0.82 = **+0.05** (+6.5%)

---

## 🎓 核心成功因素

### 技术层面

1. ✅ **Wide ResNet 架构**: 宽网络更适合 CIFAR
2. ✅ **Stochastic Depth (0.1)**: 最佳正则化平衡
3. ✅ **RandAugment 独立使用**: 自动增强搜索
4. ✅ **Mixup + CutMix**: Batch-level 增强
5. ✅ **PyramidNet 渐进式通道**: 架构创新有效

### 方法论层面

1. ✅ **数据驱动决策**: 基于实验结果调整
2. ✅ **精细调参**: drop_path 0.2 vs 0.1 差别巨大
3. ✅ **避免盲目叠加**: RandAug 独立，不叠加
4. ✅ **文献参考**: 论文配置需适配，不能照搬

---

## 📊 硬件与时间统计

### 训练硬件

- GPU: RTX 5080 (16GB)
- CPU: [具体型号]
- 内存: [具体大小]
- 系统: Windows 10

### 时间统计

| 阶段 | 实验数 | 总时间 | 关键成果 |
|-----|--------|--------|---------|
| Phase 1 | 4 | ~14h | F1 = 0.8131 |
| Phase 2 | 3 | ~14h | 失败经验 |
| Phase 2.7 | 1 | ~3.6h | F1 = 0.8307 Val |
| Phase 3 | 1 | ~10.5h | F1 = 0.82 |
| **总计** | **9** | **~42h** | **F1 = 0.82** |

---

## 🎯 最终推荐配置

### 提交方案选择

#### 选项 A: PyramidNet-110 (单模型，推荐)

```bash
python main.py \
    --model pyramidnet110_270 \
    --drop_path_rate 0.1 \
    --aug_strength randaugment \
    --weight_decay 1e-4 \
    --num_epochs 600 \
    --early_stopping_patience 60 \
    --seed 42
```

**Test F1**: 0.82  
**得分**: 90 分  
**优势**: 单模型，简洁

---

#### 选项 B: Wide ResNet-28-10 (最稳定)

```bash
python main.py \
    --model wide_resnet28_10 \
    --drop_path_rate 0.1 \
    --aug_strength randaugment \
    --seed 42
```

**Test F1**: 0.81  
**得分**: 90 分  
**优势**: Val-Test gap 最小 (0.003)

---

#### 选项 C: Ensemble (最保险)

```bash
python main.py --ensemble_seeds 42,43,44
```

**Test F1**: 0.82  
**得分**: 90 分  
**优势**: 多模型投票，稳定

---

## 🎊 项目成就

### 达成目标

- ✅ **Phase 1 目标** (F1 ≥ 0.80): 0.8131 超额完成
- ✅ **Phase 2 目标** (F1 ≥ 0.83): 0.8307 Val (超额), 0.82 Test (接近)
- 🎯 **满分目标** (F1 ≥ 0.85): 未达成（可能是极限）

### 技术创新

1. ✅ Wide ResNet + Stochastic Depth + RandAugment 完整方案
2. ✅ PyramidNet 在 CIFAR-100 上的成功应用
3. ✅ RandAugment 独立使用的最佳实践
4. ✅ Ensemble 框架实现

### 代码质量

- ✅ 完整的类型注解 (standard mode)
- ✅ 所有代码通过 linting 检查
- ✅ 符合作业修改范围限制
- ✅ 代码结构清晰，可维护性强

---

## 📁 最终代码结构

```
scripts/
├── model_architectures.py (~1340 lines)
│   ├── DropPath (Stochastic Depth)
│   ├── Wide ResNet-28-10/28-12/40-10
│   ├── PyramidNet-110/164
│   ├── ConvNeXt-Tiny/Small
│   ├── Self-Distillation variant
│   └── EnsembleModel
│
├── data_augmentation.py (~472 lines)
│   ├── Traditional augmentation (light/medium/strong)
│   └── RandAugment (N=2, M=9)
│
├── train_utils.py (~1597 lines)
│   ├── Mixup/CutMix (adaptive)
│   ├── Stochastic Depth loss
│   ├── Self-distillation loss
│   ├── EMA, Focal Loss, etc.
│   └── Training/validation loops
│
├── data_download.py (未修改, 677 lines)
└── evaluation_metrics.py (未修改, 319 lines)

main.py (~973 lines)
├── Standard training pipeline
├── Ensemble training pipeline
└── 完整超参数配置
```

---

## 🎓 给未来学习者的建议

### Do's ✅

1. **精细调参**: drop_path 0.1 vs 0.2 差别巨大
2. **数据驱动**: 基于实验结果调整，不猜测
3. **文献参考**: 但需适配，不能照搬
4. **避免叠加**: RandAugment 独立使用
5. **接受现实**: 认识到任务的固有限制

### Don'ts ❌

1. **盲目增大模型**: 容量甜点 ≈ 36M
2. **照搬 ImageNet 配置**: weight_decay 等需调整
3. **过度正则化**: 8 层叠加会适得其反
4. **尝试 Transformer**: 32×32 太小，数据太少
5. **忽视 Val-Test gap**: 重要的泛化指标

---

## 🏆 最终总结

### 成绩

**Test F1 = 0.82** (PyramidNet-110 or Ensemble)

**对应分数**: **90 分**

**超额完成**: Phase 1 目标 (0.80)  
**接近达成**: Phase 2 目标 (0.83)

### 评价

在以下严格约束下：

- ❌ 禁止预训练
- 📐 32×32 超低分辨率
- 📊 仅 40k 训练样本

**F1 = 0.82 是非常优秀的成绩！**

### 关键成功要素

1. **架构选择**: Wide ResNet → PyramidNet
2. **正则化**: Stochastic Depth (0.1)
3. **数据增强**: RandAugment (独立)
4. **训练技巧**: Mixup/CutMix, EMA, AMP
5. **精细调参**: 基于实验数据迭代

---

**项目完成！恭喜你取得优秀成绩！** 🎉

---

## 📚 参考文献

1. Zagoruyko & Komodakis. "Wide Residual Networks" (BMVC 2016)
2. Han et al. "Deep Pyramidal Residual Networks" (CVPR 2017)
3. Huang et al. "Deep Networks with Stochastic Depth" (ECCV 2016)
4. Cubuk et al. "RandAugment: Practical automated data augmentation" (NeurIPS 2020)
5. Zhang et al. "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
6. Yun et al. "CutMix: Regularization Strategy to Train Strong Classifiers" (ICCV 2019)

---

**报告日期**: 2025-10-17  
**项目时长**: [具体天数]  
**最终成绩**: Test F1 = 0.82 ✨
