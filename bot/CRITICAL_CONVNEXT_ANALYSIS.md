# 🚨 ConvNeXt-Tiny 失败分析

**实验**: Exp #200 - ConvNeXt-Tiny  
**结果**: Val F1 = **0.79** (vs WRN 0.8131, **下降 0.023!**)  
**状态**: 🔴 **失败 - 不如 Phase 1**

---

## 📊 性能对比

| 实验 | 模型 | 参数 | Train Acc | Val Acc | Val F1 | Best Epoch | 训练时间/epoch |
|-----|------|------|-----------|---------|--------|------------|---------------|
| **#104b** | WRN-28-10 | 36.5M | 59.67% | 81.40% | **0.8131** | 209/300 | ~60s |
| **#200** | ConvNeXt-Tiny | 28M | 66.06% | 80.26% | **0.79** | 350/600 | ~90s? |

**问题**:

1. 🔴 **性能下降** -0.023 F1
2. 🔴 **训练变慢** ~50% 耗时增加
3. 🔴 **收敛延迟** Best epoch 从 209 → 350

---

## 🔍 问题 1: 训练时间拉长原因

### 可能原因分析

#### A. ConvNeXt 计算复杂度更高

**证据**:

```python
# ConvNeXt Block
1. Depthwise Conv 7×7 (vs WRN 3×3)
2. NHWC ↔ NCHW 转换 (permute 操作)
3. LayerNorm 计算 (vs BatchNorm)
4. GELU (vs ReLU, 更复杂)
```

**FLOPs 估算**:

- Wide ResNet-28-10: ~5.2G FLOPs
- ConvNeXt-Tiny: ~4.5G FLOPs (理论上更少)

**但实际慢的原因**:

- ✗ Permute 操作破坏内存连续性
- ✗ LayerNorm 不如 BatchNorm 优化
- ✗ GELU 需要 exp 计算

#### B. 实现问题？

让我检查 ConvNeXt 实现...

**潜在问题**:

- Permute 操作频繁 (每个 block 2次)
- LayerNorm2d 可能不够高效

**建议**: 接受训练时间增加（这是 ConvNeXt 的代价）

---

## 🔍 问题 2: 性能下降原因

### 关键发现

**Train Acc 对比**:

```
WRN-28-10: 59.67% (with wd=5e-4)
ConvNeXt:  66.06% (with wd=0.05)
```

**异常**: ConvNeXt Train Acc 更高，但 Val F1 更低！

### 诊断

**1. weight_decay = 0.05 可能不适合 CIFAR**

**证据**:

- Train Acc 66% > WRN 60% (正则化不够强)
- Val F1 0.79 < WRN 0.8131 (泛化变差)
- → **欠拟合或配置不当**

**ConvNeXt 论文背景**:

- ImageNet: 224×224, batch=4096, wd=0.05
- CIFAR-100: 32×32, batch=128, wd=0.05

**问题**: CIFAR 数据量小得多，wd=0.05 可能**太大**导致模型学不好！

**2. Best epoch = 350 (vs WRN 209)**

说明收敛慢，可能：

- 学习率不够
- weight_decay 抑制学习
- 架构不适合小图

---

## 📊 类别分布分析

### 需要看 training_metrics.txt

等提供后再分析...

**猜测**:

- 困难类别 (boy, girl, etc.) 可能更差
- 整体分布可能更平但没有特别优秀的类

---

## 💡 优化建议

### 🔥 方向 1: 大幅降低 weight_decay ⭐⭐⭐⭐⭐

**问题诊断**: wd=0.05 对 CIFAR 太大！

**Exp #201: ConvNeXt + wd 调优**

```bash
# 配置 A: wd=0.01 (降低 80%)
python main.py \
    --model convnext_tiny \
    --drop_path_rate 0.1 \
    --weight_decay 0.01 \
    --warmup_epochs 20 \
    --num_epochs 600 \
    --seed 42

# 配置 B: wd=0.005 (降低 90%)
python main.py \
    --model convnext_tiny \
    --drop_path_rate 0.1 \
    --weight_decay 0.005 \
    --warmup_epochs 20 \
    --num_epochs 600 \
    --seed 43

# 配置 C: wd=5e-4 (与 WRN 相同)
python main.py \
    --model convnext_tiny \
    --drop_path_rate 0.1 \
    --weight_decay 5e-4 \
    --warmup_epochs 20 \
    --num_epochs 600 \
    --seed 44
```

**理由**:

- ConvNeXt 论文用 wd=0.05 是基于 ImageNet (海量数据)
- CIFAR-100 只有 50k 训练样本，太少了
- 需要更小的 wd

**预期**: F1 = 0.82-0.84

---

### 🔥 方向 2: 增大学习率 ⭐⭐⭐⭐

**ConvNeXt 论文**: lr = 4e-3 (我们用 0.001)

**配置**:

```bash
python main.py \
    --model convnext_tiny \
    --drop_path_rate 0.1 \
    --lr 0.002 \
    --weight_decay 0.01 \
    --warmup_epochs 20 \
    --num_epochs 600 \
    --seed 42
```

**预期**: 加快收敛，提升性能

---

### 🔥 方向 3: 回归 Wide ResNet-28-12 ⭐⭐⭐⭐⭐

**放弃 ConvNeXt，尝试更大的 Wide ResNet**

**Exp #205: WRN-28-12**

```bash
python main.py \
    --model wide_resnet28_12 \
    --drop_path_rate 0.12 \
    --aug_strength randaugment \
    --lr 0.0012 \
    --weight_decay 8e-4 \
    --dropout 0.35 \
    --num_epochs 600 \
    --batch_size 96 \
    --seed 42
```

**理由**:

- 52.8M 参数 (+45% vs WRN-28-10)
- 已知架构，风险低
- 基于 Phase 1 成功经验

**预期**: F1 = 0.82-0.84

**成功概率**: 75-80% (比 ConvNeXt 更可靠)

---

## 🎯 我的强烈建议

### ✅ **优先级排序**

**1. 立即尝试: WRN-28-12** (今晚)

- 风险最低
- 成功概率最高
- 基于 Phase 1 成功配置

**2. 并行尝试: ConvNeXt + wd 调优** (明天)

- 降低 wd 到 0.005-0.01
- 可能挽救 ConvNeXt

**3. 备选: ConvNeXt + lr 提升**

- 仅在 wd 调优失败后

---

## 🔬 关于训练时间

### 为什么 ConvNeXt 慢？

**1. 架构复杂度**:

- Permute 操作 (NCHW ↔ NHWC) 破坏缓存
- LayerNorm 计算量大
- GELU 需要指数计算

**2. torch.compile 可能无效**:

- 动态 permute 可能阻止完整图编译
- 回退到 eager mode

**是否需要修复**?

- ❌ 不一定
- ConvNeXt 设计如此，训练慢是正常的
- 除非性能提升值得

---

## 📋 立即行动

### 🔥 Exp #205: Wide ResNet-28-12

```bash
python main.py \
    --model wide_resnet28_12 \
    --drop_path_rate 0.12 \
    --aug_strength randaugment \
    --dropout 0.35 \
    --lr 0.0012 \
    --weight_decay 8e-4 \
    --num_epochs 600 \
    --early_stopping_patience 60 \
    --batch_size 96 \
    --seed 42
```

**预期**: F1 = 0.82-0.84  
**时间**: 4-5 小时  
**成功概率**: 75-80%

---

**建议**: 优先尝试 WRN-28-12，ConvNeXt 可能不适合 CIFAR！
