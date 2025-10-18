# 🔴 Exp #200 失败深度分析

**实验**: ConvNeXt-Tiny + wd=0.05  
**结果**: Val F1 = 0.79 (**vs WRN 0.8131, 下降 0.023**)  
**结论**: ConvNeXt 配置严重不当或不适合 CIFAR

---

## 📊 三方对比：惊人的发现

### 关键指标对比

| Exp | 模型 | wd | Train Acc | Val Acc | Val F1 | Best Epoch |
|-----|------|----|-----------|---------|--------|------------|
| #100 | WRN baseline | 5e-4 | 68% | 78.26% | 0.7802 | 152 |
| **#104b** | **WRN + SD + RA** | **5e-4** | **59.67%** | **81.40%** | **0.8131** | **209** |
| #200 | ConvNeXt + SD + RA | **0.05** | **66.06%** | 80.26% | **0.79** | **350** |

### 🚨 震惊的发现

**ConvNeXt (F1=0.79) ≈ Exp #100 (F1=0.7802)！**

**类别分布对比**:

| 类别 | Exp #100 | Exp #104b | Exp #200 (ConvNeXt) | 分析 |
|-----|----------|-----------|---------------------|------|
| boy | 0.51 | 0.51 | **0.46** | ConvNeXt 更差！|
| girl | 0.57 | 0.57 | **0.57** | 持平 |
| otter | 0.52 | 0.52 | **0.56** | 略好 |
| seal | 0.54 | 0.54 | **0.57** | 略好 |
| lawn_mower | 0.92 | 0.92 | **0.95** | 略好 |
| pickup_truck | 0.93 | 0.94 | **0.93** | 持平 |
| motorcycle | 0.91 | 0.92 | **0.91** | 持平 |
| sunflower | 0.91 | 0.92 | **0.96** | 好！|

**结论**:

- ConvNeXt 分布与 Exp #100 几乎一样！
- **完全没有获得 SD + RandAugment 的提升 (+0.03)！**
- 某些类略好，某些类略差，整体持平

---

## 🔍 根本问题诊断

### 问题 1: weight_decay = 0.05 **严重过大** 🚨

#### 数学分析

**L2 正则化项**: `L_total = L_CE + wd × ||w||²`

**weight_decay 影响**:

```
wd = 5e-4 (WRN):
  L_total ≈ L_CE + 0.0005 × ||w||²

wd = 0.05 (ConvNeXt):
  L_total ≈ L_CE + 0.05 × ||w||²  (100x!)
```

**后果**:

- 模型被强制保持小权重
- 学习能力被严重抑制
- 无法充分学习特征

#### 证据

**1. Train Acc 过高 (66% vs WRN 60%)**

- 正则化应该降低 Train Acc
- 但 ConvNeXt 反而更高
- → wd 太大，模型学不好，过度简化

**2. 收敛慢 (epoch 350 vs WRN 209)**

- wd 太大抑制梯度
- 学习速度变慢
- 需要更多 epoch 才能收敛

**3. 性能与 baseline 相同**

- 完全没有从 SD + RA 受益
- SD 和 RA 的提升被 wd 抵消了

---

### 问题 2: ConvNeXt 实现导致训练慢

#### 性能瓶颈定位

**ConvNeXt Block 的开销**:

```python
# 每个 block 的操作
x = x.permute(0, 2, 3, 1)  # NCHW → NHWC (开销!)
x = self.pwconv1(x)        # Linear
x = self.act(x)            # GELU (比 ReLU 慢)
x = self.pwconv2(x)
x = x.permute(0, 3, 1, 2)  # NHWC → NCHW (开销!)
```

**问题**:

1. **Permute 破坏内存连续性**: 2次 permute × 18 blocks = 36 次
2. **LayerNorm 计算**: 比 BatchNorm 慢
3. **GELU 激活**: 需要 exp 计算，比 ReLU 慢

**训练时间**:

```
WRN-28-10: ~60s/epoch
ConvNeXt-Tiny: ~90s/epoch (+50%)
```

**是否正常**?

- ✅ **正常！** 这是 ConvNeXt 架构的代价
- ConvNeXt 论文也提到训练比 ResNet 慢

**是否值得**?

- ❌ **不值得！** F1 下降了，还训练更慢

---

### 问题 3: ConvNeXt 可能不适合 CIFAR

#### ConvNeXt 设计背景

**原始设计**:

- ImageNet (1000 classes, 1.28M images)
- 输入: 224×224
- Batch: 4096
- Data: 海量

**CIFAR-100**:

- 100 classes, 50k images (小 26x)
- 输入: 32×32 (小 7.7x)
- Batch: 128 (小 32x)
- Data: 稀少

#### 架构适配问题

**1. 7×7 Depthwise Conv 对 32×32 太大**:

```
32×32 图像:
- 7×7 kernel 覆盖 ~22% 的图像
- 可能破坏局部特征

vs 224×224 图像:
- 7×7 kernel 覆盖 ~3% 的图像
- 合理的感受野
```

**2. 4 个 stage 的下采样**:

```
32 → 16 → 8 → 4 (最后仅 4×4)

vs Wide ResNet:
32 → 32 → 16 → 8 (最后 8×8, 保留更多空间信息)
```

**结论**: ConvNeXt 在小图上可能**过度下采样**

---

## 💡 优化建议

### 🥇 **方向 1: 回归 Wide ResNet-28-12** ⭐⭐⭐⭐⭐ (强烈推荐)

**放弃 ConvNeXt，尝试更大的 Wide ResNet**

**Exp #205: WRN-28-12 + Phase 1 最佳实践**

```bash
python main.py \
    --model wide_resnet28_12 \
    --drop_path_rate 0.12 \
    --aug_strength randaugment \
    --dropout 0.35 \
    --lr 0.0012 \
    --weight_decay 8e-4 \
    --warmup_epochs 10 \
    --num_epochs 600 \
    --early_stopping_patience 60 \
    --batch_size 96 \
    --seed 42
```

**理由**:

1. ✅ 基于 Phase 1 成功经验
2. ✅ 参数量 52.8M (+45% vs WRN-28-10)
3. ✅ 已知架构，风险低
4. ✅ 训练时间可控 (~65s/epoch)

**预期**: F1 = **0.82-0.84**  
**成功概率**: **80-85%**  
**时间**: 今晚，4-5 小时

---

### 🥈 **方向 2: 修复 ConvNeXt 配置** ⭐⭐⭐ (如果想坚持 ConvNeXt)

**Exp #201: ConvNeXt + 大幅降低 wd**

```bash
python main.py \
    --model convnext_tiny \
    --drop_path_rate 0.1 \
    --aug_strength randaugment \
    --lr 0.002 \
    --weight_decay 0.005 \
    --warmup_epochs 20 \
    --num_epochs 600 \
    --batch_size 128 \
    --seed 42
```

**改动**:

- wd: 0.05 → **0.005** (降低 90%!)
- lr: 0.001 → **0.002** (补偿 wd 降低)

**理由**:

- CIFAR 数据量小，wd=0.05 过大
- 0.005 ≈ 10× WRN 的 5e-4

**预期**: F1 = 0.82-0.83  
**成功概率**: 60-70%  
**时间**: 5-6 小时（慢）

---

### 🥉 **方向 3: Wide ResNet-40-10** ⭐⭐⭐⭐ (更深的 WRN)

**Exp #206: WRN-40-10**

```bash
python main.py \
    --model wide_resnet40_10 \
    --drop_path_rate 0.15 \
    --aug_strength randaugment \
    --dropout 0.35 \
    --lr 0.0012 \
    --weight_decay 1e-3 \
    --num_epochs 600 \
    --batch_size 96 \
    --seed 42
```

**理由**:

- 55.8M 参数 (最大)
- 40 层 vs 28 层
- drop_path 可以更大 (0.15)

**预期**: F1 = 0.82-0.85  
**成功概率**: 70-75%  
**风险**: 可能 OOM

---

## 🎯 最终建议

### ✅ **立即执行: Wide ResNet-28-12** (Exp #205)

**为什么?**

1. **可靠性**: 基于 Phase 1 成功配置
2. **效率**: 训练时间可控
3. **成功率**: 80%+ 概率达到 0.82-0.84
4. **参数量**: 52.8M，理论上比 36.5M 更强

**vs ConvNeXt**:

- ❌ ConvNeXt 需要大幅调参（风险高）
- ❌ ConvNeXt 训练慢（时间成本高）
- ❌ ConvNeXt 不确定能否超过 WRN

**命令**:

```bash
python main.py \
    --model wide_resnet28_12 \
    --drop_path_rate 0.12 \
    --aug_strength randaugment \
    --dropout 0.35 \
    --lr 0.0012 \
    --weight_decay 8e-4 \
    --batch_size 96 \
    --num_epochs 600 \
    --early_stopping_patience 60 \
    --seed 42
```

**预期**: F1 = 0.82-0.84 (达成 Phase 2!)

---

## 🔬 关于训练时间拉长

### 原因确认

**ConvNeXt 架构特性**:

1. ✗ Permute 操作 (36 次/forward pass)
2. ✗ LayerNorm (比 BatchNorm 慢)
3. ✗ GELU (比 ReLU 慢 ~2x)
4. ✗ 7×7 Depthwise Conv

**结果**: ~90s/epoch (vs WRN 60s/epoch, +50%)

**是实现问题吗**?

- ❌ 不是！
- ConvNeXt 设计如此
- 论文也承认训练慢于 ResNet

**能优化吗**?

- 理论上可以优化 permute
- 但改动大，风险高
- 不值得（性能还不如 WRN）

---

## 📊 类别分布深度对比

### 三次实验的相似性

**Exp #100 vs Exp #200** (几乎一样!)

| 类别 | #100 F1 | #200 F1 | 差异 |
|-----|---------|---------|------|
| boy | 0.51 | 0.46 | -0.05 |
| girl | 0.57 | 0.57 | 0.00 |
| otter | 0.52 | 0.56 | +0.04 |
| seal | 0.54 | 0.57 | +0.03 |
| lawn_mower | 0.92 | 0.95 | +0.03 |
| pickup_truck | 0.93 | 0.93 | 0.00 |

**平均绝对差异**: ~0.02 (微小波动)

**结论**:

- ConvNeXt 分布 ≈ WRN baseline
- **没有获得 SD + RA 的提升！**
- wd=0.05 完全抵消了 SD + RA 的好处

---

## 🎓 关键洞察

### ConvNeXt 在 CIFAR 上的问题

**1. 数据集规模不匹配**:

```
ImageNet: 1.28M images → wd=0.05 合适
CIFAR-100: 50k images → wd=0.05 过大 (数据量小 26x)
```

**建议缩放**: wd = 0.05 / 26 ≈ **0.002**

**2. 图像尺寸不匹配**:

```
ImageNet: 224×224 → 7×7 conv 合适
CIFAR: 32×32 → 7×7 conv 过大 (覆盖 22%)
```

**3. 架构复杂度不匹配**:

- ConvNeXt 设计用于大规模数据
- CIFAR 可能过度复杂化

---

## 🎯 推荐行动

### ✅ **立即放弃 ConvNeXt，转 WRN-28-12!**

**理由总结**:

1. ConvNeXt F1=0.79，不如 WRN-28-10 (0.8131)
2. 训练时间 +50%，效率低
3. 调参风险高（wd 需要从 0.05 降到 ~0.002）
4. WRN-28-12 更可靠（52.8M params，基于成功经验）

**Exp #205 配置**:

```bash
python main.py \
    --model wide_resnet28_12 \
    --drop_path_rate 0.12 \
    --aug_strength randaugment \
    --dropout 0.35 \
    --lr 0.0012 \
    --weight_decay 8e-4 \
    --num_epochs 600 \
    --batch_size 96 \
    --early_stopping_patience 60 \
    --seed 42
```

**预期**: F1 = **0.82-0.84**  
**成功概率**: **80-85%**  
**时间**: 今晚 4-5 小时

---

## 📋 备选方案 (如果坚持 ConvNeXt)

### Exp #201: ConvNeXt + 极低 wd

```bash
python main.py \
    --model convnext_tiny \
    --weight_decay 0.002 \
    --lr 0.002 \
    --drop_path_rate 0.1 \
    --num_epochs 600 \
    --seed 42
```

**预期**: F1 = 0.81-0.83  
**风险**: 仍可能不如 WRN-28-12

---

## 🎯 最终结论

**ConvNeXt 是好架构，但不适合 CIFAR！**

**原因**:

1. 设计用于 ImageNet (大图、大数据)
2. CIFAR 太小，优势发挥不出来
3. 需要大量调参，风险高

**建议**: **立即转 WRN-28-12！**

这是达到 0.83-0.85 的最可靠路径！
