# 关键发现: Dropout 与 Drop_path 的权衡

**发现时间**: 2025-10-24  
**实验阶段**: Phase 1 - 正则化优化  
**重要性**: ⭐⭐⭐⭐⭐

---

## 🔍 核心发现

### 发现 1: Dropout 调整的平台期

**实验数据**:

```
dropout 0.3  → F1 = 0.82 (baseline)
dropout 0.2  → F1 = 0.81
dropout 0.15 → F1 = 0.81
```

**关键洞察**:

1. ✅ dropout 0.2 和 0.15 **效果完全相同**
2. ✅ 说明 0.15-0.2 区间是"平台期"
3. ✅ 继续降低 dropout 无意义
4. ⚠️ dropout 不是主要瓶颈

**影响**:

- 确定 dropout 0.2 为最优值
- 需要寻找新的优化点
- Drop_path 成为下一个候选

---

### 发现 2: Detail vs Local Classes 的系统性权衡

**类别表现对比**:

| 配置 | Detail Classes | Local Classes | Overall |
|------|----------------|---------------|---------|
| dropout 0.3 | 0.610 | 0.810 | 0.82 |
| dropout 0.2 | 0.666 (+0.056) | 0.763 (-0.047) | 0.81 |
| dropout 0.15 | 0.664 (+0.054) | 0.760 (-0.050) | 0.81 |

**洞察**:

1. ✅ Detail classes 随 dropout 降低而提升
2. ✅ Local classes 随 dropout 降低而下降
3. ❌ 但 Detail 的提升无法弥补 Local 的损失
4. ❌ Overall F1 反而下降 0.01

**深层原因**:

```
Detail classes (人类/小动物):
- 需要更细致的特征学习
- 受益于更大的模型容量
- dropout ↓ → 容量 ↑ → 性能 ↑

Local classes (树木/纹理):
- 依赖局部模式和纹理
- 过强的容量可能过拟合
- dropout ↓ → 过拟合 ↑ → 性能 ↓

其他 85 个类:
- 表现稳定，略有波动
- 微小的下降累积成整体下降
```

---

### 发现 3: Detail Classes 的三个瓶颈

#### 瓶颈 A: 物理分辨率限制 (不可解决)

```
32×32 CIFAR-100 图片:
- 面部区域: ~20×20 像素
- 眼睛: ~3×3 像素
- 鼻子: ~2×2 像素
- 嘴巴: ~3×2 像素

人类类别:
- boy (F1 = 0.46) ❌❌
- girl (F1 = 0.57) ❌
- man (F1 = 0.58) ❌
- woman (F1 = 0.60) ⚠️
- baby (F1 = 0.60) ⚠️

结论: 信息不足以精确分类
```

**无法通过调参解决！**

---

#### 瓶颈 B: Mixup/CutMix 的破坏性 (可能可解决)

```
Mixup (alpha=0.4):
- 混合两张图片 (人脸 A + 人脸 B)
- 细节进一步模糊
- 类别边界更难区分

CutMix (alpha=1.0):
- 切割并拼接
- 可能切掉关键特征 (眼睛/嘴巴)
- 对小物体破坏性极强

实验证据:
- dual-loader 禁用 Mixup: boy F1 提升 0.05
- 但整体下降 (类别不平衡)
```

**可能的解决方案**:

- 降低 Detail classes 的 Mixup alpha (0.4 → 0.2)
- 完全禁用 Detail classes 的 CutMix
- 需要平衡泛化和细节保留

---

#### 瓶颈 C: 模型容量不足 (可能可解决)

```
当前正则化:
- Dropout: 0.2 (80% 神经元)
- Drop_path: 0.1 (90% 层)
- 总容量: 0.8 × 0.9 = 72%

假设 drop_path 0.0:
- Dropout: 0.2 (80% 神经元)
- Drop_path: 0.0 (100% 层)
- 总容量: 0.8 × 1.0 = 80%

容量提升: +8%
预期 F1 提升: +1-2%
```

**当前正在测试！**

---

### 发现 4: Drop_path 在 WRN-28 中可能过强

**理论分析**:

```
Drop_path 的最佳实践:
  网络深度      推荐 drop_path
  ResNet-34     0.0
  ResNet-50     0.05-0.1
  ResNet-110    0.1-0.2
  ResNet-152    0.2

WRN-28-10:
  深度: 28 层 (接近 ResNet-34)
  宽度: 10× (参数量大)
  当前 drop_path: 0.1 (可能过高)
```

**影响**:

1. 每个 batch 随机丢弃 ~3 层
2. 梯度流不稳定
3. 深层特征学习困难
4. 限制了模型容量

**假设**: 降低或禁用 drop_path 可能提升性能

---

## 🎯 实验支撑

### 实验 1: Dropout 0.3 → 0.2 (F1: 0.82 → 0.81)

**类别级别分析**:

```
提升类别 (F1 增加):
- baby: +0.04
- man: +0.03
- woman: +0.05
- mouse: +0.02
- shrew: +0.03

平均: +0.034 (Detail classes)

下降类别 (F1 下降):
- oak_tree: -0.05
- willow_tree: -0.04
- pine_tree: -0.03
- bus: -0.04

平均: -0.04 (Local classes)

持平类别:
- 大部分高性能类 (F1 > 0.90)
- 微小波动 (±0.01)
```

**结论**: Detail ↑ Local ↓ 是系统性的权衡

---

### 实验 2: Dropout 0.2 → 0.15 (F1: 0.81 → 0.81)

**类别级别分析**:

```
Detail classes:
  0.2:  0.666
  0.15: 0.664
  差异: -0.002 (误差范围内)

Local classes:
  0.2:  0.763
  0.15: 0.760
  差异: -0.003 (误差范围内)

低性能类别:
  0.2:  6 个 (F1 < 0.65)
  0.15: 6 个 (相同)
```

**结论**: 0.15 和 0.2 完全相同，说明 dropout 优化已饱和

---

## 💡 关键洞察

### 洞察 1: 正则化的叠加效应

**当前正则化堆栈**:

```
1. Dropout: 0.2 (神经元级别)
2. Drop_path: 0.1 (层级别)
3. Mixup: alpha=0.4 (数据级别)
4. CutMix: alpha=1.0 (数据级别)
5. RandAugment: N=2, M=9 (数据级别)

总正则化强度: 非常高
```

**问题**:

- 过多的正则化可能限制模型容量
- Detail classes 需要更多容量来学习细节
- 可能需要减少某些正则化

**优先级**:

1. Drop_path (层级正则化) - 最容易调整
2. Mixup alpha (数据正则化) - 需要修改代码
3. RandAugment (数据正则化) - 影响不确定

---

### 洞察 2: 模型容量 vs 泛化能力

**权衡曲线**:

```
dropout 0.1 (高容量)
  ↑
  │  Detail ✅ Local ❌
  │
dropout 0.2 (平衡)
  │
  │  Detail ⚠️ Local ⚠️
  │
dropout 0.3 (高泛化)
  ↓
     Detail ❌ Local ✅
```

**最优点**: 可能在 drop_path 维度，而非 dropout 维度

---

### 洞察 3: 不同类别对容量的需求不同

**高需求类别** (受益于高容量):

- Detail classes (人类/小动物)
- 高类间相似类 (otter/seal/shark)
- 低分辨率类 (bowl/cup/plate)

**低需求类别** (受益于高正则化):

- 高对比度类 (sunflower/lawn_mower)
- 形状鲜明类 (bicycle/motorcycle)
- 纹理类 (树木/植物) ← Local classes

**矛盾**:

- 无法同时满足两类需求
- 需要找到最佳平衡点
- 或使用 Ensemble 融合不同配置

---

## 🎯 下一步策略

### 策略 1: Drop_path 优化 (当前)

```bash
# 当前实验
python main.py --dropout 0.2 --drop_path_rate 0.0

# 预期
F1 = 0.825-0.835
Detail: +0.03-0.04
Local: +0.01-0.02
```

**理由**:

- Drop_path 可能是主要瓶颈
- WRN-28 不需要强 drop_path
- 禁用可能释放 8% 容量

---

### 策略 2: 基于类别的增强强度 (未来)

**概念**:

```python
# Detail classes
mixup_alpha = 0.2  # 降低混合强度
use_cutmix = False  # 禁用切割

# Local classes
mixup_alpha = 0.3
use_cutmix = True
cutmix_alpha = 0.8

# Others
mixup_alpha = 0.4
use_cutmix = True
cutmix_alpha = 1.0
```

**预期**:

- Detail: +0.03-0.05
- 其他: -0.01 (可接受)
- Overall: +0.02-0.03

---

### 策略 3: Ensemble 不同正则化配置

**概念**:

```
Model 1: dropout 0.2, drop_path 0.0 (高容量)
  → 擅长 Detail classes

Model 2: dropout 0.3, drop_path 0.1 (高泛化)
  → 擅长 Local classes

Model 3: dropout 0.25, drop_path 0.05 (平衡)
  → 擅长其他类

Ensemble: 软投票
  → 综合优势
```

**预期**:

- 单模型: 0.83-0.84
- Ensemble: 0.84-0.86

---

## 📊 数据总结

### 完整对比表

| 配置 | Overall | Detail | Local | 低性能类 | 高性能类 | 训练时间 |
|------|---------|--------|-------|----------|----------|----------|
| **0.3/0.1** | **0.82** | 0.610 | **0.810** | 未知 | 未知 | 2h |
| **0.2/0.1** | 0.81 | **0.666** | 0.763 | 6 | 18 | 2h |
| **0.15/0.1** | 0.81 | 0.664 | 0.760 | 6 | 未知 | 2h |
| **0.2/0.0** | **?** | **?** | **?** | **?** | **?** | 2h+ |

---

## 📝 元数据

- **文档类型**: 关键发现笔记
- **创建时间**: 2025-10-24
- **实验次数**: 3 轮完成, 1 轮进行中
- **置信度**: 高 (数据充分)
- **影响**: 指导后续所有优化策略
