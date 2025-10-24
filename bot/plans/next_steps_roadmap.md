# 后续优化路线图

**创建时间**: 2025-10-24  
**当前进度**: dropout 0.2 + drop_path 0.0 实验进行中  
**目标**: F1 ≥ 0.85  
**当前最佳**: F1 = 0.82 (dropout 0.3, drop_path 0.1)

---

## 🗺️ 总体路线图

```
Phase 1: 正则化优化 (当前阶段)
  ├─ dropout 调整 ✅ (完成: 0.2 最优)
  ├─ drop_path 调整 🔄 (进行中: 测试 0.0)
  └─ 确定最优正则化配置

Phase 2: 架构优化 (下一阶段)
  ├─ 注意力机制 (SE-Net / CBAM)
  ├─ 增强策略微调 (Mixup alpha)
  └─ 学习率调度优化

Phase 3: 模型融合 (最终阶段)
  ├─ 单模型优化到 0.83-0.84
  ├─ Ensemble (3-5 个模型)
  └─ 突破 0.85 目标

时间预算: 总计 8-12 轮实验
预计总时间: 16-24 小时训练
```

---

## 📍 当前位置

### 已完成的里程碑

#### ✅ Phase 0: 基础建立 (已完成)

```
✅ Wide ResNet-28-10 实现
✅ Self-Distillation (SD) 集成
✅ RandAugment (RA) 集成
✅ Mixup/CutMix 集成
✅ EMA 集成
✅ AMP + Gradient Clipping
✅ 基础性能: F1 = 0.82
```

#### ✅ Phase 1.1: Dropout 调整 (已完成)

| 实验 | 配置 | F1 | Detail | Local | 结论 |
|------|------|-----|--------|-------|------|
| baseline | dropout 0.3 | 0.82 | 0.610 | 0.810 | 最佳 overall |
| exp-1 | dropout 0.15 | 0.81 | 0.664 | 0.760 | Detail ↑, Local ↓ |
| exp-2 | dropout 0.2 | 0.81 | 0.666 | 0.763 | 与 0.15 无差异 |

**关键发现**:

- Dropout 0.2 是合理值
- 进一步降低 dropout 无效
- Detail classes 提升饱和 (+0.056)
- 需要寻找新的优化点

---

#### 🔄 Phase 1.2: Drop_path 调整 (进行中)

**当前实验**: dropout 0.2 + drop_path 0.0

**目标**:

- 验证 drop_path 是否是新的瓶颈
- 尝试释放更多模型容量
- 预期 F1: 0.825-0.835

**备选方案**:

- 如果 0.0 过拟合 → 尝试 0.05
- 如果 0.0 成功 → 进入 Phase 2

---

## 🎯 Phase 2: 架构优化 (待启动)

### 优先级 1: SE-Net 重新评估 ⭐⭐⭐⭐⭐

**前提**: Phase 1 找到最优正则化配置

#### 实验 2.1: 最优配置 + SE-Net

```bash
python main.py --model wide_resnet28_10 \
  --dropout <最优> \
  --drop_path_rate <最优> \
  --use_se \
  --se_reduction 16
```

**理由**:

1. 之前 SE-Net 失败可能是正则化过强
2. SE-Net 提供 channel-wise attention
3. 对 Detail classes 应该有帮助
4. 参数增加少 (~5%)

**预期**:

- F1: +0.01-0.02 (vs 最优 baseline)
- Detail classes: +0.02-0.03
- 训练时间: +5-10%

**成功标准**: F1 ≥ 0.83

---

### 优先级 2: Mixup Alpha 微调 ⭐⭐⭐⭐

**需要修改代码** (`train_utils.py`)

#### 实验 2.2: Detail classes 使用更低的 Mixup alpha

**修改**:

```python
# train_utils.py, line ~315
def adaptive_mixup_cutmix(...):
    if batch has detail_sensitive classes:
        mixup_alpha = 0.2  # 从 0.4 降低
        use_cutmix = False  # 禁用 cutmix
    elif batch has local_feature classes:
        mixup_alpha = 0.2
        use_cutmix = True
        cutmix_alpha = 0.8  # 从 1.0 降低
    else:
        mixup_alpha = 0.4
        use_cutmix = True
        cutmix_alpha = 1.0
```

**理由**:

1. Detail classes 对 Mixup 极度敏感
2. alpha=0.4 可能过强，破坏细节
3. alpha=0.2 保留更多原始特征
4. 之前 alpha=0.6 失败，说明 0.4 附近是边界

**预期**:

- Detail classes: +0.02-0.04
- 其他类: -0.005-0.01 (泛化略降)
- Overall: +0.01-0.02

**风险**: 需要修改代码，可能超出作业范围

---

### 优先级 3: RandAugment 调整 ⭐⭐⭐

#### 实验 2.3: 提高 RandAugment 强度

```bash
python main.py --model wide_resnet28_10 \
  --dropout <最优> \
  --drop_path_rate <最优> \
  --randaugment_n 3 \
  --randaugment_m 11
```

**理由**:

- 当前 N=2, M=9 可能偏保守
- 更强的增强可能提升泛化
- 对高性能类别 (>0.90) 应该有帮助

**风险**:

- 可能进一步损害 Detail classes
- 需要权衡利弊

**预期**:

- 高性能类别: +0.01
- Detail classes: -0.01-0.02
- Overall: +0.005-0.01 (不确定)

---

### 优先级 4: 学习率调度优化 ⭐⭐

#### 实验 2.4: Cosine Annealing with Restarts

```bash
python main.py --model wide_resnet28_10 \
  --dropout <最优> \
  --drop_path_rate <最优> \
  --scheduler cosine_restart \
  --restart_period 50
```

**理由**:

- 周期性重启可能跳出局部最优
- 多个学习率周期可能找到更好的解

**预期**:

- F1: +0.005-0.015
- 更平滑的收敛

---

## 🎯 Phase 3: 模型融合 (最终阶段)

### 优先级 1: Ensemble ⭐⭐⭐⭐⭐

**前提**: 单模型达到 F1 = 0.83-0.84

#### 实验 3.1: Ensemble (3 模型)

```bash
python main.py --model wide_resnet28_10 \
  --dropout <最优> \
  --drop_path_rate <最优> \
  --ensemble_seeds "42,123,456"
```

**配置**:

- 3 个模型，不同随机种子
- Soft Voting (平均概率)
- 相同的最优超参数

**预期**:

- 单模型: 0.83-0.84
- Ensemble: +0.01-0.02
- **最终 F1: 0.84-0.85** ✅

**时间**: 3× 单模型时间 (~6 小时)

---

#### 实验 3.2: Ensemble (5 模型) - 如果需要

```bash
python main.py --model wide_resnet28_10 \
  --dropout <最优> \
  --drop_path_rate <最优> \
  --ensemble_seeds "42,123,456,789,1024"
```

**预期**:

- 相比 3 模型 Ensemble: +0.005-0.01
- **最终 F1: 0.845-0.86** ✅✅

**时间**: 5× 单模型时间 (~10 小时)

---

### 优先级 2: 不同架构 Ensemble ⭐⭐⭐

#### 实验 3.3: WRN + PyramidNet Ensemble

```bash
# 训练 2× WRN-28-10 + 1× PyramidNet-110
# Ensemble
```

**理由**:

- 不同架构的多样性更高
- PyramidNet: 0.82 (验证 0.83)
- WRN: 0.82-0.83 (预期)
- Ensemble 可能达到 0.84-0.85

**预期**:

- **F1: 0.84-0.86**

**时间**: ~8 小时

---

## 📊 决策树

```
当前: dropout 0.2 + drop_path 0.0 实验
  │
  ├─ F1 ≥ 0.83 ✅
  │   ├─ 尝试 SE-Net
  │   │   ├─ F1 ≥ 0.84 ✅ → Ensemble (目标 0.85)
  │   │   └─ F1 < 0.84 → 尝试 Mixup 微调
  │   └─ 尝试 Mixup 微调
  │       ├─ F1 ≥ 0.84 → Ensemble
  │       └─ F1 < 0.84 → Ensemble (3 模型)
  │
  ├─ 0.82 ≤ F1 < 0.83 ⚠️
  │   ├─ 尝试 drop_path 0.05
  │   │   ├─ F1 ≥ 0.83 → 进入 Phase 2
  │   │   └─ F1 < 0.83 → 保持 baseline, 尝试 SE-Net
  │   └─ 直接尝试 SE-Net
  │
  └─ F1 < 0.82 ❌
      ├─ 回退到 drop_path 0.05
      ├─ 或回退到 drop_path 0.1 + dropout 0.3 (baseline)
      └─ 探索其他方向 (架构/增强)
```

---

## 🎯 各阶段目标

| Phase | 目标 F1 | 预期时间 | 关键策略 |
|-------|---------|----------|----------|
| Phase 1.1 ✅ | 0.81-0.82 | 6 小时 | Dropout 调整 |
| Phase 1.2 🔄 | 0.825-0.83 | 2-4 小时 | Drop_path 调整 |
| Phase 2.1 ⏳ | 0.83-0.84 | 2-4 小时 | SE-Net / Mixup 微调 |
| Phase 2.2 ⏳ | 0.835-0.845 | 2 小时 | 组合优化 |
| Phase 3.1 ⏳ | **0.85+** | 6-10 小时 | Ensemble |

**总时间预算**: 18-28 小时

---

## 🎓 重要原则

### 1. 控制变量原则 ✅

```
每次只改变一个变量:
✅ Dropout: 0.3 → 0.2 → 0.15
✅ Drop_path: 0.1 → 0.0
❌ 不要同时改变 dropout + drop_path + SE-Net
```

### 2. 增量优化原则 ✅

```
逐步积累提升:
- dropout 0.2: F1 = 0.81
- + drop_path 0.0: F1 = 0.825 (+0.015)
- + SE-Net: F1 = 0.84 (+0.015)
- + Ensemble: F1 = 0.85 (+0.01)

总提升: +0.04 (0.81 → 0.85)
```

### 3. 风险管理原则 ✅

```
每个实验都有备选方案:
- drop_path 0.0 失败 → 尝试 0.05
- SE-Net 失败 → 尝试 Mixup 微调
- 单模型无法突破 → Ensemble
```

### 4. 时间效率原则 ✅

```
优先测试高成功率策略:
- drop_path 优化: 65-75% 成功率 ✅
- SE-Net: 60-70% 成功率
- Ensemble: 80-90% 成功率 ✅
```

---

## 📋 实验队列 (按优先级)

### 队列 1: 高优先级 (立即执行)

1. ✅ dropout 0.2 + drop_path 0.1 (已完成: F1 = 0.81)
2. 🔄 dropout 0.2 + drop_path 0.0 (进行中)
3. ⏳ dropout 0.2 + drop_path 0.05 (如果 #2 失败)

### 队列 2: 中优先级 (Phase 2)

4. ⏳ 最优配置 + SE-Net
5. ⏳ 最优配置 + Mixup 0.2 (Detail classes)
6. ⏳ 最优配置 + SE-Net + Mixup 0.2

### 队列 3: 低优先级 (如果前面不够)

7. ⏳ 最优配置 + RandAugment (3, 11)
8. ⏳ 最优配置 + Cosine Restart

### 队列 4: 终极策略 (Phase 3)

9. ⏳ Ensemble (3 模型)
10. ⏳ Ensemble (5 模型) - 如果需要
11. ⏳ WRN + PyramidNet Ensemble

---

## 🎯 成功路径预测

### 路径 A: 顺利突破 (概率: 60%)

```
1. drop_path 0.0 成功 (F1 = 0.83)
2. + SE-Net 成功 (F1 = 0.84)
3. + Ensemble (3 模型) → F1 = 0.85 ✅

总时间: 10-12 小时
总实验: 5-6 轮
```

### 路径 B: 中等困难 (概率: 30%)

```
1. drop_path 0.0 部分成功 (F1 = 0.825)
2. + drop_path 0.05 验证 (F1 = 0.83)
3. + SE-Net 失败 (F1 = 0.83)
4. + Mixup 微调 (F1 = 0.835)
5. + Ensemble (5 模型) → F1 = 0.85 ✅

总时间: 18-22 小时
总实验: 8-10 轮
```

### 路径 C: 高难度 (概率: 10%)

```
1. drop_path 优化效果有限 (F1 = 0.82)
2. SE-Net / Mixup 微调效果有限 (F1 = 0.825)
3. 必须依赖 Ensemble (5-7 模型) → F1 = 0.85 ✅

总时间: 24-30 小时
总实验: 12-15 轮
```

---

## ⚠️ 风险与缓解

### 风险 1: 单模型无法突破 0.83

**缓解**:

- 早期尝试 Ensemble
- 测试不同架构组合
- 接受 0.84 作为最终目标

### 风险 2: 时间不足

**缓解**:

- 降低 epochs (300 → 200)
- 使用 early stopping
- 优先测试高成功率策略

### 风险 3: 过拟合

**缓解**:

- 保持合理的正则化
- 监控训练/验证曲线
- 及时回退到稳定配置

---

## 📝 元数据

- **文档版本**: 1.0
- **创建时间**: 2025-10-24
- **状态**: 🔄 Phase 1.2 进行中
- **预计完成**: Phase 3 (6-12 天后)
- **最终目标**: F1 ≥ **0.85** ✨
