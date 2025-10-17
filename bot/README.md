# Bot 工作目录 - CIFAR-100 优化项目

**创建日期**: 2025-10-17  
**项目目标**: CIFAR-100 分类，从头训练达到 F1 ≥ 0.85  
**当前最佳**: F1 = 0.77 (ResNet34/50)

---

## 📂 目录结构

```
bot/
├── README.md                    # 本文件 - 项目索引
├── plans/                       # 优化计划
│   └── optimization_roadmap.md  # 三阶段优化路线图 ⭐
├── experiments/                 # 实验记录
│   ├── experiment_template.md   # 实验记录模板
│   ├── experiment_tracker.md    # 实验追踪表 ⭐
│   ├── phase1/                  # Phase 1 实验 (目标: F1≥0.80)
│   ├── phase2/                  # Phase 2 实验 (目标: F1≥0.83)
│   └── phase3/                  # Phase 3 实验 (目标: F1≥0.85)
├── implementations/             # 新模型/技术实现
│   ├── wide_resnet.py          # Wide ResNet-28-10 (待实现)
│   ├── convnext_cifar.py       # ConvNeXt-Tiny for CIFAR (待实现)
│   ├── sam_optimizer.py        # SAM 优化器 (待实现)
│   └── augmentations/          # 高级数据增强
│       ├── randaugment.py      # RandAugment (待实现)
│       ├── gridmask.py         # GridMask (待实现)
│       └── fmix.py             # FMix (待实现)
├── notes/                      # 技术笔记与参考
│   ├── technical_notes.md      # 实现要点与最佳实践 ⭐
│   └── quick_reference.md      # 快速参考指南 ⭐
└── configs/                    # 配置文件
    ├── phase1_baseline.yaml    # Phase 1 基线配置 (待创建)
    ├── phase2_convnext.yaml    # Phase 2 ConvNeXt 配置 (待创建)
    └── final_submission.yaml   # 最终提交配置 (待创建)
```

---

## 🎯 三阶段优化路线

### Phase 1: 冲击 0.80 (Week 1)

**策略**: 低风险、高回报的快速优化

| 任务 | 优先级 | 状态 | 预期提升 |
|-----|--------|------|---------|
| 实现 Wide ResNet-28-10 | 🔥🔥🔥🔥🔥 | 🟡 待开始 | +0.02-0.03 |
| 添加 RandAugment | 🔥🔥🔥🔥 | 🟡 待开始 | +0.01-0.02 |
| 实现 GridMask | 🔥🔥🔥🔥 | 🟡 待开始 | +0.005-0.01 |
| Stochastic Depth | 🔥🔥🔥🔥 | 🟡 待开始 | +0.01-0.015 |
| 超参数精调 | 🔥🔥🔥 | 🟡 待开始 | +0.005-0.01 |

**目标**: Val F1 ≥ 0.80  
**详情**: [优化路线图](plans/optimization_roadmap.md#-phase-1-冲击-080)

---

### Phase 2: 冲击 0.83 (Week 2)

**策略**: 中等风险、中等回报的架构探索

| 任务 | 优先级 | 状态 | 预期提升 |
|-----|--------|------|---------|
| 实现 ConvNeXt-Tiny | 🔥🔥🔥🔥🔥 | 🟡 待开始 | +0.02-0.03 |
| SAM 优化器 | 🔥🔥🔥🔥 | 🟡 待开始 | +0.01-0.02 |
| 自蒸馏 | 🔥🔥🔥 | 🟡 待开始 | +0.01-0.015 |
| FMix 增强 | 🔥🔥🔥 | 🟡 待开始 | +0.005-0.01 |

**目标**: Val F1 ≥ 0.83  
**详情**: [优化路线图](plans/optimization_roadmap.md#-phase-2-冲击-083)

---

### Phase 3: 冲刺 0.85 (Week 3)

**策略**: 高风险、高回报的极致优化

| 任务 | 优先级 | 状态 | 预期提升 |
|-----|--------|------|---------|
| 模型集成 | 🔥🔥🔥🔥🔥 | 🟡 待开始 | +0.02-0.03 |
| Wide ResNet-40-10 | 🔥🔥🔥🔥 | 🟡 待开始 | +0.01-0.02 |
| TTA | 🔥🔥🔥 | 🟡 待开始 | +0.01-0.015 |
| 贝叶斯优化 | 🔥🔥🔥 | 🟡 待开始 | +0.005-0.01 |

**目标**: Val F1 ≥ 0.85  
**详情**: [优化路线图](plans/optimization_roadmap.md#-phase-3-冲刺-085)

---

## 📖 快速开始

### 1. 查看优化计划

```bash
# 完整的三阶段路线图
cat bot/plans/optimization_roadmap.md

# 或在 VS Code 中打开
code bot/plans/optimization_roadmap.md
```

### 2. 开始 Phase 1

```bash
# 查看 Phase 1 待办事项
grep -A 10 "Phase 1 实施检查清单" bot/plans/optimization_roadmap.md

# 查看实验追踪表
cat bot/experiments/experiment_tracker.md
```

### 3. 创建新实验

```bash
# 复制模板
cp bot/experiments/experiment_template.md \
   bot/experiments/phase1/exp_100.md

# 编辑实验配置
code bot/experiments/phase1/exp_100.md
```

### 4. 查阅技术文档

```bash
# 查看 Wide ResNet 实现要点
grep -A 50 "Wide ResNet 实现要点" bot/notes/technical_notes.md

# 查看常用命令
cat bot/notes/quick_reference.md
```

---

## 🔥 当前工作重点

### 本周任务 (Week 1 - Phase 1)

#### ✅ 已完成

- [x] 创建项目结构和文档
- [x] 制定三阶段优化路线图
- [x] 准备实验追踪系统

#### 🔵 进行中

- [ ] 无

#### 🟡 待开始

1. **实现 Wide ResNet-28-10** (优先级: 🔥🔥🔥🔥🔥)
   - 文件: `bot/implementations/wide_resnet.py`
   - 集成: `scripts/model_architectures.py`
   - 参考: [技术笔记 - Wide ResNet](notes/technical_notes.md#-wide-resnet-实现要点)

2. **添加 RandAugment** (优先级: 🔥🔥🔥🔥)
   - 文件: `bot/implementations/augmentations/randaugment.py`
   - 集成: `scripts/data_augmentation.py`

3. **运行基线实验** (优先级: 🔥🔥🔥🔥🔥)
   - 实验: Exp #100
   - 配置: Wide ResNet-28-10 + 当前最佳超参
   - 目标: 验证实现正确性

---

## 📊 实验管理

### 实验编号规则

- `#000-#099`: 历史实验 / 基准
- `#100-#199`: Phase 1 实验
- `#200-#299`: Phase 2 实验
- `#300-#399`: Phase 3 实验

### 实验记录规范

每个实验必须记录：

1. ✅ 实验目标与假设
2. ✅ 完整的配置参数
3. ✅ 训练/验证/测试指标
4. ✅ 分析与下一步行动
5. ✅ 相关代码和日志路径

**模板**: [experiment_template.md](experiments/experiment_template.md)

### 实验追踪

所有实验在 [实验追踪表](experiments/experiment_tracker.md) 中登记。

---

## 🛠️ 开发流程

### 1. 实现新模型/技术

```bash
# 在 bot/implementations/ 中创建新文件
code bot/implementations/wide_resnet.py

# 实现并测试
python -c "from bot.implementations.wide_resnet import WideResNet; print(WideResNet(28, 10, 100))"

# 集成到主项目
# 修改 scripts/model_architectures.py
```

### 2. 运行实验

```bash
# 使用快速参考中的命令
python main.py --model wide_resnet28_10 --no_pretrained ...

# 或使用配置文件 (待实现)
python main.py --config bot/configs/phase1_baseline.yaml
```

### 3. 记录结果

```bash
# 创建实验记录
cp bot/experiments/experiment_template.md \
   bot/experiments/phase1/exp_100_wide_resnet_baseline.md

# 填写实验结果
code bot/experiments/phase1/exp_100_wide_resnet_baseline.md

# 更新实验追踪表
code bot/experiments/experiment_tracker.md
```

### 4. 提交代码

```bash
git add bot/ scripts/
git commit -m "Exp #100: Wide ResNet-28-10 baseline, F1=0.XX"
git push
```

---

## 📈 进度追踪

### 当前最佳结果

- **模型**: ResNet50 (从头训练)
- **Val F1**: 0.77
- **Test F1**: TBD
- **配置**: [quick_reference.md](notes/quick_reference.md#resnet50-当前最佳配置)

### 距离目标还差

- **Phase 1 (0.80)**: +0.03 F1 ⚠️
- **Phase 2 (0.83)**: +0.06 F1 ⚠️⚠️
- **Phase 3 (0.85)**: +0.08 F1 ⚠️⚠️⚠️

### 时间规划

- **Week 1 (剩余)**: 实现 Wide ResNet + RandAugment, 运行基线实验
- **Week 2**: ConvNeXt + SAM 优化器
- **Week 3**: 模型集成 + 最终调优

---

## 🔗 重要文档链接

### 规划与追踪

- 📋 [**优化路线图**](plans/optimization_roadmap.md) - 三阶段完整计划
- 📊 [**实验追踪表**](experiments/experiment_tracker.md) - 所有实验总览
- 📝 [**实验模板**](experiments/experiment_template.md) - 标准化记录模板

### 技术参考

- 🔧 [**技术笔记**](notes/technical_notes.md) - 实现要点与最佳实践
- ⚡ [**快速参考**](notes/quick_reference.md) - 常用命令速查
- 📚 主项目 [README.md](../README.md) - 项目整体说明

### 实验记录

- 📁 [Phase 1 实验](experiments/phase1/) - 目标 F1 ≥ 0.80
- 📁 [Phase 2 实验](experiments/phase2/) - 目标 F1 ≥ 0.83
- 📁 [Phase 3 实验](experiments/phase3/) - 目标 F1 ≥ 0.85

---

## ⚠️ 重要约束

### 必须遵守

1. ❌ **禁止使用预训练模型** (ImageNet, 迁移学习, 蒸馏)
2. ✅ **必须从头训练** (--no_pretrained)
3. ✅ **模型架构无限制** (CNN, Transformer, 任意架构)
4. ⏱️ **训练时间 < 12 小时**
5. 💾 **显存限制**: RTX 5080 / 4080 Super

### 可修改的文件

- ✅ `scripts/data_augmentation.py`
- ✅ `scripts/model_architectures.py`
- ✅ `scripts/train_utils.py`
- ✅ `main.py` (默认超参数)

### 不可修改的文件

- ❌ 所有 `.ipynb` 笔记本
- ❌ `scripts/data_download.py`
- ❌ `scripts/evaluation_metrics.py`

---

## 🎯 成功标准

| F1-score | 得分 | 难度评估 |
|----------|------|---------|
| ≥ 0.85 | 100% | 🔴 困难 |
| ≥ 0.80 | 90% | 🟡 中等 |
| ≥ 0.75 | 80% | 🟢 简单 |
| < 0.55 | 0% | - |

**当前目标**: Phase 1 达到 0.80 (90分), 最终冲击 0.85 (100分)

---

## 📞 联系与支持

- **项目维护**: [你的名字]
- **最后更新**: 2025-10-17
- **Git 分支**: `resnet-best-practice`

---

## 🏷️ 版本历史

### v0.1.0 (2025-10-17)

- ✅ 创建 bot/ 目录结构
- ✅ 编写优化路线图 (三阶段)
- ✅ 准备实验追踪系统
- ✅ 撰写技术文档和快速参考

### v0.2.0 (待发布)

- 🟡 实现 Wide ResNet-28-10
- 🟡 添加 RandAugment
- 🟡 运行 Exp #100 基线实验

---

**开始你的优化之旅！** 🚀

首先查看 [优化路线图](plans/optimization_roadmap.md)，然后开始 Phase 1 的第一个任务。
