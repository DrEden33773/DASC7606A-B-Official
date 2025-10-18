# 实验追踪表

**目标**: F1 ≥ 0.85  
**当前最佳**: F1 = 0.77 (ResNet50 从头训练)  
**最后更新**: 2025-10-17

---

## 📊 实验概览

| Exp# | 日期 | 阶段 | 模型 | Val F1 | Test F1 | 状态 | 备注 |
|------|------|------|------|--------|---------|------|------|
| #000 | 历史 | - | ResNet34 (from scratch) | 0.77 | - | ✅ | 历史基准 |
| #001 | 历史 | - | ResNet50 (from scratch) | 0.77 | - | ✅ | 历史基准 |
| - | - | - | - | - | - | - | - |
| #100 | 2025-10-17 | P1 | Wide ResNet-28-10 (baseline) | 0.7802 | 0.78 | ✅ | +0.01 vs ResNet50 |
| #101 | - | - | (Merged into #103) | - | - | ⏭️ | 跳过 |
| #102 | TBD | P1 | Wide ResNet-28-10 + GridMask | - | - | 🟡 | 低优先级 |
| #103 | 2025-10-17 | P1.5 | WRN-28-10 + SD(0.2) + RA(叠加) | - | 0.75 | 🔴 | 增强过度 |
| #103-Rev | 2025-10-17 | P1.5 | WRN-28-10 + SD(0.2) only | 0.7791 | 0.77 | ⚠️ | SD过强 |
| #104a | TBD | P1 | WRN-28-10 + SD(0.1) + medium | - | - | 🟡 | 计划中 |
| #104b | TBD | P1 | WRN-28-10 + SD(0.1) + RandAug(pure) | - | - | 🟡 | 推荐 |
| #104c | TBD | P1 | WRN-28-10 + SD(0.15) + medium | - | - | 🟡 | 备选 |
| #104d | TBD | P1 | WRN-28-10 + SD(0.1) + RandAug(M=7) | - | - | 🟡 | 备选 |
| - | - | - | - | - | - | - | - |
| #200 | TBD | P2 | ConvNeXt-Tiny (baseline) | - | - | 🟡 | 计划中 |
| #201 | TBD | P2 | ConvNeXt-Tiny + SAM | - | - | 🟡 | 计划中 |
| #202 | TBD | P2 | Wide ResNet + SAM | - | - | 🟡 | 计划中 |
| #203 | TBD | P2 | 自蒸馏 (ensemble → single) | - | - | 🟡 | 计划中 |
| - | - | - | - | - | - | - | - |
| #300 | TBD | P3 | 模型集成 (3-5 models) | - | - | 🟡 | 计划中 |
| #301 | TBD | P3 | Wide ResNet-40-10 | - | - | 🟡 | 计划中 |
| #302 | TBD | P3 | ConvNeXt-Small | - | - | 🟡 | 计划中 |
| #303 | TBD | P3 | 最终模型 + TTA | - | - | 🟡 | 计划中 |

---

## 🎯 阶段里程碑

### Phase 1: 冲击 0.80

- **目标**: Val F1 ≥ 0.80
- **截止**: Week 1
- **状态**: 🟡 未开始
- **关键实验**: #100-#104

### Phase 2: 冲击 0.83

- **目标**: Val F1 ≥ 0.83
- **截止**: Week 2
- **状态**: 🟡 未开始
- **关键实验**: #200-#203

### Phase 3: 冲刺 0.85

- **目标**: Val F1 ≥ 0.85
- **截止**: Week 3
- **状态**: 🟡 未开始
- **关键实验**: #300-#303

---

## 📈 F1 提升趋势

```
0.77 (Baseline) 
  │
  ├─> 0.80 (Phase 1 目标, +0.03)
  │     │
  │     ├─> Wide ResNet (#100-#104)
  │     └─> 数据增强优化
  │
  ├─> 0.83 (Phase 2 目标, +0.06)
  │     │
  │     ├─> ConvNeXt (#200-#201)
  │     └─> SAM 优化器 (#202-#203)
  │
  └─> 0.85 (Phase 3 目标, +0.08)
        │
        ├─> 模型集成 (#300)
        └─> 超大模型 (#301-#303)
```

---

## 🏆 最佳配置记录

### 当前最佳 (Exp #001)

```bash
python main.py \
    --model resnet50 \
    --no_pretrained \
    --dropout 0.45 \
    --lr 0.0008 \
    --weight_decay 1.2e-3 \
    --warmup_epochs 20 \
    --num_epochs 350 \
    --early_stopping_patience 35 \
    --batch_size 96 \
    --mixup_alpha 0.2 \
    --use_cutmix \
    --cutmix_alpha 0.6 \
    --aug_strength medium \
    --use_online_aug
```

**结果**: Val F1 = 0.77

---

### Phase 1 最佳 (TBD)

```bash
[待更新]
```

**结果**: Val F1 = TBD

---

### Phase 2 最佳 (TBD)

```bash
[待更新]
```

**结果**: Val F1 = TBD

---

### 最终提交配置 (TBD)

```bash
[待更新]
```

**结果**: Test F1 = TBD

---

## 📝 实验笔记

### 2025-10-17

- 创建实验追踪系统
- 制定三阶段优化路线图
- 下一步: 实现 Wide ResNet-28-10

---

## 🔗 快速链接

- [优化路线图](./optimization_roadmap.md)
- [实验模板](./experiment_template.md)
- Phase 1 实验:
  - [Exp #100](./phase1/exp_100.md) - Wide ResNet Baseline
  - [Exp #101](./phase1/exp_101.md) - + RandAugment
  - [Exp #102](./phase1/exp_102.md) - + GridMask
  - [Exp #103](./phase1/exp_103.md) - + Stochastic Depth
  - [Exp #104](./phase1/exp_104.md) - 超参数优化

---

**维护**: 每次实验后更新此表格
