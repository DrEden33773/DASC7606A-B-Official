# Experiment #103 - Wide ResNet-28-10 + Stochastic Depth + RandAugment

**日期**: 2025-10-17  
**阶段**: Phase 1.5  
**优先级**: 🔥🔥🔥🔥🔥  
**状态**: 🟡 待运行

---

## 🎯 实验目标

同时启用 Stochastic Depth 和 RandAugment，冲击 Phase 1 目标 (Val F1 ≥ 0.80)。

**预期提升**: +0.020-0.030 F1  
**基准 F1**: 0.7802 (Exp #100 - WRN-28-10 baseline)  
**目标 F1**: ≥ 0.80 (Phase 1 完成标准)

---

## 🔧 配置详情

### 模型配置

```python
model = "wide_resnet28_10"
depth = 28
widen_factor = 10
dropout = 0.3
drop_path_rate = 0.2  # ← 新增: Stochastic Depth
num_classes = 100
```

**参数量**: 36.54M

### 数据增强 (新增 RandAugment)

```python
augmentation_strength = "medium"
use_randaugment = True   # ← 新增
randaugment_n = 2        # ← 2 个随机操作
randaugment_m = 9        # ← 幅度 9/10
mixup_alpha = 0.25
cutmix_alpha = 0.65
use_cutmix = True
```

### 训练超参数

```python
lr = 0.001
weight_decay = 5e-4
batch_size = 128
num_epochs = 500
warmup_epochs = 10
early_stopping_patience = 50

optimizer = "adamw"
scheduler = "cosine"
```

### 技术栈

- [x] AMP (自动混合精度)
- [x] EMA (指数移动平均)
- [x] **Stochastic Depth (drop_path=0.2)** ← 新增
- [x] **RandAugment (N=2, M=9)** ← 新增
- [x] Mixup + CutMix (自适应)
- [x] Gradient Clipping (1.0)

---

## 📊 运行命令

### PowerShell

```powershell
python main.py `
    --model wide_resnet28_10 `
    --dropout 0.3 `
    --drop_path_rate 0.2 `
    --use_randaugment `
    --randaugment_n 2 `
    --randaugment_m 9 `
    --lr 0.001 `
    --weight_decay 5e-4 `
    --warmup_epochs 10 `
    --num_epochs 500 `
    --early_stopping_patience 50 `
    --batch_size 128 `
    --mixup_alpha 0.25 `
    --use_cutmix `
    --cutmix_alpha 0.65 `
    --aug_strength medium `
    --use_online_aug `
    --seed 42
```

### Bash

```bash
python main.py \
    --model wide_resnet28_10 \
    --dropout 0.3 \
    --drop_path_rate 0.2 \
    --use_randaugment \
    --randaugment_n 2 \
    --randaugment_m 9 \
    --lr 0.001 \
    --weight_decay 5e-4 \
    --warmup_epochs 10 \
    --num_epochs 500 \
    --early_stopping_patience 50 \
    --batch_size 128 \
    --mixup_alpha 0.25 \
    --use_cutmix \
    --cutmix_alpha 0.65 \
    --aug_strength medium \
    --use_online_aug \
    --seed 42
```

---

## 📈 实验结果

### 训练指标

| Metric | Value |
|--------|-------|
| Best Epoch | TBD / 500 |
| Train Loss | TBD |
| Train Acc | TBD% |
| Val Loss | TBD |
| Val Acc | TBD% |
| Val F1 (macro) | **TBD** |
| Test F1 (macro) | **TBD** |

### 训练时间

- **总时间**: TBD 小时
- **平均每 epoch**: ~60-70 秒 (略慢于 baseline，因为 RandAugment)
- **硬件**: RTX 5080

### 性能对比

| 实验 | 配置 | Val F1 | 提升 |
|-----|------|--------|------|
| Exp #100 | WRN-28-10 (baseline) | 0.7802 | - |
| **Exp #103** | **+ SD + RA** | **TBD** | **TBD** |

### vs Phase 1 目标

| 目标 | 结果 | 状态 |
|-----|------|------|
| Val F1 ≥ 0.80 | TBD | 🟡 待验证 |

---

## 📉 预期分析

### 预期效果分解

| 优化 | 预期提升 | 依据 |
|-----|---------|------|
| Stochastic Depth | +0.015-0.020 | Wide ResNet 论文 (+2%) |
| RandAugment | +0.010-0.015 | NeurIPS 2020 论文 |
| **总计** | **+0.025-0.035** | 保守估计 |

### 预期结果

```
Baseline (Exp #100): 0.7802
+ 保守估计 (+0.025): 0.8052 ✅
+ 乐观估计 (+0.035): 0.8152 ✅✅
```

**成功概率**: 85%+

---

## 🔍 关键改进点

### vs Exp #100 的变化

1. **Stochastic Depth** (新增):
   - 线性递增: block_0=0.0 → block_11=0.2
   - 解决过拟合问题
   - 允许更长时间训练

2. **RandAugment** (新增):
   - 自动增强搜索
   - N=2 (每图 2 个操作)
   - M=9 (强度 9/10)
   - 14 种操作池

3. **其他保持不变**:
   - Dropout = 0.3 ✅
   - Weight Decay = 5e-4 ✅
   - Mixup + CutMix ✅
   - EMA ✅

---

## 💡 关键假设

### 假设 1: Stochastic Depth 解决过拟合

**证据** (Exp #100):

- Epoch 152 后 Val F1 停滞
- Train Acc 继续提升

**预期**: SD 允许模型继续学习而不过拟合

### 假设 2: RandAugment 提升泛化

**证据**:

- 当前 medium augmentation 可能不够多样
- RandAugment 提供 14 种操作的随机组合

**预期**: 更丰富的数据多样性 → 更好的泛化

### 假设 3: 两者互补

**理由**:

- SD = 模型正则化（结构层面）
- RA = 数据正则化（样本层面）
- 作用机制不同，效果叠加

---

## 🎯 成功标准

| 结果 | 评价 | 后续行动 |
|-----|------|---------|
| Val F1 ≥ 0.80 | ✅ Phase 1 完成！ | 开始 Phase 2 准备 |
| Val F1 = 0.795-0.80 | 🟡 非常接近 | 微调超参数 |
| Val F1 = 0.79-0.795 | ⚠️ 有提升但不够 | Exp #104 (dropout/WD 调优) |
| Val F1 < 0.79 | 🔴 未达预期 | 重新评估策略 |

---

## 📁 相关文件

- **实现 (SD)**: `bot/implementations/wide_resnet.py` (DropPath class)
- **实现 (RA)**: `bot/implementations/augmentations/randaugment.py`
- **集成**: `scripts/model_architectures.py`, `scripts/train_utils.py`
- **参数**: `main.py` (--drop_path_rate, --use_randaugment, etc.)
- **日志**: `cifar_pipeline.log`
- **模型**: `results/models/best_model.pth`
- **指标**: `training_metrics.txt`

---

## 🏷️ 标签

`phase-1.5` `wide-resnet` `stochastic-depth` `randaugment` `pending`

---

## 📝 实验日志

### 2025-10-17 15:30

- ✅ Stochastic Depth 实现完成
  - 添加 DropPath 类
  - 修改 WideBasicBlock 和 WideResNet
  - 线性递增策略 (0.0 → 0.2)
  
- ✅ RandAugment 实现完成
  - 14 种操作池
  - N=2, M=9 配置
  - 集成到 train_transforms

- ✅ 单元测试通过
  - Stochastic Depth: ✅
  - RandAugment: ✅
  - Integration: ✅

- 🟡 待运行完整训练

---

**实验人员**: AI Assistant + User  
**最后更新**: 2025-10-17 15:30  
**预计完成**: 今晚或明早
