# Experiment #100 - Wide ResNet-28-10 Baseline

**日期**: 2025-10-17  
**阶段**: Phase 1  
**优先级**: 🔥🔥🔥🔥🔥  
**状态**: 🟡 待开始

---

## 🎯 实验目标

验证 Wide ResNet-28-10 实现正确性，并建立新的基线。

**预期提升**: Val F1 ≥ 0.77 (当前 ResNet50 水平)  
**基准 F1**: 0.77 (ResNet50/ResNet34 从头训练)

---

## 🔧 配置详情

### 模型配置

```python
model = "wide_resnet28_10"
depth = 28  # 6n+4, n=4
widen_factor = 10
dropout = 0.3  # Wide ResNet 原论文推荐
num_classes = 100
```

**参数量**: 36.54M (vs ResNet50 23.5M)

### 数据增强

```python
augmentation_strength = "medium"
use_randaugment = False  # 基线不使用
mixup_alpha = 0.25
cutmix_alpha = 0.65
use_cutmix = True
```

### 训练超参数

```python
lr = 0.001
weight_decay = 5e-4  # Wide ResNet 推荐
batch_size = 128
num_epochs = 500
warmup_epochs = 10
early_stopping_patience = 50

optimizer = "adamw"
scheduler = "cosine"
```

### 其他技巧

- [x] AMP (自动混合精度)
- [x] EMA (指数移动平均)
- [ ] Stochastic Depth (待 Phase 1.2)
- [ ] SAM 优化器 (待 Phase 2)
- [ ] Label Smoothing (与 Mixup/CutMix 互斥)

---

## 📊 运行命令

```bash
python main.py \
    --model wide_resnet28_10 \
    --dropout 0.3 \
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
- **平均每 epoch**: TBD 秒
- **硬件**: RTX 5080

### 性能对比

| 实验 | 模型 | 参数量 | Val F1 | 提升 |
|-----|-----|--------|--------|------|
| Baseline (历史) | ResNet50 | 23.5M | 0.77 | - |
| Baseline (历史) | ResNet34 | 21M | 0.77 | - |
| Current (Exp #100) | WRN-28-10 | 36.5M | TBD | TBD |

---

## 📉 训练曲线

### Loss Curve

[TODO: 插入 loss 曲线图或描述趋势]

- 训练 loss 在第 XX epoch 收敛
- 验证 loss 在第 XX epoch 达到最低

### F1 Curve

[TODO: 插入 F1 曲线图或描述趋势]

- F1 在第 XX epoch 达到峰值
- 早停触发: 第 XX epoch (patience=50)

---

## 🔍 详细分析

### ✅ 成功之处

[TODO: 实验完成后填写]

### ❌ 失败/问题

[TODO: 实验完成后填写]

### 🤔 观察与洞察

[TODO: 实验完成后填写]

### 📊 困难类别分析

```
Top-5 最难分类的类别:
[TODO: 实验完成后填写]
```

---

## 💡 下一步行动

### 优先级 1 (立即尝试)

- [ ] **Exp #101**: 添加 RandAugment (N=2, M=9)
- [ ] **Exp #102**: 添加 GridMask
- [ ] **Exp #103**: 添加 Stochastic Depth (drop_path=0.1)

### 优先级 2 (后续考虑)

- [ ] **超参数调优**: lr, weight_decay, dropout 网格搜索
- [ ] **更长训练**: 尝试 600-800 epochs

### 放弃的想法

- ❌ 使用预训练模型（禁止）

---

## 📁 相关文件

- **实现**: `bot/implementations/wide_resnet.py`
- **集成**: `scripts/model_architectures.py`
- **配置**: 见上述运行命令
- **日志**: `cifar_pipeline.log`
- **模型**: `results/models/exp_100_best_model.pth`
- **指标**: `results/results/exp_100_metrics.txt`

---

## 🏷️ 标签

`phase-1` `wide-resnet` `baseline` `wrn-28-10` `pending`

---

## 📝 实验日志

### 2025-10-17 14:00

- ✅ Wide ResNet-28-10 实现完成
- ✅ 集成到 `model_architectures.py`
- ✅ 删除 SimpleCNN 和预训练相关代码
- ✅ 单元测试通过 (参数量: 36.54M)
- 🟡 待运行完整训练实验

---

**实验人员**: AI Assistant + User  
**最后更新**: 2025-10-17 14:00
