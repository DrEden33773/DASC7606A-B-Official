# Experiment #XXX - [实验名称]

**日期**: YYYY-MM-DD  
**阶段**: Phase X  
**优先级**: 🔥🔥🔥🔥🔥 / 🔥🔥🔥🔥 / 🔥🔥🔥  
**状态**: 🟡 计划中 / 🔵 进行中 / 🟢 完成 / 🔴 失败

---

## 🎯 实验目标

[描述本次实验的具体目标和假设]

**预期提升**: +0.0XX F1  
**基准 F1**: 0.XX (Experiment #YYY)

---

## 🔧 配置详情

### 模型配置

```python
model = "wide_resnet28_10"  # 或其他
dropout = 0.5
num_classes = 100
```

### 数据增强

```python
augmentation_strength = "medium"
use_randaugment = True
randaugment_n = 2
randaugment_m = 9
mixup_alpha = 0.2
cutmix_alpha = 0.6
```

### 训练超参数

```python
lr = 0.001
weight_decay = 1e-3
batch_size = 128
num_epochs = 500
warmup_epochs = 20
early_stopping_patience = 50

optimizer = "adamw"
scheduler = "cosine"
```

### 其他技巧

- [ ] AMP (自动混合精度)
- [ ] EMA (指数移动平均)
- [ ] Stochastic Depth
- [ ] SAM 优化器
- [ ] Label Smoothing

---

## 📊 运行命令

```bash
python main.py \
    --model wide_resnet28_10 \
    --no_pretrained \
    --dropout 0.5 \
    --lr 0.001 \
    --weight_decay 1e-3 \
    --warmup_epochs 20 \
    --num_epochs 500 \
    --early_stopping_patience 50 \
    --batch_size 128 \
    --mixup_alpha 0.2 \
    --use_cutmix \
    --cutmix_alpha 0.6 \
    --aug_strength medium \
    --use_online_aug \
    --seed 42
```

---

## 📈 实验结果

### 训练指标

| Metric | Value |
|--------|-------|
| Best Epoch | XXX / 500 |
| Train Loss | X.XXXX |
| Train Acc | XX.XX% |
| Val Loss | X.XXXX |
| Val Acc | XX.XX% |
| Val F1 (macro) | **0.XXXX** |
| Test F1 (macro) | **0.XXXX** |

### 训练时间

- **总时间**: X.XX 小时
- **平均每 epoch**: XX.X 秒
- **硬件**: RTX 5080

### 性能对比

| 实验 | Val F1 | Test F1 | 提升 |
|-----|--------|---------|------|
| Baseline (Exp #YYY) | 0.XXXX | 0.XXXX | - |
| Current (Exp #XXX) | 0.XXXX | 0.XXXX | +0.0XXX |

---

## 📉 训练曲线

### Loss Curve

```
[TODO: 插入 loss 曲线图或描述趋势]
- 训练 loss 在第 XX epoch 收敛
- 验证 loss 在第 XX epoch 达到最低
```

### F1 Curve

```
[TODO: 插入 F1 曲线图或描述趋势]
- F1 在第 XX epoch 达到峰值
- 早停触发: 第 XX epoch (patience=50)
```

---

## 🔍 详细分析

### ✅ 成功之处

1. [列出本次实验成功的方面]
2. [...]

### ❌ 失败/问题

1. [列出遇到的问题]
2. [...]

### 🤔 观察与洞察

- [记录有趣的现象]
- [意外的发现]
- [对下一步的启发]

### 📊 困难类别分析

```
Top-5 最难分类的类别:
1. girl (F1=0.XX)
2. seal (F1=0.XX)
3. otter (F1=0.XX)
4. boy (F1=0.XX)
5. shrew (F1=0.XX)

与 baseline 的对比:
- [哪些类别有改进]
- [哪些类别退步了]
```

---

## 💡 下一步行动

### 优先级 1 (立即尝试)

- [ ] [Action item 1]
- [ ] [Action item 2]

### 优先级 2 (后续考虑)

- [ ] [Action item 3]
- [ ] [Action item 4]

### 放弃的想法

- ❌ [不再尝试的方向及原因]

---

## 📁 相关文件

- **代码**: `bot/experiments/exp_XXX/train.py`
- **配置**: `bot/experiments/exp_XXX/config.yaml`
- **日志**: `bot/experiments/exp_XXX/training.log`
- **模型**: `results/models/exp_XXX_best_model.pth`
- **指标**: `results/results/exp_XXX_metrics.txt`

---

## 🏷️ 标签

`phase-1` `wide-resnet` `randaugment` `baseline` `completed`

---

**实验人员**: [你的名字]  
**最后更新**: YYYY-MM-DD HH:MM
