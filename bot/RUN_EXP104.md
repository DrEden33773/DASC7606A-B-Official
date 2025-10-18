# 🚀 立即运行 Exp #104b

**配置**: WRN-28-10 + SD(0.1) + RandAugment(pure)  
**目标**: Val F1 ≥ 0.80  
**预期**: 0.79-0.82

---

## 📋 快速命令

### 最简化版本 (推荐)

```bash
python main.py \
    --drop_path_rate 0.1 \
    --aug_strength randaugment
```

**说明**: 其他参数使用最优默认值

---

### 完整版本

```bash
python main.py \
    --model wide_resnet28_10 \
    --dropout 0.3 \
    --drop_path_rate 0.1 \
    --aug_strength randaugment \
    --randaugment_n 2 \
    --randaugment_m 9 \
    --lr 0.001 \
    --weight_decay 5e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --warmup_epochs 10 \
    --num_epochs 500 \
    --early_stopping_patience 50 \
    --batch_size 128 \
    --mixup_alpha 0.25 \
    --use_cutmix \
    --cutmix_alpha 0.65 \
    --use_amp \
    --use_ema \
    --seed 42
```

---

## 🎯 关键改动

| 参数 | Exp #100 | Exp #103-Rev | Exp #104b |
|-----|---------|--------------|-----------|
| drop_path_rate | 0.0 | 0.2 | **0.1** |
| aug_strength | medium | medium | **randaugment** |

---

## 📊 预期效果

```
Baseline (Exp #100): 0.7802
+ SD 轻量化 (0.1):  +0.005-0.010
+ RandAug (纯净):   +0.010-0.015
= 0.7952-0.8052 ✅
```

**成功概率**: 70-75%

---

## 🔍 监控指标

### 关键指标

**健康信号**:

- Train Acc ≈ 65-70% (恢复正常)
- Val Acc ≈ 78-80%
- Train/Val gap ≈ 10-15% (合理)

**警报信号**:

- Train Acc < 60% → 增强仍过强
- Train Acc > 75% → 过拟合
- Val F1 < 0.78 → 配置有问题

---

## ⚡ 备选配置

### 如果 M=9 太强

```bash
python main.py \
    --drop_path_rate 0.1 \
    --aug_strength randaugment \
    --randaugment_m 7
```

### 如果想测试 SD(0.15)

```bash
python main.py \
    --drop_path_rate 0.15 \
    --aug_strength medium
```

---

**开始训练吧！** 🚀
